import os
import collections
import itertools
import pickle
import numpy
import gensim
import copy
import torch
import random

import xml.etree.ElementTree as ET

from conllu import parse
from torch.utils.data import Dataset
from torch.utils.data.sampler import RandomSampler

CUSTOM_FIELDS = ('form', 'lemma', 'pos', 'lu', 'senses', 'entity')

POS_MAP = {"!": ".", "#": ".", "$": ".", "''": ".", "(": ".", ")": ".", ",": ".", "-LRB-": ".", "-RRB-": ".",
           ".": ".", ":": ".", "?": ".", "CC": "CONJ", "CD": "NUM", "CD|RB": "X", "DT": "DET", "EX": "DET",
           "FW": "X", "IN": "ADP", "IN|RP": "ADP", "JJ": "ADJ", "JJR": "ADJ", "JJRJR": "ADJ", "JJS": "ADJ",
           "JJ|RB": "ADJ", "JJ|VBG": "ADJ", "LS": "X", "MD": "VERB", "NN": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
           "NNS": "NOUN", "NN|NNS": "NOUN", "NN|SYM": "NOUN", "NN|VBG": "NOUN", "NP": "NOUN", "PDT": "DET",
           "POS": "PRT", "PRP": "PRON", "PRP$": "PRON", "PRP|VBP": "PRON", "PRT": "PRT", "RB": "ADV", "RBR": "ADV",
           "RBS": "ADV", "RB|RP": "ADV", "RB|VBG": "ADV", "RN": "X", "RP": "PRT", "SYM": "X", "TO": "PRT",
           "UH": "X", "VB": "VERB", "VBD": "VERB", "VBD|VBN": "VERB", "VBG": "VERB", "VBG|NN": "VERB",
           "VBN": "VERB", "VBP": "VERB", "VBP|TO": "VERB", "VBZ": "VERB", "VP": "VERB", "WDT": "DET", "WH": "X",
           "WP": "PRON", "WP$": "PRON", "WRB": "ADV", "``": "."}
POS_NORMALIZE = {"MD|VB": "MD", "NNP|NP": "NNP", "NPS": "POS", "PR": "WRB", "NNP|VBN": "VBN", "PP": "PRP"}
POS_MAP_SIMPLE = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

def get_pos(old_pos):
    if old_pos in globals.pos_map:
        new_pos = globals.pos_map[old_pos]
        if new_pos in globals.pos_map_simple:
            new_pos = globals.pos_map_simple[new_pos]
        else:
            new_pos = new_pos.lower()
    else:
        new_pos = old_pos.lower()
    return new_pos

class ConcatDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders
        self.thresholds = []
        init_threshold = 0.0
        for d in self.dataloaders:
            init_threshold += len(d._index_sampler)
            self.thresholds.add(init_threshold)

    def __getitem__(self, i):

        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return sum(len(d._index_sampler) for d in self.dataloaders)

'''Modified from: https://github.com/bomri/code-for-posts/tree/master/mtl-data-loading'''
class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.total_length = sum(len(dataset) for dataset in dataset.datasets)
        self.partitions = [cur_length*1.0/self.total_length for cur_length in dataset.cumulative_sizes]

    def __len__(self):
        return len(self.dataset) * self.number_of_datasets

    def __iter__(self):
        samplers_list = []
        sampler_iterators = []
        datasets_length = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset)
            samplers_list.append(sampler)
            cur_sampler_iterator = sampler.__iter__()
            sampler_iterators.append(cur_sampler_iterator)
            datasets_length.append(len(cur_dataset))

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size # * self.number_of_datasets
        samples_to_grab = self.batch_size
        largest_dataset_index = torch.argmax(torch.as_tensor(datasets_length)).item()
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        # epoch_samples = datasets_length[largest_dataset_index] * self.number_of_datasets
        # iterate over the total length of the combined datasets (slightly oversampling some datasets, and undersampling others)
        epoch_samples = self.total_length

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            #TODO instead of alternating between datasets, flip a (weighted) coin every time
            coin_toss = torch.rand(1).item()
            for threshold in self.partitions:
                if threshold > coin_toss:
                    i = self.partitions.index(threshold)
                    break
            # for i in range(self.number_of_datasets):
            cur_batch_sampler = sampler_iterators[i]
            cur_samples = []
            for _ in range(samples_to_grab):
                try:
                    cur_sample_org = cur_batch_sampler.__next__()
                    cur_sample = cur_sample_org + push_index_val[i]
                    cur_samples.append(cur_sample)
                except StopIteration:
                    if i == largest_dataset_index:
                        # largest dataset iterator is done we can break
                        samples_to_grab = len(cur_samples)  # adjusting the samples_to_grab
                        # got to the end of iterator - extend final list and continue to next task if possible
                        break
                    else:
                        # restart the iterator - we want more samples until finishing with the largest dataset
                        sampler_iterators[i] = samplers_list[i].__iter__()
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
            final_samples_list.extend(cur_samples)

        return iter(final_samples_list)


class Sample():

    def __init__(self):
        ''' Simple structure to hold the sentence features '''
        self.length = 0
        self.forms = []
        self.lemmas = []
        self.pos = []
        self.lemmas_pos = []
        self.senses = []
        self.entities = []
        self.sentence_str = ""

class WSDataset(Dataset):

    def __init__(self, device, tsv_data, src2id, embeddings, embeddings_dim, embeddings_input, max_labels, lu2senses,
                 single_softmax, batch_layers, known_senses=None, pos_map=None, pos_filter=False):
        # Our data has some pretty long sentences, so we will set a large max length
        # Alternatively, can throw them out or truncate them
        self.device = device
        self.src2id = src2id
        self.embeddings = embeddings
        self.embeddings_dim = embeddings_dim
        self.embeddings_input = embeddings_input
        self.max_labels = max_labels
        self.lu2senses = lu2senses
        self.pos_filter = pos_filter
        self.known_lus, self.known_pos, self.known_entity_tags = set(), set(), set()
        if pos_map is not None:
            pos_map = get_pos_tagset(pos_map, "medium")
        self.data = self.parse_tsv(tsv_data, 300, pos_map, pos_filter)
        self.known_lus, self.known_pos, self.known_entity_tags = \
            sorted(self.known_lus), sorted(self.known_pos), sorted(self.known_entity_tags)
        self.single_softmax = single_softmax
        self.batch_layers = batch_layers
        if self.single_softmax is True:
            if known_senses is None:
                self.known_senses = {"UNKNOWN" : 0}
                id = 1
                for lu in self.known_lus:
                    if lu in lu2senses:
                        for sense in lu2senses[lu]:
                            if sense not in self.known_senses:
                                self.known_senses[sense] = id
                                id += 1
            else:
                self.known_senses = known_senses
        self.known_pos = {pos_tag:i for i, pos_tag in enumerate(self.known_pos)}
        self.known_entity_tags = {entity_tag: i for i, entity_tag in enumerate(self.known_entity_tags)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' Prepare one sample sentence of the data '''
        # Get a sentence
        sample = self.data[idx]
        # Get an integer ID for each lemma in the sentence (<UNK> if unfamiliar)
        # Note that we are working with lemmas for the input, not the word forms
        input_source = sample.lemmas if self.embeddings_input == "lemma" else sample.forms
        inputs = [self.src2id[token] if token in self.src2id
                  else self.src2id["<UNK>"] for token in input_source]
        targets_embed, neg_targets, targets_classify, targets_pos, targets_ner, mask, pos_mask, ner_mask, lengths_labels = \
            [], [], [], [], [], [], [], [], 0
        for i, label in enumerate(sample.senses):
            lemma = sample.lemmas[i]
            lemma_pos = sample.lemmas_pos[i]
            dict_key = lemma_pos.lower() if self.pos_filter is True else lemma.lower()
            pos = sample.pos[i]
            entity = sample.entities[i]
            target_embed, neg_target, target_classify = \
                torch.zeros(self.embeddings_dim), torch.zeros(self.embeddings_dim), torch.zeros(self.max_labels)
            # Get labels for POS tags
            targets_pos.append(self.known_pos[pos] if pos in self.known_pos else -1)
            targets_ner.append(self.known_entity_tags[entity] if entity in self.known_entity_tags else -1)
            if label == "_":
                mask.append(False)
                if sample.pos[i] is not "_":
                    pos_mask.append(True)
                else:
                    pos_mask.append(False)
                if sample.entities[i] is not "_":
                    ner_mask.append(True)
                else:
                    ner_mask.append(False)
                targets_classify.append(-1)
            else:
                mask.append(True)
                pos_mask.append(True)
                ner_mask.append(True)
                # lemma_pos = sample.lemmas[i] + "-" + sample.pos[i] # e.g. "bear-n"
                all_senses = self.lu2senses[dict_key] # TODO: parametrize this
                # Take care of cases of multiple labels, e.g. "01104026-a,00357790-a"
                these_senses = label.split(",")
                num_labels = len(these_senses)
                for sense in these_senses:
                    if sense in self.src2id:
                        synset_embedding = torch.Tensor(self.embeddings[self.src2id[sense]])
                        target_embed += synset_embedding
                        # target_classify[self.lemma2synsets[lemma_pos].index(synset)] = 1.0
                target_embed /= num_labels
                # Either use separate softmax layers per lemma/LU, or use one large softmax for all senses
                if self.single_softmax is True:
                    sense = random.choice(these_senses)
                    if sense in self.known_senses:
                        targets_classify.append(self.known_senses[sense])
                    else:
                        targets_classify.append(self.known_senses["UNKNOWN"])
                else:
                    targets_classify.append(self.lu2senses[dict_key].index(random.choice(these_senses)))
                lengths_labels += 1
                # Pick negative targets too
                # Copy the list of synsets, so that we don't change the dict
                neg_options = copy.copy(all_senses)
                for sense in these_senses:
                    # Get rid of the gold synsets
                    neg_options.remove(sense)
                while True:
                    # If no synsets remain in the list, pick any synset at random
                    if len(neg_options) == 0:
                        neg_sense = random.choice(list(self.src2id))
                        break
                    neg_sense = random.choice(neg_options)
                    # Make sure the chosen synset has a matching embedding, else remove from list
                    if neg_sense in self.src2id:
                        break
                    else:
                        neg_options.remove(neg_sense)
                neg_target = torch.Tensor(self.embeddings[self.src2id[neg_sense]])
            targets_embed.append(target_embed)
            neg_targets.append(neg_target)
            # targets_classify.append(target_classify)
        data = {"forms": sample.forms,
                "lemmas": sample.lemmas,
                "lemmas_pos": sample.lemmas_pos,
                "length": sample.length,
                # "lengths_labels": torch.tensor(lengths_labels, dtype=torch.long),
                "lengths_labels": lengths_labels,
                "pos": sample.pos,
                "senses": sample.senses,
                "entities": sample.entities,
                "inputs": torch.tensor(inputs, dtype=torch.long).to(self.device),
                "targets_embed": torch.stack(targets_embed).clone().detach().to(self.device),
                "neg_targets": torch.stack(neg_targets).clone().detach().to(self.device),
                "sentence": sample.sentence_str,
                "targets_classify": torch.tensor(targets_classify, dtype=torch.long).to(self.device),
                "targets_pos": torch.tensor(targets_pos, dtype=torch.long).to(self.device),
                "targets_ner": torch.tensor(targets_ner, dtype=torch.long).to(self.device),
                "mask": torch.tensor(mask, dtype=torch.bool).to(self.device),
                "pos_mask": torch.tensor(pos_mask, dtype=torch.bool).to(self.device),
                "ner_mask": torch.tensor(ner_mask, dtype=torch.bool).to(self.device),
                "batch_layers": self.batch_layers}
        return data

    def parse_tsv(self, dataset_path, max_length, pos_map=None, pos_filter=False):
        files = []
        if os.path.isfile(dataset_path):
            files = [dataset_path]
        elif os.path.isdir(dataset_path):
            files = [os.path.join(dataset_path, f_name) for f_name in os.listdir(dataset_path)]
        data = []
        for f in files:
            sentences = parse(open(f, "r").read(), CUSTOM_FIELDS)
            for sentence in sentences:
                sample = Sample()
                sentence_str = []
                for token in sentence:
                    sample.forms.append(token["form"])
                    lemma = token["lemma"]
                    # lemma = lemma.replace("'", "APOSTROPHE_")
                    # lemma = lemma.replace(".", "DOT_")
                    pos = token["pos"]
                    if pos_map is not None:
                        if pos in pos_map:
                            pos = pos_map[pos]
                    # pos = POS_MAP[pos] if pos in POS_MAP else pos
                    # lemma_pos = lemma + "-" + POS_MAP_SIMPLE[pos] if pos in POS_MAP_SIMPLE else lemma # TODO: handle different POS tags
                    senses = token["senses"]
                    # if senses != "_":
                    #     pos_id = senses.split(',')[0][-1]
                    # else:
                    #     pos_id = ""
                    if token["lu"] == "_":
                        if pos in POS_MAP_SIMPLE:
                            pos_id = POS_MAP_SIMPLE[pos]
                        else:
                            pos_id = pos
                        lemma_pos = lemma + "-" + pos_id
                    else:
                        lemma_pos = token["lu"].replace(" ", "_").lower()
                    entity = token['entity']
                    sample.lemmas.append(lemma)
                    sample.pos.append(pos)
                    sample.lemmas_pos.append(lemma_pos)
                    sample.entities.append(entity)
                    sample.senses.append(senses)
                    if pos_filter is True:
                        lemma_id = lemma_pos
                    else:
                        lemma_id = lemma
                    # TODO Why do I need this?
                    # if senses != "_":
                    #     if lemma_id not in self.lu2senses:
                    #         self.lu2senses[lemma_id] = [senses]
                    #     else:
                    #         if senses not in self.lu2senses[lemma_id]:
                    #             self.lu2senses[lemma_id].append(senses)
                    self.known_lus.add(lemma_id)
                    self.known_pos.add(pos)
                    self.known_entity_tags.add(entity)
                    sentence_str.append(token["form"])
                sample.sentence_str = (" ").join(sentence_str)
                sample.length = len(sample.forms) if len(sample.forms) < max_length else max_length
                # Take care to pad all sequences to the same length
                sample.forms = (sample.forms + (max_length - len(sample.forms)) * ["<PAD>"])[:max_length]
                sample.lemmas = (sample.lemmas + (max_length - len(sample.lemmas)) * ["<PAD>"])[:max_length]
                sample.pos = (sample.pos + (max_length - len(sample.pos)) * ["_"])[:max_length]
                sample.lemmas_pos = (sample.lemmas_pos + (max_length - len(sample.lemmas_pos)) * ["<PAD>"])[:max_length]
                sample.senses = (sample.senses + (max_length - len(sample.senses)) * ["_"])[:max_length]
                sample.entities = (sample.entities + (max_length - len(sample.entities)) * ["_"])[:max_length]
                data.append(sample)
        return data

def transform_uef2tsv(path_to_dataset, output_path):
    data, keys = "", ""
    for f in os.listdir(path_to_dataset):
        if f.endswith(".xml"):
            data = f
        elif f.endswith(".txt"):
            keys = f
    codes2keys = {}
    f_codes2keys = open(os.path.join(path_to_dataset, keys), "r")
    for line in f_codes2keys.readlines():
        fields = line.strip().split()
        codes2keys[fields[0]] = fields[1:]
    corpora = ET.parse(os.path.join(path_to_dataset, data)).getroot().findall("corpus")
    for corpus in corpora:
        texts = corpus.findall("text")
        sentence_str = []
        for text in texts:
            sentences = text.findall("sentence")
            for sentence in sentences:
                this_sent = ""
                elements = sentence.findall(".//")
                for element in elements:
                    wordform = element.text
                    lemma = element.get("lemma")
                    pos = element.get("pos")
                    if element.tag == "instance":
                        synsets = [sensekey2synset[key] for key in codes2keys[element.get("id")]]
                    else:
                        synsets = ["_"]
                    this_sent += "\t".join([wordform, lemma, pos, "_", ",".join(synsets), "_"]) + "\n"
                sentence_str.append(this_sent)
    dataset_str = "\n".join(sentence_str)
    with open(os.path.join(output_path, data.split(".")[0] + ".tsv"), "w", encoding="utf-8") as f:
        f.write(dataset_str)
    return

# def transform_original2tsv(path_to_dataset, output_path):
#     for f_name in os.listdir(path_to_dataset):
#         print(f_name)
#         # with open(os.path.join(path_to_dataset, f_name), "r") as f:
#         # # context = ET.parse(os.path.join(path_to_dataset, f_name)).getroot().get("context")
#         # #     it = itertools.chain('<root>', f, '</root>')
#         # #     f_contents = f.read()
#         #     # root = ET.fromstring('<root>\n' + f_contents + '\n</root>')
#         #     doc = ET.parse(f)
#         paragraphs = ET.parse(os.path.join(path_to_dataset, f_name)).getroot().findall("contextfile")[0].findall("context")[0].findall("p")
#         sentence_str = []
#         for p in paragraphs:
#             sentences = p.findall("s")
#             for sent in sentences:
#                 this_sent = ""
#                 wfs = sent.findall("wf") + sent.findall("punc")
#                 for wf in wfs:
#                     wordform = wf.text
#                     lemma = wf.get("lemma")
#                     if lemma is None:
#                         lemma = wordform
#                     pos = wf.get("pos")
#                     if pos is None:
#                         pos = "."
#                     synsets = wf.get("lexsn")
#                     if synsets is not None:
#                         synsets = synsets.split(";")
#                         # if lemma == "rotting":
#                         #     continue
#                         # for synset in synsets:
#                         #     sense = lemma + "%" + synset
#                         #     if sense not in sensekey2synset:
#                         #         continue
#                         synsets = [sensekey2synset[lemma + "%" + synset] for synset in synsets
#                                    if lemma + "%" + synset in sensekey2synset]
#                     else:
#                         synsets = ["_"]
#                     this_sent += "\t".join([wordform, lemma, pos, ",".join(synsets)]) + "\n"
#                 sentence_str.append(this_sent)
#         dataset_str = "\n".join(sentence_str)
#         with open(os.path.join(output_path, f_name + ".tsv"), "w") as f:
#             f.write(dataset_str)
#     return

def transform_original2tsv(path_to_dataset, output_path):
    for f_name in os.listdir(path_to_dataset):
        print(f_name)
        # with open(os.path.join(path_to_dataset, f_name), "r") as f:
        # # context = ET.parse(os.path.join(path_to_dataset, f_name)).getroot().get("context")
        # #     it = itertools.chain('<root>', f, '</root>')
        # #     f_contents = f.read()
        #     # root = ET.fromstring('<root>\n' + f_contents + '\n</root>')
        #     doc = ET.parse(f)
        paragraphs = ET.parse(os.path.join(path_to_dataset, f_name)).getroot().findall("contextfile")[0].findall("context")[0].findall("p")
        sentence_str = []
        for p in paragraphs:
            sentences = p.findall("s")
            for sent in sentences:
                this_sent = ""
                wfs = sent.findall("wf") + sent.findall("punc")
                for wf in wfs:
                    wordform = wf.text
                    lemma = wf.get("lemma")
                    if lemma is None:
                        lemma = wordform
                    pos = wf.get("pos")
                    if pos is None:
                        pos = "."
                    synsets = wf.get("lexsn")
                    if synsets is not None:
                        synsets = synsets.split(";")
                        # if lemma == "rotting":
                        #     continue
                        # for synset in synsets:
                        #     sense = lemma + "%" + synset
                        #     if sense not in sensekey2synset:
                        #         continue
                        synsets = [sensekey2synset[lemma + "%" + synset] for synset in synsets
                                   if lemma + "%" + synset in sensekey2synset]
                    else:
                        synsets = ["_"]
                    this_sent += "\t".join([wordform, lemma, pos, ",".join(synsets), "_"]) + "\n"
                sentence_str.append(this_sent)
        dataset_str = "\n".join(sentence_str)
        with open(os.path.join(output_path, f_name + ".tsv"), "w") as f:
            f.write(dataset_str)
    return

def transform_bsnlp2tsv(path_to_dataset, output_path, btb=False, separator="    "):
    for f_name in os.listdir(path_to_dataset):
        new_content = ""
        with open(os.path.join(path_to_dataset, f_name), "r") as f:
            text = f.read()
            text = text.replace("</S>" + "\t" + "E-SENT\n<S>" + "\t" + "B-SENT", "")
            text = text.replace("</S>" + "    " + "E-SENT\n<S>" + "    " + "B-SENT", "")
            text = text.replace("<S>" + "\t" + "B-SENT\n", "")
            text = text.replace("<S>" + "    " + "B-SENT\n", "")
            text = text.replace("\n</S>" + "\t" + "E-SENT\n", "")
            text = text.replace("\n</S>" + "    " + "E-SENT\n", "")
            if btb is not True:
                for line in text.split("\n"):
                    if line == "":
                        new_content += "\n"
                    else:
                        fields = line.split("\t")
                        if len(fields) < 2:
                            pass
                        new_line = fields[0] + "\t_\t_\t_\t" + fields[1] + "\n"
                        new_content += new_line
            else:
                text = text.replace("\t0\t", "\t_\t")
                # text = text.replace("\tLat", "\tunknown\tLat")
                # text = text.replace("Unknown\t_", "unknown\tunknown\t_")
                new_content = text
        with open(os.path.join(output_path, f_name + ".tsv"), "w") as new_f:
            new_f.write(new_content.rstrip("\n"))
    return

def fix_semcor_xml(path_to_dataset, output_path):
    for f_name in os.listdir(path_to_dataset):
        new_content = "<root>\n"
        with open(os.path.join(path_to_dataset, f_name), "r") as f:
            for line in f.readlines():
                new_line = line
                new_line = new_line.replace(">&<", ">&#038;<")
                opening_tag = line.split(">")[0][1:]
                fields = opening_tag.split(" ")
                for field in fields:
                    if "=" in field and "\"-\"" not in field:
                        value = field.split("=")[1]
                        new_field = field.split("=")[0] + "=" + "\"" + value + "\""
                        new_line = new_line.replace(field, new_field)
                new_content += new_line
            new_content += "</root>"
        with open(os.path.join(output_path, f_name + ".xml"), "w") as new_f:
            new_f.write(new_content)
    return

def load_embeddings(embeddings_path, use_flair=False):
    """Loads an embedding model with gensim

    Args:
        embeddings_path: A string, the path to the model

    Returns:
        embeddings: A list of vectors
        src2id: A dictionary, maps strings to integers in the list
        id2src: A dictionary, maps integers in the list to strings

    """
    # if use_flair is True:
    #     binary = True
    # else:
    #     binary = False
    model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path,
                                                            binary=False,
                                                            datatype=numpy.float32)
    embeddings = model.vectors
    zeros = numpy.zeros(len(embeddings[0]), dtype=numpy.float32)
    # Gensim provides a dict mapping ints to words (strings)
    id2src = model.index2word
    # We want to be able to go from words to ints as well
    src2id = {v: (k + 1) for k, v in enumerate(id2src)}
    # Insert a zero vector for the padding symbol
    src2id["<PAD>"] = 0
    embeddings = numpy.insert(embeddings, 0, copy.copy(zeros), axis=0)
    # Make sure we have a vector for unknown inputs
    if "UNK" not in src2id:
        if "unk" in src2id:
            src2id["<UNK>"] = src2id["unk"]
            id2src[src2id["<UNK>"]] = "<UNK>"
            del src2id["unk"]
        else:
            unk = numpy.zeros(len(embeddings[0]), dtype=numpy.float32)
            src2id["<UNK>"] = len(src2id)
            embeddings = numpy.concatenate((embeddings, [unk]))
    return embeddings, src2id, id2src

def fiter_embeddings(data, f_embeddings, f_emb_out, f_oov):
    attested_forms = set()
    embedded_forms = set()
    filtered_embeddings = ""
    for f_name in os.listdir(data):
        with open(os.path.join(data, f_name), "r") as f:
            for line in f.readlines():
                attested_forms.add(line.split("\t")[0])
    with open(f_embeddings, "r") as embeddings:
        for line in embeddings.readlines():
            word = line.split(" ")[0]
            if word in attested_forms:
                filtered_embeddings += line + "\n"
            embedded_forms.add(word)
    with open(f_emb_out, "w") as emb_out:
        emb_out.write(filtered_embeddings)
    oov_forms = attested_forms.difference(embedded_forms)
    with open(f_oov, "a") as oov:
        for form in oov_forms:
            oov.write(form + "\n")
    return


def get_wordnet_lexicon(lexicon_path, pos_filter=False):
    """Reads the WordNet dictionary

    Args:
        lexicon_path: A string, the path to the dictionary

    Returns:
        lemma2synsets: A dictionary, maps lemmas to synset IDs

    """
    lemma2synsets, max_labels = {}, 0
    lexicon = open(lexicon_path, "r")
    for line in lexicon.readlines():
        fields = line.split(" ")
        lemma_base, synsets = fields[0], fields[1:]
        if len(synsets) > max_labels:
            max_labels = len(synsets)
        for i, entry in enumerate(synsets):
            synset = entry[:10].strip()
            if pos_filter:
                pos = synset[-1]
                lemma = lemma_base + "-" + pos
            else:
                lemma = lemma_base
            # lemma = lemma.replace("'", "APOSTROPHE_")
            # lemma = lemma.replace(".", "DOT_")
            if lemma not in lemma2synsets:
                lemma2synsets[lemma] = [synset]
            else:
                lemma2synsets[lemma].append(synset)
    lemma2synsets = collections.OrderedDict(sorted(lemma2synsets.items()))
    return lemma2synsets, max_labels

def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)

def get_framenet_lexicon(lexicon_path):
    lu2frames = {}
    with open(lexicon_path, "r") as lexicon:
        for line in lexicon.readlines():
            fields = line.strip().split(" ")
            lu, frames = fields[0], fields[1:]
            lu = rreplace(lu, ".", "-", 1)
            lu2frames[lu] = frames
    return lu2frames

def get_pos_tagset(f_mapping, granularity="medium"):

    postag2postag = {}
    known_postags = set()
    doc = ET.parse(f_mapping)
    root = doc.getroot()
    pairs = root.findall("pair")
    if granularity == "medium":
        for pair in pairs:
            specific_tag = pair.find("item").text
            coarse_tag = pair.find("item1").text
            postag2postag[specific_tag] = coarse_tag
            known_postags.add(coarse_tag)
    elif granularity == "coarse":
        for pair in pairs:
            specific_tag = pair.find("item").text
            coarse_tag = pair.find("item2")
            postag2postag[specific_tag] = coarse_tag
            known_postags.add(coarse_tag)
    return postag2postag

if __name__ == "__main__":
    f_sensekey2synset = "/home/lenovo/dev/neural-wsd/data/sensekey2synset.pkl"
    sensekey2synset = pickle.load(open(f_sensekey2synset, "rb"))
    transform_uef2tsv("/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/WSD_Training_Corpora/SemCor",
                      "/home/lenovo/dev/neural-wsd/data/combined_WSD_FN")
    # f_dataset = "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv/semeval2007.tsv"
    # sentences = parse_tsv(open(f_dataset, "r").read(), 50)
    # transform_original2tsv("/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/WSD_Training_Corpora/SemCor",
    #                        "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv2")
    # fix_semcor_xml("/home/lenovo/dev/neural-wsd/data/semcor3.0/all", "/home/lenovo/dev/neural-wsd/data/semcor3.0/all_fixed1")
    # transform_bsnlp2tsv('/home/lenovo/dev/PostDoc/LREC/BSLNE/BSNLP_test', '/home/lenovo/dev/PostDoc/LREC/BSLNE/tsv_test')
    # fiter_embeddings('/home/lenovo/dev/PostDoc/LREC/BSLNE/tsv_all',
    #                  '/home/lenovo/dev/PostDoc/LREC/cc.bg.300.vec',
    #                  '/home/lenovo/dev/PostDoc/LREC/cc.bg.300.vec_FILTERED',
    #                  '/home/lenovo/dev/PostDoc/LREC/oov.txt')
    # transform_bsnlp2tsv("/home/lenovo/dev/PostDoc/LREC/BTB_BIOFull_WSD_021119",
    #                     "/home/lenovo/dev/PostDoc/LREC/BTB_BIOFull_WSD_021119_TSV",
    #                     True,
    #                     "\t")
    print("This is the end.")