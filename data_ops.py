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

f_sensekey2synset = "/home/lenovo/dev/neural-wsd/data/sensekey2synset.pkl"
sensekey2synset = pickle.load(open(f_sensekey2synset, "rb"))
CUSTOM_FIELDS = ('form', 'lemma', 'pos', 'synsets', 'entity')

POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


class Sample():

    def __init__(self):
        ''' Simple structure to hold the sentence features '''
        self.length = 0
        self.forms = []
        self.lemmas = []
        self.pos = []
        self.lemmas_pos = []
        self.synsets = []
        self.entities = []

class WSDataset(Dataset):

    def __init__(self, tsv_data, src2id, embeddings, embeddings_dim, embeddings_input, max_labels, lemma2synsets,
                 single_softmax, known_synsets=None, pos_map=None):
        # Our data has some pretty long sentences, so we will set a large max length
        # Alternatively, can throw them out or truncate them
        self.src2id = src2id
        self.embeddings = embeddings
        self.embeddings_dim = embeddings_dim
        self.embeddings_input = embeddings_input
        self.max_labels = max_labels
        self.lemma2synsets = lemma2synsets
        self.known_lemmas, self.known_pos, self.known_entity_tags = set(), set(), set()
        if pos_map is not None:
            pos_map = get_pos_tagset(pos_map, "medium")
        self.data = self.parse_tsv(tsv_data, 300, pos_map)
        self.known_lemmas, self.known_pos, self.known_entity_tags = \
            sorted(self.known_lemmas), sorted(self.known_pos), sorted(self.known_entity_tags)
        self.single_softmax = single_softmax
        if self.single_softmax is True:
            if known_synsets is None:
                self.known_synsets = {"UNKNOWN" : 0}
                id = 1
                for lemma in self.known_lemmas:
                    if lemma in lemma2synsets:
                        for synset in lemma2synsets[lemma]:
                            if synset not in self.known_synsets:
                                self.known_synsets[synset] = id
                                id += 1
            else:
                self.known_synsets = known_synsets
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
        for i, label in enumerate(sample.synsets):
            lemma = sample.lemmas[i]
            lemma_pos = sample.lemmas_pos[i]
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
                all_synsets = self.lemma2synsets[lemma] # TODO: parametrize this
                # Take care of cases of multiple labels, e.g. "01104026-a,00357790-a"
                these_synsets = label.split(",")
                num_labels = len(these_synsets)
                for synset in these_synsets:
                    if synset in self.src2id:
                        synset_embedding = torch.Tensor(self.embeddings[self.src2id[synset]])
                        target_embed += synset_embedding
                        # target_classify[self.lemma2synsets[lemma_pos].index(synset)] = 1.0
                target_embed /= num_labels
                # Either use separate softmax layers per lemma/LU, or use one large softmax for all senses
                if self.single_softmax is True:
                    synset = random.choice(these_synsets)
                    if synset in self.known_synsets:
                        targets_classify.append(self.known_synsets[synset])
                    else:
                        targets_classify.append(self.known_synsets["UNKNOWN"])
                else:
                    targets_classify.append(self.lemma2synsets[lemma_pos].index(random.choice(these_synsets)))
                lengths_labels += 1
                # Pick negative targets too
                # Copy the list of synsets, so that we don't change the dict
                neg_options = copy.copy(all_synsets)
                for synset in these_synsets:
                    # Get rid of the gold synsets
                    neg_options.remove(synset)
                while True:
                    # If no synsets remain in the list, pick any synset at random
                    if len(neg_options) == 0:
                        neg_synset = random.choice(list(self.src2id))
                        break
                    neg_synset = random.choice(neg_options)
                    # Make sure the chosen synset has a matching embedding, else remove from list
                    if neg_synset in self.src2id:
                        break
                    else:
                        neg_options.remove(neg_synset)
                neg_target = torch.Tensor(self.embeddings[self.src2id[neg_synset]])
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
                "synsets": sample.synsets,
                "entities": sample.entities,
                "inputs": torch.tensor(inputs, dtype=torch.long),
                "targets_embed": torch.stack(targets_embed).clone().detach(),
                "neg_targets": torch.stack(neg_targets).clone().detach(),
                "targets_classify": torch.tensor(targets_classify, dtype=torch.long),
                "targets_pos": torch.tensor(targets_pos, dtype=torch.long),
                "targets_ner": torch.tensor(targets_ner, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.bool),
                "pos_mask": torch.tensor(pos_mask, dtype=torch.bool),
                "ner_mask": torch.tensor(ner_mask, dtype=torch.bool)}
        return data

    def parse_tsv(self, dataset_path, max_length, pos_map=None):
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
                for token in sentence:
                    sample.forms.append(token["form"])
                    lemma = token["lemma"]
                    lemma = lemma.replace("'", "APOSTROPHE_")
                    lemma = lemma.replace(".", "DOT_")
                    pos = token["pos"]
                    if pos_map is not None:
                        if pos in pos_map:
                            pos = pos_map[pos]
                    # pos = POS_MAP[pos] if pos in POS_MAP else pos
                    lemma_pos = lemma + "-" + POS_MAP[pos] if pos in POS_MAP else pos
                    synsets = token["synsets"]
                    entity = token['entity']
                    sample.lemmas.append(lemma)
                    sample.pos.append(pos)
                    sample.lemmas_pos.append(lemma_pos)
                    sample.entities.append(entity)
                    sample.synsets.append(synsets)
                    if synsets != "_":
                        if lemma not in self.lemma2synsets:
                            self.lemma2synsets[lemma] = [synsets]
                        else:
                            if synsets not in self.lemma2synsets[lemma]:
                                self.lemma2synsets[lemma].append(synsets)
                    # self.known_lemmas.add(lemma_pos) TODO: parametrize this
                    self.known_lemmas.add(lemma)
                    self.known_pos.add(pos)
                    self.known_entity_tags.add(entity)
                sample.length = len(sample.forms) if len(sample.forms) < max_length else max_length
                # Take care to pad all sequences to the same length
                sample.forms = (sample.forms + (max_length - len(sample.forms)) * ["<PAD>"])[:max_length]
                sample.lemmas = (sample.lemmas + (max_length - len(sample.lemmas)) * ["<PAD>"])[:max_length]
                sample.pos = (sample.pos + (max_length - len(sample.pos)) * ["_"])[:max_length]
                sample.lemmas_pos = (sample.lemmas_pos + (max_length - len(sample.lemmas_pos)) * ["<PAD>"])[:max_length]
                sample.synsets = (sample.synsets + (max_length - len(sample.synsets)) * ["_"])[:max_length]
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
                    this_sent += "\t".join([wordform, lemma, pos, ",".join(synsets)]) + "\n"
                sentence_str.append(this_sent)
    dataset_str = "\n".join(sentence_str)
    with open(os.path.join(output_path, data.split(".")[0] + ".tsv"), "w", encoding="utf-8") as f:
        f.write(dataset_str)
    return

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
                    this_sent += "\t".join([wordform, lemma, pos, ",".join(synsets)]) + "\n"
                sentence_str.append(this_sent)
        dataset_str = "\n".join(sentence_str)
        with open(os.path.join(output_path, f_name + ".tsv"), "w") as f:
            f.write(dataset_str)
    return

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
                    this_sent += "\t".join([wordform, lemma, pos, ",".join(synsets)]) + "\n"
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
                text = text.replace("\tLat", "\tunknown\tLat")
                text = text.replace("Unknown\t_", "unknown\tunknown\t_")
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

def load_embeddings(embeddings_path):
    """Loads an embedding model with gensim

    Args:
        embeddings_path: A string, the path to the model

    Returns:
        embeddings: A list of vectors
        src2id: A dictionary, maps strings to integers in the list
        id2src: A dictionary, maps integers in the list to strings

    """
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
            lemma = lemma.replace("'", "APOSTROPHE_")
            lemma = lemma.replace(".", "DOT_")
            if lemma not in lemma2synsets:
                lemma2synsets[lemma] = [synset]
            else:
                lemma2synsets[lemma].append(synset)
    lemma2synsets = collections.OrderedDict(sorted(lemma2synsets.items()))
    return lemma2synsets, max_labels

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
    # transform_uef2tsv("C:\Work\dev\wsd-resources\WSD_Unified_Evaluation_Datasets\senseval2",
    #                   "C:\Work\dev\wsd-resources\data_tsv")
    # f_dataset = "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv/semeval2007.tsv"
    # sentences = parse_tsv(open(f_dataset, "r").read(), 50)
    # transform_original2tsv("C:\Work\dev\wsd-resources\SemCor",
    #                        "C:\Work\dev\wsd-resources\data_tsv")
    # fix_semcor_xml("/home/lenovo/dev/neural-wsd/data/semcor3.0/all", "/home/lenovo/dev/neural-wsd/data/semcor3.0/all_fixed1")
    # transform_bsnlp2tsv('/home/lenovo/dev/PostDoc/LREC/BSLNE/BSNLP_test', '/home/lenovo/dev/PostDoc/LREC/BSLNE/tsv_test')
    # fiter_embeddings('/home/lenovo/dev/PostDoc/LREC/BSLNE/tsv_all',
    #                  '/home/lenovo/dev/PostDoc/LREC/cc.bg.300.vec',
    #                  '/home/lenovo/dev/PostDoc/LREC/cc.bg.300.vec_FILTERED',
    #                  '/home/lenovo/dev/PostDoc/LREC/oov.txt')
    transform_bsnlp2tsv("/home/lenovo/dev/PostDoc/LREC/BSLNE/TestReady-20191129",
                        "/home/lenovo/dev/PostDoc/LREC/BSLNE/tsv_test",
                        False,
                        "\t")
    print("This is the end.")