import os
import collections
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
CUSTOM_FIELDS = ('form', 'lemma', 'pos', 'synsets')

POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

class WSDataset(Dataset):

    def __init__(self, tsv_file, src2id, embeddings, embeddings_dim, max_labels, lemma2synsets):
        # Our data has some pretty long sentences, so we will set a large max length
        # Alternatively, can throw them out or truncate them
        self.data = parse_tsv(open(tsv_file, "r").read(), max_length=300)
        self.src2id = src2id
        self.embeddings = embeddings
        self.embeddings_dim = embeddings_dim
        self.max_labels = max_labels
        self.lemma2synsets = lemma2synsets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ''' Prepare one sample sentence of the data '''
        # Get a sentence
        sample = self.data[idx]
        # Get an integer ID for each lemma in the sentence (<UNK> if unfamiliar)
        # Note that we are working with lemmas for the input, not the word forms
        inputs = [self.src2id[lemma] if lemma in self.src2id
                  else self.src2id["<UNK>"] for lemma in sample.lemmas]
        targets, neg_targets, targets_labels, mask, lengths_labels = [], [], [], [], []
        for i, label in enumerate(sample.synsets):
            target, neg_target, = torch.zeros(self.embeddings_dim), torch.zeros(self.embeddings_dim)
            if label == "_":
                mask.append(False)
                lengths_labels.append(0)
                targets_labels.append(0)
            else:
                mask.append(True)
                lemma_pos = sample.lemmas[i] + "-" + POS_MAP[sample.pos[i]] # e.g. "bear-n"
                all_synsets = self.lemma2synsets[lemma_pos]
                # Take care of cases of multiple labels, e.g. "01104026-a,00357790-a"
                these_synsets = label.split(",")
                num_labels = len(these_synsets)
                for synset in these_synsets:
                    if synset in self.src2id:
                        synset_embedding = torch.Tensor(self.embeddings[self.src2id[synset]])
                        target += synset_embedding
                targets_labels.append(self.lemma2synsets[lemma_pos].index(random.choice(these_synsets)))
                target /= num_labels
                lengths_labels.append(len(all_synsets))
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
            targets.append(target)
            neg_targets.append(neg_target)
        data = {"lemmas": sample.lemmas,
                "length": sample.length,
                "lengths_labels": torch.tensor(lengths_labels, dtype=torch.long),
                "pos": sample.pos,
                "synsets": sample.synsets,
                "inputs": torch.tensor(inputs, dtype=torch.long),
                "targets": torch.stack(targets).clone().detach(),
                "neg_targets": torch.stack(neg_targets).clone().detach(),
                "targets_labels": torch.tensor(targets_labels, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.bool)}
        return data

class Sample():

    def __init__(self):
        ''' Simple structure to hold the sentence features '''
        self.length = 0
        self.forms = []
        self.lemmas = []
        self.pos = []
        self.synsets = []

def parse_tsv(f_dataset, max_length):
    sentences = parse(f_dataset, CUSTOM_FIELDS)
    data = []
    for sentence in sentences:
        sample = Sample()
        for token in sentence:
            sample.forms.append(token["form"])
            lemma = token["lemma"]
            lemma = lemma.replace("'", "APOSTROPHE_")
            lemma = lemma.replace(".", "DOT_")
            sample.lemmas.append(lemma)
            sample.pos.append(token["pos"])
            sample.synsets.append(token["synsets"])
        sample.length = len(sample.forms)
        # Take care to pad all sequences to the same length
        sample.forms += (max_length - len(sample.forms)) * ["<PAD>"]
        sample.lemmas += (max_length - len(sample.lemmas)) * ["<PAD>"]
        sample.pos += (max_length - len(sample.pos)) * "_"
        sample.synsets += (max_length - len(sample.synsets)) * "_"
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
    with open(os.path.join(output_path, data.split(".")[0] + ".tsv"), "w") as f:
        f.write(dataset_str)
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

if __name__ == "__main__":
    # transform_uef2tsv("/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/WSD_Training_Corpora/SemCor",
    #                   "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv")
    f_dataset = "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv/semeval2007.tsv"
    sentences = parse_tsv(open(f_dataset, "r").read(), 50)
    print("This is the end.")