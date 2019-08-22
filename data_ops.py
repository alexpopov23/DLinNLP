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

pos_map = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

class WSDataset(Dataset):

    def __init__(self, tsv_file, src2id, embeddings, embeddings_dim, lemma2synsets, input_synsets=False):
        self.data = parse_tsv(open(tsv_file, "r").read(), 300)
        self.src2id = src2id
        self.embeddings = embeddings
        self.embeddings_dim = embeddings_dim
        self.lemma2synsets = lemma2synsets
        self.input_synsets = input_synsets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.input_synsets:
            inputs = []
            for i, synsets in enumerate(sample.synsets):
                lemma = sample.lemmas[i]
                if synsets != "_":
                    synsets_lemma = synsets.split(",") + [lemma]
                    input = random.choice(synsets_lemma)
                    inputs.append(self.src2id[input] if input in self.src2id else self.src2id["<UNK>"])
                else:
                    inputs.append(self.src2id[lemma] if lemma in self.src2id else self.src2id["<UNK>"])
        else:
            inputs = [self.src2id[lemma] if lemma in self.src2id else self.src2id["<UNK>"] for lemma in sample.lemmas]
        mask, targets, lemmas, gold_synsets, neg_targets = [], [], [] , [], []
        for i, label in enumerate(sample.synsets):
            target, neg_target = torch.zeros(300), torch.zeros(300)
            if label == "_":
                mask.append(False)
            else:
                mask.append(True)
                these_synsets = label.split(",")
                for synset in these_synsets:
                    if synset in self.src2id:
                        synset_embedding = torch.Tensor(self.embeddings[self.src2id[synset]])
                        target += synset_embedding
                target /= len(these_synsets)
                # Pick negative targets too
                lemma_pos = sample.lemmas[i] + "-" + pos_map[sample.pos[i]]
                neg_options = copy.copy(self.lemma2synsets[lemma_pos])
                for synset in these_synsets:
                    neg_options.remove(synset)
                while True:
                    if len(neg_options) == 0:
                        neg_synset = random.choice(list(self.src2id))
                        break
                    neg_synset = random.choice(neg_options)
                    if neg_synset in self.src2id:
                        break
                    else:
                        neg_options.remove(neg_synset)
                neg_target = torch.Tensor(self.embeddings[self.src2id[neg_synset]])
            targets.append(target)
            neg_targets.append(neg_target)
        data = {"lemmas": sample.lemmas,
                "pos": sample.pos,
                "inputs": torch.tensor(inputs, dtype=torch.long),
                "targets": torch.stack(targets).clone().detach(),
                "synsets": sample.synsets,
                "mask": torch.tensor(mask, dtype=bool),
                "length": sample.length,
                "neg_targets": torch.stack(neg_targets).clone().detach()}
        return data

class Sample():

    def __init__(self):
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
            sample.lemmas.append(token["lemma"])
            sample.pos.append(token["pos"])
            sample.synsets.append(token["synsets"])
        sample.length = len(sample.forms)
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
    embeddings_model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=False,
                                                                       datatype=numpy.float32)
    embeddings = embeddings_model.syn0
    zeros = numpy.zeros(len(embeddings[0]), dtype=numpy.float32)
    id2src = embeddings_model.index2word
    src2id = {v:(k+1) for k, v in enumerate(id2src)}
    src2id["<PAD>"] = 0
    # src2id["<START>"] = 1
    embeddings = numpy.insert(embeddings, 0, copy.copy(zeros), axis=0)
    embeddings = numpy.insert(embeddings, 0, copy.copy(zeros), axis=0)
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
    lemma2synsets = {}
    lexicon = open(lexicon_path, "r")
    # max_synsets = 0
    for line in lexicon.readlines():
        fields = line.split(" ")
        lemma_base, synsets = fields[0], fields[1:]
        # if len(synsets) > max_synsets:
        #     max_synsets = len(synsets)
        for i, entry in enumerate(synsets):
            synset = entry[:10].strip()
            if pos_filter:
                pos = synset[-1]
                lemma = lemma_base + "-" + pos
            else:
                lemma = lemma_base
            if lemma not in lemma2synsets:
                lemma2synsets[lemma] = [synset]
            else:
                lemma2synsets[lemma].append(synset)
    lemma2synsets = collections.OrderedDict(sorted(lemma2synsets.items()))
    return lemma2synsets

if __name__ == "__main__":
    # transform_uef2tsv("/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/WSD_Training_Corpora/SemCor",
    #                   "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv")
    f_dataset = "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv/semeval2007.tsv"
    sentences = parse_tsv(open(f_dataset, "r").read(), 50)
    print("This is the end.")