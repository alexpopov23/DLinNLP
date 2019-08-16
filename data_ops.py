import os
import pickle
import numpy
import gensim
import copy
import torch

import xml.etree.ElementTree as ET

from conllu import parse
from torch.utils.data import Dataset

f_sensekey2synset = "/home/lenovo/dev/neural-wsd/data/sensekey2synset.pkl"
sensekey2synset = pickle.load(open(f_sensekey2synset, "rb"))
CUSTOM_FIELDS = ('form', 'lemma', 'pos', 'synsets')

class WSDataset(Dataset):

    def __init__(self, tsv_file, src2id, embeddings, embeddings_dim):
        self.data = parse_tsv(open(tsv_file, "r").read(), 100)
        self.src2id = src2id
        self.embeddings = embeddings
        self.embeddings_dim = embeddings_dim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        inputs = [self.src2id[lemma] if lemma in self.src2id else self.src2id["<UNK>"] for lemma in sample.lemmas]
        mask, targets = [], []
        for label in sample.synsets:
            target = torch.zeros(300)
            if label == "_":
                mask.append(False)
            else:
                mask.append(True)
                synsets = label.split(",")
                for synset in synsets:
                    if synset in self.src2id:
                        synset_embedding = torch.Tensor(self.embeddings[self.src2id[synset]])
                        target += synset_embedding
                target /= len(synsets)
            targets.append(target)
        return sample.lemmas, sample.pos, torch.tensor(inputs, dtype=torch.long), torch.tensor(torch.stack(targets)), \
               sample.synsets, torch.tensor(mask, dtype=bool), sample.length

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


if __name__ == "__main__":
    # transform_uef2tsv("/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/WSD_Training_Corpora/SemCor",
    #                   "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv")
    f_dataset = "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv/semeval2007.tsv"
    sentences = parse_tsv(open(f_dataset, "r").read(), 50)
    print("This is the end.")