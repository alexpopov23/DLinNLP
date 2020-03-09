import collections

import numpy
import torch
import torch.nn as nn

from flair.data import Sentence
from flair.embeddings import BytePairEmbeddings, CharacterEmbeddings, FastTextEmbeddings, FlairEmbeddings, StackedEmbeddings, WordEmbeddings
from torch.nn.utils.rnn import pad_sequence

POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

class WSDModel(nn.Module):

    def __init__(self, lang, embeddings_dim, embedding_weights, hidden_dim, hidden_layers, dropout,
                 output_layers=["embed_wsd"], lemma2synsets=None, synsets2id={}, pos_tags={},
                 entity_tags={}, use_flair=False, combine_WN_FN=False):
        super(WSDModel, self).__init__()
        self.use_flair = use_flair
        self.combine_WN_FN = combine_WN_FN
        self.output_layers = output_layers
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.num_wsd_classes = 0
        self.synsets2id = synsets2id
        output_emb_dim = embeddings_dim
        if use_flair is True:
            if lang == "Bulgarian":
                # BG EMBEDDINGS:
                self.word_embeddings = StackedEmbeddings([
                    WordEmbeddings('/home/lenovo/dev/PostDoc/LREC/Embeddings/cc.bg.300.vec_FILTERED_OOV.gensim'),
                    # WordEmbeddings('bg'),
                    # FastTextEmbeddings('/home/lenovo/dev/PostDoc/LREC/Embeddings/cc.bg.300.vec_FILTERED_OOV'),
                    # Byte pair embeddings for English
                    BytePairEmbeddings('bg'),
                    FlairEmbeddings('bg-forward-fast'),
                    FlairEmbeddings('bg-backward-fast'),
                    CharacterEmbeddings()
                ])
            elif lang == "English":
                # EN EMBEDDINGS:
                self.word_embeddings = StackedEmbeddings([
                    WordEmbeddings('/home/lenovo/dev/word-embeddings/glove.6B/glove.6B.300d_MOD.gensim'),
                    WordEmbeddings('/home/lenovo/dev/word-embeddings/lemma_sense_embeddings/'
                                   'WN30WN30glConOne-C15I7S7N5_200M_syn_and_lemma_WikipediaLemmatized_FILTERED.gensim'),
                    # WordEmbeddings('bg'),
                    # FastTextEmbeddings('/home/lenovo/dev/PostDoc/LREC/Embeddings/cc.bg.300.vec_FILTERED_OOV'),
                    # Byte pair embeddings for English
                    BytePairEmbeddings('en'),
                    FlairEmbeddings('en-forward-fast'),
                    FlairEmbeddings('en-backward-fast'),
                    CharacterEmbeddings()
                ])
            else:
                print("Unknown language!")
                exit(1)
            embeddings_dim = self.word_embeddings.embedding_length
        else:
            self.word_embeddings = nn.Embedding.from_pretrained(embedding_weights,
                                                                freeze=True)
        self.lstm = nn.LSTM(embeddings_dim,
                            hidden_dim,
                            hidden_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout)
        if "embed_wsd" in self.output_layers:
            # We want output with the size of the lemma&synset embeddings
            self.emb_relu = nn.ReLU()
            self.output_emb = nn.Linear(2*hidden_dim, output_emb_dim)
        if "embed_frameID" in self.output_layers:
            self.emb_relu_frames = nn.ReLU()
            self.output_emb_frames = nn.Linear(2 * hidden_dim, output_emb_dim)
        if "classify_wsd" in self.output_layers:
            if len(self.synsets2id) > 0:
                self.output_classify = nn.Linear(2*hidden_dim, len(self.synsets2id))
                self.num_wsd_classes = len(self.synsets2id)
            else:
                lemma2layers = collections.OrderedDict()
                for lemma, synsets in lemma2synsets.items():
                    lemma2layers[lemma] = nn.Linear(2*hidden_dim, len(synsets))
                    if len(synsets) > self.num_wsd_classes:
                        self.num_wsd_classes = len(synsets)
                self.classifiers = nn.Sequential(lemma2layers)
        if "pos_tagger" in self.output_layers:
            self.pos_tags = nn.Linear(2 * hidden_dim, len(pos_tags))
        if "ner" in self.output_layers:
            self.ner = nn.Linear(2 * hidden_dim, len(entity_tags))
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, lemmas, source_ids=None):
        if self.use_flair is True:
            X = data["sentence"]
            X = [Sentence(sent) for sent in X]
            self.word_embeddings.embed(X)
            X = [torch.stack([token.embedding for token in sentence]) for sentence in X]
            # pad_vector = torch.zeros(self.word_embeddings.embedding_length)
            X = pad_sequence(X, batch_first=True, padding_value=0.0)
        else:
            X = data["inputs"]
            X = self.word_embeddings(X)  # shape is [batch_size,max_length,embeddings_dim]
        X_lengths, mask, lemmas = data["length"], data["mask"], lemmas
        X = self.dropout(X)
        X = torch.nn.utils.rnn.pack_padded_sequence(X,
                                                    X_lengths,
                                                    batch_first=True,
                                                    enforce_sorted=False)
        X, _ = self.lstm(X)
        # pad_packed_sequence cuts the sequences in the batch to the greatest sequence length
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True) # shape is [batch_size, max_length_of_X, 2* hidden_layer]
        # Therefore, make sure mask has the same shape as X
        mask = mask[:, :X.shape[1]] # shape is [batch_size, max_length_of_X]
        mask = torch.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        X_wsd = torch.masked_select(X, mask)
        X_wsd = X_wsd.view(-1, 2 * self.hidden_dim)  # shape is [num_labels, 2*hidden_dim]
        outputs = {}
        for layer in self.output_layers:
            # if self.combine_WN_FN is True:
            #     outputs_emb = []
            #     for i, x in enumerate(torch.unbind(X_wsd)):
            #         if source_ids[i] == "WSD":
            #             output = self.dropout(self.output_emb(self.emb_relu(x)))
            #         elif source_ids[i] == "FrameID":
            #             output = self.dropout(self.output_emb_frames(self.emb_relu_frames(x)))
            #         outputs_emb.append(output)
            #     outputs["embed_wsd"] = outputs_emb
            # else:
            if layer == "embed_wsd" and layer in data["batch_layers"][0]:
                outputs["embed_wsd"] = self.dropout(self.output_emb(self.emb_relu(X_wsd)))
            if layer == "embed_frameID" and layer in data["batch_layers"][0]:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        outputs["embed_frameID"] = self.dropout(self.output_emb_frames(self.emb_relu_frames(X_wsd)))
            if layer == "classify_wsd" and layer in data["batch_layers"][0]:
                if len(self.synsets2id) > 0:
                    outputs["classify_wsd"] = self.dropout(self.output_classify(X_wsd))
                else:
                    outputs_classif = []
                    for i, x in enumerate(torch.unbind(X_wsd)):
                        # lemma_pos = lemmas[i] + "-" + POS_MAP[pos[i]]
                        output_classif = self.dropout(self.classifiers._modules[lemmas[i]](x))
                        outputs_classif.append(output_classif)
                    outputs_classif = pad_sequence(outputs_classif, batch_first=True, padding_value=-100)
                    outputs["classify_wsd"] = outputs_classif
            if layer == "pos_tagger" and layer in data["batch_layers"][0]:
                outputs["pos_tagger"] = pad_sequence(self.dropout(self.pos_tags(X)),
                                                     batch_first=True,
                                                     padding_value=-100)
            if layer == "ner" and layer in data["batch_layers"][0]:
                outputs["ner"] = pad_sequence(self.dropout(self.ner(X)),
                                              batch_first=True,
                                              padding_value=-100)
        return outputs

    def forward_old(self, X, X_lengths, mask, pos_mask, lemmas):
        X = self.word_embeddings(X) # shape is [batch_size,max_length,embeddings_dim]
        X = torch.nn.utils.rnn.pack_padded_sequence(X,
                                                    X_lengths,
                                                    batch_first=True,
                                                    enforce_sorted=False)
        X, _ = self.lstm(X)
        # pad_packed_sequence cuts the sequences in the batch to the greatest sequence length
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True) # shape is [batch_size, max_length_of_X, 2* hidden_layer]
        # Therefore, make sure mask has the same shape as X
        mask = mask[:, :X.shape[1]] # shape is [batch_size, max_length_of_X]
        pos_mask = pos_mask[:, :X.shape[1]] # shape is [batch_size, max_length_of_X]
        # Make mask broadcastable to X
        mask = torch.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        pos_mask = torch.reshape(pos_mask, (pos_mask.shape[0], pos_mask.shape[1], 1))
        # Select only RNN outputs that correspond to synset-tagged words in the data
        X_wsd = torch.masked_select(X, mask)
        # masked_select flattens the tensor, but we need it as matrix
        X_wsd = X_wsd.view(-1, 2 * self.hidden_dim) # shape is [num_labels, 2*hidden_dim]
        # Select also the words to be POS tagged
        X_pos = torch.masked_select(X, pos_mask)
        X_pos = X_pos.view(-1, 2 * self.hidden_dim)
        outputs = {}
        for layer in self.output_layers:
            if layer == "embed_wsd":
                outputs["embed_wsd"] = self.dropout(self.output_emb(X_wsd))
            elif layer == "classify_wsd":
                if len(self.synsets2id) > 0:
                    outputs["classify_wsd"] = self.dropout(self.output_classify(X_wsd))
                else:
                    outputs_classif = []
                    for i, x in enumerate(torch.unbind(X_wsd)):
                        # lemma_pos = lemmas[i] + "-" + POS_MAP[pos[i]]
                        output_classif = self.dropout(self.classifiers._modules[lemmas[i]](x))
                        outputs_classif.append(output_classif)
                    outputs_classif = pad_sequence(outputs_classif, batch_first=True, padding_value=-100)
                    outputs["classify_wsd"] = outputs_classif
            elif layer == "pos_tagger":
                outputs["pos_tagger"] = self.dropout(self.pos_tags(X_pos))
        return outputs

def embed_concepts(data, outputs, query, embedding_dim, loss_func, alpha):
    mask_embed = torch.reshape(data["mask"], (data["mask"].shape[0], data["mask"].shape[1], 1))
    targets_embed = torch.masked_select(data["targets_embed"], mask_embed)
    targets_embed = targets_embed.view(-1, embedding_dim)
    neg_targets = torch.masked_select(data["neg_targets"], mask_embed)
    neg_targets = neg_targets.view(-1, embedding_dim)
    # targets_classify = targets_classify.view(-1, max_labels)
    loss_embed = alpha * loss_func(outputs[query], targets_embed) + \
                 (1 - alpha) * (1 - loss_func(outputs[query], neg_targets))
    return

def classify_wsd():
    return

def pos_tagger():
    return

def ner():
    return


def calculate_accuracy_embedding(outputs, lemmas, gold_synsets, lemma2synsets, embeddings, src2id, pos_filter=True):
    matches, total = 0, 0
    cosine_similarity = torch.nn.CosineSimilarity()
    for i, output in enumerate(torch.unbind(outputs)):
        # if pos_filter:
        #     lemma = lemmas[i] + "-" + POS_MAP[pos[i]]
        # else:
        #     lemma = lemmas[i]
        lemma = lemmas[i]
        possible_synsets = lemma2synsets[lemma]
        synset_choice, max_similarity = "", -100.0
        for synset in possible_synsets:
            synset_embedding = embeddings[src2id[synset]] if synset in src2id else embeddings[src2id["<UNK>"]]
            cos_sim = cosine_similarity(output.view(1, -1), synset_embedding.view(1, -1))
            # cos_sim = torch.nn.CosineSimilarity(output.view(1, -1).detach().numpy(), synset_embedding.view(1, -1).detach().numpy())[0][0]
            # cos_sim = cosine_similarity(output.view(1, -1).detach().numpy(),
            #                             synset_embedding.view(1, -1).detach().numpy())[0][0]
            if cos_sim > max_similarity:
                max_similarity = cos_sim
                synset_choice = synset
        if synset_choice in gold_synsets[i].split(","):
            matches += 1
        total += 1
    return matches, total

def calculate_accuracy_classification_wsd(outputs, targets, default_disambiguations, lemmas=None, known_lemmas=None,
                                          synsets=None, lemma2synsets=None, synset2id=None, single_softmax=False):
    matches, total = 0, 0
    log = ""
    if single_softmax is False:
        choices = numpy.argmax(outputs, axis=1)
        # This loop makes sure that we take the 1st sense heuristics for lemmas unseen in training
        for i in default_disambiguations:
            choices[i] = 0
    else:
        choices = outputs
    for i, choice in enumerate(choices):
        if single_softmax is True:
            if targets[i] == 0: # i.e. if the synset is not attested in the training data
                if lemma2synsets[lemmas[i]][0] in synsets[i].split(","):
                    matches += 1
            else:
                permitted_synsets = lemma2synsets[lemmas[i]]
                max, max_synset = -100, ""
                for synset in permitted_synsets:
                    if synset in synset2id:
                        synset_activation = choice[synset2id[synset]]
                        if synset_activation > max:
                            max = synset_activation
                            max_synset = synset
                if max_synset in synsets[i].split(","):
                    matches += 1
                log += lemmas[i] + "\t" + ",".join(permitted_synsets) + "\t" + max_synset + "\t" + synsets[i] + "\n"
        elif lemma2synsets[lemmas[i]][choice] in synsets[i].split(","):
                matches += 1
        total += 1
    return matches, total, log

def calculate_accuracy_classification(outputs, targets):
    choices = torch.argmax(outputs, dim=1)
    comparison_tensor = torch.eq(choices, targets)
    matches = torch.sum(comparison_tensor).numpy()
    total = comparison_tensor.shape[0]
    return matches, total

def calculate_accuracy_crf(loss_func, outputs, mask, targets):
    choices = loss_func.decode(outputs, mask=mask)
    matches, total = 0, 0
    for i, seq in enumerate(choices):
        for j, choice in enumerate(seq):
            if choice == targets[i][j].item():
                matches += 1
            total += 1
    return matches, total

def calculate_f1_ner(outputs, targets, lengths, entity2id):
    id2entity = {i:entity for entity, i in entity2id.items()}
    buffer, entities, verbose = [], [], ""
    tag_collections = [outputs, targets]
    for collection in tag_collections:
        tags = []
        for i, seq in enumerate(collection):
            if len(buffer) != 0:
                tags.append(buffer)
                buffer = []
            for j, tag in enumerate(seq):
                if j >= lengths[i]:
                    break
                if tag == entity2id["O"]:
                    if len(buffer) != 0:    # if buffer not empty, then write the NE to list
                        tags.append(buffer)
                        buffer = []
                else:
                    if len(id2entity[tag].split("-")) < 2:
                        print("Here")
                    id1, id2 = id2entity[tag].split("-")    # get B/I and EVT/LOC/ORG/PER/PRO/etc tags
                    if id1 == "B":
                        if len(buffer) != 0:    # if buffer not empty, write the NE to list
                            tags.append(buffer)
                        buffer = [i, id2, j, j]    # initiate new entity entry in buffer
                    elif id1 == "I":
                        if len(buffer) == 0:    # if I-tag is an island, disregard it
                            continue
                        elif id2 != buffer[1]:    # if type of entity doesn't match what's in buffer, write buffer without current tag
                            tags.append(buffer)
                            buffer = []
                        else:    # otherwise, add new end index to buffer
                            buffer[3] = j
        if len(buffer) != 0:
            tags.append(buffer)
        entities.append(tags)
    # get counts on false positives, true positives and false negatives
    tps, fps, fns = 0, 0, 0
    entity_types = set([entity.split("-")[1] for entity in entity2id.keys() if entity != "O"])
    verbose_info = {entity + "_" + qualifier: 0 for qualifier in ["TP", "FP", "FN"] for entity in entity_types}
    for ent in entities[0]:
        if ent in entities[1]:
            tps += 1
            verbose_info[ent[1] + "_TP"] += 1
        else:
            fps += 1
            verbose_info[ent[1] + "_FP"] += 1
    for ent in entities[1]:
        if ent not in entities[0]:
            fns += 1
            verbose_info[ent[1] + "_FN"] += 1
    _, _, f1 = get_granular_f1(tps, fps, fns)
    macro_avg = 0.0
    for entity in entity_types:
        tps, fps, fns = verbose_info[entity + "_TP"], verbose_info[entity + "_FP"], verbose_info[entity + "_FN"]
        precision, recall, f1_verbose = get_granular_f1(tps, fps, fns)
        verbose_line = entity + "\t" + "tp: " + str(tps) + " - fp: " + str(fps) + " - fn: " + str(fns) \
                      + " - precision: " + str(precision) + " - recall: " + str(recall) + " - f1-score: "\
                       + str(f1_verbose) + "\n"
        macro_avg += f1_verbose
        verbose += verbose_line
    macro_avg /= len(entities)
    verbose = "MICRO_AVG:\tf1-score: " + str(f1) + "\n" + "MACRO_AVG:\tf1-score: " + str(macro_avg) + "\n" + verbose
    return f1, [tps, fps, fns], verbose

def get_granular_f1(tps, fps, fns):
    if tps + fps == 0:
        precision = 0.0
    else:
        precision = 1.0 * tps / (tps + fps)
    if tps + fns == 0:
        recall = 0.0
    else:
        recall = 1.0 * tps / (tps + fns)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

if __name__ == "__main__":
    test_seq_out = [[10, 10, 0, 5, 5, 10, 10, 2, 10, 3, 4, 9, 8, 10, 8, 1 ], [10, 10, 10]]
    test_seq_gold = [[10, 10, 1, 6, 6, 10, 10, 2, 10, 3, 4, 10, 3, 10, 10, 1], [10, 10, 2]]
    entity2id = {'B-EVT': 0, 'B-LOC': 1, 'B-ORG': 2, 'B-PER': 3, 'B-PRO': 4, 'I-EVT': 5,
                 'I-LOC': 6, 'I-ORG': 7, 'I-PER': 8, 'I-PRO': 9, 'O': 10}
    print(calculate_f1_ner(test_seq_out, test_seq_gold, entity2id))