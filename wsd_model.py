import torch
import torch.nn as nn

from sklearn.metrics.pairwise import cosine_similarity

POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}

class WSDModel(nn.Module):

    def __init__(self, embeddings_dim, embedding_weights, hidden_dim,
                 hidden_layers, dropout):
        super(WSDModel, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weights,
                                                            freeze=False)
        self.lstm = nn.LSTM(embeddings_dim,
                            hidden_dim,
                            hidden_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout)
        # We want output with the size of the lemma&synset embeddings
        self.output = nn.Linear(2*hidden_dim, embeddings_dim)

    def forward(self, X, X_lengths, mask):
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
        # Make mask broadcastable to X
        mask = torch.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        # Select only RNN outputs that correspond to synset-tagged words in the data
        X = torch.masked_select(X, mask)
        # masked_select flattens the tensor, but we need it as matrix
        X = X.view(-1, 2 * self.hidden_dim) # shape is [num_labels, 2*hidden_dim]
        outputs = self.output(X)
        return outputs


def calculate_accuracy(outputs, lemmas, pos, gold_synsets, lemma2synsets, embeddings, src2id, pos_filter=True):
    matches, total = 0, 0
    for i, output in enumerate(torch.unbind(outputs)):
        if pos_filter:
            lemma = lemmas[i] + "-" + POS_MAP[pos[i]]
        else:
            lemma = lemmas[i]
        possible_synsets = lemma2synsets[lemma]
        synset_choice, max_similarity = "", -100.0
        for synset in possible_synsets:
            synset_embedding = embeddings[src2id[synset]] if synset in src2id else embeddings[src2id["<UNK>"]]
            cos_sim = cosine_similarity(output.view(1, -1).detach().numpy(),
                                        synset_embedding.view(1, -1).detach().numpy())[0][0]
            if cos_sim > max_similarity:
                max_similarity = cos_sim
                synset_choice = synset
        if synset_choice in gold_synsets[i].split(","):
            matches += 1
        total += 1
    return matches, total