import torch
import torch.nn as nn

class WSDModel(nn.Module):

    def __init__(self, embedding_dim, embedding_weights, hidden_dim, hidden_layers, output_size):
        super(WSDModel, self).__init__()
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weights)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.output = nn.Linear(2 * hidden_dim, output_size)

    def forward(self, X, X_lengths, mask):
        X = self.word_embeddings(X)
        #TODO check the sorting issue
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False)
        hidden = (torch.randn(self.hidden_layers * 2, len(X_lengths), self.hidden_dim),
                  torch.randn(self.hidden_layers * 2, len(X_lengths), self.hidden_dim))
        X, _ = self.lstm(X, hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        mask = mask[:, :X.shape[1]]
        mask = torch.reshape(mask, (mask.shape[0], mask.shape[1], 1))
        X = torch.masked_select(X, mask)
        X = X.view(-1, 2 * self.hidden_dim)
        # X_masked = torch.index_select(X, indices)
        outputs = self.output(X)
        return outputs