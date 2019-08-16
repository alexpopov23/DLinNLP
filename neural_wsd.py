import torch

from data_ops import WSDataset, load_embeddings
from wsd_model import WSDModel

embeddings_path = "/home/lenovo/dev/word-embeddings/lemma_sense_embeddings/" \
                  "WN30WN30glConOne-C15I7S7N5_200M_syn_and_lemma_WikipediaLemmatized_FILTERED.txt"
embeddings, src2id, id2src = load_embeddings(embeddings_path)
embeddings = torch.Tensor(embeddings)

f_dataset = "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv/semeval2007.tsv"
trainset = WSDataset(f_dataset, src2id, embeddings, 300)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

model = WSDModel(300, embeddings, 100, 1, 300)
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
for epoch in range(10):
    for i, data in enumerate(trainloader):
        print(data)
        outputs = model(data[2], data[6], data[5])
        mask = torch.reshape(data[5], (data[5].shape[0], data[5].shape[1], 1))
        targets = torch.masked_select(data[3], mask)
        targets = targets.view(-1, 300)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
print("This is the end.")