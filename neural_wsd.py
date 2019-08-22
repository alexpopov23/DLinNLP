import numpy
import torch
import os

from data_ops import WSDataset, load_embeddings, get_wordnet_lexicon
from wsd_model import WSDModel, calculate_accuracy

embeddings_path = "/home/lenovo/dev/word-embeddings/lemma_sense_embeddings/" \
                  "WN30WN30glConOne-C15I7S7N5_200M_syn_and_lemma_WikipediaLemmatized_FILTERED.txt"
embeddings, src2id, id2src = load_embeddings(embeddings_path)
embeddings = torch.Tensor(embeddings)
lexicon_path = "/home/lenovo/tools/ukb_wsd/lkb_sources/wn30.lex"
lemma2synsets = get_wordnet_lexicon(lexicon_path, True)

save_path = "/home/lenovo/dev/DLinNLP/experiments/test"
f_train = "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv/semcor.tsv"
f_test = "/home/lenovo/dev/neural-wsd/data/Unified-WSD-framework/tsv/senseval2.tsv"
trainset = WSDataset(f_train, src2id, embeddings, 300, lemma2synsets, input_synsets=True)
testset = WSDataset(f_test, src2id, embeddings, 300, lemma2synsets)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

model = WSDModel(300, embeddings, 1000, 1, 300)
loss_function = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)

# Train loop
least_loss = 0.0
best_accuracy = 0.0
for epoch in range(100):
    print("***** Start of epoch " + str(epoch) + " *****")
    average_loss = 0.0
    for step, data in enumerate(trainloader):
        outputs = model(data["inputs"], data["length"], data["mask"])
        mask = torch.reshape(data["mask"], (data["mask"].shape[0], data["mask"].shape[1], 1))
        targets = torch.masked_select(data["targets"], mask)
        targets = targets.view(-1, 300)
        neg_targets = torch.masked_select(data["neg_targets"], mask)
        neg_targets = neg_targets.view(-1, 300)
        loss = 0.85 * loss_function(outputs, targets) + 0.15 * (1 - loss_function(outputs, neg_targets))
        loss.backward()
        average_loss += loss
        optimizer.step()
        if step % 20 == 0:
            print("Step " + str(step))
            lemmas = numpy.asarray(data['lemmas']).transpose()[data["mask"]]
            synsets = numpy.asarray(data['synsets']).transpose()[data["mask"]]
            pos = numpy.asarray(data['pos']).transpose()[data["mask"]]
            matches, total = calculate_accuracy(outputs, lemmas, pos, synsets, lemma2synsets,
                                                embeddings, src2id, pos_filter=True)
            train_accuracy = matches * 1.0 / total
            print("Training accuracy at step " + str(step) + ": " + str(train_accuracy))
            average_loss /= 20
            print("Average loss (training) is: " + str(average_loss.detach().numpy()))
            average_loss = 0.0
            # Loop over the eval data
            matches_all, total_all = 0, 0
            for eval_data in testloader:
                outputs = model(eval_data["inputs"], eval_data["length"], eval_data["mask"])
                lemmas = numpy.asarray(eval_data['lemmas']).transpose()[eval_data["mask"]]
                synsets = numpy.asarray(eval_data['synsets']).transpose()[eval_data["mask"]]
                pos = numpy.asarray(eval_data['pos']).transpose()[eval_data["mask"]]
                matches, total = calculate_accuracy(outputs, lemmas, pos, synsets, lemma2synsets,
                                                    embeddings, src2id, pos_filter=True)
                matches_all += matches
                total_all += total
            test_accuracy = matches_all * 1.0 / total_all
            print("Test accuracy at step " + str(step) + " is: " + str(test_accuracy))
            if average_loss < least_loss or test_accuracy > best_accuracy:
                least_loss = average_loss
                best_accuracy = test_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(save_path, "epoch" + str(epoch) + "step" + str(step)))


# Evaluation loop
print("This is the end.")