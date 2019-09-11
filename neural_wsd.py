import argparse
import numpy
import torch
import os

from data_ops import WSDataset, load_embeddings, get_wordnet_lexicon
from wsd_model import WSDModel, calculate_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and evaluate a neural WSD model.', fromfile_prefix_chars='@')
    parser.add_argument('-alpha', dest='alpha', required=False, default=1,
                        help='Coefficient to weight the positive and negative labels.')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-dev_data_path', dest='dev_data_path', required=False,
                        help='The path to the gold corpus used for development.')
    parser.add_argument('-dropout', dest='dropout', required=False, default="0.",
                        help='The probability of keeping an element output in a layer (for dropout)')
    parser.add_argument('-embeddings_path', dest='embeddings_path', required=True,
                        help='The path to the pretrained model with the primary embeddings.')
    parser.add_argument('-embeddings2_path', dest='embeddings2_path', required=False,
                        help='The path to the pretrained model with the additional embeddings.')
    parser.add_argument('-embeddings_case', dest='embeddings_case', required=False, default="lowercase",
                        help='Are the embeddings trained on lowercased or mixedcased text? Options: lowercase, '
                             'mixedcase')
    parser.add_argument('-embeddings2_case', dest='embeddings2_case', required=False, default="lowercase",
                        help='Are the embeddings trained on lowercased or mixedcased text? Options: lowercase, '
                             'mixedcase')
    parser.add_argument('-embeddings_input', dest='embeddings_input', required=False, default="wordform",
                        help='Are these embeddings of wordforms or lemmas? Options are: wordform, lemma')
    parser.add_argument('-embeddings2_input', dest='embeddings2_input', required=False, default="lemma",
                        help='Are these embeddings of wordforms or lemmas? Options are: wordform, lemma')
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.2,
                        help='How fast the network should learn.')
    parser.add_argument('-lexicon_path', dest='lexicon_path', required=False,
                        help='The path to the location of the lexicon file.')
    parser.add_argument('-max_seq_length', dest='max_seq_length', required=False, default=63,
                        help='Maximum length of a sentence to be passed to the network (the rest is cut off).')
    parser.add_argument('-mode', dest='mode', required=False, default="train",
                        help="Is this is a training, evaluation or application run? Options: train, evaluation, "
                             "application")
    parser.add_argument('-n_hidden_neurons', dest='n_hidden_neurons', required=False, default=200,
                        help='Size of the hidden layer.')
    parser.add_argument('-n_hidden_layers', dest='n_hidden_layers', required=False, default=1,
                        help='Number of the hidden LSTMs in the forward/backward modules.')
    parser.add_argument('-pos_filter', dest='pos_filter', required=False, default="False",
                        help='Whether to use POS information to filter out irrelevant synsets.')
    parser.add_argument('-save_path', dest='save_path', required=False,
                        help='Path to where the model should be saved, or path to the folder with a saved model.')
    parser.add_argument('-sensekey2synset_path', dest='sensekey2synset_path', required=False,
                        help='Path to mapping between sense annotations in the corpus and synset IDs in WordNet.')
    parser.add_argument('-test_data_path', dest='test_data_path', required=False,
                        help='The path to the gold corpus used for testing.')
    parser.add_argument('-train_data_path', dest='train_data_path', required=False,
                        help='The path to the gold corpus used for training.')
    parser.add_argument('-training_epochs', dest='training_epochs', required=False, default=100001,
                        help='How many epochs over the data the network should train for.')
    parser.add_argument('-wsd_method', dest='wsd_method', required=True,
                        help='Which method for WSD? Options: classification, context_embedding, multitask')

    # Get the embeddings and lexicon
    args = parser.parse_args()
    embeddings_path = args.embeddings_path
    embeddings, src2id, id2src = load_embeddings(embeddings_path)
    embeddings = torch.Tensor(embeddings)
    embedding_dim = embeddings.shape[1]
    lexicon_path = args.lexicon_path
    lemma2synsets = get_wordnet_lexicon(lexicon_path, args.pos_filter)

    # Get the training/dev/testing data
    save_path = args.save_path
    f_train = args.train_data_path
    f_dev = args.dev_data_path
    trainset = WSDataset(f_train, src2id, embeddings, embedding_dim, lemma2synsets)
    devset = WSDataset(f_dev, src2id, embeddings, embedding_dim, lemma2synsets)

    # Get model parameters
    alpha = float(args.alpha)
    batch_size = int(args.batch_size)
    hiden_neurons = int(args.n_hidden_neurons)
    hiden_layers = int(args.n_hidden_layers)
    dropout = float(args.dropout)
    learning_rate = float(args.learning_rate)

    # Construct data loaders for this specific model
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    devloader = torch.utils.data.DataLoader(devset, batch_size=len(devset.data), shuffle=False)

    # Construct the model
    model = WSDModel(embedding_dim, embeddings, hiden_neurons, hiden_layers, dropout)
    loss_function = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train loop
    least_loss = 0.0
    best_accuracy = 0.0
    eval_at = 100
    for epoch in range(100):
        print("***** Start of epoch " + str(epoch) + " *****")
        average_loss = 0.0
        for step, data in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            outputs = model(data["inputs"], data["length"], data["mask"])
            mask = torch.reshape(data["mask"], (data["mask"].shape[0], data["mask"].shape[1], 1))
            targets = torch.masked_select(data["targets"], mask)
            targets = targets.view(-1, embedding_dim)
            neg_targets = torch.masked_select(data["neg_targets"], mask)
            neg_targets = neg_targets.view(-1, embedding_dim)
            loss = alpha * loss_function(outputs, targets) + (1 - alpha) * (1 - loss_function(outputs, neg_targets))
            loss.backward()
            average_loss += loss
            optimizer.step()
            # Eval loop during training
            if step % eval_at == 0:
                model.eval()
                print("Step " + str(step))
                lemmas = numpy.asarray(data['lemmas']).transpose()[data["mask"]]
                synsets = numpy.asarray(data['synsets']).transpose()[data["mask"]]
                pos = numpy.asarray(data['pos']).transpose()[data["mask"]]
                matches, total = calculate_accuracy(outputs, lemmas, pos, synsets, lemma2synsets,
                                                    embeddings, src2id, pos_filter=True)
                train_accuracy = matches * 1.0 / total
                print("Training accuracy at step " + str(step) + ": " + str(train_accuracy))
                average_loss /= (eval_at if step != 0 else 1)
                print("Average loss (training) is: " + str(average_loss.detach().numpy()))
                average_loss = 0.0
                # Loop over the eval data
                matches_all, total_all = 0, 0
                for eval_data in devloader:
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

    # Evaluation loop on the test data
    print("This is the end.")