import argparse
import numpy
import torch
import os

from data_ops import WSDataset, load_embeddings, get_wordnet_lexicon
from wsd_model import WSDModel, calculate_accuracy_embedding, calculate_accuracy_classification

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
    lemma2synsets, max_labels = get_wordnet_lexicon(lexicon_path, args.pos_filter)

    # Get the training/dev/testing data
    save_path = args.save_path
    f_train = args.train_data_path
    f_dev = args.dev_data_path
    trainset = WSDataset(f_train, src2id, embeddings, embedding_dim, max_labels, lemma2synsets)
    devset = WSDataset(f_dev, src2id, embeddings, embedding_dim, max_labels, lemma2synsets)

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
    model = WSDModel(embedding_dim, embeddings, hiden_neurons, hiden_layers, dropout,
                     ["context_embedding", "classification"], lemma2synsets)
    loss_function = torch.nn.MSELoss()
    loss_function_classif = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train loop
    least_loss = 0.0
    best_accuracy = 0.0
    eval_at = 100
    for epoch in range(100):
        print("***** Start of epoch " + str(epoch) + " *****")
        average_loss_embed, average_loss_classif, average_loss_overall = 0.0, 0.0, 0.0
        for step, data in enumerate(trainloader):
            model.train()
            optimizer.zero_grad()
            lemmas = numpy.asarray(data['lemmas']).transpose()[data["mask"]]
            pos = numpy.asarray(data['pos']).transpose()[data["mask"]]
            synsets = numpy.asarray(data['synsets']).transpose()[data["mask"]]
            lengths_labels = numpy.asarray(data["lengths_labels"])[data["mask"]]
            outputs = model(data["inputs"], data["length"], data["mask"], lemmas, pos)
            mask = torch.reshape(data["mask"], (data["mask"].shape[0], data["mask"].shape[1], 1))
            # Calculate loss for the context embedding method
            targets = torch.masked_select(data["targets"], mask)
            targets = targets.view(-1, embedding_dim)
            neg_targets = torch.masked_select(data["neg_targets"], mask)
            neg_targets = neg_targets.view(-1, embedding_dim)
            targets_labels = torch.from_numpy(numpy.asarray(data["targets_labels"])[data["mask"]])
            loss_embed = alpha * loss_function(outputs[0], targets) + (1 - alpha) * (1 - loss_function(outputs[0], neg_targets))
            # Calculate loss for the classification method
            loss_classif = loss_function_classif(outputs[1], targets_labels)
            loss = loss_embed + loss_classif
            loss.backward()
            average_loss_embed += loss_embed
            average_loss_classif += loss_classif
            optimizer.step()
            # Eval loop during training
            if step % eval_at == 0:
                model.eval()
                print("Step " + str(step))
                matches_embed, total_embed = calculate_accuracy_embedding(outputs[0],
                                                                          lemmas,
                                                                          pos,
                                                                          synsets,
                                                                          lemma2synsets,
                                                                          embeddings,
                                                                          src2id,
                                                                          pos_filter=True)
                train_accuracy_embed = matches_embed * 1.0 / total_embed
                matches_classif, total_classif = calculate_accuracy_classification(outputs[1], targets_labels)
                train_accuracy_classif = matches_classif * 1.0 / total_classif
                print("Training embedding accuracy at step " + str(step) + ": " + str(train_accuracy_embed))
                average_loss_embed /= (eval_at if step != 0 else 1)
                print("Average embedding loss (training) is: " + str(average_loss_embed.detach().numpy()))
                print("Training classification accuracy at step " + str(step) + ": " + str(train_accuracy_classif))
                average_loss_classif /= (eval_at if step != 0 else 1)
                print("Average classification loss (training) is: " + str(average_loss_classif.detach().numpy()))
                average_loss_overall = average_loss_embed.detach().numpy() + average_loss_classif.detach().numpy()
                average_loss_embed, average_loss_classif = 0.0, 0.0
                print("Average overall loss (training) is: " + str(average_loss_overall))
                # Loop over the eval data
                matches_embed_all, total_embed_all = 0, 0
                matches_classif_all, total_classif_all = 0, 0
                for eval_data in devloader:
                    lemmas = numpy.asarray(eval_data['lemmas']).transpose()[eval_data["mask"]]
                    synsets = numpy.asarray(eval_data['synsets']).transpose()[eval_data["mask"]]
                    pos = numpy.asarray(eval_data['pos']).transpose()[eval_data["mask"]]
                    targets = torch.from_numpy(numpy.asarray(eval_data["targets_labels"])[eval_data["mask"]])
                    outputs = model(eval_data["inputs"], eval_data["length"], eval_data["mask"], lemmas, pos)
                    matches_embed, total_embed = calculate_accuracy_embedding(outputs[0],
                                                                              lemmas,
                                                                              pos,
                                                                              synsets,
                                                                              lemma2synsets,
                                                                              embeddings,
                                                                              src2id,
                                                                              pos_filter=True)
                    matches_embed_all += matches_embed
                    total_embed_all += total_embed
                    matches_classif, total_classif = calculate_accuracy_classification(outputs[1], targets)
                    matches_classif_all += matches_classif
                    total_classif_all += total_classif
                test_accuracy_embed = matches_embed_all * 1.0 / total_embed_all
                test_accuracy_classif = matches_classif_all * 1.0 / total_classif_all
                print("Test embedding accuracy at step " + str(step) + " is: " + str(test_accuracy_embed))
                print("Test classification accuracy at step " + str(step) + " is: " + str(test_accuracy_classif))
                if average_loss_overall < least_loss or test_accuracy_classif > best_accuracy:
                    least_loss = average_loss_overall
                    best_accuracy = test_accuracy_classif
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, os.path.join(save_path, "epoch" + str(epoch) + "step" + str(step)))

    # Evaluation loop on the test data
    print("This is the end.")