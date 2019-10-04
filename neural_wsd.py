import argparse
import numpy
import torch
import os
import sys

from auxiliary import Logger
from data_ops import WSDataset, load_embeddings, get_wordnet_lexicon
from wsd_model import WSDModel, calculate_accuracy_embedding, calculate_accuracy_classification

def disambiguate_by_default(lemmas, known_lemmas):
    default_disambiguations = []
    for i, lemma in enumerate(lemmas):
        if lemma not in known_lemmas:
            default_disambiguations.append(i)
    return default_disambiguations

def eval_loop(data_loader, known_lemmas, model, output_layers):
    matches_embed_all, total_embed_all = 0, 0
    matches_classify_all, total_classify_all = 0, 0
    for eval_data in data_loader:
        lemmas = numpy.asarray(eval_data['lemmas_pos']).transpose()[eval_data["mask"]]
        default_disambiguations = disambiguate_by_default(lemmas, known_lemmas)
        synsets = numpy.asarray(eval_data['synsets']).transpose()[eval_data["mask"]]
        pos = numpy.asarray(eval_data['pos']).transpose()[eval_data["mask"]]
        targets_classify = torch.from_numpy(numpy.asarray(eval_data["targets_classify"])[eval_data["mask"]])
        outputs = model(eval_data["inputs"], eval_data["length"], eval_data["mask"], lemmas, pos)
        accuracy_embed, accuracy_classify = 0.0, 0.0
        if "embed_wsd" in output_layers:
            matches_embed, total_embed = calculate_accuracy_embedding(outputs["embed_wsd"],
                                                                      lemmas,
                                                                      synsets,
                                                                      lemma2synsets,
                                                                      embeddings,
                                                                      src2id,
                                                                      pos_filter=True)
            matches_embed_all += matches_embed
            total_embed_all += total_embed
            accuracy_embed = matches_embed_all * 1.0 / total_embed_all
        if "classify_wsd" in output_layers:
            matches_classify, total_classify = calculate_accuracy_classification(
                outputs["classify_wsd"].detach().numpy(),
                targets_classify.detach().numpy(),
                default_disambiguations)
            matches_classify_all += matches_classify
            total_classify_all += total_classify
            accuracy_classify = matches_classify_all * 1.0 / total_classify_all
    return accuracy_embed, accuracy_classify

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
    parser.add_argument('-output_layers', dest='output_layers', required=False, default="embed_wsd",
                        help='What tasks will the NN solve at output? Options: embed_wsd, classify_wsd.'
                             'More than one can be provided, delimiting them by commas, e.g. "embed_wsd,classiy_wsd"')
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
    f_test = args.test_data_path
    trainset = WSDataset(f_train, src2id, embeddings, embedding_dim, max_labels, lemma2synsets)
    devset = WSDataset(f_dev, src2id, embeddings, embedding_dim, max_labels, lemma2synsets)
    testset = WSDataset(f_test, src2id, embeddings, embedding_dim, max_labels, lemma2synsets)

    # Redirect print statements to both terminal and logfile
    sys.stdout = Logger(os.path.join(save_path, "results.txt"))

    # Get model parameters
    output_layers = [str.strip(layer) for layer in args.output_layers.split(",")]
    alpha = float(args.alpha)
    batch_size = int(args.batch_size)
    hiden_neurons = int(args.n_hidden_neurons)
    hiden_layers = int(args.n_hidden_layers)
    dropout = float(args.dropout)
    learning_rate = float(args.learning_rate)

    # Construct data loaders for this specific model
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    devloader = torch.utils.data.DataLoader(devset, batch_size=len(devset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data), shuffle=False)

    # Construct the model
    model = WSDModel(embedding_dim, embeddings, hiden_neurons, hiden_layers, dropout, output_layers, lemma2synsets)
    loss_func_embed = torch.nn.MSELoss()
    loss_func_classify = torch.nn.CrossEntropyLoss(ignore_index=-100)
    # loss_func_classify = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Eval loop
    if args.mode == "evaluate":
        model.load_state_dict(torch.load(args.save_path))
        model.eval()
        test_accuracy_embed, test_accuracy_classify = eval_loop(testloader, model, output_layers)
        print("Test embedding accuracy: " + str(test_accuracy_embed))
        print("Test classification accuracy: " + str(test_accuracy_classify))

    # Train loop
    else:
        least_loss = 0.0
        best_accuracy_embed, best_accuracy_classify = 0.0, 0.0
        eval_at = 100
        for epoch in range(100):
            print("***** Start of epoch " + str(epoch) + " *****")
            average_loss_embed, average_loss_classify, average_loss_overall = 0.0, 0.0, 0.0
            for step, data in enumerate(trainloader):
                model.train()
                optimizer.zero_grad()
                lemmas = numpy.asarray(data['lemmas_pos']).transpose()[data["mask"]]
                default_disambiguations = disambiguate_by_default(lemmas, trainset.known_lemmas)
                pos = numpy.asarray(data['pos']).transpose()[data["mask"]]
                synsets = numpy.asarray(data['synsets']).transpose()[data["mask"]]
                lengths_labels = numpy.asarray(data["lengths_labels"])[data["mask"]]
                outputs = model(data["inputs"], data["length"], data["mask"], lemmas, pos)
                mask = torch.reshape(data["mask"], (data["mask"].shape[0], data["mask"].shape[1], 1))
                loss = 0.0
                # Calculate loss for the context embedding method
                if "embed_wsd" in output_layers:
                    targets_embed = torch.masked_select(data["targets_embed"], mask)
                    targets_embed = targets_embed.view(-1, embedding_dim)
                    neg_targets = torch.masked_select(data["neg_targets"], mask)
                    neg_targets = neg_targets.view(-1, embedding_dim)
                    # targets_classify = targets_classify.view(-1, max_labels)
                    loss_embed = alpha * loss_func_embed(outputs["embed_wsd"], targets_embed) + \
                                 (1 - alpha) * (1 - loss_func_embed(outputs["embed_wsd"], neg_targets))
                    loss += loss_embed
                    average_loss_embed += loss_embed
                # Calculate loss for the classification method
                if "classify_wsd" in output_layers:
                    targets_classify = torch.from_numpy(numpy.asarray(data["targets_classify"])[data["mask"]])
                    loss_classify = loss_func_classify(outputs["classify_wsd"],
                                                       targets_classify)
                    loss += loss_classify
                    average_loss_classify += loss_classify
                # loss = loss_embed + loss_classify
                loss.backward()
                optimizer.step()
                # Eval loop during training
                if step % eval_at == 0:
                    model.eval()
                    print("Step " + str(step))
                    if "embed_wsd" in output_layers:
                        matches_embed, total_embed = calculate_accuracy_embedding(outputs["embed_wsd"],
                                                                                  lemmas,
                                                                                  synsets,
                                                                                  lemma2synsets,
                                                                                  embeddings,
                                                                                  src2id,
                                                                                  pos_filter=True)
                        train_accuracy_embed = matches_embed * 1.0 / total_embed
                        print("Training embedding accuracy: " + str(train_accuracy_embed))
                        average_loss_embed /= (eval_at if step != 0 else 1)
                        print("Average embedding loss (training): " + str(average_loss_embed.detach().numpy()))
                        average_loss_overall += average_loss_embed.detach().numpy()
                    if "classify_wsd" in output_layers:
                        matches_classify, total_classify = calculate_accuracy_classification(outputs["classify_wsd"].detach().numpy(),
                                                                                             targets_classify.detach().numpy(),
                                                                                             default_disambiguations)
                        train_accuracy_classify = matches_classify * 1.0 / total_classify

                        print("Training classification accuracy: " + str(train_accuracy_classify))
                        average_loss_classify /= (eval_at if step != 0 else 1)
                        print("Average classification loss (training): " + str(average_loss_classify.detach().numpy()))
                        average_loss_overall += average_loss_classify.detach().numpy()
                    print("Average overall loss (training): " + str(average_loss_overall))
                    average_loss_embed, average_loss_classif, average_loss_overall = 0.0, 0.0, 0.0

                    # Loop over the dev dataset
                    dev_accuracy_embed, dev_accuracy_classify = eval_loop(devloader, trainset.known_lemmas, model, output_layers)
                    print("Dev embedding accuracy: " + str(dev_accuracy_embed))
                    print("Dev classification accuracy: " + str(dev_accuracy_classify))
                    best_result = False
                    if dev_accuracy_embed > best_accuracy_embed:
                        for file in os.listdir(save_path):
                            if "embed_wsd" in file:
                                os.remove(os.path.join(save_path, file))
                        torch.save(model.state_dict(), os.path.join(save_path, "epoch" + str(epoch) + "-step" + str(step)
                                                                    + "-embed_wsd=" + str(dev_accuracy_embed)[:7] + ".pt"))

                        best_accuracy_embed = dev_accuracy_embed
                        best_result = True
                    if dev_accuracy_classify > best_accuracy_classify:
                        for file in os.listdir(save_path):
                            if "classify_wsd" in file:
                                os.remove(os.path.join(save_path, file))
                        torch.save(model.state_dict(), os.path.join(save_path, "epoch" + str(epoch) + "-step" + str(step)
                                                                    + "-classify_wsd=" + str(dev_accuracy_classify)[:7] + ".pt"))
                        best_accuracy_classify = dev_accuracy_classify
                        best_result = True
                    if best_result is True:
                        # Eval on the test dataset as well
                        test_accuracy_embed, test_accuracy_classify = eval_loop(testloader, trainset.known_lemmas, model, output_layers)
                        print("Test embedding accuracy: " + str(test_accuracy_embed))
                        print("Test classification accuracy: " + str(test_accuracy_classify))

    print("This is the end.")