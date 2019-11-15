import argparse
import itertools
import numpy
import os
import sys
import torch

from torchcrf import CRF

from auxiliary import Logger
from data_ops import WSDataset, load_embeddings, get_wordnet_lexicon
from wsd_model import WSDModel, calculate_accuracy_embedding, calculate_accuracy_classification, calculate_accuracy_pos

def disambiguate_by_default(lemmas, known_lemmas):
    default_disambiguations = []
    for i, lemma in enumerate(lemmas):
        if lemma not in known_lemmas:
            default_disambiguations.append(i)
    return default_disambiguations

def slice_and_pad(tensor, lengths, tag_length=1):
    max_length = max(lengths)
    iterable = iter(tensor)
    padded_tensor = []
    padding = tag_length * [0]
    for l in lengths:
        if l == 0:
            continue
        padded_seq = list(itertools.islice(iterable, l*tag_length))
        if tag_length > 1:
            padded_seq = [padded_seq[i*tag_length:((i+1)*tag_length)] for i in range(l)]
            padded_tensor.append(padded_seq + (max_length - l).item() * [padding])
        else:
            padded_tensor.append(padded_seq + (max_length - l).item() * padding)
    return torch.tensor(padded_tensor)

def length_to_mask(lengths, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    from: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/2
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    # lengths = [l for l in lengths if l != 0]
    lengths = lengths[lengths.nonzero().squeeze()]
    max_len = max_len or lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device,
                        dtype=lengths.dtype).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask


def eval_loop(data_loader, known_lemmas, model, output_layers):
    matches_embed_all, total_embed_all = 0, 0
    matches_classify_all, total_classify_all = 0, 0
    for eval_data in data_loader:
        lemmas = numpy.asarray(eval_data['lemmas_pos']).transpose()[eval_data["mask"]]
        default_disambiguations = disambiguate_by_default(lemmas, known_lemmas)
        synsets = numpy.asarray(eval_data['synsets']).transpose()[eval_data["mask"]]
        pos = numpy.asarray(eval_data['pos']).transpose()[eval_data["mask"]]
        targets_classify = torch.from_numpy(numpy.asarray(eval_data["targets_classify"])[eval_data["mask"]])
        targets_pos = torch.from_numpy(numpy.asarray(eval_data["targets_pos"])[eval_data["pos_mask"]])
        outputs = model(eval_data["inputs"], eval_data["length"], eval_data["mask"], eval_data["pos_mask"], lemmas)
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
                default_disambiguations,
                lemmas,
                known_lemmas,
                synsets,
                lemma2synsets,
                synset2id,
                single_softmax)
            matches_classify_all += matches_classify
            total_classify_all += total_classify
            accuracy_classify = matches_classify_all * 1.0 / total_classify_all
        if "pos_tagger" in output_layers:
            matches_pos, total_pos = calculate_accuracy_pos(
                outputs["pos_tagger"],
                targets_pos)
            accuracy_pos = matches_pos * 1.0 / total_pos
    return accuracy_embed, accuracy_classify, accuracy_pos

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and evaluate a neural WSD model.', fromfile_prefix_chars='@')
    parser.add_argument('-alpha', dest='alpha', required=False, default=1,
                        help='Coefficient to weight the positive and negative labels.')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-crf_layer', dest='crf_layer', required=False, default=False,
                        help='Whether to use a CRF layer on top of the LSTM. Indicate True if yes.')
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
    parser.add_argument('-n_classifiers', dest='n_classifiers', required=False, default="multiple",
                        help="Shoud there be a separate classification layer per lemma, or should the model use a "
                             "single layer for all lemmas? Options: single, multiple")
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

    # Get the embeddings and lexicon
    args = parser.parse_args()
    embeddings_path = args.embeddings_path
    embeddings, src2id, id2src = load_embeddings(embeddings_path)
    embeddings = torch.Tensor(embeddings)
    embedding_dim = embeddings.shape[1]
    lexicon_path = args.lexicon_path
    lemma2synsets, max_labels = get_wordnet_lexicon(lexicon_path, args.pos_filter)
    crf_layer = True if args.crf_layer == "True" else False
    # if crf_layer is True:
    #     single_softmax = True
    # else:
    single_softmax = True if args.n_classifiers == "single" else False

    # Get the training/dev/testing data
    save_path = args.save_path
    train_path = args.train_data_path
    dev_path = args.dev_data_path
    test_path = args.test_data_path
    trainset = WSDataset(train_path, src2id, embeddings, embedding_dim, max_labels, lemma2synsets, single_softmax)
    if single_softmax is True:
        synset2id = trainset.known_synsets
    else:
        synset2id = {}
    devset = WSDataset(dev_path, src2id, embeddings, embedding_dim, max_labels, lemma2synsets, single_softmax, synset2id)
    testset = WSDataset(test_path, src2id, embeddings, embedding_dim, max_labels, lemma2synsets, single_softmax, synset2id)

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
    model = WSDModel(embedding_dim, embeddings, hiden_neurons, hiden_layers, dropout, output_layers, lemma2synsets,
                     synset2id, trainset.known_pos)
    loss_func_embed = torch.nn.MSELoss()
    if crf_layer is True:
        loss_func_classify = CRF(model.num_wsd_classes, batch_first=True)
        loss_func_pos = CRF(len(trainset.known_pos), batch_first=True)
    else:
        loss_func_classify = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss_func_pos = torch.nn.CrossEntropyLoss()
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
        best_accuracy_embed, best_accuracy_classify, best_accuracy_pos = 0.0, 0.0, 0.0
        eval_at = 100
        for epoch in range(100):
            print("***** Start of epoch " + str(epoch) + " *****")
            average_loss_embed, average_loss_classify, average_loss_pos, average_loss_overall = 0.0, 0.0, 0.0, 0.0
            for step, data in enumerate(trainloader):
                model.train()
                optimizer.zero_grad()
                lemmas = numpy.asarray(data['lemmas_pos']).transpose()[data["mask"]]
                default_disambiguations = disambiguate_by_default(lemmas, trainset.known_lemmas)
                synsets = numpy.asarray(data['synsets']).transpose()[data["mask"]]
                # lengths_labels = numpy.asarray(data["lengths_labels"])[data["mask"]]
                outputs = model(data["inputs"], data["length"], data["mask"], data["pos_mask"], lemmas)
                loss = 0.0
                # Calculate loss for the context embedding method
                if "embed_wsd" in output_layers:
                    mask_embed = torch.reshape(data["mask"], (data["mask"].shape[0], data["mask"].shape[1], 1))
                    targets_embed = torch.masked_select(data["targets_embed"], mask_embed)
                    targets_embed = targets_embed.view(-1, embedding_dim)
                    neg_targets = torch.masked_select(data["neg_targets"], mask_embed)
                    neg_targets = neg_targets.view(-1, embedding_dim)
                    # targets_classify = targets_classify.view(-1, max_labels)
                    loss_embed = alpha * loss_func_embed(outputs["embed_wsd"], targets_embed) + \
                                 (1 - alpha) * (1 - loss_func_embed(outputs["embed_wsd"], neg_targets))
                    loss += loss_embed
                    average_loss_embed += loss_embed
                # Calculate loss for the classification method
                if "classify_wsd" in output_layers:
                    # targets_classify = torch.from_numpy(numpy.asarray(data["targets_classify"])[data["mask"]])
                    # mask_classify = torch.reshape(data["mask"], (data["mask"].shape[0], data["mask"].shape[1], 1))
                    # mask_classify = data["mask"][:, :outputs["classify_wsd"].shape[1]]
                    # outputs_classify = torch.masked_select(outputs["classify_wsd"],
                    #                                        torch.reshape(mask_classify, (mask_classify.shape[0],
                    #                                                                      mask_classify.shape[1],
                    #                                                                      1)))
                    targets_classify = torch.masked_select(data["targets_classify"], data["mask"])
                    # loss_classify = loss_func_classify(outputs["classify_wsd"], targets_classify, mask_classify)
                    outputs_classify = slice_and_pad(outputs["classify_wsd"],
                                                     data["lengths_labels"],
                                                     tag_length=model.num_wsd_classes)
                    targets_classify = slice_and_pad(targets_classify, data["lengths_labels"])
                    mask_crf = length_to_mask(data["lengths_labels"], outputs_classify.shape[1])
                    loss_classify = loss_func_classify(outputs_classify, targets_classify, mask_crf, reduction='mean')
                    loss += loss_classify * (-1.0 if crf_layer is True else 1.0)
                    average_loss_classify += loss_classify
                if "pos_tagger" in output_layers:
                    # targets_pos = torch.from_numpy(numpy.asarray(data["targets_pos"])[data["pos_mask"]])
                    pos_mask = data["pos_mask"][:, :outputs["pos_tagger"].shape[1]]
                    targets_pos = data["targets_pos"][:, :outputs["pos_tagger"].shape[1]]
                    loss_pos = loss_func_pos(outputs["pos_tagger"], targets_pos, pos_mask, reduction="mean")
                    loss += loss_pos * (-1.0 if crf_layer is True else 1.0)
                    average_loss_pos += loss_pos
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
                        if crf_layer is True:
                            choices = loss_func_classify.decode(outputs_classify, mask=mask_crf)
                            matches_classify, total_classify = 0, 0
                            for i, seq in enumerate(choices):
                                for j, choice in enumerate(seq):
                                    if choice == targets_classify[i][j].item():
                                        matches_classify += 1
                                    total_classify += 1
                        else:
                            matches_classify, total_classify = calculate_accuracy_classification(
                                    outputs["classify_wsd"].detach().numpy(),
                                    targets_classify.detach().numpy(),
                                    default_disambiguations,
                                    lemmas,
                                    trainset.known_lemmas,
                                    synsets,
                                    lemma2synsets,
                                    synset2id,
                                    single_softmax)
                        train_accuracy_classify = matches_classify * 1.0 / total_classify

                        print("Training classification accuracy: " + str(train_accuracy_classify))
                        average_loss_classify /= (eval_at if step != 0 else 1)
                        print("Average classification loss (training): " + str(average_loss_classify.detach().numpy()))
                        average_loss_overall += average_loss_classify.detach().numpy()
                    if "pos_tagger" in output_layers:
                        matches_pos, total_pos = calculate_accuracy_pos(
                            outputs["pos_tagger"],
                            targets_pos)
                        train_accuracy_pos = matches_pos * 1.0 / total_pos

                        print("Training pos tagger accuracy: " + str(train_accuracy_pos))
                        average_loss_pos /= (eval_at if step != 0 else 1)
                        print("Average pos tagger loss (training): " + str(average_loss_pos.detach().numpy()))
                        average_loss_overall += average_loss_pos.detach().numpy()
                    print("Average overall loss (training): " + str(average_loss_overall))
                    average_loss_embed, average_loss_classif, average_loss_poss, average_loss_overall = 0.0, 0.0, 0.0, 0.0

                    # Loop over the dev dataset
                    dev_accuracy_embed, dev_accuracy_classify, dev_accuracy_pos = \
                        eval_loop(devloader, trainset.known_lemmas, model, output_layers)
                    print("Dev embedding accuracy: " + str(dev_accuracy_embed))
                    print("Dev classification accuracy: " + str(dev_accuracy_classify))
                    print("Dev pos tagging accuracy: " + str(dev_accuracy_pos))
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
                    if dev_accuracy_pos > best_accuracy_pos:
                        for file in os.listdir(save_path):
                            if "pos_tagger" in file:
                                os.remove(os.path.join(save_path, file))
                        torch.save(model.state_dict(), os.path.join(save_path, "epoch" + str(epoch) + "-step" + str(step)
                                                                    + "-pos_tagger=" + str(dev_accuracy_pos)[:7] + ".pt"))
                        best_accuracy_pos = dev_accuracy_pos
                        best_result = True
                    if best_result is True:
                        # Eval on the test dataset as well
                        test_accuracy_embed, test_accuracy_classify, test_pos_accuracy = \
                            eval_loop(testloader, trainset.known_lemmas, model, output_layers)
                        print("Test embedding accuracy: " + str(test_accuracy_embed))
                        print("Test classification accuracy: " + str(test_accuracy_classify))
                        print("Test pos tagging accuracy: " + str(test_pos_accuracy))

    print("This is the end.")