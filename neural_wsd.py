import argparse
import copy
import os
import pickle
import sys

import itertools
import numpy
import torch
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchcrf import CRF

from auxiliary import Logger
from data_ops import WSDataset, BatchSchedulerSampler, load_embeddings, get_wordnet_lexicon, get_framenet_lexicon
from wsd_model import WSDModel, calculate_accuracy_embedding, calculate_accuracy_classification, \
    calculate_accuracy_classification_wsd, calculate_accuracy_crf, calculate_f1_ner, get_granular_f1


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
    matches_embed_wsd_all, total_embed_wsd_all = 0, 0
    matches_embed_frameID_all, total_embed_frameID_all = 0, 0
    matches_classify_all, total_classify_all = 0, 0
    matches_pos_all, total_pos_all = 0, 0
    tps_all, fps_all, fns_all = 0, 0, 0
    accuracy_embed_wsd, accuracy_embed_frameID, accuracy_classify, accuracy_pos, f1_ner = 0.0, 0.0, 0.0, 0.0, 0.0
    log = "LEMMA\tALL OPTIONS\tSELECTED SENSE\tGOLD SENSES\n"
    for eval_data in data_loader:
        # lemmas = numpy.asarray(eval_data['lemmas_pos']).transpose()[eval_data["mask"]]
        batch_layers = [layers[0] for layers in eval_data['batch_layers']]
        if pos_filter is True:
            lemmas = numpy.asarray(eval_data['lemmas_pos']).transpose()[eval_data["mask"].cpu()]
        else:
            lemmas = numpy.asarray(eval_data['lemmas']).transpose()[eval_data["mask"].cpu()]
        default_disambiguations = disambiguate_by_default(lemmas, known_lemmas)
        senses = numpy.asarray(eval_data['senses']).transpose()[eval_data["mask"].cpu()]
        # pos = numpy.asarray(eval_data['pos']).transpose()[eval_data["mask"]]
        # targets_classify = torch.from_numpy(numpy.asarray(eval_data["targets_classify"])[eval_data["mask"]])
        # targets_pos = torch.from_numpy(numpy.asarray(eval_data["targets_pos"])[eval_data["pos_mask"]])
        # outputs = model(eval_data["inputs"], eval_data["length"], eval_data["mask"], eval_data["pos_mask"], lemmas)
        outputs = model(eval_data, lemmas)
        if "embed_wsd" in output_layers and "embed_wsd" in batch_layers:
            matches_embed_wsd, total_embed_wsd = calculate_accuracy_embedding(outputs["embed_wsd"].cpu(),
                                                                              lemmas,
                                                                              senses,
                                                                              lemma2synsets,
                                                                              embeddings,
                                                                              src2id,
                                                                              pos_filter=True)
            matches_embed_wsd_all += matches_embed_wsd
            total_embed_wsd_all += total_embed_wsd
        if "embed_frameID" in output_layers and "embed_frameID" in batch_layers:
            matches_embed_frameID, total_embed_frameID = calculate_accuracy_embedding(outputs["embed_frameID"].cpu(),
                                                                                      lemmas,
                                                                                      senses,
                                                                                      lu2frames,
                                                                                      embeddings.cpu(),
                                                                                      src2id,
                                                                                      pos_filter=True)
            matches_embed_frameID_all += matches_embed_frameID
            total_embed_frameID_all += total_embed_frameID
        if "classify_wsd" in output_layers and "classify_wsd" in batch_layers:
            targets_classify = torch.masked_select(eval_data["targets_classify"], eval_data["mask"])
            matches_classify, total_classify, batch_log = calculate_accuracy_classification_wsd(
                outputs["classify_wsd"].cpu().detach().numpy(),
                targets_classify.cpu().detach().numpy(),
                default_disambiguations,
                lemmas,
                known_lemmas,
                senses,
                lemma2synsets,
                synset2id,
                single_softmax)
            log += batch_log
            matches_classify_all += matches_classify
            total_classify_all += total_classify
        if "pos_tagger" in output_layers and "pos_tagger" in batch_layers:
            if crf_layer is True:
                targets_pos = eval_data["targets_pos"][:, :outputs["pos_tagger"].shape[1]]
                mask_crf_pos = eval_data["pos_mask"][:, :outputs["pos_tagger"].shape[1]]
                matches_pos, total_pos = calculate_accuracy_crf(loss_func_pos,
                                                                outputs["pos_tagger"],
                                                                mask_crf_pos,
                                                                targets_pos)
            else:
                targets_pos = torch.from_numpy(numpy.asarray(eval_data["targets_pos"])[eval_data["pos_mask"].cpu()])
                mask_pos = eval_data["pos_mask"][:, :outputs["pos_tagger"].shape[1]]
                mask_pos = torch.reshape(mask_pos, (mask_pos.shape[0], mask_pos.shape[1], 1))
                outputs_pos = torch.masked_select(outputs["pos_tagger"], mask_pos).view(-1, len(trainset.known_pos))
                matches_pos, total_pos = calculate_accuracy_classification(outputs_pos, targets_pos)
            matches_pos_all += matches_pos
            total_pos_all += total_pos

        if "ner" in output_layers and "ner" in batch_layers:
            outputs_ner = outputs["ner"]
            if crf_layer is True:
                mask_crf_ner = eval_data["ner_mask"].cpu()[:, :outputs["ner"].shape[1]]
                targets_ner = eval_data["targets_ner"][:, :outputs["ner"].shape[1]]
                outputs_ner = loss_func_ner.decode(outputs_ner.cpu(), mask=mask_crf_ner.cpu())
            else:
                targets_ner = torch.from_numpy(numpy.asarray(eval_data["targets_ner"])[eval_data["ner_mask"].cpu()])
                outputs_ner = outputs["ner"]
                outputs_ner = torch.argmax(outputs_ner, dim=2)
                targets_ner = slice_and_pad(targets_ner, eval_data["length"])
                outputs_ner = outputs_ner.numpy()
            f1_ner, [tps, fps, fns], _ = calculate_f1_ner(outputs_ner,
                                                          targets_ner.cpu().numpy(),
                                                          eval_data["length"],
                                                          trainset.known_entity_tags)
            tps_all += tps
            fps_all += fps
            fns_all += fns
        accuracy_embed_wsd = matches_embed_wsd_all * 1.0 / total_embed_wsd_all if total_embed_wsd_all > 0 else 0
        accuracy_embed_frameID = matches_embed_frameID_all * 1.0 / total_embed_frameID_all if total_embed_frameID_all > 0 else 0
        accuracy_classify = matches_classify_all * 1.0 / total_classify_all if total_classify_all > 0 else 0
        accuracy_pos = matches_pos_all * 1.0 / total_pos_all if total_pos_all > 0 else 0
        _, _, f1_ner = get_granular_f1(tps_all, fps_all, fns_all)
    return accuracy_embed_wsd, accuracy_embed_frameID, accuracy_classify, accuracy_pos, f1_ner, log

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and evaluate a neural WSD model.', fromfile_prefix_chars='@')
    parser.add_argument('-alpha', dest='alpha', required=False, default=1,
                        help='Coefficient to weight the positive and negative labels.')
    parser.add_argument('-batch_size', dest='batch_size', required=False, default=128,
                        help='Size of the training batches.')
    parser.add_argument('-combine_WN_FN', dest='combine_WN_FN', required=False, default=False,
                        help='True if information is provided from both WordNet and FrameNet corpora.')
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
    parser.add_argument('-epochs', dest='epochs', required=False, default=100,
                        help='How many epochs should the NN train for?')
    parser.add_argument('-f_indices', dest='f_indices', required=False, default="",
                        help='File storing the indices for the train/dev/test split of the data, if reading from 1 dataset.')
    parser.add_argument('-f_pos_map', dest='f_pos_map', required=False,
                        help='File with mapping between more and less granular tagsets.')
    parser.add_argument('-frame_path_dev', dest='frame_path_dev', required=False,
                        help='Path to frame dev data.')
    parser.add_argument('-frame_path_test', dest='frame_path_test', required=False,
                        help='Path to frame test data.')
    parser.add_argument('-frame_path_train', dest='frame_path_train', required=False,
                        help='Path to frame training data.')
    parser.add_argument('-language', dest='language', required=True,
                        help='What language is processing done in: English or Bulgarian?')
    parser.add_argument('-learning_rate', dest='learning_rate', required=False, default=0.2,
                        help='How fast the network should learn.')
    parser.add_argument('-lexicon_path', dest='lexicon_path', required=False,
                        help='The path to the location of the lexicon file (WordNet).')
    parser.add_argument('-lexicon_fn_path', dest='lexicon_fn_path', required=False,
                        help='The path to the location of the lexicon file (FrameNet).')
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
                             'More than one can be provided, delimiting them by commas, e.g. "embed_wsd,classify_wsd"')
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
    parser.add_argument('-use_flair', dest='use_flair', required=False, default="False",
                        help='Whether the Flair library should be used to embed the inputs.')

    # Figure out what device to run the network on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is :" + str(device))
    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Get the embeddings and lexicon
    args = parser.parse_args()
    lang = args.language
    embeddings_path = args.embeddings_path
    use_flair = True if args.use_flair == "True" else False
    combine_WN_FN = True if args.combine_WN_FN == "True" else False
    embeddings, src2id, id2src = load_embeddings(embeddings_path)
    embeddings = torch.Tensor(embeddings)
    embedding_dim = embeddings.shape[1]
    embeddings_input = args.embeddings_input
    lexicon_path = args.lexicon_path
    lexicon_fn_path = args.lexicon_fn_path
    pos_filter = True if args.pos_filter == "True" else False
    if lexicon_path is not None:
        lemma2synsets, max_labels = get_wordnet_lexicon(lexicon_path, pos_filter)
    else:
        lemma2synsets, max_labels = {}, 0
    if lexicon_fn_path is not None:
        lu2frames = get_framenet_lexicon(lexicon_fn_path)
    crf_layer = True if args.crf_layer == "True" else False
    # if crf_layer is True:
    #     single_softmax = True
    # else:
    single_softmax = True if args.n_classifiers == "single" else False
    synset2id = {}

    # Get model parameters
    output_layers = [str.strip(layer) for layer in args.output_layers.split(",")]
    alpha = float(args.alpha)
    batch_size = int(args.batch_size)
    hidden_neurons = int(args.n_hidden_neurons)
    hidden_layers = int(args.n_hidden_layers)
    dropout = float(args.dropout)
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)

    # Get the training/dev/testing data
    save_path = args.save_path
    train_path = args.train_data_path
    dev_path = args.dev_data_path
    test_path = args.test_data_path
    frame_path_train = args.frame_path_train
    frame_path_dev = args.frame_path_dev
    frame_path_test = args.frame_path_test
    f_indices = args.f_indices
    f_pos_map = args.f_pos_map
    split_dataset = False

    if dev_path is None and test_path is None:
        split_dataset = True
    primary_batch_layers = copy.copy(output_layers)
    if "embed_frameID" in output_layers:
        primary_batch_layers.remove("embed_frameID")
        frame_batch_layers = ["embed_frameID"]
    # if "embed_frameID" in primary_batch_layers:
    #     lu2concepts = lu2frames
    #     train_path, dev_path, test_path = frame_path_train, frame_path_dev, frame_path_test
    # else:
    #     lu2concepts = lemma2synsets
    # trainset = WSDataset(device, train_path, src2id, embeddings, embedding_dim, embeddings_input, max_labels,
    #                      lu2concepts, single_softmax, primary_batch_layers, pos_map=f_pos_map, pos_filter=pos_filter)
    # known_pos = trainset.known_pos
    # known_lemmas = trainset.known_lus
    # known_entity_tags = trainset.known_entity_tags
    #
    # if single_softmax is True:
    #     synset2id = trainset.known_senses
    # else:
    #     synset2id = {}

    # if there is only a single dataset for train/dev/test purposes, sample from it; else, just load the dev/test-sets
    trainsampler, devsampler, testsampler = None, None, None
    known_pos, known_lemmas, known_entity_tags = {}, {}, {}

    if split_dataset is False:
        if len(primary_batch_layers) > 0:
            trainset = WSDataset(device, train_path, src2id, embeddings, embedding_dim, embeddings_input, max_labels,
                                 lemma2synsets, single_softmax, primary_batch_layers, pos_map=f_pos_map,
                                 pos_filter=pos_filter)
            known_pos = trainset.known_pos
            known_lemmas = trainset.known_lus
            known_entity_tags = trainset.known_entity_tags

            if single_softmax is True:
                synset2id = trainset.known_senses

            devset = WSDataset(device, dev_path, src2id, embeddings, embedding_dim, embeddings_input, max_labels,
                               lemma2synsets, single_softmax, primary_batch_layers, synset2id, pos_filter=pos_filter)
            testset = WSDataset(device, test_path, src2id, embeddings, embedding_dim, embeddings_input, max_labels,
                                lemma2synsets, single_softmax, primary_batch_layers, synset2id, pos_filter=pos_filter)
            devloader = torch.utils.data.DataLoader(dataset=devset, batch_size=batch_size, shuffle=False)
            testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
        if "embed_frameID" in output_layers:
            trainset_frameID = WSDataset(device, frame_path_train, src2id, embeddings, embedding_dim, embeddings_input,
                                         max_labels, lu2frames, single_softmax, frame_batch_layers, pos_map=f_pos_map,
                                         pos_filter=True)
            devset_frameID = WSDataset(device, frame_path_dev, src2id, embeddings, embedding_dim, embeddings_input,
                                       max_labels, lu2frames, single_softmax, frame_batch_layers, pos_map=f_pos_map,
                                       pos_filter=True)
            testset_frameID = WSDataset(device, frame_path_test, src2id, embeddings, embedding_dim, embeddings_input,
                                        max_labels, lu2frames, single_softmax, frame_batch_layers, pos_map=f_pos_map,
                                        pos_filter=True)
            devloader_frameID = torch.utils.data.DataLoader(dataset=devset_frameID, batch_size=batch_size, shuffle=False)
            testloader_frameID = torch.utils.data.DataLoader(dataset=testset_frameID, batch_size=batch_size, shuffle=False)
            if combine_WN_FN is True:
                trainset = ConcatDataset([trainset, trainset_frameID])
                trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                                          sampler=BatchSchedulerSampler(dataset=trainset,
                                                                                        batch_size=batch_size),
                                                          batch_size=batch_size,
                                                          shuffle=False)
            else:
                trainloader = torch.utils.data.DataLoader(trainset_frameID, batch_size=batch_size, shuffle=True)
                devloader = torch.utils.data.DataLoader(dataset=devset_frameID, batch_size=batch_size,
                                                                shuffle=False)
                testloader = torch.utils.data.DataLoader(dataset=testset_frameID, batch_size=batch_size,
                                                                 shuffle=False)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # else:
    #     if os.path.exists(f_indices):
    #         with open(f_indices, "rb") as f:
    #             train_indices = pickle.load(f)
    #             dev_indices = pickle.load(f)
    #             test_indices = pickle.load(f)
    #     else:
    #         dataset_size = len(trainset)
    #         indices = list(range(dataset_size))
    #         split = int(numpy.floor(0.1 * dataset_size))
    #         numpy.random.seed(42)
    #         numpy.random.shuffle(indices)
    #         dev_indices, test_indices, train_indices = indices[:split], indices[split:2*split], indices[2*split:]
    #         with open(f_indices, "wb") as f:
    #             pickle.dump(train_indices, f, protocol=2)
    #             pickle.dump(dev_indices, f, protocol=2)
    #             pickle.dump(test_indices, f, protocol=2)
    #         f.close()
    #     trainsampler = SubsetRandomSampler(train_indices)
    #     devsampler = SubsetRandomSampler(dev_indices)
    #     testsampler = SubsetRandomSampler(test_indices)
    #     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=trainsampler)
    #     devloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=devsampler)
    #     testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=testsampler)

    # Construct data loaders for this specific model


    # Redirect print statements to both terminal and logfile
    results_file = open(os.path.join(save_path, "results.txt"), 'w+')
    # sys.stdout = Logger(os.path.join(save_path, "results.txt"))
    sys.stdout = Logger(os.path.join(save_path, "results.txt"))

    # Construct the model
    model = WSDModel(lang, embedding_dim, embeddings, hidden_neurons, hidden_layers, dropout, output_layers, lemma2synsets,
                     synset2id, known_pos, known_entity_tags, use_flair=use_flair, combine_WN_FN=combine_WN_FN)
    model.to(device)
    loss_func_embed = torch.nn.MSELoss()
    if crf_layer is True:
        if "classify_wsd" in output_layers:
            loss_func_classify = torch.nn.CrossEntropyLoss(ignore_index=-100)
        if "pos_tagger" in output_layers:
            loss_func_pos = CRF(len(known_pos), batch_first=True)
        if "ner" in output_layers:
            loss_func_ner = CRF(len(known_entity_tags), batch_first=True)
    else:
        loss_func_classify = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss_func_pos = torch.nn.CrossEntropyLoss()
        loss_func_ner = torch.nn.CrossEntropyLoss()
    # loss_func_classify = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Eval loop
    if args.mode == "evaluate":
        model.load_state_dict(torch.load(args.save_path))
        model.eval()
        test_accuracy_embed, test_accuracy_classify, log = eval_loop(testloader, model, output_layers)
        print("Test embedding accuracy: " + str(test_accuracy_embed))
        print("Test classification accuracy: " + str(test_accuracy_classify))

    # Train loop
    else:
        print(args)
        least_loss = 0.0
        best_accuracy_embed_wsd, best_accuracy_embed_frameID, best_accuracy_classify, best_accuracy_pos, best_f1_ner = \
            0.0, 0.0, 0.0, 0.0, 0.0
        best_test_embed_wsd, best_test_frameID, best_test_classify, best_test_pos, best_test_ner = 0.0, 0.0, 0.0, 0.0, 0.0
        eval_at = 100
        for epoch in range(epochs):
            print("***** Start of epoch " + str(epoch) + " *****")
            average_loss_embed_wsd, average_loss_embed_frameID, average_loss_classify, average_loss_pos, \
            average_loss_ner, average_loss_overall = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            count_embed_wsd, count_embed_frameID, count_classify, count_pos, count_ner = 0, 0, 0, 0, 0
            for step, data in enumerate(trainloader):
                model.train()
                optimizer.zero_grad()
                # lemmas = numpy.asarray(data['lemmas_pos']).transpose()[data["mask"]]
                batch_layers = [layers[0] for layers in data['batch_layers']]
                if pos_filter is True:
                    lemmas = numpy.asarray(data['lemmas_pos']).transpose()[data["mask"].cpu()]
                else:
                    lemmas = numpy.asarray(data['lemmas']).transpose()[data["mask"].cpu()] # TODO: parametrize
                default_disambiguations = disambiguate_by_default(lemmas, known_lemmas)
                senses = numpy.asarray(data['senses']).transpose()[data["mask"].cpu()]
                # lengths_labels = numpy.asarray(data["lengths_labels"])[data["mask"]]
                # outputs = model(data["inputs"], data["length"], data["mask"], data["pos_mask"], lemmas)
                outputs = model(data, lemmas)
                loss = 0.0
                # Calculate loss for the context embedding method
                if "embed_wsd" in output_layers and "embed_wsd" in batch_layers:
                    mask_embed = torch.reshape(data["mask"], (data["mask"].shape[0], data["mask"].shape[1], 1))
                    targets_embed = torch.masked_select(data["targets_embed"], mask_embed)
                    targets_embed = targets_embed.view(-1, embedding_dim)
                    neg_targets = torch.masked_select(data["neg_targets"], mask_embed)
                    neg_targets = neg_targets.view(-1, embedding_dim)
                    # targets_classify = targets_classify.view(-1, max_labels)
                    loss_embed = alpha * loss_func_embed(outputs["embed_wsd"], targets_embed) + \
                                 (1 - alpha) * (1 - loss_func_embed(outputs["embed_wsd"], neg_targets))
                    loss += loss_embed
                    average_loss_embed_wsd += loss_embed
                    count_embed_wsd += 1
                # Calculate loss for the frame identification embedding method
                if "embed_frameID" in output_layers and "embed_frameID" in batch_layers:
                    mask_embed = torch.reshape(data["mask"], (data["mask"].shape[0], data["mask"].shape[1], 1))
                    targets_embed = torch.masked_select(data["targets_embed"], mask_embed)
                    targets_embed = targets_embed.view(-1, embedding_dim)
                    neg_targets = torch.masked_select(data["neg_targets"], mask_embed)
                    neg_targets = neg_targets.view(-1, embedding_dim)
                    # targets_classify = targets_classify.view(-1, max_labels)
                    loss_embed = alpha * loss_func_embed(outputs["embed_frameID"], targets_embed) + \
                                 (1 - alpha) * (1 - loss_func_embed(outputs["embed_frameID"], neg_targets))
                    loss += loss_embed
                    average_loss_embed_frameID += loss_embed
                    count_embed_frameID += 1
                # Calculate loss for the classification method
                if "classify_wsd" in output_layers and "classify_wsd" in batch_layers:
                    # targets_classify = torch.from_numpy(numpy.asarray(data["targets_classify"].cpu())[data["mask"].cpu()])
                    targets_classify = data["targets_classify"][data["mask"]]
                    loss_classify = loss_func_classify(outputs["classify_wsd"], targets_classify)
                    loss += loss_classify
                    average_loss_classify += loss_classify
                    count_classify += 1
                if "pos_tagger" in output_layers and "pos_tagger" in batch_layers:
                    if crf_layer is True:
                        mask_crf_pos = data["pos_mask"][:, :outputs["pos_tagger"].shape[1]]
                        targets_pos = data["targets_pos"][:, :outputs["pos_tagger"].shape[1]]
                        loss_pos = loss_func_pos(outputs["pos_tagger"], targets_pos, mask_crf_pos, reduction="mean")
                        loss_pos *= (-1.0 if crf_layer is True else 1.0)
                    else:
                        targets_pos = torch.from_numpy(numpy.asarray(data["targets_pos"])[data["pos_mask"].cpu()])
                        mask_pos = data["pos_mask"][:, :outputs["pos_tagger"].shape[1]]
                        mask_pos = torch.reshape(mask_pos, (mask_pos.shape[0], mask_pos.shape[1], 1))
                        outputs_pos = torch.masked_select(outputs["pos_tagger"], mask_pos).view(-1, len(trainset.known_pos))
                        loss_pos = loss_func_pos(outputs_pos, targets_pos)
                    loss += loss_pos
                    average_loss_pos += loss_pos
                    count_pos += 1
                if "ner" in output_layers and "ner" in batch_layers:
                    if crf_layer is True:
                        mask_crf_ner = data["ner_mask"][:, :outputs["ner"].shape[1]]
                        targets_ner = data["targets_ner"][:, :outputs["ner"].shape[1]]
                        loss_ner = loss_func_ner(outputs["ner"].cpu(), targets_ner.cpu(), mask_crf_ner.cpu(), reduction="mean")
                        loss_ner *= (-1.0 if crf_layer is True else 1.0)
                    else:
                        targets_ner = torch.from_numpy(numpy.asarray(data["targets_ner"])[data["ner_mask"].cpu()])
                        mask_ner = data["ner_mask"][:, :outputs["ner"].shape[1]]
                        # mask_ner = mask_ner[:, :outputs["ner"].shape[1]]
                        mask_ner = torch.reshape(mask_ner, (mask_ner.shape[0], mask_ner.shape[1], 1))
                        outputs_ner = torch.masked_select(outputs["ner"], mask_ner).view(-1, len(trainset.known_entity_tags))
                        loss_ner = loss_func_ner(outputs_ner, targets_ner)
                    loss += loss_ner
                    average_loss_ner += loss_ner
                    count_ner += 1
                # loss = loss_embed + loss_classify
                loss.backward()
                optimizer.step()
                # Eval loop during training
                if step % eval_at == 0:
                    model.eval()
                    test_accuracy_embed_wsd, test_accuracy_classify = 0.0, 0.0
                    print("Step " + str(step))
                    if "embed_wsd" in output_layers and count_embed_wsd > 0:
                        average_loss_embed_wsd /= count_embed_wsd
                        print("Average embedding loss (training): " + str(average_loss_embed_wsd.cpu().detach().numpy()))
                        average_loss_overall += average_loss_embed_wsd.cpu().detach().numpy()
                    if "embed_frameID" in output_layers and count_embed_frameID > 0:
                        average_loss_embed_frameID /= count_embed_frameID
                        print("Average frameID loss (training): " + str(average_loss_embed_frameID.cpu().detach().numpy()))
                        average_loss_overall += average_loss_embed_frameID.cpu().detach().numpy()
                    if "classify_wsd" in output_layers and count_classify > 0:
                        average_loss_classify /= count_classify
                        print("Average classification loss (training): " + str(average_loss_classify.cpu().detach().numpy()))
                        average_loss_overall += average_loss_classify.cpu().detach().numpy()
                    if "pos_tagger" in output_layers:
                        average_loss_pos /= count_pos
                        print("Average pos tagger loss (training): " + str(average_loss_pos.cpu().detach().numpy()))
                        average_loss_overall += average_loss_pos.cpu().detach().numpy()
                    if "ner" in output_layers:
                        average_loss_ner /= count_ner
                        print("Average ner tagger loss (training): " + str(average_loss_ner.cpu().detach().numpy()))
                        average_loss_overall += average_loss_ner.cpu().detach().numpy()
                        if crf_layer is True:
                            matches_ner, total_ner = calculate_accuracy_crf(loss_func_ner,
                                                                            outputs["ner"].cpu(),
                                                                            mask_crf_ner.cpu(),
                                                                            targets_ner.cpu())
                            outputs_ner = loss_func_ner.decode(outputs["ner"].cpu(), mask=mask_crf_ner.cpu())
                        else:
                            matches_ner, total_ner = calculate_accuracy_classification(outputs_ner, targets_ner)
                            outputs_ner = torch.argmax(outputs_ner, dim=1)
                            outputs_ner = slice_and_pad(outputs_ner, data["length"])
                            targets_ner = slice_and_pad(targets_ner, data["length"])
                            outputs_ner = outputs_ner.numpy()
                        f1_ner, _, _ = calculate_f1_ner(outputs_ner,
                                                        targets_ner.cpu().numpy(),
                                                        data["length"],
                                                        trainset.known_entity_tags)
                        train_accuracy_ner = matches_ner * 1.0 / total_ner

                        print("Training ner tagger accuracy: " + str(train_accuracy_ner))
                        print("F1-score for NER: " + str(f1_ner))
                        average_loss_ner /= (eval_at if step != 0 else 1)
                        print("Average ner tagger loss (training): " + str(average_loss_ner.detach().numpy()))
                        average_loss_overall += average_loss_ner.detach().numpy()
                    print("Average overall loss (training): " + str(average_loss_overall))
                    average_loss_embed_wsd, average_loss_embed_frameID, average_loss_classif, average_loss_pos, \
                    average_loss_ner, average_loss_overall = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    count_embed_wsd, count_embed_frameID, count_classify, count_pos, count_ner = 0, 0, 0, 0, 0

                    # Loop over the dev dataset
                    if combine_WN_FN is True:
                        dev_accuracy_embed_wsd, _, dev_accuracy_classify, dev_accuracy_pos, dev_f1_ner, _ = \
                            eval_loop(devloader, known_lemmas, model, output_layers)
                        _, dev_accuracy_embed_frameID, _, _, _, _ = eval_loop(devloader_frameID, known_lemmas, model, output_layers)
                    else:
                        dev_accuracy_embed_wsd, dev_accuracy_embed_frameID, dev_accuracy_classify, dev_accuracy_pos, dev_f1_ner,\
                        dev_log = eval_loop(devloader, known_lemmas, model, output_layers)
                    if "embed_wsd" in output_layers:
                        print("Dev embedding accuracy (WSD): " + str(dev_accuracy_embed_wsd))
                    if "embed_frameID" in output_layers:
                        print("Dev embedding accuracy (frameID): " + str(dev_accuracy_embed_frameID))
                    if "classify_wsd" in output_layers:
                        print("Dev classification accuracy: " + str(dev_accuracy_classify))
                    if "pos_tagger" in output_layers:
                        print("Dev pos tagging accuracy: " + str(dev_accuracy_pos))
                    if "ner" in output_layers:
                        print("Dev ner F1 score: " + str(dev_f1_ner))
                    best_result = False
                    if dev_accuracy_embed_wsd > best_accuracy_embed_wsd and "embed_wsd" in output_layers:
                        for file in os.listdir(save_path):
                            if "embed_wsd" in file:
                                os.remove(os.path.join(save_path, file))
                        torch.save(model.state_dict(), os.path.join(save_path, "epoch" + str(epoch) + "-step" + str(step)
                                                                    + "-embed_wsd=" + str(dev_accuracy_embed_wsd)[:7] + ".pt"))

                        best_accuracy_embed_wsd = dev_accuracy_embed_wsd
                        if test_accuracy_embed_wsd == 0:
                            test_accuracy_embed_wsd, _, test_accuracy_classify, test_pos_accuracy, test_n1_fscore, \
                            test_log = eval_loop(testloader, known_lemmas, model, output_layers)
                        best_test_embed_wsd = test_accuracy_embed_wsd \
                            if test_accuracy_embed_wsd > best_test_embed_wsd else best_test_embed_wsd
                        print("Test embedding accuracy (wsd): " + str(test_accuracy_embed_wsd))
                        test_accuracy_embed_wsd = 0.0
                        best_result = True
                    if dev_accuracy_embed_frameID > best_accuracy_embed_frameID and "embed_frameID" in output_layers:
                        for file in os.listdir(save_path):
                            if "frameID" in file:
                                os.remove(os.path.join(save_path, file))
                        torch.save(model.state_dict(), os.path.join(save_path, "epoch" + str(epoch) + "-step" + str(step)
                                                                    + "-frameID=" + str(dev_accuracy_embed_frameID)[:7] + ".pt"))
                        best_accuracy_embed_frameID = dev_accuracy_embed_frameID
                        _, test_accuracy_frameID, _, _, _, _ = eval_loop(testloader_frameID, known_lemmas, model,
                                                                         output_layers)
                        best_test_frameID = test_accuracy_frameID \
                            if test_accuracy_frameID > best_test_frameID else best_test_frameID
                        print("Test embedding accuracy (frameID): " + str(test_accuracy_frameID))
                        test_accuracy_frameID = 0.0
                        best_result = True
                    if dev_accuracy_classify > best_accuracy_classify and "classify_wsd" in output_layers:
                        for file in os.listdir(save_path):
                            if "classify_wsd" in file:
                                os.remove(os.path.join(save_path, file))
                        torch.save(model.state_dict(), os.path.join(save_path, "epoch" + str(epoch) + "-step" + str(step)
                                                                    + "-classify_wsd=" + str(dev_accuracy_classify)[:7] + ".pt"))
                        best_accuracy_classify = dev_accuracy_classify
                        if test_accuracy_classify == 0:
                            test_accuracy_embed_wsd, _, test_accuracy_classify, test_pos_accuracy, test_n1_fscore, \
                            test_log = eval_loop(testloader, known_lemmas, model, output_layers)
                        best_test_classify = test_accuracy_classify \
                            if test_accuracy_classify > best_test_classify else best_test_classify
                        print("Test WSD accuracy (classification): " + str(test_accuracy_classify))
                        test_accuracy_classify = 0.0
                        best_result = True
                    if dev_accuracy_pos > best_accuracy_pos and "pos_tagger" in output_layers:
                        for file in os.listdir(save_path):
                            if "pos_tagger" in file:
                                os.remove(os.path.join(save_path, file))
                        torch.save(model.state_dict(), os.path.join(save_path, "epoch" + str(epoch) + "-step" + str(step)
                                                                    + "-pos_tagger=" + str(dev_accuracy_pos)[:7] + ".pt"))
                        best_accuracy_pos = dev_accuracy_pos
                        best_result = True
                    if dev_f1_ner > best_f1_ner and "ner" in output_layers:
                        for file in os.listdir(save_path):
                            if "ner" in file:
                                os.remove(os.path.join(save_path, file))
                        torch.save(model.state_dict(), os.path.join(save_path, "epoch" + str(epoch) + "-step" + str(step)
                                                                    + "-ner=" + str(dev_accuracy_pos)[:7] + ".pt"))
                        best_f1_ner = dev_f1_ner
                        best_result = True
        print("Best context embedding accuracy on the test data: " + str(best_test_embed_wsd))
        print("Best frameID accuracy on the test data: " + str(best_test_frameID))
        print("Best WSD accuracy on the test data (classification): " + str(best_test_classify))
        print("Best POS tagging accuracy on the test data: " + str(best_test_pos))
        print("Best F1-score on the test data: " + str(best_test_ner))

    print("This is the end.")