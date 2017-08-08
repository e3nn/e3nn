#pylint: disable=C,R,E1101
import os
import imp
import argparse
import logging
import csv
import glob
import random
import numpy as np
import torch
import gc
from time import perf_counter
from se3_cnn.utils import gpu_memory
from se3_cnn.utils import time_logging

def import_module(path):
    file_name = os.path.basename(path)
    model_name = file_name.split('.')[0]
    module = imp.load_source(model_name, path)
    return module

class Dataset:
    def __init__(self, files, ids, labels):
        self.files = files
        self.ids = ids
        self.labels = labels

def load_data(csv_file, files_pattern, classes=None):
    labels = {}

    with open(csv_file, 'rt') as file:
        reader = csv.reader(file)
        for row in reader:
            labels[row[0]] = row[1]

    files = glob.glob(files_pattern)
    random.shuffle(files)

    ids = [file.split("/")[-1].split(".")[0] for file in files]

    # keep only files that appears in the csv
    files, ids = zip(*[(f, i) for f, i in zip(files, ids) if i in labels])

    labels = [labels[i] for i in ids]
    if classes is None:
        classes = sorted(set(labels))
    labels = [classes.index(x) for x in labels]

    return Dataset(files, ids, labels), classes


def train_one_epoch(epoch, model, train_files, train_labels, optimizer, criterion):
    cnn = model.get_cnn()
    bs = model.get_batch_size()
    logger = logging.getLogger("trainer")

    indicies = list(range(len(train_files)))
    random.shuffle(indicies)

    losses = []
    total_correct = 0
    total_trained = 0

    cnn.train()
    cnn.cuda()

    for i in range(0, len(train_files), bs):
        t0 = perf_counter()
        gc.collect()
        j = min(i + bs, len(train_files))

        images = model.load_train_files([train_files[g] for g in indicies[i:j]])
        images = torch.autograd.Variable(images.cuda())

        labels = [train_labels[g] for g in indicies[i:j]]
        labels = torch.autograd.Variable(torch.LongTensor(labels).cuda())

        t = time_logging.start()

        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        time_logging.end("gradient step", t)

        loss_ = float(loss.data.cpu().numpy())
        losses.append(loss_)
        correct = sum(outputs.data.cpu().numpy().argmax(-1) == labels.data.cpu().numpy())
        total_correct += correct
        total_trained += j - i

        logger.info("[%d|%d/%d] Loss=%.2f <Loss>=%.2f Accuracy=%d/%d <Accuracy>=%.2f%% Memory=%s Time=%.2fs",
            epoch, i, len(train_files),
            loss_, np.mean(losses),
            correct, j-i, 100 * total_correct / total_trained,
            gpu_memory.format_memory(gpu_memory.used_memory()),
            perf_counter() - t0)

        del images
        del labels
        del outputs
        del loss
    return (np.mean(losses), total_correct / total_trained)


def evaluate(model, files):
    cnn = model.get_cnn()
    bs = model.get_batch_size()
    logger = logging.getLogger("trainer")

    cnn.eval()
    cnn.cuda()

    all_outputs = []

    for i in range(0, len(files), bs):
        j = min(i + bs, len(files))
        gc.collect()
        images = model.load_eval_files(files[i:j])
        images = images.cuda()

        outputs = model.evaluate(images)

        all_outputs.append(outputs)

        logger.info("Evaluation [%d/%d] Memory=%s",
            i, len(files),
            gpu_memory.format_memory(gpu_memory.used_memory()))

        del images
        del outputs
    return np.concatenate(all_outputs, axis=0)


def save_evaluation(eval_ids, logits, labels, log_dir, number):
    logits = np.array(logits)
    labels = np.array(labels)
    filename = os.path.join(log_dir, "eval{}.csv".format(number))

    with open(filename, "wt") as file:
        writer = csv.writer(file)

        for i, label, ilogits in zip(eval_ids, labels, logits):
            writer.writerow([i, label] + list(ilogits))

    logging.getLogger("trainer").info("Evaluation saved into %s", filename)


def train(args):

    if os.path.isdir(args.log_dir):
        print("{} exists already".format(args.log_dir))
        return

    os.mkdir(args.log_dir)

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("Arguments = %s", repr(args))

    ############################################################################
    # Files and labels
    classes = None

    train_data = None
    eval_datas = []

    if args.train_csv_path is not None or args.train_data_path is not None:
        train_data, classes = load_data(args.train_csv_path, args.train_data_path, classes)
        logger.info("%s=%d training files", "+".join([str(train_data.labels.count(x)) for x in set(train_data.labels)]), len(train_data.files))

    if args.eval_data_path is not None or args.eval_csv_path is not None:
        assert len(args.eval_data_path) == len(args.eval_csv_path)

        for csv_file, pattern in zip(args.eval_csv_path, args.eval_data_path):
            eval_data, classes = load_data(csv_file, pattern, classes)
            eval_datas.append(eval_data)
            logger.info("%s=%d evaluation files", "+".join([str(eval_data.labels.count(x)) for x in set(eval_data.labels)]), len(eval_data.files))

    ############################################################################
    # Import model
    module = import_module(args.model_path)
    model = module.MyModel()
    model.initialize(len(classes))
    cnn = model.get_cnn()

    logger.info("There is %d parameters to optimize", sum([x.numel() for x in cnn.parameters()]))

    if args.restore_path is not None:
        checkpoint = torch.load(os.path.join(args.restore_path, "model.pkl"))
        args.start_epoch = checkpoint['epoch']
        cnn.load_state_dict(checkpoint['state_dict'])
        logger.info("Restoration from file %s", os.path.join(args.restore_path, "model.pkl"))

    ############################################################################
    # Only evaluation
    if train_data is None:
        if args.restore_path is None:
            logger.info("Evalutation with randomly initialized parameters")
        for i, data in enumerate(eval_datas):
            outputs = evaluate(model, data.files)
            save_evaluation(data.ids, outputs, data.labels, args.log_dir, i)
            correct = np.sum(np.argmax(outputs, axis=1) == np.array(data.labels, np.int64))
            logger.info("%d / %d = %.2f%%", correct, len(data.labels), 100 * correct / len(data.labels))
        return

    ############################################################################
    # Optimizer
    optimizer = model.get_optimizer()
    criterion = model.get_criterion()
    criterion.cuda()

    for param_group in optimizer.param_groups:
        param_group['lr'] = model.get_learning_rate(args.start_epoch)

    if args.restore_path is not None:
        checkpoint = torch.load(os.path.join(args.restore_path, "model.pkl"))
        optimizer.load_state_dict(checkpoint['optimizer'])

    ############################################################################
    # Training
    statistics_train = []
    statistics_eval = [[] for _ in eval_datas]

    for epoch in range(args.start_epoch, args.number_of_epochs):
        t = time_logging.start()
        avg_loss, accuracy = train_one_epoch(epoch, model, train_data.files, train_data.labels, optimizer, criterion)
        statistics_train.append([epoch, avg_loss, accuracy])
        time_logging.end("training epoch", t)

        cnn.cpu()
        path = os.path.join(args.log_dir, 'model.pkl')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': cnn.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, path)
        logger.info("Saved in %s", path)

        if epoch % args.eval_each == args.eval_each - 1:
            for i, (data, stat) in enumerate(zip(eval_datas, statistics_eval)):
                outputs = evaluate(model, data.files)
                save_evaluation(data.ids, outputs, data.labels, args.log_dir, i)
                correct = np.sum(np.argmax(outputs, axis=1) == np.array(data.labels, np.int64))
                logger.info("Evaluation accuracy %d / %d = %.2f%%", correct, len(data.labels), 100 * correct / len(data.labels))
                stat.append([epoch, correct / len(data.labels)])

        logger.info("%s", time_logging.text_statistics())
        time_logging.clear()

    statistics_train = np.array(statistics_train)
    np.save(os.path.join(args.log_dir, "statistics_train.npy"), statistics_train)
    statistics_eval = np.array(statistics_eval)
    np.save(os.path.join(args.log_dir, "statistics_eval.npy"), statistics_eval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_epochs", type=int, required=True)
    parser.add_argument("--start_epoch", type=int, default=0)

    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--train_csv_path", type=str)

    parser.add_argument("--eval_data_path", type=str, nargs="+")
    parser.add_argument("--eval_csv_path", type=str, nargs="+")
    parser.add_argument("--eval_each", type=int, default=1)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--restore_path", type=str)

    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
