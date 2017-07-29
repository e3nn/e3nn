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


def load_data(csv_file, files_pattern, classes=None):
    labels = {}

    with open(csv_file, 'rt') as file:
        reader = csv.reader(file)
        for row in reader:
            labels[row[0]] = row[1]

    files = glob.glob(files_pattern)
    random.shuffle(files)

    ids = [file.split("/")[-1].split(".")[0] for file in files]
    labels = [labels[i] for i in ids]
    if classes is None:
        classes = sorted(set(labels))
    labels = [classes.index(x) for x in labels]

    return files, ids, labels, classes


def train_one_epoch(epoch, model, train_files, train_labels, optimizer, criterion):
    cnn = model.get_cnn()
    bs = model.get_batch_size()
    logger = logging.getLogger("trainer")

    losses = []
    accuracies = []

    cnn.train()
    cnn.cuda()

    for i in range(0, len(train_files), bs):
        t0 = perf_counter()

        j = min(i + bs, len(train_files))
        gc.collect()
        images = model.load_files(train_files[i:j])
        images = images.cuda()

        labels = train_labels[i:j]
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
        accuracies.append(correct / bs)

        logger.info("[%d|%d/%d] Loss=%.2f <Loss>=%.2f Accuracy=%d/%d <Accuracy>=%.2f%% Memory=%s Time=%.2fs",
        epoch, i, len(train_files), loss_, np.mean(losses), correct, bs, 100 * np.mean(accuracies), gpu_memory.format_memory(gpu_memory.used_memory()), perf_counter() - t0)

        del images
        del labels
        del outputs
        del loss



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
        images = model.load_files(files[i:j])
        images = images.cuda()
        # images.volatile = True

        outputs = model.evaluate(images)

        all_outputs.append(outputs.data.cpu().numpy())

        logger.info("Evaluation [%d/%d] Memory=%s", i, len(files), gpu_memory.format_memory(gpu_memory.used_memory()))

        del images
        del outputs
    return np.concatenate(all_outputs, axis=0)


def save_evaluation(eval_ids, logits, labels, log_dir):
    logits = np.array(logits)
    labels = np.array(labels)

    with open(os.path.join(log_dir, "eval.csv"), "wt") as file:
        writer = csv.writer(file)

        for i, label, ilogits in zip(eval_ids, labels, logits):
            writer.writerow([i, label] + list(ilogits))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_epochs", type=int)
    parser.add_argument("--start_epoch", type=int, default=0)

    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--train_csv_path", type=str)

    parser.add_argument("--eval_data_path", type=str)
    parser.add_argument("--eval_csv_path", type=str)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--restore_path", type=str)

    args = parser.parse_args()

    if os.path.isdir(args.log_dir):
        print("{} exists already".format(args.log_dir))
        return

    os.mkdir(args.log_dir)

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    ch = logging.StreamHandler()
    logger.addHandler(fh)
    logger.addHandler(ch)

    ############################################################################
    # Import model
    module = import_module(args.model_path)
    model = module.MyModel()
    cnn = model.get_cnn()

    logger.info("There is %d parameters to optimize", sum([x.numel() for x in cnn.parameters()]))

    if args.restore_path is not None:
        checkpoint = torch.load(args.restore_path)
        args.start_epoch = checkpoint['epoch']
        cnn.load_state_dict(checkpoint['state_dict'])
        logger.info("Restoration from file %s", args.restore_path)

    ############################################################################
    # Files and labels
    classes = None
    train_files = eval_files = None
    if args.train_csv_path is not None and args.train_data_path is not None:
        train_files, train_ids, train_labels, classes = load_data(args.train_csv_path, args.train_data_path, classes)
        logger.info("%s=%d training files", "+".join([str(train_labels.count(x)) for x in set(train_labels)]), len(train_files))
    if args.eval_csv_path is not None and args.eval_data_path is not None:
        eval_files, eval_ids, eval_labels, classes = load_data(args.eval_csv_path, args.eval_data_path, classes)
        logger.info("%s=%d evaluation files", "+".join([str(eval_labels.count(x)) for x in set(eval_labels)]), len(eval_files))

    ############################################################################
    # Only evaluation
    if train_files is None:
        if args.restore_path is None:
            logger.info("Evalutation with randomly initialized paramters")
        outputs = evaluate(model, eval_files)
        save_evaluation(eval_ids, outputs, eval_labels, args.log_dir)
        correct = np.sum(np.argmax(outputs, axis=1) == np.array(eval_labels, np.int64))
        logger.info("%d / %d = %.2f%%", correct, len(eval_labels), 100 * correct / len(eval_labels))
        return

    ############################################################################
    # Optimizer
    optimizer = model.get_optimizer()
    criterion = model.get_criterion()
    criterion.cuda()

    for param_group in optimizer.param_groups:
        param_group['lr'] = model.get_learning_rate(args.start_epoch)

    if args.restore_path is not None:
        checkpoint = torch.load(args.restore_path)
        optimizer.load_state_dict(checkpoint['optimizer'])

    ############################################################################
    # Training
    for epoch in range(args.start_epoch, args.number_of_epochs):
        t = time_logging.start()
        train_one_epoch(epoch, model, train_files, train_labels, optimizer, criterion)
        time_logging.end("training epoch", t)

        cnn.cpu()
        path = os.path.join(args.log_dir, 'model.pkl')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': cnn.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, path)
        logger.info("Saved in %s", path)

        if eval_files is not None:
            outputs = evaluate(model, eval_files)
            save_evaluation(eval_ids, outputs, eval_labels, args.log_dir)
            correct = np.sum(np.argmax(outputs, axis=1) == np.array(eval_labels, np.int64))
            logger.info("Evaluation accuracy %d / %d = %.2f%%", correct, len(eval_labels), 100 * correct / len(eval_labels))

        logger.info("%s", time_logging.text_statistics())
        time_logging.clear()

if __name__ == '__main__':
    main()
