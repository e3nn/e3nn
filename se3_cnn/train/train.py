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

    labels = [labels[file.split("/")[-1].split(".")[0]] for file in files]
    if classes is None:
        classes = sorted(set(labels))
    labels = [classes.index(x) for x in labels]

    return files, labels, classes


def train_one_epoch(epoch, model, train_files, train_labels, optimizer, criterion):
    cnn = model.get_cnn()
    bs = model.get_batch_size()
    logger = logging.getLogger("trainer")

    losses = []
    accuracies = []

    cnn.train()
    cnn.cuda()

    for i in range(0, len(train_files), bs):
        j = min(i + bs, len(train_files))
        images = model.load_files(train_files[i:j])
        images = images.cuda()

        labels = train_labels[i:j]
        labels = torch.autograd.Variable(torch.LongTensor(labels).cuda())

        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss = float(loss.data.cpu().numpy())
        losses.append(loss)
        correct = sum(outputs.data.cpu().numpy().argmax(-1) == labels.data.cpu().numpy())
        accuracies.append(correct / bs)

        logger.info("[%d] Loss (Current,Mean) = %.2f, %.2f Training Accuracy (Current,Mean) = %d / %d, %.3f",
            epoch, loss, np.mean(losses), correct, bs, np.mean(accuracies))


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

    ############################################################################
    # Files and labels
    train_files, train_labels, classes = load_data(args.train_csv_path, args.train_data_path)
    eval_files, eval_labels, _ = load_data(args.eval_csv_path, args.eval_data_path, classes)

    ############################################################################
    # Optimizer
    optimizer = model.get_optimizer()
    criterion = model.get_criterion()
    criterion.cuda()

    for param_group in optimizer.param_groups:
        param_group['lr'] = model.get_learning_rate(args.start_epoch)

    for epoch in range(args.start_epoch, args.number_of_epochs):
        train_one_epoch(epoch, model, train_files, train_labels, optimizer, criterion)

        cnn.cpu()
        path = os.path.join(args.log_dir, 'model.pkl')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': cnn.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, path)
        logger.info("Saved in %s", path)

if __name__ == '__main__':
    main()
