# pylint: disable=C,R
# takes m7.py m8.py m9.py
# takes (train_normal, val_normal), (train_perturbed, val_normal) ....

# train each N times with each (train,eval) pair
#
# output all data to make graphs

import os
from se3_cnn.train.train import train
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--number_of_epochs", type=int)
parser.add_argument("--eval_each", type=int, default=1)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--log_dir", type=str)
parser.add_argument("--model_path", type=str, nargs="+")
parser.add_argument("--repeat", type=int)

main_args = parser.parse_args()

os.mkdir(main_args.log_dir)

class Arguments:
    def __init__(self):
        self.number_of_epochs = None
        self.start_epoch = None
        self.eval_each = None
        self.gpu = None
        self.eval_data_path = None
        self.eval_csv_path = None
        self.restore_path = None
        self.model_path = None
        self.train_data_path = None
        self.train_csv_path = None
        self.log_dir = None

args = Arguments()

args.number_of_epochs = main_args.number_of_epochs
args.start_epoch = 0
args.eval_each = main_args.eval_each
args.gpu = main_args.gpu
args.train_csv_path = "voxels/train_6x32.csv"
args.eval_data_path = ["voxels/val_normal/*.npy", "voxels/val_perturbed/*.npy"]
args.eval_csv_path = ["voxels/val_6x194.csv"] * 2
args.restore_path = None

xxx = []
yyy = []
for model in main_args.model_path:
    args.model_path = model

    name = os.path.splitext(os.path.basename(model))[0]
    model_dir = os.path.join(main_args.log_dir, name)
    os.mkdir(model_dir)

    xx = []
    yy = []
    for i in range(main_args.repeat):
        x = []
        y = []
        for data_type in ["train_normal", "train_perturbed"]:
            args.train_data_path = "voxels/{}/*.npy".format(data_type)
            args.log_dir = os.path.join(model_dir, "{}_{}".format(data_type, i))
            train(args)

            x.append(np.load(os.path.join(args.log_dir, "statistics_train.npy")))
            y.append(np.load(os.path.join(args.log_dir, "statistics_eval.npy")))
        xx.append(x)
        yy.append(y)

    xx = np.array(xx) # [repeat, train_type,           row, column]
    yy = np.array(yy) # [repeat, train_type, val_type, row, column]

    np.save(os.path.join(model_dir, "statistics_train.npy"), xx)
    np.save(os.path.join(model_dir, "statistics_eval.npy"), yy)
    xxx.append(xx)
    yyy.append(yy)

xxx = np.array(xxx) # [model, repeat, train_type,           row, column]
yyy = np.array(yyy) # [model, repeat, train_type, val_type, row, column]

np.save(os.path.join(main_args.log_dir, "statistics_train.npy"), xxx)
np.save(os.path.join(main_args.log_dir, "statistics_eval.npy"), yyy)

eval_accuracies = np.mean(yyy, axis=1)[:, :, :, -1, 1] # [model, train_type, val_type]
eval_accuracies_std = np.std(yyy, axis=1)[:, :, :, -1, 1] # [model, train_type, val_type]

for i, model in enumerate(main_args.model_path):
    name = os.path.splitext(os.path.basename(model))[0]

    print(name)
    for j, data_type in enumerate(["train_normal", "train_perturbed"]):
        print(data_type)
        print("Mean Accuracy = {}".format(eval_accuracies[i][j]))
        print("Std Accuracy = {}".format(eval_accuracies_std[i][j]))
