# pylint: disable=not-callable, no-member, invalid-name, line-too-long
'''
Architecture to predict the structural categories of proteins according to the CATH
classification (www.cathdb.info).

'''
import torch
import torch.utils.data
import torch.nn as nn

import numpy as np
import os
import time
import importlib
from shutil import copyfile
import argparse

from experiments.datasets.cath.cath import Cath
from experiments.util import *

def train_loop(model, train_loader, optimizer, epoch):
    """Main training loop
    :param model: Model to be trained
    :param train_loader: DataLoader object for training set
    :param optimizer: Optimizer object
    :param epoch: Current epoch index
    """
    model.train()
    training_losses = []
    training_outs = []
    training_accs = []
    for batch_idx, (data, target) in enumerate(train_loader):
        time_start = time.perf_counter()

        target = torch.LongTensor(target)
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data)
        y = torch.autograd.Variable(target)
        # forward and backward propagation
        out = model(x)
        losses = nn.functional.cross_entropy(out, y, reduce=False)
        loss = losses.mean()
        loss.backward()
        if batch_idx % args.batchsize_multiplier == args.batchsize_multiplier-1:
            optimizer.step()
            optimizer.zero_grad()

        _, argmax = torch.max(out, 1)
        acc = (argmax.squeeze() == y).float().mean()

        training_losses.append(losses.data.cpu().numpy())
        training_outs.append(out.data.cpu().numpy())
        training_accs.append(acc.data[0])

        log_obj.write("[{}:{}/{}] loss={:.4} acc={:.3} time={:.2}".format(
            epoch, batch_idx, len(train_loader),
            float(loss.data[0]), float(acc.data[0]),
            time.perf_counter() - time_start))

    loss_avg = np.mean(training_losses)
    acc_avg = np.mean(training_accs)
    training_outs = np.concatenate(training_outs)
    training_losses = np.concatenate(training_losses)
    return loss_avg, acc_avg, training_outs, training_losses


def infer(model, loader):
    """Make prediction for all entries in loader
    :param model: Model used for prediction
    :param loader: DataLoader object
    """
    model.eval()
    losses = []
    outs = []
    ys = []
    for data,target in loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data, volatile=True)
        y = torch.autograd.Variable(target, volatile=True)
        out = model(x)
        outs.append(out.data.cpu().numpy())
        ys.append(y.data.cpu().numpy())
        losses.append(nn.functional.cross_entropy(out, y, reduce=False).data.cpu().numpy())
    outs = np.concatenate(outs)
    ys = np.concatenate(ys)
    return outs, ys, np.concatenate(losses)


def main(checkpoint):

    # Build datasets
    if args.mode == 'train':
        train_set = torch.utils.data.ConcatDataset([
            Cath(args.data_filename, split=i, download=True,
                 randomize_orientation=args.randomize_orientation,
                 discretization_bins=args.data_discretization_bins,
                 discretization_bin_size=args.data_discretization_bin_size) for i in range(7)])
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
        n_input = train_set.datasets[0].n_atom_types
        n_output = len(train_set.datasets[0].label_set)
        log_obj.write("Training set: " + str([len(dataset) for dataset in train_set.datasets]))

    if args.mode in ['train', 'validate']:
        validation_set = Cath(
            args.data_filename, split=7,
            discretization_bins=args.data_discretization_bins,
            discretization_bin_size=args.data_discretization_bin_size)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        n_input = validation_set.n_atom_types
        n_output = len(validation_set.label_set)
        log_obj.write("Validation set: " + str(len(validation_set)))

    if args.mode == 'test' or args.report_on_test_set:
        test_set = torch.utils.data.ConcatDataset([Cath(
            args.data_filename, split=i,
            discretization_bins=args.data_discretization_bins,
            discretization_bin_size=args.data_discretization_bin_size) for i in range(8, 10)])
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        n_input = test_set.datasets[0].n_atom_types
        n_output = len(test_set.datasets[0].label_set)
        log_obj.write("Test set: " + str([len(dataset) for dataset in test_set.datasets]))

    # Build model and set up optimizer
    model = network_module.network(n_input=n_input, n_output=n_output, args=args)
    if use_gpu:
        model.cuda()
    log_obj.write(str(model))
    log_obj.write("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    param_groups = get_param_groups.get_param_groups(model, args)
    optimizer = optimizers_L1L2.Adam(param_groups, lr=args.initial_lr)
    optimizer.zero_grad()


    # restore state from checkpoint
    epoch_start_index = 0
    best_validation_acc = -float('inf')
    best_avg_validation_acc = -float('inf')
    latest_validation_accs = []
    global timestamp
    if checkpoint is not None:
        log_obj.write("Restoring model from: " + checkpoint_path_restore)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start_index = checkpoint['epoch']+1
        best_validation_acc = checkpoint['best_validation_acc']
        best_avg_validation_acc = checkpoint['best_avg_validation_acc']
        latest_validation_accs = checkpoint['latest_validation_accs']

    # Set the logger
    tf_logger, tensorflow_available = tensorflow_logger.get_tf_logger(basepath=basepath, timestamp=timestamp)

    if args.mode == 'train':

        for epoch in range(epoch_start_index, args.training_epochs):

            # decay learning rate
            optimizer, _ = lr_schedulers.lr_scheduler_exponential(optimizer, epoch, args.initial_lr, args.lr_decay_start,
                                                                  args.lr_decay_base, verbose=True)

            loss_avg, acc_avg, training_outs, training_losses = train_loop(model, train_loader, optimizer, epoch)

            if (epoch+1) % args.report_frequency != 0:
                continue

            validation_outs, ys, validation_losses = infer(model, validation_loader)

            # compute the accuracy
            validation_acc = np.sum(validation_outs.argmax(-1) == ys) / len(ys)

            validation_loss_avg = np.mean(validation_losses)

            log_obj.write('TRAINING SET [{}:{}/{}] loss={:.4} acc={:.3}'.format(
                epoch, len(train_loader)-1, len(train_loader),
                loss_avg, acc_avg))
            log_obj.write('VALIDATION SET [{}:{}/{}] loss={:.4} acc={:.3}'.format(
                epoch, len(train_loader)-1, len(train_loader),
                validation_loss_avg, validation_acc))

            log_obj.write('VALIDATION losses: ' + str(validation_losses))

            if args.report_on_test_set:
                test_outs, test_ys, test_losses = infer(model,
                                                   test_loader)

                # compute the accuracy
                test_acc = np.sum(test_outs.argmax(-1) == test_ys) / len(test_ys)

                test_loss_avg = np.mean(test_losses)

                log_obj.write(
                    'TEST SET [{}:{}/{}] loss={:.4} acc={:.3}'.format(
                        epoch, len(train_loader) - 1, len(train_loader),
                        test_loss_avg, test_acc))

            # ============ TensorBoard logging ============ #
            if tensorflow_available:
                # (1) Log the scalar values
                info = {'training set avg loss': loss_avg,
                        'training set accuracy': acc_avg,
                        'validation set avg loss': validation_loss_avg,
                        'validation set accuracy': validation_acc}
                if args.report_on_test_set:
                    info.update({'test set avg loss': test_loss_avg,
                                 'test set accuracy': test_acc})
                for tag, value in info.items():
                    tf_logger.scalar_summary(tag, value, step=epoch+1)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    tf_logger.histo_summary(tag,         value.data.cpu().numpy(),      step=epoch+1)
                    tf_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step=epoch+1)

                # (3) Log losses for all datapoints in validation and training set
                tf_logger.histo_summary("losses/validation/", validation_losses, step=epoch+1)
                tf_logger.histo_summary("losses/training",    training_losses,   step=epoch+1)

                # (4) Log logits for all datapoints in validation and training set
                for i in range(n_output):
                    tf_logger.histo_summary("logits/%d/validation" % i, validation_outs[:, i], step=epoch+1)
                    tf_logger.histo_summary("logits/%d/training" % i,   training_outs[:, i],   step=epoch+1)


            # saving of latest state
            for n in range(0, checkpoint_latest_n-1)[::-1]:
                source = checkpoint_path_latest_n.replace('__n__', '_'+str(n))
                target = checkpoint_path_latest_n.replace('__n__', '_'+str(n+1))
                if os.path.exists(source):
                    os.rename(source, target)

            torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_validation_acc': best_validation_acc,
                        'best_avg_validation_acc': best_avg_validation_acc,
                        'latest_validation_accs': latest_validation_accs,
                        'timestamp': timestamp},
                        checkpoint_path_latest_n.replace('__n__', '_0'))

            latest_validation_accs.append(validation_acc)
            if len(latest_validation_accs) > checkpoint_latest_n:
                latest_validation_accs.pop(0)
            latest_validation_accs_avg = -float('inf')
            if len(latest_validation_accs) == checkpoint_latest_n:
                latest_validation_accs_avg = np.average(latest_validation_accs)

            # optional saving of best validation state
            improved = False
            if epoch > args.burnin_epochs:
                if validation_acc > best_validation_acc:
                    best_validation_acc = validation_acc
                    copyfile(src=checkpoint_path_latest_n.replace('__n__', '_0'), dst=checkpoint_path_best)
                    log_obj.write('Best validation accuracy until now - updated best model')
                    improved = True
                if latest_validation_accs_avg > best_avg_validation_acc:
                    best_avg_validation_acc = latest_validation_accs_avg
                    copyfile(src=checkpoint_path_latest_n.replace('__n__', '_'+str(checkpoint_latest_n//2)),
                             dst=checkpoint_path_best_window_avg)
                    log_obj.write('Best validation accuracy (window average)_until now - updated best (window averaged) model')
                    improved = True
                if not improved:
                    log_obj.write('Validation loss did not improve')


    elif args.mode == 'validate':
        out, y, validation_losses = infer(model, validation_loader)

        # compute the accuracy
        validation_acc = np.sum(out.argmax(-1) == y) / len(y)
        validation_loss_avg = np.mean(validation_losses)

        log_obj.write('VALIDATION SET: loss={:.4} acc={:.3}'.format(
            validation_loss_avg, validation_acc))

    elif args.mode == 'test':
        out, y, test_losses = infer(model, test_loader)

        # compute the accuracy
        test_acc = np.sum(out.argmax(-1) == y) / len(y)
        test_loss_avg = np.mean(test_losses)

        log_obj.write('TEST SET: loss={:.4} acc={:.3}'.format(
            test_loss_avg, test_acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--model", required=True,
                        help="Which model definition to use")
    parser.add_argument("--data-filename", required=True,
                        help="The name of the data file, e.g. cath_3class.npz, cath_10arch.npz (will automatically downloaded if not found)")
    parser.add_argument("--report-frequency", default=1, type=int,
                        help="The frequency with which status reports will be written")
    parser.add_argument("--report-on-test-set", action="store_true", default=False,
                        help="Whether to include accuracy on test set in output")
    parser.add_argument("--burnin-epochs", default=0, type=int,
                        help="Number of epochs to discard when dumping the best model")
    # cath specific
    parser.add_argument("--data-discretization-bins", type=int, default=50,
                        help="Number of bins used in each dimension for the discretization of the input data")
    parser.add_argument("--data-discretization-bin-size", type=float, default=2.0,
                        help="Size of bins used in each dimension for the discretization of the input data")

    parser.add_argument("--mode", choices=['train', 'test', 'validate'], default="train",
                        help="Mode of operation (default: %(default)s)")
    parser.add_argument("--training-epochs", default=100, type=int,
                        help="Which model definition to use")
    parser.add_argument("--randomize-orientation", action="store_true", default=False,
                        help="Whether to randomize the orientation of the structural input during training (default: %(default)s)")
    parser.add_argument("--batch-size", default=18, type=int,
                        help="Size of mini batches to use per iteration, can be accumulated via argument batchsize_multiplier (default: %(default)s)")
    parser.add_argument("--batchsize-multiplier", default=1, type=int,
                        help="number of minibatch iterations accumulated before applying the update step, effectively multiplying batchsize (default: %(default)s)")
    parser.add_argument("--restore-checkpoint-filename", type=str, default=None,
                        help="Read model from checkpoint given by filename (assumed to be in checkpoint folder)")
    parser.add_argument("--initial_lr", default=1e-1, type=float,
                        help="Initial learning rate (without decay)")
    parser.add_argument("--lr_decay_start", type=int, default=40,
                        help="epoch after which the exponential learning rate decay starts")
    parser.add_argument("--lr_decay_base", type=float, default=.94,
                        help="exponential decay factor per epoch")
    # model
    parser.add_argument("--kernel-size", type=int, default=3,
                        help="convolution kernel size")
    parser.add_argument("--p-drop-conv", type=float, default=None,
                        help="convolution/capsule dropout probability")
    parser.add_argument("--p-drop-fully", type=float, default=None,
                        help="fully connected layer dropout probability")
    parser.add_argument("--bandlimit-mode", choices={"conservative", "compromise", "sfcnn"}, default="compromise",
                        help="bandlimiting heuristic for spherical harmonics")
    parser.add_argument("--SE3-nonlinearity", choices={"gated", "norm"}, default="gated",
                        help="Which nonlinearity to use for non-scalar capsules")
    parser.add_argument("--normalization", choices={'batch', 'group', 'instance', None}, default='batch',
                        help="Which nonlinearity to use for non-scalar capsules")
    parser.add_argument("--downsample-by-pooling", action='store_true', default=True,
                        help="Switches from downsampling by striding to downsampling by pooling")
    # WEIGHTS
    parser.add_argument("--lamb_conv_weight_L1", default=0, type=float,
                        help="L1 regularization factor for convolution weights")
    parser.add_argument("--lamb_conv_weight_L2", default=0, type=float,
                        help="L2 regularization factor for convolution weights")
    parser.add_argument("--lamb_bn_weight_L1", default=0, type=float,
                        help="L1 regularization factor for batchnorm weights")
    parser.add_argument("--lamb_bn_weight_L2", default=0, type=float,
                        help="L2 regularization factor for batchnorm weights")
    parser.add_argument("--lamb_linear_weight_L1", default=0, type=float,
                        help="L1 regularization factor for fully connected layer weights (except last / classification layer)")
    parser.add_argument("--lamb_linear_weight_L2", default=0, type=float,
                        help="L2 regularization factor for fully connected layer weights (except last / classification layer)")
    parser.add_argument("--lamb_softmax_weight_L1", default=0, type=float,
                        help="L1 regularization factor for classification layer weights")
    parser.add_argument("--lamb_softmax_weight_L2", default=0, type=float,
                        help="L2 regularization factor for classification layer weights")
    # BIASES
    parser.add_argument("--lamb_conv_bias_L1", default=0, type=float,
                        help="L1 regularization factor for convolution biases")
    parser.add_argument("--lamb_conv_bias_L2", default=0, type=float,
                        help="L2 regularization factor for convolution biases")
    parser.add_argument("--lamb_norm_activ_bias_L1", default=0, type=float,
                        help="L1 regularization factor for norm activation biases")
    parser.add_argument("-lamb_norm_activ_bias_L2", default=0, type=float,
                        help="L2 regularization factor for norm activation biases")
    parser.add_argument("--lamb_bn_bias_L1", default=0, type=float,
                        help="L1 regularization factor for batchnorm biases")
    parser.add_argument("--lamb_bn_bias_L2", default=0, type=float,
                        help="L2 regularization factor for batchnorm biases")
    parser.add_argument("--lamb_linear_bias_L1", default=0, type=float,
                        help="L1 regularization factor for fully connected layer biases (except last / classification layer)")
    parser.add_argument("--lamb_linear_bias_L2", default=0, type=float,
                        help="L2 regularization factor for fully connected layer biases (except last / classification layer)")
    parser.add_argument("--lamb_softmax_bias_L1", default=0, type=float,
                        help="L1 regularization factor for classification layer biases")
    parser.add_argument("--lamb_softmax_bias_L2", default=0, type=float,
                        help="L2 regularization factor for classification layer biases")

    args, unparsed = parser.parse_known_args()

    if len(unparsed) != 0:
        print('\n{:d} unparsed (unknown arguments):'.format(len(unparsed)))
        for u in unparsed:
            print('  ', u)
        print()
        raise ValueError('unparsed / unknown arguments')

    network_module = importlib.import_module('networks.{:s}.{:s}'.format(args.model, args.model))

    basepath = 'networks/{:s}'.format(args.model)

    # load checkpoint
    if args.restore_checkpoint_filename is not None:
        checkpoint_path_restore = '{:s}/checkpoints/{:s}'.format(basepath, args.restore_checkpoint_filename)
        checkpoint = torch.load(checkpoint_path_restore)
        timestamp = checkpoint['timestamp']
    else:
        checkpoint = None
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        os.makedirs('{:s}/checkpoints'.format(basepath), exist_ok=True)

    checkpoint_latest_n = 5
    checkpoint_path_latest_n = '{:s}/checkpoints/{:s}_latest__n__.ckpt'.format(basepath, timestamp)
    checkpoint_path_best   = '{:s}/checkpoints/{:s}_best.ckpt'.format(basepath, timestamp)
    checkpoint_path_best_window_avg   = '{:s}/checkpoints/{:s}_best_window_avg.ckpt'.format(basepath, timestamp)

    # instantiate simple logger
    log_obj = logger.logger(basepath=basepath, timestamp=timestamp)
    if checkpoint != None:
        log_obj.write('\n' + 42*'=' + '\n')
        log_obj.write('\n model restored from checkpoint\n')
    log_obj.write('basepath = {:s}'.format(basepath))
    log_obj.write('timestamp = {:s}'.format(timestamp))
    log_obj.write('\n# Options')
    for key, value in sorted(vars(args).items()):
        log_obj.write('\t'+str(key)+'\t'+str(value))

    torch.backends.cudnn.benchmark = True
    use_gpu = torch.cuda.is_available()

    main(checkpoint)