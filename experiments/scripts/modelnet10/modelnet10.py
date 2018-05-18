# pylint: disable=C,R,E1101,W0622
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

import numpy as np
import os
import time
import importlib
from shutil import copyfile
import argparse

from experiments.datasets.modelnet.modelnet import *
from experiments.util import *



def train_loop(model, train_loader, optimizer, epoch):
    model.train()
    training_losses = []
    training_outs = []
    training_accs = []
    for batch_idx, (data, target) in enumerate(train_loader):
        time_start = time.perf_counter()
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        x = Variable(data)
        y = Variable(target)
        # forward and backward propagation
        out = model(x)
        losses = F.cross_entropy(out, y, reduce=False)
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

        log_obj.write("[{}:{}/{}] loss={:.4} acc={:.2} time={:.2}".format(
            epoch, batch_idx, len(train_loader),
            float(loss.data[0]), float(acc.data[0]),
            time.perf_counter() - time_start))

    loss_avg = np.mean(training_losses)
    acc_avg = np.mean(training_accs)
    training_outs = np.concatenate(training_outs)
    training_losses = np.concatenate(training_losses)
    return loss_avg, acc_avg, training_outs, training_losses



def infer(model, loader):
    model.eval()
    losses = []
    outs = []
    ys = []
    for batch_idx, (data,target) in enumerate(loader):
        print('inference on batch {}/{}'.format(batch_idx,len(loader)), end='\r')
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data, volatile=True)
        y = torch.autograd.Variable(target, volatile=True)
        out = model(x)
        outs.append(out.data.cpu().numpy())
        ys.append(y.data.cpu().numpy())
        losses.append(F.cross_entropy(out, y, reduce=False).data.cpu().numpy())
    outs = np.concatenate(outs)
    ys = np.concatenate(ys)
    return outs, ys, np.concatenate(losses)



def main(checkpoint):

    # load datasets
    data_set_kwargs = {'root_dir': args.dataset_root_dir,
                       'dataset': args.dataset,
                       'size': args.sample_vol_size}
    data_loader_kwargs = {'batch_size': args.batch_size,
                          'num_workers': 32,
                          'pin_memory': True}
    if args.mode == 'train_full':
        data_set_kwargs.update({'mode': 'train_full'})
        data_loader_kwargs.update({'shuffle': True,
                                   'drop_last': True})
        train_set, train_loader = get_modelnet_loader(**data_set_kwargs, data_loader_kwargs=data_loader_kwargs, args=args)
    if args.mode == 'train':
        data_set_kwargs.update({'mode': 'train'})
        data_loader_kwargs.update({'shuffle': True,
                                   'drop_last': True})
        train_set, train_loader = get_modelnet_loader(**data_set_kwargs, data_loader_kwargs=data_loader_kwargs, args=args)
    if args.mode in ['train', 'validation']:
        data_set_kwargs.update({'mode': 'validation'})
        data_loader_kwargs.update({'shuffle': False,
                                   'drop_last': False})
        validation_set, validation_loader = get_modelnet_loader(**data_set_kwargs, data_loader_kwargs=data_loader_kwargs, args=args)
    if args.mode == 'test':
        data_set_kwargs.update({'mode': 'test'})
        data_loader_kwargs.update({'shuffle': False,
                                   'drop_last': False})
        test_set, test_loader = get_modelnet_loader(**data_set_kwargs, data_loader_kwargs=data_loader_kwargs, args=args)


    # Build model and set up optimizer
    model = network_module.network(args=args)
    if use_gpu:
        model.cuda()
    log_obj.write(str(model))
    log_obj.write("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    param_groups = get_param_groups.get_param_groups(model, args)
    optimizer = optimizers_L1L2.Adam(param_groups, lr=args.initial_lr)
    optimizer.zero_grad()


    # restore state from checkpoint
    epoch_start_index = 0
    best_validation_loss_avg = float('inf')
    global timestamp
    if checkpoint is not None:
        log_obj.write("Restoring model from: " + checkpoint_path_restore)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start_index = checkpoint['epoch']+1
        best_validation_loss_avg = checkpoint['best_validation_loss_avg']


    # Set the logger
    tf_logger, tensorflow_available = tensorflow_logger.get_tf_logger(basepath=basepath, timestamp=timestamp)

    if args.mode == 'train':

        for epoch in range(epoch_start_index, args.training_epochs):

            # decay learning rate
            optimizer, _ = lr_schedulers.lr_scheduler_exponential(optimizer, epoch, args.initial_lr, args.lr_decay_start,
                                                                  args.lr_decay_base, verbose=True)

            loss_avg, acc_avg, training_outs, training_losses = train_loop(model, train_loader, optimizer, epoch)

            validation_outs, ys, validation_losses = infer(model, validation_loader)
            # # compute the accuracy
            validation_acc = np.sum(validation_outs.argmax(-1) == ys) / len(ys)
            validation_loss_avg = np.mean(validation_losses)

            log_obj.write('TRAINING SET [{}:{}/{}] loss={:.4} acc={:.2}'.format(
                epoch, len(train_loader)-1, len(train_loader),
                loss_avg, acc_avg))
            log_obj.write('VALIDATION SET [{}:{}/{}] loss={:.4} acc={:.2}'.format(
                epoch, len(train_loader)-1, len(train_loader),
                validation_loss_avg, validation_acc))
            # log_obj.write('VALIDATION losses: ' + str(validation_losses))

            # ============ TensorBoard logging ============ #
            if tensorflow_available:
                # (1) Log the scalar values
                info = {'training set avg loss': loss_avg,
                        'training set accuracy': acc_avg,
                        'validation set avg loss': validation_loss_avg,
                        'validation set accuracy': validation_acc
            }
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
                for i in range(10):
                    tf_logger.histo_summary("logits/%d/validation" % i, validation_outs[:, i], step=epoch+1)
                    tf_logger.histo_summary("logits/%d/training" % i,   training_outs[:, i],   step=epoch+1)



            # saving of latest state
            torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_validation_loss_avg': best_validation_loss_avg,
                        'timestamp': timestamp},
                        checkpoint_path_latest)




            # if validation_loss_avg < best_validation_loss_avg:
            #     best_validation_loss_avg = validation_loss_avg
            #     copyfile(src=checkpoint_path_latest, dst=checkpoint_path_best)
            #     log_obj.write('Best validation loss until now - updated best model')
            # else:
            #     log_obj.write('Validation loss did not improve')





    elif args.mode == 'validate':
        out, y, validation_loss_sum = infer(model, validation_loader)

        # compute the accuracy
        validation_acc = np.sum(out.argmax(-1) == y) / len(y)
        validation_loss_avg = validation_loss_sum / len(validation_loader.dataset)

        log_obj.write('VALIDATION SET: loss={:.4} acc={:.2}'.format(
            validation_loss_avg, validation_acc))

    elif args.mode == 'test':
        out, y, test_loss_sum = infer(model, test_loader)

        # compute the accuracy
        test_acc = np.sum(out.argmax(-1) == y) / len(y)
        test_loss_avg = test_loss_sum.mean() / len(test_loader.dataset)

        log_obj.write('TEST SET: loss={:.4} acc={:.2}'.format(
            test_loss_avg, test_acc))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model and dataset
    parser.add_argument("--model", required=True,
                        help="Which model definition to use")
    parser.add_argument("--dataset_root_dir", required=True,
                        help="Root directory of dataset")
    parser.add_argument("--dataset", choices=['ModelNet10', 'ModelNet40'], required=True,
                        help="Dataset to use")
    parser.add_argument("--sample_vol_size", required=True, type=int,
                        help="size of the voxelized volume")
    parser.add_argument("--mode", choices=['train', 'test', 'validate', 'train_full'], default="train",
                        help="Mode of operation (default: %(default)s)")
    # augmentation params
    parser.add_argument("--augment_affine", action='store_true', default=False, 
                        help="Switch to turn on full affine augmentation (scaling up to 1.1, flipping, translations and rotations)")
    parser.add_argument("--augment_scales", default=False, 
                        help="Augment by scaling (value > 1 required since only zooming out is allowed)")
    parser.add_argument("--augment_flip", action='store_true', default=False,
                        help="Augment by random flipping")
    parser.add_argument("--augment_translate", action='store_true', default=False,
                        help="Augment by random translation")
    parser.add_argument("--augment_rotate", action='store_true', default=False,
                        help="Augment by random rotation")
    parser.add_argument("--add_z_axis", action='store_true', default=False,
                        help="Augment by adding a z-axis to the data")
    # training params
    parser.add_argument("--training-epochs", default=100, type=int,
                        help="Which model definition to use")
    parser.add_argument("--batch-size", default=64, type=int,
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
    parser.add_argument("--normalization", choices={'batch', 'group', 'instance', None}, default='group',
                        help="Which nonlinearity to use for non-scalar capsules")
    parser.add_argument("--downsample-by-pooling", action='store_true', default=False,
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

    if args.augment_affine:
        args.augment_scales = 1.1
        args.augment_flip = True
        args.augment_translate = True
        args.augment_rotate = True

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

    checkpoint_path_latest = '{:s}/checkpoints/{:s}_latest.ckpt'.format(basepath, timestamp)
    checkpoint_path_best   = '{:s}/checkpoints/{:s}_best.ckpt'.format(basepath, timestamp)

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
