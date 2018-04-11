# pylint: disable=C,R,E1101
'''
training/evaluation script for MRI image segmentation.

'''
import torch
import torch.utils.data

import numpy as np
import os
import time
import importlib
from shutil import copyfile
import argparse
from functools import partial

from experiments.datasets.MRI.mri import MRISegmentation
from experiments.util import *


def train_loop(model, train_loader, loss_function, optimizer, epoch):
    model.train()
    train_losses = []
    train_dice_accs = []
    for batch_idx, (data, target, img_index, patch_index, valid) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data)
        y = torch.autograd.Variable(target)

        out = model(x)

        time_start = time.perf_counter()
        loss = loss_function(out, y)
        loss.backward()
        if batch_idx % args.batchsize_multiplier == args.batchsize_multiplier-1:
            optimizer.step()
            optimizer.zero_grad()

        loss_value = float(loss.data[0])
        binary_dice_acc = losses.dice_coefficient_orig_binary(out, y, y_pred_is_dist=True).data[0]
        train_losses.append(loss_value)
        train_dice_accs.append(binary_dice_acc)

        log_obj.write("[{}:{:3}/{:3}] loss={:.4} dice_acc={:.4} time={:.2}".format(
            epoch, batch_idx, len(train_loader),
            loss_value, binary_dice_acc,
            time.perf_counter() - time_start))

        # for debugging to arrive at validation early...
        if (batch_idx+1)%4 == 0:
            break

    return np.mean(train_losses), np.mean(train_dice_accs)


def infer(model, loader, loss_function):
    model.eval()
    losses_numerator = []
    losses_denominator = []
    out_images = []
    for i in range(len(loader.dataset.unpadded_data_shape)):
        out_images.append(np.full(loader.dataset.unpadded_data_shape[i], -1))
    for i, (data, target, img_index, patch_index, valid) in enumerate(loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        x = torch.autograd.Variable(data, volatile=True)
        y = torch.autograd.Variable(target, volatile=True)
        out = model(x)
        _, out_predict = torch.max(out, dim=1)
        mask = get_mask.get_mask(out_predict.shape, valid)
        patch_index = patch_index.cpu().numpy()
        for j in range(out.size(0)):
            out_predict_masked = out_predict[j][mask[j]]
            patch_start = patch_index[j,0] + valid[j,0]
            patch_end = patch_start + (valid[j,1]-valid[j,0])
            if (patch_end-patch_start > 0).all():
                out_images[img_index[j]][patch_start[0]:patch_end[0],
                                         patch_start[1]:patch_end[1],
                                         patch_start[2]:patch_end[2]] = out_predict_masked.view((valid[j,1] - valid[j,0]).tolist()).data.cpu().numpy()
        numerator, denominator = loss_function(out, y, valid=valid, reduce=False)
        del out, out_predict
        try:
            numerator = numerator.data
            denominator = denominator.data
        except:
            pass
        losses_numerator.append(numerator.cpu().numpy())
        losses_denominator.append(denominator.cpu().numpy())
        # print(np.mean(np.sum(losses_numerator[-1], axis=0)/np.sum(losses_denominator[-1], axis=0)), loss_function(out, y, valid).data.cpu().numpy())
        # loss_function = lambda *x: cross_entropy_loss(*x, class_weight=class_weight)
    # Check that entire image was filled in
    for out_image in out_images:
        assert not (out_image == -1).any()
    losses_numerator = np.concatenate(losses_numerator)
    losses_denominator = np.concatenate(losses_denominator)
    loss = np.mean(np.sum(losses_numerator, axis=0) / np.sum(losses_denominator, axis=0))
    return out_images, loss


def calc_binary_dice_score(dataset, ys):
    # Calculate binary dice score on predicted images
    numerators = []
    denominators = []
    for i in range(len(dataset.data)):
        y_true = torch.LongTensor(dataset.get_original(i)[1])
        y_pred = torch.LongTensor(ys[i])
        if use_gpu:
            y_true = y_true.cuda()
            y_pred = y_pred.cuda()
        numerator, denominator = losses.dice_coefficient_orig_binary(
            y_pred.unsqueeze(0),
            y_true.unsqueeze(0),
            classes=output_size,
            reduce=False)
        numerators.append(numerator)
        denominators.append(denominator)
    numerators = torch.cat(numerators)
    denominators = torch.cat(denominators)
    binary_dice_acc = torch.mean(torch.sum(numerators, dim=0)/(torch.sum(denominators, dim=0)))
    return binary_dice_acc



def main(checkpoint):

    h5_filename = '../../datasets/MRI/MICCAI2012/miccai12.h5'
    if args.mode == 'train':
        train_set = MRISegmentation(h5_filename=h5_filename,
                                    patch_shape=args.patch_size,
                                    filter=train_filter)
                                    # log10_signal=args.log10_signal)
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=8,
                                                   pin_memory=False,
                                                   drop_last=True)
        np.set_printoptions(threshold=np.nan)
        print(np.unique(train_set.labels[0]))
    if args.mode in ['train', 'validate']:
        validation_set = MRISegmentation(h5_filename=h5_filename,
                                         patch_shape=args.patch_size,
                                         filter=validation_filter,
                                         randomize_patch_offsets=False)
                                         # log10_signal=args.log10_signal)
        validation_loader = torch.utils.data.DataLoader(validation_set,
                                                        batch_size=args.batch_size,
                                                        shuffle=False,
                                                        num_workers=8,
                                                        pin_memory=False,
                                                        drop_last=False)
    if args.mode == 'test':
        test_set = MRISegmentation(h5_filename=h5_filename,
                                   patch_shape=args.patch_size,
                                   filter=test_filter,
                                   randomize_patch_offsets=False)
                                   # log10_signal=args.log10_signal)
        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  pin_memory=False,
                                                  drop_last=False)


    model = network_module.network(output_size=output_size)
    if use_gpu:
        model.cuda()

    log_obj.write("The model contains {} parameters".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)))


    param_groups = get_param_groups.get_param_groups(model, args)
    optimizer = optimizers_L1L2.Adam(param_groups, lr=args.initial_lr)
    optimizer.zero_grad()

    loss_function = None
    if args.loss == "dice":
        loss_function = losses.dice_coefficient_loss
    elif args.loss == "cross_entropy":
        if args.class_weighting:
            class_weight = torch.Tensor(1/train_set.class_count)
            class_weight *= np.sum(train_set.class_count)/len(train_set.class_count)
            if use_gpu:
                class_weight = class_weight.cuda()
        else:
            class_weight = None
        loss_function = partial(losses.cross_entropy_loss, class_weight=class_weight)

    # restore state from checkpoint
    epoch_start_index = 0
    best_validation_loss = float('inf')
    global timestamp
    if checkpoint is not None:
        log_obj.write("Restoring model from: " + checkpoint_path_restore)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch_start_index = checkpoint['epoch']+1
        best_validation_loss = checkpoint['best_validation_loss']

    tf_logger, tensorflow_available = tensorflow_logger.get_tf_logger(basepath=basepath, timestamp=timestamp)

    if args.mode == 'train':
        for epoch in range(epoch_start_index, args.training_epochs):
            optimizer, _ = lr_schedulers.lr_scheduler_exponential(optimizer, epoch, args.initial_lr,
                                                                  args.lr_decay_start, args.lr_decay_base, verbose=True)

            training_loss, training_binary_dice_acc = train_loop(model, train_loader, loss_function, optimizer, epoch)

            validation_ys, validation_loss = infer(model, validation_loader, loss_function)
            validation_binary_dice_acc = calc_binary_dice_score(validation_set, validation_ys)

            log_obj.write('VALIDATION SET [{}:{}/{}] loss={:.4} dice_acc={:.2}'.format(
                                        epoch, len(train_loader)-1, len(train_loader),
                                        validation_loss, validation_binary_dice_acc))

            # ============ TensorBoard logging ============ #
            if tensorflow_available:
                # (1) Log the scalar values
                info = {'training set loss': training_loss,
                        'training set binary dice accuracy': training_binary_dice_acc,
                        'validation set loss': validation_loss,
                        'validation set binary dice accuracy': validation_binary_dice_acc}
                for tag, value in info.items():
                    tf_logger.scalar_summary(tag, value, step=epoch+1)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    tf_logger.histo_summary(tag,         value.data.cpu().numpy(),      step=epoch+1)
                    tf_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), step=epoch+1)

                # # (3) Log losses for all datapoints in validation and training set
                # tf_logger.histo_summary("losses/validation/", validation_losses, step=epoch+1)
                # tf_logger.histo_summary("losses/training",    training_losses,   step=epoch+1)

                # # (4) Log logits for all datapoints in validation and training set
                # for i in range(n_output):
                #     tf_logger.histo_summary("logits/%d/validation" % i, validation_outs[:, i], step=epoch+1)
                #     tf_logger.histo_summary("logits/%d/training" % i,   training_outs[:, i],   step=epoch+1)

            # saving of latest state
            torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_validation_loss': best_validation_loss,
                        'timestamp': timestamp},
                        checkpoint_path_latest)
            # optional saving of best validation state
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                copyfile(src=checkpoint_path_latest, dst=checkpoint_path_best)
                log_obj.write('Best validation loss until now - updated best model')

            # Adjust patch indices at end of each epoch
            train_set.initialize_patch_indices()

    elif args.mode == 'validate':
        raise NotImplementedError('validation mode')

    elif args.mode == 'test':
        raise NotImplementedError('test mode')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--model", required=True,
                        help="Which model definition to use")
    # MRI specific
    parser.add_argument("--patch-size", default=64, type=int,
                        help="Size of patches (default: %(default)s)")
    parser.add_argument("--loss", choices=['dice', 'dice_onehot', 'cross_entropy'],
                        default="cross_entropy",
                        help="Which loss function to use(default: %(default)s)")
    parser.add_argument("--class-weighting", action='store_true', default=False,
                        help="switches on class weighting, only used in cross entropy loss (default: %(default)s)")

    parser.add_argument("--mode", choices=['train', 'test', 'validate'],
                        default="train",
                        help="Mode of operation (default: %(default)s)")
    parser.add_argument("--training-epochs", default=100, type=int,
                        help="Which model definition to use")
    parser.add_argument("--randomize-orientation", action="store_true", default=False,
                        help="Whether to randomize the orientation of the structural input during training (default: %(default)s)")
    parser.add_argument("--batch-size", default=2, type=int,
                        help="Size of mini batches to use per iteration, can be accumulated via argument batchsize_multiplier (default: %(default)s)")
    parser.add_argument("--batchsize-multiplier", default=1, type=int,
                        help="number of minibatch iterations accumulated before applying the update step, effectively multiplying batchsize (default: %(default)s)")
    parser.add_argument("--restore-checkpoint-filename", type=str, default=None,
                        help="Read model from checkpoint given by filename (assumed to be in checkpoint folder)")
    parser.add_argument("--initial_lr", default=1e-2, type=float,
                        help="Initial learning rate (without decay)")
    parser.add_argument("--lr_decay_start", type=int, default=1,
                        help="epoch after which the exponential learning rate decay starts")
    parser.add_argument("--lr_decay_base", type=float, default=1,
                        help="exponential decay factor per epoch")
    # WEIGHTS
    parser.add_argument("--lamb_conv_weight_L1", default=0, type=float,
                        help="L1 regularization factor for convolution weights")
    parser.add_argument("--lamb_conv_weight_L2", default=0, type=float,
                        help="L2 regularization factor for convolution weights")
    parser.add_argument("--lamb_normalization_weight_L1", default=0, type=float,
                        help="L1 regularization factor for normalization layer weights")
    parser.add_argument("--lamb_normalization_weight_L2", default=0, type=float,
                        help="L2 regularization factor for normalization weights")
    # BIASES
    parser.add_argument("--lamb_conv_bias_L1", default=0, type=float,
                        help="L1 regularization factor for convolution biases")
    parser.add_argument("--lamb_conv_bias_L2", default=0, type=float,
                        help="L2 regularization factor for convolution biases")
    parser.add_argument("--lamb_norm_activ_bias_L1", default=0, type=float,
                        help="L1 regularization factor for norm activation biases")
    parser.add_argument("-lamb_norm_activ_bias_L2", default=0, type=float,
                        help="L2 regularization factor for norm activation biases")
    parser.add_argument("--lamb_normalization_bias_L1", default=0, type=float,
                        help="L1 regularization factor for normalization biases")
    parser.add_argument("--lamb_normalization_bias_L2", default=0, type=float,
                        help="L2 regularization factor for normalization biases")

    args, unparsed = parser.parse_known_args()

    if len(unparsed) != 0:
        print('\n{:d} unparsed (unknown arguments):'.format(len(unparsed)))
        for u in unparsed:
            print('  ', u)
        print()
        raise ValueError('unparsed / unknown arguments')

    network_module = importlib.import_module('networks.MICCAI2012.{:s}.{:s}'.format(args.model, args.model))

    basepath = 'networks/MICCAI2012/{:s}'.format(args.model)

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

    output_size = 135

    train_filter = ["1000_3",
                    # "1001_3",
                    # "1002_3",
                    # "1006_3",
                    # "1007_3",
                    # "1008_3",
                    # "1009_3",
                    # "1010_3",
                    # "1011_3",
                    # "1012_3",
                    # "1013_3",
                    "1014_3"
                    ]
    validation_filter = ["1015_3",
                         # "1017_3",
                         "1036_3"
                         ]
    test_filter = ["1003_3",
                   "1004_3",
                   "1005_3",
                   "1018_3",
                   "1019_3",
                   "1023_3",
                   "1024_3",
                   "1025_3",
                   "1038_3",
                   "1039_3",
                   "1101_3",
                   "1104_3",
                   "1107_3",
                   "1110_3",
                   "1113_3",
                   "1116_3",
                   "1119_3",
                   "1122_3",
                   "1125_3",
                   "1128_3"]

    # Check that sets are non-overlapping
    assert len(set(validation_filter).intersection(train_filter)) == 0
    assert len(set(test_filter).intersection(train_filter)) == 0

    main(checkpoint)