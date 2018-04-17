def lr_scheduler_exponential(optimizer, epoch, init_lr, epoch_start, base_factor, verbose=False, printfct=print):
    """
    Decay initial learning rate exponentially starting after epoch_start epochs
    The learning rate is multiplied with base_factor every lr_decay_epoch epochs
    :param optimizer: the optimizer inheriting from torch.optim.Optimizer
    :param epoch: the current epoch
    :param init_lr: initial learning rate before decaying
    :param epoch_start: epoch after which the learning rate is decayed exponentially. Constant lr before epoch_start
    :param base_factor: factor by which the learning rate is decayed in each epoch after epoch_start
    :param verbose: print current learning rate
    """
    if epoch <= epoch_start:
        lr = init_lr
    else:
        lr = init_lr * (base_factor**(epoch - epoch_start))
    if verbose:
        printfct('learning rate = {:6f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr


def lr_scheduler_step(optimizer, epoch, init_lr, decay_epochs, decay_factors, verbose=False, printfct=print):
    """
    Decay initial learning rate in steps at given points
    The learning rate is decreased by decay_factor[i] at decay_epochs[i]
    :param optimizer: the optimizer inheriting from torch.optim.Optimizer
    :param epoch: the current epoch
    :param init_lr: initial learning rate before decaying
    :param decay_epochs: list of epochs when the learning rate should be decayed
    :param decay_factors: float or list of factors by which the learning rate should be decayed at decay_epochs.
                                              when int, same factor at each decay step
                                              factors should be larger than 1
    :param verbose: print current learning rate
    """
    if type(decay_factors) == list:
        assert len(decay_factors) == len(decay_epochs)
    else:
        assert type(decay_factors) == int
        decay_factors = [decay_factors] * len(decay_epochs)  # same factor at each decay step
    lr = init_lr
    for e, f in zip(decay_epochs, decay_factors):
        if epoch >= e:
            lr /= f
        else:
            break
    if verbose:
        printfct('learning rate = {:6f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer, lr
