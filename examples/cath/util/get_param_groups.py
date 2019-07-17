import torch.nn as nn

from se3cnn.blocks import GatedBlock
from se3cnn.blocks import NormBlock
from se3cnn import SE3BatchNorm
from se3cnn import SE3GroupNorm
from se3cnn import SE3Convolution
from se3cnn import SE3BNConvolution
from se3cnn import SE3GNConvolution

from se3cnn.non_linearities import NormRelu
from se3cnn.non_linearities import NormSoftplus
from se3cnn.non_linearities import ScalarActivation
from se3cnn.non_linearities import GatedActivation

# ONLY REGULATIZATION PARAMETERS REGISTERED HERE WILL BE CONSIDERED
lamb_dict = {
    'lamb_conv_weight_L1':          0, 'lamb_conv_weight_L2':          0,
    'lamb_conv_bias_L1':            0, 'lamb_conv_bias_L2':            0,
    'lamb_normalization_weight_L1': 0, 'lamb_normalization_weight_L2': 0,
    'lamb_normalization_bias_L1':   0, 'lamb_normalization_bias_L2':   0,
    'lamb_linear_weight_L1':        0, 'lamb_linear_weight_L2':        0,
    'lamb_linear_bias_L1':          0, 'lamb_linear_bias_L2':          0,
    'lamb_softmax_weight_L1':       0, 'lamb_softmax_weight_L2':       0,
    'lamb_softmax_bias_L1':         0, 'lamb_softmax_bias_L2':         0,
    'lamb_norm_activ_bias_L1':      0, 'lamb_norm_activ_bias_L2':      0,
}

def get_param_groups(model, args):
    """ split up parameters into groups, named_parameters() returns tupels ('name', parameter)
        each group gets its own regularization gain
    """
    # update lamb_dict with command line argument values
    for k in lamb_dict.keys():
        if k in vars(args).keys():
            lamb_dict[k] = vars(args).get(k)

    convLayers = [m for m in model.modules()
                  if isinstance(m, (SE3Convolution,
                                    # SE3BNConvolution, # TO BE INCLUDED SINCE CONVOLUTION DOES NOT USE SE3CONVOLUTION BUT DIRECTLY nn.functional.conv3d AND IS HENCE NOT COVERED
                                    # SE3GNConvolution, # NOT TO BE INCLUDED SINCE SE3CONVOLUTION IS USED
                                    nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d,
                                    nn.ConvTranspose2d, nn.ConvTranspose3d))]
    normActivs = [m for m in model.modules() if isinstance(m, (NormSoftplus, NormRelu, ScalarActivation))]
    normalizationLayers = [m for m in model.modules() if isinstance(m, (SE3BatchNorm, SE3GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    linearLayers = [m for m in model.modules() if isinstance(m, nn.Linear)]

    weights_conv = [p for m in convLayers for n, p in m.named_parameters() if n.endswith('weight')]
    biases_conv  = [p for m in convLayers for n, p in m.named_parameters() if n.endswith('bias')]
    weights_normalization = [p for m in normalizationLayers for n, p in m.named_parameters() if n.endswith('weight')]
    biases_normalization  = [p for m in normalizationLayers for n, p in m.named_parameters() if n.endswith('bias')]
    weights_fully = [p for m in linearLayers for n, p in m.named_parameters() if n.endswith('weight')]  # CROP OFF LAST WEIGHT !!!!! (classification layer)
    biases_fully  = [p for m in linearLayers for n, p in m.named_parameters() if n.endswith('bias')]  # CROP OFF LAST WEIGHT !!!!! (classification layer)
    (weights_fully, weights_softmax) = (weights_fully[:-1], [weights_fully[-1]]) if weights_fully != [] else ([], [])
    (biases_fully,  biases_softmax)  = (biases_fully[:-1],  [biases_fully[-1]])  if biases_fully  != [] else ([], [])
    biases_activs = [p for m in normActivs for n, p in m.named_parameters() if n.endswith('bias')]

    for np_tuple in model.named_parameters():
        if not np_tuple[0].endswith(('weight', 'weights_re', 'weights_im', 'bias')):
            raise Exception('named parameter encountered which is neither a weight nor a bias but `{:s}`'.format(np_tuple[0]))
    param_groups = [dict(params=weights_conv,          lamb_L1=lamb_dict['lamb_conv_weight_L1'],          lamb_L2=lamb_dict['lamb_conv_weight_L2']),
                    dict(params=biases_conv,           lamb_L1=lamb_dict['lamb_conv_bias_L1'],            lamb_L2=lamb_dict['lamb_conv_bias_L2']),
                    dict(params=weights_normalization, lamb_L1=lamb_dict['lamb_normalization_weight_L1'], lamb_L2=lamb_dict['lamb_normalization_weight_L2']),
                    dict(params=biases_normalization,  lamb_L1=lamb_dict['lamb_normalization_bias_L1'],   lamb_L2=lamb_dict['lamb_normalization_bias_L2']),
                    dict(params=weights_fully,         lamb_L1=lamb_dict['lamb_linear_weight_L1'],        lamb_L2=lamb_dict['lamb_linear_weight_L2']),
                    dict(params=biases_fully,          lamb_L1=lamb_dict['lamb_linear_bias_L1'],          lamb_L2=lamb_dict['lamb_linear_bias_L2']),
                    dict(params=weights_softmax,       lamb_L1=lamb_dict['lamb_softmax_weight_L1'],       lamb_L2=lamb_dict['lamb_softmax_weight_L2']),
                    dict(params=biases_softmax,        lamb_L1=lamb_dict['lamb_softmax_bias_L1'],         lamb_L2=lamb_dict['lamb_softmax_bias_L2']),
                    dict(params=biases_activs,         lamb_L1=lamb_dict['lamb_norm_activ_bias_L1'],      lamb_L2=lamb_dict['lamb_norm_activ_bias_L2'])]

    # Check whether all parameters are in groups
    params_in_groups = [id(param) for group in param_groups for param in group['params']]
    if len(list(params_in_groups)) != len(list(model.parameters())):
        error_msg = "Mismatch between number of total parameters and number of parameters in groups. "
        if len(list(params_in_groups)) < len(list(model.parameters())):
            error_msg += "The following parameters will not be optimized:\n"
            for name, param in model.named_parameters():
                if id(param) not in params_in_groups:
                    error_msg += "\t" + name + "\n"
        else:
            error_msg += "Total number in groups: {}. Total number of parameters: {}\n".format(
                len(list(params_in_groups)), len(list(model.parameters())))
            param_names = {id(param): name for name, param in
                           model.named_parameters()}
            counter = 0
            for group_idx, group in enumerate(param_groups):
                error_msg += "\t{}\n".format(group_idx)
                for param in group['params']:
                    counter += 1
                    error_msg += "\t\t{} {}\n".format(counter,
                                                      param_names[id(param)])
        # import ipdb; ipdb.set_trace()
        raise RuntimeError(error_msg)

    return param_groups


# old version, does not differentiate between parameter groups
# param_groups = [dict(params=model.parameters(), lamb_L1=lambda_L1,  lamb_L2=lambda_L2)]
# You can set different regularization for different parameter groups by splitting them up