"""
This script loads the MRBrainS13 dataset stored in nifty format (.nii) exports it to hdf5 files.
Preprocessing includes:
- resempling
- cropping of background
- normalization of the signal
"""
import os
import glob
import sys
import numpy as np
import h5py
import nibabel as nib
nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6
import scipy.ndimage


def resample_volume(volume, order):
    """resample anisotropic data of resolution (0.958mm x 0.958mm x 3.0mm) by a factor of 3 over the z-axis"""
    return scipy.ndimage.zoom(volume, zoom=(1,1,3), order=order)


def crop_background(signal_volumes, label_volumes, border=16, verbose=True):
    """Crop out the cube in the signal and label volume which contain only background
       We define a region as background where the label is zero
       Since the skull, skin and face are labeled as background we add a border
    """
    bg_mask = label_volumes[0] == 0
    # generate 1d arrays over axes which are false iff only bg is found in the corresponding slice
    only_bg_x = 1-np.all(bg_mask, axis=(1,2))
    only_bg_y = 1-np.all(bg_mask, axis=(0,2))
    only_bg_z = 1-np.all(bg_mask, axis=(0,1))
    # get start and stop index of non bg mri volume
    x_start = np.argmax(only_bg_x)
    x_stop  = np.argmax(1 - only_bg_x[x_start:]) + x_start
    x_stop  = x_stop if x_start!=x_stop else len(only_bg_x)
    x_start = max(0, x_start-border)
    x_stop  = min(len(only_bg_x), x_stop+border)
    y_start = np.argmax(only_bg_y)
    y_stop  = np.argmax(1 - only_bg_y[y_start:]) + y_start
    y_stop  = y_stop if y_start!=y_stop else len(only_bg_y)
    y_start = max(0, y_start-border)
    y_stop  = min(len(only_bg_y), y_stop+border)
    z_start = np.argmax(only_bg_z)
    z_stop  = np.argmax(1 - only_bg_z[z_start:]) + z_start
    z_stop  = z_stop if z_start!=z_stop else len(only_bg_z)
    z_start = max(0, z_start-border)
    z_stop  = min(len(only_bg_z), z_stop+border)
    if verbose:
        print('cropped x ({} - {}), of len {} ({}%)'.format(x_start, x_stop, len(only_bg_x), 100*(x_stop-x_start)/len(only_bg_x)))
        print('cropped y ({} - {}), of len {} ({}%)'.format(y_start, y_stop, len(only_bg_y), 100*(y_stop-y_start)/len(only_bg_y)))
        print('cropped z ({} - {}), of len {} ({}%)'.format(z_start, z_stop, len(only_bg_z), 100*(z_stop-z_start)/len(only_bg_z)))
        print('volume fraction left = {}%'.format(100*(x_stop-x_start)*(y_stop-y_start)*(z_stop-z_start)/np.prod(signal_volumes[0].shape)))
    # crop out non bg signal
    signal_volumes_cropped = [sv[x_start:x_stop, y_start:y_stop, z_start:z_stop] for sv in signal_volumes]
    label_volumes_cropped  = [lv[x_start:x_stop, y_start:y_stop, z_start:z_stop] for lv in label_volumes]
    return signal_volumes_cropped, label_volumes_cropped


def normalize_signals(signals_train, signals_test):
    signals = signals_train+signals_test
    T1       = np.concatenate([s[0].flatten() for s in signals])
    T1_IR    = np.concatenate([s[1].flatten() for s in signals])
    T2_FLAIR = np.concatenate([s[2].flatten() for s in signals])
    T1_mean,       T1_std       = T1.mean(),       T1.std()
    T1_IR_mean,    T1_IR_std    = T1_IR.mean(),    T1_IR.std()
    T2_FLAIR_mean, T2_FLAIR_std = T2_FLAIR.mean(), T2_FLAIR.std()
    for i in range(len(signals_train)):
        signals_train[i][0] = (signals_train[i][0] - T1_mean)/T1_std
        signals_train[i][1] = (signals_train[i][1] - T1_IR_mean)/T1_IR_std
        signals_train[i][2] = (signals_train[i][2] - T2_FLAIR_mean)/T2_FLAIR_std
    for i in range(len(signals_test)):
        signals_test[i][0] = (signals_test[i][0] - T1_mean)/T1_std
        signals_test[i][1] = (signals_test[i][1] - T1_IR_mean)/T1_IR_std
        signals_test[i][2] = (signals_test[i][2] - T2_FLAIR_mean)/T2_FLAIR_std
    channel_means = np.array([T1_mean, T1_IR_mean, T2_FLAIR_mean])
    channel_stds  = np.array([T1_std , T1_IR_std , T2_FLAIR_std ])
    return signals_train, signals_test, channel_means, channel_stds


def compute_class_counts(labels):
    labels_full    = np.concatenate([l[0].flatten() for l in labels])
    labels_reduced = np.concatenate([l[1].flatten() for l in labels])
    class_counts_full    = np.array([np.sum(labels_full==label)    for label in np.arange(9)])
    class_counts_reduced = np.array([np.sum(labels_reduced==label) for label in np.arange(4)])
    return class_counts_full, class_counts_reduced


mri_dirs_train = ['MRBrainS13DataNii/TrainingData/{}/'.format(i) for i in np.arange(1,6)]
mri_dirs_test  = ['MRBrainS13DataNii/TestData/{}/'.format(i) for i in np.arange(1,16)]
volume_names = ['T1.nii', 'T1_IR.nii', 'T2_FLAIR.nii'] #, 'T1_1mm.nii']
label_names  = ['LabelsForTraining.nii', 'LabelsForTesting.nii']

signals_train = []
labels_train = []
signals_test = []
print('load and resample data...')
for dir in mri_dirs_train:
    print('\t', dir)
    signals_train.append([])
    labels_train.append([])
    for vol_name in volume_names:
        data = nib.load(dir+vol_name).get_data()
        data = resample_volume(data, order=3)
        signals_train[-1].append(data)
    for lab_name in label_names:
        data = nib.load(dir+lab_name).get_data()
        data = resample_volume(data, order=1)
        labels_train[-1].append(data)
for dir in mri_dirs_test:
    print('\t', dir)
    signals_test.append([])
    for vol_name in volume_names:
        data = nib.load(dir+vol_name).get_data()
        data = resample_volume(data, order=3)
        signals_test[-1].append(data)

# bg_estimates = np.array([np.array(signals_test)[:,ch,0,:,:].mean() for i in range(3)])

# cropping after resampling is a bit slower but allows for the same border in each axis
print('crop background in training set...')
for i,(s,l) in enumerate(zip(signals_train, labels_train)):
    signals_train[i], labels_train[i] = crop_background(s,l)

print('normalize signals...')
signals_train, signals_test, channel_means, channel_stds = normalize_signals(signals_train, signals_test)

print('compute class counts...')
class_counts_full, class_counts_reduced = compute_class_counts(labels_train)

print('save datasets...')
with h5py.File('MRBrainS13.h5', 'w') as hf:
    for i,(signals,labels) in enumerate(zip(signals_train, labels_train)):
        # stack together signals to signal channels
        signals_stacked = np.stack(signals, axis=0)
        hf.create_dataset('train_signal_{}'.format(i),        data=signals_stacked, compression="gzip")
        hf.create_dataset('train_label_full_{}'.format(i),    data=labels[0],       compression="gzip")
        hf.create_dataset('train_label_reduced_{}'.format(i), data=labels[1],       compression="gzip")
    for i,signals in enumerate(signals_test):
        # stack together signals to signal channels
        signals_stacked = np.stack(signals, axis=0)
        hf.create_dataset('test_signal_{}'.format(i), data=signals_stacked, compression="gzip")
    hf.create_dataset('class_counts_full',    data=class_counts_full,    compression='gzip')
    hf.create_dataset('class_counts_reduced', data=class_counts_reduced, compression='gzip')
    hf.create_dataset('channel_means', data=channel_means, compression='gzip')
    hf.create_dataset('channel_stds',  data=channel_stds,  compression='gzip')




# import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
# vmin = signals_train[0][0].min()
# vmax = signals_train[0][0].max()
# for i,s in enumerate(signals_train[0][0]):
#     plt.imsave('{}.png'.format(i), s, vmin=vmin, vmax=vmax) #cmap='gray')
# import ipdb; ipdb.set_trace()





# # DEPRECATED: SIGNAL VOLUME NOISE NEVER DROPS TO ZERO IN ALL AXES
# def crop_background(signal_volumes, label_volumes, verbose=True):
#     """Crop out the cube in the signal and label volume which contain only background
#        We define a region as background when the signals T1.nii and T2_FLAIR.nii and the labels are are all zero
#        The signal T1_IR.nii is noisy, we don't check it here
#     """
#     # DEPRECATED: SIGNAL VOLUME NOISE NEVER DROPS TO ZERO IN ALL AXES
#     all_stacked = np.stack(signal_volumes[::2] + label_volumes, axis=-1) # [::2] throws out T1_IR.nii
#     bg_mask = np.all(all_stacked==0, axis=-1) # contracts over channels, gives boolean voxels which are True iff all signals/labels are bg==0
#     # generate 1d arrays over axes which are false iff only bg is found in the corresponding slice
#     only_bg_x = 1-np.all(bg_mask, axis=(1,2))
#     only_bg_y = 1-np.all(bg_mask, axis=(0,2))
#     only_bg_z = 1-np.all(bg_mask, axis=(0,1))
#     # get start and stop index of non bg mri volume
#     x_start = np.argmax(only_bg_x)
#     x_stop  = np.argmax(1 - only_bg_x[x_start:]) + x_start
#     x_stop  = x_stop if x_start!=x_stop else len(only_bg_x)
#     y_start = np.argmax(only_bg_y)
#     y_stop  = np.argmax(1 - only_bg_y[y_start:]) + y_start
#     y_stop  = y_stop if y_start!=y_stop else len(only_bg_y)
#     z_start = np.argmax(only_bg_z)
#     z_stop  = np.argmax(1 - only_bg_z[z_start:]) + z_start
#     z_stop  = z_stop if z_start!=z_stop else len(only_bg_z)
#     if verbose:
#         print('cropped x ({} - {}), of len {} ({}%)'.format(x_start, x_stop, len(only_bg_x), 100*(x_stop-x_start)/len(only_bg_x)))
#         print('cropped y ({} - {}), of len {} ({}%)'.format(y_start, y_stop, len(only_bg_y), 100*(y_stop-y_start)/len(only_bg_y)))
#         print('cropped z ({} - {}), of len {} ({}%)'.format(z_start, z_stop, len(only_bg_z), 100*(z_stop-z_start)/len(only_bg_z)))
#         print('volume fraction left = {}%'.format(100*(x_stop-x_start)*(y_stop-y_start)*(z_stop-z_start)/np.prod(signal_volumes[0].shape)))
#     # crop out non bg signal
#     signal_volumes_cropped = [sv[x_start:x_stop, y_start:y_stop, z_start:z_stop] for sv in signal_volumes]
#     label_volumes_cropped  = [lv[x_start:x_stop, y_start:y_stop, z_start:z_stop] for lv in label_volumes]
#     return signal_volumes_cropped, label_volumes_cropped