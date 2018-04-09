"""
This script laods the MICCAI 2012 Multi-Atlas Challenge dataset stored in nifty format (.nii)
and exports the labels to npz files (.mat).

Note that labels are converted into 135 categories.
"""
import os
import glob
import sys
import numpy as np
import h5py
import nibabel as nib
nib.nifti1.Nifti1Header.quaternion_threshold = -1e-6

ignored_labels = list(range(1,4))+list(range(5,11))+list(range(12,23))+list(range(24,30))+[33,34]+[42,43]+[53,54]+list(range(63,69))+[70,74]+\
                    list(range(80,100))+[110,111]+[126,127]+[130,131]+[158,159]+[188,189]

# 47: right hippocampus, 48: left hippocampus, 0: others
true_labels = [4, 11, 23, 30, 31, 32, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57,
                58, 59, 60, 61, 62, 69, 71, 72, 73, 75, 76, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 112,
                113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 128, 129, 132, 133, 134, 135, 136,
                137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
                157, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
                179, 180, 181, 182, 183, 184, 185, 186, 187, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
                201, 202, 203, 204, 205, 206, 207]


def label_filtering(lab, ignore_labels, true_labels):
    for ignored_label in ignored_labels:
        lab[lab == ignored_label] = 0
    for idx, label in enumerate(true_labels):
        lab[lab==label] = idx+1
    return lab

def crop_background(signal_volume, label_volume, signal_bg=0, verbose=True):
    """Crop out the cube in the signal and label volume in which the signal is non-zero"""
    bg_mask = signal_volume == signal_bg
    # the following DOES NOT work since there is skull/skin signal labeled as background
    # bg_mask = label_volume == bg_label
    # generate 1d arrays over axes which are false iff only bg is found in the corresponding slice
    only_bg_x = 1-np.all(bg_mask, axis=(1,2))
    only_bg_y = 1-np.all(bg_mask, axis=(0,2))
    only_bg_z = 1-np.all(bg_mask, axis=(0,1))
    # get start and stop index of non bg mri volume
    x_start = np.argmax(only_bg_x)
    x_stop  = np.argmax(1 - only_bg_x[x_start:]) + x_start
    x_stop  = x_stop if x_start!=x_stop else len(only_bg_x)
    y_start = np.argmax(only_bg_y)
    y_stop  = np.argmax(1 - only_bg_y[y_start:]) + y_start
    y_stop  = y_stop if y_start!=y_stop else len(only_bg_y)
    z_start = np.argmax(only_bg_z)
    z_stop  = np.argmax(1 - only_bg_z[z_start:]) + z_start
    z_stop  = z_stop if z_start!=z_stop else len(only_bg_z)
    if verbose:
        print('cropped x ({} - {}), of len {} ({}%)'.format(x_start, x_stop, len(only_bg_x), 100*(x_stop-x_start)/len(only_bg_x)))
        print('cropped y ({} - {}), of len {} ({}%)'.format(y_start, y_stop, len(only_bg_y), 100*(y_stop-y_start)/len(only_bg_y)))
        print('cropped z ({} - {}), of len {} ({}%)'.format(z_start, z_stop, len(only_bg_z), 100*(z_stop-z_start)/len(only_bg_z)))
        print('volume fraction left = {}%'.format(100*(x_stop-x_start)*(y_stop-y_start)*(z_stop-z_start)/np.prod(signal_volume.shape)))
    # crop out non bg signal
    signal_volume = signal_volume[x_start:x_stop, y_start:y_stop, z_start:z_stop]
    label_volume  = label_volume[ x_start:x_stop, y_start:y_stop, z_start:z_stop]
    return signal_volume, label_volume


mri_dir = sys.argv[1]
label_dir = sys.argv[2]

mri_files = glob.glob(os.path.join(mri_dir, '*.nii'))
data_list = []
label_list = []
with h5py.File('miccai12.h5', 'w') as hf:
    for mri_file in mri_files:
        basename = os.path.basename(mri_file).replace('.nii', '')
        label_file = glob.glob(os.path.join(label_dir, basename + "*.nii"))[0]
        print('Processing {}'.format(basename))

        data = nib.load(mri_file).get_data().squeeze()
        labels = nib.load(label_file).get_data().squeeze()
        labels = labels.astype(int, copy=False)
        labels = label_filtering(labels, ignored_labels, true_labels)

        data, labels = crop_background(data, labels)
        data_list.append(data)
        label_list.append(labels)

    # compute data normalization coefficients
    data_flat = np.concatenate([data_vol.flatten() for data_vol in data_list])
    data_mean, data_std = data_flat.mean(), data_flat.std()

    for i,mri_file in enumerate(mri_files):
        basename = os.path.basename(mri_file).replace('.nii', '')
        data_i = (data_list[i] - data_mean)/data_std
        labels_i = label_list[i]
        data_labels = np.stack([data_i, labels_i], axis=-1)
        hf.create_dataset(basename, data=data_labels, compression="gzip")
        print('saved {}, normalized to mean={} and std={}'.format(basename, data_i.mean(), data_i.std()))


    print('compute class count for class imbalance')
    stacked_labels = np.concatenate([label_vol.flatten() for label_vol in label_list])
    labels_unique = np.unique(stacked_labels)
    class_counts = np.array([np.sum(stacked_labels==label) for label in labels_unique])
    # np.savez('class_counts.npz', class_counts)
    hf.create_dataset('class_counts', data=class_counts, compression='gzip')