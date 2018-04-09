import numpy as np
import h5py
import os
from skimage import io

def _crop_background(signal_volume, label_volume, signal_bg=0, verbose=True):
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
    y_start = np.argmax(only_bg_y)
    y_stop  = np.argmax(1 - only_bg_y[y_start:]) + y_start
    z_start = np.argmax(only_bg_z)
    z_stop  = np.argmax(1 - only_bg_z[z_start:]) + z_start
    if verbose:
        print('cropped x ({} - {}), of len {} ({}%)'.format(x_start, x_stop, len(only_bg_x), 100*(x_stop-x_start)/len(only_bg_x)))
        print('cropped y ({} - {}), of len {} ({}%)'.format(y_start, y_stop, len(only_bg_y), 100*(y_stop-y_start)/len(only_bg_y)))
        print('cropped z ({} - {}), of len {} ({}%)'.format(z_start, z_stop, len(only_bg_z), 100*(z_stop-z_start)/len(only_bg_z)))
        print('volume fraction left = {}%'.format(100*(x_stop-x_start)*(y_stop-y_start)*(z_stop-z_start)/np.prod(signal_volume.shape)))
    # crop out non bg signal
    signal_volume = signal_volume[x_start:x_stop, y_start:y_stop, z_start:z_stop]
    label_volume  = label_volume[ x_start:x_stop, y_start:y_stop, z_start:z_stop]
    return signal_volume, label_volume


def plot_slices(volume, path, axis):
    assert axis in ['x', 'y', 'z']
    axis2num = {'x':0, 'y':1, 'z':2}
    volume = np.moveaxis(volume, axis2num[axis], 0) # move axis to slice over to pos 0
    for i,sliced in enumerate(volume):
        io.imsave('{}/{}.png'.format(path, str(i).zfill(3)), sliced)


def main(args):
    # Read H5 file
    with h5py.File(args.data_filename, 'r') as hf:
        joined = hf[args.filter][:]
        data  = joined[:,:,:,0].squeeze()
        label = joined[:,:,:,1].squeeze()
        data,label = _crop_background(data, label)
    print('read {}, filter {} volume - shape {}'.format(args.data_filename, args.filter, data.shape))

    # for i,s in enumerate(data):
    #     print(i, s.sum())
    # import ipdb; ipdb.set_trace()

    data = np.log10(data + data.min() + 1) # shift min amplitude to 1 and logarithmize -> min is 0
    data = 2*(data/data.max()) - 1 # signal range in [-1,1] for skimage.io.imwrite

    # import ipdb; ipdb.set_trace()

    for axis in ['x', 'y', 'z']:
        path = './plots_crop_bg/{}/log_signal_slice_{}'.format(args.filter, axis)
        if not os.path.exists(path):
            os.makedirs(path)
        plot_slices(data, path, axis)
        # path = './plots_crop_bg/{}/label_slice_{}'.format(args.filter, axis)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # plot_slices(label, path, axis)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-filename", required=True,
                        help="The name of the data file.")
    parser.add_argument("--filter", required=True,
                        help="The volume identifier in the hdf5 file.")
    args = parser.parse_args()
    main(args=args)