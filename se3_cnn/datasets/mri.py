# pylint: disable=C,R,E1101
import torch
import h5py
import numpy as np
import numbers
import sys


class MRISegmentation(torch.utils.data.Dataset):
    """Read 3D medical image files in .nii format, and provide it to the
       user as 3D patches of the requested size. Setting
       randomize_patch_offsets=True will add random offsets to the
       patches to reduce the affect of patch boundaries."""

    def __init__(self, h5_filename, filter, patch_shape,
                 randomize_patch_offsets=True,
                 pad_mode='constant',
                 pad_constant=0,
                 log10_signal=False):

        if isinstance(patch_shape, numbers.Integral):
            patch_shape = np.repeat(patch_shape, 3)
        self.patch_shape = patch_shape
        self.randomize_patch_offsets = randomize_patch_offsets
        self.pad_mode = pad_mode
        self.pad_constant = pad_constant
        self.log10_signal = log10_signal

        self.data = []
        self.labels = []
        self.unpadded_data_shape = []
        self.padding_boundary = []
        self.patch_indices = []

        # Read H5 file
        print("Reading data...", end="")
        sys.stdout.flush()
        with h5py.File(h5_filename, 'r') as hf:
            for name in filter:
                data = hf[name][:]
                # Assumption: voxel value and pixel are stored in last dim
                signal_volume = data[:,:,:,0].squeeze()
                label_volume  = data[:,:,:,1].squeeze()
                signal_volume, label_volume = self._crop_background(signal_volume, label_volume)
                self.data.append(signal_volume)
                self.labels.append(label_volume)
                self.unpadded_data_shape.append(self.data[-1].shape)
                self.padding_boundary.append(None)
        print("done.")

        # This first call to initialize_patch_indices will calculate the
        # total padding around each image, and store it in self.padding_boundary
        self.initialize_patch_indices()

        # The total padding is fixed to a multiple of the patch size - we
        # can therefore add the full padding in the setup fase
        print("Applying padding...", end="")
        sys.stdout.flush()
        for i, image in enumerate(self.data):
            pad_width = self.padding_boundary[i]
            self.data[i] = np.pad(self.data[i], pad_width,
                                  mode=self.pad_mode,
                                  constant_values=self.pad_constant)
            self.labels[i] = np.pad(self.labels[i], pad_width,
                                    mode=self.pad_mode,
                                    constant_values=self.pad_constant)
        print("done.")
        # optionally logarithmize zero shifted input signal
        if self.log10_signal:
            signal_min = min([np.min(data_i for data_i in self.data)])
            for i in range(len(self.data)):
                self.data[i] = np.log10(self.data[i] + signal_min + 1) # add 1 to prevent -inf from the log


    def _crop_background(self, signal_volume, label_volume, signal_bg=0, verbose=True):
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


    def initialize_patch_indices(self):
        """For each image, calculate the indices for each patch, possibly
           shifted by a random offset"""

        self.patch_indices = []
        for i, image in enumerate(self.data):
            patch_indices, overflow = self.calc_patch_indices(
                self.unpadded_data_shape[i],
                self.patch_shape,
                randomize_offset=self.randomize_patch_offsets)
            patch_indices = np.append(np.full(shape=(patch_indices.shape[0],1),
                                              fill_value=i),
                                      patch_indices, axis=1)
            self.patch_indices += patch_indices.tolist()

            # Keep track of how much each image has been padded
            if self.padding_boundary[i] is None:
                pad_width = np.stack([overflow, overflow], axis=1)
                self.padding_boundary[i] = pad_width

    def __getitem__(self, index):
        """Retrieve a single patch"""

        # Which image to retrieve patch from
        dataset_index = self.patch_indices[index][0]

        # Extract image and label
        image = self.data[dataset_index]
        labels = self.labels[dataset_index]

        # Obtain patch indices into original image
        patch_index_start = np.array(self.patch_indices[index][1:],
                                     dtype=np.int16)
        patch_index_end = patch_index_start + self.patch_shape

        # Update patch indices to padded image
        patch_index_start += self.padding_boundary[dataset_index][:,0]
        patch_index_end += self.padding_boundary[dataset_index][:,0]

        # Lookup image and add channel dimension
        image_patch = np.expand_dims(
            image[patch_index_start[0]:patch_index_end[0],
                  patch_index_start[1]:patch_index_end[1],
                  patch_index_start[2]:patch_index_end[2]],
            axis=0).astype(np.float32)
        labels_patch = np.expand_dims(
            labels[patch_index_start[0]:patch_index_end[0],
                   patch_index_start[1]:patch_index_end[1],
                   patch_index_start[2]:patch_index_end[2]],
            axis=0)

        # Check that patch has the correct size
        assert np.all(image_patch[0].shape == self.patch_shape)
        assert np.all(labels_patch[0].shape == self.patch_shape)

        return image_patch, labels_patch

    def __len__(self):
        return len(self.patch_indices)

    @staticmethod
    def calc_patch_indices(image_shape, patch_shape,
                           overlap=0,
                           randomize_offset=True,
                           minimum_overflow_fraction=0.25):
        """
        Given the image shape and the patch shape, calculate the placement
        of patches. If randomize_offset is on, it will randomize the placement,
        so that the patch boundaries affect different regions in each epoch.
        There is natural room for this randomization whenever the image size
        if not divisible by the patch size, in the sense that the overflow
        can be placed arbitrarily in the beginning on the end. If you want
        to ensure that some randomization will always occur, you can set
        minimum_overflow_fraction, which will ensure that an extra patch will
        be added to provide extra overflow if necessary.

        :param image_shape: Shape if input image
        :param patch_shape: Shape of patch
        :param overlap: Allow patches to overlap with this number of voxels
        :param randomize_offset: Whether patch placement should be normalized
        :param minimum_overflow_fraction: Specify to force overflow beyond image
               to be at least this fraction of the patch size
        :return:
        """
        if isinstance(overlap, numbers.Integral):
            overlap = np.repeat(overlap, len(image_shape))

        # Effective patch shape
        eff_patch_shape = (patch_shape - overlap)

        # Number of patches (rounding up)
        n_patches = np.ceil(image_shape / eff_patch_shape).astype(int)

        # Overflow of voxels beyond image
        overflow = eff_patch_shape * n_patches - image_shape + overlap

        if randomize_offset:

            # Add extra patch for dimensions where minimum is below fraction
            extra_patch = (overflow/patch_shape) < minimum_overflow_fraction
            overflow += extra_patch*eff_patch_shape
            n_patches += 1

            # Select random start index so that overlap is spread randomly
            # on both sides. If overflow is larger than patch_shape
            max_start_offset = overflow
            start_index = -np.array([np.random.choice(offset + 1)
                                     for offset in max_start_offset])
            # max_start_offset = np.minimum(overflow, patch_shape-1)
            # min_start_offset = np.maximum(0, overflow-max_start_offset)
            # minmax_start_offset = list(zip(min_start_offset, max_start_offset))
            # start_index = -np.array([np.random.choice(np.arange(offset[0],
            #                                                     offset[1]+1))
            #                          for offset in minmax_start_offset])
        else:

            # Set start index to overflow is spread evenly on both sides
            start_index = -np.ceil(overflow/2).astype(int)

        stop_index = image_shape + start_index + overflow
        step_size = eff_patch_shape

        return (np.mgrid[start_index[0]:stop_index[0]:step_size[0],
                         start_index[1]:stop_index[1]:step_size[1],
                         start_index[2]:stop_index[2]:step_size[2]].reshape(3, -1).T,
                overflow)






