# pylint: disable=C,R,E1101
import torch
import h5py
import numpy as np
import numbers
import sys

miccai_filters = {
    'train': ["1000_3",
              "1001_3",
              "1002_3",
              "1006_3",
              "1007_3",
              "1008_3",
              "1009_3",
              "1010_3",
              "1011_3",
              "1012_3",
              "1013_3",
              "1014_3"],
    'validation': ["1015_3",
                   "1017_3",
                   "1036_3"],
    'test': ["1003_3",
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
    }
# check that filters are non-overlapping
assert len(set(miccai_filters['train']).intersection(miccai_filters['validation'])) == 0
assert len(set(miccai_filters['train']).intersection(miccai_filters['test'])) == 0

def get_miccai_dataloader(dataset,
                          h5_filename,
                          mode,
                          patch_shape,
                          patch_overlap,
                          batch_size,
                          num_workers,
                          pin_memory,
                          **read_data_kwargs):
    if mode == 'train':
        data_set = MRISegmentation(dataset=dataset,
                                   h5_filename=h5_filename,
                                   mode='train',
                                   patch_shape=patch_shape,
                                   patch_overlap=patch_overlap,
                                   **read_data_kwargs)
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory,
                                                  drop_last=True)
        np.set_printoptions(threshold=np.nan)

    if mode in ['train', 'validation']:
        data_set = MRISegmentation(dataset=dataset,
                                   h5_filename=h5_filename,
                                   mode='validation',
                                   patch_shape=patch_shape,
                                   patch_overlap=patch_overlap,
                                   randomize_patch_offsets=False,
                                   **read_data_kwargs)
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory,
                                                  drop_last=False)
    if mode == 'test':
        data_set = MRISegmentation(dataset=dataset,
                                   h5_filename=h5_filename,
                                   mode='test',
                                   patch_shape=patch_shape,
                                   patch_overlap=patch_overlap,
                                   randomize_patch_offsets=False,
                                   **read_data_kwargs)
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory,
                                                  drop_last=False)
    return data_set, data_loader

def get_mrbrains_dataloader(dataset,
                            h5_filename,
                            mode,
                            patch_shape,
                            patch_overlap,
                            batch_size,
                            num_workers,
                            pin_memory,
                            N_train):
    if mode == 'train':
        data_set = MRISegmentation(dataset=dataset,
                                   h5_filename=h5_filename,
                                   mode='train',
                                   patch_shape=patch_shape,
                                   patch_overlap=patch_overlap)
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory,
                                                  drop_last=True)
        np.set_printoptions(threshold=np.nan)
        print(np.unique(data_set.labels[0]))
    if mode in ['train', 'validation']:
        data_set = MRISegmentation(dataset=dataset,
                                   h5_filename=h5_filename,
                                   mode='validation',
                                   patch_shape=patch_shape,
                                   patch_overlap=patch_overlap,
                                   randomize_patch_offsets=False)
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory,
                                                  drop_last=False)
    if mode == 'test':
        data_set = MRISegmentation(dataset=dataset,
                                   h5_filename=h5_filename,
                                   mode='test',
                                   patch_shape=patch_shape,
                                   patch_overlap=patch_overlap,
                                   randomize_patch_offsets=False)
        data_loader = torch.utils.data.DataLoader(data_set,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  pin_memory=pin_memory,
                                                  drop_last=False)
    return data_set, data_loader

def read_h5_data(dataset, h5_filename, mode, **read_data_kwargs):
    ''' read MRI datasets from h5 files
        :param dataset: selects miccai or mrbrains, the latter with either reduced or full labels
        :param h5_filename: path to the h5 file
        :param mode: load train, validation or test set
    '''
    assert dataset in ['miccai', 'mrbrains_reduced', 'mrbrains_full']
    assert mode in ['train', 'validation', 'test']
    if dataset == 'miccai':
        return read_h5_data_miccai(h5_filename, mode, **read_data_kwargs)
    else:
        label_mode = dataset.split('_')[-1] # 'reduced' or 'full'
        return read_h5_data_mrbrains(h5_filename, mode, label_mode, **read_data_kwargs)

def read_h5_data_miccai(h5_filename, mode, filter=None):
    ''' to be called from read_h5_data '''
    data = []
    labels = []
    unpadded_data_spatial_shape = []
    padding_boundary = []
    patch_indices = []
    if filter == None:
        filter = miccai_filters.get(mode)
    with h5py.File(h5_filename, 'r') as hf:
        for name in filter:
            # Assumption: voxel value and pixel are stored in last dim
            signal_volume = hf[name][:][:,:,:,0].squeeze()[np.newaxis,...]
            label_volume  = hf[name][:][:,:,:,1].squeeze()
            data.append(signal_volume)
            labels.append(label_volume)
            unpadded_data_spatial_shape.append(data[-1].shape[1:])
            padding_boundary.append(None)
        class_count = hf['class_counts'][:]
        # signal_mean = hf['signal_mean'][:]
        # signal_std  = hf['signal_std'][:]
        # bg_values = -signal_mean/signal_std # since original bg value was zero
    return data, labels, unpadded_data_spatial_shape, padding_boundary, class_count

def read_h5_data_mrbrains(h5_filename, mode, label_mode, N_train=4):
    ''' to be called from read_h5_data
        training set is split into N_train training samples and 5-N_train validation samples
    '''
    with h5py.File(h5_filename, 'r') as hf:
        if mode == 'train':
            data   = [hf['train_signal_{}'.format(i)][()]               for i in range(N_train)]
            labels = [hf['train_label_{}_{}'.format(label_mode, i)][()] for i in range(N_train)]
        elif mode == 'validation':
            data   = [hf['train_signal_{}'.format(i)][()]               for i in range(N_train, 5)]
            labels = [hf['train_label_{}_{}'.format(label_mode, i)][()] for i in range(N_train, 5)]
        elif mode == 'test':
            data   = [hf['test_signal_{}'.format(i)][()] for i in range(15)]
            labels = None
        class_count = hf['class_counts_{}'.format(label_mode)][:]
        unpadded_data_spatial_shape = [d.shape[1:] for d in data]
        padding_boundary = [None for d in data]
        # channel_means = hf['channel_means'][:]
        # channel_stds  = hf['channel_stds'][:]
    return data, labels, unpadded_data_spatial_shape, padding_boundary, class_count


class MRISegmentation(torch.utils.data.Dataset):
    ''' Read 3D medical image files in .nii format, and provide it to the
        user as 3D patches of the requested size. Setting
        randomize_patch_offsets=True will add random offsets to the
        patches to reduce the affect of patch boundaries.
        :param dataset: dataset to be loaded. options are 'miccai', 'mrbrains_reduced' and 'mrbrains_full'
        :param h5_filename: path to the hdf5 file
        :param mode: load 'train', 'validation' or 'test' set
        :param patch_shape:
        :param filter: optional - only for miccai select exactly which scans to load
        :param randomize_patch_offsets:
        # :param pad_mode:
        :param pad_constant:
        :param read_data_kwargs: keywordargs options for different datasets
                                 used to pass `filter` for miccai and `N_train` for mrbrains
    '''
    def __init__(self,
                 dataset,
                 h5_filename,
                 mode,
                 patch_shape,
                 patch_overlap,
                 randomize_patch_offsets=True,
                 # pad_mode='constant',
                 # pad_constant=0,
                 **read_data_kwargs):

        if isinstance(patch_shape, numbers.Integral):
            patch_shape = np.repeat(patch_shape, 3)
        if isinstance(patch_overlap, numbers.Integral):
            patch_overlap = np.repeat(patch_overlap, 3)
        self.patch_shape = patch_shape
        self.patch_overlap = patch_overlap
        self.randomize_patch_offsets = randomize_patch_offsets
        # self.pad_mode = pad_mode
        # self.pad_constant = pad_constant
        # self.log10_signal = log10_signal

        # Read H5 file
        print("Reading data...", end="")
        sys.stdout.flush()
        self.data, self.labels, self.unpadded_data_spatial_shape, self.padding_boundary, self.class_count = \
             read_h5_data(dataset, h5_filename, mode, **read_data_kwargs)
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
            # for data which contains a channel dimension add an entry to pad_width
            if len(self.data[i].shape) == 4:
                pad_width_data = np.insert(pad_width, 0, values=0, axis=0)
            # self.data[i] = np.pad(self.data[i], pad_width_data,
            #                       mode=self.pad_mode,
            #                       constant_values=self.pad_constant)
            # self.labels[i] = np.pad(self.labels[i], pad_width,
            #                         mode=self.pad_mode,
            #                         constant_values=self.pad_constant).astype(np.int64)
            pad_mode = 'symmetric' # constant value padding unclear for some signal channels
            self.data[i]   = np.pad(self.data[i],   pad_width_data, mode=pad_mode)
            self.labels[i] = np.pad(self.labels[i], pad_width,      mode=pad_mode).astype(np.int64)

        print("done.")
        # # optionally logarithmize zero shifted input signal
        # if self.log10_signal:
        #     print('logarithmize signal')
        #     signal_min = min([np.min(data_i) for data_i in self.data])
        #     for i in range(len(self.data)):
        #         self.data[i] = np.log10(self.data[i] + signal_min + 1) # add 1 to prevent -inf from the log


    def get_original(self, dataset_index):
        """Get full input image at specified index"""
        size = self.unpadded_data_spatial_shape[dataset_index]
        patch_index_start = self.padding_boundary[dataset_index][:,0]
        patch_index_end = patch_index_start + size
        patch_index = np.stack((patch_index_start, patch_index_end))
        return (self.data[dataset_index][patch_index[0, 0]:patch_index[1, 0],
                                         patch_index[0, 1]:patch_index[1, 1],
                                         patch_index[0, 2]:patch_index[1, 2]],
                self.labels[dataset_index][patch_index[0, 0]:patch_index[1, 0],
                                         patch_index[0, 1]:patch_index[1, 1],
                                         patch_index[0, 2]:patch_index[1, 2]])

    def initialize_patch_indices(self):
        """For each image, calculate the indices for each patch, possibly
           shifted by a random offset"""

        self.patch_indices = []
        for i, image in enumerate(self.data):
            patch_indices, overflow = self.calc_patch_indices(
                self.unpadded_data_spatial_shape[i],
                self.patch_shape,
                overlap=self.patch_overlap,
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

        patch_index = np.stack((patch_index_start, patch_index_end))
        patch_valid = np.stack(patch_index).clip(
            min=0, max=self.unpadded_data_spatial_shape[dataset_index]) - patch_index[0]

        # Update patch indices to padded image
        patch_index_padded = patch_index + self.padding_boundary[dataset_index][:,0]


        # # OLD VERSION
        # # assumed images not to have channel dimension yet and hence adds it
        # Lookup image and add channel dimension
        # image_patch = np.expand_dims(
        #     image[patch_index_padded[0, 0]:patch_index_padded[1, 0],
        #           patch_index_padded[0, 1]:patch_index_padded[1, 1],
        #           patch_index_padded[0, 2]:patch_index_padded[1, 2]],
        #     axis=0).astype(np.float32)

        # Slice image patch
        image_patch = image[:, patch_index_padded[0, 0]:patch_index_padded[1, 0],
                               patch_index_padded[0, 1]:patch_index_padded[1, 1],
                               patch_index_padded[0, 2]:patch_index_padded[1, 2]].astype(np.float32)
        # Slice label patch and add dimension
        labels_patch = np.expand_dims(
            labels[patch_index_padded[0, 0]:patch_index_padded[1, 0],
                   patch_index_padded[0, 1]:patch_index_padded[1, 1],
                   patch_index_padded[0, 2]:patch_index_padded[1, 2]],
            axis=0)

        # print("image: ", image_patch)


        # Check that patch has the correct size
        assert np.all(image_patch[0].shape == self.patch_shape)
        assert np.all(labels_patch[0].shape == self.patch_shape)

        return image_patch, labels_patch, dataset_index, patch_index, patch_valid

    def __len__(self):
        return len(self.patch_indices)

    @staticmethod
    def calc_patch_indices(image_shape,
                           patch_shape,
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
        n_patches = np.ceil((image_shape - patch_shape) / eff_patch_shape + 1).astype(int)

        # Overflow of voxels beyond image
        overflow = eff_patch_shape * n_patches - image_shape + overlap

        if randomize_offset:

            # Add extra patch for dimensions where minimum is below fraction
            extra_patch = (overflow / patch_shape) < minimum_overflow_fraction
            while extra_patch.any():
                overflow += extra_patch*eff_patch_shape
                n_patches += extra_patch
                extra_patch = (overflow / patch_shape) < minimum_overflow_fraction

            # Select random start index so that overlap is spread randomly
            # on both sides. If overflow is larger than patch_shape
            max_start_offset = overflow
            start_index = -np.array([np.random.choice(offset + 1)
                                     for offset in max_start_offset])
        else:

            # Set start index to overflow is spread evenly on both sides
            start_index = -np.ceil(overflow/2).astype(int)

            # In the non-randomize setting, we still one to make sure that the
            # last patch sticks outside the image with at least overlap/2, i.e,
            # that overflow/2 > overlap/2 (since the overflow is distributed
            # evenly on both sides when randomize_offset=True
            extra_patch = (overflow < overlap)
            while extra_patch.any():
                overflow += extra_patch*eff_patch_shape
                n_patches += extra_patch
                extra_patch = (overflow < overlap)


        # stop_index = image_shape + start_index + overflow
        step_size = eff_patch_shape
        stop_index = start_index + step_size*n_patches

        return (np.mgrid[start_index[0]:stop_index[0]:step_size[0],
                         start_index[1]:stop_index[1]:step_size[1],
                         start_index[2]:stop_index[2]:step_size[2]].reshape(3, -1).T,
                overflow)





