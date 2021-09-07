import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Dataset(Dataset):
    """Class loading RTK and image timestamps data for the whole Oxford dataset"""

    def __init__(self, dataset_root, nsc_images=True, sc_images=False, transforms=None):

        """
        CLEVR-Change dataset class.

        Args:
            dataset_root (string): Path to the root of the CLEVR-Change dataset.
            nsc_images (bool): Use images with non-semantic change as positives.
            sc_images (bool): Use images with semantic change as positives.
            transforms (list of callables): What transforms apply to the images?
        """

        self.dataset_root = dataset_root
        self.nsc_images = nsc_images
        self.sc_images = sc_images
        self.transforms = transforms

        self.image_dirpaths = os.path.join(self.dataset_root, 'images')
        self.image_filenames = [f for f in os.listdir(self.image_dirpaths) if '.jpg' in f or '.png' in f]
        self.image_filenames = sorted(self.image_filenames)
        self.image_filepaths = [os.path.join(self.image_dirpaths, f) for f in self.image_filenames]

        self.image_sc_dirpaths = os.path.join(self.dataset_root, 'sc_images')
        self.image_sc_filenames = [f for f in os.listdir(self.image_sc_dirpaths) if '.jpg' in f or '.png' in f]
        self.image_sc_filenames = sorted(self.image_sc_filenames)
        self.image_sc_filepaths = [os.path.join(self.image_sc_dirpaths, f) for f in self.image_sc_filenames]

        self.image_nsc_dirpaths = os.path.join(self.dataset_root, 'nsc_images')
        self.image_nsc_filenames = [f for f in os.listdir(self.image_nsc_dirpaths) if '.jpg' in f or '.png' in f]
        self.image_nsc_filenames = sorted(self.image_nsc_filenames)
        self.image_nsc_filepaths = [os.path.join(self.image_nsc_dirpaths, f) for f in self.image_nsc_filenames]

    ###########################################################################
    # Magic methods
    ###########################################################################

    def __iter__(self):
        """
        Magic function for iteration start. At each start of iteration (start of each epoch) we sample new sequences
        and indices to be used in this epoch.
        """
        self.iterator_n = 0
        return self

    def __next__(self):
        if self.iterator_n < len(self):
            self.iterator_n += 1
            return self[[self.iterator_n - 1]]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, indices):

        # Read images
        images = []
        for idx in indices:
            if idx < len(self):
                img_filepath = self.image_filepaths[idx]
            elif idx < 2*len(self):
                img_filepath = self.image_nsc_filepaths[idx - len(self)]
            else:
                img_filepath = self.image_sc_filepaths[idx - 2*len(self)]
            img = cv2.cvtColor(cv2.imread(img_filepath), cv2.COLOR_BGR2RGB)
            images.append(img)

        # Transforms
        if self.transforms:
            data = self.transforms((images, None))

        return data


class DatasetSampler(Sampler):

    def __init__(self, data_source: Dataset, batch_size: int, samples_per_epoch=10000, mode='nsc', random_seed=None):
        """
        Sampler constructor.

        Args:
            data_source (Dataset): CLEVR-Change dataset object.
            batch_size (int): Size of the batch
            samples_per_epoch (int): How many images should I produce in each epoch?
            mode (str): one of: 'nsc', 'sc' or 'both'.
            random_seed (int): If passed will be used as a seed for numpy random generator.
        """

        self.data_source = data_source
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode
        assert self.mode == 'nsc' or self.mode == 'sc' or self.mode == 'both', 'mode should be either' \
            '\'nsc\', \'sc\', \'both\''
        self.random_seed = random_seed
        if self.random_seed is not None:
            self.random_state = np.random.RandomState(self.random_seed)

    def generate_batches(self):
        if self.random_seed is not None:
            self.iterated_idcs = self.random_state.choice(len(self.data_source), self.samples_per_epoch)
        else:
            self.iterated_idcs = np.random.choice(len(self.data_source), self.samples_per_epoch)

    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches()

        #
        self.sampled_batch = []
        for sample_idx in range(self.samples_per_epoch):
            samples = self.sample(sample_idx)
            self.sampled_batch.append(samples)
            if sample_idx % self.batch_size == self.batch_size - 1:
                yield self.sampled_batch
                self.sampled_batch = []

    def sample(self, sample_idx):

        # Patch_1
        patch_1_idx = self.iterated_idcs[sample_idx]

        # Patch_2
        patch_2_idx = patch_1_idx
        mode = self.mode
        if mode == 'both':
            mode = np.random.choice(['nsc', 'sc'])
        if mode == 'nsc':
            patch_2_idx += len(self.data_source)
        elif mode == 'sc':
            patch_2_idx += 2*len(self.data_source)

        # Return
        return patch_1_idx, patch_2_idx


def main_nsc_scenario():

    # Root of the dataset
    dataset_root = '/data/input/datasets/ChangeDetection/CLEVR_Change'

    ###########################################################################
    # Create dataset and transforms
    ###########################################################################

    # Transforms
    patch_size = 320
    from src.data.transforms import CenterCrop, ChangeAwarePrep, DictToTensor
    composed_transforms = transforms.Compose([CenterCrop(patch_size),
                                              ChangeAwarePrep(['patch_1', 'patch_2']),
                                              DictToTensor(['patch_1', 'patch_2'])])

    # Load the dataset
    from timeit import default_timer as timer
    start = timer()
    clevr_change_dataset = Dataset(dataset_root=dataset_root, nsc_images=True, sc_images=False,
                                   transforms=composed_transforms)
    end = timer()
    print('Dataset created in {} seconds'.format(end - start))

    ###########################################################################
    # Use dataloader
    ###########################################################################

    clevr_change_sampler = DatasetSampler(data_source=clevr_change_dataset, batch_size=2, samples_per_epoch=100,
                                          mode='nsc')
    dataloader = DataLoader(clevr_change_dataset, batch_sampler=clevr_change_sampler, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):

        sample_idx_in_batch = 0
        patch_1 = sample_batched['patch_1'][sample_idx_in_batch].numpy()
        patch_2 = sample_batched['patch_2'][sample_idx_in_batch].numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(patch_1.transpose((1, 2, 0)))
        ax1.set_title('patch_1')
        ax2.imshow(patch_2.transpose((1, 2, 0)))
        ax2.set_title('patch_2')
        plt.show()


def main_sc_scenario():

    # Root of the dataset
    dataset_root = '/data/input/datasets/ChangeDetection/CLEVR_Change'

    ###########################################################################
    # Create dataset and transforms
    ###########################################################################

    # Transforms
    patch_size = 320
    from src.data.transforms import CenterCrop, ChangeAwarePrep, DictToTensor
    composed_transforms = transforms.Compose([CenterCrop(patch_size),
                                              ChangeAwarePrep(['patch_1', 'patch_2']),
                                              DictToTensor(['patch_1', 'patch_2'])])

    # Load the dataset
    from timeit import default_timer as timer
    start = timer()
    clevr_change_dataset = Dataset(dataset_root=dataset_root, nsc_images=False, sc_images=True,
                                   transforms=composed_transforms)
    end = timer()
    print('Dataset created in {} seconds'.format(end - start))

    ###########################################################################
    # Use dataloader
    ###########################################################################

    clevr_change_sampler = DatasetSampler(data_source=clevr_change_dataset, batch_size=2, samples_per_epoch=100,
                                          mode='sc')
    dataloader = DataLoader(clevr_change_dataset, batch_sampler=clevr_change_sampler, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):

        sample_idx_in_batch = 0
        patch_1 = sample_batched['patch_1'][sample_idx_in_batch].numpy()
        patch_2 = sample_batched['patch_2'][sample_idx_in_batch].numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(patch_1.transpose((1, 2, 0)))
        ax1.set_title('patch_1')
        ax2.imshow(patch_2.transpose((1, 2, 0)))
        ax2.set_title('patch_2')
        plt.show()


def main_both_scenario():

    # Root of the dataset
    dataset_root = '/data/input/datasets/ChangeDetection/CLEVR_Change'

    ###########################################################################
    # Create dataset and transforms
    ###########################################################################

    # Transforms
    patch_size = 320
    from src.data.transforms import CenterCrop, ChangeAwarePrep, DictToTensor
    composed_transforms = transforms.Compose([CenterCrop(patch_size),
                                              ChangeAwarePrep(['patch_1', 'patch_2']),
                                              DictToTensor(['patch_1', 'patch_2'])])

    # Load the dataset
    from timeit import default_timer as timer
    start = timer()
    clevr_change_dataset = Dataset(dataset_root=dataset_root, nsc_images=True, sc_images=True,
                                   transforms=composed_transforms)
    end = timer()
    print('Dataset created in {} seconds'.format(end - start))

    ###########################################################################
    # Use dataloader
    ###########################################################################

    clevr_change_sampler = DatasetSampler(data_source=clevr_change_dataset, batch_size=2, samples_per_epoch=100,
                                          mode='both')
    dataloader = DataLoader(clevr_change_dataset, batch_sampler=clevr_change_sampler, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):

        sample_idx_in_batch = 0
        patch_1 = sample_batched['patch_1'][sample_idx_in_batch].numpy()
        patch_2 = sample_batched['patch_2'][sample_idx_in_batch].numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(patch_1.transpose((1, 2, 0)))
        ax1.set_title('patch_1')
        ax2.imshow(patch_2.transpose((1, 2, 0)))
        ax2.set_title('patch_2')
        plt.show()


if __name__ == "__main__":
    # main_nsc_scenario()
    # main_sc_scenario()
    main_both_scenario()
