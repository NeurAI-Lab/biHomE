import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.data.utils import warp_image
from src.data.utils import four_point_to_homography


# class Dataset(Dataset):
#     """Class loading RTK and image timestamps data for the whole Oxford dataset"""
#
#     def __init__(self, dataset_root, transforms=None):
#
#         """
#         COCO dataset class.
#
#         Args:
#             dataset_root (string): Path to the root of the COCO images.
#             transforms (list of callables): What transforms apply to the images?
#         """
#
#         self.dataset_root = dataset_root
#         self.transforms = transforms
#         self.img_filenames = [f for f in os.listdir(self.dataset_root) if '.jpg' in f or '.npy' in f]
#         self.img_filepaths = [os.path.join(self.dataset_root, f) for f in self.img_filenames]
#
#     ###########################################################################
#     # Conversion method
#     ###########################################################################
#
#     def preprocess_offline(self, output_dataset_root):
#
#         # Make dirs if needed
#         if not os.path.exists(output_dataset_root):
#             os.makedirs(output_dataset_root)
#
#         # Copy files
#         for idx in tqdm(range(len(self))):
#
#             # Get current seq path
#             image = self.load_image(idx)
#
#             # Apply transforms
#             if self.transforms:
#                 data = self.transforms(([image], None))
#
#             # Save the image
#             filename = '.'.join(self.img_filenames[idx].rsplit('.')[:-1]) + '.npy'
#             data_filename = os.path.join(output_dataset_root, filename)
#             np.save(data_filename, data[0][0], allow_pickle=True)
#
#     ###########################################################################
#     # Magic methods
#     ###########################################################################
#
#     def __iter__(self):
#         """
#         Magic function for iteration start. At each start of iteration (start of each epoch) we sample new sequences
#         and indices to be used in this epoch.
#         """
#         self.iterator_n = 0
#         return self
#
#     def __next__(self):
#         if self.iterator_n < len(self):
#             self.iterator_n += 1
#             return self[[self.iterator_n - 1]]
#         else:
#             raise StopIteration
#
#     def __len__(self):
#         return len(self.img_filenames)
#
#     def __getitem__(self, indices):
#         # Read images
#         images = []
#         for idx in indices:
#             img = self.load_image(idx)
#         images.append(img)
#
#         # Transforms
#         if self.transforms:
#             data = self.transforms((images, None))
#
#         return data
#
#     def load_image(self, idx):
#         filepath = self.img_filepaths[idx]
#         if '.jpg' in filepath:
#             img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
#         elif '.npy' in filepath:
#             img = np.load(filepath, allow_pickle=True)
#         else:
#             assert False, 'I dont know this format'
#         return img
#

import pickle
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class Dataset(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(Dataset, self).__init__(root, transform=transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(([img], None))

        return data

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class DatasetSampler(Sampler):

    def __init__(self, data_source: Dataset, batch_size: int, samples_per_epoch=10000, mode='pair', random_seed=None):
        """
        Sampler constructor.

        There is 77 sequences with RTK data and each sequence has on average about 30k images, which results in about
        2.5 million of images. I've assumed that each epoch will have 10k images (specified with @samples_per_epoch).
        Positive sample will be randomly chosen between +-positive_max_frame_dist, stereo camera frame rate is 16Hz,
        so I would recommend to choose positive_max_frame_dist=16.

        Args:
            data_source (Dataset): Oxford dataset object.
            batch_size (int): Size of the batch
            samples_per_epoch (int): How many images should I produce in each epoch?
            mode (str): Should I sample single image from the dataset, pair of images from the same sequence but distant
                in time specified by @pair_max_frame_dist, or triplet of frames with corresponding pose, but captured in
                different sequences?
            random_seed (int): If passed will be used as a seed for numpy random generator.
        """

        self.data_source = data_source
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode
        assert self.mode == 'single', 'COCO dataset support only single mode'
        self.random_seed = random_seed
        if self.random_seed is not None:
            self.random_state = np.random.RandomState(self.random_seed)

    def generate_batches(self):
        if self.random_seed is not None:
            self.iterated_idcs = self.random_state.choice(np.arange(len(self.data_source.img_filepaths)),
                                                          self.samples_per_epoch)
        else:
            self.iterated_idcs = np.random.choice(np.arange(len(self.data_source.img_filepaths)),
                                                  self.samples_per_epoch)

    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches()

        #
        self.sampled_batch = []
        for sample_idx, idx in enumerate(self.iterated_idcs):
            self.sampled_batch.append([idx])
            if sample_idx % self.batch_size == self.batch_size - 1:
                yield self.sampled_batch
                self.sampled_batch = []


def main_single_scenario():

    # Root of the dataset
    dataset_root = '/data/input/datasets/CIFAR-10'

    ###########################################################################
    # Create dataset and transforms
    ###########################################################################

    # Transforms
    patch_size = 128
    from src.data.transforms import Rescale, HomographyNetPrep, DictToGrayscale, DictToTensor
    composed_transforms = transforms.Compose([Rescale((192, 192)),
                                              HomographyNetPrep(rho=patch_size//4, patch_size=patch_size,
                                                                photometric_distort_keys=[], target_gen='4_points'),
                                              DictToGrayscale(keys=['image_1', 'image_2', 'patch_1', 'patch_2']),
                                              DictToTensor(['patch_1', 'patch_2', 'image_1', 'image_2'])])

    # Load the dataset
    start = timer()
    #cifar10_trainset = Dataset(root=dataset_root, train=True, transform=composed_transforms)
    cifar10_testset = Dataset(root=dataset_root, train=False, transform=composed_transforms)
    end = timer()
    print('Dataset created in {} seconds'.format(end - start))

    ###########################################################################
    # Using iteration
    ###########################################################################

    # for images in cifar10_testset:
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    #     query = images['image_1'].numpy()
    #     query = np.transpose(np.tile(query.astype(np.uint8), (3, 1, 1)), (1, 2, 0))
    #     ax1.imshow(query)
    #     ax1.set_title('query image')
    #     patch_1 = images['patch_1'].numpy()
    #     patch_1[patch_1 > 255] = 255
    #     patch_1[patch_1 < 0] = 0
    #     patch_1 = np.transpose(np.tile(patch_1.astype(np.uint8), (3, 1, 1)), (1, 2, 0))
    #     ax2.imshow(patch_1)
    #     ax2.set_title('patch_1')
    #     patch_2 = images['patch_2'].numpy()
    #     patch_2[patch_2 > 255] = 255
    #     patch_2[patch_2 < 0] = 0
    #     patch_2 = np.transpose(np.tile(patch_2.astype(np.uint8), (3, 1, 1)), (1, 2, 0))
    #     ax3.imshow(patch_2)
    #     ax3.set_title('patch_2')
    #     plt.show()

    ###########################################################################
    # Use dataloader
    ###########################################################################

    #coco_sampler = DatasetSampler(data_source=coco_dataset, batch_size=4, samples_per_epoch=100, mode='single')
    testloader = DataLoader(cifar10_testset, batch_size=4, shuffle=True, num_workers=2)
    for i_batch, sample_batched in enumerate(testloader):

        # Get patches and delta_gt
        sample_idx_in_batch = 0
        image_1 = sample_batched['image_1'][sample_idx_in_batch].numpy().transpose(1, 2, 0)
        image_2 = sample_batched['image_2'][sample_idx_in_batch].numpy().transpose(1, 2, 0)
        patch_1 = sample_batched['patch_1'][sample_idx_in_batch].numpy().transpose(1, 2, 0)
        patch_2 = sample_batched['patch_2'][sample_idx_in_batch].numpy().transpose(1, 2, 0)
        delta_gt = sample_batched['delta'][sample_idx_in_batch].numpy()
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(3, 10))

        # Image 1
        ax1.imshow(np.tile(image_1.astype(np.uint8), (1, 1, 3)))
        ax1.set_title('image_1')

        # Patch 1
        ax2.imshow(np.tile(patch_1.astype(np.uint8), (1, 1, 3)))
        ax2.set_title('patch_1')

        # Patch 1 warped
        image_corners = sample_batched['corners'][sample_idx_in_batch].numpy()
        homography = four_point_to_homography(np.expand_dims(image_corners, axis=0), np.expand_dims(delta_gt, axis=0),
                                              crop=False)
        image_1_w = warp_image(image_1, homography, target_h=image_1.shape[0], target_w=image_1.shape[1])
        patch_1_w = image_1_w[image_corners[0, 1]:image_corners[3, 1], image_corners[0, 0]:image_corners[1, 0]]
        patch_1_w = np.expand_dims(patch_1_w, axis=-1)
        ax3.imshow(np.tile(patch_1_w.astype(np.uint8), (1, 1, 3)))
        ax3.set_title('patch_1 warped')

        # Image 2
        ax4.imshow(np.tile(image_2.astype(np.uint8), (1, 1, 3)))
        ax4.set_title('image_2')

        # Patch 2
        ax5.imshow(np.tile(patch_2.astype(np.uint8), (1, 1, 3)))
        ax5.set_title('patch_2')

        # Patch 2 warped
        image_corners_2 = image_corners + delta_gt
        delta_gt_2 = -delta_gt
        homography_2 = four_point_to_homography(np.expand_dims(image_corners_2, axis=0),
                                                np.expand_dims(delta_gt_2, axis=0), crop=False)
        image_2_w = warp_image(image_2, homography_2, target_h=image_2.shape[0], target_w=image_2.shape[1])
        patch_2_w = image_2_w[image_corners[0, 1]:image_corners[3, 1], image_corners[0, 0]:image_corners[1, 0]]
        patch_2_w = np.expand_dims(patch_2_w, axis=-1)
        ax6.imshow(np.tile(patch_2_w.astype(np.uint8), (1, 1, 3)))
        ax6.set_title('patch_2 warped')
        plt.show()


def main_single_photometric_distort_test():

    # Root of the dataset
    dataset_root = '/data/input/datasets/CIFAR-10'

    ###########################################################################
    # Create dataset and transforms
    ###########################################################################

    # Transforms
    patch_size = 128
    from src.data.transforms import Rescale, HomographyNetPrep, DictToGrayscale, DictToTensor
    composed_transforms = transforms.Compose([Rescale((192, 192)),
                                              HomographyNetPrep(rho=patch_size//4, patch_size=patch_size,
                                                                photometric_distort_keys=['image_1', 'image_2'],
                                                                target_gen='4_points'),
                                              DictToGrayscale(keys=['image_1', 'image_2', 'patch_1', 'patch_2']),
                                              DictToTensor(['patch_1', 'patch_2', 'image_1', 'image_2'])])

    # Load the dataset
    start = timer()
    #cifar10_trainset = Dataset(root=dataset_root, train=True, transform=composed_transforms)
    cifar10_testset = Dataset(root=dataset_root, train=False, transform=composed_transforms)
    end = timer()
    print('Dataset created in {} seconds'.format(end - start))

    ###########################################################################
    # Using iteration
    ###########################################################################

    # for images in cifar10_testset:
    #     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    #     query = images['image_1'].numpy()
    #     query[query > 255] = 255
    #     query[query < 0] = 0
    #     query = np.transpose(np.tile(query.astype(np.uint8), (3, 1, 1)), (1, 2, 0))
    #     ax1.imshow(query)
    #     ax1.set_title('query image')
    #     patch_1 = images['patch_1'].numpy()
    #     patch_1[patch_1 > 255] = 255
    #     patch_1[patch_1 < 0] = 0
    #     patch_1 = np.transpose(np.tile(patch_1.astype(np.uint8), (3, 1, 1)), (1, 2, 0))
    #     ax2.imshow(patch_1)
    #     ax2.set_title('patch_1')
    #     patch_2 = images['patch_2'].numpy()
    #     patch_2[patch_2 > 255] = 255
    #     patch_2[patch_2 < 0] = 0
    #     patch_2 = np.transpose(np.tile(patch_2.astype(np.uint8), (3, 1, 1)), (1, 2, 0))
    #     ax3.imshow(patch_2)
    #     ax3.set_title('patch_2')
    #     plt.show()

    ###########################################################################
    # Use dataloader to show transformed image
    ###########################################################################

    #coco_sampler = DatasetSampler(data_source=coco_dataset, batch_size=4, samples_per_epoch=100, mode='single')
    testloader = DataLoader(cifar10_testset, batch_size=4, shuffle=True, num_workers=2)
    for i_batch, sample_batched in enumerate(testloader):

        # Get patches and delta_gt
        sample_idx_in_batch = 0
        image_1 = sample_batched['image_1'][sample_idx_in_batch].numpy().transpose(1, 2, 0)
        image_2 = sample_batched['image_2'][sample_idx_in_batch].numpy().transpose(1, 2, 0)
        patch_1 = sample_batched['patch_1'][sample_idx_in_batch].numpy().transpose(1, 2, 0)
        patch_2 = sample_batched['patch_2'][sample_idx_in_batch].numpy().transpose(1, 2, 0)
        delta_gt = sample_batched['delta'][sample_idx_in_batch].numpy()
        fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(3, 10))

        # Clip patches
        image_1[image_1 > 255] = 255
        image_1[image_1 < 0] = 0
        image_2[image_2 > 255] = 255
        image_2[image_2 < 0] = 0
        patch_1[patch_1 > 255] = 255
        patch_1[patch_1 < 0] = 0
        patch_2[patch_2 > 255] = 255
        patch_2[patch_2 < 0] = 0

        # Image 1
        ax1.imshow(np.tile(image_1.astype(np.uint8), (1, 1, 3)))
        ax1.set_title('image_1')

        # Patch 1
        ax2.imshow(np.tile(patch_1.astype(np.uint8), (1, 1, 3)))
        ax2.set_title('patch_1')

        # Patch 1 warped
        image_corners = sample_batched['corners'][sample_idx_in_batch].numpy()
        homography = four_point_to_homography(np.expand_dims(image_corners, axis=0), np.expand_dims(delta_gt, axis=0),
                                              crop=False)
        image_1_w = warp_image(image_1, homography, target_h=image_1.shape[0], target_w=image_1.shape[1])
        patch_1_w = image_1_w[image_corners[0, 1]:image_corners[3, 1], image_corners[0, 0]:image_corners[1, 0]]
        patch_1_w = np.expand_dims(patch_1_w, axis=-1)
        ax3.imshow(np.tile(patch_1_w.astype(np.uint8), (1, 1, 3)))
        ax3.set_title('patch_1 warped')

        # Image 2
        ax4.imshow(np.tile(image_2.astype(np.uint8), (1, 1, 3)))
        ax4.set_title('image_2')

        # Patch 2
        ax5.imshow(np.tile(patch_2.astype(np.uint8), (1, 1, 3)))
        ax5.set_title('patch_2')

        # Patch 2 warped
        image_corners_2 = image_corners + delta_gt
        delta_gt_2 = -delta_gt
        homography_2 = four_point_to_homography(np.expand_dims(image_corners_2, axis=0),
                                                np.expand_dims(delta_gt_2, axis=0), crop=False)
        image_2_w = warp_image(image_2, homography_2, target_h=image_2.shape[0], target_w=image_2.shape[1])
        patch_2_w = image_2_w[image_corners[0, 1]:image_corners[3, 1], image_corners[0, 0]:image_corners[1, 0]]
        patch_2_w = np.expand_dims(patch_2_w, axis=-1)
        ax6.imshow(np.tile(patch_2_w.astype(np.uint8), (1, 1, 3)))
        ax6.set_title('patch_2 warped')
        plt.show()


if __name__ == "__main__":

    # Fix seed
    np.random.seed(2)

    # Test
    #main_single_scenario()
    main_single_photometric_distort_test()
