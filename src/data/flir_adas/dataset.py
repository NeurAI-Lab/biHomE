import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.data.utils import warp_image
from src.data.utils import four_point_to_homography


class Dataset(Dataset):
    """Class loading RTK and image timestamps data for the whole Oxford dataset"""

    def __init__(self, dataset_root, transforms=None):

        """
        COCO dataset class.

        Args:
            dataset_root (string): Path to the root of the COCO images.
            transforms (list of callables): What transforms apply to the images?
        """

        self.dataset_root = dataset_root
        self.transforms = transforms
        self.img_filenames = [f for f in os.listdir(self.dataset_root) if '.jpeg' in f or '.npy' in f]
        self.img_filepaths = [os.path.join(self.dataset_root, f) for f in self.img_filenames]

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
        return len(self.img_filenames)

    def __getitem__(self, indices):
        # Read images
        images = []
        for idx in indices:
            img = self.load_image(idx)
        images.append(img)

        # Transforms
        if self.transforms:
            data = self.transforms((images, None))

        return data

    def load_image(self, idx):
        filepath = self.img_filepaths[idx]
        if '.jpeg' in filepath:
            img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        elif '.npy' in filepath:
            img = np.load(filepath, allow_pickle=True)
        else:
            assert False, 'I dont know this format'
        return img


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
    dataset_root = '/data/input/datasets/FLIR_ADAS_1_3/train/thermal_8_bit'

    ###########################################################################
    # Create dataset and transforms
    ###########################################################################

    # Transforms
    patch_size = 128
    from src.data.transforms import Rescale, CenterCrop, ToGrayscale, HomographyNetPrep, DictToTensor
    composed_transforms = transforms.Compose([Rescale((320, 256)), CenterCrop((320, 256)),
                                              ToGrayscale(),
                                              HomographyNetPrep(rho=patch_size//4, patch_size=patch_size,
                                                                photometric_distort_keys=[], target_gen='4_points'),
                                              DictToTensor(['patch_1', 'patch_2', 'image_1', 'image_2'])])

    # Load the dataset
    start = timer()
    coco_dataset = Dataset(dataset_root=dataset_root, transforms=composed_transforms)
    end = timer()
    print('Dataset created in {} seconds'.format(end - start))

    ###########################################################################
    # Using iteration
    ###########################################################################

    for images in coco_dataset:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
        query = images['image_1'].numpy()
        query = np.transpose(np.tile(query.astype(np.uint8), (3, 1, 1)), (1, 2, 0))
        ax1.imshow(query)
        ax1.set_title('query image')
        patch_1 = images['patch_1'].numpy()
        patch_1[patch_1 > 255] = 255
        patch_1[patch_1 < 0] = 0
        patch_1 = np.transpose(np.tile(patch_1.astype(np.uint8), (3, 1, 1)), (1, 2, 0))
        ax2.imshow(patch_1)
        ax2.set_title('patch_1')
        patch_2 = images['patch_2'].numpy()
        patch_2[patch_2 > 255] = 255
        patch_2[patch_2 < 0] = 0
        patch_2 = np.transpose(np.tile(patch_2.astype(np.uint8), (3, 1, 1)), (1, 2, 0))
        ax3.imshow(patch_2)
        ax3.set_title('patch_2')
        plt.show()

    ###########################################################################
    # Use dataloader
    ###########################################################################

    coco_sampler = DatasetSampler(data_source=coco_dataset, batch_size=4, samples_per_epoch=100, mode='single')
    dataloader = DataLoader(coco_dataset, batch_sampler=coco_sampler, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):

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
    dataset_root = '/data/input/datasets/FLIR_ADAS_1_3/train/thermal_8_bit'

    ###########################################################################
    # Create dataset and transforms
    ###########################################################################

    # Transforms
    patch_size = 128
    from src.data.transforms import Rescale, CenterCrop, ToGrayscale, HomographyNetPrep, DictToGrayscale, DictToTensor
    composed_transforms = transforms.Compose([Rescale((320, 256)), CenterCrop((320, 256)),
                                              HomographyNetPrep(rho=patch_size//4, patch_size=patch_size,
                                                                photometric_distort_keys=['image_1', 'image_2'],
                                                                target_gen='4_points'),
                                              DictToGrayscale(keys=['image_1', 'image_2', 'patch_1', 'patch_2']),
                                              # DictStandardize(0.443, 0.129, keys=['patch_1', 'patch_2']),
                                              DictToTensor(keys=['image_1', 'image_2', 'patch_1', 'patch_2'])])

    # Load the dataset
    start = timer()

    coco_dataset = Dataset(dataset_root=dataset_root, transforms=composed_transforms)
    end = timer()
    print('Dataset created in {} seconds'.format(end - start))

    ###########################################################################
    # Using iteration
    ###########################################################################

    # for images in coco_dataset:
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

    coco_sampler = DatasetSampler(data_source=coco_dataset, batch_size=4, samples_per_epoch=100, mode='single')
    dataloader = DataLoader(coco_dataset, batch_sampler=coco_sampler, num_workers=1)
    for i_batch, sample_batched in enumerate(dataloader):

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

    #main_single_scenario()
    main_single_photometric_distort_test()
