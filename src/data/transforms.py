import cv2
import numpy as np
# import porespy as ps

from src.data.utils import warp_image
from src.data.utils import four_point_to_homography

import torch


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, bigger of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, random_seed=None):
        assert isinstance(output_size, (int, tuple, list))
        self.output_size = output_size

    def __call__(self, data):

        images, targets = data
        for i in range(len(images)):

            h, w = images[i].shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                src_ratio = h / w
                target_w, target_h = self.output_size
                if src_ratio < target_h / target_w:
                    new_w, new_h = (int(np.round(target_h / src_ratio)), target_h)
                else:
                    new_w, new_h = (target_w, int(np.round(target_w * src_ratio)))

            new_h, new_w = int(new_h), int(new_w)
            images[i] = cv2.resize(images[i], (new_w, new_h))

        return images, targets


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):

        images, targets = data
        for i in range(len(images)):

            h, w = images[i].shape[:2]
            new_h, new_w = self.output_size

            if h != new_h:
                top = np.random.randint(0, h - new_h)
            else:
                top = 0
            if w != new_w:
                left = np.random.randint(0, w - new_w)
            else:
                left = 0

            images[i] = images[i][top: top + new_h, left: left + new_w]

        return images, targets


class CenterCrop(object):
    """Crop center the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, random_seed=None):
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):

        images, targets = data
        for i in range(len(images)):

            h, w = images[i].shape[:2]
            new_w, new_h = self.output_size

            if h != new_h:
                top = (h - new_h)//2
            else:
                top = 0
            if w != new_w:
                left = (w - new_w)//2
            else:
                left = 0

            images[i] = images[i][top: top + new_h, left: left + new_w]

        return images, targets


class ImageConvertFromInts(object):
    def __call__(self, image):
        return image.astype(np.float32)


class ImageConvertToInts(object):
    def __call__(self, image):
        return np.rint(image).astype(np.uint8)


class ImageCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ImageRandomBrightness(object):
    def __init__(self, max_delta=32, random_state=None):
        assert max_delta >= 0.0
        assert max_delta <= 255.0
        self.delta = max_delta
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            delta = self.random_state.uniform(-self.delta, self.delta)
            image += delta
        return image


class ImageRandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5, random_state=None):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.random_state = random_state

    # expects float image
    def __call__(self, image):
        if self.random_state.randint(2):
            alpha = self.random_state.uniform(self.lower, self.upper)
            image *= alpha
        return image


class ImageConvertColor(object):
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'RGB' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == 'BGR' and self.transform == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == 'HSV' and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image


class ImageRandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5, random_state=None):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            image[:, :, 1] *= self.random_state.uniform(self.lower, self.upper)
        return image


class ImageRandomHue(object):
    def __init__(self, delta=18.0, random_state=None):
        assert 0.0 <= delta <= 360.0
        self.delta = delta
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            image[:, :, 0] += self.random_state.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image


class ImageSwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class ImageRandomLightingNoise(object):
    def __init__(self, random_state):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        self.random_state = random_state

    def __call__(self, image):
        if self.random_state.randint(2):
            swap = self.perms[self.random_state.randint(len(self.perms))]
            shuffle = ImageSwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image


class PhotometricDistort(object):
    def __init__(self, keys, random_state=None):
        self.random_state = random_state
        self.pd = [
            ImageRandomContrast(random_state=self.random_state),  # RGB
            ImageConvertColor(current="RGB", transform='HSV'),  # HSV
            ImageRandomSaturation(random_state=self.random_state),  # HSV
            ImageRandomHue(random_state=self.random_state),  # HSV
            ImageConvertColor(current='HSV', transform='RGB'),  # RGB
            ImageRandomContrast(random_state=self.random_state)  # RGB
        ]
        self.from_int = ImageConvertFromInts()
        self.rand_brightness = ImageRandomBrightness(random_state=self.random_state)
        self.rand_light_noise = ImageRandomLightingNoise(random_state=self.random_state)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            im = data[key].copy()
            im = self.from_int(im)
            im = self.rand_brightness(im)
            if self.random_state.randint(2):
                distort = ImageCompose(self.pd[:-1])
            else:
                distort = ImageCompose(self.pd[1:])
            im = distort(im)
            im = self.rand_light_noise(im)
            data[key] = im
        return data


class PhotometricDistortSimple(object):
    def __init__(self, keys, max_delta=32, random_state=None):
        self.random_state = random_state
        self.max_delta = max_delta

        lower = 1.0 - self.max_delta / 32 * 0.5
        upper = 1.0 + self.max_delta / 32 * 0.5
        self.pd = [
            ImageRandomContrast(lower=lower, upper=upper, random_state=self.random_state),  # RGB
            ImageConvertColor(current="RGB", transform='HSV'),  # HSV
            ImageRandomSaturation(lower=lower, upper=upper, random_state=self.random_state),  # HSV
            ImageRandomHue(delta=max_delta/2, random_state=self.random_state),  # HSV
            ImageConvertColor(current='HSV', transform='RGB'),  # RGB
            ImageRandomContrast(lower=lower, upper=upper, random_state=self.random_state)  # RGB
        ]
        self.from_int = ImageConvertFromInts()
        self.rand_brightness = ImageRandomBrightness(max_delta=max_delta, random_state=self.random_state)
        if max_delta > 0:
            self.rand_light_noise = ImageRandomLightingNoise(random_state=self.random_state)
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            im = data[key].copy()
            im = self.from_int(im)
            im = self.rand_brightness(im)
            if self.random_state.randint(2):
                distort = ImageCompose(self.pd[:-1])
            else:
                distort = ImageCompose(self.pd[1:])
            im = distort(im)
            if self.max_delta > 0:
                im = self.rand_light_noise(im)
            data[key] = im
        return data


class ToGrayscale(object):
    def __call__(self, data):
        images, targets = data
        for i in range(len(images)):
            # RGB 2 GRAY
            images[i] = np.expand_dims(images[i][:, :, 0] * 0.299 +
                                       images[i][:, :, 1] * 0.587 +
                                       images[i][:, :, 2] * 0.114, axis=-1)
        return images, targets


class DictToGrayscale(object):
    def __init__(self, keys, *args):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            # RGB 2 GRAY
            data[key] = np.expand_dims(data[key][:, :, 0] * 0.299 +
                                       data[key][:, :, 1] * 0.587 +
                                       data[key][:, :, 2] * 0.114, axis=-1)
        return data


class Standardize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        images, targets = data
        for i in range(len(images)):
            images[i] = (images[i].astype(np.float32)/255 - self.mean) / self.std
        return images, targets


class DictStandardize(object):
    def __init__(self, mean, std, keys, *args):
        self.mean = mean
        self.std = std
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            data[key] = (data[key].astype(np.float32)/255 - self.mean) / self.std
        return data


class ToTensorWithTarget(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        images, targets = data
        for i in range(len(images)):
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            images[i] = images[i].transpose((2, 0, 1))

        # Targets
        if targets is not None:
            targets = torch.from_numpy(np.array(targets))

        return torch.from_numpy(np.array(images)), targets


class ChangeAwarePrep(object):
    """
    @TODO: Describe it!
    """

    def __init__(self, keys=['image', 'positive', 'weak_positive']):
        self.keys = keys

    def __call__(self, data):

        images, targets = data
        assert len(images) == len(self.keys), 'Something is weid: len(images)={}  len(self.keys)=={}'.format(
            len(images), len(self.keys)
        )

        ret_dict = {}
        for i, k in enumerate(self.keys):
            ret_dict[k] = images[i]

        return ret_dict


class HomographyNetPrep(object):
    """
    Data preparation procedure like in the [1].

    "To  generate  a  single  training  example,  we  first  randomly crop a square patch from the larger image
     I at position p (we avoid  the  borders  to  prevent  bordering  artifacts  later  in  the data  generation
     pipeline). This  random  crop  is Ip.  Then,  the four  corners  of  Patch  A  are  randomly  perturbed  by
     values within  the  range  [-ρ,ρ]. The  four  correspondences  define a  homography HAB.  Then, the  inverse
     of  this  homography HBA= (HAB)−1 is  applied  to the  large  image  to  produce image I′. A second patch I′p
     is cropped from I′ at position p. The two grayscale patches, Ip and I'p are then stacked channelwise to create
     2-channel image which is fed directly to our ConvNet. The 4-point parametrization of HAB is then used as the
     associated ground-truth training label."

    [1] DeTone, D., Malisiewicz, T., & Rabinovich, A. (2016). Deep Image Homography Estimation. ArXiv, abs/1606.03798.

    Args:
        rho (int): point perturbation range.
        patch_size (int): size of patch.
    """

    def __init__(self, rho, patch_size, photometric_distort_keys=None, max_delta=32, target_gen='4_points',
                 random_seed=None):
        self.rho = rho
        self.patch_size = patch_size
        self.target_gen = target_gen
        self.photometric_distort_keys = photometric_distort_keys
        self.max_delta = max_delta
        self.random_seed = random_seed
        if self.random_seed is not None:
            self.random_state = np.random.RandomState(self.random_seed)
            self.randint_fn = self.random_state.randint
        else:
            self.random_state = np.random
            self.randint_fn = np.random.randint

    def __call__(self, data):

        images, targets = data
        assert len(images) == 1, ' HomographyNetPrep transform should be used only with single mode'

        # Get image
        image = images[0]
        h, w = image.shape[:2]

        ###################################################################
        # Should we distort images?
        ###################################################################

        # # VISONLY
        # cv2.imshow('coco_image', cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        # cv2.imwrite('appendix_coco_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        # # VISONLY

        image_1 = np.copy(image)
        if 'image_1' in self.photometric_distort_keys:
            image_1 = PhotometricDistortSimple(keys=['image_1'], max_delta=self.max_delta,
                                               random_state=self.random_state)({'image_1': image_1})['image_1']
        image_2 = np.copy(image)
        if 'image_2' in self.photometric_distort_keys:
            image_2 = PhotometricDistortSimple(keys=['image_2'], max_delta=self.max_delta,
                                               random_state=self.random_state)({'image_2': image_2})['image_2']

        # # VISONLY
        # image_1[image_1 > 255] = 255
        # image_1[image_1 < 0] = 0
        # image_1 = image_1.astype(np.uint8)
        # print(image_1.shape, image_1[0][0])
        # #cv2.imshow('image_1', cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY))
        # cv2.imwrite('appendix_image_1.png', cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY))
        #
        # image_2[image_2 > 255] = 255
        # image_2[image_2 < 0] = 0
        # image_2 = image_2.astype(np.uint8)
        # #cv2.imshow('image_2', cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY))
        # cv2.imwrite('appendix_image_2.png', cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY))
        # # VISONLY
        # exit()

        ###################################################################
        # Find patch
        ###################################################################

        # Calc position of patch center
        if self.patch_size != w:
            pos_x = self.randint_fn(self.rho + self.patch_size // 2, w - self.rho - self.patch_size // 2 + 1)
            pos_y = self.randint_fn(self.rho + self.patch_size // 2, h - self.rho - self.patch_size // 2 + 1)
        else:
            pos_x = w//2
            pos_y = h//2

        # # VISONLY
        # pos_x = w//2 + 20
        # pos_y = h//2 + 25
        # # VISONLY

        # Get patch coords (x/y)
        corners = np.array([(pos_x - self.patch_size // 2, pos_y - self.patch_size // 2),
                            (pos_x + self.patch_size // 2, pos_y - self.patch_size // 2),
                            (pos_x + self.patch_size // 2, pos_y + self.patch_size // 2),
                            (pos_x - self.patch_size // 2, pos_y + self.patch_size // 2)])
        patch_1 = image_1[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]

        # # VISONLY
        # from eval import draw_rect
        # #image_1_with_rect = draw_rect(cv2.UMat(image_1), corners, color='b')
        # image_1_with_rect = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
        # image_1_with_rect = np.tile(np.expand_dims(image_1_with_rect, axis=-1), (1, 1, 3))
        # image_1_with_rect = draw_rect(image_1_with_rect, corners, color='b', thickness=2)
        # cv2.imshow('image_1_with_rect', cv2.cvtColor(image_1_with_rect, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('appendix_image_1_with_rect.png', cv2.cvtColor(image_1_with_rect, cv2.COLOR_RGB2BGR))
        # # VISONLY

        ###################################################################
        # Find second patch
        ###################################################################

        # Randomly distort coordinates
        delta = self.randint_fn(-self.rho, self.rho, 8).reshape(4, 2)
        #delta = self.random_state.choice([-self.rho, self.rho], 8, replace=True).reshape(4, 2)

        # # VISONLY
        # delta = [[-20, -20], [+15, +10], [-8, +4], [+15, +25]]
        # image_2_with_rect = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
        # image_2_with_rect = np.tile(np.expand_dims(image_2_with_rect, axis=-1), (1, 1, 3))
        # image_2_with_rect = draw_rect(image_2_with_rect, corners + delta, color='b', thickness=2)
        # cv2.imshow('image_2_with_rect', cv2.cvtColor(image_2_with_rect, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('appendix_image_2_with_rect.png', cv2.cvtColor(image_2_with_rect, cv2.COLOR_RGB2BGR))
        # # VISONLY

        # Debug
        # delta_1 = np.random.randint(0, self.rho, 2).reshape(1, 2)
        # delta_2 = np.concatenate(
        #     [np.random.randint(-self.rho, 0, 1).reshape(1, 1), np.random.randint(0, self.rho, 1).reshape(1, 1)],
        #     axis=-1)
        # delta_3 = np.random.randint(-self.rho, 0, 2).reshape(1, 2)
        # delta_4 = np.concatenate(
        #     [np.random.randint(0, self.rho, 1).reshape(1, 1), np.random.randint(-self.rho, 0, 1).reshape(1, 1)],
        #     axis=-1)
        # delta = np.concatenate([delta_1, delta_2, delta_3, delta_4], axis=0)
        # delta_1 = np.array([30, 15]).reshape(1, 2)
        # delta_2 = np.array([0, 0]).reshape(1, 2)
        # delta_3 = np.array([-15, -30]).reshape(1, 2)
        # delta_4 = np.array([0, 0]).reshape(1, 2)
        # delta = np.concatenate([delta_1, delta_2, delta_3, delta_4], axis=0)

        #######################################################################
        # Get perspective transform NEW
        #######################################################################

        # Calc homography between
        homography = four_point_to_homography(np.expand_dims(corners, axis=0), np.expand_dims(delta, axis=0),
                                              crop=False)
        image_2 = warp_image(image_2, homography, target_h=image_2.shape[0], target_w=image_2.shape[1])
        if len(image_2.shape) == 2:
            image_2 = np.expand_dims(image_2, axis=-1)
        patch_2 = image_2[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]
        if len(patch_2.shape) == 2:
            patch_2 = np.expand_dims(patch_2, axis=-1)

        # # VISONLY
        # image_2_with_rect_hba = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)
        # image_2_with_rect_hba = np.tile(np.expand_dims(image_2_with_rect_hba, axis=-1), (1, 1, 3))
        # image_2_with_rect_hba = draw_rect(image_2_with_rect_hba, corners, color='b', thickness=2)
        # cv2.imshow('image_2_with_rect_hba', cv2.cvtColor(image_2_with_rect_hba, cv2.COLOR_RGB2BGR))
        # cv2.imwrite('appendix_image_2_with_rect_hba.png', cv2.cvtColor(image_2_with_rect_hba, cv2.COLOR_RGB2BGR))
        #
        #
        # cv2.imshow('patch_1', cv2.cvtColor(patch_1, cv2.COLOR_RGB2GRAY))
        # cv2.imwrite('appendix_patch_1.png', cv2.cvtColor(patch_1, cv2.COLOR_RGB2GRAY))
        # cv2.imshow('patch_2', cv2.cvtColor(patch_2, cv2.COLOR_RGB2GRAY))
        # cv2.imwrite('appendix_patch_2.png', cv2.cvtColor(patch_2, cv2.COLOR_RGB2GRAY))
        #
        # cv2.waitKey()
        # exit()
        # # VISONLY

        ###################################################################
        # How to reconstruct patch_2, having only patch_1
        ###################################################################

        # image_width, image_height = patch_1.shape[:2]
        # corners = np.float32([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]])
        # corners = np.tile(np.expand_dims(corners, axis=0), (1, 1, 1))
        # homography = four_point_to_homography(corners, np.expand_dims(delta, axis=0), crop=True)
        # patch_2_ = warp_image(patch_1, homography, target_h=image_2.shape[0], target_w=image_2.shape[1])
        # patch_2_ = patch_2_[:patch_1.shape[0], :patch_1.shape[1]]
        # cv2.imshow('image_1', image_1.astype(np.uint8))
        # cv2.imshow('patch_1', patch_1.astype(np.uint8))
        # cv2.imshow('image_2', image_2.astype(np.uint8))
        # cv2.imshow('patch_2', patch_2.astype(np.uint8))
        # cv2.imshow('patch_2_', patch_2_.astype(np.uint8))
        # cv2.waitKey()
        # exit()

        ###################################################################
        # Should we distort patches?
        ###################################################################

        # if 'patch_1' in self.photometric_distort_keys:
        #     patch_1 = PhotometricDistort(keys=['patch_1'])({'patch_1': patch_1}, self.random_seed)['patch_1']
        # if 'patch_2' in self.photometric_distort_keys:
        #     patch_2 = PhotometricDistort(keys=['patch_2'])({'patch_2': patch_2}, self.random_seed)['patch_2']

        ###################################################################
        # Prepare output data - 4 points
        ###################################################################

        if self.target_gen == '4_points':
            target = delta

        ###################################################################
        # Prepare output data - all points
        ###################################################################

        elif self.target_gen == 'all_points':

            # Create grid of targets
            y_grid, x_grid = np.mgrid[0:h, 0:w]
            point_grid = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

            # Transform grid o points
            point_grid_t = cv2.perspectiveTransform(np.array([point_grid], dtype=np.float32), homography).squeeze()
            diff_grid_t = point_grid_t - point_grid
            diff_x_grid_t = diff_grid_t[:, 0]
            diff_y_grid_t = diff_grid_t[:, 1]
            diff_x_grid_t = diff_x_grid_t.reshape((h, w))
            diff_y_grid_t = diff_y_grid_t.reshape((h, w))

            pf_patch_x_branch = diff_x_grid_t[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]
            pf_patch_y_branch = diff_y_grid_t[corners[0, 1]:corners[3, 1], corners[0, 0]:corners[1, 0]]


            # Debug
            # print('p1', diff_x_grid_t[corners[0, 1], corners[0, 0]], diff_y_grid_t[corners[0, 1], corners[0, 0]])
            # print('p3', diff_x_grid_t[corners[2, 1], corners[2, 0]], diff_y_grid_t[corners[2, 1], corners[2, 0]])
            # print(pf_patch_x_branch[0, 0], pf_patch_y_branch[0, 0])
            #
            # target = np.tile(image_1.astype(np.uint8), (1, 1, 3))
            # for h, w in zip([0, 0, target.shape[0] - 1, target.shape[0] - 1, corners[0, 1]],
            #                 [0, target.shape[1] - 1, 0, target.shape[1] - 1, corners[0, 0]]):
            #     x = diff_x_grid_t[h, w].astype(int)
            #     y = diff_y_grid_t[h, w].astype(int)
            #     print('x: {} y: {}'.format(x, y))
            #     cv2.arrowedLine(target, (w, h), (w + x, h + y), color=(255, 0, 0), thickness=1)
            #
            # import matplotlib.pyplot as plt
            # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
            #
            # ax1.imshow(np.tile(patch_1.astype(np.uint8), (1, 1, 3)))
            # ax1.set_title('patch-1')
            #
            # patch_2 = np.tile(patch_2.astype(np.uint8), (1, 1, 3))
            # x = diff_x_grid_t[corners[0, 0], corners[0, 1]].astype(int)
            # y = diff_y_grid_t[corners[0, 0], corners[0, 1]].astype(int)
            # cv2.arrowedLine(patch_2, (0, 0), (0 + x, 0 + y), color=(255, 0, 0), thickness=1)
            # ax2.imshow(patch_2)
            # ax2.set_title('patch-2')
            # ax3.imshow(target)
            # ax3.set_title('pf')
            # plt.show()
            # exit(0)

            target = np.zeros((self.patch_size, self.patch_size, 2))
            target[:, :, 0] = pf_patch_x_branch
            target[:, :, 1] = pf_patch_y_branch

        else:
            assert False, 'I do not know this, it should be either \'4_points\' ar \'all_points\''

        ###################################################################
        # Debug display
        ###################################################################

        # # Display
        # i1 = np.copy(image)
        # i1 = cv2.rectangle(i1, tuple(corners[0]), tuple(corners[2]), (255, 0, 0),
        #                    thickness=15)
        # cv2.imshow('i1', i1)
        #
        # i1d = np.copy(image)
        # i1d = cv2.line(i1d, tuple(patch_coords_clock_wise_distorted[0]),
        #                tuple(patch_coords_clock_wise_distorted[1]),
        #                (255, 0, 0), thickness=5)
        # i1d = cv2.line(i1d, tuple(patch_coords_clock_wise_distorted[1]), tuple(patch_coords_clock_wise_distorted[2]),
        #                (255, 0, 0), thickness=5)
        # i1d = cv2.line(i1d, tuple(patch_coords_clock_wise_distorted[2]), tuple(patch_coords_clock_wise_distorted[3]),
        #                (255, 0, 0), thickness=5)
        # i1d = cv2.line(i1d, tuple(patch_coords_clock_wise_distorted[3]), tuple(patch_coords_clock_wise_distorted[0]),
        #                (255, 0, 0), thickness=5)
        # cv2.imshow('i1d', i1d)
        # cv2.imshow('p1', patch_1)
        #
        # i3 = np.copy(warped_image)
        # cv2.imshow('i3', i3)
        # cv2.imshow('p2', patch_2)
        #
        # if self.target_gen == 'all_points':
        #     cv2.imshow('target_x', (target[:, :, 0] + self.rho)/self.rho/2)
        #     cv2.imshow('target_y', (target[:, :, 1] + self.rho)/self.rho/2)
        #
        # cv2.waitKey(0)
        # exit()

        return {'image_1': image_1, 'image_2': image_2, 'patch_1': patch_1, 'patch_2': patch_2, 'corners': corners,
                'target': target, 'delta': delta, 'homography': homography}


class DictToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, keys=['image', 'positive', 'weak_positive'], *args):
        self.keys = keys

    def __call__(self, data):
        for key in data:
            if key in self.keys:
                if len(data[key].shape) == 3:
                    # swap color axis because
                    # numpy image: H x W x C
                    # torch image: C X H X W
                    data[key] = data[key].transpose((2, 0, 1))
            data[key] = torch.from_numpy(np.array(data[key]))
        return data


class CollatorWithBlobs(object):

    def __init__(self, patch_1_key=None, patch_2_key=None, blob_porosity=None, blobiness=None, random_seed=None):
        self.patch_1_key = patch_1_key
        self.patch_2_key = patch_2_key
        self.blob_porosity = blob_porosity
        self.blobiness = blobiness
        self.random_seed = random_seed
        if self.random_seed is not None:
            self.random_state = np.random.RandomState(self.random_seed)
            self.rand_choice_fn = self.random_state.choice
        else:
            self.rand_choice_fn = np.random.choice

    def __call__(self, batch):

        ###################################################################
        # Collate
        ###################################################################

        keys = list(batch[0].keys())
        output_dict = {key: [] for key in keys}
        for elem in batch:
            for key in keys:
                output_dict[key].append(elem[key])
        for key in keys:
            output_dict[key] = torch.stack(output_dict[key])

        ###################################################################
        # Generate blobs
        ###################################################################
        if self.patch_1_key is not None:

            h, w = output_dict[self.patch_1_key].shape[-2:]
            for elem_idx in range(len(batch)):
                # Pick image to copy content from
                possible_indices = np.arange(len(batch))
                possible_indices = np.delete(possible_indices, np.where(possible_indices == elem_idx))
                other_index = self.rand_choice_fn(possible_indices, 1)[0]

                # Create blob
                blobs = ps.generators.blobs(shape=[h, w], porosity=self.blob_porosity, blobiness=self.blobiness)
                blobs = torch.from_numpy(blobs)

                # Copy
                patch_1 = output_dict[self.patch_1_key][other_index]
                patch_2 = output_dict[self.patch_2_key][elem_idx]
                blobs = blobs.unsqueeze(0).repeat((patch_2.shape[0], 1, 1))
                patch_2_new = torch.mul(patch_1, blobs)
                patch_2_old = torch.mul(patch_2, torch.bitwise_not(blobs))
                patch_2_augmented = patch_2_old + patch_2_new
                output_dict[self.patch_2_key][elem_idx] = patch_2_augmented

        return output_dict
