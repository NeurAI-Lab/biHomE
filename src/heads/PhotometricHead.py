import torch
import torch.nn as nn

from src.data.utils import warp_image
from src.data.utils import image_shape_to_corners
from src.data.utils import four_point_to_homography


class Model(nn.Module):

    def __init__(self, backbone, **kwargs):
        super(Model, self).__init__()

        # image_1, image_2, delta_1to2_hat
        self.learning_keys = kwargs['LEARNING_KEYS']

    def forward(self, data):

        # Get corners
        if 'corners' in data:
            corners = data['corners']
        else:
            assert False, 'Check this twice!'
            corners = image_shape_to_corners(patch=data[self.learning_keys[1]])

        # Estimate homography
        delta_hat = data[self.learning_keys[3]]
        homography_hat = four_point_to_homography(corners=corners, deltas=delta_hat, crop=False)

        # Warp the image
        image_warped = warp_image(image=data[self.learning_keys[1]], homography=homography_hat,
                                  target_h=data[self.learning_keys[1]].shape[-2],
                                  target_w=data[self.learning_keys[1]].shape[-1])

        # Get the patch iteratively
        patch_hat = []
        corners = corners.int()
        for idx in range(corners.shape[0]):
            patch = image_warped[idx, :, corners[idx, 0, 1]:corners[idx, 3, 1], corners[idx, 0, 0]:corners[idx, 1, 0]]
            patch_hat.append(patch)
        patch_hat = torch.stack(patch_hat)

        # Return: ground_truth, network_output, delta_gt, delta_hat
        patch_gt = data[self.learning_keys[0]]
        delta_gt = data[self.learning_keys[2]]
        return patch_gt, patch_hat, delta_gt, delta_hat

    def predict_homography(self, data):

        # Get corners
        if 'corners' in data:
            corners = data['corners']
        else:
            assert False, 'How to handle it?'

        # Estimate homography
        delta_hat = data[self.learning_keys[3]]
        homography_hat = four_point_to_homography(corners=corners, deltas=delta_hat, crop=False)

        # Return the patch
        return delta_hat, homography_hat
