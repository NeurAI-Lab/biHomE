import numpy as np

import cv2
import torch
import torch.nn as nn

from src.data.utils import four_point_to_homography


class Model(nn.Module):

    def __init__(self, backbone, **kwargs):
        super(Model, self).__init__()

        # Target gen: '4_points' or 'all_points'
        self.target_gen = kwargs['TARGET_GEN']

        # ground_truth, network_output, delta_gt, delta_hat
        self.learning_keys = kwargs['LEARNING_KEYS']

    def forward(self, data):
        ret = []

        # First 3 elements could be fetched directly
        for key in self.learning_keys[:-1]:
            ret.append(data[key])

        # Last element could require transformation
        if self.target_gen == '4_points':
            ret.append(data[self.learning_keys[-1]])
        elif self.target_gen == 'all_points':

            target_hat = data[self.learning_keys[-1]]
            delta_hat = torch.zeros((target_hat.shape[0], 4, 2), device=target_hat.device, dtype=target_hat.dtype)
            h, w = target_hat.shape[-2:]

            delta_hat[:, 0, 0] = target_hat[:, 0, 0, 0]             # top left x
            delta_hat[:, 0, 1] = target_hat[:, 1, 0, 0]             # top left y

            delta_hat[:, 1, 0] = target_hat[:, 0, 0, w - 1]         # top right x
            delta_hat[:, 1, 1] = target_hat[:, 1, 0, w - 1]         # top right y

            delta_hat[:, 2, 0] = target_hat[:, 0, h - 1, w - 1]     # bottom right x
            delta_hat[:, 2, 1] = target_hat[:, 1, h - 1, w - 1]     # bottom right y

            delta_hat[:, 3, 0] = target_hat[:, 0, h - 1, 0]         # bottom left x
            delta_hat[:, 3, 1] = target_hat[:, 1, h - 1, 0]         # bottom left y

            ret.append(delta_hat)

        else:
            assert False, 'I didnt understand that!'

        return ret

    def predict_homography(self, data):
        if self.target_gen == '4_points':

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

        elif self.target_gen == 'all_points':
            return self._postprocess(data[self.learning_keys[1]])

    @staticmethod
    def _postprocess(perspective_field):

        # Move data to numpy?
        if torch.is_tensor(perspective_field):
            perspective_field = perspective_field.cpu().detach().numpy()

        # Create field of the coordinates
        y_patch_grid, x_patch_grid = np.mgrid[0:perspective_field.shape[-2], 0:perspective_field.shape[-1]]
        x_patch_grid = np.tile(x_patch_grid.reshape(1, -1), (perspective_field.shape[0], 1))
        y_patch_grid = np.tile(y_patch_grid.reshape(1, -1), (perspective_field.shape[0], 1))
        coordinate_field = np.stack((x_patch_grid, y_patch_grid), axis=1).transpose(0, 2, 1)

        # Create prediction field
        pf_x1_img_coord, pf_y1_img_coord = np.split(perspective_field, 2, axis=1)
        pf_x1_img_coord = pf_x1_img_coord.reshape(perspective_field.shape[0], -1)
        pf_y1_img_coord = pf_y1_img_coord.reshape(perspective_field.shape[0], -1)
        mapping_field = coordinate_field + np.stack((pf_x1_img_coord, pf_y1_img_coord), axis=1).transpose(0, 2, 1)

        # Find best homography fit and its delta
        predicted_h = []
        predicted_delta = []
        four_points = [[0, 0], [perspective_field.shape[-1], 0], [perspective_field.shape[-1],
                                                                  perspective_field.shape[-2]],
                       [0, perspective_field.shape[-2]]]
        for i in range(perspective_field.shape[0]):
            h = cv2.findHomography(np.float32(coordinate_field[i]), np.float32(mapping_field[i]), cv2.RANSAC, 10)[0]
            delta = cv2.perspectiveTransform(np.asarray([four_points], dtype=np.float32), h).squeeze() - four_points
            predicted_h.append(h)
            predicted_delta.append(delta)
        predicted_h = np.array(predicted_h)
        predicted_delta = np.array(predicted_delta)

        # Find delta
        return predicted_delta, predicted_h
