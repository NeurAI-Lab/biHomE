import torch
import numpy as np

from src.data.utils import warp_image
from src.data.utils import image_shape_to_corners
from src.data.utils import four_point_to_homography


class Model(torch.nn.Module):

    def __init__(self, backbone, **kwargs):
        super(Model, self).__init__()
        self.backbone = backbone

        self.patch_keys = kwargs['PATCH_KEYS']
        self.mask_keys = kwargs['MASK_KEYS']
        self.feature_keys = kwargs['FEATURE_KEYS']
        self.target_keys = kwargs['TARGET_KEYS']

        self.ld = kwargs['LD']
        self.mu = kwargs['MU']
        assert self.ld == 2, 'Only ld==2 is supported at the moment'

        self.variant = str.lower(kwargs['VARIANT'])
        assert self.variant == 'oneline' or self.variant == 'doubleline', 'Supported variants: OneLine or DoubleLine'
        self.triplet_margin = kwargs['TRIPLET_MARGIN']
        self.triplet_channel_aggregation = kwargs['TRIPLET_AGGREGATION']

    @staticmethod
    def _warp(image, delta_hat, corners=None):
        if corners is None:
            corners = image_shape_to_corners(patch=image)
        homography = four_point_to_homography(corners=corners, deltas=delta_hat, crop=False)
        image_warped = warp_image(image, homography, target_h=image.shape[-2], target_w=image.shape[-1])
        return image_warped, homography

    def forward(self, data):

        # Keys
        e1, e2 = self.patch_keys
        m1, m2 = self.mask_keys
        f1, f2 = self.feature_keys
        o1, o2 = self.target_keys

        # Get first patch data
        patch_1 = data[e1]
        patch_1_m = data[m1]
        patch_1_f = data[f1]

        # Get second patch data
        patch_2 = data[e2]
        patch_2_m = data[m2]
        patch_2_f = data[f2]

        #######################################################################
        # Prepare first patch warped
        #######################################################################

        patch_1_prime, _ = self._warp(patch_1, delta_hat=data[o1])
        patch_1_f_prime = self.backbone.feature_extractor(patch_1_prime)
        patch_1_m_prime, h1 = self._warp(data[m1], delta_hat=data[o1])

        #######################################################################
        # Prepare second patch warped
        #######################################################################

        # Prepare second patch warped
        if self.variant == 'doubleline':
            patch_2_prime, _ = self._warp(patch_2, delta_hat=data[o2])
            patch_2_f_prime = self.backbone.feature_extractor(patch_2_prime)
            patch_2_m_prime, h2 = self._warp(patch_2_m, delta_hat=data[o2])

        #######################################################################
        # Final loss - oneline variant
        #######################################################################

        # Loss elements
        l1 = torch.abs(patch_1_f_prime - patch_2_f)
        l3 = torch.abs(patch_1_f - patch_2_f)

        # Prepare masks
        patch_2_m = torch.squeeze(patch_2_m, dim=1)
        patch_1_m_prime = torch.squeeze(patch_1_m_prime, dim=1)

        ln1_den = torch.sum(torch.sum(patch_1_m_prime * patch_2_m, dim=-1), dim=-1)
        if isinstance(self.triplet_margin, str):
            if self.triplet_channel_aggregation == 'channel-aware':
                loss_mat_1 = torch.sum(l1 - l3, dim=1)
            elif self.triplet_channel_aggregation == 'channel-agnostic':
                loss_mat_1 = torch.sum(l1, dim=1) - torch.sum(l3, dim=1)
            else:
                assert False, 'Do not know this aggregation technique'
        else:
            if self.triplet_channel_aggregation == 'channel-aware':
                loss_mat_1 = torch.sum(torch.max(l1 - l3 + self.triplet_margin, torch.zeros_like(l1)), dim=1)
            elif self.triplet_channel_aggregation == 'channel-agnostic':
                loss_mat_1 = torch.max(torch.sum(l1, dim=1) - torch.sum(l3, dim=1) + self.triplet_margin,
                                       torch.zeros_like(l1))
            else:
                assert False, 'Do not know this aggregation technique'
        ln1 = torch.sum(torch.sum(patch_1_m_prime * patch_2_m * loss_mat_1, dim=-1), dim=-1) / \
              torch.max(ln1_den, torch.ones_like(ln1_den))

        # Sum over batch
        ln1 = torch.sum(ln1)

        # Final loss (will be overwritten if doubline variant)
        loss = ln1

        #######################################################################
        # Final loss - doubleline variant
        #######################################################################

        if self.variant == 'doubleline':

            # Third loss elem
            l1 = torch.abs(patch_1_f_prime - patch_2_f)
            l2 = torch.abs(patch_2_f_prime - patch_1_f)

            # Prepare masks
            patch_1_m = torch.squeeze(patch_1_m, dim=1)
            patch_2_m_prime = torch.squeeze(patch_2_m_prime, dim=1)

            # Second loss elem
            ln2_den = torch.sum(torch.sum(patch_2_m_prime * patch_1_m, dim=-1), dim=-1)
            if isinstance(self.triplet_margin, str):
                if self.triplet_channel_aggregation == 'channel-aware':
                    loss_mat_2 = torch.sum(l2 - l3, dim=1)
                elif self.triplet_channel_aggregation == 'channel-agnostic':
                    loss_mat_2 = torch.sum(l2, dim=1) - torch.sum(l3, dim=1)
                else:
                    assert False, 'Do not know this aggregation technique'
            else:
                if self.triplet_channel_aggregation == 'channel-aware':
                    loss_mat_2 = torch.sum(torch.max(l2 - l3 + self.triplet_margin, torch.zeros_like(l2)), dim=1)
                elif self.triplet_channel_aggregation == 'channel-agnostic':
                    loss_mat_2 = torch.max(torch.sum(l2, dim=1) - torch.sum(l3, dim=1) + self.triplet_margin,
                                           torch.zeros_like(l2))
                else:
                    assert False, 'Do not know this aggregation technique'
            ln2 = torch.sum(torch.sum(patch_2_m_prime * patch_1_m * loss_mat_2, dim=-1), dim=-1) / \
                  torch.max(ln2_den, torch.ones_like(ln2_den))

            # Sum over batch
            ln2 = torch.sum(ln2)

            # Forth loss elem
            batch_size = data[e1].shape[0]
            eye = torch.eye(3, dtype=h1.dtype, device=h1.device).unsqueeze(dim=0).repeat(batch_size, 1, 1)
            ln3 = torch.sum((torch.matmul(h1, h2) - eye) ** 2)

            # Final loss
            loss = ln1 + ln2 + self.mu * ln3

        #######################################################################
        # Tensorboard logs
        #######################################################################

        if 'summary_writer' in data:
            step = data['summary_writer_step']

            # Feature space
            data['summary_writer'].add_scalars('feature_space', {'patch_2_f': torch.mean(patch_2_f).item()}, step)
            data['summary_writer'].add_scalars('feature_space', {'patch_1_f_prime':
                                                                     torch.mean(patch_1_f_prime).item()}, step)
            data['summary_writer'].add_scalars('feature_space', {'patch_1_f': torch.mean(patch_1_f).item()}, step)

            if self.variant == 'doubleline':
                data['summary_writer'].add_scalars('feature_space', {'patch_2_f_prime':
                                                                         torch.mean(patch_2_f_prime).item()}, step)

            # Loss componets
            data['summary_writer'].add_scalars('loss_comp', {
                'l1': torch.mean(torch.abs(patch_2_f - patch_1_f_prime)).item()}, step)
            data['summary_writer'].add_scalars('loss_comp', {
                'l3': torch.mean(torch.abs(patch_1_f - patch_2_f)).item()}, step)
            eye = torch.eye(3, dtype=h1.dtype, device=h1.device).unsqueeze(dim=0).repeat(h1.shape[0], 1, 1)
            data['summary_writer'].add_scalars('h', {'h1': (torch.sum((h1 - eye) ** 2)).item()}, step)
            if self.variant == 'doubleline':
                data['summary_writer'].add_scalars('loss_comp', {
                    'l2': torch.mean(torch.abs(patch_1_f - patch_2_f_prime)).item()}, step)
                data['summary_writer'].add_scalars('loss_comp', {'ln1': ln1.item()}, step)
                data['summary_writer'].add_scalars('loss_comp', {'ln2': ln2.item()}, step)
                data['summary_writer'].add_scalars('loss_comp', {'ln3': self.mu*ln3.item()}, step)
                data['summary_writer'].add_scalars('h', {'h2': (torch.sum((h2 - eye) ** 2)).item()}, step)

        #######################################################################
        # Delta GT
        #######################################################################

        # delta
        delta_gt, delta_hat = None, None
        if 'delta' in data:
            delta_gt = data['delta']
        if self.target_keys[0] in data:
            delta_hat = data[self.target_keys[0]]

        # Return loss
        return loss, delta_gt, delta_hat

    def predict_homography(self, data):

        # Keys
        e1, e2 = self.patch_keys
        o1, o2 = self.target_keys

        # Estimate homography
        delta_hat = data[o1]
        patch_1_prime, homography_hat = self._warp(data[e1], delta_hat=delta_hat)

        # Return
        return delta_hat, homography_hat
