import numpy as np

import torch
import kornia
import torch.nn as nn
import torchvision.models as models

from src.data.utils import warp_image
from src.data.utils import image_shape_to_corners
from src.data.utils import four_point_to_homography

from src.heads.ransac_utils import DSACSoftmax


class AuxiliaryResnet(nn.Module):

    def __init__(self, **kwargs):
        super(AuxiliaryResnet, self).__init__()

        # Define resnet model
        resnet_fn = getattr(models, kwargs['AUXILIARY_RESNET'])
        self.resnet = resnet_fn(pretrained=True, progress=True)

        # Clear unnecessary layers
        self.auxiliary_resnet_output_layer = kwargs['AUXILIARY_RESNET_OUTPUT_LAYER']
        if self.auxiliary_resnet_output_layer < 2:
            self.resnet.layer2 = torch.nn.Identity()
        if self.auxiliary_resnet_output_layer < 3:
            self.resnet.layer3 = torch.nn.Identity()
        if self.auxiliary_resnet_output_layer < 4:
            self.resnet.layer4 = torch.nn.Identity()
        self.resnet.avgpool = torch.nn.Identity()
        self.resnet.fc = torch.nn.Identity()

        # Freeze the model
        self.freeze = kwargs['AUXILIARY_RESNET_FREEZE'] if 'AUXILIARY_RESNET_FREEZE' in kwargs else True
        if self.freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Add projection head
        self.with_projection_head = kwargs['WITH_PROJECTION_HEAD'] if 'WITH_PROJECTION_HEAD' in kwargs else None
        self.projection_head = nn.ModuleList()
        if self.with_projection_head is not None:
            for idx, layer in enumerate(self.with_projection_head):
                self.projection_head.append(torch.nn.Linear(layer[0], layer[1]))
                if idx != len(self.with_projection_head) - 1:
                    self.projection_head.append(torch.nn.ReLU())

    def forward(self, x):

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)

        if self.auxiliary_resnet_output_layer > 1:
            x = self.resnet.layer2(x)
        if self.auxiliary_resnet_output_layer > 2:
            x = self.resnet.layer3(x)
        if self.auxiliary_resnet_output_layer > 3:
            x = self.resnet.layer4(x)

        # Projection head
        if self.with_projection_head is not None:
            x = x.permute(0, 2, 3, 1)
            for layer in self.projection_head:
                x = layer(x)
            x = x.permute(0, 3, 1, 2)

        return x


class Model(nn.Module):

    def __init__(self, backbone, **kwargs):
        super(Model, self).__init__()
        self.backbone = backbone

        # Patch keys
        self.four_points_12 = None
        self.four_points_21 = None
        self.patch_size = kwargs['PATCH_SIZE']
        self.patch_keys = kwargs['PATCH_KEYS']
        self.delta_hat_keys = kwargs['DELTA_HAT_KEYS']

        # No DSAC needed
        if len(self.delta_hat_keys):
            self.hypothesis_no = 1

        # DSAC if required
        else:
            self.coordinate_field_12 = None
            self.coordinate_field_21 = None
            self.pf_keys = kwargs['PF_KEYS']
            self.hypothesis_no = kwargs['RANSAC_HYPOTHESIS_NO']
            self.point_per_hypothesis = kwargs['POINTS_PER_HYPOTHESIS']
            self.dsac = DSACSoftmax(**kwargs)

        # Triplet version
        self.triplet_version = kwargs['TRIPLET_LOSS']
        if self.triplet_version != '':
            self.mask_keys = kwargs['MASK_KEYS']
            self.change_detection_mask = kwargs['MASK_CRD'] if 'MASK_CRD' in kwargs else False
            self.triplet_margin = kwargs['TRIPLET_MARGIN']
            self.triplet_channel_aggregation = kwargs['TRIPLET_AGGREGATION']
            self.sampling_strategy = kwargs['SAMPLING_STRATEGY']
            self.triplet_distance = kwargs['TRIPLET_DISTANCE']
            if 'one-line' in self.triplet_version:
                self.triplet_loss = torch.nn.TripletMarginLoss(margin=self.triplet_margin, p=1, reduction='none')
            elif 'double-line' in self.triplet_version:
                self.triplet_mu = kwargs['TRIPLET_MU']

        #######################################################################
        # Auxiliary resnet
        #######################################################################

        self.auxiliary_resnet = AuxiliaryResnet(**kwargs)

    def forward_map_field(self, perspective_field, self_coordinate_field, self_four_points):

        #######################################################################
        # Create field of the coordinates if needed
        #######################################################################

        batch_size = perspective_field.shape[0]
        pf_predicted_size = (batch_size, perspective_field.shape[-2] * perspective_field.shape[-1], 2)
        if self_coordinate_field is None or self_coordinate_field.shape != pf_predicted_size:
            y_patch_grid, x_patch_grid = np.mgrid[0:perspective_field.shape[-2], 0:perspective_field.shape[-1]]
            x_patch_grid = np.tile(x_patch_grid.reshape(1, -1), (batch_size, 1))
            y_patch_grid = np.tile(y_patch_grid.reshape(1, -1), (batch_size, 1))
            coordinate_field = np.stack((x_patch_grid, y_patch_grid), axis=1).transpose(0, 2, 1)
            self_coordinate_field = torch.from_numpy(coordinate_field).float().to(perspective_field.device)
            four_points = np.array([[0, 0], [perspective_field.shape[-1], 0],
                                    [perspective_field.shape[-1], perspective_field.shape[-2]],
                                    [0, perspective_field.shape[-2]]])
            four_points = torch.from_numpy(four_points).float().to(perspective_field.device)
            self_four_points = torch.unsqueeze(four_points, dim=0).repeat(batch_size*self.hypothesis_no, 1, 1)

        perspective_field = perspective_field.reshape(batch_size, 2, -1).permute(0, 2, 1)
        return self_coordinate_field + perspective_field, self_coordinate_field, self_four_points

    def forward(self, data):

        #######################################################################
        # DSAC is required
        #######################################################################

        if not len(self.delta_hat_keys):

            #######################################################################
            # Prepare data
            #######################################################################

            # Fetch perspective field data
            perspective_field_12 = data[self.pf_keys[0]]

            # Create coord field
            map_field_12, self.coordinate_field_12, self.four_points_12 = self.forward_map_field(
                perspective_field_12, self.coordinate_field_12, self.four_points_12)

            #######################################################################
            # DSAC with SoftMax
            #######################################################################

            batch_size = perspective_field_12.shape[0]
            homography_hats_12, homography_scores_12 = self.dsac(self.coordinate_field_12, map_field_12,
                                                                 hypothesis_no=self.hypothesis_no,
                                                                 points_per_hypothesis=self.point_per_hypothesis)
            four_points_transformed_12 = kornia.transform_points(homography_hats_12.reshape(-1, 3, 3),
                                                                 self.four_points_12)
            delta_hats_12 = (four_points_transformed_12 - self.four_points_12).reshape(batch_size, self.hypothesis_no,
                                                                                       4, 2)

            #######################################################################
            # Doubleline
            #######################################################################

            if 'double-line' in self.triplet_version:

                # Fetch perspective field data
                perspective_field_21 = data[self.pf_keys[1]]

                # Create coord field
                map_field_21, self.coordinate_field_21, self.four_points_21 = self.forward_map_field(
                    perspective_field_21, self.coordinate_field_21, self.four_points_21)

                #######################################################################
                # DSAC with SoftMax
                #######################################################################

                batch_size = perspective_field_21.shape[0]
                homography_hats_21, homography_scores_21 = self.dsac(self.coordinate_field_21, map_field_21,
                                                                     hypothesis_no=self.hypothesis_no,
                                                                     points_per_hypothesis=self.point_per_hypothesis)
                four_points_transformed_21 = kornia.transform_points(homography_hats_21.reshape(-1, 3, 3),
                                                                     self.four_points_21)
                delta_hats_21 = (four_points_transformed_21 - self.four_points_21).reshape(batch_size,
                                                                                           self.hypothesis_no,
                                                                                           4, 2)

        #######################################################################
        # No DSAC needed
        #######################################################################

        else:

            # Get delta_hats
            delta_hats_12 = data[self.delta_hat_keys[0]]
            homography_scores_12 = None

            if 'double-line' in self.triplet_version:
                delta_hats_21 = data[self.delta_hat_keys[1]]

        #######################################################################
        # Triplet loss
        #######################################################################

        # Triplet loss
        if 'one-line' in self.triplet_version:
            return self.triplet_resnet_loss(data, delta_hats_12, scores=homography_scores_12)
        elif 'double-line' in self.triplet_version:
            return self.triplet_resnet_loss(data, delta_hats_12, delta_hats_21=delta_hats_21)

        #######################################################################
        # Multihead loss
        #######################################################################

        else:
            return self.multihead_resnet_loss(data, delta_hats_12, scores=homography_scores_12)

    @staticmethod
    def _warp(image, delta_hat, corners=None):
        if corners is None:
            corners = image_shape_to_corners(patch=image)
        homography = four_point_to_homography(corners=corners, deltas=delta_hat, crop=False)
        image_warped = warp_image(image, homography, target_h=image.shape[-2], target_w=image.shape[-1])
        return image_warped, homography

    def multihead_resnet_loss(self, data, delta_hats, scores=None):

        #######################################################################
        # Fetch keys and data
        #######################################################################

        e1, e2 = self.patch_keys
        patch_1 = data[e1]
        patch_2 = data[e2]

        #######################################################################
        # Prepare first patch warped
        #######################################################################

        # for every hypothesis
        b = delta_hats.shape[0]
        n = self.hypothesis_no
        i = self.patch_size

        # Repeat patch_1
        patch_1 = patch_1.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)        # [B*N, 1, 128, 128]

        # Extract resnet features
        patch_2 = patch_2.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)        # [B*N, 1, 128, 128]
        patch_2_f = self.auxiliary_resnet(patch_2)                                              # [B*N, C, H, W]

        delta_hats = delta_hats.reshape(b * n, 4, 2)                                            # [B*N, 4, 2]
        patch_1_prime, h1 = self._warp(patch_1, delta_hat=delta_hats)                           # [B*N, 1, 128, 128]
        patch_1_f_prime = self.auxiliary_resnet(patch_1_prime)                                  # [B*N, C, H, W]

        # Add scores
        _, f_c, f_h, f_w = patch_1_f_prime.shape
        if scores is not None:
            scores_f = scores.reshape(b*n, 1, 1, 1).repeat(1, f_c, f_h, f_w)                    # [B*N, C, H, W]
            patch_1_f_prime = patch_1_f_prime * scores_f                                        # [B*N, C, H, W]
            patch_2_f = patch_2_f * scores_f                                                    # [B*N, C, H, W]

        #######################################################################
        # Tensorboard logs
        #######################################################################

        if 'summary_writer' in data:
            step = data['summary_writer_step']

            # Feature space
            data['summary_writer'].add_scalars('feature_space', {'patch_2_f': torch.mean(patch_2_f).item()}, step)
            data['summary_writer'].add_scalars('feature_space', {'patch_1_f_prime': torch.mean(patch_1_f_prime).item()},
                                               step)

            # Loss componets
            data['summary_writer'].add_scalars('loss_comp', {
                'l1': torch.mean(torch.abs(patch_2_f - patch_1_f_prime)).item()}, step)
            eye = torch.eye(3, dtype=h1.dtype, device=h1.device).unsqueeze(dim=0).repeat(h1.shape[0], 1, 1)
            data['summary_writer'].add_scalars('h', {'h1': (torch.sum((h1 - eye) ** 2)).item()}, step)

        #######################################################################
        # Delta GT
        #######################################################################

        # delta
        delta_gt, delta_hat = None, None
        if 'delta' in data:
            delta_gt = data['delta']

        # Calc average of delta_hat
        if scores is not None:
            delta_hats = delta_hats * scores.reshape(b * n, 1, 1).repeat(1, 4, 2)   # [B*N, 4, 2]
            delta_hats = torch.sum(delta_hats.reshape(b, n, 4, 2), dim=1)           # [B, 4, 2]

        # Return loss: ground_truth, original_non_patched_image, delta_gt, delta_hat
        return patch_2_f, patch_1_f_prime, delta_gt, delta_hats

    def __upsample(self, img, scale_factor):
        return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)(img)

    def triplet_resnet_loss(self, data, delta_hats, delta_hats_21=None, scores=None):

        assert (delta_hats_21 is not None and scores is None) or (delta_hats_21 is None and scores is not None) or\
               (delta_hats_21 is None and scores is None), \
            'They should not be on at the same time - at least its not implemented yet'

        #######################################################################
        # Fetch keys and data
        #######################################################################

        e1, e2 = self.patch_keys
        patch_1 = data[e1]
        patch_2 = data[e2]

        if len(self.mask_keys):
            m1, m2 = self.mask_keys
            patch_1_m = data[m1]
            patch_2_m = data[m2]
        else:
            patch_1_m = torch.ones_like(patch_1)
            patch_2_m = torch.ones_like(patch_2)

        #######################################################################
        # Prepare first patch warped
        #######################################################################

        # for every hypothesis
        b = delta_hats.shape[0]
        n = self.hypothesis_no
        i = self.patch_size

        # Repeat patch_1 and extract features
        patch_1 = patch_1.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)        # [B*N, 1, 128, 128]
        if self.sampling_strategy == 'upsample-patch-4x':
            patch_1_f = self.auxiliary_resnet(self.__upsample(patch_1, scale_factor=4))
        elif self.sampling_strategy == 'upsample-patch-2x':
            patch_1_f = self.auxiliary_resnet(self.__upsample(patch_1, scale_factor=2))
        else:
            patch_1_f = self.auxiliary_resnet(patch_1)  # [B*N, C, H, W]

        # Repeat patch 2 and extract features
        patch_2 = patch_2.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)        # [B*N, 1, 128, 128]
        if self.sampling_strategy == 'upsample-patch-4x':
            patch_2_f = self.auxiliary_resnet(self.__upsample(patch_2, scale_factor=4))
        elif self.sampling_strategy == 'upsample-patch-2x':
            patch_2_f = self.auxiliary_resnet(self.__upsample(patch_2, scale_factor=2))
        else:
            patch_2_f = self.auxiliary_resnet(patch_2)                                              # [B*N, C, H, W]

        # Warp patch 1 and extract features
        delta_hats = delta_hats.reshape(b * n, 4, 2)  # [B*N, 4, 2]
        patch_1_prime, h1 = self._warp(patch_1, delta_hat=delta_hats)                           # [B*N, 1, 128, 128]
        if self.sampling_strategy == 'upsample-patch-4x':
            patch_1_f_prime = self.auxiliary_resnet(self.__upsample(patch_1_prime, scale_factor=4))
        elif self.sampling_strategy == 'upsample-patch-2x':
            patch_1_f_prime = self.auxiliary_resnet(self.__upsample(patch_1_prime, scale_factor=2))
        else:
            patch_1_f_prime = self.auxiliary_resnet(patch_1_prime)  # [B*N, C, H, W]

        # Repeat mask_1 and warp it
        patch_1_m = patch_1_m.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)    # [B*N, 1, 128, 128]
        patch_2_m = patch_2_m.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)    # [B*N, 1, 128, 128]
        patch_1_m_prime, h1 = self._warp(patch_1_m, delta_hat=delta_hats)                       # [B*N, 1, 128, 128]

        #######################################################################
        # Prepare second patch warped
        #######################################################################

        if 'double-line' in self.triplet_version:

            # Warp patch 2 and extract features
            delta_hats_21 = delta_hats_21.reshape(b * n, 4, 2)                                  # [B*N, 4, 2]
            patch_2_prime, _ = self._warp(patch_2, delta_hat=delta_hats_21)
            if self.sampling_strategy == 'upsample-patch-4x':
                patch_2_f_prime = self.auxiliary_resnet(self.__upsample(patch_2_prime, scale_factor=4))
            if self.sampling_strategy == 'upsample-patch-2x':
                patch_2_f_prime = self.auxiliary_resnet(self.__upsample(patch_2_prime, scale_factor=2))
            else:
                patch_2_f_prime = self.auxiliary_resnet(patch_2_prime)

            # Warp mask 2
            patch_2_m_prime, h2 = self._warp(patch_2_m, delta_hat=delta_hats_21)

        #######################################################################
        # Old loss to be added to AFM
        #######################################################################

        if 'dual' in self.triplet_version:

            # Extract features
            patch_1_f_dual = self.backbone.feature_extractor(patch_1)
            patch_2_f_dual = self.backbone.feature_extractor(patch_2)
            patch_1_f_prime_dual = self.backbone.feature_extractor(patch_1_prime)

            # Distance L1
            l1_dual = torch.sum(torch.abs(patch_1_f_prime_dual - patch_2_f_dual), axis=1)
            l3_dual = torch.sum(torch.abs(patch_1_f_dual - patch_2_f_dual), axis=1)

            # Prepare masks
            patch_1_m_dual = torch.squeeze(patch_1_m, dim=1)
            patch_2_m_dual = torch.squeeze(patch_2_m, dim=1)
            patch_1_m_prime_dual = torch.squeeze(patch_1_m_prime, dim=1)
            patch_2_m_prime_dual = torch.squeeze(patch_2_m_prime, dim=1)

            # First loss elem
            ln1_den_dual = torch.sum(torch.sum(patch_1_m_prime_dual * patch_2_m_dual, dim=-1), dim=-1)
            ln1_dual = torch.sum(torch.sum(patch_1_m_prime_dual * patch_2_m_dual * (l1_dual - l3_dual), dim=-1), dim=-1) / \
                  torch.max(ln1_den_dual, torch.ones_like(ln1_den_dual))

            # Sum losses over batch
            loss_dual = torch.sum(ln1_dual)

            if 'double-line' in self.triplet_version:
                patch_2_f_prime_dual = self.backbone.feature_extractor(patch_2_prime)
                l2_dual = torch.sum(torch.abs(patch_2_f_prime_dual - patch_1_f_dual), axis=1)

                # Second loss elem
                ln2_den_dual = torch.sum(torch.sum(patch_2_m_prime_dual * patch_1_m_dual, dim=-1), dim=-1)
                ln2_dual = torch.sum(torch.sum(patch_2_m_prime_dual * patch_1_m_dual * (l2_dual - l3_dual), dim=-1),
                                     dim=-1) / \
                           torch.max(ln2_den_dual, torch.ones_like(ln2_den_dual))
                loss_dual = loss_dual + torch.sum(ln2_dual)

        #######################################################################
        # Size mismatch fix strategies
        #######################################################################

        _, f_c, f_h, f_w = patch_1_f_prime.shape
        if self.sampling_strategy == 'downsample-mask' or True:

            # Downsample mask
            downsample_factor = patch_1_m.shape[-1] // f_h
            downsample_layer = torch.nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor, padding=0)
            patch_1_m_prime = downsample_layer(patch_1_m_prime)
            patch_2_m = downsample_layer(patch_2_m)

            # Prepare second patch warped
            if 'double-line' in self.triplet_version:
                patch_1_m = downsample_layer(patch_1_m)
                patch_2_m_prime = downsample_layer(patch_2_m_prime)

        #######################################################################
        # LOSS
        #######################################################################

        if 'one-line' in self.triplet_version:

            # Distances L1
            if self.triplet_distance == 'l1':

                if self.auxiliary_resnet.with_projection_head is not None:

                    norm = torch.norm(patch_1_f_prime, p=2, dim=1).unsqueeze(dim=1)
                    patch_1_f_prime = patch_1_f_prime / norm

                    norm = torch.norm(patch_2_f, p=2, dim=1).unsqueeze(dim=1)
                    patch_2_f = patch_2_f / norm

                    norm = torch.norm(patch_1_f, p=2, dim=1).unsqueeze(dim=1)
                    patch_1_f = patch_1_f / norm

                l1 = torch.sum(torch.abs(patch_1_f_prime - patch_2_f), axis=1)
                l3 = torch.sum(torch.abs(patch_1_f - patch_2_f), axis=1)

            # Distances cosine
            elif self.triplet_distance == 'cosine':

                if self.auxiliary_resnet.with_projection_head is not None:

                    norm = torch.norm(patch_1_f_prime, p=2, dim=1).unsqueeze(dim=1)
                    patch_1_f_prime = patch_1_f_prime / norm

                    norm = torch.norm(patch_2_f, p=2, dim=1).unsqueeze(dim=1)
                    patch_2_f = patch_2_f / norm

                    norm = torch.norm(patch_1_f, p=2, dim=1).unsqueeze(dim=1)
                    patch_1_f = patch_1_f / norm

                l1 = 1 - torch.cosine_similarity(patch_1_f_prime, patch_2_f, dim=1)
                l3 = 1 - torch.cosine_similarity(patch_1_f, patch_2_f, dim=1)

            else:
                assert False, 'Do not know this distance metric'

            # Triplet Margin Loss
            loss_mat = torch.max(l1-l3 + torch.ones_like(l1) * self.triplet_margin, torch.zeros_like(l1))

            # Add scores
            if scores is not None:
                _, f_h, f_w = loss_mat.shape
                scores_f = scores.reshape(b * n, 1, 1).repeat(1, f_h, f_w)                       # [B*n, H/8, W/8]
                loss_mat = loss_mat * scores_f                                                      # [B*N, H/8, W/8]

            # Prepare masks
            patch_2_m = torch.squeeze(patch_2_m, dim=1)
            patch_1_m_prime = torch.squeeze(patch_1_m_prime, dim=1)

            ###################################################################
            # Apply mask Zhang way
            ###################################################################

            if not self.change_detection_mask:

                loss_den = torch.sum(torch.sum(patch_1_m_prime * patch_2_m, dim=-1), dim=-1)
                loss = torch.sum(torch.sum(patch_1_m_prime * patch_2_m * loss_mat, dim=-1), dim=-1) /\
                       torch.max(loss_den, torch.ones_like(loss_den))

            ###################################################################
            # Apply mask our way
            ###################################################################

            else:

                loss_den = torch.sum(torch.sum(patch_1_m_prime, dim=-1), dim=-1)
                loss = torch.sum(torch.sum(patch_1_m_prime * loss_mat, dim=-1), dim=-1) /\
                       torch.max(loss_den, torch.ones_like(loss_den))

            # Sum losses over batch
            loss = torch.sum(loss)

        elif 'double-line' in self.triplet_version:

            # Distance L1
            if self.triplet_distance == 'l1':

                # if self.auxiliary_resnet.with_projection_head is not None:
                # 
                #     norm = torch.norm(patch_1_f_prime, p=2, dim=1).unsqueeze(dim=1)
                #     patch_1_f_prime = patch_1_f_prime / norm
                # 
                #     norm = torch.norm(patch_2_f, p=2, dim=1).unsqueeze(dim=1)
                #     patch_2_f = patch_2_f / norm
                # 
                #     norm = torch.norm(patch_2_f_prime, p=2, dim=1).unsqueeze(dim=1)
                #     patch_2_f_prime = patch_2_f_prime / norm
                # 
                #     norm = torch.norm(patch_1_f, p=2, dim=1).unsqueeze(dim=1)
                #     patch_1_f = patch_1_f / norm

                l1 = torch.abs(patch_1_f_prime - patch_2_f)
                l2 = torch.abs(patch_2_f_prime - patch_1_f)
                l3 = torch.abs(patch_1_f - patch_2_f)

            # Distance L2
            elif self.triplet_distance == 'l2':

                # if self.auxiliary_resnet.with_projection_head is not None:
                #
                #     norm = torch.norm(patch_1_f_prime, p=2, dim=1).unsqueeze(dim=1)
                #     patch_1_f_prime = patch_1_f_prime / norm
                #
                #     norm = torch.norm(patch_2_f, p=2, dim=1).unsqueeze(dim=1)
                #     patch_2_f = patch_2_f / norm
                #
                #     norm = torch.norm(patch_2_f_prime, p=2, dim=1).unsqueeze(dim=1)
                #     patch_2_f_prime = patch_2_f_prime / norm
                #
                #     norm = torch.norm(patch_1_f, p=2, dim=1).unsqueeze(dim=1)
                #     patch_1_f = patch_1_f / norm

                l1 = torch.mean(torch.square(patch_1_f_prime - patch_2_f), axis=1)
                l2 = torch.mean(torch.square(patch_2_f_prime - patch_1_f), axis=1)
                l3 = torch.mean(torch.square(patch_1_f - patch_2_f), axis=1)

            # Distances cosine
            elif self.triplet_distance == 'cosine':

                # if self.auxiliary_resnet.with_projection_head is not None:
                # 
                #     norm = torch.norm(patch_1_f_prime, p=2, dim=1).unsqueeze(dim=1)
                #     patch_1_f_prime = patch_1_f_prime / norm
                # 
                #     norm = torch.norm(patch_2_f, p=2, dim=1).unsqueeze(dim=1)
                #     patch_2_f = patch_2_f / norm
                # 
                #     norm = torch.norm(patch_2_f_prime, p=2, dim=1).unsqueeze(dim=1)
                #     patch_2_f_prime = patch_2_f_prime / norm
                # 
                #     norm = torch.norm(patch_1_f, p=2, dim=1).unsqueeze(dim=1)
                #     patch_1_f = patch_1_f / norm

                l1 = 1 - torch.cosine_similarity(patch_1_f_prime, patch_2_f, dim=1)
                l2 = 1 - torch.cosine_similarity(patch_2_f_prime, patch_1_f, dim=1)
                l3 = 1 - torch.cosine_similarity(patch_1_f, patch_2_f, dim=1)

            else:
                assert False, 'Do not know this distance metric'

            # Prepare masks
            patch_1_m = torch.squeeze(patch_1_m, dim=1)
            patch_2_m = torch.squeeze(patch_2_m, dim=1)
            patch_1_m_prime = torch.squeeze(patch_1_m_prime, dim=1)
            patch_2_m_prime = torch.squeeze(patch_2_m_prime, dim=1)

            # First loss elem
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
                    assert l2.shape[:2] == torch.Size([64, 64]), 'Hardcoded: batchsize=64, channels=64'
                    loss_mat_2 = torch.max(torch.sum(l2, dim=1) - torch.sum(l3, dim=1) + self.triplet_margin*64,
                                           torch.zeros_like(l2))
                else:
                    assert False, 'Do not know this aggregation technique'
            ln2 = torch.sum(torch.sum(patch_2_m_prime * patch_1_m * loss_mat_2, dim=-1), dim=-1) /\
                  torch.max(ln2_den, torch.ones_like(ln2_den))

            # Sum losses over batch
            ln1 = torch.sum(ln1)
            ln2 = torch.sum(ln2)

            # Forth loss elem
            batch_size = data[e1].shape[0]
            eye = torch.eye(3, dtype=h1.dtype, device=h1.device).unsqueeze(dim=0).repeat(batch_size, 1, 1)
            ln3 = torch.sum((torch.matmul(h1, h2) - eye) ** 2)

            # Final loss
            loss = ln1 + ln2 + self.triplet_mu * ln3

        #######################################################################
        # Dual loss needs to be merged
        #######################################################################

        if 'dual' in self.triplet_version:
            loss = loss + loss_dual

        #######################################################################
        # Tensorboard logs
        #######################################################################

        if 'summary_writer' in data:
            step = data['summary_writer_step']

            # Feature space
            data['summary_writer'].add_scalars('feature_space', {'patch_1_f': torch.mean(patch_1_f).item()}, step)
            data['summary_writer'].add_scalars('feature_space', {'patch_2_f': torch.mean(patch_2_f).item()}, step)
            data['summary_writer'].add_scalars('feature_space', {'patch_1_f_prime': torch.mean(patch_1_f_prime).item()},
                                               step)

            # Loss componets
            data['summary_writer'].add_scalars('loss_comp', {
                'l1': torch.mean(torch.abs(patch_2_f - patch_1_f_prime)).item()}, step)
            data['summary_writer'].add_scalars('loss_comp', {
                'l3': torch.mean(torch.abs(patch_2_f - patch_1_f)).item()}, step)
            eye = torch.eye(3, dtype=h1.dtype, device=h1.device).unsqueeze(dim=0).repeat(h1.shape[0], 1, 1)
            data['summary_writer'].add_scalars('h', {'h1': (torch.sum((h1 - eye) ** 2)).item()}, step)

            if 'double-line' in self.triplet_version:
                data['summary_writer'].add_scalars('loss_den', {'l1_den': torch.min(ln1_den).item()}, step)
                data['summary_writer'].add_scalars('loss_den', {'l2_den': torch.min(ln2_den).item()}, step)

        #######################################################################
        # Delta GT
        #######################################################################

        # delta
        delta_gt, delta_hat = None, None
        if 'delta' in data:
            delta_gt = data['delta']

        # Calc average of delta_hat
        if scores is not None:
            delta_hats = delta_hats * scores.reshape(b * n, 1, 1).repeat(1, 4, 2)   # [B*N, 4, 2]
            delta_hats = torch.sum(delta_hats.reshape(b, n, 4, 2), dim=1)           # [B, 4, 2]

        # Return loss: ground_truth, original_non_patched_image, delta_gt, delta_hat
        return loss, delta_gt, delta_hats

    def predict_homography(self, data):

        #######################################################################
        # No DSAC needed
        #######################################################################

        if len(self.delta_hat_keys):

            # Get delta_hats
            delta_hats = data[self.delta_hat_keys[0]]
            return delta_hats, None

        #######################################################################
        # DSAC is required
        #######################################################################

        else:

            #######################################################################
            # Prepare data
            #######################################################################

            # Fetch perspective field data
            perspective_field_12 = data[self.pf_keys[0]]

            # Create coord field
            map_field_12, self.coordinate_field_12, self.four_points_12 = self.forward_map_field(
                perspective_field_12, self.coordinate_field_12, self.four_points_12)

            #######################################################################
            # DSAC with SoftMax
            #######################################################################

            batch_size = perspective_field_12.shape[0]
            homography_hats, homography_scores = self.dsac(self.coordinate_field_12, map_field_12,
                                                           hypothesis_no=self.hypothesis_no,
                                                           points_per_hypothesis=self.point_per_hypothesis)

            # Find the best homography
            indices = torch.argmax(homography_scores, dim=-1, keepdim=False)
            indices = indices.reshape(-1, 1, 1, 1).repeat(1, 1, 3, 3)
            homography_hats = torch.gather(homography_hats, dim=1, index=indices)

            # Find delta hats
            four_points = np.array([[0, 0], [perspective_field_12.shape[-1], 0],
                                    [perspective_field_12.shape[-1], perspective_field_12.shape[-2]],
                                    [0, perspective_field_12.shape[-2]]])
            four_points = torch.from_numpy(four_points).float().to(perspective_field_12.device)
            four_points = torch.unsqueeze(four_points, dim=0).repeat(batch_size, 1, 1)
            four_points_transformed = kornia.transform_points(homography_hats.reshape(-1, 3, 3), four_points)
            delta_hats = (four_points_transformed - four_points).reshape(batch_size, 4, 2)
            return delta_hats, None
