import math

import torch
import torch.nn as nn
import torchvision.models as models

import kornia


class ScoreCNN(torch.nn.Module):

    def __init__(self, pretrained):
        super(ScoreCNN, self).__init__()

        # Get ResNet model
        self.resnet18 = models.resnet18(pretrained=pretrained, progress=True)
        # First conv got only 1 channel
        self.resnet18.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Last FC has 8 output units
        self.resnet18.fc = nn.Linear(512, 1, bias=True)

    def forward(self, x):
        return self.resnet18(x)


class DSACSoftmax(torch.nn.Module):

    def __init__(self, **kwargs):
        super(DSACSoftmax, self).__init__()

        # Scoring method
        self.scoring_method = kwargs['SCORING_METHOD'] if 'SCORING_METHOD' in kwargs else 'repr_error'

        # Inliers ratio
        if self.scoring_method == 'inliers_ratio':
            self.scoring_distance_threshold = kwargs['SCORING_DISTANCE_THRESHOLD']

        # Soft inliers ratio
        if self.scoring_method == 'soft_inliers_ratio':
            self.scoring_distance_beta = kwargs['SCORING_DISTANCE_BETA']
            self.scoring_distance_threshold = kwargs['SCORING_DISTANCE_THRESHOLD']

        # If use CNN to score
        if self.scoring_method == 'score_cnn':
            self.score_cnn = ScoreCNN(kwargs['SCORE_CNN_PRETRAINED'])

    @staticmethod
    def __sample_hypotheses(points1, points2, batch_size, points_per_hypothesis, hypothesis_no):

        ###########################################################################
        # Randomly sample points to be used in homography estimation
        ###########################################################################

        points_to_sample = batch_size * points_per_hypothesis * hypothesis_no
        range_to_sample = torch.arange(start=0, end=points1.shape[1], dtype=torch.float32, device=points1.device)
        choice = torch.multinomial(range_to_sample, points_to_sample, replacement=True)
        choice = choice.reshape(batch_size, -1, 1).repeat((1, 1, 2))

        ###########################################################################
        # Extract points1 and points2
        ###########################################################################

        points1_sampled = torch.gather(points1, dim=1, index=choice)
        points2_sampled = torch.gather(points2, dim=1, index=choice)

        ###########################################################################
        # Find homography using kornia dlt
        ###########################################################################

        points1_sampled = points1_sampled.reshape(batch_size * hypothesis_no, points_per_hypothesis, 2)
        points2_sampled = points2_sampled.reshape(batch_size * hypothesis_no, points_per_hypothesis, 2)
        homographies = kornia.find_homography_dlt(points1_sampled, points2_sampled)
        homographies = homographies.reshape(batch_size, hypothesis_no, 3, 3)
        return homographies

    def __score_hypotheses(self, points1, points2, homographies, batch_size, hypothesis_no):

        # Reshape input points
        points_no = points1.shape[1]
        points1 = points1.reshape(batch_size, 1, points_no, 2).repeat(1, hypothesis_no, 1, 1)           # [B, n, N, 2]
        points1 = points1.reshape(batch_size * hypothesis_no, points_no, 2)                             # [B*n, N, 2]

        points2 = points2.reshape(batch_size, 1, points_no, 2).repeat(1, hypothesis_no, 1, 1)           # [B, n, N, 2]
        points2 = points2.reshape(batch_size * hypothesis_no, points_no, 2)                             # [B*n, N, 2]

        # Reshape homographies
        homographies = homographies.reshape(batch_size * hypothesis_no, 3, 3)                           # [B*n, 3, 3]

        # Transform points
        points1_transformed = kornia.transform_points(homographies, points1)                            # [B*n, N, 2]

        # L1 reprojection error score
        if self.scoring_method == 'repr_error':
            reprojection_error = torch.sum(torch.abs(points1_transformed - points2), dim=-1)            # [B*n, N]
            reprojection_error = torch.sum(reprojection_error, dim=-1)
            scores = reprojection_error.reshape(batch_size, hypothesis_no)                              # [B, n]

        # inliers ratio score
        elif self.scoring_method == 'inliers_ratio':
            reprojection_error = torch.norm(points1_transformed - points2, p=None, dim=-1)              # [B*n, N]
            scores = torch.mean((reprojection_error < self.scoring_distance_threshold).to(torch.float32),
                                axis=-1)                                                                # [B*n]
            scores = scores.reshape(batch_size, hypothesis_no)                                          # [B, n]

        # inliers ratio score
        elif self.scoring_method == 'soft_inliers_ratio':
            reprojection_error = torch.norm(points1_transformed - points2, p=None, dim=-1)              # [B*n, N]
            reprojection_error = torch.sigmoid(self.scoring_distance_beta *
                                               (reprojection_error - self.scoring_distance_threshold))  # [B*n, N]
            scores = torch.sum(reprojection_error, dim=-1)
            scores = scores.reshape(batch_size, hypothesis_no)                                          # [B, n]

        # Using CNN to estimate score
        elif self.scoring_method == 'score_cnn':
            reprojection_error = points1_transformed - points2                                          # [B*n, N, 2]
            image_shape = int(math.sqrt(reprojection_error.shape[1]))
            reprojection_error = reprojection_error.permute((0, 2, 1))                                  # [B*n, 2, N]
            reprojection_error = reprojection_error.reshape(batch_size * hypothesis_no, 2, image_shape,
                                                            image_shape)                            # [B*n, 1, im, im]
            scores = self.score_cnn(reprojection_error)
            scores = scores.reshape(batch_size, hypothesis_no)                                          # [B, n]
        else:
            assert False, 'I do not know this scoring method'

        # Softmax of the score
        scores = torch.nn.Softmax(dim=-1)(-scores)                                                      # [B, n]
        reprojection_error = reprojection_error.reshape(batch_size, hypothesis_no, -1)                  # [B, n, N]
        return scores, reprojection_error

    @staticmethod
    def __refine_hypotheses(points1, points2, distances, hypothesis_no):

        batch_size, N, _ = points1.shape

        points1_repeated = points1.reshape(batch_size, 1, N, 2).repeat(1, hypothesis_no, 1, 1)      # [B, n, N, 2]
        points2_repeated = points2.reshape(batch_size, 1, N, 2).repeat(1, hypothesis_no, 1, 1)      # [B, n, N, 2]
        points1_repeated = points1_repeated.reshape(batch_size * hypothesis_no, N, 2)               # [B*n, N, 2]
        points2_repeated = points2_repeated.reshape(batch_size * hypothesis_no, N, 2)               # [B*n, N, 2]

        weights = 1 - distances                                                                     # [B, n, N]
        weights = weights.reshape(batch_size * hypothesis_no, -1)                                   # [B*n, N]

        homographies = kornia.find_homography_dlt(points1_repeated, points2_repeated, weights)      # [B*n, 3, 3]
        homographies = homographies.reshape(batch_size, hypothesis_no, 3, 3)                        # [B, n, 3, 3]
        return homographies

    def forward(self, points1: torch.Tensor, points2: torch.Tensor, points_per_hypothesis: int = 4,
                hypothesis_no: int = 128):

        batch_size = points1.shape[0]

        # Sample hypotheses
        homographies = self.__sample_hypotheses(points1, points2, batch_size, points_per_hypothesis, hypothesis_no)

        # Score hypotheses
        scores, distances = self.__score_hypotheses(points1, points2, homographies, batch_size, hypothesis_no)

        # Refine hypothesis
        #homographies = self.__refine_hypotheses(points1, points2, distances, hypothesis_no)

        return homographies, scores