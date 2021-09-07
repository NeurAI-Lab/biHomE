import cv2
import torch
import kornia
import numpy as np


def four_point_to_homography(corners, deltas, crop=False):
    """
    Args:
        corners ():
        deltas ():
        crop (bool): If set to true, homography will aready contain cropping part.
    """

    assert len(corners.shape) == 3, 'corners should be of size B, 4, 2, but got: {}'.format(corners.shape)
    assert len(deltas.shape) == 3, 'deltas should be of size B, 4, 2, but got: {}'.format(deltas.shape)

    # in order to apply transform and center crop,
    # subtract points by top-left corner (corners[N, 0])
    if 'torch' in str(type(corners)):
        if crop:
            corners = corners - corners[:, 0].view(-1, 1, 2)
        corners_hat = corners + deltas
        return kornia.get_perspective_transform(corners, corners_hat)

    elif 'numpy' in str(type(corners)):
        if crop:
            corners = corners - corners[:, 0].reshape(-1, 1, 2)
        corners_hat = corners + deltas
        return cv2.getPerspectiveTransform(np.float32(corners), np.float32(corners_hat))

    else:
        assert False, 'Wrong type?'


def image_shape_to_corners(patch):
    assert len(patch.shape) == 4, 'patch should be of size B, C, H, W'
    batch_size = patch.shape[0]
    image_width = patch.shape[-2]
    image_height = patch.shape[-1]
    if 'torch' in str(type(patch)):
        corners = torch.tensor([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]],
                               device=patch.device, dtype=patch.dtype, requires_grad=False)
        corners = corners.repeat(batch_size, 1, 1)
    elif 'numpy' in str(type(patch)):
        corners = np.float32([[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]])
        corners = np.tile(np.expand_dims(corners, axis=0), (batch_size, 1, 1))
    else:
        assert False, 'Wrong type?'

    return corners


def warp_image(image, homography, target_h, target_w, inverse=True):

    if 'torch' in str(type(homography)):
        if inverse:
            homography = torch.inverse(homography)
        return kornia.warp_perspective(image, homography, tuple((target_h, target_w)))

    elif 'numpy' in str(type(homography)):
        if inverse:
            homography = np.linalg.inv(homography)
        return cv2.warpPerspective(image, homography, dsize=tuple((target_w, target_h)))

    else:
        assert False, 'Wrong type?'


def perspectiveTransform(points, homography):
    """
    Transform point with given homography.

    Args:
        points (np.array of size Nx2) - 2D points to be transformed
        homography (np.array of size 3x3) - homography matrix

    Returns:
        (np.array of size Nx2) - transformed 2D points
    """

    # Asserts
    assert len(points.shape) == 2 and points.shape[1] == 2, 'points arg should be of size Nx2, but has size: {}'. \
        format(points.shape)
    assert homography.shape == (3, 3), 'homography arg should be of size 3x3, but has size: {}'.format(homography.shape)

    if 'torch' in str(type(homography)) and 'torch' in str(type(points)):

        # if inverse:
        #     homography = torch.inverse(homography)

        points = torch.nn.functional.pad(points, (0, 1), "constant", 1.)
        points_transformed = homography @ (points.permute(1, 0))
        points_transformed = points_transformed.permute(1, 0)
        return points_transformed[:, :2] / points_transformed[:, 2:].repeat(1, 2)

    elif 'numpy' in str(type(homography)) and 'numpy' in str(type(points)):

        # if inverse:
        #     homography = np.linalg.inv(homography)

        return cv2.perspectiveTransform([points], homography).squeeze()

    else:
        assert False, 'Wrong or inconsistent types?'


def perspectiveTransformBatched(points, homography):
    """
    Transform point with given homography.

    Args:
        points (np.array of size BxNx2) - 2D points to be transformed
        homography (np.array of size Bx3x3) - homography matrix

    Returns:
        (np.array of size BxNx2) - transformed 2D points
    """

    # Asserts
    assert len(points.shape) == 3 and points.shape[2] == 2, 'points arg should be of size Nx2, but has size: {}'. \
        format(points.shape)
    assert homography.shape[1:] == (3, 3), 'homography arg should be of size 3x3, but has size: {}'\
        .format(homography.shape)

    if 'torch' in str(type(homography)) and 'torch' in str(type(points)):

        points = torch.nn.functional.pad(points, (0, 1), "constant", 1.)
        points_transformed = homography @ (points.permute(0, 2, 1))
        points_transformed = points_transformed.permute(0, 2, 1)
        return points_transformed[:, :, :2] / points_transformed[:, :, 2:].repeat(1, 1, 2)

    elif 'numpy' in str(type(homography)) and 'numpy' in str(type(points)):
        assert False, 'Not implemented - I was too lazy, sorry!'
    else:
        assert False, 'Wrong or inconsistent types?'


def calc_reprojection_error(source_points, target_points, homography):
    """
    Calculate reprojection error for a given homography.

    Args:
        source_points (np.array of size Nx2) - 2D points to be transformed
        target_points (np.array of size Nx2) - target 2D points
        homography (np.array of size 3x3) - homography matrix

    Returns:
        (float) - reprojection error
    """

    # Asserts
    assert len(source_points.shape) == 2 and source_points.shape[1] == 2, 'source_points arg should be of size Nx2, ' \
                                                                          'but has size: {}'.format(source_points.shape)
    assert len(target_points.shape) == 2 and target_points.shape[1] == 2, 'target_points arg should be of size Nx2, ' \
                                                                          'but has size: {}'.format(target_points.shape)
    assert homography.shape == (3, 3), 'homography arg should be of size 3x3, but has size: {}'.format(homography.shape)

    if 'torch' in str(type(homography)) and 'torch' in str(type(source_points)) and 'torch' in str(type(target_points)):

        transformed_points = perspectiveTransform(source_points, homography)
        reprojection_error = torch.sum((transformed_points - target_points) ** 2)
        return reprojection_error

    if 'numpy' in str(type(homography)) and 'numpy' in str(type(source_points)) and 'numpy' in str(type(target_points)):

        transformed_points = cv2.perspectiveTransform(np.expand_dims(source_points, axis=0), homography).squeeze()
        reprojection_error = np.sum((transformed_points - target_points) ** 2)
        return reprojection_error

    else:
        assert False, 'Wrong or inconsistent types?'
