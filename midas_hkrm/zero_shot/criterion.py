import logging

import torch
import torch.nn as nn
from midas_hkrm.utils import compute_scale_and_shift

logger = logging.getLogger()

"""
Much of this code was adapted from: https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022
Note that this loss function are computed only on valid pixels
"""


class ComputeDepthThenCriterion(nn.Module):
    """
    Convenience criterion.
    Given a disparity map, compute the corresponding depth map (with the least squares fitting method)
    The depth is capped (anything more than the cap is set to the max), and then a loss function is computed
    """

    def __init__(self, depth_cap: int, criterion: nn.Module):
        """
        Args:
            depth_cap (int): maximum depth possible, anything more than this will be set to this maximum
            criterion (nn.Module): Loss function in depth space
        """
        super().__init__()
        self._depth_module = DispToDepth(depth_cap)
        self._criterion = criterion

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._criterion(self._depth_module(prediction, target), target)


class DispToDepth(nn.Module):
    """
    Compute the depth map from a disparity map and a target depth map,
    with the least squares criterion. Then cap the depth values at a maximum value.
    """

    def __init__(self, depth_cap: int):
        """

        Args:
            depth_cap (int): maximum depth. Values will be capped at this maximum.
        """
        super().__init__()
        self._depth_cap = depth_cap

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target > 0
        logger.debug(f"Number of valid pixels {mask.sum()}")

        target_disparity = torch.zeros_like(target)
        target_disparity[mask] = 1.0 / target[mask]

        scale, shift = compute_scale_and_shift(prediction, target_disparity)
        logger.debug(
            f"Computed scale of shape {scale.shape} and shift of shape {shift.shape}"
        )
        prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        disparity_cap = 1.0 / self._depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        prediction_depth = 1.0 / prediction_aligned
        return prediction_depth


class BadPixelMetric(nn.Module):
    """
    Bad Pixel metric as computed for e.g, on NYUv2 (threshold is 1.25).
    Please see the report for the mathematical details.
    """

    def __init__(self, threshold: float):
        """
        Args:
            threshold (float): threshold after which a pixel is considered "wrong" (this is usually 1.25).
        """
        super().__init__()
        self._threshold = threshold

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logger.debug(
            f"Computing BadPixelMetric from prediction of shape {prediction.shape} and target of shape {target.shape}"
        )

        mask = target > 0
        err = torch.zeros_like(prediction, dtype=torch.float)

        err[mask] = torch.max(
            prediction[mask] / target[mask],
            target[mask] / prediction[mask],
        ).float()

        err[mask] = (err[mask] > self._threshold).float()
        logger.debug(f"Error sum {err.sum((1,2))}")

        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        return 100 * torch.mean(p)


class AbsRel(nn.Module):
    """
    Mean Absolute value of the relative error, as evaluated on ETH3D and TUM.
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target > 0

        err = torch.zeros_like(prediction, dtype=torch.float)

        err[mask] = ((prediction[mask] - target[mask]).abs() / target[mask]).float()
        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        return torch.mean(p)


class MaskedRMSE(nn.Module):
    """
    Root Mean Squared Error masked only on valid pixels.
    """

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target > 0
        valid = mask.sum((1, 2)) > 0

        if not torch.any(valid):  # Some samples from BlendedMVS have all invalid pixels
            return 0

        prediction, target = prediction[valid], target[valid]

        err = torch.zeros_like(prediction, dtype=torch.float)
        err[mask] = ((prediction[mask] - target[mask]) ** 2).float()

        p = torch.sqrt(torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2)))
        return torch.mean(p)
