import torch
import torch.nn as nn
from midas_hkrm.utils import compute_scale_and_shift
import logging

logger = logging.getLogger()


class ComputeDepthThenCriterion(nn.Module):
    def __init__(self, depth_cap: int, criterion: nn.Module):
        super().__init__()
        self._depth_module = DispToDepth(depth_cap)
        self._criterion = criterion

    def forward(self, prediction, target):
        return self._criterion(self._depth_module(prediction, target), target)


class DispToDepth(nn.Module):
    def __init__(self, depth_cap):
        super().__init__()
        self._depth_cap = depth_cap

    def forward(self, prediction, target):
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


# Please see: https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022
# code was adapted from this gist
class BadPixelMetric(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self._threshold = threshold

    def forward(self, prediction, target):
        logger.debug(
            f"Computing BadPixelMetric from prediction of shape {prediction.shape} and target of shape {target.shape}"
        )

        mask = target > 0
        err = torch.zeros_like(prediction, dtype=torch.float)

        err[mask] = torch.max(
            prediction[mask] / target[mask],
            target[mask] / prediction[mask],
        )

        err[mask] = (err[mask] > self._threshold).float()
        logger.debug(f"Error sum {err.sum((1,2))}")

        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        return 100 * torch.mean(p)


class AbsRel(nn.Module):
    def forward(self, prediction, target):
        mask = target > 0
        err = torch.zeros_like(prediction, dtype=torch.float)
        err[mask] = ((prediction[mask] - target[mask]).abs() / target[mask]).float()
        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        return torch.mean(p)


class MaskedRMSE(nn.Module):
    def forward(self, prediction, target):
        mask = target > 0
        valid = mask.sum((1, 2)) > 0
        if not torch.any(valid):
            return 0
        prediction, target = prediction[valid], target[valid]
        err = torch.zeros_like(prediction, dtype=torch.float)
        err[mask] = ((prediction[mask] - target[mask]) ** 2).float()
        p = torch.sqrt(torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2)))
        return torch.mean(p)
