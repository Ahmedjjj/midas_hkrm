import torch
import torch.nn as nn
from src.utils import compute_scale_and_shift

# Please see: https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022
# code was adapted from this gist


class BadPixelMetric(nn.Module):
    def __init__(self, threshold, depth_cap):
        super().__init__()
        self._threshold = threshold
        self._depth_cap = depth_cap

    def forward(self, prediction, target):
        mask = target > 0
        target_disparity = torch.zeros_like(target)
        target_disparity[mask == 1] = 1.0 / target[mask == 1]

        scale, shift = compute_scale_and_shift(prediction, target_disparity, mask)
        prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        disparity_cap = 1.0 / self._depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        prediction_depth = 1.0 / prediction_aligned

        # bad pixel
        err = torch.zeros_like(prediction_depth, dtype=torch.float)

        err[mask == 1] = torch.max(
            prediction_depth[mask == 1] / target[mask == 1],
            target[mask == 1] / prediction_depth[mask == 1],
        )

        err[mask == 1] = (err[mask == 1] > self._threshold).float()

        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        return 100 * torch.mean(p)
