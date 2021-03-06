import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


"""
Implements the Shift Invariant Trimmed MAE loss.
Some of this code was taken from: https://gist.github.com/ranftlr/1d6194db2e1dffa0a50c9b0a9549cbd2
Please see the paper: https://arxiv.org/abs/1907.01341v3 for the mathematical details
All operation are done in masked fashion, i.e only valid ground truth pixels are used.
It is assumed all invalid ground truth pixels have value 0.
"""


def normalize_prediction_robust(
    target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Obtain a scale and shift invariant version of the tensor target


    Args:
        target (torch.Tensor): torch.tensor
        mask (torch.Tensor): mask on the valid pixels in target.

    Returns:
        torch.Tensor: torch.tensor (normalized target)
    """

    logger.debug(f"normalizing input of shape {target.shape}")
    logger.debug(f"mask has shape : {mask.shape}")
    orig_shape = target.shape

    target = target.flatten(start_dim=1)
    mask = mask.flatten(start_dim=1)
    shift = torch.zeros((target.shape[0], 1), device=target.device)

    for i, (t, m) in enumerate(zip(target, mask)):
        shift[i] = torch.median(t[m])
    logger.debug(f"Computed shift shape: {shift.shape}")
    logger.debug(f"Shift has nans: {torch.any(torch.isnan(shift))}")
    scale = (target - shift).abs()
    scale *= mask
    scale = scale.sum(dim=1) / mask.sum(dim=1)

    logger.debug(f"Computed scale has shape: {scale.shape}")
    logger.debug(f"Shift has nans: {torch.any(torch.isnan(scale))}")

    scale[scale == 0] = 1
    normalized_targets = (target - shift) / scale.unsqueeze(1)
    logger.debug(
        f"nomalized targets has nans: {torch.any(torch.isnan(normalized_targets))}"
    )

    return normalized_targets.reshape(orig_shape)


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    (Masked) gradient loss function.

    Args:
        prediction (torch.Tensor): prediction tensor.
        target (torch.Tensor): ground truth tensor.

    Returns:
        torch.Tensor: tensor of one entry, the gradient loss
    """
    mask = target > 0

    diff = prediction - target

    grad_x = (diff[:, :, 1:] - diff[:, :, :-1]).abs()
    grad_y = (diff[:, 1:, :] - diff[:, :-1, :]).abs()

    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]

    normalization = mask.sum((1, 2))
    loss = (grad_x * mask_x).sum((1, 2)) + (grad_y * mask_y).sum((1, 2))
    valid = normalization > 0
    if valid.sum() == 0:
        return 0
    loss *= valid
    loss[valid] /= normalization[valid]
    return loss.mean()


class MultiScaleGradientLoss(nn.Module):
    """
    Gradient loss at multiple scales.
    """

    def __init__(self, scales: int = 4):
        """
        Args:
            scales (int, optional): number of scales to use. Defaults to 4.
        """
        super().__init__()
        self.__scales = scales

    def forward(self, prediction, target):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)
            total += gradient_loss(
                prediction[:, ::step, ::step], target[:, ::step, ::step]
            )

        return total


class SSITrimmedMAELoss(nn.Module):
    """
    Final loss of the model.
    Concretely this does the following:
    . Normalize the prediction and the targets into scale and shift invariant versions.
    . Compute the trimmed MAE and the multi-scale gradient loss and sum them
    """

    def __init__(self, trim: float = 0.2, alpha: float = 0.5, scales: int = 4):
        """
        Args:
            trim (float, optional): Percentage of residuals to trim. Defaults to 0.2.
            alpha (float, optional): weight of the gradient loss. Defaults to 0.5.
            scales (int, optional): number of scales to use. Defaults to 4.
        """

        super().__init__()
        self.__trim_mae = TrimmedMAELoss(trim)
        self.__alpha = alpha
        self.__grad_loss = MultiScaleGradientLoss(scales=scales)

    def forward(self, prediction, target):
        logger.debug(
            f"Computing SSITrimmedMAELoss for predictions of shape {prediction.shape} and target of shape {target.shape}"
        )
        mask = target > 0
        valid_samples = mask.sum((1, 2)) > 0
        if not torch.any(valid_samples):
            return 0

        logger.debug(f"Number of valid samples {valid_samples.sum()}")
        prediction, target, mask = (
            prediction[valid_samples],
            target[valid_samples],
            mask[valid_samples],
        )
        logger.debug(f"Valid predictions {len(prediction)}")
        logger.debug(f"Valid target {len(target)}")

        prediction = normalize_prediction_robust(prediction, mask)
        logger.debug(f"normalized prediction shape: {prediction.shape}")
        target = normalize_prediction_robust(target, mask)

        logger.debug(f"normalized target shape: {target.shape}")
        loss = self.__trim_mae(prediction, target)
        if self.__alpha > 0:
            loss += self.__alpha * self.__grad_loss(prediction, target)
        return loss


class TrimmedMAELoss(nn.Module):
    """
    TrimmedMAELoss: Compute the MAE and discard a fraction of the largest residuals
    """

    def __init__(self, trim: float = 0.2):
        """
        Args:
            trim (float, optional): trimmimg percentage. Defaults to 0.2.
        """
        super().__init__()
        self.trim = trim

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logger.debug(
            f"Computing TrimmedMAELoss for predictions of shape {prediction.shape} and target of shape {target.shape}"
        )
        # we can treat images as vectors
        prediction = prediction.flatten(start_dim=1)
        target = target.flatten(start_dim=1)

        # 0 disparity indicates invalid ground truth pixels
        mask = target > 0

        # MAE
        res = torch.zeros_like(prediction)
        res[mask] = (prediction[mask] - target[mask]).abs()

        # Trim largest residuals
        sorted_res, _ = torch.sort(res)
        max_index = (~mask).sum(dim=1)  # do not count invalid pixels
        max_index += int((1 - self.trim) * res.shape[1])
        max_index[max_index >= res.shape[1]] = res.shape[1] - 1
        max_res_per_target = sorted_res[range(res.shape[0]), max_index]
        res[res > max_res_per_target.unsqueeze(1)] = 0

        # each loss should be normalized independently
        normalization = 2 * mask.sum(1)
        valid = normalization > 0
        if valid.sum() == 0:
            return 0
        loss = res.sum(dim=1)
        loss *= valid
        loss[valid] /= normalization[valid]
        return loss.mean()
