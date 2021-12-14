import torch
import torch.nn as nn


def normalize_prediction_robust(target, mask):
    orig_shape = target.shape

    target = target.flatten(start_dim=1)
    mask = mask.flatten(start_dim=1)
    shift = torch.zeros((target.shape[0], 1), device=target.device)

    for i, (t, m) in enumerate(zip(target, mask)):
        shift[i] = torch.median(t[m])

    scale = (target - shift).abs()
    scale *= mask
    scale = scale.sum(dim=1) / mask.sum(dim=1)

    normalized_targets = (target - shift) / scale.unsqueeze(1)

    return normalized_targets.reshape(orig_shape)


def gradient_loss(prediction, target):
    mask = target > 0

    diff = prediction - target

    grad_x = (prediction[:, :, 1:] - prediction[:, :, :-1]).abs()
    grad_y = (prediction[:, 1:, :] - prediction[:, :-1, :]).abs()

    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]

    normalization = mask.sum((1, 2))
    loss = (grad_x * mask_x).sum((1, 2)) + (grad_y * mask_y).sum((1, 2))
    valid = normalization > 0
    loss *= valid
    loss[valid] /= normalization[valid]
    return loss.mean()


class MultiScaleGradientLoss(nn.Module):
    def __init__(self, scales=4):
        super().__init__()
        self.__scales = scales

    def forward(self, prediction, target):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)
            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step])

        return total


class SSITrimmedMAELoss(nn.Module):
    def __init__(self, trim=0.2, alpha=0.5, scales=4):
        super().__init__()
        self.__trim_mae = TrimmedMAELoss(trim)
        self.__alpha = alpha
        self.__grad_loss = MultiScaleGradientLoss(scales=4)

    def forward(self, prediction, target):
        mask = target > 0
        prediction = normalize_prediction_robust(prediction, mask)
        target = normalize_prediction_robust(target, mask)

        loss = self.__trim_mae(prediction, target)
        if self.__alpha > 0:
            loss += self.__alpha * self.__grad_loss(prediction, target)
        return loss


class TrimmedMAELoss(nn.Module):

    def __init__(self, trim=0.2):
        super().__init__()
        self.trim = trim

    def forward(self, prediction, target):
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
        max_res_per_target = res[range(res.shape[0]), max_index]
        res[res > max_res_per_target.unsqueeze(1)] = 0

        # each loss should be normalized independently
        normalization = 2 * mask.sum(1)
        valid = normalization > 0
        loss = res.sum(dim=1)
        loss *= valid
        loss[valid] /= normalization[valid]
        return loss.mean()
