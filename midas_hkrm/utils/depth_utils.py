from typing import List

import torch
import torch.nn.functional as F


def pad_masks(masks: List[torch.tensor], max_objects: int):
    padded_masks = []
    for mask in masks:
        padded_mask = F.pad(
            mask[:max_objects], (0, 0, 0, 0, 0, max(0, max_objects - mask.shape[0]))
        )
        padded_masks.append(padded_mask)

    return padded_masks


def pad_features(obj_features: torch.tensor, num_objects: List[int], max_objects: int):
    cur = 0
    final_tensors = []
    for num_obj in num_objects:
        img_features = obj_features[cur : cur + num_obj]
        final_tensors.append(
            F.pad(
                img_features[:max_objects],
                (0, 0, 0, max(0, max_objects - img_features.shape[0])),
            )
        )
        cur += num_obj

    return torch.cat(final_tensors)
