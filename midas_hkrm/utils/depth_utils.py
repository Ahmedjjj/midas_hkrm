from typing import List

import torch
import torch.nn.functional as F


def pad_masks(masks: List[torch.tensor], max_objects: int) -> List[torch.Tensor]:
    """
    Pads (and clips) each mask in masks to max_objects. Adds 0 filled masks if the number
    of masks is less than the number of objects.

    Args:
        masks (List[torch.tensor]): the list of masks to pad
        max_objects (int): the maximum number of objects

    Returns:
        List[torch.Tensor]: padded mask list
    """
    padded_masks = []
    for mask in masks:
        padded_mask = F.pad(
            mask[:max_objects], (0, 0, 0, 0, 0, max(0, max_objects - mask.shape[0]))
        )
        padded_masks.append(padded_mask)

    return padded_masks


def pad_features(
    obj_features: torch.tensor, num_objects: List[int], max_objects: int
) -> torch.Tensor:
    """
    Given a list of features (which may correspond to different images in batch mode),
    Pad the features in each image with zeros to the desired number of objects.

    Args:
        obj_features (torch.tensor): feature tensor
        num_objects (List[int]): number of objects for each image
        max_objects (int): desired number of objects for each image

    Returns:
        torch.Tensor: -> padded feature tensor
    """
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
