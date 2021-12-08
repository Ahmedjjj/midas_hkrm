import torch.nn as nn
from nn import functional as F
import torch
import numpy as np

from midas.blocks import _make_encoder
from src.objects.features import ObjectFeatureExtractor
import torchvision.transforms.functional as TF

import enum


class ResNetSizes(enum.Enum):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    LAYER_1 = (72, 96)
    LAYER_2 = (36, 48)
    LAYER_3 = (18, 24)
    LAYER_4 = (9, 12)


class MidasHKRMNet(nn.Module):
    """
    A modified MiDaS network that uses object features as extra feature maps in the decoder
    """
    def __init__(object_feature_extractor: ObjectFeatureExtractor,
                 object_feature_dimension: int = 1024,
                 max_objects: int,
                 pretrained_resnet: bool = True,
                 features=256):

        super().__init__()

        # basic resnet encoder + midas specific num channels reduction
        self.backbone, self.channel_reduc = _make_encoder(
            backbone="resnext101_wsl", features=features, use_pretrained=pretrained_resnet)

        # backbone object feature extractor
        self.object_feature_extractor = object_feature_extractor
        self.object_feature_dimension = object_feature_dimension

        # project object features into vectors of the desired sizes
        self.object_feature_mapper_4 = nn.Linear(
            object_feature_dimension, ResNetSizes.LAYER_4.height * ResNetSizes.LAYER_4.width)
        self.object_feature_mapper_3 = nn.Linear(
            object_feature_dimension,  ResNetSizes.LAYER_3.height * ResNetSizes.LAYER_3.width)
        self.object_feature_mapper_2 = nn.Linear(
            object_feature_dimension,  ResNetSizes.LAYER_2.height * ResNetSizes.LAYER_2.width)
        self.object_feature_mapper_1 = nn.Linear(
            object_feature_dimension,  ResNetSizes.LAYER_1.height * ResNetSizes.LAYER_1.width)

        self.max_objects = max_objects

        # We have two extra feature maps (object features + mask) for each object in the image
        self.refinenet_4 = FeatureFusionBlock(features + max_objects * 2)
        self.refinenet_3 = FeatureFusionBlock(features + max_objects * 2)
        self.refinenet_2 = FeatureFusionBlock(features + max_objects * 2)
        self.refinenet_1 = FeatureFusionBlock(features + max_objects * 2)

        # same as MidaS
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

    def forward(self, x: torch.tensor):
        """
        Args:
            x (torch.tensor): tensor containing the input images to the model, assumed to have shape num images x num channels x height x width

        Returns:
            torch.tensor: disparity map
        """

        # Resnet
        layer_1 = self.backbone.layer1(x)
        layer_2 = self.backbone.layer2(layer_1)
        layer_3 = self.backbone.layer3(layer_2)
        layer_4 = self.backbone.layer4(layer_3)

        # Get multiscale feature maps with same number of channels
        layer_1_rn = self.channel_reduc.layer1_rn(layer_1)
        layer_2_rn = self.channel_reduc.layer2_rn(layer_2)
        layer_3_rn = self.channel_reduc.layer3_rn(layer_3)
        layer_4_rn = self.channel_reduc.layer4_rn(layer_4)

        features, num_objects, classes, masks = self.object_feature_extractor.get_object_features(
            x, outputs=["features", "num_objects", "classes", "masks"])

        obj_features = self._pad_image_features(features, num_objects)
        masks = self._pad_masks(masks)

        # convert object features into feature maps
        obj_features_1 = self.object_feature_mapper_1(
            obj_features).reshape(-1, self.max_objects, ResNetSizes.LAYER_1.height, ResNetSizes.LAYER_1.width)
        obj_features_2 = self.object_feature_mapper_2(
            obj_features).reshape(-1, self.max_objects, ResNetSizes.LAYER_2.height, ResNetSizes.LAYER_2.width)
        obj_features_3 = self.object_feature_mapper_3(
            obj_features).reshape(-1, self.max_objects, ResNetSizes.LAYER_3.height, ResNetSizes.LAYER_3.width)
        obj_features_4 = self.object_feature_mapper_4(
            obj_features).reshape(-1, self.max_objects, ResNetSizes.LAYER_4.height, ResNetSizes.LAYER_4.width)

        # Resize masks
        masks_4 = TF.resize(masks, size=[ResNetSizes.LAYER_4.height, ResNetSizes.LAYER_4.width])
        masks_3 = TF.resize(masks, size=[ResNetSizes.LAYER_3.height, ResNetSizes.LAYER_3.width])
        masks_2 = TF.resize(masks, size=[ResNetSizes.LAYER_2.height, ResNetSizes.LAYER_2.width])
        masks_1 = TF.resize(masks, size=[ResNetSizes.LAYER_1.height, ResNetSizes.LAYER_1.width])

        decoder_input_4 = torch.cat((layer_4_rn, obj_features_4, masks_4), dim=1)
        decoder_input_3 = torch.cat((layer_3_rn, obj_features_3, masks_3), dim=1)
        decoder_input_2 = torch.cat((layer_2_rn, obj_features_2, masks_2), dim=1)
        decoder_input_1 = torch.cat((layer_1_rn, obj_features_1, masks_1), dim=1)

        # same as MiDaS, decoder
        path_4 = self.scratch.refinenet4(decoder_input_4)
        path_3 = self.scratch.refinenet3(path_4, decoder_input_3)
        path_2 = self.scratch.refinenet2(path_3, decoder_input_2)
        path_1 = self.scratch.refinenet1(path_2, decoder_input_1)

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)

    def _pad_masks(masks: List[torch.tensor]):
        padded_masks = []
        height, width = masks[0].shape[1:]  # assumes all masks have the same shape
        for mask in masks:
            padded_mask = torch.pad(mask[: self.max_objects], (0, 0, 0, 0, 0, max(0, self.max_objects - mask.shape[0])))
            padded_masks.append(padded_mask)
        return torch.cat(padded_masks).reshape(-1, self.max_objects, height, width)

    def _pad_image_features(obj_features: torch.tensor, num_objects: List[int]):
        cur = 0
        for num_obj in num_objects:
            final_tensors.append(self._pad_zeros(obj_features[cur: cur + num_obj]))
            cur += num_obj

        return torch.cat(final_tensors)

    def _pad_zeros(self, input_tensor):
        return F.pad(input_tensor[:self.max_objects], (0, 0, 0, max(0, self.max_objects - input_tensor.shape[0]))
