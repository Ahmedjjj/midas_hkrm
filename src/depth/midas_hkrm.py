import torch.nn as nn
from nn import functional as F
import torch
import numpy as np

from midas.blocks import _make_encoder
from src.objects.features import ObjectFeatureExtractor

class MidasHKRMNet(nn.Module):
    """
    A modified MiDaS network that uses object features as extra feature maps in the decoder
    """
    def __init__(object_feature_extractor: ObjectFeatureExtractor,
                 object_feature_dimension: int = 1024,
                 max_objects: int,
                 pretrained_resnet: bool=True,
                 features = 256):
        
        super().__init__()

        # basic resnet encoder
        self.backbone, self.channel_reduc = _make_encoder(backbone="resnext101_wsl", features=features, use_pretrained=pretrained_resnet)
        
        # backbone object feature extractor
        self.object_feature_extractor = object_feature_extractor
        self.object_feature_dimension = object_feature_dimension

        # map object features into vectors of the desired sizes
        self.object_feature_mapper_4 = nn.Sequential(nn.Linear(object_feature_dimension, 9 * 12), nn.ReLU())
        self.object_feature_mapper_3 = nn.Sequential(nn.Linear(object_feature_dimension, 18 * 24), nn.ReLU())
        self.object_feature_mapper_2 = nn.Sequential(nn.Linear(object_feature_dimension, 36 * 48), nn.ReLU())
        self.object_feature_mapper_1 = nn.Sequential(nn.Linear(object_feature_dimension, 72 * 96), nn.ReLU())

        self.max_objects = max_objects

        self.refinenet_4 = FeatureFusionBlock(features + max_objects)
        self.refinenet_3 = FeatureFusionBlock(features + max_objects)
        self.refinenet_2 = FeatureFusionBlock(features + max_objects)
        self.refinenet_1 = FeatureFusionBlock(features + max_objects)


        # same as MidaS
        self.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True)
        )

    def forward(self, x):
        layer_1 = self.backbone.layer1(x)
        layer_2 = self.backbone.layer2(layer_1)
        layer_3 = self.backbone.layer3(layer_2)
        layer_4 = self.backbone.layer4(layer_3)

        layer_1_rn = self.channel_reduc.layer1_rn(layer_1)
        layer_2_rn = self.channel_reduc.layer2_rn(layer_2)
        layer_3_rn = self.channel_reduc.layer3_rn(layer_3)
        layer_4_rn = self.channel_reduc.layer4_rn(layer_4)

        obj_features = self._get_per_image_features(*self.object_feature_extractor(x))

        # convert object features into feature maps
        obj_features_1 = self.object_feature_mapper_1(obj_features).reshape(-1, 72, 96)
        obj_features_2 = self.object_feature_mapper_2(obj_features).reshape(-1, 36, 48)
        obj_features_3 = self.object_feature_mapper_3(obj_features).reshape(-1, 18, 24)
        obj_features_4 = self.object_feature_mapper_4(obj_features).reshape(-1, 9 , 12)

        # same as MiDaS
        path_4 = self.scratch.refinenet4(torch.cat((layer_4_rn, obj_features_4), dim=-3))
        path_3 = self.scratch.refinenet3(path_4, torch.cat((layer_3_rn, obj_features_3), dim=-3))
        path_2 = self.scratch.refinenet2(path_3, torch.cat((layer_2_rn, obj_features_2), dim=-3))
        path_1 = self.scratch.refinenet1(path_2, torch.cat((layer_1_rn, object_features_1), dim=-3))

        out = self.scratch.output_conv(path_1)

        return torch.squeeze(out, dim=1)

    def _get_per_image_features(obj_features, indices):
        indices = list(indices)
        starts = np.array([0] + indices)[:-1]
        ends = np.array(indices)

        final_tensors = []
        for start, end in zip(starts, ends):
            final_tensors.append(self._pad_features(obj_features[start: end]))
        
        return torch.cat(final_tensors).reshape(-1, self.max_objects, self.object_feature_dimension)

    def _pad_features(self, input_tensor):
        if input_tensor.shape[0] > self.max_objects:
            return input_tensor [: self.max_objects]

        return F.pad(input_tensor, (0, 0 , 0, self.max_objects - input_tensor.shape[0]))
        


