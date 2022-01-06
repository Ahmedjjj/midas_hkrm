import logging
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from midas.blocks import (  # the midas code should be on PATH or PYTHONPATH
    FeatureFusionBlock,
    Interpolate,
    _make_encoder,
)

from midas_hkrm.objects import GeneralizedRCNNObjectDetector
from midas_hkrm.objects.features import ObjectDetector
from midas_hkrm.utils import construct_config, get_baseline_config
from midas_hkrm.utils.depth_utils import pad_features, pad_masks
from midas_hkrm.utils.errors import require

logger = logging.getLogger(__name__)


class MidasHKRMNet(nn.Module):
    """
    A modified Midas network that uses object features and object masks as extra feature maps in the decoder.
    This code is inspired from: https://github.com/isl-org/MiDaS/blob/master/midas/midas_net.py
    And directly uses the blocks from: https://github.com/isl-org/MiDaS/blob/master/midas/blocks.py
    """

    def __init__(
        self,
        object_feature_extractor: ObjectDetector,
        max_objects: int,
        object_feature_dimension: int = 1024,
        pretrained_resnet: bool = True,
        features=256,
        device="cuda",
    ):
        """
        Create a new MidasHKRM network

        Args:
            object_feature_extractor (ObjectDetector): the object detection network.
            max_objects (int): maximum number of objects the network uses.
            object_feature_dimension (int, optional): the dimension of the object features. Defaults to 1024.
                                                      this should be perfect a square since the feaature vectors are mapped into square images
            pretrained_resnet (bool, optional): whether to load pre-trained resnet weights. Defaults to True.
            features (int, optional): number of features (channels) that the different resolution encoder outputs will be mapped to. Defaults to 256.
            device (str, optional): device of the model. Defaults to "cuda".
        """
        super().__init__()

        require(
            int(np.sqrt(object_feature_dimension)) ** 2 == object_feature_dimension,
            "object_feature_dimension should be a perfect square",
        )

        self.device = device

        # basic resnet encoder + midas specific num channels reduction
        self.backbone, self.channel_reduc = _make_encoder(
            backbone="resnext101_wsl",
            features=features,
            use_pretrained=pretrained_resnet,
        )

        # backbone object feature extractor
        self.object_feature_extractor = object_feature_extractor
        self.object_feature_dimension = object_feature_dimension

        self.max_objects = max_objects

        # We have two extra feature maps (object features + mask) for each object in the image
        self.refinenet_4 = FeatureFusionBlock(features + max_objects * 2)
        self.refinenet_3 = FeatureFusionBlock(features + max_objects * 2)
        self.refinenet_2 = FeatureFusionBlock(features + max_objects * 2)
        self.refinenet_1 = FeatureFusionBlock(features + max_objects * 2)

        # same as MidaS
        self.output_conv = nn.Sequential(
            nn.Conv2d(
                features + max_objects * 2, 128, kernel_size=3, stride=1, padding=1
            ),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )
        self.to(device)

    def forward(
        self, object_detector_input: List[np.array], depth_input: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x (List[np.array]): list of images, assumed to have shape height x width x num_channels
            depth_input (input_of the depth encoder): torch.tensor
        Returns:
            torch.tensor: disparity map
        """

        require(
            len(object_detector_input) == depth_input.shape[0],
            "detector image number is different from from number of inputs!",
        )

        # Resnet
        layer_1 = self.backbone.layer1(depth_input)
        layer_2 = self.backbone.layer2(layer_1)
        layer_3 = self.backbone.layer3(layer_2)
        layer_4 = self.backbone.layer4(layer_3)

        # Get multiscale feature maps with same number of channels
        layer_1_rn = self.channel_reduc.layer1_rn(layer_1)
        layer_2_rn = self.channel_reduc.layer2_rn(layer_2)
        layer_3_rn = self.channel_reduc.layer3_rn(layer_3)
        layer_4_rn = self.channel_reduc.layer4_rn(layer_4)

        layer_1_height, layer_1_width = layer_1_rn.shape[2:]
        layer_2_height, layer_2_width = layer_2_rn.shape[2:]
        layer_3_height, layer_3_width = layer_3_rn.shape[2:]
        layer_4_height, layer_4_width = layer_4_rn.shape[2:]

        (
            features,
            num_objects,
            classes,
            masks,
        ) = self.object_feature_extractor.get_object_features(
            object_detector_input,
            outputs=["features", "num_objects", "classes", "masks"],
        )

        obj_features = pad_features(features, num_objects, self.max_objects)

        for i in range(len(masks)):
            if len(masks[i]) > 0:
                masks[i] *= classes[i].reshape(-1, 1, 1)

        masks = pad_masks(masks, self.max_objects)

        # convert object features into feature maps
        obj_features_1 = self._resize_features(
            obj_features, size=(layer_1_height, layer_1_width)
        ).reshape(-1, self.max_objects, layer_1_height, layer_1_width)
        obj_features_2 = self._resize_features(
            obj_features, size=(layer_2_height, layer_2_width)
        ).reshape(-1, self.max_objects, layer_2_height, layer_2_width)
        obj_features_3 = self._resize_features(
            obj_features, size=(layer_3_height, layer_3_width)
        ).reshape(-1, self.max_objects, layer_3_height, layer_3_width)
        obj_features_4 = self._resize_features(
            obj_features, size=(layer_4_height, layer_4_width)
        ).reshape(-1, self.max_objects, layer_4_height, layer_4_width)

        # Resize masks
        masks_4 = self._batch_resize_masks(masks, layer_4_rn.shape[2:])
        masks_3 = self._batch_resize_masks(masks, layer_3_rn.shape[2:])
        masks_2 = self._batch_resize_masks(masks, layer_2_rn.shape[2:])
        masks_1 = self._batch_resize_masks(masks, layer_1_rn.shape[2:])

        decoder_input_4 = torch.cat((layer_4_rn, obj_features_4, masks_4), dim=1)
        decoder_input_3 = torch.cat((layer_3_rn, obj_features_3, masks_3), dim=1)
        decoder_input_2 = torch.cat((layer_2_rn, obj_features_2, masks_2), dim=1)
        decoder_input_1 = torch.cat((layer_1_rn, obj_features_1, masks_1), dim=1)

        # same as MiDaS, decoder
        path_4 = self.refinenet_4(decoder_input_4)
        path_3 = self.refinenet_3(path_4, decoder_input_3)
        path_2 = self.refinenet_2(path_3, decoder_input_2)
        path_1 = self.refinenet_1(path_2, decoder_input_1)

        out = self.output_conv(path_1)

        return torch.squeeze(out, dim=1)

    def _batch_resize_masks(
        self, masks: List[torch.Tensor], size: Tuple[int]
    ) -> torch.Tensor:
        """
        Resize (using bilinear interpolation) a list of n tensors into one tensor of the desired size

        Args:
            masks (List[torch.tensor]): list of tensors
            size (Tuple[int]): desired size: HxW

        Returns:
            torch.Tensor: tensor of shape (len(masks) x *size)
        """
        return torch.cat([TF.resize(m, size=size) for m in masks]).reshape(
            -1, self.max_objects, *size
        )

    def _resize_features(
        self, features: torch.Tensor, size: Tuple[int]
    ) -> torch.Tensor:
        """
        Given a tensor of feature vectors reshape each feature vector into a square image
        then biliniearly interpolate to the desired size

        Args:
            features (torch.Tensor): tensor of feature vectors
            size (Tuple[int]): desired size: HxW

        Returns:
            torch.Tensor: torch of len(features) "feature images" of shape size
        """
        square_side = int(np.sqrt(self.object_feature_dimension))
        feature_squares = features.reshape(features.shape[0], square_side, square_side)
        return TF.resize(feature_squares, size)


def create_midas_hkrm_model(
    max_objects: int,
    object_detection_threshold: float,
    load_weights: bool = True,
    object_model_weights: str = None,
    random_init_missing: bool = True,
    device: str = "cuda",
    midas_hkrm_weights: torch.Tensor = None,
    use_hkrm: bool = True,
) -> MidasHKRMNet:
    """
    This is a convenience wrapper around recurring boilerplate model creating code
    Create a MidasHKRM net, optionally initialize weights
    Args:
        max_objects (int): maximum number of objects the model uses.
        object_detection_threshold (float): probability threshold for the object detection backbone,
        load_weights (bool, optional): whether to load existing weights for the model. Defaults to True.
        object_model_weights (str, optional): weights for the object detection model.
                                              should be provided in load_weights is True.
                                              Defaults to None.
        random_init_missing (bool, optional): If True, missing parameters are randomly initialized. Defaults to True.
        device (str, optional): device of the model. Defaults to "cuda".
        midas_hkrm_weights (torch.Tensor, optional): pretrained midas hkrm models.
                                                     if provided the model state_dict loading will be done in strict fashion.
                                                     Defaults to None.
        use_hkrm (bool, optional): if True use a HKRM based object detector.
                                   Otherwise uses a Faster-RCNN-FPN architecture.
                                   Defaults to True.
    Returns:
        MidasHKRMNet: [description]
    """
    if use_hkrm:
        cfg = construct_config()
        cfg.MODEL.WEIGHTS = object_model_weights
    else:
        cfg = get_baseline_config()

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = object_detection_threshold
    cfg.MODEL.DEVICE = device
    object_detector = GeneralizedRCNNObjectDetector(cfg)

    model = MidasHKRMNet(
        object_detector,
        max_objects=max_objects,
        device=device,
        pretrained_resnet=load_weights,
    )

    if load_weights:
        logger.info("Loading pretrained model_weights")

        if midas_hkrm_weights:
            final_sd = midas_hkrm_weights
            model.load_state_dict(final_sd, strict=True)

        else:
            checkpoint = "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt"
            pretrained_sd = torch.hub.load_state_dict_from_url(
                checkpoint,
                map_location=torch.device("cpu"),
                progress=True,
                check_hash=True,
            )
            final_sd = pretrained_sd
            if random_init_missing:
                logger.info("Randomnly intializing missing model weights")
                model_sd = model.state_dict()
                final_sd = OrderedDict(zip(model_sd.keys(), pretrained_sd.values()))
                for k in final_sd.keys():
                    if "output_conv" in k or "refinenet" in k:
                        weights = final_sd[k]
                        m_shape = model_sd[k].shape
                        p_shape = final_sd[k].shape
                        for i in range(len(weights.shape)):
                            m = m_shape[i]
                            p = p_shape[i]
                            if p != m:
                                assert m > p
                                added_shape = list(weights.shape)
                                added_shape[i] = m - p
                                random_w = torch.randn(tuple(added_shape))
                                weights = torch.cat([weights, random_w], dim=i)
                        final_sd[k] = weights
                model.load_state_dict(final_sd, strict=True)
            else:
                model.load_state_dict(final_sd, strict=False)
    return model.to(device)
