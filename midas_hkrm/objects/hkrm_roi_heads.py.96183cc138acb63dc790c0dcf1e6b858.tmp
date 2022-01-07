import pickle
from typing import Dict, List, Optional, Tuple, Union
from detectron2.config.config import CfgNode

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, ROIHeads
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures.image_list import ImageList
from detectron2.structures.instances import Instances
from midas_hkrm.utils.errors import require
from torch import nn

"""
Modified HKRM model code. The model relies on many componenents from detectron2.
Please see the report, or the HKRM paper for the details of the model: https://arxiv.org/abs/1810.12681
The modified HKRM uses a Feature Pyramid Network as a RoIPooling layer, and does not make use of the Implicit
Knowledge Modules from the original paper.
"""


class ExplicitFeatureRelationshipModule(nn.Module):
    """
    Explicit Knowledge Routed Module.
    Uses semantic information about the relationships of the object classes in order to do object feature transformation,
    after the ROIPooling layer
    """

    def __init__(
        self,
        input_shape: int,
        num_layers: int,
        layer_sizes: List[int],
        transformed_feature_size: int,
        knowledge_matrix: np.ndarray = None,
    ):
        """
        Args:
            input_shape (int): feature vector input shape.
            num_layers (int): number of hidden layers in the FFN.
            layer_sizes (List[int]): hidden layer sizes, should have length equal to num_layers.
            transformed_feature_size (int): target feature dimension.
            knowledge_matrix (np.ndarray, optional): the external original knowledge matrix.
                                                     Needs to be provided only at train time.
                                                     Defaults to None.
        """
        super(ExplicitFeatureRelationshipModule, self).__init__()

        require(
            len(layer_sizes) == num_layers,
            "number of layers doesn't match number of layer sizes",
        )

        self.relationship_scorer = nn.Sequential()
        layer_sizes = [input_shape] + layer_sizes

        for layer_index in range(1, len(layer_sizes)):
            self.relationship_scorer.add_module(
                f"fc_{layer_index}",
                nn.Sequential(
                    nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]),
                    nn.ReLU(),
                ),
            )

        self.relationship_scorer.add_module("fc_output", nn.Linear(layer_sizes[-1], 1))

        # Simple linear transformation to desired dimension
        self.feature_transform = nn.Linear(input_shape, transformed_feature_size)

        self.knowledge_matrix = knowledge_matrix

    def forward(
        self, features: torch.Tensor, gt_classes: torch.Tensor = None
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        num_objects, num_features = features.shape

        # Compute pairwise L1 difference between feature vectors
        dist = torch.abs(features.unsqueeze(-2) - features.unsqueeze(0)).reshape(
            -1, num_features
        )

        # predicted_scores (using the FFN)
        scores = self.relationship_scorer(dist).reshape(-1)

        # transform features (E F W in the paper)
        transformation_matrix = F.softmax(scores.reshape(num_objects, num_objects), -1)
        transformed_features = self.feature_transform(
            torch.matmul(transformation_matrix, features)
        )

        if self.training:
            # ground truth scores
            gt_scores = self.knowledge_matrix[
                np.repeat(gt_classes.cpu(), num_objects),
                np.tile(gt_classes.cpu(), num_objects),
            ]
            loss = F.mse_loss(
                scores,
                torch.tensor(gt_scores, dtype=scores.dtype, device=scores.device),
            )
            return transformed_features, loss

        else:
            return transformed_features


class HKRMBoxHead(nn.Module):
    """
    HKRM based box head. Given a tensor of features use two Knowledge Routed Modules
    (attribute and relationship based) in order to transform the features using semantic information.
    The box head will keep the input dimension, meaning that the output features will have the same dimension as the input features.
    """

    def __init__(
        self,
        base_box_head: nn.Module,
        attribute_knowledge_matrix: np.array = None,
        relationship_knowledge_matrix: np.array = None,
        device="cuda",
    ):
        """
        Args:
            base_box_head (nn.Module): the base box head outputs the feature vectors.
                                       This module will be used after the RoIPooling layer in order to
                                       map the multi-scale region features into a feature vector.
            attribute_knowledge_matrix (np.array, optional): Knowledge matrix.
                                                             Should be symmetric and square.
                                                             Needs to be provided at train time.
                                                             Defaults to None.
            relationship_knowledge_matrix (np.array, optional): same as the previous parameter. Defaults to None.
            device (str, optional): device of the box head. Defaults to "cuda".
        """

        super(HKRMBoxHead, self).__init__()

        self.base_box_head = base_box_head
        in_features = base_box_head.output_shape.channels

        self.attribute_feature_transform = ExplicitFeatureRelationshipModule(
            input_shape=in_features,
            num_layers=3,
            layer_sizes=[256, 128, 64],  # same layer sizes as the paper
            transformed_feature_size=in_features // 2,
            knowledge_matrix=attribute_knowledge_matrix,
        )

        self.relationship_feature_transform = ExplicitFeatureRelationshipModule(
            input_shape=in_features,
            num_layers=3,
            layer_sizes=[256, 128, 64],
            transformed_feature_size=in_features // 2,
            knowledge_matrix=relationship_knowledge_matrix,
        )
        self.device = device

    def _forward_inference(
        self, features: torch.Tensor, proposals: List[Instances]
    ) -> torch.Tensor:
        """
        Test time inference. Can be regarded as a form of feature transformation.

        Args:
            features (torch.Tensor): base features
            proposals (List[Instances]): Region proposals (output of the RPN)

        Returns:
            torch.Tensor: transformed features
        """
        with torch.no_grad():
            start = 0
            base_box_head_features = self.base_box_head(features).float()

            result_features = torch.empty(
                0,
                *base_box_head_features.shape[1:],
                dtype=base_box_head_features.dtype,
                device=self.device,
            )

            for instance in proposals:
                image_features = base_box_head_features[start : start + len(instance)]
                start += len(instance)
                attrib_transformed_features = self.attribute_feature_transform(
                    image_features
                )
                relation_transformed_features = self.relationship_feature_transform(
                    image_features
                )
                image_transformed_features = torch.cat(
                    (attrib_transformed_features, relation_transformed_features), dim=1
                )

                result_features = torch.cat(
                    (result_features, image_transformed_features), dim=0
                )

            return result_features

    def forward(
        self, features: torch.Tensor, proposals: List[Instances]
    ) -> Union[torch.Tensor, Tuple[torch.tensor, Dict[str, torch.Tensor]]]:
        if not self.training:
            return self._forward_inference(features, proposals)

        attrib_loss = 0
        relationship_loss = 0

        start = 0

        # base box head gives us a vector for each object
        base_box_head_features = self.base_box_head(features).float()

        result_features = torch.empty(
            0,
            *base_box_head_features.shape[1:],
            dtype=base_box_head_features.dtype,
            device=self.device,
        )

        for instance in proposals:
            image_features = base_box_head_features[start : start + len(instance)]
            start += len(instance)
            gt_classes = instance.gt_classes

            (
                attrib_transformed_features,
                attrib_loss_img,
            ) = self.attribute_feature_transform(image_features, gt_classes)

            (
                relation_transformed_features,
                relation_loss_img,
            ) = self.relationship_feature_transform(image_features, gt_classes)

            attrib_loss += attrib_loss_img
            relationship_loss += relation_loss_img

            image_transformed_features = torch.cat(
                (attrib_transformed_features, relation_transformed_features), dim=1
            )

            result_features = torch.cat(
                (result_features, image_transformed_features), dim=0
            )

        return result_features, {
            "attrib_module_loss": attrib_loss,
            "relation_module_loss": relation_loss_img,
        }


@ROI_HEADS_REGISTRY.register()
class HKRMROIHeads(ROIHeads):
    """
    HKRM based RoIHeads. Follows the detectron2 interface for ROIHeads.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

    @classmethod
    def _init_hkrm_box_head(cls, cfg: CfgNode, input_shape: int):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in in_features]
        assert len(set(in_channels)) == 1, in_channels

        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        base_box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        relationship_matrix_path = cfg.MODEL.HKRM.RELATIONSHIP_PATH

        attrib_matrix_path = cfg.MODEL.HKRM.ATTRIB_PATH

        with open(relationship_matrix_path, "rb") as f:
            relationship_matrix = pickle.load(f)

        with open(attrib_matrix_path, "rb") as f:
            attrib_matrix = pickle.load(f)

        hkrm_box_head = HKRMBoxHead(
            base_box_head=base_box_head,
            attribute_knowledge_matrix=attrib_matrix,
            relationship_knowledge_matrix=relationship_matrix,
            device=cfg.MODEL.DEVICE,
        )

        box_predictor = FastRCNNOutputLayers(cfg, base_box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": hkrm_box_head,
            "box_predictor": box_predictor,
        }

    @classmethod
    def from_config(cls, cfg, input_shape):
        super_config = super().from_config(cfg)
        super_config.update(cls._init_hkrm_box_head(cfg, input_shape))

        return super_config

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        losses = {}
        if self.training:
            box_features, relation_loss = self.box_head(box_features, proposals)
            losses.update(relation_loss)
        else:
            box_features = self.box_head(box_features, proposals)

        predictions = self.box_predictor(box_features)
        if self.training:
            losses.update(self.box_predictor.losses(predictions, proposals))
            return predictions, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, {}
