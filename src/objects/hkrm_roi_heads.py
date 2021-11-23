from typing import Dict, List, Iterable, Optional, Tuple
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
import pickle
from detectron2.structures.image_list import ImageList
from detectron2.structures.instances import Instances

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY, ROIHeads

from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import ShapeSpec

from src.utils.errors import require


class ExplicitFeatureRelationshipModule(nn.Module):
    def __init__(
        self,
        input_shape,
        num_layers,
        layer_sizes,
        transformed_feature_size,
        knowledge_matrix=None,
    ):
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
                    nn.Linear(layer_sizes[layer_index - 1],
                              layer_sizes[layer_index]),
                    nn.ReLU(),
                ),
            )

        self.relationship_scorer.add_module(
            "fc_output", nn.Linear(layer_sizes[-1], 1))

        # Simple linear projection to desired dimension
        self.feature_transform = nn.Linear(
            input_shape, transformed_feature_size)

        self.knowledge_matrix = knowledge_matrix

    def forward(self, features: torch.tensor, gt_classes=None):
        num_objects, num_features = features.shape

        # Compute pairwise L1 difference between feature vectors
        dist = torch.abs(features.unsqueeze(-2) -
                         features.unsqueeze(0)).reshape(-1, num_features)

        # predicted_scores
        scores = self.relationship_scorer(dist)

        # transform features (E F W in the paper)
        transformation_matrix = F.softmax(
            scores.reshape(num_objects, num_objects), -1)
        transformed_features = self.feature_transform(
            torch.matmul(transformation_matrix, features))

        if self.training:
            # ground truth scores
            gt_scores = self.knowledge_matrix[np.repeat(
                gt_classes, num_objects), np.tile(gt_classes, num_objects)]
            loss = F.mse_loss(scores, gt_scores)
            return transformed_features, loss
        else:
            return transformed_features


class HKRMBoxHead(nn.Module):
    def __init__(
        self,
        base_box_head: nn.Module,
        attribute_knowledge_matrix: np.array = None,
        relationship_knowledge_matrix: np.array = None,
    ):

        super(HKRMBoxHead, self).__init__()

        self.base_box_head = base_box_head
        in_features = base_box_head.output_shape.channels

        # for now the feature transforms are hardcoded, since they are not the real point of the project
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

    def _forward_inference(self, features, proposals):
        with torch.no_grad():
            start = 0
            result_features = torch.empty(
                0, *features.shape[1:], dtype=features.dtype)
            base_box_head_features = self.base_box_head(features).float()
            for instance in proposals:
                image_features = base_box_head_features[start:start + len(
                    instance)]
                start += len(instance)
                attrib_transformed_features = self.attribute_feature_transform(
                    image_features)
                relation_transformed_features = self.relationship_feature_transform(
                    image_features)
                image_transformed_features = torch.cat(
                    (attrib_transformed_features, relation_transformed_features), dim=1)

                result_features = torch.cat(
                    (result_features, image_transformed_features), dim=0)
            return result_features

    def forward(self, features, proposals):
        if not self.training:
            return self._forward_inference(features, proposals)

        attrib_loss = 0
        relationship_loss = 0

        start = 0
        # base box head gives us a vector for each object
        result_features = torch.empty(
            0, *features.shape[1:], dtype=features.dtype)
        base_box_head_features = self.base_box_head(features).float()

        for instance in proposals:
            image_features = base_box_head_features[start:start +
                                                    len(instance)]
            start += len(instance)
            gt_classes = instance.gt_classes

            attrib_transformed_features, attrib_loss_img = self.attribute_feature_transform(
                image_features, gt_classes)

            relation_transformed_features, relation_loss_img = self.relationship_feature_transform(
                image_features, gt_classes)

            attrib_loss += attrib_loss_img
            relationship_loss += relation_loss_img

            result_features = torch.cat(
                result_features,
                torch.cat(attrib_transformed_features,
                          relation_transformed_features,
                          dim=1),
                dim=0,
            )

        return result_features, {
            "attrib_module_loss": attrib_loss,
            "relation_module_loss": relation_loss_img,
        }


@ROI_HEADS_REGISTRY.register()
class HKRMROIHeads(ROIHeads):

    @configurable
    def __init__(self,
                 *,
                 box_in_features: List[str],
                 box_pooler: ROIPooler,
                 box_head: nn.Module,
                 box_predictor: nn.Module,
                 **kwargs):

        super().__init__(**kwargs)

        self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor

    @classmethod
    def _init_hkrm_box_head(cls, cfg, input_shape):

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
            cfg, ShapeSpec(channels=in_channels,
                           height=pooler_resolution, width=pooler_resolution)
        )
        relationship_matrix_path = cfg.MODEL.HKRM.RELATIONSHIP_PATH

        attrib_matrix_path = cfg.MODEL.HKRM.ATTRIB_PATH

        with open(relationship_matrix_path, 'rb') as f:
            relationship_matrix = pickle.load(f)

        with open(attrib_matrix_path, 'rb') as f:
            attrib_matrix = pickle.load(f)

        hkrm_box_head = HKRMBoxHead(base_box_head=base_box_head, attribute_knowledge_matrix=attrib_matrix,
                                    relationship_knowledge_matrix=relationship_matrix)

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

    def forward(self, images: ImageList,
                features: Dict[str, torch.Tensor],
                proposals: List[Instances],
                targets: Optional[List[Instances]] = None) -> Tuple[List[Instances],
                                                                    Dict[str, torch.Tensor]]:
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals])
        losses = {}
        if self.training:
            relation_loss, box_features = self.box_head(
                box_features, proposals)
            losses.update(relation_loss)
        else:
            box_features = self.box_head(box_features, proposals)

        predictions = self.box_predictor(box_features)
        if self.training:
            losses.update(self.box_predictor.losses(predictions, proposals))
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(
                predictions, proposals)
            return pred_instances
