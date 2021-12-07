import abc
import enum
from typing import List

import detectron2.data.transforms as T
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.config import CfgNode
from detectron2.modeling import build_model
import numpy as np


class ObjectDetector(abc.ABC):

    @abc.abstractmethod
    def get_object_features(self, imgs: List[np.array], outputs: List[str], **kwargs):
        """
        Extract object features, mask, classes from a list of images

        Args:
            imgs (List[np.array]): image list, should be in the same format as the model expected format
            outputs (List[str]): a combination of ["features", "classes", "masks"]

        """
        raise NotImplementedError()

    @classmethod
    def get_object_mask(cls, img, box_coords):
        res = torch.zeros(img.shape[:2])
        obj_coords = [int(coord) for coord in box_coords]
        print(obj_coords)
        x0, y0, x1, y1 = obj_coords
        res[y0: y1, x0:x1] = 1
        return res

    @classmethod
    def get_batch_object_masks(cls, img, boxes_tensor):
        # num boxes x img_width x img_height tensor
        res = torch.zeros((boxes_tensor.shape[0], *img.shape[:2]))
        for index, box_coords in enumerate(boxes_tensor):
            res[index] = cls.get_object_mask(img, box_coords)

        return res


class GeneralizedRCNNObjectDetector(ObjectDetector):
    """ A Wrapper around a detectron2 based object detector
    """

    def __init__(self, cfg: CfgNode):
        self.model = build_model(cfg)
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        self.device = cfg.MODEL.DEVICE
        self.cfg = cfg
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def get_object_features(self, imgs: List[np.array], outputs: List[str], **kwargs):
        with torch.no_grad():
            model_input = []
            for img in imgs:
                height, width = img.shape[:2]
                img = self.aug.get_transform(img).apply_image(img).astype("float32").transpose(2, 0, 1)
                model_input.append({"image": torch.tensor(img, device=self.device),
                                    "height": height, "width": width})

            preprocessed_images = self.model.preprocess_image(model_input)

            # Backbone
            features = self.model.backbone(preprocessed_images.tensor)

            # Proposals
            proposals, _ = self.model.proposal_generator(preprocessed_images, features, None)

            # Final instances
            instances, _ = self.model.roi_heads(preprocessed_images, features, proposals)

            # small detail, rescale the output instances
            instances = [x["instances"] for x in self.model._postprocess(
                instances, model_input, preprocessed_images.image_sizes)]

            final_output = []

            if "features" in outputs:
                object_features = [features[f] for f in self.model.roi_heads.in_features]
                object_features = self.model.roi_heads.box_pooler(object_features, [x.pred_boxes for x in instances])
                object_features = self.model.roi_heads.box_head(object_features)

                indices = [len(i) for i in instances]

                final_output.append(object_features)
                final_output.append(indices)

            if "classes" in outputs:
                classes = [instance.get_fields()['pred_classes'] for instance in instances]
                final_output.append(classes)

            if "masks" in outputs:
                masks = []
                for img, instance in zip(imgs, instances):
                    boxes = instance.get_fields()['pred_boxes'].tensor
                    masks.append(super().get_batch_object_masks(img, boxes).to(self.device))

                final_output.append(masks)

            return final_output
