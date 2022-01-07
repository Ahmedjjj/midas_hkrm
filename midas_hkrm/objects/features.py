import abc
from typing import List, Tuple

import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.modeling import build_model
from midas_hkrm.objects import HKRMROIHeads


class ObjectDetector(abc.ABC):
    """
    Abstraction for any object detection backbone that can provide feature vectors, object classes and object masks for
    an input image.
    """

    @abc.abstractmethod
    def get_object_features(self, imgs: List[np.array], outputs: List[str], **kwargs):
        """
        Extract object features, mask, classes from a list of images

        Args:
            imgs (List[np.array]): image list, should be in the same format as the model expected format.
            outputs (List[str]): a combination of ["features", "num_objects", "classes", "masks"]
                                . if "features" is present, return a tensor of feature vectors,
                                      return one tensor for all images in the batch (as done in detectron2).
                                . if "num_objects" is present, return a list with length equal to len(imgs),
                                      with the number of detected objects for each image.
                                . if "classes" is present, return a list of vectors, on for each image,
                                      with the class of each detected object.
                                . if "masks" is present return a list of tensors, one for each image,
                                      each tensor contains a mask for each detected object.
                                The order should be strictly "features", "num_objects", "classes", "masks".
        """
        raise NotImplementedError()

    @classmethod
    def get_object_mask(cls, img: np.ndarray, box_coords: Tuple[float]) -> torch.Tensor:
        """
        Given an input image, construct an object mask at the given coordinates.
        Args:
            img (np.ndarray): original image
            box_coords (Tuple[float]): box coordinates, should be in the format
                                       (upper_corner_x, upper_corner_y, lower_corner_x, lower_corner_y)
        Returns:
            torch.Tensor: object mask
        """
        res = torch.zeros(img.shape[:2])
        obj_coords = [int(coord) for coord in box_coords]
        x0, y0, x1, y1 = obj_coords
        res[y0:y1, x0:x1] = 1
        return res

    @classmethod
    def get_batch_object_masks(
        cls, img: np.ndarray, boxes_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Convenience function to get masks in batch fashion
        Args:
            img (np.ndarray): original image
            boxes_tensor (torch.Tensor): A 2D tensor of shape (num_objects, 4), one for each box.

        Returns:
            torch.Tensor: tensor of masks, of shape (len(boxes_tensor), H, W)
        """
        # num boxes x img_width x img_height tensor
        res = torch.zeros((boxes_tensor.shape[0], *img.shape[:2]))
        for index, box_coords in enumerate(boxes_tensor):
            res[index] = cls.get_object_mask(img, box_coords)

        return res


class GeneralizedRCNNObjectDetector(ObjectDetector):
    """
    A Wrapper around a detectron2 based object detector. Please see the GeneralizedRCNN architecture in detectron2.
    This class will:
        . load the model weights.
        . use the model in eval mode and run the evaluation with no gradients.
        . Optionally resize the input before passing it through the model.
    """

    def __init__(self, cfg: CfgNode, resize=False):
        """
        Args:
            cfg (CfgNode): detectron2 config node
            resize (bool, optional): if True, resize the input to the model. Defaults to False.
        """
        self.model = build_model(cfg)
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        self.device = cfg.MODEL.DEVICE
        self.cfg = cfg
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        self.resize = resize

    def get_object_features(self, imgs: List[np.array], outputs: List[str], **kwargs):
        with torch.no_grad():
            model_input = []
            for img in imgs:
                height, width = img.shape[:2]
                if self.resize:
                    img = self.aug.get_transform(img).apply_image(img)
                img = img.astype("float32").transpose(2, 0, 1)
                model_input.append(
                    {
                        "image": torch.tensor(img, device=self.device),
                        "height": height,
                        "width": width,
                    }
                )

            preprocessed_images = self.model.preprocess_image(model_input)

            # Backbone
            features = self.model.backbone(preprocessed_images.tensor)

            # Proposals
            proposals, _ = self.model.proposal_generator(
                preprocessed_images, features, None
            )

            # Final instances
            instances, _ = self.model.roi_heads(
                preprocessed_images, features, proposals
            )

            # small detail, rescale the output instances
            instances_r = [
                x["instances"]
                for x in self.model._postprocess(
                    instances, model_input, preprocessed_images.image_sizes
                )
            ]

            final_output = []

            if "features" in outputs:
                object_features = [
                    features[f] for f in self.model.roi_heads.in_features
                ]
                object_features = self.model.roi_heads.box_pooler(
                    object_features, [x.pred_boxes for x in instances]
                )

                if isinstance(self.model.roi_heads, HKRMROIHeads):
                    object_features = self.model.roi_heads.box_head(
                        object_features, instances
                    )
                else:
                    object_features = self.model.roi_heads.box_head(object_features)

                final_output.append(object_features)

            if "num_objects" in outputs:
                num_objects = [len(i) for i in instances]
                final_output.append(num_objects)

            if "classes" in outputs:
                classes = [
                    instance.get_fields()["pred_classes"] for instance in instances
                ]
                final_output.append(classes)

            if "masks" in outputs:
                masks = []
                for img, instance in zip(imgs, instances_r):
                    boxes = instance.get_fields()["pred_boxes"].tensor
                    masks.append(
                        super().get_batch_object_masks(img, boxes).to(self.device)
                    )

                final_output.append(masks)

            return tuple(final_output)
