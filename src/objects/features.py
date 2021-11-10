import abc
import enum

import detectron2.data.transforms as T
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model


class ObjectDetectionBackbone(enum.Enum):
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights

    MASK_RCNN_R_50_FPN = ("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml", \
                          "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    MASK_RCNN_R_101_FPN = ("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", \
                           "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")


class ObjectFeatureExtractor(abc.ABC):
    @abc.abstractmethod
    def __call__(self, images, img_format):
        raise not NotImplementedError()


class MRCNNFeatureExtractor(ObjectFeatureExtractor):
    def __init__(self, backbone: ObjectDetectionBackbone, score_threshold=0.5, device='gpu'):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(backbone.config))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(backbone.weights)
        cfg.MODEL.device = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

        self.model = build_model(cfg)

        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        self.cfg = cfg

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, images, img_format="BGR"):
        with torch.no_grad():
            # Input pre-processing
            if self.input_format != img_format:
                images = [image[::-1] for image in images]
            model_input = [
                {'image': torch.tensor(
                    self.aug.get_transform(image).apply_image(image).astype("float32").transpose(2, 0, 1)),
                    'height': image.shape[0],
                    'width': image.shape[1]}
                for image in images]
            preprocessed_images = self.model.preprocess_image(model_input)

            # compute the object features
            features = self.model.backbone(preprocessed_images.tensor)
            proposals, _ = self.model.proposal_generator(preprocessed_images, features, None)
            instances, _ = self.model.roi_heads(preprocessed_images, features, proposals)
            object_features = [features[f] for f in self.model.roi_heads.in_features]
            object_features = self.model.roi_heads.box_pooler(object_features, [x.pred_boxes for x in instances])
            object_features = self.model.roi_heads.box_head(object_features)

            return object_features, [i for i, instance in enumerate(instances)
                                     for _ in range(len(instance))]
