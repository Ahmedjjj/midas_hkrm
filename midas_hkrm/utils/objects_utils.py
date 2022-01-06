from pathlib import Path

import torch.nn as nn
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode
from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from midas_hkrm.utils.errors import require

FILE_FOLDER = Path(__file__).parent


def get_baseline_config() -> CfgNode:
    """
    Returns the config for the pre-trained detectron2 faster_rcnn_R_101_FPN_3x.yaml model.
    """
    cfg = get_cfg()

    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )
    return cfg


def construct_config(
    save=False,
    save_path=None,
    relationship_matrix_path: str = None,
    attribute_matrix_path: str = None,
) -> CfgNode:
    """
    construct default config for the modified HKRM model

    Args:
        save (bool, optional): whether to save the config. Defaults to False.
        save_path ([type], optional): where to save the config. Defaults to None.
        relationship_matrix_path (str, optional): path to the relationship matrix ground truth.
                                                  if None, defaults to ../../data/relationship_matrices/COCO_graph_r.pkl
                                                  (convenience for our project).
                                                  Defaults to None.
        attribute_matrix_path (str, optional): same as relationship_matrix_path.

    Returns:
        CfgNode: config for the model.
    """

    require(not save or (save and save_path))

    cfg = get_cfg()

    # Our model is a modification of the Faster-RCNN
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    )

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    )

    cfg.MODEL.ROI_HEADS.NAME = "HKRMROIHeads"
    cfg.MODEL.HKRM = CfgNode()

    if relationship_matrix_path is None:
        relationship_matrix_path = str(
            (
                FILE_FOLDER.parent.parent
                / "data"
                / "relationship_matrices"
                / "COCO_graph_r.pkl"
            ).absolute()
        )

    if attribute_matrix_path is None:
        attribute_matrix_path = str(
            (
                FILE_FOLDER.parent.parent
                / "data"
                / "relationship_matrices"
                / "COCO_graph_a.pkl"
            ).absolute()
        )

    cfg.MODEL.HKRM.RELATIONSHIP_PATH = relationship_matrix_path
    cfg.MODEL.HKRM.ATTRIB_PATH = attribute_matrix_path

    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 128  # like the paper
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 128

    if save:
        with open(save_path, "w") as f:
            f.write(cfg.dump())

    return cfg


def get_hkrm_model() -> nn.Module:
    """
    Build the HKRM model from the default config

    Returns:
        nn.Module: model
    """
    cfg = construct_config()
    return build_model(cfg)


class HKRMTrainer(DefaultTrainer):
    """
    This is a trainer that runs an evaluation of the COCO val dataset periodically
    Please see detectron2's DefaultTrainer for documentation
    """

    def __init__(self, cfg: CfgNode):
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
