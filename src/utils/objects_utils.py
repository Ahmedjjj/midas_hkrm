# Utils for training the HKRM based object detection network

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config.config import CfgNode
from detectron2.modeling import build_model

from pathlib import Path
from src.utils.errors import require

FILE_FOLDER = Path(__file__).parent


def construct_config(save=False, save_path=None):

    require(not save or (save and save_path))

    cfg = get_cfg()

    # Our model is a modification of the Faster-RCNN
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))

    cfg.MODEL.ROI_HEADS.NAME = "HKRMROIHeads"
    cfg.MODEL.HKRM = CfgNode()

    cfg.MODEL.HKRM.RELATIONSHIP_PATH = str(
        (FILE_FOLDER.parent.parent / 'data' / 'relationship_matrices' / 'COCO_graph_r.pkl').absolute())
    cfg.MODEL.HKRM.ATTRIB_PATH = str(
        (FILE_FOLDER.parent.parent / 'data' / 'relationship_matrices' / 'COCO_graph_a.pkl').absolute())

    if save:
        with open(save_path, 'w') as f:
            f.write(cfg.dump())

    return cfg


def get_hkrm_model():
    cfg = construct_config()
    return build_model(cfg)
