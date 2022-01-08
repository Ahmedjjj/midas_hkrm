import argparse

import torch
from detectron2.data import DatasetCatalog, DatasetMapper, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model

import midas_hkrm.objects  # needed to register the HKRMROIHeads
from midas_hkrm.utils.objects_utils import construct_config

OUTPUT_DIR = "/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output"


def eval_hkrm(model_state_path):
    # Load checkpoint
    cfg = construct_config()
    model = build_model(cfg)
    checkpoint = torch.load(model_state_path)
    model.load_state_dict(checkpoint["model"])

    evaluator = COCOEvaluator("coco_2017_val")
    data = DatasetCatalog["coco_2017_val"]()
    dataset = build_detection_test_loader(
        dataset=data, mapper=DatasetMapper(cfg, is_train=False)
    )
    inference_on_dataset(model, dataset, evaluator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval a HKRM model")
    parser.add_argument("model_state", type=str)
    args = parser.parse_args()
    eval_hkrm(args.model_state)
