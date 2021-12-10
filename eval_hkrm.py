import glob
import pickle
import os


import torch
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, build_detection_test_loader, DatasetMapper
from detectron2.evaluation import inference_on_dataset

from src.utils.objects_utils import construct_config
import src.objects
from detectron2.modeling import build_model

OUTPUT_DIR = '/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output'


def main():

    cfg = construct_config()
    model = build_model(cfg)

    evaluator = COCOEvaluator('coco_2017_val')
    data = DatasetCatalog['coco_2017_val']()
    dataset = build_detection_test_loader(dataset=data, mapper=DatasetMapper(cfg, is_train=False))

    models = glob.glob(os.path.join(OUTPUT_DIR, 'model_*.pth'))
    res = dict()
    with open(os.path.join('/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output', 'eval.pickle'), 'wb') as handle:
        for checkpoint_f in sorted(models):
            checkpoint = torch.load(checkpoint_f)
            model.load_state_dict(checkpoint['model'])
            res[checkpoint_f] = (inference_on_dataset(model, dataset, evaluator))

            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
