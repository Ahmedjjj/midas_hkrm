from src.utils.objects_utils import construct_config, HKRMTrainer
from detectron2.utils.logger import setup_logger
from src.objects.hkrm_roi_heads import HKRMROIHeads

import os

def main():
    cfg = construct_config()

    COCO_DATASET_SIZE = 117266 

    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.TEST.EVAL_PERIOD = COCO_DATASET_SIZE // cfg.SOLVER.IMS_PER_BATCH
    cfg.OUTPUT_DIR = '/ivrldata1/students/2021-fall-sp-jellouli/output'
    cfg.SOLVER.BASE_LR = 0.02 / 16 

    setup_logger(os.path.join(cfg.OUTPUT_DIR, 'logs'))

    trainer = HKRMTrainer(cfg)
    
    for m in trainer.model.backbone.bottom_up.parameters():
        m.requires_grad = False

    trainer.resume_or_load()

    trainer.train()


if __name__ == '__main__':
    main()
