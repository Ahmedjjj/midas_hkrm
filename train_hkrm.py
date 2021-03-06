import os

from detectron2.utils.logger import setup_logger

from midas_hkrm.utils.objects_utils import HKRMTrainer, construct_config


def main():
    cfg = construct_config()

    COCO_DATASET_SIZE = 117266

    cfg.SOLVER.IMS_PER_BATCH = 10
    cfg.TEST.EVAL_PERIOD = 0
    cfg.OUTPUT_DIR = "/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output"
    cfg.SOLVER.BASE_LR = 0.01

    setup_logger(os.path.join(cfg.OUTPUT_DIR, "logs"))

    trainer = HKRMTrainer(cfg)

    for m in trainer.model.backbone.bottom_up.parameters():
        m.requires_grad = False

    trainer.resume_or_load()

    trainer.train()


if __name__ == "__main__":
    main()
