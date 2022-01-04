import logging

import torch
from torch.optim import Adam

from midas_hkrm.datasets import (
    HRWSI,
    ApolloScape,
    BlendedMVS,
    MegaDepth,
    RedWeb,
    TartanAir,
)
from midas_hkrm.depth.midas_hkrm import create_midas_hkrm_model
from midas_hkrm.optim import SSITrimmedMAELoss
from midas_hkrm.optim.trainer import MidasHKRMTrainer
from midas_hkrm.utils import midas_eval_transform, midas_train_transform, setup_logger

MAX_OBJECTS = 15
OBJECT_DETECTION_WEIGHTS = (
    "/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output/model_final.pth"
)

SAVE_DIR = "/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output_midas_obj_baseline"
START_STATE = "/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output_midas_hkrm/state_290000.tar"

DEVICE = "cuda"

SAVE_AFTER = 10000
EVAL_AFTER = 100000
STEPS = 300000

TEST_SET_SPLIT = 0.1
BATCH_SIZE = 1

SEED = 42
DEBUG = False

logger = logging.getLogger(__name__)
setup_logger(debug=DEBUG)


def main():
    model = create_midas_hkrm_model(
        max_objects=MAX_OBJECTS,
        object_detection_threshold=0.5,
        use_hkrm=False,
    )

    logger.info("Freezing encoder weights")
    for m in model.backbone.parameters():
        m.requires_grad = False

    logger.info("Preparing Adam optimizer")
    optimizer = Adam(
        [
            {
                "params": model.channel_reduc.parameters(),
                "lr": 1e-6,
            },
            {"params": model.refinenet_1.parameters()},
            {"params": model.refinenet_2.parameters()},
            {"params": model.refinenet_3.parameters()},
            {"params": model.refinenet_4.parameters()},
            {"params": model.output_conv.parameters()},
        ],
        lr=1e-5,
        betas=(0.9, 0.999),
    )

    logger.info("Preparing loss function (Scale and Shift Invariant Trimmed MAE)")
    loss_func = SSITrimmedMAELoss()

    datasets = [
        ApolloScape(),
        TartanAir(),
        RedWeb(),
        BlendedMVS(),
        MegaDepth(),
        HRWSI(),
    ]

    trainer = MidasHKRMTrainer(
        model=model,
        criterion=loss_func,
        optimizer=optimizer,
        datasets=datasets,
        train_transform=midas_train_transform,
        test_transform=midas_eval_transform,
        batch_size=BATCH_SIZE,
        save=True,
        save_path=SAVE_DIR,
        save_after=SAVE_AFTER,
        test=True,
        test_after=EVAL_AFTER,
        test_split_size=TEST_SET_SPLIT,
        test_batch_size=BATCH_SIZE,
        max_iter=STEPS,
        seed=SEED,
        device=DEVICE,
    )
    final_state = trainer.train()


if __name__ == "__main__":
    main()
