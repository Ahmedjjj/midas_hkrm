import logging
import operator
import pickle
from functools import reduce
from itertools import product

from torch.optim import Adam
import torch
from torch.utils.data import random_split

from midas_hkrm.datasets import (
    HRWSI,
    ApolloScape,
    BlendedMVS,
    MegaDepth,
    RedWeb,
    TartanAir,
)
from midas_hkrm.optim import MidasHKRMTrainer, SSITrimmedMAELoss
from midas_hkrm.utils import (
    midas_eval_transform,
    midas_train_transform,
    setup_logger,
)
from midas_hkrm.depth import create_midas_hkrm_model

DEBUG = False
TEST_AFTER = 300
DEVICE = "cuda"
OBJECT_DETECTION_WEIGHTS = (
    "/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output/model_final.pth"
)
TEST_SPLIT_SIZE = 0.02
BATCH_SIZE = 1
SEED = 42

logger = logging.getLogger(__name__)
setup_logger(debug=DEBUG)


def main():
    lrs = [(1e-2, 1e-3), (1e-3, 1e-4), (1e-4, 1e-5)]
    object_detection_thresholds = [0.1, 0.3, 0.5]
    max_objects = range(10, 20, 3)
    test_losses = dict()

    criterion = SSITrimmedMAELoss()

    datasets = [
        ApolloScape(),
        TartanAir(),
        RedWeb(),
        BlendedMVS(),
        MegaDepth(),
        HRWSI(),
    ]

    final_datasets = []
    for dataset in datasets:
        test_size = int(len(dataset) * TEST_SPLIT_SIZE)
        train_size = len(dataset) - test_size
        _, train_dataset = random_split(
            dataset,
            [test_size, train_size],
            generator=torch.Generator().manual_seed(SEED),
        )
        train_dataset.__setattr__("name", dataset.name)
        final_datasets.append(train_dataset)

    for (lr_new, lr_old), thresh, max_o in product(
        lrs, object_detection_thresholds, max_objects
    ):
        logger.info(
            f"""Trying combination:
                    lr_new : {lr_new}, 
                    lr_old:{lr_old}, 
                    thresh: {thresh}, 
                    max object: {max_o}"""
        )

        model = create_midas_hkrm_model(
            max_objects=max_o,
            object_detection_threshold=thresh,
            object_model_weights=OBJECT_DETECTION_WEIGHTS,
            device=DEVICE,
        )

        for m in model.backbone.parameters():
            m.requires_grad = False

        optimizer = Adam(
            [
                {
                    "params": model.channel_reduc.parameters(),
                    "lr": 1e-5,
                },
                {"params": model.refinenet_1.parameters()},
                {"params": model.refinenet_2.parameters()},
                {"params": model.refinenet_3.parameters()},
                {"params": model.refinenet_4.parameters()},
                {"params": model.output_conv.parameters()},
            ],
            lr=lr_new,
            betas=(0.9, 0.999),
        )

        trainer = MidasHKRMTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            datasets=final_datasets,
            train_transform=midas_train_transform,
            test_transform=midas_eval_transform,
            save=False,
            test=True,
            test_split_size=TEST_SPLIT_SIZE,
            same_test_size=True,
            test_after=TEST_AFTER,
            batch_size=BATCH_SIZE,
            test_batch_size=BATCH_SIZE,
            device=DEVICE,
            max_iter=TEST_AFTER,
            seed=SEED,
        )

        final_state = trainer.train()
        test_losses[(lr_new, lr_old, thresh, max_objects)] = final_state["test_losses"][
            TEST_AFTER - 1
        ]
        mean_loss = reduce(
            operator.add, final_state["test_losses"][TEST_AFTER - 1].values()
        ) / len(datasets)
        logger.info(f"Mean loss: {mean_loss}")

    with open("res.pickle", "wb") as f:
        pickle.dump(test_losses, f)


if __name__ == "__main__":
    main()
