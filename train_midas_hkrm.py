import logging
import argparse

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

OBJECT_DETECTION_WEIGHTS = (
    "/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output/model_final.pth"
)

SAVE_DIR = "/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output_midas_hkrm_v4"


SAVE_AFTER = 10000

BATCH_SIZE = 1

DEBUG = False

logger = logging.getLogger(__name__)
setup_logger(debug=DEBUG)


def train(
    max_objects: int,
    object_detection_threshold: float,
    use_hkrm: bool,
    lr_old: float,
    lr_new: float,
    max_iter: int = 300000,
    hkrm_weights: str = None,
    save: bool = True,
    output_dir: str = None,
    save_after: int = 10000,
    device: int = "cuda",
    seed: int = 42,
    test: bool = True,
    test_split_size: float = 0.1,
    test_after: int = 300000,
    load_state: bool = False,
):

    model = create_midas_hkrm_model(
        max_objects=max_objects,
        object_detection_threshold=object_detection_threshold,
        object_model_weights=hkrm_weights if use_hkrm else None,
        use_hkrm=use_hkrm,
    )

    logger.info("Freezing encoder weights")
    for m in model.backbone.parameters():
        m.requires_grad = False

    logger.info("Preparing Adam optimizer")
    optimizer = Adam(
        [
            {
                "params": model.channel_reduc.parameters(),
                "lr": lr_old,
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

    if load_state:
        trainer = MidasHKRMTrainer(
            model=model,
            criterion=loss_func,
            optimizer=optimizer,
            datasets=datasets,
            train_transform=midas_train_transform,
            test_transform=midas_eval_transform,
            save_path=output_dir,
        )
        trainer.load_state()
        trainer.test_after = test_after
        trainer.device = device

    else:
        trainer = MidasHKRMTrainer(
            model=model,
            criterion=loss_func,
            optimizer=optimizer,
            datasets=datasets,
            train_transform=midas_train_transform,
            test_transform=midas_eval_transform,
            batch_size=BATCH_SIZE,
            save=save,
            save_path=output_dir,
            save_after=save_after,
            test=test,
            test_after=test_after,
            test_split_size=test_split_size,
            test_batch_size=BATCH_SIZE,
            max_iter=max_iter,
            seed=seed,
            device=device,
        )
    final_state = trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MidasHKRM model")
    parser.add_argument("--max_obj", "-m", type=int, required=True)
    parser.add_argument("--threshold", "-t", type=float, required=True)
    parser.add_argument("--lr_old", type=float, required=True)
    parser.add_argument("--lr_new", type=float, required=True)

    parser.add_argument("--max_iter", type=int, default=300000)

    parser.add_argument("--base", action="store_true")
    parser.add_argument("--obj_weights", type=str)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_after", type=int, default=1000)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--test_after", type=int, default=300000)

    parser.add_argument("--load_state", action="store_true")

    args = parser.parse_args()
    train(
        max_objects=args.max_obj,
        object_detection_threshold=args.threshold,
        use_hkrm=not args.base,
        lr_old=args.lr_old,
        lr_new=args.lr_new,
        max_iter=args.max_iter,
        hkrm_weights=args.obj_weights,
        save=args.save,
        output_dir=args.output_dir,
        save_after=args.save_after,
        device=args.device,
        seed=args.seed,
        test=args.test,
        test_split_size=args.test_size,
        test_after=args.test_after,
        load_state=args.load_state,
    )
