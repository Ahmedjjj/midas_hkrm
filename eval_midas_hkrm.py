import argparse
import logging
import os
import pickle
import sys

sys.path.append(os.path.normpath(os.path.join(".", "external", "MiDaS")))
import torch

from midas_hkrm.datasets import (
    HRWSI,
    ApolloScape,
    BlendedMVS,
    MegaDepth,
    RedWeb,
    TartanAir,
)
from midas_hkrm.depth import create_midas_hkrm_model
from midas_hkrm.optim import MidasHKRMTester, SSITrimmedMAELoss
from midas_hkrm.utils import midas_test_transform, midas_eval_transform, setup_logger
from midas_hkrm.zero_shot import ETH3DEvaluator, NyuV2Evaluator, TUMEvaluator

logger = logging.getLogger(__name__)
setup_logger()


def eval_model(
    object_detection_weights: str,
    object_detection_threshold: float,
    num_objects: int,
    midas_hkrm_state: str,
    base=False,
    nyu=False,
    eth=False,
    tum=False,
    test_set=False,
    device="cuda",
    save=False,
    save_dir=".",
):
    logger.info("Preparing model")
    logger.info(f"Object Detection weights {object_detection_weights}")
    logger.info(f"Object Detection threshold {object_detection_threshold}")
    logger.info(f"Num objects {num_objects}")

    logger.info(f"MidasHKRM state {midas_hkrm_state}")

    state = torch.load(midas_hkrm_state)
    model = create_midas_hkrm_model(
        max_objects=num_objects,
        object_detection_threshold=object_detection_threshold,
        load_weights=True,
        object_model_weights=object_detection_weights,
        random_init_missing=False,
        device=device,
        midas_hkrm_weights=state["model"],
        use_hkrm=not base,
    )

    eval_results = dict()

    if nyu:
        evaluator = NyuV2Evaluator(
            model=model, transform=midas_test_transform, device=device, pass_input=True
        )
        eval_results["nyu"] = evaluator.evaluate()
        logger.info(f"Loss on NYUv2: {eval_results['nyu']}")

    if eth:
        evaluator = ETH3DEvaluator(
            model=model, transform=midas_test_transform, device=device, pass_input=True
        )
        eval_results["eth"] = evaluator.evaluate()
        logger.info(f"Loss on ETH3D: {eval_results['eth']}")

    if tum:
        evaluator = TUMEvaluator(
            model=model, transform=midas_test_transform, device=device, pass_input=True
        )
        eval_results["tum"] = evaluator.evaluate()
        logger.info(f"Loss on TUM: {eval_results['tum']}")

    if test_set:
        tester = MidasHKRMTester(
            model=model,
            criterion=SSITrimmedMAELoss(),
            datasets=[
                ApolloScape(),
                TartanAir(),
                RedWeb(),
                BlendedMVS(),
                MegaDepth(),
                HRWSI(),
            ],
            test_transform=midas_eval_transform,
            seed=state["seed"],
            split_size=state["test_split_size"],
            device=device,
        )
        eval_results["test"] = tester.test()
        logger.info(f"Test set loss: {eval_results['test']}")

    if save:
        filename = os.path.splitext(os.path.basename(midas_hkrm_state))[0] + ".pickle"
        with open(os.path.join(save_dir, filename), "wb") as handle:
            pickle.dump(eval_results, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval a MidasHKRM model")
    parser.add_argument("--object_weights", "-o", required=True, type=str)
    parser.add_argument("--max_objects", "-m", default=15, type=int)
    parser.add_argument("--detection_threshold", "-t", default=0.5, type=float)
    parser.add_argument("--state", "-s", required=True, type=str)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nyu", action="store_true")
    parser.add_argument("--tum", action="store_true")
    parser.add_argument("--eth", action="store_true")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--test_set", action="store_true")
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    eval_model(
        object_detection_weights=args.object_weights,
        object_detection_threshold=args.detection_threshold,
        num_objects=args.max_objects,
        midas_hkrm_state=args.state,
        nyu=args.nyu,
        eth=args.eth,
        tum=args.tum,
        test_set=args.test_set,
        device="cpu" if args.cpu else "cuda",
        save=args.save_path is not None,
        save_dir=args.save_path,
    )
