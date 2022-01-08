import argparse
import logging
import os
import pickle
import sys

sys.path.append(os.path.normpath(os.path.join(".", "external", "MiDaS")))
import torch
from midas.midas_net import MidasNet

from midas_hkrm.utils import midas_test_transform, setup_logger
from midas_hkrm.zero_shot import ETH3DEvaluator, NyuV2Evaluator, TUMEvaluator

logger = logging.getLogger(__name__)
setup_logger()


def eval_model(
    nyu=False,
    eth=False,
    tum=False,
    device="cuda",
    save=False,
    save_dir=".",
):
    logger.info("Preparing model")

    model = MidasNet()

    checkpoint = (
        "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt"
    )
    state_dict = torch.hub.load_state_dict_from_url(
        checkpoint, map_location=torch.device("cpu"), progress=True, check_hash=True
    )
    model.load_state_dict(state_dict)

    eval_results = dict()

    if nyu:
        evaluator = NyuV2Evaluator(
            model=model, transform=midas_test_transform, device=device, pass_input=False
        )
        eval_results["nyu"] = evaluator.evaluate()
        logger.info(f"Loss on NYUv2: {eval_results['nyu']}")

    if eth:
        evaluator = ETH3DEvaluator(
            model=model, transform=midas_test_transform, device=device, pass_input=False
        )
        eval_results["eth"] = evaluator.evaluate()
        logger.info(f"Loss on ETH3D: {eval_results['eth']}")

    if tum:
        evaluator = TUMEvaluator(
            model=model, transform=midas_test_transform, device=device, pass_input=False
        )
        eval_results["tum"] = evaluator.evaluate()
        logger.info(f"Loss on TUM: {eval_results['tum']}")

    if save:
        filename = "eval_midas_v2.1" + ".pickle"
        with open(os.path.join(save_dir, filename), "wb") as handle:
            pickle.dump(eval_results, handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Midas V2.1 model")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--nyu", action="store_true")
    parser.add_argument("--tum", action="store_true")
    parser.add_argument("--eth", action="store_true")
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    eval_model(
        nyu=args.nyu,
        eth=args.eth,
        tum=args.tum,
        device="cpu" if args.cpu else "cuda",
        save=args.save_path is not None,
        save_dir=args.save_path,
    )
