import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm
from midas_hkrm.datasets import DIW, ETH3D, NYU, TUM
from midas_hkrm.zero_shot import (
    AbsRel,
    BadPixelMetric,
    ComputeDepthThenCriterion,
)
from midas_hkrm.zero_shot.criterion import AbsRel
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass(eq=False, frozen=True)
class Evaluator:
    """
    Convenience class for running zero-shot evaluation on a MidasHKRM net.
    This class always uses a batch size of 1 for simplicity, and computes the mean loss over all samples in the dataset.
    Note that the prediction is bilinearly interpolated into the target size.
    """

    model: nn.Module  # model to test
    criterion: nn.Module  # criterion to compute, usually from .criterion
    transform: nn.Module  # input transform at test time
    dataset: Dataset  # Dataset to test on
    device: str = "cuda"  # device to run analysis on
    pass_input: bool = False  # if True, the original image is passed to the model as well, as required in MidasHKRM

    def __post_init__(self):
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            collate_fn=lambda x: x,
            shuffle=False,
        )
        self.model.to(self.device).eval()

    def evaluate(self):
        with torch.no_grad():
            loss = 0
            for index, batch in tqdm.tqdm(enumerate(self.dataloader)):
                logger.debug(f"Processing input {index}")

                input, target = batch[0]
                # input transform
                input_t = self.transform(input).to(self.device)
                if self.pass_input:
                    prediction = self.model([input], input_t)
                else:
                    prediction = self.model(input_t)

                # Resize to desired shape
                resized_prediction = TF.resize(prediction, size=input.shape[:2])
                target = torch.tensor(target, dtype=float).to(self.device).unsqueeze(0)

                step_loss = self.criterion(resized_prediction, target)
                logger.debug(f"loss at step {index} is {step_loss}")
                loss += step_loss

            logger.info(f"Final loss: {loss / len(self.dataset)}")
            return loss / len(self.dataset)


class NyuV2Evaluator(Evaluator):
    """
    Compute the BadPixelMetric (with delta > 1.25) on the NYUv2 dataset.
    The predicted depth is capped at 10, since this is the maximum depth in the dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        transform: nn.Module,
        device: str = "cuda",
        pass_input: bool = False,
    ):
        super().__init__(
            model=model,
            criterion=ComputeDepthThenCriterion(
                depth_cap=10, criterion=BadPixelMetric(threshold=1.25)
            ),
            transform=transform,
            dataset=NYU(),
            device=device,
            pass_input=pass_input,
        )

    def evaluate(self):
        logger.info(
            f"Evaluating Bad pixel metric > 1.25 (with depth capped at 10) on NYU_V2 and model {self.model.__class__.__name__}"
        )
        loss = super().evaluate()
        return float(loss)


class ETH3DEvaluator(Evaluator):
    """
    Compute the mean Absolute value of the relative error on ETH3D training set.
    The predicted depth is capped at 72, since this is the maximum depth in the dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        transform: nn.Module,
        device: str = "cuda",
        pass_input: bool = False,
    ):
        super().__init__(
            model=model,
            criterion=ComputeDepthThenCriterion(depth_cap=72, criterion=AbsRel()),
            transform=transform,
            dataset=ETH3D(),
            device=device,
            pass_input=pass_input,
        )

    def evaluate(self):
        logger.info(
            f"Evaluating Absolute relative error metric on ETH3D and model {self.model.__class__.__name__}"
        )
        loss = super().evaluate()
        return float(loss)


class TUMEvaluator(Evaluator):
    """
    Compute the mean Absolute value of the relative error on TUM training set.
    The predicted depth is capped at 10, since this is the maximum depth in the dataset.
    """

    def __init__(
        self,
        model: nn.Module,
        transform: nn.Module,
        device: str = "cuda",
        pass_input: bool = False,
    ):
        super().__init__(
            model=model,
            criterion=ComputeDepthThenCriterion(depth_cap=10, criterion=AbsRel()),
            transform=transform,
            dataset=TUM(),
            device=device,
            pass_input=pass_input,
        )

    def evaluate(self):
        logger.info(
            f"Evaluating Absolute relative error metric on TUM and model {self.model.__class__.__name__}"
        )
        loss = super().evaluate()
        return float(loss)


class DIWEvaluator(Evaluator):
    """
    Compute the Ordinal Loss on the DIW test set.
    This was not used in the final project.
    """

    def __init__(
        self,
        model: nn.Module,
        transform: nn.Module,
        device: str = "cuda",
        pass_input: bool = False,
    ):
        super().__init__(
            model=model,
            criterion=self.loss,
            transform=transform,
            dataset=DIW(),
            device=device,
            pass_input=pass_input,
        )

    def loss(self, prediction: torch.Tensor, target: torch.Tensor) -> int:
        sample = prediction[0]
        x1, y1, x2, y2 = target[0]
        return int(sample[int(x1), int(y1)] < sample[int(x2), int(y2)])
