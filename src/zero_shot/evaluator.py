from cv2 import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from src.datasets import NYU
from src.zero_shot import BadPixelMetric

import logging
import tqdm
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)


@dataclass(eq=False, frozen=True)
class Evaluator:
    model: nn.Module
    criterion: nn.Module
    transform: nn.Module
    dataset: Dataset
    device: str = "cuda"

    def __post_init__(self):
        self.dataloader = DataLoader(
            self.batch_size,
            batch_size=1,
            collate_fn=lambda x: x,
            shuffle=False,
        )
        self.model.to(self.device).eval()

    def evaluate(self):
        with torch.no_grad():
            loss = 0
            for index, (input, target) in tqdm.tqdm(enumerate(self.dataloader)):
                logger.info(f"Processing input {index}")
                input_t = self.transform(input).to(self.device)
                prediction = self.model([input], input_t)
                resized_prediction = TF.resize(prediction, size=input.shape)
                loss += self.criterion(resized_prediction, target)

            logger.info(f"Final loss: {loss / len(self.dataset)}")
            return loss


class NyuV2Evaluator(Evaluator):
    def __init__(
        self,
        model: nn.Module,
        transform: nn.Module,
        batch_size: int,
        device: str = "cuda",
        save_path: str = ".",
    ):
        super().__init__(
            model=model,
            criterion=BadPixelMetric(10, 1.25),
            transform=transform,
            dataset=NYU(),
            batch_size=batch_size,
            device=device,
            save_path=save_path,
        )

    def evaluate(self):
        logger.info(
            f"Evaluating Bad pixel metric > 1.25 (with depth capped at 10) on NYU_V2 and model {self.model.__class__.__name__}"
        )
        loss = super().evaluate()
        return loss
