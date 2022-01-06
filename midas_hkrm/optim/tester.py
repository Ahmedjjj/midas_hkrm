import logging
from dataclasses import dataclass
from itertools import groupby
from typing import Dict, List

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


@dataclass
class MidasHKRMTester:
    """
    A convenience class for running testing on a MidasHKRMModel
    """

    model: nn.Module  # model to test
    criterion: nn.Module  # loss function
    test_transform: nn.Module  # input transform
    datasets: List[Dataset]  # datasets to test on
    seed: int  # seed for creating the test sets
    split_size: float  # percentage of test samples for each dataset
    device: str = "cuda"  # device to run on
    batch_size: int = 1  # batch size for each dataset

    def __post_init__(self):
        self.model.eval()

        self.dataset_names = [dataset.name for dataset in self.datasets]
        logger.info(f"MidasHKRM tester initialized on datasets: {self.dataset_names}")
        datasets = dict(zip(self.dataset_names, self.datasets))

        self.test_loaders = dict()
        self.test_datasets_lengths = dict()

        logger.info("Preparing test sets")
        for name, dataset in datasets.items():
            test_size = int(len(dataset) * self.split_size)
            train_size = len(dataset) - test_size
            self.test_datasets_lengths[name] = test_size
            test_dataset, _ = random_split(
                dataset,
                [test_size, train_size],
                generator=torch.Generator().manual_seed(self.seed),
            )
            logger.info(f"Created a test set for dataset {name} of size {test_size}")
            self.test_loaders[name] = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                collate_fn=lambda x: x,
                shuffle=False,
            )

    def test(self) -> Dict[str, float]:
        """
        Run a full test on the test sets
        Returns:
            Dict[str, float]: mapping from dataset name to loss
        """
        test_losses = dict()
        with torch.no_grad():
            for name, loader in self.test_loaders.items():
                logger.info(f"Testing on dataset {name}")
                dataset_loss = 0
                for batch in tqdm.tqdm(loader):
                    dataset_loss += self._optimized_forward_pass(samples=batch)
                logger.info(f"Test loss on {name}: {dataset_loss}")
                test_losses[name] = dataset_loss
            return test_losses

    def _optimized_forward_pass(self, samples: torch.Tensor) -> torch.Tensor:
        """
        See .trainer.MidasHKRMTrainer._optimized_forward_pass
        """
        imgs, gt_disps = zip(*[self.test_transform(sample) for sample in samples])
        orig_imgs, _ = zip(*samples)
        img_shapes = enumerate([i.shape for i in imgs])
        grouped_shapes = groupby(
            sorted(img_shapes, key=lambda x: x[1]), key=lambda x: x[1]
        )
        loss = 0
        for _, indices in grouped_shapes:
            indices = [i[0] for i in indices]
            batch_imgs = torch.cat([imgs[i] for i in indices]).to(self.device)
            batch_disps = torch.cat([gt_disps[i] for i in indices]).to(self.device)
            batch_orig = [orig_imgs[i] for i in indices]
            batch_predictions = self.model(batch_orig, batch_imgs)
            batch_loss = self.criterion(batch_predictions, batch_disps)
            loss += batch_loss
        return loss
