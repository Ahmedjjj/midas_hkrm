from src.datasets import Mix6Dataset
from src.depth import MidasHKRMNet
from logging import Logger

from torch.utils.data import Dataset,  DataLoader
from torch.utils.data import random_split
from torch.optim import Optimizer
import torch.nn as nn

logger = Logger.getLogger(__name__)


class MultiDatasetMidasHkrmTrainer:

    def __init__(datasets: List[Dataset],
                 loss_func: nn.Module,
                 model: nn.Module,
                 optimizer: Optimizer,
                 augmentation: nn.Module = None,
                 batch_size: int,
                 dataset_names: List[str] = None,
                 eval_set=True,
                 eval_percent=0.1,
                 eval_after=None,
                 device='cuda',
                 save_checkpoints=True,
                 save_figure=True,
                 save_dir='.',
                 seed=42):

        self._loss_func = loss_func
        self._model = model
        self._model.to(device)
        self._loss_func.to(device)

        self._optimizer = optimizer
        self._batch_size = batch_size // len(datasets)

        self._train_sets = []
        self._eval_sets = []

        if dataset_names is None:
            dataset_names = [str(i + 1) for in range(len(datasets))]

        if augmentation is None:
            augmentations = nn.Identity()

        self._augmentation = augmentation

        if eval_set:
            for dataset in datasets:
                test_size = int(len(dataset) * eval_percent)
                train_size = len(dataset) - test_size
                test_dataset, train_dataset = random_split(
                    dataset, [test_size, train_size], generator=torch.Generator().manual_seed(seed))

                self._train_sets.append(self._get_simple_data_iter(train_dataset))
                self._test_sets.append(self._get_simple_data_iter(test_dataset))

            self._test_sets = dict(zip(dataset_names, self._test_sets))
        else:
            self._train_sets = [self._get_simple_data_iter(dataset) for dataset in datasets]

        self._train_sets = dict(zip(dataset_names, self._train_sets))

        self._save_checkpoints = save_checkpoints
        self._save_figure = save_figure
        self._save_dir = save_dir

        self._eval_after = eval_after
        self._train_losses = dict()
        self._test_losses = dict()

    def _get_simple_data_iter(self, dataset):
        return iter(DataLoader(dataset, self._batch_size, collate_fn=lambda x: x))

    def train(self, num_steps: int):

        for step in range(num_steps):
            self._optimizer.zero_grad()
            
            samples = [*next(d) for d in self._train_sets]
            transformed_samples = [self._augmentation(s) for s in samples]
            imgs, disparities = zip(*transformed_samples)
            network_inputs = zip(samples, imgs, disparities)
            loss = 0
            for orig_img, trans_image, disparity in network_inputs:
                prediction = self._model([orig_img], trans_image)
                loss += self._loss_func(prediction, disparity)

            loss.backward()
            optimizer.step()

            if step % self._eval_after == 0 and self._eval_set:
                

