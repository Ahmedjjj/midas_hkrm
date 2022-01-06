import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Callable, Dict, Iterator, List, Tuple

import torch
import torch.nn as nn
import tqdm
from midas_hkrm.utils import require
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


@dataclass
class MidasHKRMTrainer:
    """
    Convenience class to train a MidasHKRM model
    """

    model: nn.Module  # model to train
    criterion: nn.Module  # loss function
    optimizer: torch.optim.Optimizer  # Optimizer
    datasets: List[Dataset]  # Datasets to train on
    train_transform: Callable  # input transform in the training phase
    test_transform: Callable = None  # input transform in the testing phase
    batch_size: int = None  # Batch size per dataset for training. Total batch size is len(datasets) x batch_size
    test_batch_size: int = None  # Batch size per dataset for testing
    save: bool = True  # if True, periodically save a checkpoint of the complete state
    save_after: int = 0  # number of iteration to save after. required if save is true
    save_path: str = None  # Folder to save the state into
    test: bool = False  # if True, periodically test the model
    test_after: int = 0  # # number of iteration to test after. required if test is true
    test_split_size: float = 0  # percentage to leave for each dataset as test set
    same_test_size: bool = False  # if True, the number of test samples from each dataset is the same (the min)
    seed: int = 42  # seed for reproducing the split sizes
    cur_iter: int = 0  # iteration the model training is at
    max_iter: int = 300000  # maximum iterations, goal is to train the model on this number of iterations
    device: str = "cuda"  # device to use for the training

    def __post_init__(self):
        self._prepare_loaders()

        self.train_losses = OrderedDict()
        self.test_losses = OrderedDict()

        self.model.to(self.device)
        self.criterion.to(self.device)

    def _prepare_loaders(self):
        """
        Creates train and test dataset loaders.
        """
        self.dataset_names = [dataset.name for dataset in self.datasets]
        logger.info(f"Trainer initialized on datasets: {self.dataset_names}")

        datasets = dict(zip(self.dataset_names, self.datasets))

        self.train_loaders = dict()
        self.test_loaders = dict()
        self.test_datasets_lengths = dict()

        logger.info("Preparing test sets")

        if self.same_test_size:
            test_size = min(
                [int(len(dataset) * self.test_split_size) for dataset in self.datasets]
            )

        for name, dataset in datasets.items():
            train_dataset = dataset
            if self.test:
                if not self.same_test_size:
                    test_size = int(len(dataset) * self.test_split_size)

                train_size = len(dataset) - test_size
                self.test_datasets_lengths[name] = test_size
                test_dataset, train_dataset = random_split(
                    dataset,
                    [test_size, train_size],
                    generator=torch.Generator().manual_seed(self.seed),
                )

                logger.info(
                    f"Created a test set for dataset {name} of size {test_size}"
                )

                self.test_loaders[name] = DataLoader(
                    test_dataset,
                    batch_size=self.test_batch_size,
                    collate_fn=lambda x: x,
                    shuffle=False,
                )

            self.train_loaders[name] = self._cycle(
                DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    collate_fn=lambda x: x,
                    shuffle=True,
                )
            )

    @staticmethod
    def _cycle(loader: DataLoader) -> Iterator:
        """
        Given a dataloader, create an (infinitely) cycling iterator on the loader

        Args:
            loader (DataLoader): base loader

        Yields:
            batch: batch from the dataloader
        """
        cycling_iterator = iter(loader)
        while True:
            try:
                item = next(cycling_iterator)
                yield item
            except StopIteration:
                cycling_iterator = iter(loader)
                yield next(cycling_iterator)

    def save_state(self):
        """
        Save a state of the Trainer, so that training can be resumed.
        State is save in self.save_path with filename state_{self.cur_iter}.tar
        """
        saved_state_filename = f"state_{self.cur_iter}.tar"

        with open(os.path.join(self.save_path, "last_iter.txt"), "w") as f:
            f.write(saved_state_filename)

        torch.save(
            self.state,
            os.path.join(self.save_path, saved_state_filename),
        )
        logger.info(f"Saved state {saved_state_filename}")

    def load_state(self, state_filename: str = None):
        """
        Load the state of the trainer either from a file or from
        self.save_path (last state available)

        Args:
            state_filename ([type], str): path to state file. Defaults to None.
        """
        if not self.save_path and not state_filename:
            logger.warning(
                "Trainer not initialized with a save path, Not loading any state"
            )
            return

        if not state_filename:
            with open(os.path.join(self.save_path, "last_iter.txt"), "r") as f:
                state_filename = f.readline().strip()

        state = torch.load(os.path.join(self.save_path, state_filename))
        self.model.load_state_dict(state["model"])
        state.pop("model")
        self.optimizer.load_state_dict(state["optimizer"])
        state.pop("optimizer")
        self.train_losses = OrderedDict(state["train_losses"])
        state.pop("train_losses")
        self.test_losses = OrderedDict(state["test_losses"])
        state.pop("test_losses")

        for k, v in state.items():
            self.__setattribute__(k, v)

        self._prepare_loaders()
        logger.info(f"Loaded state: {state_filename}")

    def _optimized_forward_pass(
        self, samples: List[Tuple[torch.Tensor]], test=False
    ) -> torch.Tensor:
        """
        Helper function for an optimal forward pass.
        Groups input image by shape and forwards images of the same shape in a batch fashion,

        Args:
            samples (List[Tuple[torch.Tensor]): list of samples
            test (bool, optional): Whether this pass is done in testing mode. Defaults to False.

        Returns:
            torch.Tensor: batch loss
        """
        logger.debug(f"Total Batch size is {len(samples)}")

        transform = self.test_transform if test else self.train_transform
        imgs, gt_disps = zip(*[transform(sample) for sample in samples])
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

    @property
    def state(self) -> Dict[str, Any]:
        """
        Complete state of the trainer.

        Returns:
            Dict[str, Any]: state dict
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cur_iter": self.cur_iter,
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "test_batch_size": self.test_batch_size,
            "save": self.save,
            "save_after": self.save_after,
            "test": self.test,
            "test_after": self.test_after,
            "test_split_size": self.test_split_size,
            "max_iter": self.max_iter,
            "device": self.device,
            "dataset_names": self.dataset_names,
        }

    def test_(self) -> Dict[str, float]:
        """
        Evaluate the loss on the test sets.

        Returns:
            Dict[str, float]: dict of mapping from dataset name to loss
        """
        self.model.eval()

        test_losses = dict()
        for name, test_dataset in self.test_loaders.items():
            dataset_loss = 0

            for batch in test_dataset:
                with torch.no_grad():
                    dataset_loss += self._optimized_forward_pass(batch, test=True)

            test_losses[name] = float(dataset_loss / self.test_datasets_lengths[name])

        return test_losses

    def train(self) -> Dict[str, Any]:
        """
        Train the model, periodically save and test

        Returns:
            Dict[str, Any]: state dict after training.
        """

        require(
            self.batch_size is not None,
            "Please Initialize the trainer with a batch size or load a saved state",
        )

        start_iter = self.cur_iter + 1

        # save initial state
        if self.save and self.cur_iter == 0:
            logger.info("Saving initial trainer state")
            self.save_state()
            start_iter = 0

        logger.info(f"Starting at iteration {start_iter}")

        for step in tqdm.tqdm(range(start_iter, self.max_iter)):
            logger.info(f"Training iteration: {step + 1}")

            self.optimizer.zero_grad()
            loss = 0
            for train_loader in self.train_loaders.values():
                batch = next(train_loader)
                dataset_loss = self._optimized_forward_pass(batch)
                dataset_loss /= self.batch_size
                loss += dataset_loss
            if loss > 0:
                loss.backward()
                self.optimizer.step()
            self.train_losses[step] = float(loss)

            self.cur_iter = step

            # Possibly test
            if self.test and (step + 1) % self.test_after == 0:
                logger.info(f"Starting evaluation at step {step + 1}")
                self.test_losses[step] = self.test_()
                logger.info(f"Test losses  at step {step}: {self.test_losses[step]}")
                self.model.train()

            # Possibly save
            if self.save and (step + 1) % self.save_after == 0:
                logger.info(f"Saving state at step {step + 1}")
                self.save_state()

        if self.save:
            logger.info("Saving final state")
            self.save_state()

        return self.state
