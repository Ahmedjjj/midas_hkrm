import glob
import os
from abc import ABC, abstractmethod

import numpy as np
from midas_hkrm.utils.img_utils import read_image
from torch.utils.data import Dataset


class MultiFolderDataset(Dataset, ABC):
    """
    This is a useful abstraction for a dataset that is spanned over many folders.
    Each Implementing class has to provide:
    . a "path_prefix": a base folder where all datasets exist
    . a "name": it is assumed that the dataset files (as specified by the documentation of each dataset) exist under path_prefix / name
    . a list of "locations": a location is a dict with keys : "imgs" (required), "labels" (optional).
                             The values of the the dict are locations (folders), relative to path_prefix / name,
                                                                                 where the images and labels exist
    Subclasses may implement any method to change the default behavior
    """

    def __init__(self, test=False):
        """Create a new Dataset
        Args:
            test (bool, optional): Whether this corresponds to a test set (used by some datasets). Defaults to False.
        """
        self._test = test
        all_samples = self.all_samples
        self._len = len(all_samples)
        self._map = dict(zip(range(self._len), all_samples))

    @property
    @abstractmethod
    def path_prefix(self) -> str:
        """
        Path to top level folder containing all datasets.
        This is used as a convenience as we place all datasets of the same type in the same folder.
        We use this to abstract away all absolute-paths.u

        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the dataset. See above for details.
        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def locations(self):
        """Locations (folders) where images and labels exist. See above for details and in implementing subclasses for examples.
        Raises:
            NotImplementedError
        """
        raise NotImplementedError()

    def get_disparity(self, labels_path: str) -> np.ndarray:
        """Given a path to a ground truth file, return the disparity / depth map

        Args:
            labels_path (str): path to the file

        Returns:
            np.ndarray: ground truth image
        """
        return read_image(labels_path, grayscale=True)

    def get_image(self, img_path: str) -> np.ndarray:
        """Given a path to an image file, return the corresponding color Image
        Note that image should be in BGR format with dimension Height x Width x channels
        Args:
            labels_path (str): path to the file

        Returns:
            np.ndarray: image
        """
        return read_image(img_path)

    def labels_filename(self, img_path: str) -> np.ndarray:
        """Given an image (given by its path) return the filename of the corresponding ground truth file

        Args:
            img_path (str): path to the file

        Returns:
            str: ground truth filename
        """
        return os.path.splitext(os.path.basename(img_path))[0] + ".png"

    def sample_validator(self, img_path: str) -> bool:
        """Given an image (given by its path) return whether this is a valid sample of the dataset

        Args:
            img_path (str): path to the file

        Returns:
            bool: whether this sample is valid
        """
        return True

    @property
    def dataset_path(self) -> str:
        """
        See above. the path to the dataset is given by: path_prefix / dataset_name

        Returns:
            str: path to the top-level folder of the dataset
        """
        return os.path.join(self.path_prefix, self.name)

    @property
    def test(self):
        return self._test

    @property
    def all_samples(self):
        """
        Return a list of paths of all samples in the dataset.
        Concretely, this does the following:
        . Look in all locations:
            . for each file in location["imgs"], if the sample is valid (sample_validator) add it to the list of samples
            . if location has "labels", find the corresponding gt (labels_filename) and add it to the list of samples

        Returns:
            List: all samples in the dataset (as a tuple of paths)
        """
        all_samples = []
        for location in self.locations:
            absolute_folder_path = os.path.join(self.dataset_path, location["imgs"])
            samples = [
                sample
                for sample in sorted(glob.glob(os.path.join(absolute_folder_path, "*")))
                if self.sample_validator(sample)
            ]

            if "labels" in location:
                absolute_labels_path = os.path.join(
                    self.dataset_path, location["labels"]
                )
                labels = [
                    os.path.join(
                        absolute_labels_path,
                        self.labels_filename(sample),
                    )
                    for sample in samples
                ]
                samples = zip(samples, labels)

            all_samples.extend(samples)

        return all_samples

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int):
        """
        Return the item with index "index" in the dataset
        if the sample has ground truth, return a tuple of (img, gt)
        else return an ndarray
        Args:
            index (int)

        Returns:
            Tuple[np.ndarray] or ndarray: sample
        """
        item = self._map[index]
        if isinstance(item, tuple):
            img_path, gt_path = item
            return self.get_image(img_path), self.get_disparity(gt_path)
        else:
            return self.get_image(item)


class Mix6Dataset(MultiFolderDataset):
    """
    This is a convenience wrapper for datasets belonging to MIX6
    The base path (where all datasets exist) is assumed to be is the environment variable MIX6_DATASETS
    Note that MIX6 datasets should:
        . return disparity as ground truth
        . 0 values in the disparity should correspond to invalid ground truth pixels
        . the ground truth should be rescaled to the 0-1 range
    """

    @property
    def path_prefix(self):
        return os.environ["MIX6_DATASETS"]


class ZeroShotDataset(MultiFolderDataset):
    """
    This is a convenience wrapper for datasets used for zero shot evaluation
    The base path (where all datasets exist) is assumed to be is the environment variable "ZERO_SHOT_DATASETS"
    """

    @property
    def path_prefix(self):
        return os.environ["ZERO_SHOT_DATASETS"]
