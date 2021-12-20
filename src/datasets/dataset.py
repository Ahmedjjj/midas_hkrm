from src.utils.img_utils import read_image
from torch.utils.data import Dataset
import os
from abc import ABC, abstractmethod
import glob


class MultiFolderDataset(Dataset, ABC):
    def __init__(self, test=False):
        self._test = test
        all_samples = self.__all_samples
        self._len = len(all_samples)
        self._map = dict(zip(range(self._len), all_samples))

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def locations(self):
        raise NotImplementedError()

    def get_disparity(self, labels_path):
        return read_image(labels_path, grayscale=True)

    def get_image(self, img_path):
        return read_image(img_path)

    def labels_filename(self, img_name):
        return os.path.splitext(img_name)[0] + ".png"

    def sample_validator(self, img_path):
        return True

    @property
    @abstractmethod
    def path_prefix(self):
        raise NotImplementedError()

    @property
    def dataset_path(self):
        return os.path.join(self.path_prefix, self.name)

    @property
    def test(self):
        return self._test

    @property
    def __all_samples(self):
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
                        self.labels_filename(os.path.basename(sample)),
                    )
                    for sample in samples
                ]
                samples = zip(samples, labels)

            all_samples.extend(samples)

        return all_samples

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        item = self._map[index]
        if isinstance(item, tuple):
            img_path, gt_path = item
            return self.get_image(img_path), self.get_disparity(gt_path)
        else:
            return self.get_image(item)


class Mix6Dataset(MultiFolderDataset):
    @property
    def path_prefix(self):
        return os.environ["MIX6_DATASETS"]


class ZeroShotDataset(MultiFolderDataset):
    @property
    def path_prefix(self):
        return os.environ["ZERO_SHOT_DATASETS"]
