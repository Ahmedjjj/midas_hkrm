import os

from midas_hkrm.datasets import ZeroShotDataset
from midas_hkrm.utils import require
from torch.utils.data import Dataset


class DIW(ZeroShotDataset):
    def __init__(self):
        super().__init__(test=True)

    @property
    def name(self):
        return "DIW"

    @staticmethod
    def __parse_annotation(line):
        x1, y1, x2, y2, sign, _, _ = line.split(",")
        if sign == ">":
            return int(x1) - 1, int(y1) - 1, int(x2) - 1, int(y2) - 1
        else:
            return int(x2) - 1, int(y2) - 1, int(x1) - 1, int(y1) - 1

    @property
    def locations(self):
        return []

    @property
    def all_samples(self):
        all_samples = []
        annotation_filepath = os.path.join(
            self.dataset_path, "DIW_Annotations", "DIW_test.csv"
        )
        with open(annotation_filepath) as f:
            lines = f.readlines()
            files = [s.rstrip() for s in lines[::2]]
            annotations = [s.rstrip() for s in lines[1::2]]

        all_samples = list(
            zip(
                [os.path.join(self.dataset_path, f) for f in files],
                [self.__parse_annotation(a) for a in annotations],
            )
        )

        return all_samples

    def __getitem__(self, index):
        item = self._map[index]
        img_path, annotation = item
        return self.get_image(img_path), annotation
