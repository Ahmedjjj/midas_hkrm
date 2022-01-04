from re import I
from midas_hkrm.datasets import ZeroShotDataset
import os
from midas_hkrm.utils import read_image, map_depth_to_disp
import numpy as np
from pathlib import Path


class TUM(ZeroShotDataset):
    def __init__(self, static=True):
        self._labels_map = dict()
        self.static = static
        for location in self.locations:
            location_folder = Path(self.dataset_path) / Path(location["imgs"]).parent
            self._labels_map[location_folder] = dict()
            with open(location_folder / "assoc.txt") as f:
                lines = f.readlines()
            for line in lines:
                depth, _, rgb, _ = line.strip().split()
                self._labels_map[location_folder][rgb] = depth
        super().__init__(test=False)

    @property
    def name(self):
        return "TUM"

    @property
    def locations(self):
        folders = os.listdir(self.dataset_path)
        if self.static:
            folders = [
                folder
                for folder in folders
                if "freiburg2" in folder or "static" in folder
            ]

        locations = [
            {"imgs": os.path.join(f, "rgb"), "labels": os.path.join(f, "depth")}
            for f in folders
        ]

        return locations

    def labels_filename(self, img_path):
        img_folder = Path(img_path).parent.parent
        return (
            self._labels_map[img_folder][
                os.path.basename(os.path.splitext(img_path)[0])
            ]
            + ".png"
        )

    def sample_validator(self, img_path):
        img_folder = Path(img_path).parent.parent
        return (
            os.path.basename(os.path.splitext(img_path)[0])
            in self._labels_map[img_folder]
        )

    def get_disparity(self, labels_path):
        depth = read_image(labels_path, grayscale=True)
        depth = depth / 5000
        depth[depth > 10] = 0
        depth[depth < 0] = 0

        return depth
