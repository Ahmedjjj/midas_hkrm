import os
from pathlib import Path

from midas_hkrm.datasets import ZeroShotDataset
from midas_hkrm.utils import read_image


class TUM(ZeroShotDataset):
    """
    TUM "dynamic" subset: https://vision.in.tum.de/data/datasets/rgbd-dataset
    Ground truth: Depth
    The ground truth is not given for every image. Rather each image and depth map has a timestamp.
    The authors provide a script to match the two: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
    """

    def __init__(self, static=True):
        """
        Args:
            static (bool, optional): Whether to use only the static subsets. Defaults to True.
                                     static subsets are: rgbd_dataset_freiburg2_desk_with_person
                                                         rgbd_dataset_freiburg3_sitting_static
                                                         rgbd_dataset_freiburg3_walking_static
        """
        self._labels_map = dict()
        self.static = static
        for location in self.locations:
            location_folder = Path(self.dataset_path) / Path(location["imgs"]).parent
            self._labels_map[location_folder] = dict()
            with open(
                location_folder / "assoc.txt"
            ) as f:  # assoc.txt contains the mapping between the images and depths
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
        # A sample is valid if it has a corresponding depth map
        return (
            os.path.basename(os.path.splitext(img_path)[0])
            in self._labels_map[img_folder]
        )

    def get_disparity(self, labels_path):
        depth = read_image(labels_path, grayscale=True)
        depth = depth / 5000  # Metric depth is obtained by dividing 5000
        depth[depth > 10] = 0  # Maximum valid depth is 10
        depth[depth < 0] = 0

        return depth
