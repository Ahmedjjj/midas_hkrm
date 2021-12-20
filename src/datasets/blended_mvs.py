import posixpath

from src.datasets.remote_dataset import RemoteDataset
from src.utils.errors import require
from src.utils import read_image
import os
from src.datasets import Mix6Dataset
import glob
import numpy as np

from src.utils import map_depth_to_disp


class BlendedMVS(Mix6Dataset):
    def __init__(self, test=False):
        super().__init__(test)
        require(test == False, "BlendedMVS has no predefined test set")

    @property
    def name(self):
        return "BlendedMVS"

    @property
    def locations(self):
        all_folders = [
            f
            for f in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, f))
        ]
        locations = [
            {
                "imgs": os.path.join(folder, "blended_images"),
                "labels": os.path.join(folder, "rendered_depth_maps"),
            }
            for folder in all_folders
        ]

        return locations

    def labels_filename(self, img_name):
        return os.path.splitext(img_name)[0] + ".pfm"

    def sample_validator(self, img_path):
        return "masked" not in img_path

    def get_disparity(self, labels_path):
        return map_depth_to_disp(read_image(labels_path, grayscale=True))
