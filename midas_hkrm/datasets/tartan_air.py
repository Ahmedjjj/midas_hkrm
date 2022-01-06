import os

import numpy as np
from midas_hkrm.datasets import Mix6Dataset
from midas_hkrm.utils import map_depth_to_disp


class TartanAir(Mix6Dataset):
    """
    TartanAir dataset: https://theairlab.org/tartanair-dataset/
    Ground truth: Disparity (mapped from depth)
    """

    def __init__(self):
        super().__init__(test=False)

    @property
    def name(self):
        return "TartanAir"

    @property
    def locations(self):
        all_folders = [
            f
            for f in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, f))
        ]
        locations = []
        for folder in all_folders:
            folder_easy = os.path.join(folder, "Easy")
            labels_folder = os.path.join(folder_easy, "depth_left")
            image_folder = os.path.join(folder_easy, "image_left")
            absolute_path_labels = os.path.join(self.dataset_path, labels_folder)
            for part in os.listdir(absolute_path_labels):
                locations.append(
                    {
                        "imgs": os.path.join(image_folder, part, "image_left"),
                        "labels": os.path.join(labels_folder, part, "depth_left"),
                    }
                )

        return locations

    def get_disparity(self, labels_path):
        depth_img = np.load(labels_path)
        return map_depth_to_disp(depth_img)

    def labels_filename(self, img_name):
        return os.path.splitext(os.path.basename(img_name))[0] + "_depth.npy"
