from src.datasets import Mix6Dataset
from src.utils import require, read_image, map_depth_to_disp
import numpy as np
import os


class TartanAir(Mix6Dataset):
    def __init__(self, test=False):
        super().__init__(test)
        require(test == False, "TartanAir has no predefined test set")

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
        return os.path.splitext(img_name)[0] + "_depth.npy"
