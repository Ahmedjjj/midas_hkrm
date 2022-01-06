import os

import h5py
import numpy as np
from midas_hkrm.datasets import Mix6Dataset
from midas_hkrm.utils import map_depth_to_disp


class MegaDepth(Mix6Dataset):
    """
    MegaDepth dataset: https://www.cs.cornell.edu/projects/megadepth/
    Ground truth: Disparity (mapped from depth)
    """

    def __init__(self):
        super().__init__(test=False)

    @property
    def name(self):
        return "MegaDepth_v1"

    @property
    def locations(self):
        all_folders = [
            f
            for f in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, f))
        ]
        locations = []
        for folder in all_folders:
            folder_dense_0 = os.path.join(folder, "dense0")
            locations.append(
                {
                    "imgs": os.path.join(folder_dense_0, "imgs"),
                    "labels": os.path.join(folder_dense_0, "depths"),
                }
            )
            folder_dense_1 = os.path.join(folder, "dense1")
            locations.append(
                {
                    "imgs": os.path.join(folder_dense_1, "imgs"),
                    "labels": os.path.join(folder_dense_1, "depths"),
                }
            )

        return locations

    def labels_filename(self, img_name):
        return os.path.splitext(os.path.basename(img_name))[0] + ".h5"

    def get_disparity(self, labels_path):
        with h5py.File(labels_path, "r") as f:
            depth = f.get("/depth")
            depth = np.array(depth)

        return map_depth_to_disp(depth)
