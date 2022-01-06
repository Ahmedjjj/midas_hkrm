import os

import h5py
import numpy as np
from midas_hkrm.datasets import ZeroShotDataset
from scipy.io import loadmat


class NYU(ZeroShotDataset):
    """
    NYUV2 dataset: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
    Ground truth: Depth
    """

    def __init__(self):
        super().__init__()
        split_path = os.path.join(self.dataset_path, "splits.mat")
        data_path = os.path.join(self.dataset_path, "nyu_depth_v2_labeled.mat")
        mat = loadmat(split_path)
        self._indices = (mat["testNdxs"] - 1).reshape(-1)
        self._dataset = h5py.File(data_path, "r")
        self._images = self._dataset.get("images")
        self._depths = self._dataset.get("rawDepths")
        self._len = len(self._indices)

    @property
    def name(self):
        return "NYU"

    def __len__(self):
        return self._len

    @property
    def locations(self):
        return []  # Dataset is in the format of MATLAB matrices

    def __getitem__(self, index):
        image = self._images[self._indices[index]]
        depth = self._depths[self._indices[index]]
        depth[depth > 10] = 0  # max depth is 10 for NYU
        depth[depth < 0] = 0
        return np.swapaxes(image, 0, 2), np.swapaxes(depth, 0, 1)
