import os

import numpy as np
from midas_hkrm.datasets import ZeroShotDataset


class ETH3D(ZeroShotDataset):
    """
    Original ETH3D depth benchmark: https://www.eth3d.net/overview
    Ground truth: Depth
    """

    def __init__(self):
        super().__init__(test=False)

    @property
    def name(self):
        return "ETH3D"

    @property
    def locations(self):
        folders = os.listdir(self.dataset_path)

        locations = [
            {
                "imgs": os.path.join(f, "images", "dslr_images"),
                "labels": os.path.join(f, "ground_truth_depth", "dslr_images"),
            }
            for f in folders
        ]

        return locations

    def labels_filename(self, img_name):
        return os.path.basename(img_name)

    def get_depth(self, labels_path, image_shape):
        """Read depth image. This is a bit particular as depth is given as a 1D binary 4-byte
        float dump. ETH3D has max depth 72. Points with larger depth are given the value 0.

        Args:
            labels_path (str): path of the binary dump
            image_shape (Tuple): shape of the corresponding image
        Returns:
            np.ndarray: depth map
        """
        depth_raw = np.fromfile(labels_path, dtype=np.float32)
        depth = depth_raw.reshape(image_shape)
        depth[depth > 72] = 0
        depth[depth < 0] = 0
        return depth

    def __getitem__(self, index):
        item = self._map[index]
        if isinstance(item, tuple):
            img_path, gt_path = item
            image = self.get_image(img_path)
            return image, self.get_depth(gt_path, image.shape[:2])
        else:
            return self.get_image(item)
