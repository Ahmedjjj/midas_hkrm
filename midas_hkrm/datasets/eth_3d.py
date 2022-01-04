from midas_hkrm.datasets import ZeroShotDataset
import os
from midas_hkrm.utils import read_image, map_depth_to_disp
import numpy as np


class ETH3D(ZeroShotDataset):
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
