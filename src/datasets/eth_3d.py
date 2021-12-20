from src.datasets import ZeroShotDataset
import os
from src.utils import read_image, map_depth_to_disp


class ETH3D(ZeroShotDataset):
    def __init__(self, test=None):
        super().__init__(test=test)

    @property
    def name(self):
        return "ETH3D"

    @property
    def locations(self):
        folders = []
        if self.test or self.test is None:
            folders.extend(
                [
                    os.path.join("test", f)
                    for f in os.listdir(os.path.join(self.dataset_path, "test"))
                ]
            )
        if not self.test or self.test is None:
            folders.extend(
                [
                    os.path.join("training", f)
                    for f in os.listdir(os.path.join(self.dataset_path, "training"))
                ]
            )

        locations = [
            {"imgs": os.path.join(f, "rgb"), "labels": os.path.join(f, "depth")}
            for f in folders
        ]

        return locations

    def get_disparity(self, labels_path):
        return read_image(labels_path, grayscale=True)
