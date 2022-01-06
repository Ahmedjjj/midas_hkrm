import os

from midas_hkrm.datasets import Mix6Dataset
from midas_hkrm.utils import map_depth_to_disp, read_image


class BlendedMVS(Mix6Dataset):
    """
    BlendedMVS dataset: https://github.com/YoYo000/BlendedMVS
    Ground truth: Disparity (mapped from Depth)
    """

    def __init__(self):
        super().__init__(test=False)

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
        return os.path.splitext(os.path.basename(img_name))[0] + ".pfm"

    def sample_validator(self, img_path):
        return "masked" not in img_path

    def get_disparity(self, labels_path):
        return map_depth_to_disp(read_image(labels_path, grayscale=True))
