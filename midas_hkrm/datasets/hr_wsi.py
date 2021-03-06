from midas_hkrm.datasets import Mix6Dataset
from midas_hkrm.utils import map_disp_to_0_1, read_image


class HRWSI(Mix6Dataset):
    """
    HRWSI dataset: https://github.com/KexianHust/Structure-Guided-Ranking-Loss
    Ground truth: disparity
    """

    @property
    def name(self):
        return "HR-WSI"

    @property
    def locations(self):
        if self.test:
            return [{"imgs": "val/imgs", "labels": "val/gts"}]
        else:
            return [{"imgs": "train/imgs", "labels": "train/gts"}]

    def get_disparity(self, labels_path):
        return map_disp_to_0_1(read_image(labels_path, grayscale=True))
