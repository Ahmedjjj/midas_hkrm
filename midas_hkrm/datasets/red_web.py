from midas_hkrm.datasets import Mix6Dataset
from midas_hkrm.utils import map_disp_to_0_1, read_image


class RedWeb(Mix6Dataset):
    """
    RedWeb dataset: https://github.com/nnizhang/SMAC
    Ground truth: Disparity
    """

    def __init__(self):
        super().__init__(test=False)

    @property
    def name(self):
        return "ReDWeb_V1"

    @property
    def locations(self):
        return [{"imgs": "Imgs", "labels": "RDs"}]

    def get_disparity(self, labels_path):
        """
        RedWeb has a particular disparity format. Please see: https://github.com/isl-org/MiDaS/issues/16
        """
        labels = read_image(labels_path, grayscale=True)
        disp = labels.astype(float)
        mask = disp != 255
        disp = disp / 255
        disp *= -1.0
        disp *= mask
        return map_disp_to_0_1(disp)
