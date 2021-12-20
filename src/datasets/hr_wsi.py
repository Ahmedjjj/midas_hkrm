from src.datasets import Mix6Dataset
from src.utils import map_disp_to_0_1, read_image


class HRWSI(Mix6Dataset):
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
