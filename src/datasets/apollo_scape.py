from src.datasets import Mix6Dataset
from src.utils import map_disp_to_0_1
from src.utils import read_image


class ApolloScape(Mix6Dataset):
    @property
    def name(self):
        return "ApolloScape"

    @property
    def locations(self):
        if self.test:
            return [{"imgs": "test/camera_5"}, {"imgs": "test/camera_6"}]

        else:
            return [
                {
                    "imgs": "stereo_train_001/camera_5",
                    "labels": "stereo_train_001/disparity",
                },
                {
                    "imgs": "stereo_train_002/camera_5",
                    "labels": "stereo_train_002/disparity",
                },
                {
                    "imgs": "stereo_train_003/camera_5",
                    "labels": "stereo_train_003/disparity",
                },
            ]

    def get_disparity(self, labels_path):
        return map_disp_to_0_1(read_image(labels_path, grayscale=True))
