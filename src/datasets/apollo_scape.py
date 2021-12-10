import posixpath

from src.datasets.remote_dataset import RemoteDataset


class ApolloScape(RemoteDataset):
    def __init__(self, remote=False, test=False, username=None):
        super().__init__(remote, username=username)
        self._test = test

    @property
    def path_prefix(self):
        return '/runai-ivrl-scratch/students/2021-fall-sp-jellouli/mix6/ApolloScape'

    @property
    def locations(self):
        if self.test:
            return [{'imgs': 'test/camera_5', 'labels': None},
                    {'imgs': 'test/camera_6', 'labels': None}]

        else:
            return [{'imgs': 'stereo_train_001/camera_5', 'labels': 'stereo_train_001/disparity'},
                    {'imgs': 'stereo_train_002/camera_5', 'labels': 'stereo_train_002/disparity'},
                    {'imgs': 'stereo_train_003/camera_5', 'labels': 'stereo_train_003/disparity'},
                    ]

    def labels_filename(self, img_name):
        return posixpath.splitext(img_name)[0] + '.png'

    @property
    def test(self):
        return self._test
