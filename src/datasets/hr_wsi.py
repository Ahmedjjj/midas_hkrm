import posixpath

from src.datasets.remote_dataset import RemoteDataset


class HRWSI(RemoteDataset):
    def __init__(self, remote=False, test=False, username=None):
        super().__init__(remote, username=username)
        self._test = test

    @property
    def path_prefix(self):
        return '/runai-ivrl-scratch/students/2021-fall-sp-jellouli/mix6/HR-WSI'

    @property
    def locations(self):
        if self.test:
            return [{'imgs': 'val/imgs', 'labels': 'val/gts'}]
        else:
            return [{'imgs': 'train/imgs', 'labels': 'train/gts'}]

    def labels_filename(self, img_name):
        return posixpath.splitext(img_name)[0] + '.png'

    @property
    def test(self):
        return self._test
