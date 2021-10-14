import posixpath

from src.datasets.remote_dataset import RemoteDataset


class RedWeb(RemoteDataset):
    def __init__(self, remote=False, username=None):
        super().__init__(remote, username=username)

    @property
    def path_prefix(self):
        return '/runai-ivrl-scratch/students/2021-fall-sp-jellouli/mix6/ReDWeb_V1'

    @property
    def locations(self):
        return [{'imgs': 'Imgs', 'labels': 'RDs'}]

    def labels_filename(self, img_name):
        return posixpath.splitext(img_name)[0] + '.png'
