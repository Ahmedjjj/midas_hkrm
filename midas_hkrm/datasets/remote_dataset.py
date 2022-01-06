import os
import posixpath
import random
from abc import ABC, abstractmethod

from midas_hkrm.utils.img_utils import read_image
from midas_hkrm.utils.sftp_utils import open_cluster_connection, read_remote_image
from torch.utils.data import IterableDataset


class RemoteDataset(IterableDataset, ABC):
    """
    Abstraction for a remote (sftp) fetched dataset. Documentation is ommited because this code is not used in the rest of the project.
    A better way to do this is to use sshfs: https://github.com/libfuse/sshfs and mount the remote directory
    """

    def __init__(self, remote, username=None):
        self._remote = remote
        if remote:
            if not username:
                raise ValueError("username must be provided for remote datasets")
            self._sftp = open_cluster_connection(username)

    @property
    def sftp(self):
        return self._sftp

    @property
    def is_remote(self):
        return self._remote

    @property
    @abstractmethod
    def path_prefix(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def locations(self):
        raise NotImplementedError()

    @abstractmethod
    def labels_filename(self, image_filename):
        raise NotImplementedError()

    def __get_all_samples(self):
        for location_dict in self.locations:
            img_folder, labels_folder = location_dict.values()
            img_folder_path = posixpath.join(self.path_prefix, img_folder)
            images = (
                self.sftp.listdir(img_folder_path)
                if self.is_remote
                else os.listdir(img_folder_path)
            )
            all_samples = [posixpath.join(img_folder_path, image) for image in images]
            if labels_folder:
                labels = [self.labels_filename(image) for image in images]
                all_samples = [
                    (image, os.path.join(self.path_prefix, labels_folder, labels))
                    for (image, labels) in zip(all_samples, labels)
                ]

            return all_samples

    def __iter__(self):
        all_samples = self.__get_all_samples()
        random.shuffle(all_samples)
        for sample in all_samples:
            if type(sample) is tuple:
                image_path, labels_path = sample
                if self._remote:
                    img, labels = read_remote_image(
                        self.sftp, image_path
                    ), read_remote_image(self.sftp, labels_path, grayscale=True)
                else:
                    img, labels = read_image(image_path), read_image(
                        labels_path, grayscale=True
                    )

                yield img, labels
            else:
                image_path = sample
                if self._remote:
                    img = read_remote_image(self.sftp, image_path)
                else:
                    img = read_image(image_path)
                yield img
