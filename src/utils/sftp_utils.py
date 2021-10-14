import getpass

import cv2
import numpy as np
import paramiko


def open_cluster_connection(username):
    password = getpass.getpass(prompt='Remote server password: ')
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.client.AutoAddPolicy)
    client.connect(hostname='iccluster138.iccluster.epfl.ch',
                   username=username,
                   password=password,
                   look_for_keys=False)
    return client.open_sftp()


def read_remote_image(sftp_client, image_path, grayscale=False):
    with sftp_client.file(image_path) as image:
        if grayscale:
            return cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
