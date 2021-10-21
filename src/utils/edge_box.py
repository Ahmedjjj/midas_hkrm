import tempfile

import cv2
import edge_boxes
import numpy as np


def get_proposal(image, num_proposals=100):
    with tempfile.NamedTemporaryFile(suffix='.jpg') as image_file:
        cv2.imwrite(image_file.name, image)
        proposals = edge_boxes.get_windows([image_file.name])[0]
        return proposals[proposals[:, -1].argsort()][-min(num_proposals, proposals.shape[0]):]


def get_batch_proposals(self, images):
    pass
