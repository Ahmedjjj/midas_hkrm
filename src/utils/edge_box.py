import tempfile

import cv2
import edge_boxes
import joblib
import torch


def get_best_n(proposals, n=100):
    @joblib.delayed
    def worker(proposal):
        return proposal[proposal[:, -1].argsort()][-min(n, proposal.shape[0]):]

    return joblib.Parallel(n_jobs=-1)(worker(p) for p in proposals)


def get_proposal(image):
    with tempfile.NamedTemporaryFile(suffix='.jpg') as image_file:
        cv2.imwrite(image_file.name, image)
        return edge_boxes.get_windows([image_file.name])


def get_batch_proposals(images):
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    temp_files = [tempfile.NamedTemporaryFile(suffix='.jpg') for img in images]
    file_names = []
    for img_file, img in zip(temp_files, images):
        cv2.imwrite(img_file.name, img)
        file_names.append(img_file.name)
    return edge_boxes.get_windows(file_names)
