import cv2
import numpy as np


def read_image(
    image_path: str, grayscale: bool = False, bgr_to_rgb: bool = False
) -> np.ndarray:
    """
    Helper function for reading an image (assumed to be BGR if color) from a file.

    Args:
        image_path (str): path to the image
        grayscale (bool, optional): whether the image has 1 or 3 channels. Defaults to False.
        bgr_to_rgb (bool, optional): whether to convert the image to RGB. Defaults to False.

    Returns:
        np.ndarray: image array
    """
    if grayscale:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
