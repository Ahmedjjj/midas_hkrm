import cv2


def read_image(image_path, grayscale=False, bgr_to_rgb=False):
    if grayscale:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
