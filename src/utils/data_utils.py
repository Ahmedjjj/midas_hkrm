from midas.transforms import Resize, NormalizeImage, PrepareForNet
import torch
import torchvision.transforms as T
import cv2

# Taken from the discussion in the paper + midas code
midas_train_transform = T.Compose(
    [
        lambda sample: {"image": sample[0] / 255.0, "disparity": sample[1]},
        Resize(
            384,
            384,
            resize_target=True,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        lambda sample: torch.cat(
            [
                torch.from_numpy(sample["image"]),
                torch.from_numpy(sample["disparity"]).unsqueeze(0),
            ]
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomResizedCrop((384, 384))], p=0.3),
        lambda tensor: (tensor[:3].unsqueeze(0), tensor[3].unsqueeze(0)),
    ]
)

midas_eval_transform = T.Compose(
    [
        lambda sample: {"image": sample[0] / 255.0, "disparity": sample[1]},
        Resize(
            384,
            384,
            resize_target=True,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        lambda sample: (
            torch.from_numpy(sample["image"]).unsqueeze(0),
            torch.from_numpy(sample["disparity"]).unsqueeze(0),
        ),
    ]
)
# Taken from midas code
midas_test_transform = T.Compose(
    [
        lambda img: {"image": img / 255.0},
        Resize(
            384,
            384,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
    ]
)

# Perform augmentation on a batch
midas_batch_test_transform = lambda x: torch.cat(
    [midas_test_transform(img) for img in x]
)
