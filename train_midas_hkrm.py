import logging
import sys
from collections import OrderedDict

from src.datasets import ApolloScape, TartanAir, RedWeb, BlendedMVS, MegaDepth, HRWSI
from src.depth import MidasHKRMNet
from src.utils import midas_train_transform, midas_eval_transform, construct_config
from src.objects import GeneralizedRCNNObjectDetector
from src.optim import SSITrimmedMAELoss

import torch
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
import tqdm
import os

MAX_OBJECTS = 20
OBJECT_DETECTION_WEIGHTS = '/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output/model_final.pth'
SAVE_DIR = '/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output_midas_hkrm'
DEVICE = 'cuda'
SAVE_AFTER = 10
EVAL_AFTER = 10
STEPS = 300000
TEST_SET_SPLIT = 0.1
BATCH_SIZE = 6
SEED = 42

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def main():
    logger.info("Constructing Object detection network")
    cfg = construct_config()
    cfg.MODEL.WEIGHTS = OBJECT_DETECTION_WEIGHTS
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4  # Higher threshold for more precise object detection
    object_detector = GeneralizedRCNNObjectDetector(cfg)

    logger.info("Preparing modified midas network")
    model = MidasHKRMNet(object_detector, max_objects=MAX_OBJECTS, device=DEVICE, pretrained_resnet=True)

    logger.info("Loading pretrained model_weights")
    checkpoint = (
        "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-f6b98070.pt"
    )
    pretrained_sd = torch.hub.load_state_dict_from_url(
        checkpoint, map_location=torch.device('cpu'), progress=True, check_hash=True
    )

    logger.info("Randomnly intializing missing model weights")
    model_sd = model.state_dict()
    final_sd = OrderedDict(zip(model_sd.keys(), pretrained_sd.values()))
    for k in final_sd.keys():
        if 'output_conv' in k or 'refinenet' in k:
            weights = final_sd[k]
            m_shape = model_sd[k].shape
            p_shape = final_sd[k].shape
            for i in range(len(weights.shape)):
                m = m_shape[i]
                p = p_shape[i]
                if p != m:
                    assert m > p
                    added_shape = list(weights.shape)
                    added_shape[i] = m - p
                    random_w = torch.randn(tuple(added_shape))
                    weights = torch.cat([weights, random_w], dim=i)
            final_sd[k] = weights
    model.load_state_dict(final_sd)
    logger.info("All weights loaded!")

    logger.info("Preparing Adam optimizer")
    # Almost like the MiDaS paper
    optimizer = Adam([{'params': model.backbone.parameters(), 'lr': 1e-5},
                      {'params': model.channel_reduc.parameters(), 'lr': 1e-5}],
                     lr=1e-4, betas=(0.9, 0.999))

    logger.info("Preparing loss function (Scale and Shift Invariant Trimmed MAE)")
    loss_func = SSITrimmedMAELoss()

    datasets = [ApolloScape(), TartanAir(), RedWeb(), BlendedMVS(), MegaDepth(), HRWSI()]
    num_datasets = len(datasets)
    dataset_names = [dataset.name for dataset in datasets]
    logger.info(f"Training on datasets: {dataset_names}")
    datasets = dict(zip(dataset_names, datasets))

    train_loaders = dict()
    test_loaders = dict()

    test_datasets_lengths = dict()

    logger.info("Preparing test sets")
    for name, dataset in datasets.items():
        train_dataset = dataset
        if TEST_SET_SPLIT > 0:

            test_size = int(len(dataset) * TEST_SET_SPLIT)
            train_size = len(dataset) - test_size

            test_datasets_lengths[name] = test_size

            test_dataset, train_dataset = random_split(
                dataset, [test_size, train_size], generator=torch.Generator().manual_seed(SEED))

            logger.info(f"Created a test set for dataset {name} of size {test_size}")
            test_loaders[name] = cycle(DataLoader(test_dataset, batch_size=test_size,
                                       collate_fn=lambda x: x, shuffle=False))

        train_loaders[name] = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE // num_datasets,
                                               collate_fn=lambda x: x, shuffle=True))

    train_losses = {}
    test_losses = {}

    logging.info(f"Starting training for {STEPS}")

    model.train()
    for step in tqdm.tqdm(range(STEPS)):
        optimizer.zero_grad()

        logger.info(f"Training iteration: {step}")
        loss = 0
        train_losses[step] = {}
        for name, loader in train_loaders.items():
            samples = next(loader)
            images, gt_disps = zip(*(midas_train_transform(sample) for sample in samples))
            orig_imgs, _ = zip(*samples)
            dataset_loss = 0
            for image, orig_img, gt_disp in zip(images, orig_imgs, gt_disps):
                prediction = model([orig_img], image.to(DEVICE))
                dataset_loss += loss_func(prediction, gt_disp.to(DEVICE))
            loss += dataset_loss
            train_losses[step][name] = dataset_loss

        loss /= BATCH_SIZE
        loss.backward()
        optimizer.step()

        if step % EVAL_AFTER == 0 and TEST_SET_SPLIT != 0:
            logger.info(f"Starting evaluation at step {step}")
            model.eval()
            test_losses[step] = {}
            with torch.no_grad():
                for name, loader in test_loaders.items():
                    samples = next(loader)
                    images, gt_disps = zip(*(midas_eval_transform(sample) for sample in samples))
                    orig_imgs, _ = zip(*samples)
                    loss = 0
                    for image, orig_img, gt_disp in zip(images, orig_imgs, gt_disps):
                        prediction = model([orig_img], image.to(DEVICE))
                        loss += loss_func(prediction, gt_disp.to(DEVICE))

                    loss /= len(test_datasets_lengths[name])
                    test_losses[step][name] = loss
                    logger.info(f"Test loss at step {step} for dataset {name}: {loss:.3f}")

        model.train()
        if step % SAVE_AFTER == 0 and SAVE_AFTER > 0:
            logger.info(f"Saving state at step {step}")
            saved_state_filename = f"state_{step}.tar"

            with open(os.path.join(SAVE_DIR, 'last_iter.txt'), 'w') as f:
                f.write(saved_state_filename)

            torch.save({'iter': step,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'train_losses': train_losses,
                        'test_losses': test_losses,
                        'seed': SEED}, os.path.join(SAVE_DIR, saved_state_filename))


def cycle(loader):
    cycling_iterator = iter(loader)
    while True:
        try:
            item = next(cycling_iterator)
            yield item
        except StopIteration:
            cycling_iterator = iter(loader)
            yield next(cycling_iterator)


if __name__ == "__main__":
    main()
