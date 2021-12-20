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
from itertools import groupby
from functools import reduce
from operator import add

MAX_OBJECTS = 20
OBJECT_DETECTION_WEIGHTS = '/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output/model_final.pth'
SAVE_DIR = '/runai-ivrl-scratch/students/2021-fall-sp-jellouli/output_midas_hkrm'
DEVICE = 'cuda'
SAVE_AFTER = 10000
EVAL_AFTER = 0
STEPS = 300000
TEST_SET_SPLIT = 0.1
BATCH_SIZE = 6
EVAL_BATCH_SIZE = 18
SEED = 42
DEBUG = False

logger = logging.getLogger(__name__)
level = logging.INFO
if DEBUG:
    level = logging.DEBUG

logger.setLevel(level)
logging.basicConfig(stream=sys.stdout, level=level)


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
    logger.info("Freezing encoder weights")
    for m in model.backbone.parameters():
        m.requires_grad = False

    logger.info("Preparing Adam optimizer")
    # Almost like the MiDaS paper
    optimizer = Adam([{'params': model.channel_reduc.parameters(), 'lr': 1e-5}],
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
            test_loaders[name] = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE // num_datasets,
                                            collate_fn=lambda x: x, shuffle=False)

        train_loaders[name] = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE // num_datasets,
                                               collate_fn=lambda x: x, shuffle=True))

    train_losses = {}
    test_losses = {}

    logger.info(f"Starting training for {STEPS} steps")
    for step in tqdm.tqdm(range(STEPS)):
        logger.info(f"Training iteration: {step}")

        optimizer.zero_grad()
        step_losses = forward_pass(train_loaders, model, loss_func, midas_train_transform, DEVICE)
        train_losses[step] = dict([(k, float(v)) for k, v in step_losses.items()])
        loss = reduce(add, step_losses.values())
        loss /= BATCH_SIZE
        loss.backward()
        optimizer.step()

        if EVAL_AFTER > 0 and step % EVAL_AFTER == 0 and TEST_SET_SPLIT != 0:
            logger.info(f"Starting evaluation at step {step}")
            model.eval()
            test_losses[step] = dict()
            for name, test_dataset in test_loaders.items():
                dataset_loss = 0
                for _, samples in enumerate(test_dataset, 0):
                    with torch.no_grad():
                        dataset_loss += sample_forward_pass(samples, model, loss_func, midas_eval_transform, DEVICE)

                test_losses[step][name] = dataset_loss / test_datasets_lengths[name]
            logger.info(f"Test losses  at step {step}: {test_losses[step]}")
            model.train()

        if SAVE_AFTER > 0 and step % SAVE_AFTER == 0 and SAVE_AFTER > 0:
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


def sample_forward_pass(samples, model, loss_func, transform, device):
    logger.debug(f"Total Batch size is {len(samples)}")

    imgs, gt_disps = zip(*[transform(sample) for sample in samples])
    orig_imgs, _ = zip(*samples)
    img_shapes = enumerate([i.shape for i in imgs])
    grouped_shapes = groupby(sorted(img_shapes, key=lambda x: x[1]), key=lambda x: x[1])
    loss = 0
    for _, indices in grouped_shapes:
        indices = [i[0] for i in indices]
        batch_imgs = torch.cat([imgs[i] for i in indices]).to(device)
        batch_disps = torch.cat([gt_disps[i] for i in indices]).to(device)
        batch_orig = [orig_imgs[i] for i in indices]

        logger.debug(f"Forward pass on mini batch (grouped) of size {len(indices)}")
        logger.debug(f"Batch input shape: {batch_imgs.shape}")

        batch_predictions = model(batch_orig, batch_imgs)
        logger.debug(f"Batch predictions shape: {batch_predictions.shape}")
        logger.debug("Computing loss for batch")
        logger.debug(f"batch ground truth shape: {batch_disps.shape}")

        batch_loss = loss_func(batch_predictions, batch_disps)

        logger.debug(f"Computed loss: {batch_loss}")

        loss += batch_loss
    return loss


def forward_pass(datasets, model, loss_func, transform, device):
    def one_pass():
        losses = {}
        for name, loader in datasets.items():
            losses[name] = 0
            samples = next(loader)
            logger.debug(f"forward pass for dataset {name}")
            losses[name] = sample_forward_pass(samples, model, loss_func, transform, device)

        return losses
    if DEBUG:
        with torch.autograd.detect_anomaly():
            return one_pass()
    else:
        return one_pass()


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
