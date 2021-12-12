from src.datasets import Mix6Dataset

from logging import Logger


class MultiDatasetTrainer:

    def __init__(datasets: List[Mix6Dataset],
                 loss_func: nn.Module,
                 model: nn.Module
                 batch_size,
                 training_iters=None,
                 eval_set=True,
                 eval_percent=0.1,
                 eval_after=None,
                 device='cuda',
                 save_checkpoints=True,
                 save_dir='.'):

    def train():
