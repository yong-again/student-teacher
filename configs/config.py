from dataclasses import dataclass
import os
from typing import Tuple, List, Dict, Any

@dataclass
class TrainResNet:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    num_workers: int = 8
    batch_size: int = 64
    image_size: int = 256
    learning_rate: float = 1e-3
    momentum: float = 0.9
    epochs: int = 100
    category: str = 'bottle'


if __name__ == '__main__':
    config = TrainResNet()
    print(config)
    print(config.batch_size)
    print(config.image_size)
    print(config.learning_rate)
    print(config.momentum)
    print(config.epochs)
    print(config.category)
    print(config.root_dir)
