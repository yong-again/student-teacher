from dataclasses import dataclass
import os
from typing import Tuple, List, Dict, Any

@dataclass
class TrainResNet:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    num_workers: int = 4
    batch_size: int = 64
    image_size: int = 256
    learning_rate: float = 1e-3
    momentum: float = 0.9
    epochs: int = 1000
    category: str = 'bottle'

@dataclass
class TrainTeacher:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    category: str = 'bottle'
    num_workers: int = 4
    batch_size: int = 64
    image_size: int = 256
    patch_size: int = 65
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    num_epochs: int = 1000

@dataclass
class TrainStudent:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    category: str = 'bottle'
    num_workers: int = 4
    batch_size: int = 1
    n_students: int = 3
    image_size: int = 256
    patch_size: int = 65
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

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