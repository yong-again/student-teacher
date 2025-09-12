from dataclasses import dataclass
import os
from typing import Tuple, List, Dict, Any

@dataclass
class TrainResNet:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_name: str = 'resnet34'
    num_workers: int = 4
    batch_size: int = 64
    image_size: int = 256
    learning_rate: float = 1e-3
    momentum: float = 0.9
    epochs: int = 1000
    category: str = 'metal_nut'

@dataclass
class TrainTeacher:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    category: str = 'metal_nut'
    num_workers: int = 4
    batch_size: int = 64
    image_size: int = 256
    patch_size: int = 17
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    num_epochs: int = 1000

@dataclass
class TrainStudent:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    category: str = 'metal_nut'
    num_workers: int = 4
    batch_size: int = 1
    n_students: int = 3
    image_size: int = 256
    patch_size: int = 17
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 15


@dataclass
class Inference:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    category: str = 'metal_nut'
    test_size: int = 20
    n_students: int = 3
    patch_size: int = 17
    image_size: int = 256
    visualize: bool = True
    batch_size: int = 1
    num_workers: int = 4

@dataclass
class InferenceMultiScale:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    category: str = 'metal_nut'
    n_students: int = 3
    patch_sizes: List[int] = (17, 33, 65)
    image_size: int = 256
    visualize: bool = True
    batch_size: int = 1
    num_workers: int = 4

if __name__ == '__main__':
    config = InferenceMultiScale()
    print(config)