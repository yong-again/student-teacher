# Student-Teacher Anomaly Detection
This repository contains an implementation of a student-teacher framework for anomaly detection using PyTorch. 
The model is trained to identify anomalies in images by learning from a teacher network.

## Requirements
```text
torch
torchvision
numpy
PIL
tqdm
matplotlib
scikit-learn
einops
```

## Installation
1. Clone the repository:
   ```bash
    git clone https://github.com/yong-again/student-teacher.git
    cd student-teacher
    ```
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
## Usage
1. Prepare your dataset in the following structure:
    ```text
        data/
        ├── csv/
        │   ├── train.csv
        │   ├── resnet18_train.csv
        │   ├── test.csv
        │   └── ground_truth.csv
        └── mvtec_AD/
            ├── bottle/
            │   ├── ground_truth/
            │   │     ├── broken_large/
            │   │     ├── broken_small/
            │   │     └── contaminated/
            │   ├── train/
            │   │     └──  good/  
            │   └── test/
            │         ├── broken_large/
            │         ├── broken_small/
            │         ├──  contaminated/
            │         └──  good/
            └── capsule/ ...
    ```
2. Make csv files:
   ```bash
   cd ./utils
   python make_csv.py
   ```
3. Train the model:
- If you want to resnet18 as teacher network
    
  ```bash
  python resnet_train.py
  ```
- teacher train
  ```bash
  python teacher_train.py
  ```
- student train
    ```bash
    python student_train.py
    ```
  
4. Evaluate the model:
    ```bash
   python anomaly_detection.py
   ```

## Configuration
You can modify the configuration parameters in the `./config/config.py` file to customize the training and evaluation process.

```python
# Example configuration parameters
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
    category: str = 'carpet'

    
@dataclass
class TrainTeacher:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    category: str = 'carpet'
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
    category: str = 'carpet'
    num_workers: int = 4
    batch_size: int = 1
    n_students: int = 3
    image_size: int = 256
    patch_size: int = 65
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 15

    
@dataclass
class Inference:
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    category: str = 'carpet'
    test_size: int = 20
    n_students: int = 3
    patch_size: int = 65
    image_size: int = 256
    visualize: bool = True
    batch_size: int = 1
    num_workers: int = 4
```

## Saved Models
You can find the pre-trained models in the `./weights/{category}` directory.

## Results
The results of the anomaly detection can be found in the `./results/{datetime}/{category}/exp` directory.

## references
- [Student-Teacher Anomaly Detection](https://arxiv.org/pdf/1911.02357v2)
- [Official git hub](https://github.com/denguir/student-teacher-anomaly-detection)

