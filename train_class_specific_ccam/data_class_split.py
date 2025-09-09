import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.models as models

import sys
import os
# Get the directory of the current script (pipeline/CCAM/train.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Calculate the root directory (COMP0197_Group)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))  # Up two levels

sys.path.append(root_dir)
from src.model import *
from src.model_utils import *
from src.data import *


if __name__ == '__main__':
    split_pet_dataset()
    split_pet_dataset_cat_dog()
