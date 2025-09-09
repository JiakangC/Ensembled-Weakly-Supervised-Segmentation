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
from src.train import *

if __name__ == '__main__':

    #torch.save(model.state_dict(), f'./pipeline/CCAM/model_saved/{save_name}_log_{epoch}.pth')
    dataloader = OxfordIIITPetDataloader(batch_size=32)
    train_loader = dataloader.get_train_loader_CCAM()
    test_loader = dataloader.get_test_loader_CCAM()
    model_2_dir = os.path.join(current_dir, 'train_classifier', 'model_saved', 'resnet50_dog_cat.pth')
    model_37_dir = os.path.join(current_dir, 'train_classifier', 'model_saved', 'resnet50_37_class.pth')
    model = get_ccam()
    model_2 = get_ccam(model_path = model_2_dir, target_class=2)
    model_37 = get_ccam(model_path = model_37_dir, target_class=37)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_name = "pretrained"
    train_CCAM(model, train_loader, device, save_name, max_epoch = 20)
    save_name_2 = "two_class"
    train_CCAM(model_2, train_loader, device, save_name_2, max_epoch = 20)
    save_name_37 = "three_seven_class"
    train_CCAM(model_37, train_loader, device, save_name_37, max_epoch = 20)


   
    