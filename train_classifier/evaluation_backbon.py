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
from src.evaluate import *

if __name__ == '__main__':
    # Load the model
    model = resnet50(pretrained=True, num_classes=1000) # or set the num_class to your co-orprate with specific setting

    #################################################
    #Uncomment following line and load your trained model
    #################################################
    # model.load_state_dict(torch.load('model.pth')) 
   
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the test dataset
    dataloader = OxfordIIITPetDataloader()
    test_loader = dataloader.get_test_loader_CCAM()
    evaluate_cam(dataloader, model, threshold=0.05)
    evaluate_cam_ecs(model, test_loader, device)