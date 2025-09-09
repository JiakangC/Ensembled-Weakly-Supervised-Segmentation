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
    datasets = ImagePseudoMaskDataset(
        root_image_dir=os.path.join(ROOT,"data", "class_folders"),
        root_mask_dir=os.path.join(ROOT,"data", "37_class_pseudo_mask_folders"),
        root_gt_dir=os.path.join(ROOT,"data", "37_class_gt")
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iou,dice = evaluate_class_specific_ccam(datasets)
    print('result for 37 specific CCAM')
    print("iou: ", iou)
    print("dice: ", dice)

    datasets = ImagePseudoMaskDataset(
        root_image_dir=os.path.join(ROOT,"data", "class_folders"),
        root_mask_dir=os.path.join(ROOT,"data", "37_class_pseudo_mask_folders_crf"),
        root_gt_dir=os.path.join(ROOT,"data", "37_class_gt")
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iou,dice = evaluate_class_specific_ccam(datasets)
    print('result for 37 specific CCAM+CRF')
    print("iou: ", iou)
    print("dice: ", dice)

    datasets = ImagePseudoMaskDataset(
        root_image_dir=os.path.join(ROOT,"data", "class_folders_cat_dog"),
        root_mask_dir=os.path.join(ROOT,"data", "cat_dog_pseudo_mask_folders"),
        root_gt_dir=os.path.join(ROOT,"data", "cat_dog_gt")
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iou,dice = evaluate_class_specific_ccam(datasets)
    print('result for cat_dog specific CCAM')
    print("iou: ", iou)
    print("dice: ", dice)

    datasets = ImagePseudoMaskDataset(
        root_image_dir=os.path.join(ROOT,"data", "class_folders_cat_dog"),
        root_mask_dir=os.path.join(ROOT,"data", "cat_dog_pseudo_mask_folders"),
        root_gt_dir=os.path.join(ROOT,"data", "cat_dog_gt")
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iou,dice = evaluate_class_specific_ccam(datasets)
    print('result for cat_dog specific CCAM+CRF')
    print("iou: ", iou)
    print("dice: ", dice)
