import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset,random_split
import torchvision.transforms as T
import os
import sys
from PIL import Image
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
# Calculate the root directory (COMP0197_Group)
ROOT = os.path.abspath(os.path.join(current_dir, '..'))  # Up one levels
sys.path.append(ROOT)

from src.model import UNet
from src.data import ImagePseudoMaskDataset

# simple evaluation function (pixel accuracy)
def pixel_accuracy(pred, target):
    correct = (pred == target).float()
    return correct.sum() / correct.numel()

if __name__ == "__main__":
    # training params
    batch_size = 32
    lr = 1e-3
    num_epochs = 15
    num_classes = 2

    # loading the dataset with pseudo masks for training
    dataset = ImagePseudoMaskDataset(
        root_image_dir=os.path.join(ROOT, 'data', 'class_folders'),
        root_mask_dir=os.path.join(ROOT,'data', '37_class_pseudo_mask_folders'),
        root_gt_dir=os.path.join(ROOT, "data", "37_class_gt")
    )
    print(f"Dataset size: {len(dataset)}")
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # set device CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize the model
    model = UNet(n_channels=3, n_classes=num_classes, filters=[32, 64, 128, 256]).to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, verbose=True)

    # training process
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for imgs, crf_cams, _, _ in train_loader:
            imgs = imgs.to(device) 
            
            crf_cams = crf_cams.long().squeeze(1).to(device)  # [B, 1, H, W] -> [B, H, W]

            outputs = model(imgs).to(device)  # [B, C, H, W] -> [B, H, W]
            # print(outputs)
            loss = criterion(outputs, crf_cams)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} training loss: {avg_train_loss:.4f}")
        save_dir = "./train_unet/saved_models"
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"unet_specific_class_37_ccam_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_path)



