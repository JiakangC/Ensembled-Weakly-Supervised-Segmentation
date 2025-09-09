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

class PetDataset(Dataset):
    def __init__(self, data_dir, class_name, transform=None):
        """
        Args:
            data_dir (str): Path to the directory containing class-specific images.
            class_name (str): Name of the class (e.g., 'Abyssinian').
            transform (callable, optional): Optional transform to be applied to samples.
        """
        self.data_dir = os.path.join(data_dir, class_name)
        self.transform = transform
        self.class_name = class_name
        # List all image files in the class directory
        self.image_paths = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)
                           if f.endswith('.jpg') and f.startswith(class_name)]
        # Assign label 0 to all images of this class
        self.labels = [0] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def train_class_model(class_name, data_dir, max_epoch=10, batch_size=32):
    torch.manual_seed(0)
    """
    Train a CCAM model for a specific class and save the model weights.
    
    Args:
        class_name (str): Name of the class to train on.
        data_dir (str): Path to the directory containing class folders.
        max_epoch (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = get_ccam()
    param_groups = model.get_parameter_groups()
    model = model.to(device)
    model.train()

    # Create dataset and dataloader
    dataset = PetDataset(data_dir=data_dir, class_name=class_name, transform=transform)
    if len(dataset) == 0:
        print(f"No images found for class {class_name}. Skipping.")
        return
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss functions and optimizer
    criterion = [
        SimMaxLoss(metric='cos', alpha=0.3).to(device),
        SimMinLoss(metric='cos').to(device),
        SimMaxLoss(metric='cos', alpha=0.3).to(device)
    ]
    
    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': 0.001, 'weight_decay': 1e-3},
        {'params': param_groups[1], 'lr': 2 * 0.001, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * 0.001, 'weight_decay': 1e-3},
        {'params': param_groups[3], 'lr': 20 * 0.001, 'weight_decay': 0},
    ])

    # Training loop
    for epoch in range(max_epoch):
        running_loss = 0
        for iteration, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            fg_feats, bg_feats, ccam = model(images)

            loss1 = criterion[0](fg_feats)
            loss2 = criterion[1](bg_feats, fg_feats)
            loss3 = criterion[2](bg_feats)

            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch} for {class_name}, Loss: {running_loss / len(train_loader):.4f}")

        # Save model
        save_dir = os.path.join(root_dir, 'train_class_specific_ccam', 'model_saved' ,'dog_cat_specific_ccam')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{class_name}_ccam_{epoch}.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Saved model for {class_name} at {save_path}")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Define data directory
    data_dir = os.path.join(root_dir, "data", "class_folders_cat_dog")  # Assumes images are in class-specific folders
    
    # Get list of class folders
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    # class_names = ['Birman']
    # save the class name
    print(class_names)
    # Train a model for each class
    for class_name in class_names:
        print(f"\nStarting training for class: {class_name}")
        train_class_model(class_name, data_dir, max_epoch=20, batch_size=32)