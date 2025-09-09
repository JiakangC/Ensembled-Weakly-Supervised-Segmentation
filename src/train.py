import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(ROOT, 'src')
sys.path.append(src_dir)
from model_utils import calc_ecs_cam, check_positive
from model import SimMaxLoss, SimMinLoss,get_ccam
from torch.nn import functional as F

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data import PetDataset


HEIGHT = 224
WIDTH = 224
IMAGE_SIZE = (HEIGHT, WIDTH)

def train_base(model, dataloader, num_epochs, lr, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = dataloader
        for images, _, labels, boxes in pbar:
            optimizer.zero_grad()
            
            images = images.to(device)
            labels = labels.to(device)
            batch_size = len(images)
            
            box_masks = torch.full((batch_size, HEIGHT, WIDTH), fill_value=1, dtype=torch.long).to(device)
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = box
                box_masks[i, y_min:y_max, x_min:x_max] = 0
                    
            outputs = model(images)
            loss = criterion(outputs, box_masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} training loss: {avg_train_loss:.4f}")
        
        
def train_sl(model, dataloader, num_epochs, lr, device):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, masks,_,_ in dataloader:
            optimizer.zero_grad()
            
            images = images.to(device)
            masks = masks.to(device)
                    
            outputs = model(images)

            # print(torch.min(outputs), torch.max(outputs))
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} training loss: {avg_train_loss:.4f}")


def train_classifier(model, train_loader, test_loader, device, num_epochs=3, learning_rate=0.0001):

    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i,data in enumerate(train_loader):
            images,_ , labels = data
            images = images.to(device)

            # choose the labels for specific training perpose
            # labels = torch.tensor([0 if label in ([ 0,  5,  6,  7,  9, 11, 20, 23, 26, 27, 32, 33]) else 1 for label in labels]) # 0: cat, 1: dog, 
            # labels = torch.tensor([0 if label in ([ 0,  5,  6,  7,  9, 11, 20, 23, 26, 27, 32, 33]) else 2 if label == 37 else 1 for label in labels]) # 0: cat, 1: dog, 2: background
            # labels = labels
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)[1]
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images,_, labels in test_loader:
                # choose the labels for specific training perpose
                # labels = torch.tensor([0 if label in ([ 0,  5,  6,  7,  9, 11, 20, 23, 26, 27, 32, 33]) else 1 for label in labels]) # 0: cat, 1: dog, 
                # labels = torch.tensor([0 if label in ([ 0,  5,  6,  7,  9, 11, 20, 23, 26, 27, 32, 33]) else 2 if label == 37 else 1 for label in labels]) # 0: cat, 1: dog, 2: background
                # labels = labels
                labels = labels.to(device)
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)[1]
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100 * test_correct / test_total
        print(f"Test Accuracy: {test_acc:.2f}%")
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_param = model.state_dict()
            print(f"Saved best model with Test Accuracy: {best_acc:.2f}%")
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_param = model.state_dict()
            print(f"Saved best model with Test Accuracy: {best_acc:.2f}%")
    
    return best_model_param


def train_CCAM(model, train_loader, device, save_name, max_epoch = 10):
    param_groups = model.get_parameter_groups()
    model = model.to(device)

    model.train()
    # define loss function and optimizer
    criterion = [SimMaxLoss(metric='cos', alpha=0.8).to(device), SimMinLoss(metric='cos').to(device),
                 SimMaxLoss(metric='cos', alpha=0.8).to(device)]

    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': 0.001, 'weight_decay': 1e-3},
        {'params': param_groups[1], 'lr': 2 * 0.001, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * 0.001, 'weight_decay': 1e-3},
        {'params': param_groups[3], 'lr': 20 * 0.001, 'weight_decay': 0},
    ])


    # define the training hyperparams
    
    # define the training step
    for epoch in range(max_epoch):
        running_loss = 0
        for iteration, (images,_, labels) in enumerate(train_loader):
            # the contrastive loss in the CCAM setup
            images= images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            fg_feats, bg_feats, ccam = model(images)
            flag = check_positive(ccam)
            if flag:
                ccam = 1 - ccam
                temp = fg_feats
                fg_feats = bg_feats
                bg_feats = temp

            loss1 = criterion[0](fg_feats)
            loss2 = criterion[1](bg_feats, fg_feats)
            loss3 = criterion[2](bg_feats)
            loss = loss1 + loss2 + loss3
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch} running_loss: {running_loss}")
        save_dir = './train_ccam/model_saved'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f'./train_ccam/model_saved/{save_name}_log_{epoch}.pth')




def train_ECS_CCAM(model, train_loader, device, save_name, max_epoch = 10):
    param_groups = model.get_parameter_groups()
    model.to(device)

    model.train()
    # define loss function and optimizer
    criterion = [SimMaxLoss(metric='cos', alpha=0.8).to(device), SimMinLoss(metric='cos').to(device),
                 SimMaxLoss(metric='cos', alpha=0.8).to(device)]

    optimizer = optim.SGD([
        {'params': param_groups[0], 'lr': 0.001, 'weight_decay': 1e-3},
        {'params': param_groups[1], 'lr': 2 * 0.001, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * 0.001, 'weight_decay': 1e-3},
        {'params': param_groups[3], 'lr': 20 * 0.001, 'weight_decay': 0},
    ])


    # define the training hyperparams
    
    # define the training step
    for epoch in range(max_epoch):
        running_loss = 0
        for iteration, (images,_, labels) in enumerate(train_loader):
            # the contrastive loss in the CCAM setup
            images= images.to(device)
            labels = torch.tensor([0 if label in ([ 0,  5,  6,  7,  9, 11, 20, 23, 26, 27, 32, 33]) else 1 for label in labels]) # 0: cat, 1: dog, 
            labels = labels.to(device)

            optimizer.zero_grad()
            fg_feats, bg_feats, ccam = model(images)
            flag = check_positive(ccam)
            if flag:
                ccam = 1 - ccam
                temp = fg_feats
                fg_feats = bg_feats
                bg_feats = temp

            loss1 = criterion[0](fg_feats)
            loss2 = criterion[1](bg_feats, fg_feats)
            loss3 = criterion[2](bg_feats)


            # calculate the calssification loss using model.backbone.init_model
            features, _, targets = model.backbone.init_model(images)
            target_class = torch.argmax(targets, dim=1)
            loss_classification = F.cross_entropy(targets, labels)
            # calculate the ecs-cam and get the ccam vs ecs-cam loss
            ## Compute original CAM
            with torch.no_grad():
                ecs_cam = calc_ecs_cam(model, images, device)

            loss_ecs = F.mse_loss(ecs_cam.unsqueeze(1), ccam)


            loss = loss1 + loss2 + loss3 +  loss_classification + loss_ecs
            # if loss <0.05:
            #     break
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch} running_loss: {running_loss}")
        save_dir = './train_ccam/model_saved'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f'./train_ccam/model_saved/{save_name}_log_{epoch}.pth')


def train_class_ccam_model(class_name, data_dir, max_epoch=20, batch_size=32):
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
    transform = transform.Compose([
                transform.Resize((224, 224)),
                transform.ToTensor(),
                transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

    # Create dataset and dataloader
    dataset = PetDataset(data_dir=data_dir, class_name=class_name, transform=transform)
    if len(dataset) == 0:
        print(f"No images found for class {class_name}. Skipping.")
        return
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    save_name = f'class_specific_ccam/{class_name}_ccam'

    train_CCAM(model, train_loader, device, save_name, max_epoch = 20)




