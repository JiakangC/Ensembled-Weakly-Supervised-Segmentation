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

def train_classifier(model,train_loader, test_loader, device, num_epochs=3, learning_rate=0.0001):
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
            labels = torch.tensor([0 if label in ([ 0,  5,  6,  7,  9, 11, 20, 23, 26, 27, 32, 33]) else 1 for label in labels])
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
                labels = torch.tensor([0 if label in ([ 0,  5,  6,  7,  9, 11, 20, 23, 26, 27, 32, 33]) else 1 for label in labels])
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



if __name__ == "__main__":
    # initialize the modified resnet50 model
    model = resnet50(num_classes=2)
    # model.eval()
    # initiate the dataloader
    dataloader = OxfordIIITPetDataloader(batch_size=32, num_workers=8)
    train_loader = dataloader.get_train_loader_CCAM()
    test_loader = dataloader.get_test_loader_CCAM()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model_param = train_classifier(model, train_loader, test_loader, device=device)
    # save the model
    save_dir = '././train_classifier/model_saved/model_saved'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(best_model_param, './train_classifier/model_saved/resnet50_dog_cat.pth')