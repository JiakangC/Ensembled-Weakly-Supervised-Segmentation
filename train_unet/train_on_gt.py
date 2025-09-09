import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import UNet
from src.data import OxfordIIITPetDataloader

def collapse_trimap(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a 3‑class trimap into a binary mask:
    original mask values ∈ {0:pet, 1:outline, 2:bg}
    → new mask values ∈ {0:background, 1:pet}
    """
    # mask == 0 (pet) → 1, everything else → 0
    return (mask == 0).long()

if __name__ == '__main__':
    # -- Configuration ---------------------------------------------------------
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size  = 32
    lr          = 1e-3
    num_epochs  = 5
    image_size  = 224
    num_classes = 2  # background vs. pet

    # -- Data ------------------------------------------------------------------
    loader = OxfordIIITPetDataloader(
        train_ratio=0.8,
        batch_size=batch_size,
        num_workers=0,
        seed=42
    )
    train_loader = loader.get_train_loader_CCAM()
    test_loader   = loader.get_test_loader_CCAM()

    # -- Model, Loss, Optim ----------------------------------------------------
    model     = UNet(n_channels=3, n_classes=num_classes, filters=[32,64,128,256]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -- Training Loop ---------------------------------------------------------
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        for batch_idx, (imgs, masks, _) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device).long()
            masks = ((masks==0 )+(masks ==2)).long() 

            # visualize_batch(imgs, masks, n=4) # NOTE : remove function and plt import
            # exit()

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            if batch_idx % 20 == 1:
                avg = total_loss / batch_idx
                print(f"[Epoch {epoch}/{num_epochs}] Batch {batch_idx}/{len(train_loader)} - loss: {avg:.4f}", end='\r')
        print(f"\nEpoch {epoch} training loss: {total_loss/len(train_loader):.4f}")

    # -- Save Model -------------------------------------------------------------
    save_dir = "./train_unet/saved_models"
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "unet_baseline.pth")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")