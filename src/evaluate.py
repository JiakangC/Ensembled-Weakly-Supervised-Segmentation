import torch
import numpy as np
import torch.nn.functional as F
from src.model import *
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import sys
from src.model_utils import *



#### MASK GENERATOR ####
def generate_cam(image, model, extractor, device, threshold=0.05):
    """
    Generate CAM and pseudo-mask for a single image.
    
    Args:
        image (torch.Tensor): Input image tensor [C, H, W]
        model (nn.Module): Pretrained ResNet50 model
        extractor (CAMExtractor): Feature extractor
        device (torch.device): Device to run the model on
        threshold (float): Threshold for creating binary pseudo-mask
    
    Returns:
        cam_np (np.ndarray): CAM heatmap [224, 224]
        pred_mask (np.ndarray): Binary pseudo-mask [224, 224]
        target_class (int): Predicted class index
    """
    image = image.unsqueeze(0)  # [1, C, H, W]
    image = image.to(device)

    
    with torch.no_grad():
        features = extractor(image)  # [1, 2048, 7, 7]
        _, _, logits = model(image)  # [1, 1000]
        probs = F.softmax(logits, dim=1)
    
    target_class = torch.argmax(probs).item()
    
    cam = torch.zeros(features.shape[2:], dtype=torch.float32)  # [7, 7]
    for i, w in enumerate(extractor.fc_weights[target_class]):
        cam += w * features[0, i, :, :]
    
    cam = F.relu(cam)
    cam = cam / cam.max()
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(224, 224), 
                       mode='bilinear', align_corners=False)[0, 0]
    cam_np = cam.cpu().detach().numpy()
    
    # Generate binary pseudo-mask
    pred_mask = (cam_np > threshold).astype(np.uint8)
    
    return cam_np, pred_mask, target_class
#### Evaluate using IoU and Dice metrics
def evaluate_metric_cam(model, testloader, device, extractor, threshold=0.05):
    """
    Evaluate CAM on the test set using IoU and Dice metrics.
    
    Args:
        model (nn.Module): Pretrained ResNet50 model
        testloader (DataLoader): Test dataloader
        device (torch.device): Device to run the model on
        extractor (CAMExtractor): Feature extractor
        threshold (float): Threshold for creating binary pseudo-mask
    
    Returns:
        iou_score (float): Mean IoU score
        dice_score (float): Mean Dice score
    """
    model.to(device)
    model.eval()
    extractor.to(device)
    iou_pet = []
    dice_pet = []

    # Loop through the test dataset with tqdm progress bar
    for i, batch in enumerate(testloader):
    #for i, batch in enumerate(testloader):
        images, gt_mask = batch[:2]  # Ignore labels
        images = images.to(device)
        gt_mask = gt_mask.to('cpu').numpy()  # [B, H, W]

        # Generate CAM pseudo-masks
        pred_masks = []
        for img in images:
            _, pred_mask, _ = generate_cam(img, model, extractor, device, threshold)
            pred_masks.append(pred_mask)
        pred_masks = np.stack(pred_masks, axis=0)  # [B, H, W]

        # Evaluate IoU and Dice
        gt_mask_pet = (gt_mask == 0).astype(np.uint8)  # Foreground (pet) is 0 in ground-truth
        intersection = np.sum(gt_mask_pet * pred_masks, axis=(1, 2))
        union = np.sum(gt_mask_pet + pred_masks, axis=(1, 2))
        iou = intersection / (union + 1e-8)  # Avoid division by zero
        dice = (2 * intersection) / (union + intersection + 1e-8)
        
        iou_pet.extend(iou.tolist())
        dice_pet.extend(dice.tolist())

    iou_score = np.mean(iou_pet)
    dice_score = np.mean(dice_pet)
    return iou_score, dice_score

def evaluate_cam(dataloader, model, threshold=0.05):
    """
    Evaluate CAM on the test set with quantitative metrics and visualizations.
    
    Args:
        dataloader (OxfordIIITPetDataloader): Dataloader object
        model (nn.Module): Pretrained ResNet50 model
        extractor (CAMExtractor): Feature extractor
        num_images (int): Number of images to visualize per class
        max_model_index (int): Number of model indices (set to 1 for pretrained)
        threshold (float): Threshold for creating binary pseudo-mask
    """
    device = torch.device("cpu")
    model = model.to(device)
    extractor = CAMExtractor(model)
    extractor = extractor.to(device)
    
    # Get test loader
    test_loader = dataloader.get_test_loader_CCAM()
    
    # Quantitative evaluation
    iou_score, dice_score = evaluate_metric_cam(model, test_loader, device, extractor, threshold)
    print(f"\nQuantitative Results:")
    print(f"Mean IoU: {iou_score:.4f}")
    print(f"Mean Dice: {dice_score:.4f}")

### cam_ecs ###
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def denormalize(tensor, device):
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor (torch.Tensor): Input tensor to denormalize
        device (torch.device): Device to place mean and std tensors on
    
    Returns:
        torch.Tensor: Denormalized tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)
def generate_ecs_cam(image, model, extractor, device, delta=0.5, theta=0.8, threshold=0.05):
    """
    Generate ECS-CAM and pseudo-mask for a single image.
    
    Args:
        image (torch.Tensor): Input image tensor [C, H, W]
        model (nn.Module): Pretrained ResNet50 model
        extractor (CAMExtractor): Feature extractor
        device (torch.device): Device to run the model on
        delta (float): Threshold for original CAM mask
        theta (float): Threshold for noise suppression
        threshold (float): Threshold for creating binary pseudo-mask
    
    Returns:
        final_cam_np (np.ndarray): Final ECS-CAM heatmap [224, 224]
        pred_mask (np.ndarray): Binary pseudo-mask [224, 224]
        target_class (int): Predicted class index
    """
    image = image.unsqueeze(0)  # [1, C, H, W]
    image = image.to(device)
    model = model.to(device)
    extractor = extractor.to(device)
    
    # Ensure fc_weights are on the correct device
    extractor.fc_weights = extractor.fc_weights.to(device)
    
    # Step 1: Generate original CAM
    with torch.no_grad():
        features = extractor(image)  # [1, 2048, 7, 7]
        _, _, logits = model(image)  # [1, 1000]
        probs = F.softmax(logits, dim=1)
    
    target_class = torch.argmax(probs).item()
    
    # Initialize original_cam on the correct device
    original_cam = torch.zeros(features.shape[2:], dtype=torch.float32, device=device)  # [7, 7]
    for i, w in enumerate(extractor.fc_weights[target_class]):
        original_cam += w * features[0, i, :, :]
    
    original_cam = F.relu(original_cam)
    original_cam = original_cam / original_cam.max()
    original_cam = F.interpolate(original_cam.unsqueeze(0).unsqueeze(0), size=(224, 224), 
                                mode='bilinear', align_corners=False)[0, 0]
    original_cam_np = original_cam.cpu().detach().numpy()
    
    # Step 2: Create erased image
    mask = (original_cam_np > delta).astype(np.uint8)  # δ=0.5 threshold
    image_np = denormalize(image[0], device).permute(1, 2, 0).cpu().numpy()  # [224, 224, 3]
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    
    blurred_image = image_pil.filter(ImageFilter.GaussianBlur(radius=20))
    erased_np = np.where(np.expand_dims(mask, -1), 
                        np.array(blurred_image.resize((224, 224))), 
                        np.array(image_pil.resize((224, 224))))
    erased_image = Image.fromarray(erased_np.astype(np.uint8))
    
    # Step 3: Generate erased CAM
    erased_tensor = preprocess(erased_image).unsqueeze(0).to(device)
    with torch.no_grad():
        erased_features = extractor(erased_tensor)  # [1, 2048, 7, 7]
    
    # Initialize erased_cam on the correct device
    erased_cam = torch.zeros(erased_features.shape[2:], dtype=torch.float32, device=device)  # [7, 7]
    for i, w in enumerate(extractor.fc_weights[target_class]):
        erased_cam += w * erased_features[0, i, :, :]
    
    erased_cam = F.relu(erased_cam)
    erased_cam = erased_cam / erased_cam.max()
    erased_cam = F.interpolate(erased_cam.unsqueeze(0).unsqueeze(0), size=(224, 224), 
                              mode='bilinear', align_corners=False)[0, 0]
    erased_cam_np = erased_cam.cpu().detach().numpy()
    
    # Step 4: Noise suppression
    score_map = np.max(erased_cam_np, axis=0)  # Simplified score map
    suppressed_mask = (score_map > theta) & (mask == 0)  # θ=0.8 and exclude erased regions
    final_cam_np = np.where(suppressed_mask, erased_cam_np, original_cam_np)
    
    # Step 5: Generate binary pseudo-mask
    pred_mask = (final_cam_np > threshold).astype(np.uint8)
    
    return final_cam_np, pred_mask, target_class
def evaluate_cam_ecs(model, testloader, device, delta=0.5, theta=0.8, threshold=0.05):
    """
    Evaluate ECS-CAM on the test set using IoU and Dice metrics.
    
    Args:
        model (nn.Module): Pretrained ResNet50 model
        testloader (DataLoader): Test dataloader
        device (torch.device): Device to run the model on
        extractor (CAMExtractor): Feature extractor
        delta (float): Threshold for original CAM mask
        theta (float): Threshold for noise suppression
        threshold (float): Threshold for creating binary pseudo-mask
    
    Returns:
        iou_score (float): Mean IoU score
        dice_score (float): Mean Dice score
    """
    model.to(device)
    model.eval()
    extractor = CAMExtractor(model)
    extractor.to(device)
    iou_pet = []
    dice_pet = []

    # Loop through the test dataset with tqdm progress bar
    for i, batch in enumerate(testloader):
        images, gt_mask, _ = batch  # Ignore labels
        images = images.to(device)
        gt_mask = gt_mask.to('cpu').numpy()  # [B, H, W]

        # Generate ECS-CAM pseudo-masks
        pred_masks = []
        for img in images:
            _, pred_mask, _ = generate_ecs_cam(img, model, extractor, device, delta, theta, threshold)
            pred_masks.append(pred_mask)
        pred_masks = np.stack(pred_masks, axis=0)  # [B, H, W]

        # Evaluate IoU and Dice
        gt_mask_pet = (gt_mask == 0).astype(np.uint8)  # Foreground (pet) is 0 in ground-truth
        intersection = np.sum(gt_mask_pet * pred_masks, axis=(1, 2))
        union = np.sum(gt_mask_pet + pred_masks, axis=(1, 2))
        iou = intersection / (union + 1e-8)  # Avoid division by zero
        dice = (2 * intersection) / (union + intersection + 1e-8)
        
        iou_pet.extend(iou.tolist())
        dice_pet.extend(dice.tolist())

    iou_score = np.mean(iou_pet)
    dice_score = np.mean(dice_pet)
    print(f"\nQuantitative Results:")
    print(f"Mean IoU: {iou_score:.4f}")
    print(f"Mean Dice: {dice_score:.4f}")
    return iou_score, dice_score

def evaluate_ccam_pet(model, testloader, device, threshold=0.05):
    # Ensure model is in evaluation mode and moved to device.
    model.to(device)
    model.eval()
    iou_pet = []
    dice_pet = []

    # Loop through the test dataset with tqdm progress bar
    for i, batch in enumerate(testloader):
        images, gt_mask= batch[:2]
        images = images.to(device)
        gt_mask = gt_mask.to('cpu').numpy()

        # Generate pseudo masks using the CCAM method.
        pred_masks = generate_pseudo_mask(images, model, threshold=threshold)
        # pred_masks = 1 - pred_masks
        gt_mask_pet = ((gt_mask == 0) + (gt_mask == 2))
        intersection = np.logical_and(pred_masks, gt_mask_pet).sum()
        union = np.logical_or(pred_masks, gt_mask_pet).sum()
        iou = intersection / union
        dice = (2 * intersection) / (union + intersection)
        iou_pet.append(iou)
        dice_pet.append(dice)
    iou_score = np.mean(iou_pet)
    dice_score = np.mean(dice_pet)
    return iou_score, dice_score

def evaluate_class_specific_ccam(dataloader):
    iou_pet = []
    dice_pet = []
    for i, data in enumerate(dataloader):
        image,pm,gt,_ = data
        gt = gt.to('cpu').numpy()
        gt_pet = ((gt == 0) + (gt == 2))
        intersection = np.logical_and(pm, gt_pet).sum()
        union = np.logical_or(pm, gt_pet).sum()
        iou = intersection / union
        dice = (2 * intersection) / (union + intersection)
        if not torch.isnan(iou) and not torch.isnan(dice):
            iou_pet.append(iou)
            dice_pet.append(dice)
    iou_score = np.mean(iou_pet)
    dice_score = np.mean(dice_pet)
    return iou_score, dice_score

def evaluate_ccam_crf(model, dataloader, device, threshold=0.05):
    # Ensure model is in evaluation mode and moved to device.
    model.to(device)
    model.eval()
    iou_pet = []
    dice_pet = []
    for i, data in enumerate(dataloader):
        images, gt_mask = data[:2]
        images = images.to(device)
        crf = calc_cam_crfs(images,model,device)
        crf = crf.to('cpu').numpy()
        gt_mask = gt_mask.to('cpu').numpy()
        gt_mask_pet = ((gt_mask == 0) + (gt_mask == 2))
        intersection = np.logical_and(crf, gt_mask_pet).sum()
        union = np.logical_or(crf, gt_mask_pet).sum()
        iou = intersection / union
        dice = (2 * intersection) / (union + intersection)
        iou_pet.append(iou)
        dice_pet.append(dice)
    iou_score = np.mean(iou_pet)
    dice_score = np.mean(dice_pet)
    return iou_score, dice_score


        

def evaluate_unet(model, dataloader, device):
    iou_pet = []
    dice_pet = []
    model.to(device)
    model.eval()
    for i, data in enumerate(dataloader):
        image,gt,_ = data
        image = image.to(device)
        gt = gt.to('cpu').numpy()
        pm = model(image)
        pm = pm.detach().to('cpu').numpy()
        pm = np.argmax(pm, axis=1)
        gt_pet = ((gt == 0) + (gt == 2))
        intersection = np.logical_and(pm, gt_pet).sum()
        union = np.logical_or(pm, gt_pet).sum()
        iou = intersection / union
        dice = (2 * intersection) / (union + intersection)
        iou_pet.append(iou)
        dice_pet.append(dice)
    iou_score = np.mean(iou_pet)
    dice_score = np.mean(dice_pet)
    return iou_score, dice_score
        
