import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
from src.model_utils import gen_hr_cams, generate_pseudo_mask
from src.model import get_ccam
from src.data import PetDataset
from src.model_utils import crf_inference_label







transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def make_all_crf_masks(class_name, data_dir, model_path, batch_size=32):
    """
    Visualize original images, CCAM heatmaps, and pseudo-masks for a given class.
    
    Args:
        class_name (str): Name of the class to visualize.
        data_dir (str): Path to the directory containing class folders.
        model_path (str): Path to the trained model weights.
        model_index (int): Index of the model (e.g., j in {class_name}_ccam_{j}.pth).
        num_images (int): Number of images to visualize.
        batch_size (int): Batch size for DataLoader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load weights
    model = get_ccam()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = PetDataset(data_dir=data_dir, class_name=class_name, transform=transform)
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) == 0:
        print(f"No images found for class {class_name}. Skipping.")
        return
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    index = 0
    # Get one batch of images
    for images, _ in dataloader:
        images = images.to(device)

        # Generate CCAMs and pseudo-masks
        ccam_list = [gen_hr_cams(image.unsqueeze(0), model).cpu() for image in images]
        mask_list = [generate_pseudo_mask(image.unsqueeze(0), model,threshold=0.25).astype(int) for image in images]
        images_np = [(np.asarray(image.cpu().permute(1, 2, 0).numpy()) * 255).astype(np.uint8) for image in images]
        crf_mask_list = [crf_inference_label(np.asarray(images_np[i]), mask_list[i], n_labels=2, t=10) for i in range(len(images_np))]
        
        # Denormalize images for visualization
        images = images.cpu().detach().numpy()
        
        save_path_crf = os.path.join(ROOT, "data", "37_class_pseudo_mask_folders_crf",class_name)
        os.makedirs(save_path_crf, exist_ok=True)
        save_path_pm = os.path.join(ROOT, "data", "37_class_pseudo_mask_folders",class_name)
        os.makedirs(save_path_pm, exist_ok=True)
        print(os.getcwd())
        for i in range(len(crf_mask_list)):
            index += 1

            mask_crf = crf_mask_list[i]
            mask_crf = (mask_crf* 255).astype(np.uint8)   # Convert to uint8 for saving
            save_name_crf = os.path.join("data", "37_class_pseudo_mask_folders_crf",class_name, f"{class_name}_{index}.png")
            mask_crf_image = Image.fromarray(mask_crf)
            mask_crf_image.save(save_name_crf)


            pm = mask_list[i]
            pm = (pm* 255).astype(np.uint8)   # Convert to uint8 for saving
            save_name_pm = os.path.join("data", "37_class_pseudo_mask_folders",class_name, f"{class_name}_{index}.png")
            pm_image = Image.fromarray(pm)
            pm_image.save(save_name_pm)

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    # Define data directory
    data_dir = os.path.join(ROOT, "data", "class_folders")
    
    # Get list of class folders
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    # class_names = ['Abyssinian']
    best_epoch = [19,19,19,19,19,19,9,19,10,19,19,19,19,5,19,19,19,16,19,16,13,19,19,19,19,15,19,19,19,19,19,19,19,19,15,19,19]

    # Visualize for each class and model index
    for i,class_name in enumerate(class_names):
        print(f"\nVisualizing for class: {class_name}")
        j = best_epoch[i]
        model_path = os.path.join(
            ROOT, 'train_class_specific_ccam', 'model_saved', 'class_specific_ccam', f'{class_name}_ccam_{j}.pth'
        )
        if os.path.exists(model_path):
            print(f"Found model at {model_path}")
            make_all_crf_masks(
                class_name=class_name,
                data_dir=data_dir,
                model_path=model_path,
                batch_size=32
            )
        else:
            print(f"Model not found for {class_name} at {model_path}. Skipping model index {j}.")
    
    data_dir = os.path.join(ROOT, "data", "class_folders_cat_dog")
    
    # # Get list of class folders
    class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    # class_names = ['Abyssinian']
    best_epoch = [3,3]


    for i,class_name in enumerate(class_names):
        print(f"\nVisualizing for class: {class_name}")
        j = best_epoch[i]
        model_path = os.path.join(
            ROOT, 'train_class_specific_ccam','model_saved', 'dog_cat_specific_ccam',  f'{class_name}_ccam_{j}.pth'
        )
        if os.path.exists(model_path):
            print(f"Found model at {model_path}")
            make_all_crf_masks(
                class_name=class_name,
                data_dir=data_dir,
                model_path=model_path,
                batch_size=32
            )
        else:
            print(f"Model not found for {class_name} at {model_path}. Skipping model index {j}.")