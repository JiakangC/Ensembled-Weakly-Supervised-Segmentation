import os
import sys
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
# Calculate the root directory (COMP0197_Group)
ROOT = os.path.abspath(os.path.join(current_dir, '..'))  # Up one levels
sys.path.append(ROOT)

from src.data import *
from src.model import *
from src.valuate import *

ROOT = os.path.abspath(os.path.join(current_dir,'..'))

def eval_ccam_unet(model_dir, datasets,device, with_crf = False):
    # List all model files in the directory that match the specific pattern
    prefix = "unet_specific_class_37_crf_ccam_epoch" if use_crf else "unet_specific_class_37_ccam_epoch"
    model_files = [
        f for f in os.listdir(model_dir)
        if f.startswith(prefix) and f.endswith(".pth")
    ]

    # Sort the model files based on the epoch number extracted from the filename
    model_files = sorted(
        model_files,
        key=lambda f: int(re.search(r"_epoch_(\d+)\.pth", f).group(1))
    )

    results = []
    for fname in model_files:
        # Extract the epoch number from the filename
        epoch = int(re.search(r"_epoch_(\d+)\.pth", fname).group(1))
        # Construct the full path to the model file
        path  = os.path.join(model_dir, fname)
        # 4) (Re-)create your model and load weights
        model = UNet(n_channels=3, n_classes=2, filters=[32, 64, 128, 256])
        # Load the model weights from the file, mapping them to the specified device
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        # Move the model to the specified device and set it to evaluation mode
        model.to(device).eval()

        iou_pet = []
        dice_pet = []
        # Disable gradient calculation for inference
        with torch.no_grad():
            for i, data in enumerate(datasets):
                image, _ , gt, _ = data
                logits = model(image.unsqueeze(0).to(device))

                # 2) turn into CPU numpy prediction mask
                pm = logits.argmax(dim=1).squeeze(0).cpu().numpy()
                gt_pet = ((gt == 0) + (gt == 2))
                intersection = np.logical_and(pm, gt_pet).sum()
                union = np.logical_or(pm, gt_pet).sum()
                iou = intersection / union
                dice = (2 * intersection) / (union + intersection)
                iou_pet.append(iou)
                dice_pet.append(dice)
        iou_score = np.mean(iou_pet)
        dice_score = np.mean(dice_pet)
        results.append({
            "epoch": epoch,
            "iou"  : float(np.mean(iou_pet )),
            "dice" : float(np.mean(dice_pet))
        })
    return results



if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(ROOT, 'train_unet','saved_models')

    base_dataLoader = OxfordIIITPetDataloader()
    base_test_loader = base_dataLoader.get_test_loader_CCAM()

    base_model = UNet(n_channels=3, n_classes=2, filters=[32, 64, 128, 256])
    base_model.load_state_dict(torch.load(os.path.join(model_dir, 'unet_baseline.pth'), map_location=device, weights_only=True))

    base_iou, base_dice = evaluate_unet(base_model, base_test_loader, device)

    # Load Model with 37-class data without CRF
    datasets_without_crf = ImagePseudoMaskDataset(
        root_image_dir=os.path.join(ROOT,"data", "class_folders"),
        root_mask_dir=os.path.join(ROOT,"data", "37_class_pseudo_mask_folders"),
        root_gt_dir=os.path.join(ROOT,"data", "37_class_gt")
    )
    
    results_without_crf = eval_ccam_unet(model_dir, datasets_with_crf, device, with_crf=False)

    # Load Model with 37-class data without CRF
    datasets_without_crf = ImagePseudoMaskDataset(
        root_image_dir=os.path.join(ROOT,"data", "class_folders"),
        root_mask_dir=os.path.join(ROOT,"data", "37_class_pseudo_mask_folders_crf"),
        root_gt_dir=os.path.join(ROOT,"data", "37_class_gt")
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_with_crf = eval_ccam_unet(model_dir, datasets_with_crf, device, with_crf=True)
    print(f'Results for baseline - iou:{base_iou} Dice:{base_dice}')
    print(f'Results without CRF: {results_without_crf}')
    print(f'Results with CRF: {results_with_crf}')



    