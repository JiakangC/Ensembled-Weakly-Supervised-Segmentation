import os
import random
import torch
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
from torchvision import datasets
import torchvision.transforms as T
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset, ConcatDataset,Dataset
from PIL import Image
import numpy as np



ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def target_transform(target):
    mask, label = target
    mask = T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST)(mask)
    mask = T.PILToTensor()(mask)
    mask = mask.squeeze(0).long()
    mask = mask-1
    return mask, label

def target_transform_bg(target):
    label = 37
    mask = torch.zeros(224, 224)
    return mask, label


class OxfordIIITPetWithBoxes(OxfordIIITPet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name_to_boxes = self._parse_boxes()
        self.no_box_indices = []
        self.has_box_indices = []
        
        for idx, image_path in enumerate(self._images):
            name = image_path.stem
            boxes = self.name_to_boxes.get(name, [])
            if len(boxes) == 0:
                self.no_box_indices.append(idx)
            else:
                self.has_box_indices.append(idx)

    def _parse_boxes(self):
        annotation_dir = os.path.join(self.root, "oxford-iiit-pet", "annotations", "xmls")
        name_to_boxes = {}
        for filename in os.listdir(annotation_dir):
            if not filename.endswith(".xml"):
                continue
            path = os.path.join(annotation_dir, filename)
            tree = ET.parse(path)
            root = tree.getroot()
            
            width = int(root.find("size/width").text)
            height = int(root.find("size/height").text)

            scale_x = 224 / width
            scale_y = 224 / height
            
            box = []
            for obj in root.findall("object"):
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)
                
                xmin = int(xmin * scale_x)
                ymin = int(ymin * scale_y)
                xmax = int(xmax * scale_x)
                ymax = int(ymax * scale_y)
            
                box = [xmin, ymin, xmax, ymax]
            name = os.path.splitext(filename)[0]
            name_to_boxes[name] = box
        return name_to_boxes

    def __getitem__(self, idx):
        image, (mask, label) = super().__getitem__(idx)
        image_path = self._images[idx]
        name = image_path.stem
        boxes = self.name_to_boxes.get(name, [])
        return image, mask, label, boxes 
    
    
class OxfordIIITPetDataloader:
    """
    This class loads the Oxford-IIIT Pet dataset and splits it into training and testing sets.
    By default, 80% of the data is used for training and 20% for testing.
    """

    def __init__(self, train_ratio=0.8, batch_size=16, num_workers=0, seed=42):
        """
        Initializes the dataset and performs the train/test split.
        
        Args:
            train_ratio (float, optional): Proportion of data to use for training. Default is 0.8.
            seed (int, optional): Random seed for reproducibility.
        """
    
        image_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),         
        ])

        # Load the full dataset
        # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.dataset_box = OxfordIIITPetWithBoxes(
            root=os.path.join(ROOT, "data"), 
            download=True, 
            target_types=('segmentation', 'category'),
            transform=image_transform,
            target_transform=target_transform,
        )
        random.seed(seed)
        box_dataset_size = len(self.dataset_box)
        # indices = list(range(box_dataset_size))
        # 
        # random.shuffle(indices)
        # split_idx = int(train_ratio * box_dataset_size)
        # self.train_indices = indices[:split_idx]
        # self.test_indices = indices[split_idx:]
        has_box_indices = self.dataset_box.has_box_indices.copy()
        random.shuffle(has_box_indices)
        split_idx = int(train_ratio * box_dataset_size)
        train_has_box = has_box_indices[:split_idx]
        test_has_box = has_box_indices[split_idx:]
        no_box_indices = self.dataset_box.no_box_indices
        self.train_indices_box = train_has_box
        self.test_indices_box = test_has_box + no_box_indices
        self.train_loader_box = DataLoader(
            Subset(self.dataset_box, self.train_indices_box),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_with_boxes
        )

        self.test_loader_box = DataLoader(
            Subset(self.dataset_box, self.test_indices_box),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_with_boxes
        )

        
        self.bg_dataset = datasets.ImageFolder(
            os.path.join(ROOT, "data", "background"),
            transform=image_transform,
            target_transform=target_transform_bg
        )


        self.whole_dataset_train = OxfordIIITPet(
            root=os.path.join(ROOT, "data"),
            download=True,
            target_types=('segmentation', 'category'),
            split='trainval',
            transform=image_transform,
            target_transform=target_transform
        )
        self.whole_dataset_test = OxfordIIITPet(
            root=os.path.join(ROOT, "data"),
            download=True,
            target_types=('segmentation', 'category'),
            split='test',
            transform=image_transform,
            target_transform=target_transform
        )
        self.whole_dataset = ConcatDataset([self.whole_dataset_train, self.whole_dataset_test, self.bg_dataset])
        whole_dataset_size = len(self.whole_dataset)
        indices_label = list(range(whole_dataset_size))
        random.shuffle(indices_label)
        split_label_idx = int(train_ratio * whole_dataset_size)
        self.train_indices_label = indices_label[:split_label_idx]
        self.test_indices_label = indices_label[split_label_idx:]
        
        
        self.train_loader_label = DataLoader(
            Subset(self.whole_dataset, self.train_indices_label),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_with_label
        )
        self.test_loader_label = DataLoader(
            Subset(self.whole_dataset, self.test_indices_label),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_with_label
        )
        
        

        self.whole_dataset_CCAM = ConcatDataset([self.whole_dataset_train, self.whole_dataset_test])
        whole_dataset_CCAM_size = len(self.whole_dataset_CCAM)
        indices_CCAM = list(range(whole_dataset_CCAM_size))
        random.shuffle(indices_CCAM)
        split_CCAM_idx = int(train_ratio * whole_dataset_CCAM_size)
        self.train_indices_CCAM = indices_CCAM[:split_CCAM_idx]
        self.test_indices_CCAM = indices_CCAM[split_CCAM_idx:]


        self.train_loader_CCAM = DataLoader(
            Subset(self.whole_dataset_CCAM, self.train_indices_CCAM),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_with_label
        )
        self.test_loader_CCAM = DataLoader(
            Subset(self.whole_dataset_CCAM, self.test_indices_CCAM),
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_with_label
        )

    def collate_with_boxes(self, batch):
        images, masks, labels, boxes = zip(*batch)
        return (
            torch.stack(images),              # [B, C, H, W]
            torch.stack(masks),               # [B, H, W]
            torch.tensor(labels),             # [B]
            list(boxes)                   
        )
    
    def collate_with_label(self, batch):
        images, targets = zip(*batch)
        masks,labels = zip(*targets)
        return (
            torch.stack(images),              # [B, C, H, W]
            torch.stack(masks),               # [B, H, W]
            torch.tensor(labels),             # [B]
        )
    
    def get_train_loader_box(self):
        return self.train_loader_box
    
    def get_test_loader_box(self):
        return self.test_loader_box
    
    def get_train_loader_label(self):
        return self.train_loader_label
    
    def get_test_loader_label(self):
        return self.test_loader_label
    
    def get_train_loader_CCAM(self):
        return self.train_loader_CCAM
    
    def get_test_loader_CCAM(self):
        return self.test_loader_CCAM


def split_pet_dataset():
    """
    Split Oxford-IIIT Pet Dataset images into folders by class.
    
    Args:
        dataset_dir (str): Path to the dataset directory containing 'images' folder
        output_dir (str): Path where class folders will be created
    """
    # Paths
    dataset_dir = os.path.join(ROOT, 'data', 'oxford-iiit-pet')
    output_dir = os.path.join(ROOT, 'data',  'class_folders_cat_dog')
    images_dir = os.path.join(dataset_dir, 'images')
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    trimaps_dir = os.path.join(annotations_dir, 'trimaps')
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    trimaps_files = [f for f in os.listdir(trimaps_dir) if f.endswith(( '.png')) and not f.startswith('.')]
    class_before = ''
    for img_file in image_files:
        
        # Extract class name from filename (e.g., 'Abyssinian_1.jpg' -> 'Abyssinian')
        class_name = '_'.join(img_file.split('_')[:-1])
        if class_name != class_before:
            i = 1
        # Create class directory
        class_dir = os.path.join(output_dir, class_name)
        Path(class_dir).mkdir(exist_ok=True)
        
        # Copy image to class directory
        target_name = f"{class_name}_{i}.jpg"
        src_path = os.path.join(images_dir, img_file)
        dst_path = os.path.join(class_dir, target_name)
        shutil.copy2(src_path, dst_path)
        
        print(f"Copied {img_file} to {class_name} folder")
        class_before = class_name
        i+=1
    class_before = ''
    for trimap in trimaps_files:
        # Extract class name from filename (e.g., 'Abyssinian_1.jpg' -> 'Abyssinian')
        class_name = '_'.join(trimap.split('_')[:-1])
        if class_name != class_before:
            i = 1
        # Create class directory
        class_dir = os.path.join(ROOT,'data','37_class_gt', class_name)
        os.makedirs(class_dir, exist_ok=True)
        Path(class_dir).mkdir(exist_ok=True)
        
        # Copy image to class directory
        target_name = f"{class_name}_{i}.png"
        src_path = os.path.join(trimaps_dir, trimap)
        dst_path = os.path.join(class_dir, target_name)
        shutil.copy2(src_path, dst_path)
        
        print(f"Copied {trimap} to {class_name} folder")
        class_before = class_name
        i+=1


def split_pet_dataset_cat_dog():
    """
    Split Oxford-IIIT Pet Dataset images into folders by class.
    """
    # path 
    dataset_dir = os.path.join(ROOT, 'data', 'oxford-iiit-pet')
    output_dir = os.path.join(ROOT, 'data',  'class_folders_cat_dog')
    images_dir = os.path.join(dataset_dir, 'images')
    annotations_dir = os.path.join(dataset_dir, 'annotations')
    trimaps_dir = os.path.join(annotations_dir, 'trimaps')
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    trimaps_files = [f for f in os.listdir(trimaps_dir) if f.endswith(( '.png')) and not f.startswith('.')]
    cat_count = 0
    dog_count = 0
    for img_file in image_files:
        
        # Extract class name from filename (e.g., 'Abyssinian_1.jpg' -> 'Abyssinian')
        class_name = '_'.join(img_file.split('_')[:-1])
        if class_name in ["Abyssinian","Bengal","Birman","Bombay","British_Shorthair","Egyptian_Mau","Maine_Coon","Persian","Ragdoll","Russian_Blue","Siamese","Sphynx"]:
            cat_or_dog = 'cat'
            cat_count += 1
            temp = cat_count
        else:
            cat_or_dog = 'dog'
            dog_count += 1
            temp = dog_count
        # Create class directory
        class_dir = os.path.join(output_dir, cat_or_dog)
        Path(class_dir).mkdir(exist_ok=True)
        
        # Copy image to class directory
        target_name = f"{cat_or_dog}_{temp}.jpg"
        src_path = os.path.join(images_dir, img_file)
        dst_path = os.path.join(class_dir, target_name)
        shutil.copy2(src_path, dst_path)
        
        print(f"Copied {img_file} to {cat_or_dog} folder")

    cat_count = 0
    dog_count = 0
    for trimap in trimaps_files:
        # Extract class name from filename (e.g., 'Abyssinian_1.jpg' -> 'Abyssinian')
        class_name = '_'.join(trimap.split('_')[:-1])
        if class_name in ["Abyssinian","Bengal","Birman","Bombay","British_Shorthair","Egyptian_Mau","Maine_Coon","Persian","Ragdoll","Russian_Blue","Siamese","Sphynx"]:
            cat_or_dog = 'cat'
            cat_count += 1
            temp = cat_count
        
        else:
            cat_or_dog = 'dog'
            dog_count += 1
            temp = dog_count
        # Create class directory
        class_dir = os.path.join(ROOT,'data','cat_dog_gt', cat_or_dog)
        os.makedirs(class_dir, exist_ok=True)
        Path(class_dir).mkdir(exist_ok=True)
        
        # Copy image to class directory
        target_name = f"{cat_or_dog}_{temp}.png"
        src_path = os.path.join(trimaps_dir, trimap)
        dst_path = os.path.join(class_dir, target_name)
        shutil.copy2(src_path, dst_path)
        
        print(f"Copied {trimap} to {cat_or_dog} folder")


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




class ImagePseudoMaskDataset(Dataset):
    def __init__(self, root_image_dir, root_mask_dir,root_gt_dir):
        
        self.root_image_dir = root_image_dir
        self.root_mask_dir = root_mask_dir
        self.root_gt_dir = root_gt_dir
        self.image_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),         
        ])
        self.mask_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        self.gt_transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.NEAREST),
            T.PILToTensor(),
            T.Lambda(lambda x: x.squeeze(0).long() - 1)
        ])
        self.classes = sorted(os.listdir(root_image_dir))  # List of class folders (e.g., Abyssinian, class2, ...)
        self.image_paths = []
        self.mask_paths = []
        self.gt_paths = []
        self.labels = []

        # Iterate through each class folder
        for label, class_name in enumerate(self.classes):
            image_dir = os.path.join(root_image_dir, class_name)
            mask_dir = os.path.join(root_mask_dir, class_name)
            gt_dir = os.path.join(root_gt_dir, class_name)

            # Get all images in the class folder
            images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
            for img_name in images:
                self.image_paths.append(os.path.join(image_dir, img_name))
                self.mask_paths.append(os.path.join(mask_dir, img_name.replace('.jpg', '.png')))  # Assumes mask has the same name
                self.gt_paths.append(os.path.join(gt_dir, img_name.replace('.jpg', '.png')))
                self.labels.append(label)  # Assign class label (0 to 36 for 37 classes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        gt_path = self.gt_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")  # Load image as RGB
        mask = Image.open(mask_path).convert("L")    # Load mask as grayscale
        gt = Image.open(gt_path).convert("L")        # Load gt as grayscale

        # Apply transformations if provided
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        gt = self.gt_transform(gt)

        return image, mask,gt, label



if __name__ == "__main__":
    # # KMP_DUPLICATE_LIB_OK=TRUE
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # # datasets = OxfordIIITPetDataloader()
    # # # split_pet_dataset()
    # # # print(len(datasets.whole_dataset_CCAM))
    # datasets = ImagePseudoMaskDataset(
    #     root_image_dir=os.path.join(ROOT, "data", "class_folders"),
    #     root_mask_dir=os.path.join(ROOT, "data", "37_class_pseudo_mask_folders_crf"),
    #     root_gt_dir=os.path.join(ROOT, "data", "37_class_gt")
    # )
    # image, mask,gt, label = datasets[40]
    # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # ax[0].imshow(image.permute(1, 2, 0))
    # ax[1].imshow(mask.squeeze(), cmap='gray')
    # ax[2].imshow(gt.squeeze(), cmap='gray')
    # plt.show()
    split_pet_dataset_cat_dog()





