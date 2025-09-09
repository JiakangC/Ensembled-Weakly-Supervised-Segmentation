# instruction of the code

1. Environment Setup
    - comp0197-cw1-pt base environment
    - cython ~pip install Cython
    - pydensecrf ~pip install pydensecrf
    - cv2 ~pip install opencv-python

2. Data Preparation
    - Download the dataset from the link provided in the assignment
    - Unzip the dataset and put it in the same directory as the code
    - For the background enhancing image, download from: https://www.kaggle.com/datasets/balraj98/stanford-background-dataset
    - The dataset should be in the following format:
        - data/
            - oxford-iiit-pet/
                - annotation/
                - images/
                - annotations.tar.gz
                - images.tar.gz
            - background
                - images/

3. models:
    - The Resnet50 backbone is modified with stride sequence [1,2,2,1] with last 2 convolutions outputs as the features for future cam.
      details in src.model.ResNet50
    - The pretrained Resnet50 can be found https://download.pytorch.org/models/resnet50-19c8e357.pth

4. Training and evaluation:
    - train backbone classifier: Run train_classifier/train_*.py file for your specific classifier finetuning
        - in the same folder, there is a evaluation_backbone.py file for the base CAM/ECS_CAM evaluation
        - for evaluation, please load the trained model path for evaluation_backbone.py
    - train CCAM: Run train_ccam/train_ccam.py file with specific parameters.
        - in the same folder, there is a evaluation_backbone.py file for the base CCAM/ECS_CCAM evaluation
        - this training may take quite a long time, you can go to https://drive.google.com/drive/folders/10sFREPCyJv_EqRCBarTwA5PabusN4_x1 for the 
        existing model
    - train class specific CCAM:
        - run trani_class_specific_ccam/data_class_split.py first
        - run train_class_specific_ccam/train_ccam_on_cat_dog.py with specific parameters to get cat_dog specific CCAM
        - run train_class_specific_ccam/train_ccam_on_specific_37_class.py with specific parameters to get 37-class specific CCAM
        - pseudomask generating: run train_class_specific_ccam/generating_mask.py with pseudo-mask generating
        - for result evaluation: run train_class_specific_ccam/evaluate_class_specific_CCAM.py
    - Train Unet:
        - for Unet trained on ground truth run: train_unet/train_on_gt.py
        - for Unet trained on 37 specific class CCAM pseudo-mask run: train_unet/train_on_37CCAM.py
        - for Unet trained on 37 specific class CCAM pseudo-mask + CRF run: train_unet/train_on_37CCAM_CRF.py
        - for evaluating the performance run: train_unet/ccam_unet_eval.py
        
