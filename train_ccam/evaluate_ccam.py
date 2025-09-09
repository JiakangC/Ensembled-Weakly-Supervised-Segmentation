import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# Calculate the root directory (COMP0197_Group)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))  # Up two levels

sys.path.append(root_dir)
from src.model import *
from src.model_utils import *
from src.data import *
from src.train import *
from src.evaluate import *

# Assuming OxfordIIITPetDataloader is defined as provided
model_2_dir = os.path.join(root_dir, 'train_classifier', 'model_saved', 'resnet50_dog_cat.pth')
model_37_dir = os.path.join(root_dir, 'train_classifier', 'model_saved', 'resnet50_37_class.pth')
model_2_ECS_dir = os.path.join(root_dir, 'train_classifier', 'model_saved', 'resnet50_dog_cat.pth')
model_37_ECS_dir = os.path.join(root_dir, 'train_classifier', 'model_saved', 'resnet50_37_class.pth')
dataloader = OxfordIIITPetDataloader()
test_loader = dataloader.get_test_loader_label()
test_loader_box = dataloader.get_test_loader_CCAM()
# model = CCAM.get_ccam()
class_num = 2
save_name = "pretrained"
save_name_2 = "two_class"
save_name_37 = "three_seven_class"

for j in range(20):
    model = get_ccam(model_37_ECS_dir, target_class=37)
    #model = get_ccam()
    model_dir = os.path.join(root_dir, 'train_ccam', 'model_saved', f'{save_name_37}_ECS_log_{j}.pth')
    print(model_dir)
    ##########################################################################################
    model.load_state_dict(torch.load(model_dir, weights_only=True))

    model_name = f'{save_name_37}_log_{j}'
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results_ccam_score = evaluate_ccam_pet(model, 
                                                test_loader_box, 
                                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                                threshold=0.8
                                                )
    print(results_ccam_score)
    

for j in range(20):
    model = get_ccam(model_37_dir, target_class=37)
    #model = get_ccam()
    model_dir = os.path.join(root_dir, 'train_ccam', 'model_saved', f'{save_name_37}_log_{j}.pth')
    print(model_dir)
    ##########################################################################################
    model.load_state_dict(torch.load(model_dir, weights_only=True))

    model_name = f'{save_name_37}_log_{j}'
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results_ccam_score = evaluate_ccam_pet(model, 
                                                test_loader_box, 
                                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                                threshold=0.8
                                                )
    print(results_ccam_score)
    
for j in range(20):
    model = get_ccam(model_2_dir, target_class=2)
    #model = get_ccam()
    model_dir = os.path.join(root_dir, 'train_ccam', 'model_saved', f'{save_name_2}_log_{j}.pth')
    print(model_dir)
    ##########################################################################################
    model.load_state_dict(torch.load(model_dir, weights_only=True))

    model_name = f'{save_name_2}_log_{j}'
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results_ccam_score = evaluate_ccam_pet(model, 
                                                test_loader_box, 
                                                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                                threshold=0.8
                                                )
    print(results_ccam_score)
    