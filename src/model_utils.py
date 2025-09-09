import os
import torch
import numpy as np
import torchvision
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels    
from torchvision import transforms

# =============================== inference setup =============================

def get_cam(model,ori_image, scale):
    image = ori_image.clone()
    
    # preprocessing
    image = F.interpolate(
    image,
    scale_factor=scale,
    mode='bicubic',
    align_corners=False
)  # Remove batch dimension
    flipped_image = image.flip(-1)
    
    images = torch.cat([image, flipped_image],dim=0)  # [2, C, H, W]
    images = images.cuda()  # Move to GPU if available
    
    # inferenece
    _, _, cams = model(images, inference=True)


    # postprocessing
    cams = F.relu(cams)
    # cams = torch.sigmoid(features)
    cams = cams[0] + cams[1].flip(-1)
    flag =( cams[0,0,0]+cams[0,0,-1]+cams[0,-1,0]+cams[0,-1,-1])/4 > 0.5
    if flag:
        cams = 1 - cams

    return cams

def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride

def make_cam(x, epsilon=1e-5):
# relu(x) = max(x, 0)
    x = F.relu(x)
    
    b, c, h, w = x.size()

    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(dim=-1)[0].view((b, c, 1, 1))
    
    return F.relu(x - epsilon) / (max_value + epsilon)

def resize_for_tensors(tensors, size, mode='bilinear', align_corners=False):
    return F.interpolate(tensors, size, mode=mode, align_corners=align_corners) 

def gen_hr_cams(image, model, scales=[0.5, 1.0, 1.5, 2.0], threshold=0.5, device='cuda', return_soft=False):
    
    # Get original size
    ori_h, ori_w = image.shape[-2], image.shape[-1]
    
    # Process image at each scale
    
    with torch.no_grad():
        label = np.array([1])
        cams_list = [get_cam(model,image, scale) for scale in scales]
        strided_size = get_strided_size((ori_h, ori_w), 4)
        strided_up_size = get_strided_up_size((ori_h, ori_w), 16)
        strided_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_size)[0] for cams in cams_list]
        strided_cams = torch.sum(torch.stack(strided_cams_list), dim=0)
        
        hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
        hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]
        
        keys = torch.nonzero(torch.from_numpy(label))[:, 0]
        
        strided_cams = strided_cams[keys]
        strided_cams /= F.adaptive_max_pool2d(strided_cams, (1, 1)) + 1e-5
        
        hr_cams = hr_cams[keys]
        hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5
        return hr_cams

def generate_pseudo_mask(image, model, scales=[0.5, 1.0, 1.5, 2.0], threshold=0.5, device='cuda', return_soft=False):
    """
    Generate a pseudo-mask for a single preprocessed input image tensor using CCAMNetwork.
    
    Args:
        image: torch.Tensor [C, H, W], preprocessed (normalized, e.g., ResNet50 mean/std)
        model: CCAMNetwork instance
        scales: List of float scales for multi-scale inference (default: [1.0])
        threshold: Float for binarizing the mask (None for soft mask)
        device: torch.device or str ('cuda' or 'cpu')
        return_soft: Bool, if True, return soft mask even if threshold is set
    
    Returns:
        pseudo_mask: np.ndarray [H, W], binary or soft mask matching input image size
    """
        # Ensure model is in eval mode
    model.eval()
    
    # Handle device
    device = torch.device(device) if isinstance(device, str) else device
    model.to(device)
    image = image.to(device)
    
    # Validate input
    if not isinstance(image, torch.Tensor):
        raise ValueError("Input image must be a torch.Tensor")
    if image.dim() == 3:  # [C, H, W]
        image = image.unsqueeze(0)  # Add batch dim: [1, C, H, W]
    elif image.dim() != 4:
        raise ValueError(f"Expected image tensor with 3 or 4 dims, got {image.dim()}")

    hr_cams = gen_hr_cams(image, model, scales=scales, threshold=threshold, device=device, return_soft=return_soft)
    pseudo_mask = hr_cams.squeeze(0).cpu().numpy()  # [H, W]
    pseudo_mask = (pseudo_mask > threshold)
    return pseudo_mask

def check_positive(am):
    edge_mean = (am[:, 0, 0:4, :].mean() + am[:, 0, :, 0:4].mean() + am[:, 0, -4:, :].mean() + am[:, 0, :, -4:].mean()) / 4
    return edge_mean > 0.5


def calc_ecs_cam(ccam_model, images, device):
    images = images.to(device)
    ccam_model = ccam_model.to(device)
    features, _, targets = ccam_model.backbone.init_model(images)
    target_class = torch.argmax(targets, dim=1)
    cams = torch.zeros((features.shape[0],14,14), dtype=torch.float32).to(device)
    cams_mask = torch.zeros((features.shape[0],224,224), dtype=torch.float32).to(device)
    for i in range(features.shape[0]):
        for j, w in enumerate(ccam_model.backbone.init_model.fc.weight[target_class[i]]):
            cams[i] += w * features[i, j, :, :]
        cams[i] = F.relu(cams[i])
        cams[i] = cams[i] / cams[i].max()
        cams_mask[i] = F.interpolate(cams[i].unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)[0, 0]
    # Create erased image
    masks = (cams_mask > 0.6)  # δ=0.6 threshold
    blur = transforms.GaussianBlur(kernel_size=15, sigma=(0.1, 2.0))
    blurred_images = blur(images)
    erased_images = torch.where(masks.unsqueeze(1), blurred_images, images)


    features_ecs, _, targets_ecs = ccam_model.backbone.init_model(erased_images)
    target_class_ecs = torch.argmax(targets_ecs, dim=1)
    
    cams_ecs = torch.zeros((features_ecs.shape[0],14,14), dtype=torch.float32).to(device)
    cams_ecs_mask = torch.zeros((features_ecs.shape[0],224,224), dtype=torch.float32).to(device)
    for i in range(features_ecs.shape[0]):
        for j, w in enumerate(ccam_model.backbone.init_model.fc.weight[target_class_ecs[i]]):
            cams_ecs[i] += w * features_ecs[i, j, :, :]
        cams_ecs[i] = F.relu(cams_ecs[i])
        cams_ecs[i] = cams_ecs[i] / cams_ecs[i].max()

    final_cams = torch.max(cams, cams_ecs)
    return final_cams

# ===================================== loss setup ================================

def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)


def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return 1 - sim


def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()

    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)

    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)

    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C


# =============================================================================
# =============================================================================
# ================================== CRF  =====================================
# =============================================================================
# =============================================================================     

def crf_inference_label(img, labels, t=10, n_labels=38, gt_prob=0.7):

    h, w = img.shape[:2]

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def calc_cam_crfs(images,model,device):
    model.eval()
    model.to(device)
    cam_crfs_list = []
    with torch.no_grad():
        for image in images:
            image = image.to(device)
            keys = np.array([0, 1], dtype=np.uint8)
            image_4d = image.unsqueeze(0).to(device)

            # calculate the hr_cam
            cam_hr = gen_hr_cams(image_4d, model).cpu().detach().numpy()
            cam = np.pad(cam_hr, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.3)
            cam = np.argmax(cam, axis=0)

            image_np = np.asarray(image.cpu().permute(1, 2, 0).numpy())
            image_np = (image_np * 255).astype(np.uint8)


            cam_crf = crf_inference_label(np.asarray(image_np), cam, n_labels=keys.shape[0], t=10)
            cam_crfs_list.append(torch.from_numpy(cam_crf).float().unsqueeze(0))
    cam_crfs_list = torch.cat(cam_crfs_list, dim=0)
    return cam_crfs_list       # (B, H, W) 


def cam_to_ir_label(dataloader, cams, ir_label_out_dir, conf_fg_thres, conf_bg_thres):
    """
    Converts high-resolution CAMs into refined label masks using CRF,
    then writes them as PNGs via torchvision, without VOC dependencies.

    Args:
        dataloader: iterable yielding dicts with keys:
            'img': Tensor[N,C,H,W] or [C,H,W] (values in [0,255] or [0,1])
            'name': list or tensor of filenames or unique identifiers (strings)
        cam_out_dir: Tensor for prob CAMs
        ir_label_out_dir: directory to save the IR label PNGs
        conf_fg_thres: float, confidence threshold to pad CAMs for foreground
        conf_bg_thres: float, confidence threshold to pad CAMs for background
    """
    os.makedirs(ir_label_out_dir, exist_ok=True)

    for pack in dataloader:
        # assume batch size 1 for simplicity; adapt if batching
        img_tensor = pack['img'][0]  
        img_name = pack['name'][0]   

        # Convert to HxWxC uint8 numpy array
        img_np = img_tensor.detach().cpu().numpy()
        if img_np.ndim == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        img_uint8 = img_np.astype(np.uint8)

        keys = np.array([0, 1], dtype=np.uint8) 

        # Foreground confidence map
        fg_cam = np.pad(
            cams,
            ((1, 0), (0, 0), (0, 0)),
            mode='constant',
            constant_values=conf_fg_thres
        )
        fg_labels = np.argmax(fg_cam, axis=0)
        fg_refined = crf_inference_label(
            img_uint8, fg_labels, n_labels=keys.shape[0]
        )
        fg_conf = keys[fg_refined]

        # Background confidence map
        bg_cam = np.pad(
            cams,
            ((1, 0), (0, 0), (0, 0)),
            mode='constant',
            constant_values=conf_bg_thres
        )
        bg_labels = np.argmax(bg_cam, axis=0)
        bg_refined = crf_inference_label(
            img_uint8, bg_labels, n_labels=keys.shape[0]
        )
        bg_conf = keys[bg_refined]

        # Combine fg/bg into final conf map
        conf = fg_conf.copy()
        conf[fg_conf == 0] = 255
        conf[(bg_conf + fg_conf) == 0] = 0

        # Save as PNG using torchvision
        conf_tensor = torch.from_numpy(conf.astype(np.uint8))[None, ...]
        out_path = os.path.join(ir_label_out_dir, img_name + '.png')
        torchvision.io.write_png(conf_tensor, out_path)