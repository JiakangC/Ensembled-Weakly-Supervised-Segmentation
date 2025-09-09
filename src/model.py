import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url as load_url

import os
import sys

current_dir = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(current_dir, 'src') 
sys.path.append(src_dir)
print(src_dir)

import src.model_utils as model_utils

# =============================================================================
# =============================================================================
# ==================================  CCAM  ===================================
# =============================================================================
# ============================================================================= 
    
# ==============================modify resnet 50===============================

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, stride=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        features = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embedded = x
        x = self.fc(x)

        return features,embedded, x

    def get_parameter_groups(self, print_fn=print):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():
            # pretrained weights
            if 'fc' not in name:
                if 'weight' in name:
                    print_fn(f'pretrained weights : {name}')
                    groups[0].append(value)
                else:
                    print_fn(f'pretrained bias : {name}')
                    groups[1].append(value)

            # scracthed weights
            else:
                if 'weight' in name:
                    if print_fn is not None:
                        print_fn(f'scratched weights : {name}')
                    groups[2].append(value)
                else:
                    if print_fn is not None:
                        print_fn(f'scratched bias : {name}')
                    groups[3].append(value)
        return groups


# ================================== model setup===============================

def resnet50(pretrained=True, model_path = None, stride=None, num_classes=1000, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param pretrained:
        :param stride:
    """
    if stride is None:
        stride = [1, 2, 2, 1]
    model = ResNet(Bottleneck, [3, 4, 6, 3], stride=stride, **kwargs)
    if pretrained:
        model.load_state_dict(load_url("https://download.pytorch.org/models/resnet50-19c8e357.pth"), strict=True)
    model.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
    if model_path is not None:
        checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
    
    return model

class ResNet50(nn.Module):
    def __init__(self,model_path = None,target_class = 1000):
        super(ResNet50, self).__init__()
        self.init_model = resnet50(pretrained=True, model_path = model_path,num_classes= target_class)
        # model.layer4 = model._make_layer(models.resnet.Bottleneck, 512,3,1)
        self.conv1 = self.init_model.conv1
        self.bn1 = self.init_model.bn1
        self.relu = self.init_model.relu
        self.maxpool = self.init_model.maxpool
        self.layer1 = self.init_model.layer1
        self.layer2 = self.init_model.layer2
        self.layer3 = self.init_model.layer3
        self.layer4 = self.init_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x2 = self.layer4(x1)
        return torch.cat([x2, x1], dim=1)

class Disentangler(nn.Module):
    def __init__(self, channel_in):
        super(Disentangler, self).__init__()
        self.activation_head = nn.Conv2d(channel_in, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)

    def forward(self, x, inference=False):
        N, C, H, W = x.size()
        if inference:
            ccam = self.bn_head(self.activation_head(x))
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))

        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam
    
class CCAMNetwork(nn.Module):
    def __init__(self, channel_in=None, model_path = None, target_class = 1000):
        super(CCAMNetwork, self).__init__()

        self.backbone = ResNet50(model_path=model_path,target_class = target_class)
        self.ac_head = Disentangler(channel_in)
        self.from_scratch_layers = [self.ac_head]

    def forward(self, x, inference=False):

        feats = self.backbone(x)
        fg_feats, bg_feats, ccam = self.ac_head(feats, inference=inference)

        return fg_feats, bg_feats, ccam

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups

def get_ccam(channel_in = 2048+1024 ,model_path = None, target_class = 1000):
    return CCAMNetwork(channel_in=channel_in,model_path = model_path,target_class = target_class)

class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction

    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = model_utils.cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = model_utils.cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)

        
# =============================================================================
# =============================================================================
# ==============================  Segmentation  ===============================
# =============================================================================
# ============================================================================= 

# ---------------------
#   基础模块：Conv → BN → ReLU × 2
# ---------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ---------------------
#       Down block
# ---------------------
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.down(x)

# ---------------------
#       Up block
# ---------------------
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ---------------------
#       Output
# ---------------------
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.out = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.out(x)

# ---------------------
#       UNet 总体结构
# ---------------------
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, filters=[32, 64, 128, 256]):
        super().__init__()
        # 编码器
        self.inc = DoubleConv(n_channels, filters[0])
        self.down1 = Down(filters[0], filters[1])
        self.down2 = Down(filters[1], filters[2])
        self.down3 = Down(filters[2], filters[3])

        # bottleneck
        self.middle = DoubleConv(filters[3], filters[3])

        # 解码器
        self.up3 = Up(filters[3], filters[2])
        self.up2 = Up(filters[2], filters[1])
        self.up1 = Up(filters[1], filters[0])

        # 输出层
        self.outc = OutConv(filters[0], n_classes)

        # dummy layer to store the middle mayer output
        self.encoder_last_features = None

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_mid = self.middle(x4)
        x = self.up3(x_mid, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits


########### cam ############
# CAM Extractor class
class CAMExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Extract features before GAP and FC layers
        self.features = nn.Sequential(*list(model.children())[:-2])
        # Get FC layer weights
        self.fc_weights = model.fc.weight

    def forward(self, x):
        return self.features(x)
