'''
#used for bottle:
import torch
import torchvision.models as models
import torch.nn as nn


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pool

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x
    
'''

#used or tile:
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.layer0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)

        feat_l2 = self.layer2(x)  # (B, 128, H, W)
        feat_l3 = self.layer3(feat_l2)  # (B, 256, H/2, W/2)

        # Downsample layer2 to match layer3 spatial size
        feat_l2_down = F.avg_pool2d(feat_l2, kernel_size=2)

        # Concatenate along channel dimension
        feat = torch.cat([feat_l2_down, feat_l3], dim=1)

        return feat  # (B, 384, H/2, W/2)

