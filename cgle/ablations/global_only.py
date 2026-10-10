"""Ablation "Global Branch Only" (paper Table 6): SEResNet + MLP head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import SEResNet


GLOBAL_FEATURE_CHANNELS = 64  # SEResNet outputs (B, 64, 8, 8)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_CHANNELS, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 15)
        )

    def forward(self, x):
        x = self.fc_layers(x)
        return x


class GlobalOnly(nn.Module):
    def __init__(self):
        super(GlobalOnly, self).__init__()
        self.backbone = SEResNet()
        self.classifier = Classifier()

    def forward(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        backbone_output = self.backbone(resized_image)  # (B, 64, 8, 8)

        global_features = F.adaptive_avg_pool2d(backbone_output, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 64)

        output = self.classifier(global_features)
        return output
