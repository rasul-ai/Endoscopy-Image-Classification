"""Ablation "Local Branch Only" (paper Table 6): 4-strip CNNwithSE + MLP head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import CNNwithSE


LOCAL_FEATURE_CHANNELS = 256  # 4 horizontal parts × 64 channels each


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(LOCAL_FEATURE_CHANNELS, 1024),
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


class LocalOnly(nn.Module):
    def __init__(self):
        super(LocalOnly, self).__init__()
        self.cnn_se_model = CNNwithSE()
        self.classifier = Classifier()

    def forward(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        B, C, H, W = image.shape
        num_horizontal_parts = 4
        h_step = H // num_horizontal_parts

        sub_module_outputs = []

        for i in range(num_horizontal_parts):
            h_start = i * h_step
            h_end = (i + 1) * h_step if i < num_horizontal_parts - 1 else H
            part = image[:, :, h_start:h_end, :]
            output = self.cnn_se_model(part)
            sub_module_outputs.append(output)

        ensemble_sub_module = torch.cat(sub_module_outputs, dim=1)  # (B, 256, H_out, W_out)

        local_features = F.adaptive_avg_pool2d(ensemble_sub_module, (1, 1)).squeeze(-1).squeeze(-1)  # (B, 256)

        output = self.classifier(local_features)
        return output
