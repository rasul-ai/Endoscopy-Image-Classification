import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import CNNwithSE, SEResNet


class Classifier(nn.Module):
    """
    Classifier for the no-cross-attention variant.

    Input dimensionality is 320:
      - 256 from 4 x 64-channel local (CNNwithSE) features (global-avg-pooled)
      -  64 from the global (SEResNet) features (global-avg-pooled)
    """
    def __init__(self, input_dim: int = 320):
        super(Classifier, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),

            nn.Linear(128, 15),
        )

    def forward(self, x):
        return self.fc_layers(x)


class LAGNoCA(nn.Module):
    """
    Ablation "Fusion (No CA)" (paper Table 6): CGLE without cross-attention.

    Local and global features are each global-average-pooled and then
    concatenated before being passed to the classifier.

    Feature layout:
      local  : 4 horizontal strips × 64 channels → pooled to (B, 256)
      global : SEResNet output (B, 64, 8, 8)     → pooled to (B, 64)
      fused  : cat → (B, 320)
    """

    def __init__(self):
        super(LAGNoCA, self).__init__()
        self.cnn_se_model = CNNwithSE()
        self.backbone = SEResNet()
        self.classifier = Classifier(input_dim=320)

        self.num_horizontal_parts = 4

    def forward(self, image):
        if image.dim() == 3:
            image = image.unsqueeze(0)

        B, C, H, W = image.shape

        # ---- Local branch ----
        h_step = H // self.num_horizontal_parts
        sub_module_outputs = []
        for i in range(self.num_horizontal_parts):
            h_start = i * h_step
            h_end = (i + 1) * h_step if i < self.num_horizontal_parts - 1 else H
            part = image[:, :, h_start:h_end, :]
            sub_module_outputs.append(self.cnn_se_model(part))

        # (B, 4*64, H_out, W_out) → pool → (B, 256)
        ensemble_local = torch.cat(sub_module_outputs, dim=1)
        local_pooled = F.adaptive_avg_pool2d(ensemble_local, (1, 1)).view(B, -1)

        # ---- Global branch ----
        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        backbone_output = self.backbone(resized_image)   # (B, 64, 8, 8)
        global_pooled = F.adaptive_avg_pool2d(backbone_output, (1, 1)).view(B, -1)  # (B, 64)

        # ---- Fusion ----
        fused = torch.cat([local_pooled, global_pooled], dim=1)  # (B, 320)

        return self.classifier(fused)
