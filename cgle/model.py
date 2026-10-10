"""
CGLE — Cross-Granularity Local Ensemble (paper Sec. 3).

`LAG` is the original class name used during development and stored in the
released checkpoints; `CGLE` is provided as an alias matching the paper.

  Local branch  (Sec. 3.1): 4 horizontal strips -> shared CNNwithSE -> concat (B, 256)
  Global branch (Sec. 3.2): image resized to 224x224 -> SEResNet -> (B, 64)
  Fusion        (Sec. 3.3): multi-head cross-attention, 8 heads, sequence length 1
  Head          (Sec. 3.4): MLP 64 -> 1024 -> 512 -> 128 -> 15
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import CNNwithSE, SEResNet, Classifier


# Define a CrossAttention module
class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, output_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads

        self.query_proj = nn.Linear(query_dim, output_dim)
        self.key_proj = nn.Linear(key_dim, output_dim)
        self.value_proj = nn.Linear(value_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

    def forward(self, query, key, value):
        # query, key, value are typically (batch_size, sequence_length, feature_dim)

        B, N_q, D_q = query.shape
        _, N_kv, D_kv = key.shape # N_kv should be same as value's second dim

        # Project to output_dim and reshape for multi-head attention
        queries = self.query_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, N_q, head_dim)
        keys = self.key_proj(key).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)     # (B, num_heads, N_kv, head_dim)
        values = self.value_proj(value).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)   # (B, num_heads, N_kv, head_dim)

        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5) # (B, num_heads, N_q, N_kv)
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, values) # (B, num_heads, N_q, head_dim)

        # Concatenate heads and project back to original dimension
        attended_values = attended_values.transpose(1, 2).contiguous().view(B, N_q, self.num_heads * self.head_dim)
        output = self.out_proj(attended_values)
        return output

class LAG(nn.Module):
    def __init__(self):
        super(LAG, self).__init__()
        self.cnn_se_model = CNNwithSE()
        self.backbone = SEResNet()
        self.classifier = Classifier() 
        self.backbone_output_channels = 64 # From SEResNet (B, 64, 8, 8)

        LOCAL_FEATURE_CHANNELS = 256 # Example: if CNNwithSE outputs 64 channels per part, then 4*64=256

        # Cross-attention where local features query global features
        self.local_to_global_attention = CrossAttention(
            query_dim=LOCAL_FEATURE_CHANNELS, # Local features as query
            key_dim=self.backbone_output_channels,   # Global features as key
            value_dim=self.backbone_output_channels,  # Global features as value
            output_dim=self.backbone_output_channels # Output dimension of the attention module
        )

        # Cross-attention where global features query local features (optional, but often beneficial)
        self.global_to_local_attention = CrossAttention(
            query_dim=self.backbone_output_channels, # Global features as query
            key_dim=LOCAL_FEATURE_CHANNELS,   # Local features as key
            value_dim=LOCAL_FEATURE_CHANNELS,  # Local features as value
            output_dim=LOCAL_FEATURE_CHANNELS # Output dimension of the attention module
        )
        
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

        ensemble_sub_module = torch.cat(sub_module_outputs, dim=1) # (B, C_ensemble, H_out, W_out)

        resized_image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        backbone_output = self.backbone(resized_image) # (B, 64, 8, 8)

        # --- Prepare features for Cross-Attention ---
        # We need to transform the feature maps (B, C, H, W) into sequences (B, N, D) for attention.
        # Option 1: Global Average Pooling to get a single vector per feature map.
        #           This results in (B, C) which can be unsqueezed to (B, 1, C).
        global_features_pooled = F.adaptive_avg_pool2d(backbone_output, (1, 1)).squeeze(-1).squeeze(-1) # (B, 64)
        local_features_pooled = F.adaptive_avg_pool2d(ensemble_sub_module, (1, 1)).squeeze(-1).squeeze(-1) # (B, LOCAL_FEATURE_CHANNELS)

        # Unsqueeze to add a sequence dimension (N=1) for the attention module
        global_features_seq = global_features_pooled.unsqueeze(1) # (B, 1, 64)
        local_features_seq = local_features_pooled.unsqueeze(1)   # (B, 1, LOCAL_FEATURE_CHANNELS)


        attended_local_from_global = self.local_to_global_attention(
            query=local_features_seq,
            key=global_features_seq,
            value=global_features_seq
        ) 
        # attended_global_from_local = self.global_to_local_attention(
        #     query=global_features_seq,
        #     key=local_features_seq,
        #     value=local_features_seq
        # ) 
        attended_local_from_global = attended_local_from_global.squeeze(1) # (B, self.backbone_output_channels)
        # attended_global_from_local = attended_global_from_local.squeeze(1) # (B, LOCAL_FEATURE_CHANNELS)

        # print(attended_local_from_global.shape) (64,64)
        # superimposed_features = torch.cat(
        #     (attended_local_from_global, attended_global_from_local),
        #     dim=1
        # )
        # print(superimposed_features.shape)
        # lag_output = self.classifier(superimposed_features) # (B, 1)
        lag_output = self.classifier(attended_local_from_global)
        return lag_output

# Paper name for the model.
CGLE = LAG
