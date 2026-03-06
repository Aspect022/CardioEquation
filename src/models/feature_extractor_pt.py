"""
PyTorch Feature Extractor (1D ResNet-18)
=========================================
Extracts a 512-dimensional patient identity vector from ECG segments.
Supports contrastive pre-training via an optional projection head.

Input:  (B, 1, 2500) — 5 seconds of ECG at 500Hz
Output: (B, 512) — patient identity embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """1D Residual Block with optional downsampling."""

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Downsample path for residual connection
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class FeatureExtractorPT(nn.Module):
    """
    1D ResNet-18 Feature Extractor for ECG patient identity.

    Architecture:
        Conv1D(1→64, k=15, s=2) → BN → ReLU → MaxPool
        → ResBlock(64, ×2) → ResBlock(128, ×2, stride=2)
        → ResBlock(256, ×2, stride=2) → ResBlock(512, ×2, stride=2)
        → GlobalAvgPool → (B, 512)
    """

    def __init__(self, in_channels=1, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim

        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Initialize weights
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResBlock1D(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, 1, T) — single-channel ECG signal
        Returns:
            z: (B, 512) — identity embedding
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)  # (B, 512, 1)
        x = x.squeeze(-1)        # (B, 512)
        return x


class ContrastiveFeatureExtractor(nn.Module):
    """
    Wraps FeatureExtractorPT with a projection head for SimCLR-style
    contrastive pre-training. The projection head is discarded after
    pre-training — only the encoder is used for downstream conditioning.
    """

    def __init__(self, in_channels=1, embed_dim=512, proj_dim=256):
        super().__init__()
        self.encoder = FeatureExtractorPT(in_channels, embed_dim)

        # 3-layer projection head (discarded after pre-training)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x, return_projection=True):
        """
        Args:
            x: (B, 1, T)
            return_projection: if True, returns L2-normalized projection (for contrastive loss)
                               if False, returns raw 512-dim embedding (for diffusion conditioning)
        """
        z = self.encoder(x)  # (B, 512)
        if return_projection:
            p = self.projector(z)  # (B, proj_dim)
            return F.normalize(p, dim=-1)
        return z


if __name__ == "__main__":
    # Quick verification
    fe = FeatureExtractorPT()
    x = torch.randn(4, 1, 2500)
    z = fe(x)
    params = sum(p.numel() for p in fe.parameters()) / 1e6
    print(f"FeatureExtractorPT: {params:.1f}M params")
    print(f"Input:  {x.shape}")
    print(f"Output: {z.shape}")
    assert z.shape == (4, 512)
    print("✅ FeatureExtractor verified!")

    # Contrastive wrapper
    cfe = ContrastiveFeatureExtractor()
    proj = cfe(x, return_projection=True)
    embed = cfe(x, return_projection=False)
    print(f"Projection: {proj.shape}, Embedding: {embed.shape}")
    print("✅ ContrastiveFeatureExtractor verified!")
