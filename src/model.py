"""
model.py — EdgeConv GNN for LiDAR Noise Classification

Takes fused per-point features [x, y, z, intensity, R, G, B] from dataset.py,
builds a dynamic k-NN graph in feature space, and runs EdgeConv layers to
classify each point as:
    0 = clean (real object surface)
    1 = noise (weather-induced ghost point)

EdgeConv operation (per layer):
    x'_i = max_{j in N(i)}  MLP( x_j - x_i || x_i )

The graph is recomputed after each layer ("dynamic graph CNN") so that the
neighborhood evolves as features are updated.

Architecture:
    Input (N, 7)
     → EdgeConv1 (7 → 64)     + dynamic k-NN
     → EdgeConv2 (64 → 128)   + dynamic k-NN
     → EdgeConv3 (128 → 256)  + dynamic k-NN
     → Global + Local concat  (256+128+64 = 448)
     → MLP head (448 → 256 → 128 → 2)
     → Per-point logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch_geometric.data import Data, Batch


class EdgeConvBlock(nn.Module):
    """Single EdgeConv layer with BatchNorm and LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, k: int):
        super().__init__()
        # EdgeConv MLP takes concatenated (x_j - x_i || x_i) → 2 * in_channels
        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.2),
            ),
            k=k,
            aggr="max",
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.conv(x, batch)


class LiDARNoiseClassifier(nn.Module):
    """
    Dynamic Graph CNN for per-point binary classification.

    Args:
        in_channels:  Input feature dimension (default 7: x,y,z,i,R,G,B)
        k:            Number of nearest neighbors for k-NN graph
        num_classes:  Output classes (default 2: clean / noise)
    """

    def __init__(self, in_channels: int = 7, k: int = 20, num_classes: int = 2):
        super().__init__()
        self.k = k

        # --- EdgeConv backbone -------------------------------------------
        self.conv1 = EdgeConvBlock(in_channels, 64, k)
        self.conv2 = EdgeConvBlock(64, 128, k)
        self.conv3 = EdgeConvBlock(128, 256, k)

        # --- Classification head -----------------------------------------
        # Concatenate multi-scale features: 64 + 128 + 256 = 448
        self.head = nn.Sequential(
            nn.Linear(448, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (N_total, in_channels) — all points concatenated across batch
            batch: (N_total,) — batch index for each point (0, 0, ..., 1, 1, ...)

        Returns:
            logits: (N_total, num_classes) — per-point class logits
        """
        # EdgeConv layers — graph is rebuilt dynamically each layer
        x1 = self.conv1(x, batch)    # (N, 64)
        x2 = self.conv2(x1, batch)   # (N, 128)
        x3 = self.conv3(x2, batch)   # (N, 256)

        # Multi-scale feature concatenation
        x_cat = torch.cat([x1, x2, x3], dim=1)  # (N, 448)

        # Per-point classification
        logits = self.head(x_cat)  # (N, 2)
        return logits


# ---------------------------------------------------------------------------
# Helper: convert dataset.py output → torch_geometric Batch
# ---------------------------------------------------------------------------

def build_graph_batch(
    features_list: list,
    labels_list: list = None,
    device: str = "cuda",
) -> Batch:
    """
    Convert a list of per-sample feature tensors into a torch_geometric Batch.

    Args:
        features_list: list of (N_i, 7) tensors from dataset collate_fn
        labels_list:   list of (N_i,) label tensors (optional, for training)
        device:        target device

    Returns:
        torch_geometric.data.Batch with .x and .batch (and .y if labels given)
    """
    data_list = []
    for i, feat in enumerate(features_list):
        data = Data(x=feat.to(device))
        if labels_list is not None:
            data.y = labels_list[i].to(device)
        data_list.append(data)

    return Batch.from_data_list(data_list)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = LiDARNoiseClassifier(in_channels=7, k=20, num_classes=2).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")
    print(f"\nArchitecture:\n{model}")

    # Simulate a batch of 2 point clouds with different sizes
    fake_features = [
        torch.randn(4096, 7),   # sample 0: 4096 points
        torch.randn(3500, 7),   # sample 1: 3500 points
    ]
    batch = build_graph_batch(fake_features, device=device)
    print(f"\nInput:  x={batch.x.shape}  batch={batch.batch.shape}")

    logits = model(batch.x, batch.batch)
    print(f"Output: logits={logits.shape}")  # (7596, 2)

    preds = logits.argmax(dim=1)
    print(f"Predictions: {preds.shape}  unique={preds.unique().tolist()}")
