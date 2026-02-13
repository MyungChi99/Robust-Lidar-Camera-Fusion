"""
dataset.py — CARLA LiDAR-Camera Fusion Dataset

Loads synchronized LiDAR (.bin), RGB image (.png), and calibration (.json)
files, then projects each LiDAR point onto the image plane to sample the
corresponding RGB pixel. The result is a per-point fused feature vector:

    [x, y, z, intensity, R, G, B]   (7-dim)

This fused representation is what the downstream GNN (EdgeConv) operates on.

Projection pipeline (per point):
    1. Load 4×4 extrinsic matrix  T_lidar2cam  from calib JSON
    2. Load 3×3 intrinsic matrix  K            from calib JSON
    3. Transform:  p_cam = T_lidar2cam @ [x, y, z, 1]^T
    4. Project:    [u, v, 1]^T = K @ p_cam[:3] / p_cam[2]
    5. Sample:     rgb = image[v, u]
    6. Concatenate: feature = [x, y, z, intensity, r, g, b]
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CARLAFusionDataset(Dataset):
    """
    Dataset that loads CARLA sensor data and produces fused point features.

    Directory structure expected:
        data_root/
        ├── image/   000000.png, 000001.png, ...
        ├── lidar/   000000.bin, 000001.bin, ...
        └── calib/   000000.json, 000001.json, ...

    Each __getitem__ returns:
        points   : (N, 3)  float32  — XYZ coordinates (for k-NN graph)
        features : (N, 7)  float32  — [x, y, z, intensity, R, G, B]
        mask     : (N,)    bool     — True for points that project onto the image
    """

    def __init__(
        self,
        data_root: str,
        max_points: Optional[int] = None,
        normalize_rgb: bool = True,
    ):
        """
        Args:
            data_root:     Path to e.g. data/carla/raw/clear/
            max_points:    If set, randomly subsample to this many points
                           (keeps GPU memory manageable).
            normalize_rgb: If True, scale RGB from [0,255] to [0,1].
        """
        self.data_root = Path(data_root)
        self.max_points = max_points
        self.normalize_rgb = normalize_rgb

        self.image_dir = self.data_root / "image"
        self.lidar_dir = self.data_root / "lidar"
        self.calib_dir = self.data_root / "calib"

        # Collect frame IDs by listing .bin files (the primary modality)
        self.frame_ids = sorted(
            f.stem for f in self.lidar_dir.glob("*.bin")
        )

        if len(self.frame_ids) == 0:
            raise FileNotFoundError(
                f"No .bin files found in {self.lidar_dir}"
            )

        print(f"[CARLAFusionDataset] Loaded {len(self.frame_ids)} frames "
              f"from {self.data_root}")

    def __len__(self) -> int:
        return len(self.frame_ids)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_lidar(self, frame_id: str) -> np.ndarray:
        """Load .bin → (N, 4) float32 [x, y, z, intensity]."""
        path = self.lidar_dir / f"{frame_id}.bin"
        return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

    def _load_image(self, frame_id: str) -> np.ndarray:
        """Load .png → (H, W, 3) uint8 RGB array."""
        path = self.image_dir / f"{frame_id}.png"
        # Use open3d or PIL — avoid heavy cv2 dependency
        from PIL import Image
        img = Image.open(str(path)).convert("RGB")
        return np.array(img)

    def _load_calib(self, frame_id: str) -> Dict:
        """Load calibration JSON → dict with 'intrinsic' and 'extrinsic_lidar2cam'."""
        path = self.calib_dir / f"{frame_id}.json"
        with open(path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    @staticmethod
    def project_lidar_to_image(
        points_xyz: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D LiDAR points onto the 2D image plane.

        Args:
            points_xyz: (N, 3) points in LiDAR frame
            intrinsic:  (3, 3) camera intrinsic matrix
            extrinsic:  (4, 4) LiDAR-to-camera transform
            img_h, img_w: image dimensions

        Returns:
            uv:   (N, 2) int pixel coordinates [u, v]
            mask: (N,)   bool — True if the point projects inside the image
                          AND is in front of the camera (z > 0)
        """
        N = points_xyz.shape[0]

        # Homogeneous coordinates: (N, 4)
        ones = np.ones((N, 1), dtype=np.float32)
        pts_hom = np.hstack([points_xyz, ones])

        # LiDAR → Camera frame (UE4 coords): (4, 4) @ (4, N) → (4, N)
        pts_cam_ue4 = (extrinsic @ pts_hom.T).T  # (N, 4)

        # Convert UE4 (X=forward, Y=right, Z=up) → standard camera
        # (X=right, Y=down, Z=forward/depth):
        #   cam_x =  ue4_y
        #   cam_y = -ue4_z
        #   cam_z =  ue4_x
        pts_cam = np.stack([
            pts_cam_ue4[:, 1],
           -pts_cam_ue4[:, 2],
            pts_cam_ue4[:, 0],
        ], axis=1)  # (N, 3)

        # Depth in camera frame (z-axis = forward)
        depth = pts_cam[:, 2]

        # Project to image plane: (3, 3) @ (3, N) → (3, N)
        pts_2d = (intrinsic @ pts_cam.T).T  # (N, 3)

        # Normalize by depth → pixel coordinates
        # Avoid division by zero for points behind the camera
        safe_depth = np.where(depth > 0, depth, 1.0)
        u = pts_2d[:, 0] / safe_depth
        v = pts_2d[:, 1] / safe_depth

        # Round to integer pixel indices
        u = np.round(u).astype(np.int32)
        v = np.round(v).astype(np.int32)

        # Valid mask: in front of camera AND within image bounds
        mask = (
            (depth > 0)
            & (u >= 0) & (u < img_w)
            & (v >= 0) & (v < img_h)
        )

        uv = np.stack([u, v], axis=1)  # (N, 2)
        return uv, mask

    # ------------------------------------------------------------------
    # Main __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        frame_id = self.frame_ids[idx]

        # --- Load raw data ------------------------------------------------
        lidar = self._load_lidar(frame_id)          # (N, 4)
        image = self._load_image(frame_id)           # (H, W, 3) uint8
        calib = self._load_calib(frame_id)

        intrinsic = np.array(calib["intrinsic"], dtype=np.float64)        # (3, 3)
        extrinsic = np.array(calib["extrinsic_lidar2cam"], dtype=np.float64)  # (4, 4)

        img_h, img_w = image.shape[:2]

        points_xyz = lidar[:, :3]       # (N, 3)
        intensity  = lidar[:, 3:4]      # (N, 1)

        # --- Project LiDAR → image plane ---------------------------------
        uv, proj_mask = self.project_lidar_to_image(
            points_xyz, intrinsic, extrinsic, img_h, img_w
        )

        # --- Sample RGB for each valid point ------------------------------
        rgb = np.zeros((len(points_xyz), 3), dtype=np.float32)
        rgb[proj_mask] = image[uv[proj_mask, 1], uv[proj_mask, 0]].astype(np.float32)

        if self.normalize_rgb:
            rgb /= 255.0

        # --- Fuse: [x, y, z, intensity, R, G, B] -------------------------
        features = np.hstack([points_xyz, intensity, rgb])  # (N, 7)

        # --- Optional subsampling ----------------------------------------
        if self.max_points is not None and len(features) > self.max_points:
            indices = np.random.choice(len(features), self.max_points, replace=False)
            features  = features[indices]
            points_xyz = points_xyz[indices]
            proj_mask  = proj_mask[indices]

        return {
            "points":   torch.from_numpy(points_xyz).float(),   # (N, 3)
            "features": torch.from_numpy(features).float(),     # (N, 7)
            "mask":     torch.from_numpy(proj_mask),             # (N,)
            "frame_id": frame_id,
        }


# ---------------------------------------------------------------------------
# Convenience: collate function for variable-size point clouds
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict[str, list]:
    """
    Custom collate — keeps each sample as a separate entry because point
    clouds have variable N.  The GNN model will handle batching via
    torch_geometric.data.Batch later.
    """
    return {
        "points":   [sample["points"]   for sample in batch],
        "features": [sample["features"] for sample in batch],
        "mask":     [sample["mask"]     for sample in batch],
        "frame_id": [sample["frame_id"] for sample in batch],
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="e.g. data/carla/raw/clear")
    parser.add_argument("--max_points", type=int, default=16384)
    args = parser.parse_args()

    ds = CARLAFusionDataset(args.data_root, max_points=args.max_points)
    sample = ds[0]

    print(f"\nSample 0 (frame {sample['frame_id']}):")
    print(f"  points   : {sample['points'].shape}")
    print(f"  features : {sample['features'].shape}")
    print(f"  mask     : {sample['mask'].sum().item()}/{sample['mask'].shape[0]} "
          f"projected onto image")
    print(f"  feature cols: [x, y, z, intensity, R, G, B]")
    print(f"  first point : {sample['features'][0].tolist()}")
