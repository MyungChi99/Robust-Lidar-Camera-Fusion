"""
visualize_fusion.py — Visual verification of the LiDAR-Camera fusion pipeline.

Generates 3 visualizations:
  1. Projection overlay : LiDAR points drawn on top of the RGB image
  2. Fused point cloud  : 3D point cloud colored by sampled RGB
  3. k-NN graph         : 3D view of EdgeConv graph edges between neighbors

Usage:
    python scripts/visualize_fusion.py data/carla/raw/clear --frame 0
    python scripts/visualize_fusion.py data/carla/raw/rain  --frame 100 --k 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Append project root so we can import src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import CARLAFusionDataset


def viz_projection_overlay(image: np.ndarray, uv: np.ndarray, mask: np.ndarray,
                           depth: np.ndarray, title: str = ""):
    """Draw projected LiDAR points on the RGB image, colored by depth."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    u_valid = uv[mask, 0]
    v_valid = uv[mask, 1]
    d_valid = depth[mask]

    scatter = ax.scatter(u_valid, v_valid, c=d_valid, cmap="jet",
                         s=1.0, alpha=0.7, vmin=0, vmax=50)
    plt.colorbar(scatter, ax=ax, label="Depth (m)", shrink=0.7)
    ax.set_title(f"LiDAR → Image Projection  |  {mask.sum():,} points  {title}")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def viz_fused_pointcloud(points: np.ndarray, rgb: np.ndarray, mask: np.ndarray):
    """Show 3D point cloud colored by sampled RGB (Open3D)."""
    import open3d as o3d

    # Points with valid projection get their true RGB color
    # Points without projection are shown in dark gray
    colors = np.full((len(points), 3), 0.2)  # dark gray default
    colors[mask] = rgb[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"\n[Fused Point Cloud]")
    print(f"  Total points:    {len(points):,}")
    print(f"  With RGB color:  {mask.sum():,}")
    print(f"  Without (gray):  {(~mask).sum():,}")
    print("  Controls: Left-drag=rotate, Scroll=zoom, Middle-drag=pan")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Fused Point Cloud (RGB from Camera)",
        width=1280, height=720,
    )


def viz_knn_graph(points: np.ndarray, rgb: np.ndarray, mask: np.ndarray,
                  k: int = 20, max_display_points: int = 2048):
    """Visualize the k-NN graph edges in 3D using Open3D."""
    import open3d as o3d
    from sklearn.neighbors import NearestNeighbors

    # Subsample for visualization (full graph is too dense to render)
    if len(points) > max_display_points:
        idx = np.random.choice(len(points), max_display_points, replace=False)
        points = points[idx]
        rgb = rgb[idx]
        mask = mask[idx]

    N = len(points)

    # Build k-NN graph
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(points)
    distances, indices = nn.kneighbors(points)

    # Collect edges (skip self-loop at index 0)
    edges = []
    for i in range(N):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            edges.append([i, j])
    edges = np.array(edges)

    # Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.full((N, 3), 0.3)
    colors[mask] = rgb[mask]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Line set for graph edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    # Color edges light cyan
    line_colors = np.full((len(edges), 3), [0.0, 0.8, 0.8])
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    print(f"\n[k-NN Graph]")
    print(f"  Points displayed: {N:,}")
    print(f"  k = {k}")
    print(f"  Edges:  {len(edges):,}")
    print(f"  Avg edge length: {distances[:, 1:].mean():.3f} m")
    print("  Cyan lines = graph edges connecting k nearest neighbors")

    o3d.visualization.draw_geometries(
        [pcd, line_set],
        window_name=f"k-NN Graph (k={k}, {N} points)",
        width=1280, height=720,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LiDAR-Camera fusion and k-NN graph."
    )
    parser.add_argument("data_root", help="e.g. data/carla/raw/clear")
    parser.add_argument("--frame", type=int, default=0, help="Frame index.")
    parser.add_argument("--k", type=int, default=20, help="k for k-NN graph.")
    parser.add_argument("--max_points", type=int, default=16384,
                        help="Subsample points (default 16384).")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    frame_id = f"{args.frame:06d}"

    # --- Load raw data ----------------------------------------------------
    lidar_path = data_root / "lidar" / f"{frame_id}.bin"
    image_path = data_root / "image" / f"{frame_id}.png"
    calib_path = data_root / "calib" / f"{frame_id}.json"

    for p in [lidar_path, image_path, calib_path]:
        if not p.exists():
            print(f"ERROR: {p} not found")
            sys.exit(1)

    points_raw = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
    image = np.array(Image.open(str(image_path)).convert("RGB"))
    with open(calib_path) as f:
        calib = json.load(f)

    intrinsic = np.array(calib["intrinsic"], dtype=np.float64)
    extrinsic = np.array(calib["extrinsic_lidar2cam"], dtype=np.float64)
    img_h, img_w = image.shape[:2]

    points_xyz = points_raw[:, :3]

    # --- Subsample --------------------------------------------------------
    if len(points_xyz) > args.max_points:
        idx = np.random.choice(len(points_xyz), args.max_points, replace=False)
        points_xyz = points_xyz[idx]
        points_raw = points_raw[idx]

    # --- Project ----------------------------------------------------------
    uv, mask = CARLAFusionDataset.project_lidar_to_image(
        points_xyz, intrinsic, extrinsic, img_h, img_w
    )

    # Compute depth for coloring the overlay (must match UE4→camera conversion)
    ones = np.ones((len(points_xyz), 1), dtype=np.float32)
    pts_hom = np.hstack([points_xyz, ones])
    pts_cam_ue4 = (extrinsic @ pts_hom.T).T
    # UE4 X-axis (forward) = camera Z-axis (depth)
    depth = pts_cam_ue4[:, 0]

    # Sample RGB
    rgb = np.zeros((len(points_xyz), 3), dtype=np.float32)
    rgb[mask] = image[uv[mask, 1], uv[mask, 0]].astype(np.float32) / 255.0

    print(f"Frame {frame_id}  |  {len(points_xyz):,} points  |  "
          f"{mask.sum():,} projected onto {img_w}x{img_h} image")

    # --- Viz 1: Projection overlay ----------------------------------------
    print("\n[1/3] Projection overlay — close the window to continue...")
    viz_projection_overlay(image, uv, mask, depth, title=f"frame {frame_id}")

    # --- Viz 2: Fused point cloud -----------------------------------------
    print("[2/3] Fused 3D point cloud — close the window to continue...")
    viz_fused_pointcloud(points_xyz, rgb, mask)

    # --- Viz 3: k-NN graph ------------------------------------------------
    print(f"[3/3] k-NN graph (k={args.k}) — close the window to finish.")
    viz_knn_graph(points_xyz, rgb, mask, k=args.k, max_display_points=2048)

    print("\nAll visualizations complete.")


if __name__ == "__main__":
    main()
