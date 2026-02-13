"""
visualize_lidar.py — View a .bin LiDAR point cloud interactively.

Usage:
    python scripts/visualize_lidar.py data/carla/raw/clear/lidar/000000.bin
    python scripts/visualize_lidar.py data/carla/raw/rain/lidar/000050.bin --top_view
"""

import argparse
import sys

import numpy as np
import open3d as o3d


def load_bin(path: str) -> np.ndarray:
    """Load KITTI-style .bin file → (N, 4) array [x, y, z, intensity]."""
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points


def visualize_3d(points: np.ndarray):
    """Open an interactive 3D window (rotate with mouse)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # Color by intensity (column 4) — brighter = higher intensity
    intensity = points[:, 3]
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = intensity_norm        # red channel
    colors[:, 1] = 1.0 - intensity_norm  # green channel (inverted)
    colors[:, 2] = 0.3                   # slight blue tint
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Points: {len(points):,}")
    print(f"X range: [{points[:,0].min():.1f}, {points[:,0].max():.1f}]")
    print(f"Y range: [{points[:,1].min():.1f}, {points[:,1].max():.1f}]")
    print(f"Z range: [{points[:,2].min():.1f}, {points[:,2].max():.1f}]")
    print(f"Intensity range: [{intensity.min():.3f}, {intensity.max():.3f}]")
    print("\nControls: Left-click drag = rotate, Scroll = zoom, Middle-click = pan")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="LiDAR Point Cloud Viewer",
        width=1280,
        height=720,
    )


def visualize_top_view(points: np.ndarray):
    """Show a 2D bird's-eye view using matplotlib (no 3D interaction needed)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scatter = ax.scatter(
        points[:, 0], points[:, 1],
        c=points[:, 3],       # color by intensity
        cmap="viridis",
        s=0.3,
        alpha=0.7,
    )
    ax.set_xlabel("X (forward)")
    ax.set_ylabel("Y (left)")
    ax.set_title(f"Bird's-Eye View  —  {len(points):,} points")
    ax.set_aspect("equal")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    plt.colorbar(scatter, label="Intensity")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a LiDAR .bin file.")
    parser.add_argument("file", help="Path to the .bin file.")
    parser.add_argument("--top_view", action="store_true",
                        help="Show 2D bird's-eye view (matplotlib) instead of 3D.")
    args = parser.parse_args()

    points = load_bin(args.file)

    if args.top_view:
        visualize_top_view(points)
    else:
        visualize_3d(points)


if __name__ == "__main__":
    main()
