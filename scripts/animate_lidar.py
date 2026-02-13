"""
animate_lidar.py — Play LiDAR .bin files as a sequential animation.

Usage:
    python scripts/animate_lidar.py data/carla/raw/clear/lidar/
    python scripts/animate_lidar.py data/carla/raw/rain/lidar/ --start 0 --end 500 --fps 10
    python scripts/animate_lidar.py data/carla/raw/clear/lidar/ --mode bev
"""

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def load_bin(path: str) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def intensity_to_colors(points: np.ndarray) -> np.ndarray:
    intensity = points[:, 3]
    norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = norm
    colors[:, 1] = 1.0 - norm
    colors[:, 2] = 0.3
    return colors


def animate_3d(files: list, fps: float):
    """Play frames in an Open3D window, updating the point cloud each frame."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="LiDAR Animation", width=1280, height=720)

    # Load first frame to initialize geometry
    pts = load_bin(str(files[0]))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(intensity_to_colors(pts))
    vis.add_geometry(pcd)

    # Set initial viewpoint: bird's-eye slightly tilted
    # CARLA LiDAR: X=forward, Y=right, Z=up
    vc = vis.get_view_control()
    vc.set_front([0.0, 0.0, 1.0])    # look from above (Z-up)
    vc.set_lookat([0.0, 0.0, 0.0])   # center on ego vehicle
    vc.set_up([-1.0, 0.0, 0.0])      # X=forward as "up" on screen
    vc.set_zoom(0.3)

    # Fit the view to the actual point cloud extent
    vis.reset_view_point(True)

    render_opt = vis.get_render_option()
    render_opt.point_size = 2.0
    render_opt.background_color = np.array([0.05, 0.05, 0.1])

    delay = 1.0 / fps
    print(f"Playing {len(files)} frames at {fps} FPS  (close window to stop)")
    print("Controls: Left-drag = rotate, Scroll = zoom, Middle-drag = pan")

    for i, f in enumerate(files):
        pts = load_bin(str(f))

        # Update point cloud in-place (no remove/add overhead)
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(intensity_to_colors(pts))

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        if i % 50 == 0:
            print(f"  frame {f.stem}  |  {pts.shape[0]:,} pts")

        # Check if the user closed the window
        if not vis.poll_events():
            break

        time.sleep(delay)

    vis.destroy_window()
    print("Done.")


def animate_bev(files: list, fps: float):
    """Bird's-eye view animation using matplotlib."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()  # interactive mode

    scatter = None

    for i, f in enumerate(files):
        pts = load_bin(str(f))
        ax.clear()
        ax.scatter(pts[:, 0], pts[:, 1], c=pts[:, 3],
                   cmap="viridis", s=0.2, alpha=0.6)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.set_aspect("equal")
        ax.set_facecolor("black")
        ax.set_title(f"Frame {f.stem}  |  {pts.shape[0]:,} pts", color="white")
        ax.tick_params(colors="gray")
        fig.patch.set_facecolor("black")

        plt.pause(1.0 / fps)

        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Animate LiDAR .bin sequence.")
    parser.add_argument("lidar_dir", help="Directory containing .bin files.")
    parser.add_argument("--start", type=int, default=0,
                        help="First frame index (default: 0).")
    parser.add_argument("--end", type=int, default=500,
                        help="Last frame index inclusive (default: 500).")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Playback speed in frames per second (default: 10).")
    parser.add_argument("--mode", choices=["3d", "bev"], default="3d",
                        help="'3d' = interactive Open3D, 'bev' = bird's-eye matplotlib.")
    args = parser.parse_args()

    lidar_dir = Path(args.lidar_dir)
    files = sorted(lidar_dir.glob("*.bin"))

    # Filter to requested range
    files = [f for f in files
             if args.start <= int(f.stem) <= args.end]

    if not files:
        print(f"No .bin files found in {lidar_dir} for range [{args.start}, {args.end}]")
        return

    print(f"Found {len(files)} frames  [{files[0].stem} → {files[-1].stem}]")

    if args.mode == "3d":
        animate_3d(files, args.fps)
    else:
        animate_bev(files, args.fps)


if __name__ == "__main__":
    main()
