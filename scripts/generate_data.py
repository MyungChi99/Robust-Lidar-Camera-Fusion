"""
generate_data.py — CARLA Synchronized Sensor Data Collection

Collects time-aligned RGB + LiDAR frames in CARLA's synchronous mode,
along with per-frame calibration matrices (intrinsic & extrinsic).

Synchronization strategy
------------------------
CARLA's default *asynchronous* mode advances the simulation clock in real-time,
so sensors may return data from different simulation ticks.  Enabling
**synchronous mode** (`synchronous_mode=True`, `fixed_delta_seconds=0.1`)
freezes the simulation until the client explicitly calls `world.tick()`.
This guarantees that every sensor callback fires for the *same* simulation
frame before the world advances, giving us perfectly aligned image/LiDAR pairs.

Usage
-----
    python scripts/generate_data.py --weather clear --frames 1000
    python scripts/generate_data.py --weather rain  --frames 500
"""

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from queue import Empty, Queue

import carla
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_intrinsic_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    """Compute a 3x3 pinhole camera intrinsic matrix from CARLA parameters."""
    fov_rad = math.radians(fov_deg)
    fx = width / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def carla_transform_to_matrix(transform: carla.Transform) -> np.ndarray:
    """Convert a carla.Transform to a 4x4 homogeneous matrix (world frame)."""
    rotation = transform.rotation
    location = transform.location

    # Degrees → radians
    roll  = math.radians(rotation.roll)
    pitch = math.radians(rotation.pitch)
    yaw   = math.radians(rotation.yaw)

    # Rotation matrix from Euler angles (CARLA uses UE4 convention)
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    matrix = np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, location.x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, location.y],
        [sp,      -cp * sr,                cp * cr,                 location.z],
        [0.0,     0.0,                     0.0,                     1.0       ],
    ], dtype=np.float64)
    return matrix


def sensor_callback(data, queue: Queue):
    """Generic callback — push sensor data into a thread-safe queue."""
    queue.put(data)


def save_lidar_bin(point_cloud: carla.LidarMeasurement, path: str):
    """Save LiDAR point cloud as a flat binary file (N x 4 float32: x y z intensity)."""
    # raw_data is a flat buffer of float32 [x, y, z, intensity, ...]
    data = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
    data.tofile(path)


def save_image(image: carla.Image, path: str):
    """Save an RGB camera image as PNG."""
    image.save_to_disk(path)


# ---------------------------------------------------------------------------
# Weather presets
# ---------------------------------------------------------------------------

WEATHER_PRESETS = {
    "clear": carla.WeatherParameters.ClearNoon,
    "rain": carla.WeatherParameters(
        cloudiness=80.0,
        precipitation=100.0,
        precipitation_deposits=60.0,
        wind_intensity=50.0,
        sun_altitude_angle=45.0,
        fog_density=20.0,
        fog_distance=0.0,
        wetness=100.0,
    ),
}

# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CARLA synchronous sensor data collection."
    )
    parser.add_argument(
        "--weather", type=str, default="clear", choices=["clear", "rain"],
        help="Weather preset to apply (default: clear).",
    )
    parser.add_argument(
        "--frames", type=int, default=1000,
        help="Number of synchronised frames to capture (default: 1000).",
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="CARLA server hostname (default: localhost).",
    )
    parser.add_argument(
        "--port", type=int, default=2000,
        help="CARLA server port (default: 2000).",
    )
    parser.add_argument(
        "--tm-port", type=int, default=8000,
        help="Traffic Manager port (default: 8000).",
    )
    parser.add_argument(
        "--num-vehicles", type=int, default=50,
        help="Number of NPC vehicles to spawn (default: 50).",
    )
    parser.add_argument(
        "--num-walkers", type=int, default=30,
        help="Number of NPC pedestrians to spawn (default: 30).",
    )
    args = parser.parse_args()

    # ----- Output directories ------------------------------------------------
    base_dir = Path("data/carla/raw") / args.weather
    img_dir   = base_dir / "image"
    lidar_dir = base_dir / "lidar"
    calib_dir = base_dir / "calib"
    for d in (img_dir, lidar_dir, calib_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ----- Connect to CARLA --------------------------------------------------
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()

    # Store original settings so we can restore them on exit
    original_settings = world.get_settings()

    # Keep track of actors we spawn so we can clean up reliably
    actor_list = []

    try:
        # ----- Enable synchronous mode --------------------------------------
        # In synchronous mode the server waits for a client tick() call before
        # advancing the simulation.  This guarantees that every sensor produces
        # data for the *same* simulation frame.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1   # 10 Hz simulation
        world.apply_settings(settings)

        # Traffic Manager must also run in synchronous mode so that NPC
        # vehicles respect the tick-based progression.
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_synchronous_mode(True)

        # ----- Weather -------------------------------------------------------
        world.set_weather(WEATHER_PRESETS[args.weather])

        # ----- Spawn ego vehicle ---------------------------------------------
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]

        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("ERROR: No spawn points available on this map.")
            sys.exit(1)

        spawn_point = np.random.choice(spawn_points)
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(ego_vehicle)
        print(f"Spawned ego vehicle: {ego_vehicle.type_id} at {spawn_point.location}")

        # Enable autopilot through the Traffic Manager
        ego_vehicle.set_autopilot(True, args.tm_port)

        # ----- Spawn NPC vehicles --------------------------------------------
        vehicle_bps = blueprint_library.filter("vehicle.*")
        # Remove bikes/motorcycles for cleaner traffic
        vehicle_bps = [bp for bp in vehicle_bps
                       if int(bp.get_attribute("number_of_wheels")) >= 4]

        available_spawns = [sp for sp in spawn_points if sp != spawn_point]
        np.random.shuffle(available_spawns)

        num_to_spawn = min(args.num_vehicles, len(available_spawns))
        npc_vehicles = []
        for i in range(num_to_spawn):
            bp = np.random.choice(vehicle_bps)
            # Randomize color if available
            if bp.has_attribute("color"):
                color = np.random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            npc = world.try_spawn_actor(bp, available_spawns[i])
            if npc is not None:
                npc.set_autopilot(True, args.tm_port)
                npc_vehicles.append(npc)
                actor_list.append(npc)

        print(f"Spawned {len(npc_vehicles)} NPC vehicles")

        # ----- Spawn NPC pedestrians -----------------------------------------
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        walker_controller_bp = blueprint_library.find("controller.ai.walker")

        walkers = []
        walker_controllers = []
        for _ in range(args.num_walkers):
            # Find a random point on the sidewalk
            spawn_loc = world.get_random_location_from_navigation()
            if spawn_loc is None:
                continue
            wp_transform = carla.Transform(location=spawn_loc)
            bp = np.random.choice(walker_bps)
            # Make some walkers run
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = world.try_spawn_actor(bp, wp_transform)
            if walker is None:
                continue
            walkers.append(walker)
            actor_list.append(walker)

        # Must tick once before spawning walker controllers
        world.tick()

        for walker in walkers:
            controller = world.spawn_actor(walker_controller_bp,
                                           carla.Transform(), attach_to=walker)
            walker_controllers.append(controller)
            actor_list.append(controller)

        # Start walking — each pedestrian picks a random destination
        for controller in walker_controllers:
            controller.start()
            dest = world.get_random_location_from_navigation()
            if dest is not None:
                controller.go_to_location(dest)
            controller.set_max_speed(1.0 + np.random.random() * 1.5)  # 1.0–2.5 m/s

        print(f"Spawned {len(walkers)} NPC pedestrians")

        # Let the scene settle for a few ticks (vehicles start moving)
        for _ in range(30):
            world.tick()
        print("Scene settled. Starting data collection...")

        # ----- Attach RGB camera ---------------------------------------------
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "800")
        cam_bp.set_attribute("image_size_y", "600")
        cam_bp.set_attribute("fov", "90")

        # Mount position: slightly above and forward on the roof
        cam_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=0.0),
        )
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle)
        actor_list.append(camera)

        # ----- Attach LiDAR sensor -------------------------------------------
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "100.0")
        lidar_bp.set_attribute("rotation_frequency", "10")
        lidar_bp.set_attribute("channels", "128")
        lidar_bp.set_attribute("points_per_second", "2400000")
        lidar_bp.set_attribute("upper_fov", "10.0")
        lidar_bp.set_attribute("lower_fov", "-30.0")
        # Noise injection — simulates realistic sensor imperfections
        lidar_bp.set_attribute("noise_stddev", "0.01")
        lidar_bp.set_attribute("dropoff_general_rate", "0.05")

        lidar_transform = carla.Transform(
            carla.Location(x=0.0, z=2.5),
            carla.Rotation(pitch=0.0),
        )
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        actor_list.append(lidar_sensor)

        # ----- Sensor queues (thread-safe) -----------------------------------
        # Each sensor pushes its data into its own queue.  After every tick()
        # we drain both queues, which guarantees frame alignment.
        cam_queue   = Queue()
        lidar_queue = Queue()

        camera.listen(lambda data: sensor_callback(data, cam_queue))
        lidar_sensor.listen(lambda data: sensor_callback(data, lidar_queue))

        # ----- Pre-computed intrinsic matrix ---------------------------------
        IMAGE_W, IMAGE_H, FOV = 800, 600, 90.0
        intrinsic = build_intrinsic_matrix(IMAGE_W, IMAGE_H, FOV)

        # ----- Collection loop -----------------------------------------------
        print(f"Collecting {args.frames} synchronised frames  "
              f"[weather={args.weather}] ...")

        for frame_idx in range(args.frames):
            # Advance the simulation by one tick.  All sensors will produce
            # data for this tick before returning.
            world.tick()

            # Retrieve sensor data (block until available, with timeout)
            try:
                image_data = cam_queue.get(timeout=5.0)
                lidar_data = lidar_queue.get(timeout=5.0)
            except Empty:
                print(f"WARNING: Sensor timeout at frame {frame_idx}, skipping.")
                continue

            frame_tag = f"{frame_idx:06d}"

            # --- Save image --------------------------------------------------
            save_image(image_data, str(img_dir / f"{frame_tag}.png"))

            # --- Save LiDAR (.bin, KITTI-style) ------------------------------
            save_lidar_bin(lidar_data, str(lidar_dir / f"{frame_tag}.bin"))

            # --- Save calibration --------------------------------------------
            # Extrinsic: LiDAR frame → Camera frame
            # We compute  T_cam_world  and  T_lidar_world  from CARLA, then
            #   T_lidar2cam = T_cam_world^{-1}  @  T_lidar_world
            cam_world   = carla_transform_to_matrix(camera.get_transform())
            lidar_world = carla_transform_to_matrix(lidar_sensor.get_transform())
            lidar2cam   = np.linalg.inv(cam_world) @ lidar_world

            calib = {
                "frame": frame_idx,
                "intrinsic": intrinsic.tolist(),
                "extrinsic_lidar2cam": lidar2cam.tolist(),
                "cam_world": cam_world.tolist(),
                "lidar_world": lidar_world.tolist(),
            }

            calib_path = calib_dir / f"{frame_tag}.json"
            with open(calib_path, "w") as f:
                json.dump(calib, f, indent=2)

            if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
                pts = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
                print(f"  frame {frame_tag}  |  "
                      f"image {IMAGE_W}x{IMAGE_H}  |  "
                      f"lidar pts {pts.shape[0]:,}")

        print(f"\nDone — {args.frames} frames saved to {base_dir.resolve()}")

    finally:
        # ----- Cleanup -------------------------------------------------------
        # Restore original world settings (disables synchronous mode) and
        # destroy all actors we spawned.
        print("Cleaning up ...")
        world.apply_settings(original_settings)
        for actor in reversed(actor_list):
            if actor is not None and actor.is_alive:
                actor.destroy()
        print("All actors destroyed.  Exiting.")


if __name__ == "__main__":
    main()
