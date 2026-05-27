# FR3 Sonopet ROS 2 Workspace

Skeleton ROS 2 Jazzy overlay workspace for the Franka Research 3 Sonopet
experiment pipeline.

The project intentionally delegates hardware ownership to upstream packages:

- Franka robot control and MoveIt integration: `frankarobotics/franka_ros2`
- RealSense D405 camera drivers: `realsenseai/realsense-ros`

Our code owns only experiment-specific orchestration, raster planning policy,
motion request policy, microphone publishing, and practical experiment
artifacts.

## Ubuntu 24.04 ROS 2 Jazzy Setup

Ubuntu 24.04 with ROS 2 Jazzy is the supported development and runtime system for
this workspace. Franka, RealSense, MoveIt, and full launch testing belong on
Ubuntu.

Install ROS 2 Jazzy from the official ROS instructions first, then install the
workspace tooling:

```bash
./scripts/bootstrap_ubuntu24_ros_jazzy.sh
```

Load the ROS environment in each new terminal:

```bash
source scripts/source_ubuntu24_ros_jazzy.sh
```

Import third-party source dependencies locally:

```bash
cd ros2_ws
vcs import src < third_party.repos
```

Install ROS package dependencies where rosdep rules are available, then build the
hardware/vendor overlay. Some upstream Franka example packages reference optional
rosdep keys that are not published for Ubuntu Noble/Jazzy; the core FR3 Sonopet
overlay does not require those optional packages. Upstream vendor tests are
disabled here because this workspace consumes their runtime packages and launch
files, while vendor CI owns their exhaustive package tests:

```bash
rosdep install --from-paths src --ignore-src -r -y || true
# Review rosdep output and install any required system packages that resolved.
colcon build --symlink-install \
  --packages-up-to franka_fr3_moveit_config realsense2_camera \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_IGNORE_PREFIX_PATH=/usr/local
colcon build --symlink-install
```

The imported upstream repositories are intentionally ignored by Git:

```text
ros2_ws/src/franka_ros2/
ros2_ws/src/realsense-ros/
```

`third_party.repos` remains the version-controlled dependency manifest.

If the host has an older manually installed libfranka under `/usr/local`, source
`scripts/source_ubuntu24_ros_jazzy.sh` before building. The script keeps the
Jazzy package-managed Franka headers ahead of stale local headers without
modifying system files.

## Light Python Checks

Install the small Python-only developer helpers when needed:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

Run the pure-Python tests without hardware:

```bash
pytest \
  ros2_ws/src/fr3_sonopet_bringup/test \
  ros2_ws/src/fr3_sonopet_microphone/test \
  ros2_ws/src/fr3_sonopet_supervisor/test/test_operator_gates.py \
  ros2_ws/src/fr3_sonopet_motion/test/test_motion_runner_node.py \
  ros2_ws/src/fr3_sonopet_recording/test \
  ros2_ws/src/fr3_sonopet_trajectory/test/test_raster_pattern.py \
  ros2_ws/src/fr3_sonopet_trajectory/test/test_surface_geometry.py
```

## Sensor Recording Workflow

Launch the sensor-only workflow for the two D405 cameras, configured microphone,
and artifact recorder:

```bash
source scripts/source_ubuntu24_ros_jazzy.sh
ros2 launch fr3_sonopet_bringup sensors.launch.py
```

The default microphone policy is configured in
`ros2_ws/src/fr3_sonopet_bringup/config/microphone.yaml`. It requires a device
whose name contains `iMM-6C` or `imm6c` unless the config is changed.

The D405 nodes use the namespace/name layout `/RealSense_D405/in_hand` and
`/RealSense_D405/fixed`. Capture the in-hand D405 intrinsics once while the
camera is publishing, then use the fixed project CameraInfo topics for downstream
project code:

```bash
ros2 launch fr3_sonopet_bringup cameras.launch.py
ros2 launch fr3_sonopet_bringup capture_d405_intrinsics.launch.py
```

The capture writes `ros2_ws/src/fr3_sonopet_bringup/intrinsics/d405_intrinsics.yaml`.
The sensor launch republishes it on `/sonopet/d405_intrinsics/<role>/<stream>/camera_info`.

Continuous RGB topics remain active. Continuous point cloud output is disabled
by default; the recorder temporarily enables each camera point cloud publisher
only to save start and end PCD snapshots. Recording artifacts are written under
`ros2_ws/artifacts/experiments/<timestamp>/`, for example
`ros2_ws/artifacts/experiments/20260527T131514/`:

```text
manifest.json
camera_in_hand/rgb.avi
camera_in_hand/rgb_timestamps.csv
camera_in_hand/pointcloud_start.pcd
camera_in_hand/pointcloud_end.pcd
camera_fixed/rgb.avi
camera_fixed/rgb_timestamps.csv
camera_fixed/pointcloud_start.pcd
camera_fixed/pointcloud_end.pcd
audio/audio.wav
audio/audio_meta.json
```

Use standard ROS tools for inspection:

```bash
rqt_graph
rqt_image_view
ros2 topic hz /microphone/audio
ros2 launch fr3_sonopet_bringup sensors.launch.py rviz:=true
```
