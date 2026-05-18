# FR3 Sonopet ROS 2 Workspace

Skeleton ROS 2 Jazzy overlay workspace for the Franka Research 3 Sonopet
experiment pipeline.

The project intentionally delegates hardware ownership to upstream packages:

- Franka robot control and MoveIt integration: `frankarobotics/franka_ros2`
- RealSense D405 camera drivers: `realsenseai/realsense-ros`
- Recording transport: `rosbag2`

Our code owns only experiment-specific orchestration, raster planning policy,
motion request policy, microphone publishing, and run manifests.

## Ubuntu 24.04 ROS 2 Jazzy Setup

Ubuntu 24.04 with ROS 2 Jazzy is the supported development and runtime system for
this workspace. macOS can be used for editing and pure-Python checks, but Franka,
RealSense, MoveIt, rosbag2, and full launch testing belong on Ubuntu.

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

Install ROS package dependencies and build:

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

The imported upstream repositories are intentionally ignored by Git:

```text
ros2_ws/src/franka_ros2/
ros2_ws/src/realsense-ros/
```

`third_party.repos` remains the version-controlled dependency manifest.

## Light Python Checks

Install the small Python-only developer helpers when needed:

```bash
python3 -m pip install -r requirements-dev.txt
```

Run the pure-Python tests without hardware:

```bash
pytest \
  ros2_ws/src/fr3_sonopet_microphone/test/test_audio_format.py \
  ros2_ws/src/fr3_sonopet_supervisor/test/test_operator_gates.py \
  ros2_ws/src/fr3_sonopet_motion/test/test_segment_policy.py \
  ros2_ws/src/fr3_sonopet_recording/test/test_topic_policy.py \
  ros2_ws/src/fr3_sonopet_trajectory/test/test_raster_pattern.py \
  ros2_ws/src/fr3_sonopet_trajectory/test/test_surface_geometry.py
```

<!-- ## Local macOS Convenience

macOS-specific Conda environments, IDE settings, and terminal hooks are local
developer conveniences and are ignored by Git. Keep them out of shared setup
unless they are explicitly made platform-neutral. -->
