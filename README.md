# FR3 Sonopet ROS 2 Workspace

Skeleton ROS 2 Jazzy overlay workspace for the Franka Research 3 Sonopet
experiment pipeline.

The project intentionally delegates hardware ownership to upstream packages:

- Franka robot control and MoveIt integration: `frankarobotics/franka_ros2`
- RealSense D405 camera drivers: `realsenseai/realsense-ros`
- Recording transport: `rosbag2`

Our code owns only experiment-specific orchestration, raster planning policy,
motion request policy, microphone publishing, and run manifests.

## Bootstrap

```bash
cd ros2_ws
vcs import src < third_party.repos
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
```

