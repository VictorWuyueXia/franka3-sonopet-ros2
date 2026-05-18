Our new repo is for the experiment of a Franka Research 3 robot arm using sonopet device as a end effector. But we do not have sonope related code in our plan yet, so everything is around the franka robot.


Take maximum advantage of repo:https://github.com/realsenseai/realsense-ros and https://github.com/frankarobotics/franka_ros2/tree/jazzy to plan our code structure for the project, to minimize our own code effort.

The empirical best practice for our own code is to generally keep every code file 100-400lines, with each function 20-100 lines. Atomize the workflow steps, logics, and nodes, but do not over-layer them. Write everything in minimum realization. 

Plan the structure as packages in a ros workspace. Write in python by defualt, but C++ when it suits well.


Here is my draft design of the code structure:
config/
├── robot.yaml
├── cameras.yaml
├── raster.yaml
├── motion.yaml
├── recording_topics.yaml

ros2_ws/
├── src/
    ├── bringup/
        ├── experiment.launch.py        # full real experiment.
        ├── fake_experiment.launch.py   # fake Franka hardware + recorded/sample cloud.
        ├── franka.launch.py            # includes upstream Franka bringup or MoveIt launch.
        ├── cameras.launch.py           # two D405 nodes using realsense2_camera/rs_launch.py.
        ├── microphone.launch.py        # microphone node
    ├── trajectory/
        ├── raster_planner_node.py      # ROS node, two point clouds and robot state in -> raster plan out
        ├── cloud_io.py                 # PointCloud2, crop points>500mm away, then crop 20% outlyers
        ├── stitching.py                # Open3D stitcing with point to plane ICP then colored ICP, holding the in-hand pointcloud still, only move the fixed cloud
        ├── patch_selection.py          # RViz-selected center
        ├── surface_geometry.py         # PCA plane, normals, tangent stabilization
        ├── raster_pattern.py           # boustrophedon / unidirectional retract
        ├── pose_frames.py              # TF transforms and frame checks
    ├── microphone/                 # stand alone node, connect to microphone and broadcast audio
    ├── motion/
        ├── motion_runner_node.py       # raster plan -> MoveIt plan/preview/execute
        ├── segment_policy.py           # current/idle/parking/raster/retract/return
        ├── moveit_client.py            # MoveIt services/actions wrapper
        ├── trajectory_checks.py        # completion, limits, frame, boundary checks
    ├── supervisor/
        ├── experiment_supervisor_node.py
        ├── run_state.py
        ├── operator_gates.py
        ├── manifest.py
    ├── recording/                      # recorder node. cloud, images, audio in -> start/end timestamps, frame rates of each modality, start/end piont clouds (each and stitched), video and audio. Saved to an experiment artifact package. 
    ├── tests/
├── third_party.repos
