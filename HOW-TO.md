Walkthrough for this repo on **Ubuntu 24.04 + ROS 2 Jazzy**

---

## 1. One-time host setup

If this machine does not have ROS 2 Jazzy yet:

1. Install **ROS 2 Jazzy** from the [official ROS 2 docs](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debian.html).
2. From the repo root, install build tools:

```bash
cd path/to/franka3-sonopet-ros2
./scripts/bootstrap_ubuntu24_ros_jazzy.sh
```

1. Pull vendor dependencies (Franka + RealSense wrapper) into the workspace:

```bash
cd ros2_ws
vcs import src < third_party.repos
```

That creates `ros2_ws/src/franka_ros2/` and `ros2_ws/src/realsense-ros/`, third party packages not tracked in git.

1. Put the RealSense SDK source beside the wrapper packages:

```bash
cd ros2_ws/src/realsense-ros
git clone https://github.com/IntelRealSense/librealsense.git -b v2.56.4
```

This workspace intentionally builds against this local `librealsense` checkout. Do **not** rely on the apt/ROS `ros-jazzy-librealsense2` CMake package; on this machine it may point at `/opt/ros/jazzy/lib/x86_64-linux-gnu/librealsense2.so.2.56.4` even when that file is missing.

1. Install system/ROS dependencies:

```bash
rosdep install --from-paths src --ignore-src -r -y --skip-keys=librealsense2 || true
```

Review any failures and install missing packages manually if needed.

---

## 2. Build the workspace

Always work from `ros2_ws` for `colcon`. Use a **fresh terminal** and source only this repo (do not also source `~/franka_ws/install/setup.bash` while building here).

```bash
cd /path/to/franka3-sonopet-ros2
source scripts/source_ubuntu24_ros_jazzy.sh
cd ros2_ws
```

**First build** (vendor stack, then everything — from README).

- Pass `CMAKE_IGNORE_PREFIX_PATH` on **both** `colcon build` commands if other manual install of libfranka exists, or `franka_hardware` may fail against the wrong headers.
- Also hide the broken apt/ROS RealSense CMake package so `realsense2_camera` finds the workspace-built `librealsense`, not `/opt/ros/jazzy/.../realsense2.disabled`.

```bash
colcon build --symlink-install \
  --packages-up-to franka_fr3_moveit_config realsense2_camera \
  --packages-skip mobile_fr3_duo_trajectory_controller \
  --cmake-args \
    -DBUILD_TESTING=OFF \
    -DCMAKE_IGNORE_PREFIX_PATH=/usr/local \
    -DCMAKE_IGNORE_PATH=/opt/ros/jazzy/lib/x86_64-linux-gnu/cmake/realsense2.disabled

colcon build --symlink-install \
  --packages-skip mobile_fr3_duo_trajectory_controller \
  --cmake-args \
    -DCMAKE_IGNORE_PREFIX_PATH=/usr/local \
    -DCMAKE_IGNORE_PATH=/opt/ros/jazzy/lib/x86_64-linux-gnu/cmake/realsense2.disabled

cd ..
```

Vendor tree includes `mobile_fr3_duo_trajectory_controller`, which currently fails under `--symlink-install` in this workspace. Skip it unless you are actively developing that package.

**Check the build succeeded** (all lines should appear):

```bash
source ./scripts/source_ubuntu24_ros_jazzy.sh
ros2 pkg list | grep fr3_sonopet
```

Expected: `fr3_sonopet_bringup`, `fr3_sonopet_description`, `fr3_sonopet_interfaces`, `fr3_sonopet_microphone`, `fr3_sonopet_motion`, `fr3_sonopet_recording`, `fr3_sonopet_supervisor`, `fr3_sonopet_tests`, `fr3_sonopet_trajectory`.

After you change only the high-iteration experiment Python packages, **a faster rebuild**:

```bash
source ./scripts/source_ubuntu24_ros_jazzy.sh
cd ./ros2_ws
colcon build --symlink-install --packages-select fr3_sonopet_bringup fr3_sonopet_trajectory fr3_sonopet_recording fr3_sonopet_microphone
cd ..
```

`--symlink-install` means Python edits are picked up without rebuilding when you only change `.py` files.

If you modified project packages **other than** trajectory, recording, or microphone, build the touched package set directly:

```bash
source ./scripts/source_ubuntu24_ros_jazzy.sh
cd ./ros2_ws
colcon build --symlink-install \
  --packages-select fr3_sonopet_bringup fr3_sonopet_description fr3_sonopet_motion fr3_sonopet_supervisor fr3_sonopet_tests
cd ..
```

If you changed `fr3_sonopet_interfaces` messages or actions, rebuild the interfaces plus downstream project packages:

```bash
source ./scripts/source_ubuntu24_ros_jazzy.sh
cd ./ros2_ws
colcon build --symlink-install \
  --packages-up-to fr3_sonopet_bringup fr3_sonopet_motion fr3_sonopet_recording fr3_sonopet_supervisor fr3_sonopet_trajectory fr3_sonopet_tests
cd ..
```

---

## 3. Source in every new terminal

You must source **both** system ROS and this overlay. The project script does that for you:

```bash
source scripts/source_ubuntu24_ros_jazzy.sh
```

That script:

- Sources `/opt/ros/jazzy/setup.bash`
- Sources `ros2_ws/install/setup.bash` if it exists
- Sets `FR3_SONOPET_REPO` to the repo root

Quick check:

```bash
echo $ROS_DISTRO          # should print: jazzy
ros2 pkg list | grep fr3_sonopet
```

If packages are missing, you forgot to source or the build failed.

---

## 4. What to launch

This project exposes three launch entry points. Always `source scripts/source_ubuntu24_ros_jazzy.sh` first.

### Full boot (sensors + raster + RViz)

```bash
source scripts/source_ubuntu24_ros_jazzy.sh
ros2 launch fr3_sonopet_bringup experiment.launch.py rviz:=true
```

Everything in **Real arm**, plus:

- RViz with `experiment.rviz`
- In-hand RealSense point cloud enabled until `raster_planner_node` captures the frozen planning cloud
- Live RGB panels and raster overlays after you pick a surface point (section 6)

**Config files:** camera serials in `config/cameras.yaml`; raster crop/trim in `config/raster.yaml`. Default mic config requires a device name containing `iMM-6C` or `imm6c` (`config/microphone.yaml`).



<!-- ### Real arm

```bash
source scripts/source_ubuntu24_ros_jazzy.sh
ros2 launch fr3_sonopet_bringup experiment.launch.py
```

Starts Franka control, tool TF, both D405s, microphone, recording node, raster planner, motion runner, and experiment supervisor. Continuous point clouds stay off; the recorder still snapshots PCDs at recording start/stop. -->

### Fake arm

```bash
source scripts/source_ubuntu24_ros_jazzy.sh
ros2 launch fr3_sonopet_bringup fake_experiment.launch.py
```

Same planning/motion/supervisor graph with `use_fake_hardware:=true` and `fake_run` enabled. No cameras, mic, recorder, or raster capture — use this to exercise TF and motion without the bench hardware.

---

## 5. Inspect topics (CLI)

With the stack running, in a **second terminal** (source again first):

```bash
source scripts/source_ubuntu24_ros_jazzy.sh
```

**Graph and list:**

```bash
ros2 node list
ros2 topic list
rqt_graph
```

**Important topics from this project:**


| Topic                                          | Type (typical)                   | Source                                       |
| ---------------------------------------------- | -------------------------------- | -------------------------------------------- |
| `/RealSense_D405/in_hand/color/image_rect_raw` | `sensor_msgs/Image`              | In-hand D405                                 |
| `/RealSense_D405/fixed/color/image_rect_raw`   | `sensor_msgs/Image`              | Fixed D405                                   |
| `/RealSense_D405/in_hand/depth/color/points`   | `sensor_msgs/PointCloud2`        | In-hand D405 (only when point cloud enabled) |
| `/sonopet/planning_cloud`                      | `sensor_msgs/PointCloud2`        | `raster_planner_node` (frozen, `fr3_link0`)  |
| `/clicked_point`                               | `geometry_msgs/PointStamped`     | RViz **Publish Point** tool                  |
| `/sonopet/raster_plan/poses`                   | `geometry_msgs/PoseArray`        | `raster_planner_node` after a pick           |
| `/sonopet/raster_plan/markers`                 | `visualization_msgs/MarkerArray` | Raster path/patch/normal preview             |
| `/microphone/audio`                            | custom `AudioChunk`              | `microphone_node`                            |
| `/tf`, `/tf_static`                            | TF                               | Robot / cameras / tool frames                |


**Rates and echo:**

```bash
ros2 topic hz /RealSense_D405/in_hand/color/image_rect_raw
ros2 topic hz /RealSense_D405/fixed/color/image_rect_raw
ros2 topic hz /microphone/audio
ros2 topic info /microphone/audio -v
ros2 topic echo /clicked_point --once
```

**Images without RViz:**

```bash
rqt_image_view
# pick /RealSense_D405/in_hand/color/image_rect_raw or the fixed camera topic
```

Continuous RealSense point clouds are **off** in headless `experiment.launch.py`. The recorder briefly enables them for start/end PCD snapshots. **Full boot** (`rviz:=true`) turns on the in-hand stream until the planner captures once, then RViz shows `/sonopet/planning_cloud` instead of the live camera topic.

---

## 6. Full boot: RViz, planning cloud, surface pick

Use **Full boot** from section 4 — RViz is only wired for that launch:

```bash
ros2 launch fr3_sonopet_bringup experiment.launch.py rviz:=true
```

Config file: `fr3_sonopet_bringup/rviz/experiment.rviz` (Fixed Frame `fr3_link0`).


| Display             | Topic                                          | Notes                                   |
| ------------------- | ---------------------------------------------- | --------------------------------------- |
| TF                  | —                                              | Robot, `sonopet_tcp`, camera mounts     |
| Planning PointCloud | `/sonopet/planning_cloud`                      | **Frozen** startup cloud in `fr3_link0` |
| Raster Poses        | `/sonopet/raster_plan/poses`                   | After a surface pick                    |
| Raster Markers      | `/sonopet/raster_plan/markers`                 | Path, patch square, surface normal      |
| In Hand RGB         | `/RealSense_D405/in_hand/color/image_rect_raw` | Live                                    |
| Fixed RGB           | `/RealSense_D405/fixed/color/image_rect_raw`   | Live                                    |


### Planning cloud (shown once)

On the first in-hand depth cloud message, `raster_planner_node`:

1. Crops to 0.5 m around the camera optical origin
2. Trims the farthest 20% of points (`config/raster.yaml`: `planning_cloud_max_distance_m`, `planning_cloud_trim_fraction`)
3. Transforms into `fr3_link0` and publishes `/sonopet/planning_cloud`

That cloud is kept for all later RViz display and raster fitting; new camera frames are ignored. Wait for the planner log (`Captured planning cloud with … points`) before picking.

### Pick a surface point

1. Toolbar → **Publish Point** (not **Select**).
2. Click on the **Planning PointCloud** in the 3D view.
3. RViz publishes `/clicked_point`; the planner responds with `/sonopet/raster_plan/poses` and `/sonopet/raster_plan/markers`.

On failure, read `raster_planner_node` for `Raster plan rejected:` (cloud not captured yet, click off the cloud, or TF missing).

---

## 7. Start and stop artifact recording

The recorder node starts with **Real arm** / **Full boot**, but it only writes files after you send a `RecordExperiment` action goal. Always source the workspace in the terminal that sends the action, or the custom action type will not resolve.

**Check the action server:**

```bash
source scripts/source_ubuntu24_ros_jazzy.sh
ros2 action list -t | grep /sonopet/record_experiment
ros2 interface show fr3_sonopet_interfaces/action/RecordExperiment
```

**Start a run:**

```bash
ros2 action send_goal /sonopet/record_experiment \
  fr3_sonopet_interfaces/action/RecordExperiment \
  "{run_id: test_run_001, start: true}"
```

**Stop the same run:**

```bash
ros2 action send_goal /sonopet/record_experiment \
  fr3_sonopet_interfaces/action/RecordExperiment \
  "{run_id: test_run_001, start: false}"
```

After stopping, artifacts are written under `artifacts/experiments/<run_id>/`, including camera videos, audio, point-cloud snapshots when available, and `manifest.json`.

---

## 8. Typical two-terminal workflow

**Terminal 1 — full boot:**

```bash
cd ./franka3-sonopet-ros2
source scripts/source_ubuntu24_ros_jazzy.sh
ros2 launch fr3_sonopet_bringup experiment.launch.py rviz:=true
```

After the planning cloud appears, use **Publish Point** on the surface (section 6). **Terminal 2:**

```bash
cd ./franka3-sonopet-ros2
source scripts/source_ubuntu24_ros_jazzy.sh
ros2 topic echo /sonopet/raster_plan/poses --once
```

For arm-only checks without sensors, use **Fake arm** (section 4) in terminal 1 instead.

---

## 9. Common pitfalls

1. `**ros2: command not found`** — source `scripts/source_ubuntu24_ros_jazzy.sh` (or `/opt/ros/jazzy/setup.bash`).
2. `**Package 'fr3_sonopet_bringup' not found**` — build with `colcon build` and source again.
3. `**franka_hardware` compile error (`getTargetFeedback`, headers under `/usr/local/include/franka`)** — source `scripts/source_ubuntu24_ros_jazzy.sh` before building and add `-DCMAKE_IGNORE_PREFIX_PATH=/usr/local` to every `colcon build` (see section 2).
4. `**mobile_fr3_duo_trajectory_controller` CMake/symlink install error** — add `--packages-skip mobile_fr3_duo_trajectory_controller` to every full-workspace `colcon build` (section 2). If `fr3_sonopet_bringup` still built, the Sonopet stack is usable.
5. **Terminal closes right after `colcon build` finishes** — treat as success when `ros2 pkg list | grep fr3_sonopet` lists all nine project packages; re-source and continue.
6. `**not found: ".../local_setup.bash"` when sourcing** — leftover from an interrupted build. Either finish a successful build or run `rm -rf ros2_ws/build ros2_ws/install ros2_ws/log` and rebuild from section 2.
7. **`realsense2_camera` fails on missing `librealsense2.so.2.56.4`** — clone `librealsense` into `ros2_ws/src/realsense-ros/`, skip the rosdep `librealsense2` key, and rebuild with the section 2 `CMAKE_IGNORE_PATH` command.
8. **Mic node dies immediately** — no matching USB mic; adjust `microphone.yaml`.
9. **No camera images** — wrong serial in `cameras.yaml`, USB bandwidth, or camera unplugged; check `ros2 topic list` for `/RealSense_D405/...`.
10. **Empty Planning PointCloud** — use **Full boot** (`rviz:=true`); wait for planner capture log; confirm TF from in-hand camera to `fr3_link0`.
11. **Click does nothing** — toolbar **Publish Point** on the planning cloud; read `raster_planner_node` for `Raster plan rejected:`.

