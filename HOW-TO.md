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

1. Pull vendor dependencies (Franka + RealSense) into the workspace:

```bash
cd ros2_ws
vcs import src < third_party.repos
```

That creates `ros2_ws/src/franka_ros2/` and `ros2_ws/src/realsense-ros/`, third party packages not tracked in git.

1. Install system/ROS dependencies:

```bash
rosdep install --from-paths src --ignore-src -r -y || true
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

```bash
colcon build --symlink-install \
  --packages-up-to franka_fr3_moveit_config realsense2_camera \
  --cmake-args -DBUILD_TESTING=OFF -DCMAKE_IGNORE_PREFIX_PATH=/usr/local

colcon build --symlink-install \
  --cmake-args -DCMAKE_IGNORE_PREFIX_PATH=/usr/local

cd ..
```

**Check the build succeeded** (all lines should appear):

```bash
source ./scripts/source_ubuntu24_ros_jazzy.sh
ros2 pkg list | grep fr3_sonopet
```

Expected: `fr3_sonopet_bringup`, `fr3_sonopet_description`, `fr3_sonopet_interfaces`, `fr3_sonopet_microphone`, `fr3_sonopet_motion`, `fr3_sonopet_recording`, `fr3_sonopet_supervisor`, `fr3_sonopet_tests`, `fr3_sonopet_trajectory`.

After you change only this project’s Python packages, **a faster rebuild**:

```bash
source ./scripts/source_ubuntu24_ros_jazzy.sh
cd ./ros2_ws
colcon build --symlink-install --packages-select fr3_sonopet_bringup fr3_sonopet_recording fr3_sonopet_microphone
cd ..
```

`--symlink-install` means Python edits are picked up without rebuilding when you only change `.py` files.

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

## 4. What to launch (pick your goal)


| Goal                         | Command                                                        | Hardware needed                                  |
| ---------------------------- | -------------------------------------------------------------- | ------------------------------------------------ |
| Cameras + mic + recorder     | `ros2 launch fr3_sonopet_bringup sensors.launch.py`            | 2× RealSense D405, iMM-6C mic (by default)       |
| Same + RViz                  | `ros2 launch fr3_sonopet_bringup sensors.launch.py rviz:=true` | Same                                             |
| Full robot experiment        | `ros2 launch fr3_sonopet_bringup experiment.launch.py`         | Franka FR3 + sensors                             |
| Robot stack without real arm | `ros2 launch fr3_sonopet_bringup fake_experiment.launch.py`    | No Franka/cameras (planning/motion in fake mode) |


### Sensor stack (best for topic + image visualization)

```bash
source scripts/source_ubuntu24_ros_jazzy.sh
ros2 launch fr3_sonopet_bringup sensors.launch.py rviz:=true
```

`sensors.launch.py` starts:

- `cameras.launch.py` — two D405s under `/in_hand_d405/...` and `/fixed_d405/...`
- `microphone.launch.py` — publishes `/microphone/audio`
- `recording.launch.py` — subscribes to those topics for artifacts
- **RViz2** only when `rviz:=true`, using `experiment.rviz`

**Microphone note:** default config requires a device whose name contains `iMM-6C` or `imm6c`. Otherwise the mic node exits unless you change `ros2_ws/src/fr3_sonopet_bringup/config/microphone.yaml`.

**Camera note:** serial numbers are in `cameras.launch.py` (`323622273258`, `427622272709`). Override if your cameras differ:

```bash
ros2 launch fr3_sonopet_bringup sensors.launch.py \
  in_hand_serial:=YOUR_SERIAL fixed_serial:=YOUR_SERIAL rviz:=true
```

### Fake robot (TF / MoveIt without hardware)

```bash
ros2 launch fr3_sonopet_bringup fake_experiment.launch.py
```

This does **not** start cameras, mic, or RViz. For RViz with the saved config:

```bash
ros2 run rviz2 rviz2 -d $(ros2 pkg prefix fr3_sonopet_bringup)/share/fr3_sonopet_bringup/rviz/experiment.rviz
```

Camera image panels will be empty until a camera launch is running.

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


| Topic                                             | Type (typical)      | Source                                   |
| ------------------------------------------------- | ------------------- | ---------------------------------------- |
| `/in_hand_d405/d405_in_hand/color/image_rect_raw` | `sensor_msgs/Image` | In-hand D405                             |
| `/fixed_d405/d405_fixed/color/image_rect_raw`     | `sensor_msgs/Image` | Fixed D405                               |
| `/microphone/audio`                               | custom `AudioChunk` | `microphone_node`                        |
| `/tf`, `/tf_static`                               | TF                  | Robot / description (if Franka launched) |


**Rates and echo:**

```bash
ros2 topic hz /in_hand_d405/d405_in_hand/color/image_rect_raw
ros2 topic hz /fixed_d405/d405_fixed/color/image_rect_raw
ros2 topic hz /microphone/audio
ros2 topic info /microphone/audio -v
```

**Images without RViz:**

```bash
rqt_image_view
# pick /in_hand_d405/d405_in_hand/color/image_rect_raw or the fixed camera topic
```

Point clouds are **off** by default (`pointcloud_enable:=false` in `cameras.launch.py`). The recorder briefly enables them for PCD snapshots; for live cloud visualization in RViz you would launch with something like:

```bash
ros2 launch fr3_sonopet_bringup sensors.launch.py pointcloud_enable:=true rviz:=true
```

(then add a PointCloud2 display in RViz on  
`/in_hand_d405/d405_in_hand/depth/color/points` or the fixed-camera equivalent).

---

## 6. RViz setup in this repo

Built-in RViz config: `fr3_sonopet_bringup/rviz/experiment.rviz`

It is preconfigured with:

- **Fixed Frame:** `fr3_link0` (needs robot TF — from Franka/fake experiment, not from `sensors.launch.py` alone)
- **TF** display
- **In Hand RGB** → `/in_hand_d405/d405_in_hand/color/image_rect_raw`
- **Fixed RGB** → `/fixed_d405/d405_fixed/color/image_rect_raw`

So for **camera-only** `sensors.launch.py rviz:=true`:

- Image panels should work if cameras are connected.
- TF may show warnings until something publishes `fr3_link0` (e.g. run `fake_experiment.launch.py` in another terminal, or full `experiment.launch.py`).

In RViz you can also use **Add → By topic** to browse everything currently publishing.

---

## 7. Start and stop artifact recording

The recorder node starts with `sensors.launch.py`, but it only writes files after you send a `RecordExperiment` action goal. Always source the workspace in the terminal that sends the action, or the custom action type will not resolve.

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

**Terminal 1 — run stack:**

```bash
cd ./franka3-sonopet-ros2
source scripts/source_ubuntu24_ros_jazzy.sh
ros2 launch fr3_sonopet_bringup sensors.launch.py rviz:=true
```

**Terminal 2 — inspect:**

```bash
cd ./franka3-sonopet-ros2
source scripts/source_ubuntu24_ros_jazzy.sh
ros2 topic list
rqt_image_view
```

---

## 9. Common pitfalls

1. `**ros2: command not found**` — source `scripts/source_ubuntu24_ros_jazzy.sh` (or `/opt/ros/jazzy/setup.bash`).
2. `**Package 'fr3_sonopet_bringup' not found**` — build with `colcon build` and source again.
3. `**franka_hardware` compile error (`getTargetFeedback`, headers under `/usr/local/include/franka`)** — source `scripts/source_ubuntu24_ros_jazzy.sh` before building and add `-DCMAKE_IGNORE_PREFIX_PATH=/usr/local` to every `colcon build` (see section 2).
4. **Terminal closes right after `colcon build` finishes** — the full workspace build can exit with code 1 when optional vendor packages fail; that is not a crash if `ros2 pkg list | grep fr3_sonopet` shows all nine packages. Re-open the terminal, source again, and run the grep check above. Do not chain `source ... && colcon build` in automation that treats any non-zero exit as fatal unless you also check the `fr3_sonopet_*` packages.
5. `**not found: ".../local_setup.bash"` when sourcing** — leftover from an interrupted build. Either finish a successful build or run `rm -rf ros2_ws/build ros2_ws/install ros2_ws/log` and rebuild from section 2.
6. **Launch fails on `realsense2_camera` or core `franka_*`** — run `vcs import` and the section 2 `colcon build` sequence. For RealSense `API version mismatch`, install `ros-jazzy-librealsense2` and rebuild `realsense2_camera`.
7. **Mic node dies immediately** — no matching USB mic; adjust `microphone.yaml` or set `fail_if_preferred_not_found: false` and a valid `device`.
8. **No camera images** — wrong serial, USB bandwidth, or camera unplugged; check `ros2 topic list` for `/in_hand_d405/...`.

