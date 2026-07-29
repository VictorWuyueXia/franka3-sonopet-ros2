>To prepare for a re-structure overhaul of our current robot experiment pipelines, we first need a detailed account of what the experiment does. The overhaul will focus on the boundaries defined by `datacollection_franky`, `/home/victor-xia/repos/sonopet/simulation`, `gui`, and `recorder`. The primary purpose is to reorganize these components for the easiest possible human maintenance effort and for improved human readability and comprehension, embracing a more human-oriented programming style. In addition, we aim to minimize the overall codebase by leveraging ROS2 Jazzy and existing ROS2 packages for Franka Research 3 control and RealSense D405 integration. To support this effort, carefully examine the relevant portions of the codebase and provide a detailed workflow document to serve as a reference for the planned overhaul.



# Sonopet Data Collection Workflow

Reference document for the planned ROS2 jazzy overhaul of `datacollection_franky/`, `simulation/`, `gui/`, and `recorder/`.

All paths and symbols below reflect the current repository state.

---

## 0. Global view: subsystems and data flow

| Subsystem | Role | Main entry | Key external deps |
|---|---|---|---|
| `datacollection_franky/` | Planning + execution (offline IK + realtime dispatch) | `run_pipeline.py` -> `pipeline.main()` | `franky-panda` (libfranka), `pybullet` |
| `simulation/` | PyBullet simulation backend + pointcloud -> pose waypoints | `simWithPyBullet.BulletRobotSim`, `cloudpoint_wrapper.PointCloudWrapper` | `pybullet`, `open3d` |
| `gui/` | Live visualization + multi-sensor experiment GUI + RealSense dual-camera stitching | `gui/run_experiment.py`, `gui/realsense_capture/cli.py` | `PyQt5`, `pyrealsense2`, `PySpin`, `sounddevice`, `tf2_ros` |
| `recorder/` | One-click multi-device recording (D405 + USB mic) | `recorder/run.sh` -> `recorder/launcher.py` | `pyrealsense2`, `arecord (alsa-utils)` |

One-way import / subprocess relations:

```
recorder/  ----------> gui/realsense_capture/  ----------> datacollection_franky/
                              |                                          |
                              v                                          v
                  realsense_boot.py (top level)                    simulation/
```

- `recorder/record_d405.py` pulls RealSense streams, pointclouds, and robot pose through `gui.realsense_capture.*`.
- `gui/realsense_capture/robot_pose.py` runs the robot probe and local PyBullet FK through `datacollection_franky.*`.
- `datacollection_franky/pipeline.py` uses `find_simulation_dir` to inject `simulation/` into `sys.path`, then imports `cloudpoint_wrapper`, `simWithPyBullet`.
- Top-level `realsense_boot.py` is the shared RealSense device-discovery / preset-loading layer.

---

## 1. `datacollection_franky/`: Planning and execution pipeline

### 1.1 Physical layout

```
datacollection_franky/
├── run_pipeline.py        # Real executable entry (adds parent dir to sys.path, imports .pipeline.main)
├── pipeline.py            # main() orchestration: parse -> prepare -> sim/compile/preview -> hardware
├── pipeline_runtime.py    # Process runtime: CPU partition, RT thread policy, robot probe subprocess
├── pipeline_simulation.py # Sim stage: pointcloud -> pose waypoints, PyBullet connect, IK compile, preview
├── pipeline_hardware.py   # Hardware stage: reconnect, per-segment dispatch, HardwareSegmentPlan
├── cli.py                 # argparse: narrow user-facing CLI
├── config.py              # All defaults (dataclass): TIMING/STAGE/FRANKY/...
├── franky_backend.py      # FrankyRobot wrapper (try_connect / execute_joint_waypoints)
├── trajectory_compile.py  # 5-stage trajectory + speed planning + velocity/acceleration hints
├── ik_compile.py          # Per-waypoint IK + FK validation + one retry
├── densify.py             # Max-step Cartesian densify (with quat slerp)
├── preview.py             # PyBullet preview loop (POSITION_CONTROL)
├── visualization.py       # Draw square / raster curve / TCP frame in PyBullet
├── robot_probe.py         # Subprocess entry: `python -m datacollection_franky.robot_probe`
├── runtime_affinity.py    # Compatibility wrapper, forwards to util/*
├── sim_paths.py           # Locate simulation/ by walking parents
├── sim_runtime.py         # GLX/PyBullet GUI probing, orientation modes, validate_joint_dict
├── tcp_config.py          # Read sonopet_tcp_joint from fr3.urdf, add 5/10/26 mm trim
├── urdf_tcp.py            # find_fr3_urdf
├── math_utils.py          # quat_slerp / quat_abs_dot / l2_dist
├── path_utils.py          # Compat: forwards find_simulation_dir / find_fr3_urdf
├── prompts.py             # info/warn/error/confirm_or_abort
└── util/
    ├── cpu_partition.py   # >=16 cores: 4+pool+2;  3-15: 2+rest;  <3: shared
    └── thread_affinity.py # SCHED_FIFO/RR detect, pin_all_rt_threads, cap_rt_priorities
```

### 1.2 End-to-end stage sequence

`pipeline.main()` (`pipeline.py:96-127`) runs the following strict order. Each stage logs through `info()`. Failures surface as `SystemExit` or `RuntimeError` (no silent try/except).

| Stage | Function | Behavior | Key side effect |
|---|---|---|---|
| 0 | `parse_pipeline_args` + `validate_pipeline_args` | Parse 15 CLI flags (see 1.3), numeric range checks | - |
| 1 | `_load_sim_environment` | `find_simulation_dir(..., max_up=6)`, add `simulation/` to `sys.path`, import `PointCloudWrapper / PoseWaypoint / RasterSurfaceSpec / BulletRobotSim` | Mutates `sys.path` |
| 2 | `initialize_runtime_state` | `get_cpu_partition()`, derive `rt_cpu_set / ik_cpu_set`, pin main thread to `ik_cpu_set` | `sched_setaffinity` |
| 3 | `acquire_initial_robot_state` | If `--preview-only` or no `--robot-ip`: return `preview_fallback_pose`. Else run **subprocess probe** (`-m datacollection_franky.robot_probe`) to get `q_current` and RT-thread stats | Spawns + reaps subprocess |
| 4 | `run_pointcloud_stage` | Temporarily clear `__GLX_VENDOR_LIBRARY_NAME` (for Open3D picker), build `PointCloudWrapper`, load PCD, optional Open3D pick -> local cube crop -> PCA tangent plane -> raster -> normal estimation -> `List[PoseWaypoint]` | Pops Open3D window |
| 5 | `prepare_sim_connect` -> `connect_bullet_sim_with_env` | Probe PyBullet GUI (`probe_pybullet_gui_subprocess`), fall back to no-GPU env if needed; connect PyBullet, load URDF, pick EE link (`sonopet_tcp` or last link) | Opens PyBullet GUI |
| 6 | `configure_pose_orientation` | Hard-reset sim joints to `q_current`, compute idle TCP quat, apply `--orientation-mode (surface_normal | current | preset)` to override raster waypoint orientation | Mutates sim joint state |
| 7 | `compile_offline_trajectory` -> `compile_full_joint_trajectory` | Offline build of the 5-segment trajectory (see 1.4), returns `CompiledTrajectory` | - |
| 8 | `run_preview_and_disconnect` | Draw square / raster / idle TCP / parking TCP. Prompt `<y>` to run preview or `<s>` to skip. Disconnect (unless `--preview-only`) | Blocks on input |
| 9 | `run_hardware_stage_if_needed` | If connected and not preview-only: confirm `EXECUTE` token, then 1.5 flow | Real robot motion |
| 10 | `info(f"Total elapsed_s=...")` | - | - |

### 1.3 Current CLI surface (`cli.py`)

Only 15 flags. Everything else has been pushed into `config.py` dataclasses.

```
--robot-ip <ip>                                 default 172.16.0.2
--preview-only                                  force no hardware
--pcd-path <path>                               PCD input
--pcd-frame {camera_optical, fr3_link0}         PCD frame
--square-side <m>                               local patch side
--square-center X Y Z                           optional, skip Open3D picker
--line-spacing <m>                              raster line spacing
--pointcloud-downsample-rate <int>              per-line decimation
--raster-pattern {boustrophedon, unidirectional_retract}
--raster-z-offset <m>                           transient world-Z lift
--dynamics-scale <float>                        velocity s, acceleration s^2
--sim-time-scale <float>                        preview speed multiplier
--orientation-mode {surface_normal, current, preset}
--final-return-mode {workflow_start, idle}      default returns to workflow start
--preset-orientation idle | --preset-rpy-deg ... | --preset-quat-xyzw ...
```

### 1.4 Offline trajectory structure (`trajectory_compile.py`)

`compile_full_joint_trajectory` builds **A -> B -> C -> D -> E1 -> E2** in one pass and inserts 2 `W_after_<seg>` hold points between segments.

| Segment | Name | Space | Method | Default vmax |
|---|---|---|---|---|
| A | `A_current_to_idle` | Joint | Linear lerp, `SEGMENT_A_INTERP_DT_S = 0.5` s | 0.05 m/s |
| B | `B_idle_to_parking` | Cartesian -> IK | `densify_by_max_step(max_step=0.01)` -> `compile_cartesian_waypoints_to_joint_waypoints` | 0.03 m/s |
| C | `C_parking_to_first` | Cartesian -> IK | Same | 0.01 m/s |
| D | `D_raster` | Cartesian -> IK | All raster points: densify, then IK | 0.005 m/s |
| E1 | `E_retract` | Cartesian -> IK | Last waypoint +Z `cfg.stages.retract_lift_m` | 0.03 m/s |
| E2 | `E2_to_workflow_start` or `E2_to_idle` | Joint | Joint lerp, `SEGMENT_E2_INTERP_DT_S = 0.05` s | 0.03 m/s |

Pipeline inside `compile_full_joint_trajectory`:
1. `_build_runtime_sequence` stitches the segments with `W_after_*` hold points.
2. `_compute_tcp_positions_for_waypoints` runs FK for every point.
3. `_assign_segment_dt_by_speed` solves a trapezoidal `vmax + acc` profile per segment for `dt_s`; hold points clamp to `HOLD_DT_S = 0.05`.
4. `_compute_velocity_hints` central-difference `dq_by_name`, with `BOUNDARY_RAMP_POINTS = 2` linear ramps that force zero velocity at segment edges.
5. `_compute_acceleration_hints` repeats for `ddq_by_name`.
6. `_check_trajectory_continuity` asserts `dt > 0`, `|dq| <= max_joint_vel_rad_s`, and strict zero velocity/acceleration at every segment boundary.
7. Returns `CompiledTrajectory{points, segments, segment_a_interp_dt_s}`.

`ik_compile.py:solve_ik_checked` per-waypoint acceptance: `pos_err <= 5 mm` AND `|quat . quat_target| >= 0.995`. On failure, retry once (`max_iters` 120 -> 240, `residual_thresh` 5e-5 -> 1e-4).

### 1.5 Hardware execution (`pipeline_hardware.py`)

```
run_hardware_stage_if_needed
  -> confirm_or_abort("Type EXECUTE")
  -> close_preview_window_if_open
  -> run_hardware_execution
       -> connect_robot_for_execution (FrankyRobot.try_connect + apply_rt_thread_policy)
       -> compile_execution_hardware_segments
            -> Spawn headless BulletRobotSim
            -> Re-run compile_full_joint_trajectory with live robot q_current
            -> split_runtime_points_into_motion_segments (drop hold)
            -> Per segment: build_single_hardware_segment_plan
                 -> downsample_runtime_points_for_hardware
                    (D_raster: keep all;  others: keep first/mid/last)
       -> Per segment: robot.execute_joint_waypoints(plan.hardware_points,
                                                     relative_dynamics_factor)
            -> franky.JointWaypointMotion -> robot.move(motion)  (blocking)
```

Notes:
- Segments C and D use `cfg.franky.segment_cd_relative_dynamics_factor = 0.005`; all other segments use `0.05`. Names live in `SEGMENT_CD_NAMES = ("C_parking_to_first", "D_raster")`.
- Franky only receives positions (`JointWaypoint(pos)`); `JointState(position, velocity)` is used only when `dq_hint_by_name is not None`.
- The preview sim must be disconnected before starting a fresh `BulletRobotSim(gui=False)` for execute-time recompile (to avoid two PyBullet clients interfering).

### 1.6 Process-level runtime (`pipeline_runtime.py` + `util/`)

| Concern | Implementation |
|---|---|
| CPU partition | `get_cpu_partition`: `>=16` cores -> `rt=[0:4]`, `ik=[4:-2]`, leave 2 for OS; `3-15` cores -> `rt=[:2]`, `ik=[2:]`; `<3` shared |
| Main-thread pinning | `pin_current_thread_to_cpus(ik_cpu_set)` using `threading.get_native_id()` |
| RT thread treatment | `apply_rt_thread_policy(rt_cpu_set, priority_cap=80)`: scan `/proc/self/task`, pin every `SCHED_FIFO/RR` thread to `rt_cpu_set`, cap priority to <=80 |
| Robot probe isolation | `subprocess.run([sys.executable, "-m", "datacollection_franky.robot_probe", ...])` grabs `q_current` once. After the subprocess exits, lingering libfranka RT threads are gone too, keeping the main process clean |
| Debug snapshot | `log_runtime_snapshot(tag, enabled)` reads `/proc/self/status` and logs `pid/tid/threads/vmrss_kb/affinity` |

---

## 2. `simulation/`: PyBullet backend + pointcloud waypoint factory

### 2.1 File inventory

| File | Role |
|---|---|
| `simWithPyBullet.py` | `BulletRobotSim`: connect / load_robot / IK / FK / TCP offset wrapper |
| `cloudpoint_wrapper.py` | `PointCloudWrapper`: PCD -> Open3D pick -> local cube -> raster -> normals -> `PoseWaypoint` |
| `raster_patterns.py` | `RasterPatternSpec`, `build_raster_points_from_selected_square` (PCA tangent + two raster modes) |
| `trajectories.py` | Legacy Cartesian raster trajectory builder (current pipeline bypasses it, but kept) |
| `RasterTraj_PyBullet.py` | Legacy stand-alone raster + PyBullet main |
| `test_franky.py`, `test_raster_patterns.py` | Local smoke tests |
| `fr3.urdf` | FR3 URDF with Sonopet tool (`EE_base_joint` + `sonopet_tcp_joint`) |
| `franka_description/` | Official Franka meshes |
| `ee_simple*.dae/stl`, `sonopet.dae` | Tool EE meshes |
| `_sonopet_fr3_tcp_configured.urdf` | Historical artifact; `urdf_tcp.write_configured_fr3_urdf` is now a passthrough |

### 2.2 `BulletRobotSim` capabilities (`simWithPyBullet.py`)

- **Construction**: `__init__` calls `load_sonopet_tcp_geometry()` to compute `link8 -> tcp` (nominal from URDF + trim from `tcp_config.py`: `X=5 mm, Y=10 mm, Z=26 mm`).
- **`connect`**: `p.GUI` or `p.DIRECT`, gravity currently zeroed (`p.setGravity(0, 0, 0)`), fixed `fixedTimeStep`, `numSubSteps=4`, `numSolverIterations=150`.
- **`load_robot`**: `URDF_USE_INERTIA_FROM_FILE | URDF_IGNORE_COLLISION_SHAPES`, then `_assign_small_inertia_for_zero_inertia_links` (1e-3 minimum inertia to avoid blow-up), `_introspect_joints` keeps revolute / prismatic only, `_reset_to_idle_pose_on_load` hard-resets to idle.
- **`solve_ik`**: aligns to PyBullet movable-joint order; first runs `_tcp_target_to_link_target` to invert TCP offset (only trim if `sonopet_tcp` link exists, else full nominal+trim); then `p.calculateInverseKinematics`.
- **`get_tcp_pose`**: `getLinkState(ee_link) * tcp_offset` -> world TCP pose.
- **`set_joint_positions`**: array-form `POSITION_CONTROL` with `positionGains=0.1`, `velocityGains=1.0`, `targetVelocities=0` (comment notes that omitting velocity gain causes oscillation).
- **Idle pose**: `_compute_idle_pose_from_urdf` is hard-coded FR3 home (`fr3_joint2=-45 deg, fr3_joint4=-135 deg, fr3_joint6=90 deg`, rest 0).
- **Extension mechanism**: `install_bullet_robot_sim_extensions()` monkey-patches `_brs_*` functions onto the class at module load. Worth turning into normal methods in the rewrite.

### 2.3 `PointCloudWrapper.get_sim_waypoints` workflow

```
load_pointcloud_camera_frame   # o3d.io.read_point_cloud(pcd_path)
-> resolve_pointcloud_base_frame  # camera_optical -> fr3_link0 via
                                  # config/camera_extrinsics/fr3_eye_to_hand.json
-> [optional] select_square_center_from_pointcloud
       - o3d.visualization.VisualizerWithEditing  (Shift+click)
       - build_local_tangent_square_corners (PCA tangent plane)
       - _draw_local_tangent_square (second Open3D window to confirm)
-> crop_pointcloud_to_local_cube   # AABB around square_center, side = square_side_len_m
-> generate_raster_points_from_center
       - simulation/raster_patterns.py:build_raster_points_from_selected_square
         - PCA -> (u, v, normal)
         - Split N lines in UV plane, downsample_line_points keeps endpoints
         - boustrophedon (snake)  OR  unidirectional_retract (small +Z lift)
-> estimate_surface_waypoints      # Scene.calculate_surface_normals_and_waypoints
       - Local PCA on base cloud -> normal -> waypoint_pcd + normal_lines
-> build_pose_waypoints
       - _extract_normals_from_line_pairs
       - _estimate_tangents               (sign-stabilized)
       - _stabilize_tangents_with_global_reference
       - _quat_from_tangent_and_normal    x=tangent, z=normal, y=z x x
       - _stabilize_quaternion_signs_in_place
-> List[PoseWaypoint(xyz, quat_xyzw, normal_xyz)]
```

`pipeline_simulation.run_pointcloud_stage` then applies `raster_z_offset` and `apply_orientation_mode` before feeding `compile_full_joint_trajectory`.

### 2.4 Bypass details to clean up

- `_debug_emit` and `_agent_debug_log` in `cloudpoint_wrapper.py` and `simWithPyBullet.py` write to `.cursor/debug-*.log`. Pure historical agent debug noise; delete.
- `Scene` from `rob_control/pointcloud_process/process_scene.py` is used only for `calculate_surface_normals_and_waypoints`; collapse into a standalone helper.
- `simulation/RasterTraj_PyBullet.py` and `simulation/trajectories.py` are legacy entry points not used by the new pipeline.

---

## 3. `gui/`: Live visualization + dual-camera pointcloud + ROS TF bridge

### 3.1 Top-level files

| File | Role |
|---|---|
| `gui/run_experiment.py` | Main experiment GUI (PyQt5) with RealSense / Blackfly x2 / mic tabs and global Start/Stop/Record/Save All |
| `gui/realsense.py` | 12-line shim: `from realsense_capture import main; main()` |
| `gui/realsense_core.py` | 46-line minimal D405 single-camera preview |
| `gui/camera.py` | Legacy single Blackfly preview |
| `gui/camera_dual.py` | Dual Blackfly sync preview |
| `gui/microphone.py` | `AudioRecorder` (QThread + sounddevice) + stand-alone mic GUI |
| `gui/usb_microphone.py` | USB-mic-only slim version |
| `gui/live_plot.py` | matplotlib realtime plot helper |
| `gui/thermal.py` | FLIR Boson preview |
| `gui/ros_tf_probe.py` | Subprocess entry; uses `importlib.util` to lazily load `realsense_capture/ros_tf_probe.py` |

### 3.2 `gui/realsense_capture/` package

```
realsense_capture/
├── __init__.py            # Exposes DualCameraCaptureApp / main / parse_args
├── cli.py                 # argparse + main(): --list-devices / --list-camera-map / --serial (legacy) / dual
├── capture_app.py         # DualCameraCaptureApp: OpenCV window + 'c' capture + 'q' quit
├── stream.py              # RealSenseStream: start/warm_up/poll_frame/capture_pointcloud/stop + filter pipeline
├── roles.py               # CameraRoleConfig (camera-fixed / camera-in-hand)
├── robot_pose.py          # RobotTcpPoseResolver: local PyBullet FK + ROS TF probe + FrankaTfHelperThread
├── ros_env.py             # discover_ros_setup_scripts + build_franka_tf_helper_command
├── ros_tf_probe.py        # rclpy + tf2_ros listener (runs only under SYSTEM_PYTHON=/usr/bin/python3 subprocess)
├── constants.py           # BASE_FRAME_NAME=fr3_link0, TCP_FRAME_NAME=fr3_hand_tcp, ROS_SETUP_BASH=/opt/ros/jazzy/setup.bash
├── preview.py             # Display panel + Open3D show_pointcloud + write_pointcloud(.pcd)
├── metadata.py            # Per-camera + global merge_summary metadata
├── legacy_capture.py      # --serial single-camera legacy path
└── pointcloud/
    ├── transforms.py      # TransformSpec + 4x4 matrices + compose / invert / apply_parent_frame_trim
    ├── alignment.py       # resolve_base_alignment: eye-in-hand needs robot pose, eye-to-hand reads JSON
    ├── registration.py    # Open3D ICP refinement (refinement_mode: always/never/auto + dofs: z/translation/full)
    └── storage.py         # PointCloudStorage: manages artifacts/pointcloud_scans/
```

### 3.3 `run_experiment.py` workflow

```
ExperimentWindow
├── _init_sensors
│   ├── RealSenseTab           single (uses RealSenseStream)
│   ├── BlackflyTab x N        N = PySpin.System.GetCameras().GetSize()
│   └── MicrophoneTab          uses microphone.AudioRecorder
├── Global buttons:
│   ├── Start All / Stop All
│   ├── Start Recording All    QFileDialog -> mkdir experiment_<ts>/
│   ├── Stop Recording All
│   └── Save All Data          per-tab save_data(save_dir):
│                                 - realsense: color_*.png + depth_*.png + metadata.json
│                                 - blackfly:  frame_*.png + metadata.json
│                                 - microphone: audio.wav (stereo) + metadata.json
```

Each sensor tab has 4 buttons: `Start/Stop` stream control, `Start/Stop Recording` push frames into `self.recorded_data: list[dict]` (in-memory only, no streaming-to-disk).

- `RealSenseThread` reuses `gui.realsense_capture.stream.RealSenseStream`, so the RealSense filter stack, preset loading, and device discovery are unified app-wide.
- `BlackflyThread` does its own `Init / BeginAcquisition` and pixel format negotiation (prefer BGR8, fall back to RGB8).

### 3.4 Dual-camera capture flow (`DualCameraCaptureApp.run`)

```
load_camera_roles  <-  config/realsense_camera_roles.json
   { "camera-fixed":   {"serial": "..."},
     "camera-in-hand": {"serial": "..."} }
-> resolve_active_cameras -> one RealSenseStream per role
-> DualCameraCaptureApp:
     RobotTcpPoseResolver(robot_ip)             # local FK + ROS TF probe
     RegistrationSettings(...)                  # ICP refinement
     camera_in_hand_parent_trim_m=(x,y,z)/1000  # manual eye-in-hand tweak
     start()                                    # warm_up=30 per camera
     while True:
         _poll_display_panels -> cv2.imshow
         key:
           'q' -> return
           'c' -> capture_current_state:
                    - _capture_snapshot_state: eye-in-hand first (frame + robot pose
                                                back-to-back), then fixed
                    - For each camera:
                        - resolve_base_alignment (eye-to-hand: JSON only;
                                                  eye-in-hand: compose robot pose)
                        - stream.capture_pointcloud (Open3D RGBD + outlier removal)
                        - transform_pointcloud -> base frame
                    - _refine_base_clouds -> optional Open3D ICP (gated by
                                              fitness / RMSE / delta thresholds)
                    - _save_capture_products: per-cam .pcd + merged .pcd + metadata
```

### 3.5 `RobotTcpPoseResolver` dual-path design

This is the most subtle coupling between GUI/recorder and `datacollection_franky`:

```
capture_base_to_frame(frame_name)
  - _probe_joint_state_if_available
      -> datacollection_franky.pipeline_runtime.run_probe_robot_subprocess
         (returns q_current dict)
  - _lookup_base_to_frame_transform
      - _lookup_base_to_frame_transform_from_fk         (only for TCP_FRAME_NAME)
          - LocalRobotLinkFkResolver(fr3_link8).capture_base_to_link(q_current)
              uses headless PyBullet sim + reset_robot_joints + getLinkState
          - _resolve_link8_to_tcp_transform   caches link8 -> tcp (first time from ROS TF)
      - _lookup_base_to_frame_transform_from_tf         (other frames)
          - _run_ros_tf_probe:
              subprocess `/bin/bash -lc "source /opt/ros/jazzy/setup.bash &&
              source ~/franka_ros2_ws/install/setup.bash &&
              /usr/bin/python3 ros_tf_probe.py
                  --base-frame fr3_link0 --target-frame <fname>
                  --timeout-sec 3.0"`
              -> JSON {ok, translation, quat, required_topics_visible, ...}
          - If /tf has no publisher -> ensure_started launches in background:
              FrankaTfHelperThread -> ros2 launch franka_bringup franka.launch.py
                                      robot_type:=fr3 robot_ip:=<ip>
                                      joint_state_rate:=30
```

ROS2 jazzy is already in use, but only as an on-demand TF bridge. The main control logic still talks to FCI through `franky` in-process.

---

## 4. `recorder/`: One-click multi-device recording

### 4.1 Layout

```
recorder/
├── run.sh             # Verifies conda env == sonopet, then exec python3 launcher.py "$@"
├── launcher.py        # Enumerate devices, fork one subprocess per device
├── record_d405.py     # Single-D405 record loop
├── record_mic.py      # arecord wrapper
└── mic_test_dayton/   # Historical audio sample (audio.wav)
```

### 4.2 `launcher.py` flow

```
parse_args
  --base-dir (default recorder/)
  --camera-serials (empty -> all detected)
  --mic-devices    (empty -> all detected via arecord -l)
  --robot-ip (172.16.0.2)
  --fps (15)
  --no-camera / --no-mic
-> mkdir experiment_<YYYYMMDD>_<HHMMSS>/
-> discover_realsense_serials() via pyrealsense2
-> discover_capture_devices()   via arecord -l (regex plughw:C,D)
-> load_camera_role_by_serial(config/realsense_camera_roles.json)
-> build_camera_targets  (skip serials not registered in roles.json)
-> build_mic_targets     (accept all)
-> Fork per camera:
     python record_d405.py --out-dir <exp>/camera_<role>/
                            --serial <sn>
                            --camera-role <fixed|in-hand>
                            --robot-ip <ip>
                            --fps <fps>
-> Fork per microphone:
     python record_mic.py --out-dir <exp>/audio_<label>/
                          --device plughw:C,D
                          --device-name "<arecord -l line>"
-> Write launcher_meta.txt (start/stop/pid/fps/robot_ip)
-> Main loop polls 0.2 s:
     - Any child exits abnormally -> SIGINT all
     - SIGINT/SIGTERM received    -> SIGINT all
-> Join children (10 s grace, then SIGKILL); append end_unix/end_iso
```

### 4.3 `record_d405.py` single-camera flow

```
D405Recorder
  - role_config = build_camera_role(camera_role, serial)
  - robot_pose_resolver = RobotTcpPoseResolver(robot_ip)         # reuses GUI package
  - stream = RealSenseStream(...)                                # reuses GUI package
  - _start_stream_and_scan
      - stream.start + warm_up(30)
      - Capture 'start' frame -> save_pointcloud_scan('start', bundle)
          internal: capture_pointcloud -> resolve_base_alignment -> transform_pointcloud
                    -> write pointcloud_start.pcd + pointcloud_start_meta.json
  - _open_recording_outputs
      - rgb.avi (MJPG, fps, 848x480) + rgb_timestamps.csv
      - optional depth.mp4 (8-bit) + depth_timestamps.csv (--depth-video)
  - _record_loop: while not stop:
        poll_frame -> writer.write + csv row
        csv: frame_index, video_file, unix_time, rs_frame_timestamp_ms
  - On exit:
      - Close writer and csv
      - save_pointcloud_scan('end', last_bundle) -> pointcloud_end.pcd
      - stream.stop + robot_pose_resolver.close
```

Output per camera:

```
camera_<role>/
├── rgb.avi
├── rgb_timestamps.csv
├── pointcloud_start.pcd
├── pointcloud_start_meta.json
├── pointcloud_end.pcd
└── pointcloud_end_meta.json
```

### 4.4 `record_mic.py` single-mic flow

`exec arecord -D plughw:C,D -f S16_LE -r 48000 -c 1 -t wav audio.wav`. On SIGINT, forwards SIGINT to arecord, then writes `audio_meta.txt` (command, timestamps, exit code). Pure alsa-utils subprocess; no Python realtime callback.

---

## 5. Real-world end-to-end timeline

```
+-----------------------------------------------------------------------+
| Step A: scan pointcloud (eye-in-hand by hand, or fixed eye-to-hand)   |
|   $ cd recorder && ./run.sh   or   gui/realsense_capture/cli.py        |
|   -> experiment_<ts>/camera_camera-fixed/pointcloud_start.pcd          |
|      (in fr3_link0 frame)                                              |
+-----------------------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------------------+
| Step B: plan + execute (robot does the raster scan)                   |
|   $ python datacollection_franky/run_pipeline.py                      |
|         --pcd-path .../pointcloud_start.pcd                            |
|         --pcd-frame fr3_link0                                          |
|         --square-side 0.02 --line-spacing 0.002                        |
|         --raster-pattern unidirectional_retract                        |
|         --orientation-mode current                                     |
|         --robot-ip 172.16.0.2                                          |
|                                                                        |
|   1) Probe robot -> q_current                                          |
|   2) Open3D pick -> local cube -> raster + normals -> PoseWaypoint[]   |
|   3) PyBullet GUI -> draw raster -> user 'y' to preview                |
|   4) User 'EXECUTE' -> reconnect robot -> recompile IK                 |
|      -> per-segment dispatch to franky                                 |
|      Trajectory: current -> idle -> parking -> first ->                |
|                  raster -> retract -> workflow_start                   |
+-----------------------------------------------------------------------+
                          |
                          v
+-----------------------------------------------------------------------+
| Step C: in parallel with B, run recorder for synchronized data        |
|   $ cd recorder && ./run.sh   (D405 RGB video + mic wav)              |
|   or click "Start Recording All" in the GUI                            |
+-----------------------------------------------------------------------+
                          |
                          v
            Data lands under recorder/experiment_<ts>/
```

---

## 6. ROS2 jazzy refactor opportunity matrix

| Currently self-implemented | ROS2 jazzy replacement | Expected gain |
|---|---|---|
| `franky_backend.FrankyRobot` + `JointWaypointMotion` + custom segmented OTG | **`franka_ros2`** + `joint_trajectory_controller` + `MoveIt2` / `pilz_industrial_motion_planner` / `cartesian_motion_controller` | Delete `franky_backend.py`, most of `pipeline_hardware.py`, and the RT/affinity machinery in `pipeline_runtime.py` (ros2_control owns the 1 kHz loop) |
| `simWithPyBullet.BulletRobotSim` + custom IK / FK / TCP offset | **`MoveIt2`** (KDL/TracIK/Pick IK) + `robot_state_publisher` + `tf2` | All IK / FK / TCP transforms go through MoveIt + URDF. `tcp_config.py` collapses into a URDF fixed joint (already most of the way there) |
| `trajectory_compile.py` 5-segment + densify + IK + speed planning | **`MoveIt2` Cartesian path** (`compute_cartesian_path`) + `time_optimal_trajectory_generation` | Delete `densify.py`, `ik_compile.py`, and all hint-generation helpers |
| `preview.py` + `visualization.py` (PyBullet) | **`RViz2`** + MoveIt motion planning panel / `rqt_joint_trajectory_controller` | Stop maintaining two simulators |
| `pipeline_runtime.py` CPU partition + RT priority | `ros2_control` + `franka_hardware` handle RT; launch the controller node under `chrt` | Delete `util/cpu_partition.py`, `util/thread_affinity.py`, `robot_probe.py` |
| `gui/realsense_capture/stream.py` self-wrapped pyrealsense2 + filters + PCD | **`realsense2_camera`** node (`/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/camera/depth/color/points`) + `pointcloud_to_pcd` or `rosbag2` | ~300 lines removed |
| `gui/realsense_capture/robot_pose.py` ROS TF probe + Franka TF helper | Subscribe directly to `/tf` + `/tf_static` with `tf2_ros.Buffer.lookup_transform` (already the idea, just wrapped in subprocesses) | Delete `FrankaTfHelperThread`; run the Franka TF bringup as a systemd unit or launch file |
| `cloudpoint_wrapper.py` Open3D + PCA + raster + normals | Keep the business logic; rewrite as a ROS2 node that subscribes `/camera/depth/color/points` and publishes `geometry_msgs/PoseArray` | Cleaner topology |
| `recorder/launcher.py` + `record_d405.py` + `record_mic.py` | **`rosbag2`** (record many topics to `.mcap`/`.db3`) + `ros2 launch` + lifecycle nodes | Replace entire launcher with a `record.launch.py` |
| `gui/run_experiment.py` PyQt5 multi-sensor GUI | `rqt` plugins + `image_view` + `plotjuggler`; or keep Qt shell that only subscribes to ROS topics | UI survives, but no longer holds device handles |

### 6.1 A plausible target topology

```
ros2 launch sonopet bringup.launch.py
├── franka_ros2 (franka_hardware + ros2_control)
│     publishes: /joint_states, /tf, /tf_static
│     services:  /joint_trajectory_controller, /cartesian_motion_controller
├── realsense2_camera (eye-to-hand)
├── realsense2_camera (eye-in-hand)
├── robot_state_publisher (Sonopet TCP as a fixed joint, replaces tcp_config.py)
├── moveit2 (move_group + servo)
├── sonopet_planner_node       <- rewritten from datacollection_franky business logic
│      subscribes /camera/.../points, RViz interactive marker for center selection,
│      builds raster Cartesian path -> MoveIt compute_cartesian_path -> executes
├── sonopet_recorder           <- `rosbag2 record /camera/*/color /camera/*/depth
│                                  /joint_states /tf /audio/*`
└── usb_audio_publisher_node or alsa_audio_capture (ROS 2 driver)
```

### 6.2 Domain knowledge worth preserving in the rewrite

- `tcp_config.py`: "read `sonopet_tcp_joint` from URDF, then layer 5/10/26 mm trim" calibration flow (can move directly into URDF xacro params).
- `simulation/raster_patterns.py`: PCA tangent plane + boustrophedon / unidirectional_retract + endpoint-preserving downsample.
- `cloudpoint_wrapper.py`: sign-stable tangent / normal -> quaternion algorithm (`_estimate_tangents` + `_stabilize_tangents_with_global_reference` + `_stabilize_quaternion_signs_in_place`). Critical for avoiding IK-branch flips.
- `trajectory_compile.py`: strict zero velocity/acceleration at segment boundaries + `BOUNDARY_RAMP_POINTS=2` ramp (translates into TOTG boundary constraints under MoveIt).
- `config/realsense_camera_roles.json` + `config/camera_extrinsics/*.json` role-based extrinsics mapping.

### 6.3 Code blocks safe to delete outright

- `_debug_emit` / `_agent_debug_log` everywhere in `simulation/cloudpoint_wrapper.py` and `simulation/simWithPyBullet.py` (write to `.cursor/debug-*.log`).
- `datacollection_franky/runtime_affinity.py` (pure compatibility shim), `path_utils.py` (same).
- `simulation/_sonopet_fr3_tcp_configured.urdf` (`write_configured_fr3_urdf` is now passthrough).
- `datacollection_franky/preview.py` + `visualization.py` (replaced by RViz2).
- `simulation/RasterTraj_PyBullet.py` and `simulation/trajectories.py` (legacy entry points).
- `BulletRobotSim.install_bullet_robot_sim_extensions()` monkey-patching (turn into regular methods).
- `gui/realsense.py`, `gui/realsense_core.py`, `gui/usb_microphone.py`, `gui/camera.py` (small stand-alone previews superseded by `run_experiment.py` or future ROS2 nodes).

---

## 7. Recommendations for the rewrite (human-oriented programming)

1. **Slice by "node = one responsibility"**, replacing today's "split by file name":
   - `sonopet_planner` node: raster generation + Cartesian path requests, reuses `raster_patterns.py`.
   - `sonopet_executor` node: subscribes to planning results, talks to `joint_trajectory_controller`.
   - `sonopet_recorder` launch file: all `rosbag2 record`.
   - `sonopet_ui` node (optional Qt shell): only displays state, subscribes to topics, never holds devices.

2. **Drop every "self-managed RT / affinity / probe subprocess"** path. `franka_ros2` + `chrt` already give us mature realtime configuration.

3. **Unify frames** under one ROS naming scheme: `fr3_link0` (base), `fr3_link8`, `fr3_hand_tcp`, `sonopet_tcp`, `camera_color_optical_frame`. `constants.py` already uses these; `tcp_config.py` should fold its trim into URDF xacro params.

4. **Push the CLI down to ROS launch params**: the 15 flags in `cli.py` become `DeclareLaunchArgument` entries injected as parameters into `sonopet_planner`. The frozen dataclass defaults in `config.py` collapse into a single YAML configuration file.

5. **Unify log prefixes**: today we have eight scattered prefixes (`[INFO]`, `[d405]`, `[mic]`, `[launcher]`, `[RealSense]`, `[PointCloudWrapper]`, `[BulletRobotSim]`, `[IK:<stage>]`). Replace with `rclpy.get_logger("sonopet.<subsystem>")`.
