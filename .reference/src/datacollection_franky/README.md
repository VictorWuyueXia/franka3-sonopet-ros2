## datacollection_franky

This directory contains the Sonopet data-collection pipeline with a stage-oriented structure.

### Main Entry

- Script entry: `datacollection_franky/run_pipeline.py`
- Pipeline orchestrator: `datacollection_franky/pipeline.py`

### Runtime Stages

`pipeline.py` now keeps `main()` thin and delegates to short stage functions:

1. Load simulation environment and modules.
2. Initialize CPU affinity runtime state.
3. Acquire initial robot state with subprocess probe.
4. Build point-cloud-based pose waypoints.
5. Connect PyBullet and prepare orientation.
6. Compile offline trajectory.
7. Preview in PyBullet.
8. Execute hardware trajectory if allowed.

### CPU Affinity Structure

CPU affinity code is separated from core task logic:

- CPU partition policy: `util/cpu_partition.py`
- Thread affinity and RT policy helpers: `util/thread_affinity.py`

The legacy file `runtime_affinity.py` is kept as a compatibility wrapper.

### Sim And URDF Integration

- Simulation path discovery: `sim_paths.py`
- TCP-configured URDF handling: `urdf_tcp.py`
- Compatibility wrappers: `path_utils.py`

### CLI Surface (Current)

The user-facing CLI is intentionally narrow and focused in `cli.py`:

- `--robot-ip`
- `--preview-only`
- `--pcd-path`
- `--square-side`
- `--square-center`
- `--line-spacing`
- `--pointcloud-downsample-rate`
- `--raster-pattern`
- `--raster-z-offset`
- `--dynamics-scale`
- `--sim-time-scale`
- `--orientation-mode`
- `--final-return-mode`
- `--preset-orientation`
- `--preset-rpy-deg`
- `--preset-quat-xyzw`

Most runtime toggles and low-level controls were retired from CLI and moved to configuration defaults.
The final return defaults to `workflow_start`; use `--final-return-mode idle` to restore the previous idle-pose return.
Use `--orientation-mode preset --preset-orientation idle` to keep waypoint orientation equal to the idle TCP orientation.

### Configuration

Primary defaults are centralized in `config.py`:

- `RuntimePolicyConfig`
- `PointCloudDefaults`
- `PreviewDefaults`
- `InputDefaults`
- `PipelineTiming`, `PathDensifyConfig`, `StageConfig`, `FrankyConfig`

### Notes

- This package expects Linux affinity APIs (`os.sched_*`) for runtime thread policy.
- In environments without runtime dependencies (for example `pybullet`), import-time or run-time errors are expected.
- `JOINT_NAMES` is treated as an internal canonical FR3 joint order constant.

