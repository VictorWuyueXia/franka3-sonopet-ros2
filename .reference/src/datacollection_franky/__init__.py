"""franky_joint_pipeline_v1

Self-contained pipeline that:
- Builds pose waypoints (e.g., from point cloud).
- Densifies in Cartesian space.
- Solves IK in PyBullet once to produce a joint waypoint list.
- Runs a PyBullet preview.
- Optionally executes the same joint waypoints on hardware using Franky.
"""
