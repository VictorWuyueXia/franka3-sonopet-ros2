#!/usr/bin/env python3
"""
cloudpoint_wrapper.py

Point cloud wrapper for the PyBullet FR3 simulation flow:
point cloud -> manual point pick -> local square raster -> surface normals -> pose waypoints.
"""

from __future__ import annotations

import os
import json
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)
if _repo_root not in os.sys.path:
    os.sys.path.insert(0, _repo_root)

_eye_to_hand_config_path = os.path.join(
    _repo_root,
    "config",
    "camera_extrinsics",
    "fr3_eye_to_hand.json",
)

try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit(
        "open3d is not installed in this environment.\n"
        "Install it in your sonopet venv with:\n"
        "  pip install open3d\n"
    ) from e

from rob_control.pointcloud_process.process_scene import Scene
from simulation.raster_patterns import (
    RasterPatternSpec,
    build_local_tangent_square_corners,
    build_raster_points_from_selected_square,
)


def load_eye_to_hand_transform_spec() -> "TransformSpec":
    """Load the shared FR3 eye-to-hand transform from disk."""
    payload = json.loads(open(_eye_to_hand_config_path, "r", encoding="utf-8").read())
    return TransformSpec(
        translation_xyz=tuple(payload["translation_xyz"]),
        quaternion_xyzw=tuple(payload["quaternion_xyzw"]),
    )


def _debug_emit(hypothesis_id: str, location: str, message: str, data: dict, run_id: str) -> None:
    payload = {
        "sessionId": "dd3fc0",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
    }
    log_path = os.path.join(_repo_root, ".cursor", "debug-dd3fc0.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")
    f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    f.close()


def _agent_debug_log(hypothesis_id, location, message, data):
    payload = {
        "sessionId": "7bb972",
        "runId": "post-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "id": f"log_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
    }
    log_path = os.path.join(_repo_root, ".cursor", "debug-7bb972.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")
    f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    f.close()


@dataclass(frozen=True)
class TransformSpec:
    """Rigid transform from camera frame to FR3 base frame."""

    translation_xyz: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]

    def as_matrix4x4(self) -> np.ndarray:
        """Return homogeneous transform matrix."""
        qx, qy, qz, qw = self.quaternion_xyzw
        tx, ty, tz = self.translation_xyz
        rot = _quat_xyzw_to_rot_matrix(qx, qy, qz, qw)
        mat = np.eye(4, dtype=float)
        mat[:3, :3] = rot
        mat[:3, 3] = [tx, ty, tz]
        return mat


@dataclass(frozen=True)
class RasterSurfaceSpec:
    """Parameters for local square raster and surface analysis."""

    square_side_len_m: float
    line_spacing_m: float
    pointcloud_downsample_rate: int
    raster_pattern: str
    search_radius_m: float
    normal_projection_m: float
    min_points_for_normal: int
    use_surface_normal: bool
    tool_offset_m: float
    tool_normal_sign: float


@dataclass(frozen=True)
class PoseWaypoint:
    """One executable waypoint with position, orientation, and local surface normal."""

    xyz: Tuple[float, float, float]
    quat_xyzw: Tuple[float, float, float, float]
    normal_xyz: Tuple[float, float, float]


class PointCloudWrapper:
    """
    Wrapper that adapts pointcloud_process workflow into simulation-friendly outputs.
    """

    def __init__(self, pcd_path: str, vis_flag: bool, pcd_frame: str = "camera_optical") -> None:
        self.pcd_path = pcd_path
        self.vis_flag = vis_flag
        self.pcd_frame = str(pcd_frame)
        self._scene = self._build_scene_adapter(vis_flag)
        self.last_square_center: np.ndarray | None = None
        self.last_square_corners: np.ndarray | None = None

    def _build_scene_adapter(self, vis_flag: bool) -> Scene:
        """
        Build Scene safely and restore original working directory.
        Scene.__init__ changes cwd; restore immediately to avoid side effects.
        """
        cwd_before = os.getcwd()
        scene = Scene(vis_flag=vis_flag)
        os.chdir(cwd_before)
        return scene

    def default_fr3_base_to_camera_transform(self) -> TransformSpec:
        """
        Transform shared with rob_control/test.launch.py:
        parent fr3_link0 -> child camera_color_optical_frame.
        """
        return load_eye_to_hand_transform_spec()

    def load_pointcloud_camera_frame(self) -> o3d.geometry.PointCloud:
        """Load raw point cloud from file, preserving original sensor frame."""
        self._log(f"Loading point cloud from {self.pcd_path}")
        pcd = o3d.io.read_point_cloud(self.pcd_path)
        num_points = np.asarray(pcd.points).shape[0]
        if num_points == 0:
            raise RuntimeError(f"Point cloud is empty: {self.pcd_path}")
        self._log(f"Loaded point cloud points: {num_points}")
        return pcd

    def resolve_pointcloud_base_frame(self, pcd_input: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Normalize supported point-cloud input frames into FR3 base frame."""
        if self.pcd_frame == "camera_optical":
            tf_base_camera = self.default_fr3_base_to_camera_transform()
            return self.transform_camera_to_fr3_base(pcd_input, tf_base_camera)
        if self.pcd_frame == "fr3_link0":
            self._log("Input point cloud already uses FR3 base frame; skipping camera transform")
            return o3d.geometry.PointCloud(pcd_input)
        raise ValueError(f"Unsupported point-cloud frame: {self.pcd_frame}")

    def transform_camera_to_fr3_base(
        self,
        pcd_camera: o3d.geometry.PointCloud,
        transform_spec: TransformSpec,
        ) -> o3d.geometry.PointCloud:
        """Apply camera->base transform and return a copied point cloud."""
        mat = transform_spec.as_matrix4x4()
        pcd_base = o3d.geometry.PointCloud(pcd_camera)
        pcd_base.transform(mat)
        self._log("Applied camera->FR3-base transform to point cloud")
        return pcd_base

    def select_square_center_from_pointcloud(
        self,
        pcd_base: o3d.geometry.PointCloud,
        square_side_len_m: float,
        ) -> np.ndarray:
        """
        Pick a surface point and draw a PCA tangent-plane square in base frame.
        """
        self._log("Select a point in the point cloud (Shift + left click), then close the picker.")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        opt = vis.get_render_option()
        if opt is not None:
            opt.show_coordinate_frame = True
        vis.add_geometry(pcd_base)
        vis.run()
        vis.destroy_window()
        picked_indices = vis.get_picked_points()
        if not picked_indices:
            raise RuntimeError("No point was selected in point cloud UI.")
        pcd_points = np.asarray(pcd_base.points, dtype=float)
        center_arr = np.asarray(pcd_points[int(picked_indices[0])], dtype=float)
        self.last_square_corners = self.build_local_tangent_square_corners(
            selected_points=pcd_points,
            square_center_xyz=center_arr,
            square_side_len_m=float(square_side_len_m),
        )
        self._draw_local_tangent_square(
            pcd_base=pcd_base,
            square_corners=self.last_square_corners,
        )
        self._log(f"Selected square center: {center_arr.tolist()}")
        return center_arr

    def build_local_tangent_square_corners(
        self,
        selected_points: np.ndarray,
        square_center_xyz: np.ndarray,
        square_side_len_m: float,
        ) -> np.ndarray:
        """Return PCA tangent-plane square corners for visualization."""

        spec = RasterPatternSpec(
            square_center_xyz=(
                float(square_center_xyz[0]),
                float(square_center_xyz[1]),
                float(square_center_xyz[2]),
            ),
            square_side_len_m=float(square_side_len_m),
            line_spacing_m=max(float(square_side_len_m) / 4.0, 1e-6),
            pointcloud_downsample_rate=1,
            raster_pattern="unidirectional_retract",
        )
        return build_local_tangent_square_corners(selected_points, spec)

    def _draw_local_tangent_square(
        self,
        pcd_base: o3d.geometry.PointCloud,
        square_corners: np.ndarray,
        ) -> None:
        """Show the selected square on the same local tangent plane as raster generation."""

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.asarray(square_corners, dtype=float))
        line_set.lines = o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [3, 0]])
        line_set.paint_uniform_color([0, 1, 0])
        self._log("Drawing PCA tangent-plane selected square.")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_base)
        vis.add_geometry(line_set)
        vis.run()
        vis.destroy_window()

    def generate_raster_points_from_center(
        self,
        pcd_local: o3d.geometry.PointCloud,
        square_center_xyz: np.ndarray,
        surface_spec: RasterSurfaceSpec,
        ) -> Tuple[np.ndarray, o3d.geometry.LineSet, o3d.geometry.PointCloud]:
        """Build raster points from selected square point-cloud points."""

        selected_points = np.asarray(pcd_local.points, dtype=float)
        pattern_spec = RasterPatternSpec(
            square_center_xyz=(
                float(square_center_xyz[0]),
                float(square_center_xyz[1]),
                float(square_center_xyz[2]),
            ),
            square_side_len_m=float(surface_spec.square_side_len_m),
            line_spacing_m=float(surface_spec.line_spacing_m),
            pointcloud_downsample_rate=int(surface_spec.pointcloud_downsample_rate),
            raster_pattern=str(surface_spec.raster_pattern),
        )
        raster_points = build_raster_points_from_selected_square(selected_points, pattern_spec)
        self.last_square_corners = build_local_tangent_square_corners(selected_points, pattern_spec)
        raster_lines = _build_raster_path_lines(raster_points)
        raster_pcd = o3d.geometry.PointCloud()
        raster_pcd.points = o3d.utility.Vector3dVector(raster_points)
        raster_pcd.paint_uniform_color([0, 0, 1])
        self._log(
            "Generated raster points from selected square: "
            f"pattern={surface_spec.raster_pattern}, points={len(raster_points)}"
        )
        return raster_points, raster_lines, raster_pcd

    def estimate_surface_waypoints(
        self,
        raster_pcd_with_elevation: o3d.geometry.PointCloud,
        pcd_base: o3d.geometry.PointCloud,
        search_radius_m: float,
        projection_dist_m: float,
        min_points_for_normal: int,
        use_surface_normal: bool,
        ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.LineSet]:
        """
        Reuse existing local PCA normal estimator and waypoint projector.
        """
        waypoints_pcd, normal_lines = self._scene.calculate_surface_normals_and_waypoints(
            raster_pcd_with_elevation,
            pcd_base,
            search_radius=search_radius_m,
            projection_dist=projection_dist_m,
            min_points=min_points_for_normal,
            use_surface_normal=use_surface_normal,
        )
        num_waypoints = np.asarray(waypoints_pcd.points).shape[0]
        self._log(f"Estimated waypoints with normals: {num_waypoints}")
        return waypoints_pcd, normal_lines

    def crop_pointcloud_to_local_cube(
        self,
        pcd_base: o3d.geometry.PointCloud,
        cube_center_xyz: np.ndarray,
        cube_side_len_m: float,
        ) -> o3d.geometry.PointCloud:
        """
        Keep only points inside an axis-aligned 3D cube centered at cube_center_xyz.
        """
        pcd_np = np.asarray(pcd_base.points, dtype=float)
        if pcd_np.shape[0] == 0:
            raise RuntimeError("Point cloud is empty before local cube crop.")
        half_side = float(cube_side_len_m) / 2.0
        cx, cy, cz = float(cube_center_xyz[0]), float(cube_center_xyz[1]), float(cube_center_xyz[2])
        mask = (
            (pcd_np[:, 0] >= cx - half_side) & (pcd_np[:, 0] <= cx + half_side) &
            (pcd_np[:, 1] >= cy - half_side) & (pcd_np[:, 1] <= cy + half_side) &
            (pcd_np[:, 2] >= cz - half_side) & (pcd_np[:, 2] <= cz + half_side)
        )
        idx = np.where(mask)[0]
        if idx.shape[0] == 0:
            raise RuntimeError(
                f"No points found inside local cube (side={float(cube_side_len_m):.4f} m)."
            )
        cropped = pcd_base.select_by_index(idx.tolist())
        self._log(
            f"Local cube crop kept {idx.shape[0]}/{pcd_np.shape[0]} points "
            f"(side={float(cube_side_len_m):.4f} m)"
        )
        return cropped

    def build_pose_waypoints(
        self,
        waypoints_pcd: o3d.geometry.PointCloud,
        normal_lines: o3d.geometry.LineSet,
        tool_offset_m: float,
        tool_normal_sign: float,
        ) -> List[PoseWaypoint]:
        """
        Convert geometric waypoints into executable pose waypoints (position + quaternion).
        """
        waypoints = np.asarray(waypoints_pcd.points, dtype=float)
        line_points = np.asarray(normal_lines.points, dtype=float)
        normals = _extract_normals_from_line_pairs(line_points)
        if waypoints.shape[0] != normals.shape[0]:
            raise RuntimeError(
                f"Mismatch in waypoints/normals count: {waypoints.shape[0]} vs {normals.shape[0]}"
            )
        tangents = _estimate_tangents(waypoints)
        tangents = _stabilize_tangents_with_global_reference(tangents, normals)
        pose_waypoints: List[PoseWaypoint] = []
        for idx in range(waypoints.shape[0]):
            nvec = normals[idx] * float(tool_normal_sign)
            tvec = tangents[idx]
            quat = _quat_from_tangent_and_normal(tvec, nvec)
            pos = waypoints[idx] + nvec * float(tool_offset_m)
            pose_waypoints.append(
                PoseWaypoint(
                    xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
                    quat_xyzw=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
                    normal_xyz=(float(nvec[0]), float(nvec[1]), float(nvec[2])),
                )
            )
        _stabilize_quaternion_signs_in_place(pose_waypoints)
        self._log(f"Built simulation pose waypoints: {len(pose_waypoints)}")
        return pose_waypoints

    def get_sim_waypoints(
        self,
        surface_spec: RasterSurfaceSpec,
        square_center_override: Optional[Tuple[float, float, float]] = None,
    ) -> List[PoseWaypoint]:
        """
        One-shot pipeline from point cloud file to simulation pose waypoints.
        """
        #region agent log
        _debug_emit(
            "H12",
            "simulation/cloudpoint_wrapper.py:PointCloudWrapper.get_sim_waypoints:entry",
            "Entered get_sim_waypoints",
            {
                "pcd_path": self.pcd_path,
                "pcd_frame": self.pcd_frame,
                "square_center_override_used": square_center_override is not None,
            },
            "post-fix",
        )
        #endregion
        pcd_input = self.load_pointcloud_camera_frame()
        #region agent log
        _debug_emit(
            "H12",
            "simulation/cloudpoint_wrapper.py:PointCloudWrapper.get_sim_waypoints:after_load_pointcloud",
            "Loaded input pointcloud",
            {
                "pcd_points": int(np.asarray(pcd_input.points).shape[0]),
            },
            "post-fix",
        )
        #endregion
        pcd_base = self.resolve_pointcloud_base_frame(pcd_input)
        #region agent log
        _debug_emit(
            "H12",
            "simulation/cloudpoint_wrapper.py:PointCloudWrapper.get_sim_waypoints:after_transform",
            "Resolved pointcloud in base frame",
            {
                "pcd_points": int(np.asarray(pcd_base.points).shape[0]),
            },
            "post-fix",
        )
        #endregion
        #region agent log
        _agent_debug_log(
            "H5",
            "simulation/cloudpoint_wrapper.py:get_sim_waypoints:entry",
            "Waypoint pipeline entry for square center selection",
            {
                "override_provided": square_center_override is not None,
                "display": os.environ.get("DISPLAY"),
                "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
                "xdg_session_type": os.environ.get("XDG_SESSION_TYPE"),
                "pcd_points": int(np.asarray(pcd_base.points).shape[0]),
            },
        )
        #endregion
        if square_center_override is None:
            square_center = self.select_square_center_from_pointcloud(
                pcd_base,
                surface_spec.square_side_len_m,
            )
        else:
            square_center = np.asarray(square_center_override, dtype=float)
            self._log(f"Using manual square center override: {square_center.tolist()}")
            #region agent log
            _agent_debug_log(
                "H6",
                "simulation/cloudpoint_wrapper.py:get_sim_waypoints:manual_override",
                "Manual square center override branch used",
                {
                    "square_center": [
                        float(square_center[0]),
                        float(square_center[1]),
                        float(square_center[2]),
                    ],
                },
            )
            #endregion
        self.last_square_center = square_center
        #region agent log
        _debug_emit(
            "H12",
            "simulation/cloudpoint_wrapper.py:PointCloudWrapper.get_sim_waypoints:after_square_center",
            "Resolved square center",
            {
                "square_center": [
                    float(square_center[0]),
                    float(square_center[1]),
                    float(square_center[2]),
                ],
            },
            "post-fix",
        )
        #endregion
        pcd_local = self.crop_pointcloud_to_local_cube(
            pcd_base=pcd_base,
            cube_center_xyz=square_center,
            cube_side_len_m=surface_spec.square_side_len_m,
        )
        #region agent log
        _debug_emit(
            "H12",
            "simulation/cloudpoint_wrapper.py:PointCloudWrapper.get_sim_waypoints:after_local_crop",
            "Cropped local pointcloud",
            {
                "local_points": int(np.asarray(pcd_local.points).shape[0]),
            },
            "post-fix",
        )
        #endregion
        raster_points, _, raster_pcd = self.generate_raster_points_from_center(
            pcd_local,
            square_center,
            surface_spec,
        )
        #region agent log
        _debug_emit(
            "H12",
            "simulation/cloudpoint_wrapper.py:PointCloudWrapper.get_sim_waypoints:after_generate_raster",
            "Generated raster structures",
            {
                "raster_points_count": int(raster_points.shape[0]),
                "raster_pcd_points": int(np.asarray(raster_pcd.points).shape[0]),
            },
            "post-fix",
        )
        #endregion
        raster_surface = raster_pcd
        #region agent log
        _debug_emit(
            "H12",
            "simulation/cloudpoint_wrapper.py:PointCloudWrapper.get_sim_waypoints:after_raster_surface",
            "Prepared selected-point raster surface",
            {
                "raster_surface_points": int(np.asarray(raster_surface.points).shape[0]),
            },
            "post-fix",
        )
        #endregion
        waypoints_pcd, normal_lines = self.estimate_surface_waypoints(
            raster_surface,
            pcd_local,
            surface_spec.search_radius_m,
            surface_spec.normal_projection_m,
            surface_spec.min_points_for_normal,
            surface_spec.use_surface_normal,
        )
        #region agent log
        _debug_emit(
            "H12",
            "simulation/cloudpoint_wrapper.py:PointCloudWrapper.get_sim_waypoints:after_estimate_surface_waypoints",
            "Estimated surface waypoints and normals",
            {
                "waypoints_pcd_count": int(np.asarray(waypoints_pcd.points).shape[0]),
                "normal_line_points_count": int(np.asarray(normal_lines.points).shape[0]),
            },
            "post-fix",
        )
        #endregion
        pose_waypoints = self.build_pose_waypoints(
            waypoints_pcd,
            normal_lines,
            surface_spec.tool_offset_m,
            surface_spec.tool_normal_sign,
        )
        #region agent log
        _debug_emit(
            "H12",
            "simulation/cloudpoint_wrapper.py:PointCloudWrapper.get_sim_waypoints:before_return",
            "Built pose waypoints",
            {
                "pose_waypoint_count": int(len(pose_waypoints)),
            },
            "post-fix",
        )
        #endregion
        return pose_waypoints

    def _log(self, msg: str) -> None:
        print(f"[PointCloudWrapper] {msg}")


def _stabilize_quaternion_signs_in_place(pose_waypoints: List[PoseWaypoint]) -> None:
    """
    Keep quaternion sequence sign-consistent.
    q and -q represent same rotation, but sign flips can create large numeric jumps.
    """
    if len(pose_waypoints) < 2:
        return
    prev = np.asarray(pose_waypoints[0].quat_xyzw, dtype=float)
    for idx in range(1, len(pose_waypoints)):
        cur = np.asarray(pose_waypoints[idx].quat_xyzw, dtype=float)
        if float(np.dot(prev, cur)) < 0.0:
            cur = -cur
            pose_waypoints[idx] = PoseWaypoint(
                xyz=pose_waypoints[idx].xyz,
                quat_xyzw=(float(cur[0]), float(cur[1]), float(cur[2]), float(cur[3])),
                normal_xyz=pose_waypoints[idx].normal_xyz,
            )
        prev = cur


def _build_raster_path_lines(raster_points: np.ndarray) -> o3d.geometry.LineSet:
    """Create a line set that follows the final raster point order."""

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.asarray(raster_points, dtype=float))
    if raster_points.shape[0] > 1:
        lines = [[idx, idx + 1] for idx in range(raster_points.shape[0] - 1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])
    return line_set


def _quat_xyzw_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert quaternion (xyzw) to 3x3 rotation matrix."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n <= 0.0:
        raise ValueError("Quaternion norm is zero.")
    x = qx / n
    y = qy / n
    z = qz / n
    w = qw / n
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _extract_normals_from_line_pairs(line_points: np.ndarray) -> np.ndarray:
    """Recover normals from line pairs [start, end] encoded in line points."""
    if line_points.shape[0] % 2 != 0:
        raise RuntimeError("Line points count is not even; cannot decode normal pairs.")
    normals: List[np.ndarray] = []
    for idx in range(0, line_points.shape[0], 2):
        p0 = line_points[idx]
        p1 = line_points[idx + 1]
        vec = p1 - p0
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-10:
            continue
        normals.append(vec / norm)
    if not normals:
        raise RuntimeError("No valid normal vectors decoded from line geometry.")
    return np.asarray(normals, dtype=float)


def _estimate_tangents(points_xyz: np.ndarray) -> np.ndarray:
    """Estimate tangent direction from neighboring points."""
    n = points_xyz.shape[0]
    tangents = np.zeros_like(points_xyz)
    for idx in range(n):
        if n == 1:
            diff = np.array([1.0, 0.0, 0.0], dtype=float)
        elif idx == 0:
            diff = points_xyz[1] - points_xyz[0]
        elif idx == n - 1:
            diff = points_xyz[n - 1] - points_xyz[n - 2]
        else:
            diff = points_xyz[idx + 1] - points_xyz[idx - 1]
        mag = float(np.linalg.norm(diff))
        if mag <= 1e-10:
            tangents[idx] = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            tangents[idx] = diff / mag
        # Keep tangent direction sign-consistent to avoid 180-deg orientation flips,
        # which can cause large IK branch jumps between neighboring waypoints.
        if idx > 0:
            prev = tangents[idx - 1]
            cur = tangents[idx]
            if float(np.dot(prev, cur)) < 0.0:
                tangents[idx] = -cur
    return tangents


def _stabilize_tangents_with_global_reference(
    tangents_xyz: np.ndarray,
    normals_xyz: np.ndarray,
) -> np.ndarray:
    """
    Stabilize tangent directions using one global reference tangent.
    This avoids sharp orientation flips at raster turning points.
    """
    if tangents_xyz.shape[0] == 0:
        return tangents_xyz
    ref = _normalize(np.asarray(tangents_xyz[0], dtype=float))
    out = np.zeros_like(tangents_xyz)
    for idx in range(tangents_xyz.shape[0]):
        n = _normalize(np.asarray(normals_xyz[idx], dtype=float))
        # Project global reference to the local tangent plane orthogonal to normal.
        t = ref - float(np.dot(ref, n)) * n
        if float(np.linalg.norm(t)) <= 1e-10:
            # Fallback axis if reference is near parallel to normal.
            fallback = np.array([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(fallback, n))) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=float)
            t = fallback - float(np.dot(fallback, n)) * n
        t = _normalize(t)
        if idx > 0 and float(np.dot(out[idx - 1], t)) < 0.0:
            t = -t
        out[idx] = t
    return out


def _quat_from_tangent_and_normal(tangent_xyz: Sequence[float], normal_xyz: Sequence[float]) -> np.ndarray:
    """
    Build quaternion (xyzw) from local frame:
      x-axis aligns to tangent
      z-axis aligns to normal
      y-axis is computed by right-hand rule
    """
    x = _normalize(np.asarray(tangent_xyz, dtype=float))
    z = _normalize(np.asarray(normal_xyz, dtype=float))
    x = x - np.dot(x, z) * z
    if np.linalg.norm(x) <= 1e-10:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(ref, z)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        x = _normalize(ref - np.dot(ref, z) * z)
    else:
        x = _normalize(x)
    y = _normalize(np.cross(z, x))
    rot = np.column_stack((x, y, z))
    return _rotation_matrix_to_quat_xyzw(rot)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-10:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _rotation_matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion in xyzw order."""
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=float)
    return _normalize(quat)
