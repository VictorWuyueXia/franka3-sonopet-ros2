from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from .constants import BASE_FRAME_NAME, CAMERA_FIXED_NAME, CAMERA_IN_HAND_NAME, REPO_ROOT, WINDOW_NAME
from .metadata import (
    build_camera_metadata,
    build_failed_raw_metadata,
    compute_merge_summary_metadata,
    log_capture_terminal_summary,
)
from .pointcloud.registration import (
    REFINEMENT_MODE_NEVER,
    RegistrationSettings,
    refine_pointcloud_alignment,
)
from .pointcloud.alignment import cache_robot_pose_for_role, resolve_base_alignment
from .pointcloud.storage import PointCloudStorage
from .pointcloud.transforms import (
    TransformSpec,
    merge_pointclouds,
    transform_pointcloud,
)
from .preview import build_display_panel, show_pointcloud, write_pointcloud
from .robot_pose import RobotCapturePose, RobotTcpPoseResolver
from .roles import ActiveCamera, CameraRoleConfig
from .stream import (
    DEFAULT_MERGE_DISTANCE_TRUNCATION_M,
    DEFAULT_WARMUP_FRAMES,
    FrameBundle,
    log_status,
    require_cv2,
)


@dataclass
class CameraCaptureResult:
    role_name: str
    raw_cloud: object | None
    base_cloud: object | None
    metadata_entry: dict
    robot_capture: RobotCapturePose | None


@dataclass(frozen=True)
class SnapshotCaptureState:
    frame_bundles_by_role: dict[str, FrameBundle | None]
    robot_capture_by_frame: dict[str, RobotCapturePose | None]


class DualCameraCaptureApp:
    """Default RealSense entrypoint: dual-camera capture with optional graceful degradation."""

    def __init__(
        self,
        active_cameras: list[ActiveCamera],
        robot_pose_resolver: RobotTcpPoseResolver,
        registration_settings: RegistrationSettings,
        camera_in_hand_parent_trim_m: tuple[float, float, float],
        camera_in_hand_parent_yaw_trim_deg: float,
    ) -> None:
        self.active_cameras = list(active_cameras)
        self.robot_pose_resolver = robot_pose_resolver
        self.registration_settings = registration_settings
        self.camera_in_hand_parent_trim_m = (
            float(camera_in_hand_parent_trim_m[0]),
            float(camera_in_hand_parent_trim_m[1]),
            float(camera_in_hand_parent_trim_m[2]),
        )
        self.camera_in_hand_parent_yaw_trim_deg = float(camera_in_hand_parent_yaw_trim_deg)
        self.storage = PointCloudStorage(REPO_ROOT)
        self.storage.ensure_directories()

    def start(self) -> None:
        started_cameras = []
        try:
            for active_camera in self.active_cameras:
                active_camera.stream.start()
                started_cameras.append(active_camera)
            for active_camera in self.active_cameras:
                active_camera.stream.warm_up(DEFAULT_WARMUP_FRAMES)
        except Exception:
            self._stop_started_streams(started_cameras)
            raise
        camera_names = ", ".join(active_camera.role_config.camera_name for active_camera in self.active_cameras)
        log_status(f"Streaming started for: {camera_names}")

    def stop(self) -> None:
        for active_camera in self.active_cameras:
            active_camera.stream.stop()
        self.robot_pose_resolver.close()
        require_cv2().destroyAllWindows()

    def run(self) -> None:
        cv2 = require_cv2()
        try:
            self.start()
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1600, 540)
            while True:
                display_panels = self._poll_display_panels()
                if not display_panels:
                    continue
                cv2.imshow(WINDOW_NAME, np.hstack(display_panels))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    log_status("Quitting stream.")
                    return
                if key == ord("c"):
                    self.capture_current_state()
        except KeyboardInterrupt:
            log_status("Interrupted by keyboard.")
        finally:
            self.stop()

    def _stop_started_streams(self, started_cameras: list[ActiveCamera]) -> None:
        for active_camera in reversed(started_cameras):
            active_camera.stream.stop()

    def capture_current_state(self) -> None:
        timestamp_s = int(time.time())
        metadata = self._build_capture_metadata(timestamp_s)
        snapshot_state = self._capture_snapshot_state()
        base_clouds_by_role = {}
        raw_clouds = []
        for active_camera in self.active_cameras:
            frame_bundle = snapshot_state.frame_bundles_by_role.get(active_camera.role_config.camera_name)
            result = self._capture_active_camera(
                active_camera,
                timestamp_s,
                snapshot_state.robot_capture_by_frame,
                frame_bundle,
            )
            metadata["cameras"][result.role_name] = result.metadata_entry
            self._collect_capture_result(result, metadata, base_clouds_by_role, raw_clouds)
        merge_clouds_by_role = self._refine_base_clouds(base_clouds_by_role, metadata)
        self._save_capture_products(timestamp_s, merge_clouds_by_role, raw_clouds, metadata)

    def _capture_snapshot_state(self) -> SnapshotCaptureState:
        frame_bundles_by_role = {}
        robot_capture_by_frame = {}
        active_cameras = self._order_snapshot_cameras()
        for active_camera in active_cameras:
            role_config = active_camera.role_config
            role_name = role_config.camera_name
            frame_bundles_by_role[role_name] = active_camera.stream.poll_frame()
            cache_robot_pose_for_role(role_config, self.robot_pose_resolver, robot_capture_by_frame)
        return SnapshotCaptureState(
            frame_bundles_by_role=frame_bundles_by_role,
            robot_capture_by_frame=robot_capture_by_frame,
        )

    def _order_snapshot_cameras(self) -> list[ActiveCamera]:
        # Capture the moving eye-in-hand frame first so its bundle and robot pose are sampled back-to-back.
        return sorted(self.active_cameras, key=lambda active_camera: 0 if active_camera.role_config.requires_robot_pose else 1)

    def _build_capture_metadata(self, timestamp_s: int) -> dict:
        return {
            "timestamp_s": int(timestamp_s),
            "merged_pcd_frame": BASE_FRAME_NAME,
            "merged_pcd_path": None,
            "used_cameras": [],
            "robot_capture": None,
            "cameras": {},
            "registration": {
                "attempted": False,
                "applied": False,
                "reason": None,
                "fitness": None,
                "inlier_rmse": None,
                "transformation_matrix": None,
                "settings": self._build_registration_settings_metadata(),
            },
            "stitching_adjustments": {
                "camera_in_hand_parent_trim_m": [
                    float(self.camera_in_hand_parent_trim_m[0]),
                    float(self.camera_in_hand_parent_trim_m[1]),
                    float(self.camera_in_hand_parent_trim_m[2]),
                ],
                "camera_in_hand_parent_yaw_trim_deg": float(self.camera_in_hand_parent_yaw_trim_deg),
            },
        }

    def _build_registration_settings_metadata(self) -> dict:
        return {
            "refinement_mode": str(self.registration_settings.refinement_mode),
            "refinement_dofs": str(self.registration_settings.refinement_dofs),
            "min_fitness": float(self.registration_settings.min_fitness),
            "max_inlier_rmse_m": float(self.registration_settings.max_inlier_rmse_m),
            "max_translation_delta_m": float(self.registration_settings.max_translation_delta_m),
            "max_rotation_delta_deg": float(self.registration_settings.max_rotation_delta_deg),
            "overlap_margin_m": float(self.registration_settings.overlap_margin_m),
        }

    def _capture_active_camera(
        self,
        active_camera: ActiveCamera,
        timestamp_s: int,
        robot_capture_by_frame: dict,
        frame_bundle: FrameBundle | None,
    ) -> CameraCaptureResult:
        role_name = active_camera.role_config.camera_name
        base_transform, skip_reason, robot_capture = self._resolve_base_transform(
            active_camera.role_config,
            robot_capture_by_frame,
        )
        raw_cloud = active_camera.stream.capture_pointcloud(
            frame_bundle,
            merge_distance_truncation_m=float(DEFAULT_MERGE_DISTANCE_TRUNCATION_M),
        )

     
  
        if raw_cloud is None or len(raw_cloud.points) <= 0:
            return self._build_failed_capture_result(active_camera, role_name)
        return self._build_success_capture_result(
            active_camera,
            role_name,
            raw_cloud,
            base_transform,
            skip_reason,
            robot_capture,
            timestamp_s,
        )

    def _build_failed_capture_result(
        self,
        active_camera: ActiveCamera,
        role_name: str,
        fail_reason: str = "No valid point cloud from depth/color (empty or failed capture)",
    ) -> CameraCaptureResult:
        log_status(f"{role_name}: base-frame point cloud NOT saved — {fail_reason}")
        return CameraCaptureResult(
            role_name=role_name,
            raw_cloud=None,
            base_cloud=None,
            metadata_entry=build_failed_raw_metadata(active_camera, fail_reason),
            robot_capture=None,
        )

    def _build_success_capture_result(
        self,
        active_camera: ActiveCamera,
        role_name: str,
        raw_cloud,
        base_transform: TransformSpec | None,
        skip_reason: str | None,
        robot_capture: RobotCapturePose | None,
        timestamp_s: int,
    ) -> CameraCaptureResult:
        if base_transform is None:
            log_status(f"{role_name}: cannot merge into {BASE_FRAME_NAME} — {skip_reason or 'unknown'}")
            return self._build_failed_capture_result(
                active_camera,
                role_name,
                fail_reason=skip_reason or f"Missing transform into {BASE_FRAME_NAME}",
            )
        base_cloud = transform_pointcloud(raw_cloud, base_transform)
        scan_path = self.storage.build_scan_path(role_name, timestamp_s)
        write_pointcloud(scan_path, base_cloud)
        log_status(f"{role_name}: base-frame point cloud SAVED — {scan_path}")
        metadata_entry = build_camera_metadata(
            active_camera=active_camera,
            scan_path=scan_path,
            scan_point_count=len(base_cloud.points),
            base_transform=base_transform,
            skip_reason=skip_reason,
            storage=self.storage,
        )
        return CameraCaptureResult(role_name, raw_cloud, base_cloud, metadata_entry, robot_capture)

    def _collect_capture_result(
        self,
        result: CameraCaptureResult,
        metadata: dict,
        base_clouds_by_role: dict,
        raw_clouds: list,
    ) -> None:
        if result.raw_cloud is not None:
            raw_clouds.append(result.raw_cloud)
        if result.base_cloud is not None:
            base_clouds_by_role[result.role_name] = result.base_cloud
            metadata["used_cameras"].append(result.role_name)
        if result.robot_capture is not None and metadata["robot_capture"] is None:
            metadata["robot_capture"] = {
                "robot_ip": result.robot_capture.robot_ip,
                "q_current": result.robot_capture.q_current,
                "base_to_parent_frame": result.robot_capture.base_to_tcp.to_dict(),
            }

    def _poll_display_panels(self) -> list[np.ndarray]:
        panels = []
        for active_camera in self.active_cameras:
            bundle = active_camera.stream.poll_frame()
            if bundle is None:
                continue
            panels.append(
                build_display_panel(
                    camera_name=active_camera.role_config.camera_name,
                    serial_number=str(active_camera.device_info["serial"]),
                    color_image=bundle.color_image,
                    depth_image=bundle.depth_image,
                )
            )
        return panels

    def _resolve_base_transform(
        self,
        role_config: CameraRoleConfig,
        robot_capture_by_frame: dict,
    ) -> tuple[TransformSpec | None, str | None, RobotCapturePose | None]:
        result = resolve_base_alignment(
            role_config=role_config,
            robot_pose_resolver=self.robot_pose_resolver,
            robot_capture_by_frame=robot_capture_by_frame,
            camera_in_hand_parent_trim_m=self.camera_in_hand_parent_trim_m,
            camera_in_hand_parent_yaw_trim_deg=self.camera_in_hand_parent_yaw_trim_deg,
        )
        return result.base_transform, result.skip_reason, result.robot_capture

    def _refine_base_clouds(self, base_clouds_by_role: dict, metadata: dict) -> dict:
        merge_clouds_by_role = dict(base_clouds_by_role)
        fixed_cloud = base_clouds_by_role.get(CAMERA_FIXED_NAME)
        in_hand_cloud = base_clouds_by_role.get(CAMERA_IN_HAND_NAME)
        if fixed_cloud is None or in_hand_cloud is None:
            metadata["registration"]["reason"] = "Need both camera-fixed and camera-in-hand base-aligned clouds."
            return merge_clouds_by_role
        if self.registration_settings.refinement_mode == REFINEMENT_MODE_NEVER:
            metadata["registration"]["reason"] = "ICP refinement disabled because refinement mode is never."
            return merge_clouds_by_role
        metadata["registration"]["attempted"] = True
        result = refine_pointcloud_alignment(
            source_cloud=in_hand_cloud,
            target_cloud=fixed_cloud,
            settings=self.registration_settings,
        )
        self._update_registration_metadata(metadata, result)
        if not result.applied:
            return merge_clouds_by_role
        merge_clouds_by_role[CAMERA_IN_HAND_NAME] = result.refined_source
        return merge_clouds_by_role

    def _update_registration_metadata(self, metadata: dict, result) -> None:
        metadata["registration"]["applied"] = bool(result.applied)
        metadata["registration"]["reason"] = result.reason
        metadata["registration"]["fitness"] = float(result.fitness)
        metadata["registration"]["inlier_rmse"] = float(result.inlier_rmse)
        metadata["registration"]["transformation_matrix"] = [
            [float(value) for value in row]
            for row in result.transformation_matrix.tolist()
        ]

    def _save_capture_products(self, timestamp_s: int, base_clouds_by_role: dict, raw_clouds: list, metadata: dict) -> None:
        merged_cloud, merged_path = self._write_merged_cloud(timestamp_s, base_clouds_by_role, metadata)
        merged_written = bool(base_clouds_by_role)
        metadata["merge_summary"] = compute_merge_summary_metadata(metadata, merged_written)
        metadata_path = self._write_metadata_file(timestamp_s, metadata)
        log_status(f"Saved capture metadata to {metadata_path}")
        merged_path_str = None if merged_path is None else str(merged_path)
        log_capture_terminal_summary(metadata, merged_written, merged_path_str)
        self._show_capture_result(merged_cloud, raw_clouds)

    def _write_merged_cloud(self, timestamp_s: int, base_clouds_by_role: dict, metadata: dict):
        base_clouds = list(base_clouds_by_role.values())
        if not base_clouds:
            return None, None
        merged_cloud = merge_pointclouds(base_clouds)
        merged_path = self.storage.build_stitched_path(timestamp_s)
        write_pointcloud(merged_path, merged_cloud)
        metadata["merged_pcd_path"] = self.storage.relative_to_repo(merged_path)
        log_status(f"Saved merged base-frame point cloud to {merged_path}")
        return merged_cloud, merged_path

    def _write_metadata_file(self, timestamp_s: int, metadata: dict):
        metadata_path = self.storage.build_metadata_path(timestamp_s)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return metadata_path

    def _show_capture_result(self, merged_cloud, raw_clouds: list) -> None:
        if merged_cloud is not None:
            show_pointcloud(merged_cloud, "Close the merged point-cloud viewer to resume streaming.")
            return
        if raw_clouds:
            show_pointcloud(raw_clouds[0], "Close the raw point-cloud viewer to resume streaming.")
