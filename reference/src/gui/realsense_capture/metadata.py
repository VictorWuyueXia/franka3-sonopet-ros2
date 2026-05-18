from __future__ import annotations

from pathlib import Path

from .pointcloud.storage import PointCloudStorage
from .pointcloud.transforms import TransformSpec
from .roles import ActiveCamera
from .stream import log_status


def build_camera_metadata(
    active_camera: ActiveCamera,
    scan_path: Path,
    scan_point_count: int,
    base_transform: TransformSpec | None,
    skip_reason: str | None,
    storage: PointCloudStorage,
) -> dict:
    relative_scan_path = storage.relative_to_repo(scan_path)
    return {
        "serial": str(active_camera.device_info["serial"]),
        "capture_role": active_camera.role_config.camera_name,
        "scan_pcd_saved": True,
        "scan_pcd_path": relative_scan_path,
        "scan_pcd_frame": "fr3_link0",
        "scan_point_count": int(scan_point_count),
        "raw_pcd_saved": True,
        "raw_pcd_path": relative_scan_path,
        "raw_pcd_frame": "fr3_link0",
        "raw_point_count": int(scan_point_count),
        "base_aligned_pcd_path": relative_scan_path,
        "merged_into_base": bool(base_transform is not None),
        "base_transform": None if base_transform is None else base_transform.to_dict(),
        "skip_reason": skip_reason,
    }


def build_failed_raw_metadata(active_camera: ActiveCamera, failure_reason: str) -> dict:
    """Metadata when depth/color did not produce a base-frame point cloud."""

    return {
        "serial": str(active_camera.device_info["serial"]),
        "capture_role": active_camera.role_config.camera_name,
        "scan_pcd_saved": False,
        "scan_pcd_path": None,
        "scan_failure_reason": str(failure_reason),
        "scan_pcd_frame": None,
        "scan_point_count": 0,
        "raw_pcd_saved": False,
        "raw_pcd_path": None,
        "raw_failure_reason": str(failure_reason),
        "raw_pcd_frame": None,
        "raw_point_count": 0,
        "base_aligned_pcd_path": None,
        "merged_into_base": False,
        "base_transform": None,
        "skip_reason": None,
    }


def log_capture_terminal_summary(metadata: dict, merged_written: bool, merged_path: str | None) -> None:
    """Print a clear English summary for saved scans and merge outcome."""

    log_status("========== Capture summary ==========")
    cameras = metadata.get("cameras") or {}
    merge_summary = metadata.get("merge_summary") or {}
    if merged_written and merged_path:
        kind_key = str(merge_summary.get("kind") or "merged")
        kind_label = {
            "full": "both cameras in base frame",
            "partial_two_scan_one_base": "partial (only one camera transformed to base frame)",
            "partial_two_raw_one_base": "partial (only one camera transformed to base frame)",
            "single_camera": "single camera only",
        }.get(kind_key, kind_key)
        log_status(f"Base-frame merge: YES — wrote {merged_path} ({kind_label})")
    else:
        log_status("Base-frame merge: NO — point clouds were not merged into one base-frame file")
        for line in merge_summary.get("not_merged_reasons") or []:
            log_status(f"  Merge blocked: {line}")

    log_status("Base-frame point cloud scans (on disk):")
    any_scan = False
    for role in sorted(cameras.keys()):
        entry = cameras[role]
        if entry.get("scan_pcd_saved") or entry.get("raw_pcd_saved"):
            any_scan = True
            rel = entry.get("scan_pcd_path") or entry.get("raw_pcd_path") or ""
            n = entry.get("scan_point_count", entry.get("raw_point_count", 0))
            log_status(f"  {role}: SAVED — {rel} ({int(n)} points)")
        else:
            reason = entry.get("scan_failure_reason") or entry.get("raw_failure_reason") or "unknown"
            log_status(f"  {role}: NOT SAVED — {reason}")
    if not any_scan and not cameras:
        log_status("  (no camera entries)")

    log_status("Base-frame inclusion (for merge):")
    for role in sorted(cameras.keys()):
        entry = cameras[role]
        if not (entry.get("scan_pcd_saved") or entry.get("raw_pcd_saved")):
            log_status(f"  {role}: skipped (no base-frame scan file)")
            continue
        if entry.get("merged_into_base"):
            log_status(f"  {role}: included in merge")
        else:
            sr = entry.get("skip_reason") or "unknown"
            log_status(f"  {role}: NOT merged into base — {sr}")
    registration = metadata.get("registration") or {}
    if registration.get("attempted"):
        if registration.get("applied"):
            log_status(
                "Open3D refinement: APPLIED — "
                f"fitness={float(registration.get('fitness', 0.0)):.4f}, "
                f"inlier_rmse={float(registration.get('inlier_rmse', 0.0)):.6f}"
            )
        else:
            log_status(f"Open3D refinement: SKIPPED — {registration.get('reason') or 'unknown'}")
    log_status("======================================")


def compute_merge_summary_metadata(metadata: dict, merged_written: bool) -> dict:
    """Structured merge outcome for JSON and for the terminal summary block."""

    cameras = metadata.get("cameras") or {}
    used_roles = list(metadata.get("used_cameras") or [])
    scan_ok_roles = [
        role
        for role, entry in cameras.items()
        if entry.get("scan_pcd_saved") or entry.get("raw_pcd_saved")
    ]
    n_scans = len(scan_ok_roles)
    n_used = len(used_roles)
    if not merged_written:
        return _build_not_merged_summary(cameras, scan_ok_roles, n_scans)
    if n_used >= 2:
        kind = "full"
    elif n_scans >= 2 and n_used == 1:
        kind = "partial_two_scan_one_base"
    else:
        kind = "single_camera"
    return {"kind": kind, "not_merged_reasons": []}


def _build_not_merged_summary(cameras: dict, scan_ok_roles: list[str], n_scans: int) -> dict:
    reasons = []
    if n_scans == 0:
        reasons.append("No base-frame scans were saved; nothing to merge.")
        return {"kind": "none", "not_merged_reasons": reasons}
    for role in scan_ok_roles:
        entry = cameras[role]
        if not entry.get("merged_into_base"):
            skip_reason = (
                entry.get("skip_reason")
                or entry.get("scan_failure_reason")
                or "cannot transform to base frame"
            )
            reasons.append(f"{role}: {skip_reason}")
    return {"kind": "none", "not_merged_reasons": reasons}
