from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REFINEMENT_MODE_ALWAYS = "always"
REFINEMENT_MODE_NEVER = "never"
REFINEMENT_MODE_AUTO = "auto"
REFINEMENT_MODES = (
    REFINEMENT_MODE_ALWAYS,
    REFINEMENT_MODE_NEVER,
    REFINEMENT_MODE_AUTO,
)
REFINEMENT_DOFS_Z = "z"
REFINEMENT_DOFS_TRANSLATION = "translation"
REFINEMENT_DOFS_FULL = "full"
REFINEMENT_DOFS = (
    REFINEMENT_DOFS_Z,
    REFINEMENT_DOFS_TRANSLATION,
    REFINEMENT_DOFS_FULL,
)

DEFAULT_REGISTRATION_MIN_FITNESS = 0.28
DEFAULT_REGISTRATION_MAX_INLIER_RMSE_M = 0.004
DEFAULT_REGISTRATION_MAX_TRANSLATION_DELTA_M = 0.2
DEFAULT_REGISTRATION_MAX_ROTATION_DELTA_DEG = 3.0
DEFAULT_REGISTRATION_OVERLAP_MARGIN_M = 0.03
DEFAULT_REGISTRATION_CROP_MIN_POINTS = 200


@dataclass(frozen=True)
class RegistrationSettings:
    """Tuning knobs that keep refinement conservative around a good TF prior."""

    refinement_mode: str
    refinement_dofs: str
    min_fitness: float
    max_inlier_rmse_m: float
    max_translation_delta_m: float
    max_rotation_delta_deg: float
    overlap_margin_m: float


@dataclass(frozen=True)
class RegistrationResult:
    """Open3D ICP refinement result for one moving cloud against one fixed cloud."""

    refined_source: object
    transformation_matrix: np.ndarray
    fitness: float
    inlier_rmse: float
    applied: bool
    reason: str | None


def build_default_registration_settings() -> RegistrationSettings:
    return RegistrationSettings(
        refinement_mode=REFINEMENT_MODE_NEVER,
        refinement_dofs=REFINEMENT_DOFS_Z,
        min_fitness=float(DEFAULT_REGISTRATION_MIN_FITNESS),
        max_inlier_rmse_m=float(DEFAULT_REGISTRATION_MAX_INLIER_RMSE_M),
        max_translation_delta_m=float(DEFAULT_REGISTRATION_MAX_TRANSLATION_DELTA_M),
        max_rotation_delta_deg=float(DEFAULT_REGISTRATION_MAX_ROTATION_DELTA_DEG),
        overlap_margin_m=float(DEFAULT_REGISTRATION_OVERLAP_MARGIN_M),
    )


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("open3d is required for point-cloud registration.") from exc
    return o3d


def _copy_cloud(pointcloud):
    o3d = _require_open3d()
    return o3d.geometry.PointCloud(pointcloud)


def _to_numpy_bounds(bbox) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(bbox.get_min_bound(), dtype=float), np.asarray(bbox.get_max_bound(), dtype=float)


def _crop_to_overlap_region(source_cloud, target_cloud, overlap_margin_m: float):
    o3d = _require_open3d()
    margin = float(overlap_margin_m)
    source_bbox = source_cloud.get_axis_aligned_bounding_box()
    target_bbox = target_cloud.get_axis_aligned_bounding_box()
    source_min, source_max = _to_numpy_bounds(source_bbox)
    target_min, target_max = _to_numpy_bounds(target_bbox)
    overlap_min = np.maximum(source_min, target_min) - margin
    overlap_max = np.minimum(source_max, target_max) + margin
    if bool(np.any(overlap_min >= overlap_max)):
        return _copy_cloud(source_cloud), _copy_cloud(target_cloud), "Overlap crop skipped because the coarse clouds do not intersect."
    overlap_bbox = o3d.geometry.AxisAlignedBoundingBox(overlap_min, overlap_max)
    source_crop = source_cloud.crop(overlap_bbox)
    target_crop = target_cloud.crop(overlap_bbox)
    if len(source_crop.points) < DEFAULT_REGISTRATION_CROP_MIN_POINTS:
        return _copy_cloud(source_cloud), _copy_cloud(target_cloud), "Overlap crop skipped because the cropped source cloud is too small."
    if len(target_crop.points) < DEFAULT_REGISTRATION_CROP_MIN_POINTS:
        return _copy_cloud(source_cloud), _copy_cloud(target_cloud), "Overlap crop skipped because the cropped target cloud is too small."
    return source_crop, target_crop, None


def _cleanup_registration_cloud(pointcloud):
    cleaned = _copy_cloud(pointcloud)
    if len(cleaned.points) <= 0:
        return cleaned
    cleaned, _ = cleaned.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cleaned


def _prepare_registration_clouds(source_cloud, target_cloud, settings: RegistrationSettings):
    source_crop, target_crop, crop_reason = _crop_to_overlap_region(
        source_cloud=source_cloud,
        target_cloud=target_cloud,
        overlap_margin_m=float(settings.overlap_margin_m),
    )
    source_ready = _cleanup_registration_cloud(source_crop)
    target_ready = _cleanup_registration_cloud(target_crop)
    return source_ready, target_ready, crop_reason


def _downsample_with_normals(pointcloud, voxel_size_m: float):
    o3d = _require_open3d()
    cloud = _copy_cloud(pointcloud)
    if float(voxel_size_m) > 0.0:
        cloud = cloud.voxel_down_sample(float(voxel_size_m))
    if len(cloud.points) <= 0:
        return cloud
    radius = float(max(voxel_size_m * 2.5, 0.005))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=30,
        )
    )
    return cloud


def _run_icp(source, target, init_transform: np.ndarray, threshold_m: float, iterations: int, estimation_kind: str):
    o3d = _require_open3d()
    estimation = _build_estimation(estimation_kind)
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(threshold_m),
        init_transform,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iterations)),
    )


def _build_estimation(estimation_kind: str):
    o3d = _require_open3d()
    if estimation_kind == "point_to_plane":
        return o3d.pipelines.registration.TransformationEstimationPointToPlane()
    if estimation_kind == "point_to_point":
        return o3d.pipelines.registration.TransformationEstimationPointToPoint()
    raise ValueError(f"Unsupported ICP estimation kind: {estimation_kind}")


def _rotation_angle_deg(transform: np.ndarray) -> float:
    rotation = np.asarray(transform[:3, :3], dtype=float)
    trace = float(np.trace(rotation))
    cos_theta = (trace - 1.0) * 0.5
    theta_rad = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return float(np.degrees(theta_rad))


def _translation_delta_m(transform: np.ndarray) -> float:
    translation = np.asarray(transform[:3, 3], dtype=float)
    return float(np.linalg.norm(translation))


def _project_transform_to_allowed_dofs(transform: np.ndarray, refinement_dofs: str) -> np.ndarray:
    projected = np.eye(4, dtype=float)
    translation = np.asarray(transform[:3, 3], dtype=float)
    if refinement_dofs == REFINEMENT_DOFS_Z:
        projected[2, 3] = float(translation[2])
        return projected
    if refinement_dofs == REFINEMENT_DOFS_TRANSLATION:
        projected[:3, 3] = translation
        return projected
    if refinement_dofs == REFINEMENT_DOFS_FULL:
        return np.asarray(transform, dtype=float)
    raise ValueError(f"Unsupported refinement DOFs: {refinement_dofs}")


def _build_rejection_reason(settings: RegistrationSettings, fitness: float, inlier_rmse: float, transform: np.ndarray) -> str | None:
    translation_delta = _translation_delta_m(transform)
    rotation_delta = _rotation_angle_deg(transform)
    if float(fitness) < float(settings.min_fitness):
        return f"Rejected ICP: fitness {float(fitness):.4f} < {float(settings.min_fitness):.4f}."
    if float(inlier_rmse) > float(settings.max_inlier_rmse_m):
        return f"Rejected ICP: inlier RMSE {float(inlier_rmse):.6f} m > {float(settings.max_inlier_rmse_m):.6f} m."
    if float(translation_delta) > float(settings.max_translation_delta_m):
        return (
            f"Rejected ICP: translation delta {float(translation_delta):.6f} m > "
            f"{float(settings.max_translation_delta_m):.6f} m."
        )
    if float(rotation_delta) > float(settings.max_rotation_delta_deg):
        return (
            f"Rejected ICP: rotation delta {float(rotation_delta):.4f} deg > "
            f"{float(settings.max_rotation_delta_deg):.4f} deg."
        )
    return None


def refine_pointcloud_alignment(source_cloud, target_cloud, settings: RegistrationSettings) -> RegistrationResult:
    """
    Refine a roughly base-aligned source cloud against a fixed target cloud.
    Uses identity as the initial transform because TF has already performed coarse alignment.
    """

    source_reference = _copy_cloud(source_cloud)
    if len(source_cloud.points) <= 0 or len(target_cloud.points) <= 0:
        return RegistrationResult(
            refined_source=source_reference,
            transformation_matrix=np.eye(4, dtype=float),
            fitness=0.0,
            inlier_rmse=0.0,
            applied=False,
            reason="Source or target point cloud is empty.",
        )
    source_ready, target_ready, crop_reason = _prepare_registration_clouds(
        source_cloud=source_cloud,
        target_cloud=target_cloud,
        settings=settings,
    )
    transform = np.eye(4, dtype=float)
    icp_result = None
    pyramid = (
        (0.010, 0.030, 30, "point_to_point"),
        (0.005, 0.012, 25, "point_to_plane"),
        (0.0025, 0.006, 20, "point_to_plane"),
    )
    for voxel_size_m, threshold_m, iterations, estimation_kind in pyramid:
        source_ds = _downsample_with_normals(source_ready, voxel_size_m)
        target_ds = _downsample_with_normals(target_ready, voxel_size_m)
        if len(source_ds.points) <= 0 or len(target_ds.points) <= 0:
            continue
        icp_result = _run_icp(
            source=source_ds,
            target=target_ds,
            init_transform=transform,
            threshold_m=threshold_m,
            iterations=iterations,
            estimation_kind=estimation_kind,
        )
        transform = np.asarray(icp_result.transformation, dtype=float)
    transform = _project_transform_to_allowed_dofs(transform, settings.refinement_dofs)
    refined_cloud = _copy_cloud(source_cloud)
    refined_cloud.transform(transform)
    if icp_result is None:
        return RegistrationResult(
            refined_source=source_reference,
            transformation_matrix=transform,
            fitness=0.0,
            inlier_rmse=0.0,
            applied=False,
            reason="ICP refinement skipped because downsampled clouds were empty.",
        )
    if settings.refinement_mode == REFINEMENT_MODE_ALWAYS:
        accept_reason = (
            "Applied ICP because refinement mode is always "
            f"with DOFs set to {settings.refinement_dofs}."
        )
        if crop_reason:
            accept_reason = f"{accept_reason} {crop_reason}"
        return RegistrationResult(
            refined_source=refined_cloud,
            transformation_matrix=transform,
            fitness=float(icp_result.fitness),
            inlier_rmse=float(icp_result.inlier_rmse),
            applied=True,
            reason=accept_reason,
        )
    rejection_reason = _build_rejection_reason(
        settings=settings,
        fitness=float(icp_result.fitness),
        inlier_rmse=float(icp_result.inlier_rmse),
        transform=transform,
    )
    if rejection_reason is not None:
        if crop_reason:
            rejection_reason = f"{rejection_reason} {crop_reason}"
        return RegistrationResult(
            refined_source=source_reference,
            transformation_matrix=transform,
            fitness=float(icp_result.fitness),
            inlier_rmse=float(icp_result.inlier_rmse),
            applied=False,
            reason=rejection_reason,
        )
    accept_reason = crop_reason
    if accept_reason:
        accept_reason = f"Accepted ICP after conservative preprocessing. {accept_reason}"
    else:
        accept_reason = f"Accepted ICP with DOFs set to {settings.refinement_dofs}."
    return RegistrationResult(
        refined_source=refined_cloud,
        transformation_matrix=transform,
        fitness=float(icp_result.fitness),
        inlier_rmse=float(icp_result.inlier_rmse),
        applied=True,
        reason=accept_reason,
    )
