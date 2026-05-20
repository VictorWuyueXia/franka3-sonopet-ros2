from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class SurfaceFrame:
    center: np.ndarray
    tangent: np.ndarray
    bitangent: np.ndarray
    normal: np.ndarray


def normalize(vector: Vector3) -> Vector3:
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def stabilize_normal(normal: Vector3, reference: Vector3 = (0.0, 0.0, 1.0)) -> Vector3:
    """Keep a surface normal pointing consistently with a reference axis."""
    unit = normalize(normal)
    dot = sum(a * b for a, b in zip(unit, reference, strict=True))
    if dot < 0.0:
        return tuple(-value for value in unit)  # type: ignore[return-value]
    return unit


def normalize_array(vector: np.ndarray) -> np.ndarray:
    """Normalize a Euclidean vector and reject degenerate geometric inputs."""
    values = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(values)
    if length == 0.0:
        raise ValueError("Cannot normalize a zero-length vector")
    return values / length


def stabilize_axis(axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Keep a fitted axis direction consistent with a reference direction."""
    unit_axis = normalize_array(axis)
    unit_reference = normalize_array(reference)
    if float(np.dot(unit_axis, unit_reference)) < 0.0:
        return -unit_axis
    return unit_axis


def fit_surface_frame(
    patch_points: np.ndarray,
    center_xyz: np.ndarray,
    camera_origin_xyz: np.ndarray,
) -> SurfaceFrame:
    """Fit the local tangent plane and orient its normal toward the camera."""
    points = np.asarray(patch_points, dtype=np.float64)
    center = np.asarray(center_xyz, dtype=np.float64)
    camera_origin = np.asarray(camera_origin_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Patch points must be an N x 3 matrix")
    if points.shape[0] < 3:
        raise ValueError("At least three patch points are required for PCA")

    # PCA gives the minimum-variance surface normal and maximum-variance raster tangent.
    demeaned = points - points.mean(axis=0)
    covariance = demeaned.T @ demeaned / points.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)
    normal = eigenvectors[:, order[0]]
    tangent = eigenvectors[:, order[-1]]

    # The surface normal must face the D405 optical origin for RViz and TCP consistency.
    normal = stabilize_axis(normal, camera_origin - center)

    # Tangent sign is anchored to the cloud-frame x-axis for repeatable raster ordering.
    tangent_reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tangent = stabilize_axis(tangent, tangent_reference)
    bitangent = normalize_array(np.cross(normal, tangent))
    tangent = normalize_array(np.cross(bitangent, normal))
    return SurfaceFrame(center=center, tangent=tangent, bitangent=bitangent, normal=normal)


def quaternion_xyzw_from_axes(tangent: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Convert a right-handed TCP basis with +X tangent and +Z normal to XYZW."""
    x_axis = normalize_array(tangent)
    z_axis = normalize_array(normal)
    y_axis = normalize_array(np.cross(z_axis, x_axis))
    x_axis = normalize_array(np.cross(y_axis, z_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return quaternion_xyzw_from_matrix(rotation)


def quaternion_xyzw_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Compute a unit quaternion from a proper 3 x 3 rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3 x 3")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ],
                dtype=np.float64,
            )

    quat /= np.linalg.norm(quat)
    if quat[3] < 0.0:
        quat = -quat
    return quat
