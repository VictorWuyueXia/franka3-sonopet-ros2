import numpy as np
from fr3_sonopet_trajectory.surface_geometry import (
    fit_surface_frame,
    normalize,
    quaternion_xyzw_from_axes,
    stabilize_normal,
)


def test_normalize_unit_length_axis():
    assert normalize((0.0, 0.0, 2.0)) == (0.0, 0.0, 1.0)


def test_stabilize_normal_flips_against_reference():
    assert stabilize_normal((0.0, 0.0, -1.0)) == (-0.0, -0.0, 1.0)


def test_fit_surface_frame_recovers_plane_and_faces_camera_origin():
    x_values, y_values = np.meshgrid(np.linspace(-0.01, 0.01, 5), np.linspace(-0.01, 0.01, 5))
    z_values = np.full_like(x_values, 0.2)
    patch = np.column_stack((x_values.ravel(), y_values.ravel(), z_values.ravel()))

    frame = fit_surface_frame(
        patch,
        center_xyz=np.array([0.0, 0.0, 0.2]),
        camera_origin_xyz=np.zeros(3),
    )

    assert np.allclose(frame.normal, (0.0, 0.0, -1.0))
    assert abs(float(np.dot(frame.tangent, frame.normal))) < 1.0e-12
    assert abs(float(np.dot(frame.bitangent, frame.normal))) < 1.0e-12


def test_quaternion_encodes_tangent_x_and_normal_z_axes():
    quaternion = quaternion_xyzw_from_axes(
        tangent=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
    )

    assert np.allclose(quaternion, (0.0, 0.0, 0.0, 1.0))
