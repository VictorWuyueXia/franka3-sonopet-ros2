import numpy as np
from fr3_sonopet_trajectory.raster_pattern import INTER_LINE_RETRACT_LIFT_M, RasterSpec
from fr3_sonopet_trajectory.raster_plan_builder import (
    build_raster_from_cloud,
    build_raster_with_mean_surface_z,
)


def test_plan_builder_interpolates_surface_and_keeps_constant_orientation():
    x_values, y_values = np.meshgrid(np.linspace(-0.01, 0.01, 11), np.linspace(-0.01, 0.01, 11))
    z_values = 0.2 + 0.1 * x_values + 0.2 * y_values
    cloud = np.column_stack((x_values.ravel(), y_values.ravel(), z_values.ravel()))
    spec = RasterSpec(square_side_m=0.02, line_spacing_m=0.002, downsample_rate=6)

    build = build_raster_from_cloud(cloud, np.array([0.0, 0.0, 0.2]), spec)

    assert build.normals.shape == build.points.shape
    assert build.quaternions_xyzw.shape == (build.points.shape[0], 4)
    assert len(build.segment_names) == build.points.shape[0]
    assert np.allclose(build.center, (0.0, 0.0, 0.2))
    assert np.allclose(np.linalg.norm(build.normals, axis=1), 1.0)
    assert np.all(build.normals @ np.array([0.0, 0.0, 1.0]) > 0.95)
    assert np.allclose(build.quaternions_xyzw, build.quaternions_xyzw[0])
    assert np.allclose(build.quaternions_xyzw[0], (0.0, 0.0, 0.0, 1.0))
    assert "retract" in build.segment_names
    assert float(np.ptp(build.points[:, 2])) > INTER_LINE_RETRACT_LIFT_M


def test_plan_builder_accepts_pre_filtered_cloud_without_origin_crop():
    x_values, y_values = np.meshgrid(np.linspace(0.99, 1.01, 7), np.linspace(-0.01, 0.01, 7))
    z_values = np.full_like(x_values, 0.2)
    cloud = np.column_stack((x_values.ravel(), y_values.ravel(), z_values.ravel()))
    spec = RasterSpec(square_side_m=0.02, line_spacing_m=0.004, downsample_rate=2)

    build = build_raster_from_cloud(cloud, np.array([1.0, 0.0, 0.2]), spec, pre_filtered=True)

    assert np.allclose(build.center, (1.0, 0.0, 0.2))
    assert build.points.shape[0] == len(build.segment_names)


def test_plan_builder_updates_center_z_from_non_retract_waypoints():
    x_values, y_values = np.meshgrid(np.linspace(-0.01, 0.01, 11), np.linspace(-0.01, 0.01, 11))
    z_values = np.full_like(x_values, 0.21)
    cloud = np.column_stack((x_values.ravel(), y_values.ravel(), z_values.ravel()))
    spec = RasterSpec(square_side_m=0.02, line_spacing_m=0.004, downsample_rate=2)

    build = build_raster_with_mean_surface_z(cloud, np.array([0.0, 0.0, 0.205]), spec)

    raster_mask = np.asarray([name != "retract" for name in build.segment_names], dtype=bool)
    assert np.allclose(build.center, (0.0, 0.0, 0.21))
    assert np.allclose(build.points[raster_mask, 2], 0.21)
    assert np.all(build.points[~raster_mask, 2] > 0.21)
