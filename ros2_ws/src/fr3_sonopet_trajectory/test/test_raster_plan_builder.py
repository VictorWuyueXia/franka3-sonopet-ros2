import numpy as np
from fr3_sonopet_trajectory.raster_pattern import RasterSpec, build_raster_uv
from fr3_sonopet_trajectory.raster_plan_builder import build_raster_from_cloud


def test_plan_builder_returns_nonempty_centered_raster_with_matching_metadata():
    x_values, y_values = np.meshgrid(np.linspace(-0.01, 0.01, 7), np.linspace(-0.01, 0.01, 7))
    z_values = np.full_like(x_values, 0.2)
    cloud = np.column_stack((x_values.ravel(), y_values.ravel(), z_values.ravel()))
    spec = RasterSpec(square_side_m=0.02, line_spacing_m=0.004, downsample_rate=2)

    build = build_raster_from_cloud(cloud, np.array([0.0, 0.0, 0.2]), spec)

    assert build.points.shape[0] == build_raster_uv(spec).shape[0]
    assert build.normals.shape == build.points.shape
    assert build.quaternions_xyzw.shape == (build.points.shape[0], 4)
    assert len(build.segment_names) == build.points.shape[0]
    assert np.allclose(build.center, (0.0, 0.0, 0.2))
    assert np.allclose(np.linalg.norm(build.normals, axis=1), 1.0)
    assert np.all(build.normals @ np.array([0.0, 0.0, -1.0]) > 0.99)


def test_plan_builder_accepts_pre_filtered_cloud_without_origin_crop():
    x_values, y_values = np.meshgrid(np.linspace(0.99, 1.01, 7), np.linspace(-0.01, 0.01, 7))
    z_values = np.full_like(x_values, 0.2)
    cloud = np.column_stack((x_values.ravel(), y_values.ravel(), z_values.ravel()))
    spec = RasterSpec(square_side_m=0.02, line_spacing_m=0.004, downsample_rate=2)

    build = build_raster_from_cloud(cloud, np.array([1.0, 0.0, 0.2]), spec, pre_filtered=True)

    assert np.allclose(build.center, (1.0, 0.0, 0.2))
    assert build.points.shape[0] == build_raster_uv(spec).shape[0]
