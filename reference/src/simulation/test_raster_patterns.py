"""Unit tests for selected-square raster pattern generation."""

from __future__ import annotations

import unittest

import numpy as np

from simulation.raster_patterns import (
    INTER_LINE_RETRACT_LIFT_M,
    RASTER_PATTERN_BOUSTROPHEDON,
    RASTER_PATTERN_UNIDIRECTIONAL_RETRACT,
    RasterPatternSpec,
    build_local_tangent_square_corners,
    build_raster_points_from_selected_square,
    downsample_line_points,
)


class RasterPatternTests(unittest.TestCase):
    def setUp(self) -> None:
        xs = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)
        ys = np.asarray([-1.0, 0.0, 1.0], dtype=float)
        points = []
        for y in ys:
            for x in xs:
                points.append([float(x), float(y), 0.0])
        self.points_xyz = np.asarray(points, dtype=float)

    def test_downsample_line_keeps_endpoints(self) -> None:
        line = np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        downsampled = downsample_line_points(line, 2)
        np.testing.assert_allclose(downsampled[:, 0], [-1.0, 0.0, 1.0])

    def test_boustrophedon_pattern_alternates_lines(self) -> None:
        spec = RasterPatternSpec(
            square_center_xyz=(0.0, 0.0, 0.0),
            square_side_len_m=2.0,
            line_spacing_m=0.8,
            pointcloud_downsample_rate=2,
            raster_pattern=RASTER_PATTERN_BOUSTROPHEDON,
        )
        raster = build_raster_points_from_selected_square(self.points_xyz, spec)
        np.testing.assert_allclose(raster[0:3, 0], [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(raster[3:6, 2], [INTER_LINE_RETRACT_LIFT_M, INTER_LINE_RETRACT_LIFT_M, 0.0])
        np.testing.assert_allclose(raster[6:9, 0], [1.0, 0.0, -1.0])

    def test_unidirectional_pattern_lifts_between_lines(self) -> None:
        spec = RasterPatternSpec(
            square_center_xyz=(0.0, 0.0, 0.0),
            square_side_len_m=2.0,
            line_spacing_m=0.8,
            pointcloud_downsample_rate=2,
            raster_pattern=RASTER_PATTERN_UNIDIRECTIONAL_RETRACT,
        )
        raster = build_raster_points_from_selected_square(self.points_xyz, spec)
        np.testing.assert_allclose(raster[0:3, 0], [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(raster[3:6, 0], [1.0, -1.0, -1.0])
        np.testing.assert_allclose(raster[3:6, 2], [INTER_LINE_RETRACT_LIFT_M, INTER_LINE_RETRACT_LIFT_M, 0.0])
        np.testing.assert_allclose(raster[6:9, 0], [-1.0, 0.0, 1.0])

    def test_line_spacing_not_smaller_than_half_square_raises(self) -> None:
        spec = RasterPatternSpec(
            square_center_xyz=(0.0, 0.0, 0.0),
            square_side_len_m=2.0,
            line_spacing_m=1.0,
            pointcloud_downsample_rate=1,
            raster_pattern=RASTER_PATTERN_BOUSTROPHEDON,
        )
        with self.assertRaises(ValueError):
            build_raster_points_from_selected_square(self.points_xyz, spec)

    def test_unidirectional_pattern_uses_local_surface_plane(self) -> None:
        points = []
        for z in [-1.0, 0.0, 1.0]:
            for y in [-1.0, -0.5, 0.0, 0.5, 1.0]:
                points.append([0.0, float(y), float(z)])
        vertical_plane = np.asarray(points, dtype=float)
        spec = RasterPatternSpec(
            square_center_xyz=(0.0, 0.0, 0.0),
            square_side_len_m=2.0,
            line_spacing_m=0.8,
            pointcloud_downsample_rate=2,
            raster_pattern=RASTER_PATTERN_UNIDIRECTIONAL_RETRACT,
        )
        raster = build_raster_points_from_selected_square(vertical_plane, spec)
        np.testing.assert_allclose(raster[:, 0], np.zeros(raster.shape[0]))
        np.testing.assert_allclose(raster[0:3, 1], [-1.0, 0.0, 1.0])
        self.assertAlmostEqual(float(np.max(raster[0:3, 2]) - np.min(raster[0:3, 2])), 0.0)

    def test_raster_interpolates_regular_points_from_noisy_cloud(self) -> None:
        points = []
        for y in [-1.0, 0.0, 1.0]:
            for idx, x in enumerate([-1.0, -0.45, 0.12, 0.48, 1.0]):
                z = 0.05 * float(idx % 2)
                points.append([float(x), float(y), z])
        noisy_cloud = np.asarray(points, dtype=float)
        spec = RasterPatternSpec(
            square_center_xyz=(0.0, 0.0, 0.0),
            square_side_len_m=2.0,
            line_spacing_m=0.8,
            pointcloud_downsample_rate=2,
            raster_pattern=RASTER_PATTERN_BOUSTROPHEDON,
        )
        raster = build_raster_points_from_selected_square(noisy_cloud, spec)
        np.testing.assert_allclose(raster[0:3, 0], [-1.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(raster[6:9, 0], [1.0, 0.0, -1.0], atol=1e-5)

    def test_local_square_corners_follow_vertical_surface(self) -> None:
        points = []
        for z in [-1.0, 0.0, 1.0]:
            for y in [-1.0, 0.0, 1.0]:
                points.append([0.0, float(y), float(z)])
        vertical_plane = np.asarray(points, dtype=float)
        spec = RasterPatternSpec(
            square_center_xyz=(0.0, 0.0, 0.0),
            square_side_len_m=2.0,
            line_spacing_m=0.8,
            pointcloud_downsample_rate=1,
            raster_pattern=RASTER_PATTERN_UNIDIRECTIONAL_RETRACT,
        )
        corners = build_local_tangent_square_corners(vertical_plane, spec)
        np.testing.assert_allclose(corners[:, 0], np.zeros(4), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
