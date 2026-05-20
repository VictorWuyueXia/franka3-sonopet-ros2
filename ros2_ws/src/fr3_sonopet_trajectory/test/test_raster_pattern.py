import numpy as np
from fr3_sonopet_trajectory.raster_pattern import RasterSpec, build_raster_uv


def test_raster_uv_count_preserves_line_endpoints_after_downsample():
    points = build_raster_uv(
        RasterSpec(square_side_m=0.02, line_spacing_m=0.002, downsample_rate=6)
    )

    assert points.shape == (33, 2)
    assert np.allclose(points[0], (-0.01, -0.01))
    assert np.allclose(points[2], (0.01, -0.01))


def test_unidirectional_raster_keeps_each_row_left_to_right():
    points = build_raster_uv(
        RasterSpec(square_side_m=0.02, line_spacing_m=0.01, downsample_rate=1)
    )

    assert points[0, 0] < points[2, 0]
    assert points[3, 0] < points[5, 0]


def test_boustrophedon_reverses_every_other_row():
    spec = RasterSpec(
        square_side_m=0.02,
        line_spacing_m=0.01,
        downsample_rate=1,
        pattern="boustrophedon",
    )
    points = build_raster_uv(spec)

    assert points[0, 0] < points[2, 0]
    assert points[3, 0] > points[5, 0]
