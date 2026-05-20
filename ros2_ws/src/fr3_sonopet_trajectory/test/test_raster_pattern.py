import numpy as np
from fr3_sonopet_trajectory.raster_pattern import (
    INTER_LINE_RETRACT_LIFT_M,
    RasterSpec,
    build_raster_uv,
)


def test_raster_uv_count_preserves_line_endpoints_and_inserts_retracts():
    points = build_raster_uv(
        RasterSpec(square_side_m=0.02, line_spacing_m=0.002, downsample_rate=6)
    )

    assert points.shape == (85, 2)
    assert np.allclose(points[0], (-0.01, -0.01))
    assert np.allclose(points[2], (0.01, -0.01))
    assert np.allclose(points[4], (-0.01, -0.01))
    assert np.isclose(points[5, 1] - points[4, 1], INTER_LINE_RETRACT_LIFT_M)


def test_unidirectional_raster_returns_to_negative_x_before_hop():
    points = build_raster_uv(
        RasterSpec(square_side_m=0.02, line_spacing_m=0.01, downsample_rate=1)
    )

    assert points[0, 0] < points[2, 0]
    assert points[2, 0] > points[4, 0]
    assert np.allclose(points[5], (-0.01, -0.005))
    assert np.allclose(points[6], (-0.01, 0.005))
    assert points[8, 0] < points[10, 0]


def test_boustrophedon_keeps_hops_on_negative_x_edge():
    spec = RasterSpec(
        square_side_m=0.02,
        line_spacing_m=0.01,
        downsample_rate=1,
        pattern="boustrophedon",
    )
    points = build_raster_uv(spec)

    assert points[0, 0] < points[2, 0]
    assert np.allclose(points[5], (-0.01, -0.005))
    assert np.allclose(points[6], (-0.01, 0.005))
