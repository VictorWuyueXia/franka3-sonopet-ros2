from fr3_sonopet_trajectory.raster_pattern import RasterSpec, build_raster_uv


def test_unidirectional_raster_keeps_each_row_left_to_right():
    points = build_raster_uv(RasterSpec(square_side_m=0.02, line_spacing_m=0.01))
    assert points[0][0] < points[1][0]
    assert points[2][0] < points[3][0]


def test_boustrophedon_reverses_every_other_row():
    spec = RasterSpec(square_side_m=0.02, line_spacing_m=0.01, pattern="boustrophedon")
    points = build_raster_uv(spec)
    assert points[0][0] < points[1][0]
    assert points[2][0] > points[3][0]

