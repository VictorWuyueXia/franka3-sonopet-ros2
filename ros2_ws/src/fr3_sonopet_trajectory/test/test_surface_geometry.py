from fr3_sonopet_trajectory.surface_geometry import normalize, stabilize_normal


def test_normalize_unit_length_axis():
    assert normalize((0.0, 0.0, 2.0)) == (0.0, 0.0, 1.0)


def test_stabilize_normal_flips_against_reference():
    assert stabilize_normal((0.0, 0.0, -1.0)) == (-0.0, -0.0, 1.0)

