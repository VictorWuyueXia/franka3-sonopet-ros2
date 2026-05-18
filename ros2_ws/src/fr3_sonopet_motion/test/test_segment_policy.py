import pytest

from fr3_sonopet_motion.segment_policy import validate_segment_order


def test_validate_segment_order_accepts_default_policy():
    validate_segment_order()


def test_validate_segment_order_rejects_missing_raster():
    with pytest.raises(ValueError):
        validate_segment_order(("current_to_idle", "return_to_start"))

