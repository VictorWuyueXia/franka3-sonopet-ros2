from pathlib import Path


def test_raster_planner_accepts_each_recorder_captured_cloud():
    package_root = Path(__file__).resolve().parents[1]
    planner_source = (
        package_root / "src" / "fr3_sonopet_trajectory" / "raster_planner_node.py"
    ).read_text(encoding="utf-8")
    raster_config = (package_root.parent / "fr3_sonopet_bringup" / "config" / "raster.yaml").read_text(
        encoding="utf-8"
    )

    assert "/sonopet/captured_planning_cloud" in planner_source
    assert "/sonopet/captured_planning_cloud" in raster_config
    assert "if self._planning_cloud_points is not None:" not in planner_source
