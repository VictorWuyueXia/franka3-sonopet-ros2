from pathlib import Path


def test_raster_planner_accepts_each_recorder_captured_cloud():
    package_root = Path(__file__).resolve().parents[1]
    planner_source = (
        package_root / "src" / "fr3_sonopet_trajectory" / "raster_planner_node.py"
    ).read_text(encoding="utf-8")
    builder_source = (
        package_root / "src" / "fr3_sonopet_trajectory" / "raster_plan_builder.py"
    ).read_text(encoding="utf-8")
    raster_config = (
        package_root.parent / "fr3_sonopet_bringup" / "config" / "raster.yaml"
    ).read_text(encoding="utf-8")

    assert "/sonopet/captured_planning_cloud" in planner_source
    assert "/sonopet/captured_planning_cloud" in raster_config
    assert "Captured point cloud must be in {self._target_frame}" in planner_source
    assert (
        "target_from_cloud = self._lookup_matrix(self._target_frame, cloud_frame)"
        not in planner_source
    )
    assert "target_xyz = _transform_points(filtered_xyz, target_from_cloud)" not in planner_source
    assert "self._planning_cloud_points = xyz" in planner_source
    assert "planning_cloud_max_distance_m" not in planner_source
    assert "planning_cloud_trim_fraction" not in planner_source
    assert "planning_cloud_max_distance_m" not in raster_config
    assert "planning_cloud_trim_fraction" not in raster_config
    assert "filter_planning_cloud" not in builder_source
    assert "pre_filtered" not in builder_source
    assert "if self._planning_cloud_points is not None:" not in planner_source
    assert "non-negative" not in planner_source
    assert "non-negative" not in builder_source
    assert "self._selected_center_base" in planner_source
    assert "_resample_raster" in planner_source
    assert "goal_handle.request.update_selected_center" in planner_source
    assert "No raster center has been received from" in planner_source
