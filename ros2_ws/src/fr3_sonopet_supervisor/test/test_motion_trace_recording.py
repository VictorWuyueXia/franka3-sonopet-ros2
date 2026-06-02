from pathlib import Path


def test_supervisor_records_motion_trace_from_motion_state_and_joint_states():
    package_root = Path(__file__).resolve().parents[1]
    node = (
        package_root / "src" / "fr3_sonopet_supervisor" / "experiment_supervisor_node.py"
    ).read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert 'ARTIFACT_PATH_TOPIC = "/sonopet/artifact_path"' in node
    assert 'MOTION_STATE_TOPIC = "/fr3/motion"' in node
    assert 'JOINT_STATES_TOPIC = "/joint_states"' in node
    assert "self.declare_parameter(\"joint_state_downsample\")" in node
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in node
    assert "self._artifact_path_sub = self.create_subscription(" in node
    assert "self._motion_state_sub = self.create_subscription(" in node
    assert "self._joint_state_sub = self.create_subscription(" in node
    assert "class ActiveMotionTrace:" in node
    assert "def _on_motion_state(self, msg: Bool) -> None:" in node
    assert "def _on_joint_state(self, msg: JointState) -> None:" in node
    assert '"joint_states.csv"' in node
    assert '"joint_states_meta.json"' in node
    assert '"real_sample_rate_hz": real_sample_rate_hz' in node
    assert '"downsample": self._joint_state_downsample' in node
    assert 'raise RuntimeError(f"Artifact path has not arrived on {ARTIFACT_PATH_TOPIC}")' in node
    assert "while csv_path.exists() or meta_path.exists():" in node
    assert "<exec_depend>sensor_msgs</exec_depend>" in package_xml
    assert "<exec_depend>std_msgs</exec_depend>" in package_xml
