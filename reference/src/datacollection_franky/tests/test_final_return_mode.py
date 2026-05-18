import sys
from types import SimpleNamespace

import pytest

from datacollection_franky.cli import parse_pipeline_args, validate_pipeline_args
from datacollection_franky.config import JOINT_NAMES, PipelineConfig
from datacollection_franky.densify import SimplePose


def _fake_reset_joint_state(*args, **kwargs):
    """Let trajectory_compile import in test environments without PyBullet."""

    return None


sys.modules.setdefault(
    "pybullet",
    SimpleNamespace(resetJointState=_fake_reset_joint_state),
)
from datacollection_franky.sim_runtime import resolve_target_quaternion  # noqa: E402
import datacollection_franky.trajectory_compile as trajectory_compile  # noqa: E402


def _joint_pose(base_value: float):
    """Build a deterministic joint dictionary for final target checks."""

    return {
        name: float(base_value) + float(idx) * 0.01
        for idx, name in enumerate(JOINT_NAMES)
    }


def _assert_joint_pose_equal(actual, expected) -> None:
    """Compare joint dictionaries without depending on key order."""

    for name in JOINT_NAMES:
        assert actual[name] == pytest.approx(expected[name])


def test_parse_pipeline_args_defaults_to_workflow_start(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"])

    args = parse_pipeline_args(PipelineConfig())

    assert args.final_return_mode == "workflow_start"
    validate_pipeline_args(args)


def test_parse_pipeline_args_accepts_idle(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--final-return-mode", "idle"])

    args = parse_pipeline_args(PipelineConfig())

    assert args.final_return_mode == "idle"
    validate_pipeline_args(args)


def test_parse_pipeline_args_accepts_idle_orientation_preset(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_pipeline.py", "--orientation-mode", "preset", "--preset-orientation", "idle"],
    )

    args = parse_pipeline_args(PipelineConfig())

    assert args.orientation_mode == "preset"
    assert args.preset_orientation == "idle"
    validate_pipeline_args(args)


def test_validate_pipeline_args_rejects_multiple_orientation_presets(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pipeline.py",
            "--orientation-mode",
            "preset",
            "--preset-orientation",
            "idle",
            "--preset-quat-xyzw",
            "0",
            "0",
            "0",
            "1",
        ],
    )

    args = parse_pipeline_args(PipelineConfig())

    with pytest.raises(ValueError, match="exactly one"):
        validate_pipeline_args(args)


def test_resolve_target_quaternion_uses_idle_preset() -> None:
    args = SimpleNamespace(
        orientation_mode="preset",
        preset_orientation="idle",
        preset_rpy_deg=None,
        preset_quat_xyzw=None,
    )

    q = resolve_target_quaternion(
        args=args,
        current_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        idle_quat_xyzw=(0.0, 0.0, 2.0, 0.0),
    )

    assert q == pytest.approx((0.0, 0.0, 1.0, 0.0))


def _install_compile_fakes(monkeypatch) -> None:
    """Replace PyBullet-heavy helpers while preserving compile orchestration."""

    stage_outputs = {
        "B_idle_to_parking": _joint_pose(0.20),
        "C_parking_to_first": _joint_pose(0.30),
        "D_raster": _joint_pose(0.40),
        "E_retract": _joint_pose(0.50),
    }

    def fake_tcp_pose_at_joint(sim, ee_link_index, q_by_name):
        return SimplePose(
            xyz=(float(q_by_name[JOINT_NAMES[0]]), 0.0, 0.0),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        )

    def fake_compile_cartesian_waypoints_to_joint_waypoints(
        sim,
        ee_link_index,
        pose_waypoints,
        initial_rest_pose_by_name,
        ik_check,
        stage,
    ):
        return [dict(stage_outputs[str(stage)])]

    def fake_compute_tcp_positions_for_waypoints(sim, ee_link_index, q_list):
        return [
            (float(idx) * 0.001, 0.0, 0.0)
            for idx, _q_by_name in enumerate(q_list)
        ]

    def fake_check_trajectory_continuity(points, joint_names, max_joint_vel_rad_s) -> None:
        return None

    monkeypatch.setattr(trajectory_compile, "_tcp_pose_at_joint", fake_tcp_pose_at_joint)
    monkeypatch.setattr(
        trajectory_compile,
        "compile_cartesian_waypoints_to_joint_waypoints",
        fake_compile_cartesian_waypoints_to_joint_waypoints,
    )
    monkeypatch.setattr(
        trajectory_compile,
        "_compute_tcp_positions_for_waypoints",
        fake_compute_tcp_positions_for_waypoints,
    )
    monkeypatch.setattr(
        trajectory_compile,
        "_check_trajectory_continuity",
        fake_check_trajectory_continuity,
    )


@pytest.mark.parametrize(
    ("final_return_mode", "expected_segment_name", "expected_pose"),
    [
        ("workflow_start", "E2_to_workflow_start", _joint_pose(1.00)),
        ("idle", "E2_to_idle", _joint_pose(0.00)),
    ],
)
def test_compile_full_joint_trajectory_selects_final_return_target(
    monkeypatch,
    final_return_mode: str,
    expected_segment_name: str,
    expected_pose,
) -> None:
    _install_compile_fakes(monkeypatch)
    q_current = _joint_pose(1.00)
    q_idle = _joint_pose(0.00)
    pose_waypoints = [
        SimpleNamespace(
            xyz=(0.1, 0.0, 0.0),
            quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
    ]

    compiled = trajectory_compile.compile_full_joint_trajectory(
        sim=object(),
        ee_link_index=0,
        pose_waypoints=pose_waypoints,
        q_current=q_current,
        q_idle=q_idle,
        cfg=PipelineConfig(),
        dynamics_scale=1.0,
        final_return_mode=str(final_return_mode),
    )

    assert compiled.segments[-1][0] == expected_segment_name
    assert compiled.points[-1].segment_name == expected_segment_name
    _assert_joint_pose_equal(compiled.points[-1].q_by_name, expected_pose)
