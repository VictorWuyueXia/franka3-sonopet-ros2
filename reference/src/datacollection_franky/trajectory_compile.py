"""trajectory_compile.py

High-level compilation:
- Build staged Cartesian waypoint sequences.
- Densify them so adjacent translation <= max_step_m.
- Solve IK once in PyBullet to obtain joint waypoint list.

Segments:
A. current -> idle (joint interpolation, no IK)
B. idle -> parking (IK)
C. parking -> first raster waypoint (IK)
D. raster waypoints (IK)
E. retract -> final return target (IK for retract waypoint, then joint interpolation)

Rationale for segment E:
- Retract pose is defined in Cartesian space to avoid moving through the surface.
- Return to idle or workflow start is a pure joint interpolation, since both targets are in joint space already.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pybullet as p

from .config import PipelineConfig
from .densify import SimplePose, densify_by_max_step
from .ik_compile import compile_cartesian_waypoints_to_joint_waypoints
from .prompts import info


JointDict = Dict[str, float]
Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]

SEGMENT_A_INTERP_DT_S = 0.5
SEGMENT_E2_INTERP_DT_S = 0.05
WAIT_WAYPOINTS_BETWEEN_SEGMENTS = 2
BOUNDARY_RAMP_POINTS = 2
EPS_CONTINUITY = 1e-9
HOLD_DT_S = 0.05
BASE_TRAJECTORY_ACC_M_S2 = 0.2
DEFAULT_SEGMENT_SPEED_M_S = {
    "A_current_to_idle": 0.05,
    "B_idle_to_parking": 0.03,
    "C_parking_to_first": 0.01,
    "D_raster": 0.005,
    "E_retract": 0.03,
    "E2_to_workflow_start": 0.03,
    "E2_to_idle": 0.03,
}
FINAL_RETURN_WORKFLOW_START = "workflow_start"
FINAL_RETURN_IDLE = "idle"


@dataclass(frozen=True)
class RuntimeTrajectoryPoint:
    q_by_name: JointDict
    dt_s: float
    dq_by_name: JointDict
    ddq_by_name: JointDict
    segment_name: str
    is_hold: bool


@dataclass(frozen=True)
class CompiledTrajectory:
    points: List[RuntimeTrajectoryPoint]
    # Each entry is a (segment_name, start_index, end_index_exclusive)
    segments: List[Tuple[str, int, int]]
    segment_a_interp_dt_s: float


@dataclass(frozen=True)
class HardwareTrajectoryPoint:
    """Sparse hardware waypoints for Franky. Optional joint velocity hints for selected points."""

    q_by_name: JointDict
    dq_hint_by_name: Optional[JointDict]


def _joint_lerp(a: float, b: float, t: float) -> float:
    return (1.0 - t) * float(a) + t * float(b)


def interpolate_joint_segment(
    q_start: JointDict,
    q_goal: JointDict,
    control_dt_s: float,
    duration_s: float,
    joint_names: Sequence[str],
    include_start: bool,
) -> List[JointDict]:
    """Dense joint interpolation, inclusive of the last point."""

    if duration_s <= 0.0:
        if include_start:
            return [dict(q_start), dict(q_goal)]
        return [dict(q_goal)]

    n = max(1, int(math.ceil(float(duration_s) / float(control_dt_s))))
    out: List[JointDict] = []
    if include_start:
        out.append(dict(q_start))

    for k in range(1, n + 1):
        t = float(k) / float(n)
        qk: JointDict = {}
        for jn in joint_names:
            qk[jn] = _joint_lerp(q_start[jn], q_goal[jn], t)
        out.append(qk)

    return out


def _resolve_final_return_target(
    q_current: JointDict,
    q_idle: JointDict,
    final_return_mode: str,
) -> Tuple[str, JointDict]:
    """Resolve the final joint-space return target and segment label."""

    mode = str(final_return_mode)
    if mode == FINAL_RETURN_WORKFLOW_START:
        return "E2_to_workflow_start", dict(q_current)
    if mode == FINAL_RETURN_IDLE:
        return "E2_to_idle", dict(q_idle)
    raise ValueError("final_return_mode must be workflow_start or idle")


def _pose_above(wp: object, lift_m: float) -> SimplePose:
    return SimplePose(
        xyz=(float(wp.xyz[0]), float(wp.xyz[1]), float(wp.xyz[2]) + float(lift_m)),
        quat_xyzw=(
            float(wp.quat_xyzw[0]),
            float(wp.quat_xyzw[1]),
            float(wp.quat_xyzw[2]),
            float(wp.quat_xyzw[3]),
        ),
    )


def _pose_from_wp(wp: object) -> SimplePose:
    return SimplePose(
        xyz=(float(wp.xyz[0]), float(wp.xyz[1]), float(wp.xyz[2])),
        quat_xyzw=(
            float(wp.quat_xyzw[0]),
            float(wp.quat_xyzw[1]),
            float(wp.quat_xyzw[2]),
            float(wp.quat_xyzw[3]),
        ),
    )


def _tcp_pose_at_joint(sim, ee_link_index: int, q_by_name: JointDict) -> SimplePose:
    """Forward kinematics for TCP pose at a given joint dict, without changing sim state."""

    cur = sim.get_current_joint_positions()
    for js in sim.joints:
        if js.name not in q_by_name:
            continue
        p.resetJointState(sim.robot_id, js.index, targetValue=float(q_by_name[js.name]), targetVelocity=0.0)
    pos, quat = sim.get_tcp_pose(ee_link_index)
    for js in sim.joints:
        if js.name not in cur:
            continue
        p.resetJointState(sim.robot_id, js.index, targetValue=float(cur[js.name]), targetVelocity=0.0)
    return SimplePose(
        xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
        quat_xyzw=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
    )


def _zero_joint_dict(joint_names: Sequence[str]) -> JointDict:
    return {name: 0.0 for name in joint_names}


def _max_abs_joint_value(q: JointDict, joint_names: Sequence[str]) -> float:
    out = 0.0
    for name in joint_names:
        v = abs(float(q[name]))
        if v > out:
            out = v
    return out


def _build_runtime_sequence(
    joint_waypoints: List[JointDict],
    segments: List[Tuple[str, int, int]],
) -> Tuple[List[JointDict], List[str], List[bool], List[Tuple[str, int, int]]]:
    if not joint_waypoints:
        raise ValueError("joint_waypoints is empty")

    runtime_q: List[JointDict] = []
    runtime_segment_names: List[str] = []
    runtime_is_hold: List[bool] = []
    runtime_segments: List[Tuple[str, int, int]] = []

    for seg_idx, (seg_name, start, end) in enumerate(segments):
        if end <= start:
            continue

        seg_start = len(runtime_q)
        for idx in range(int(start), int(end)):
            runtime_q.append(dict(joint_waypoints[idx]))
            runtime_segment_names.append(str(seg_name))
            runtime_is_hold.append(False)
        runtime_segments.append((str(seg_name), seg_start, len(runtime_q)))

        if seg_idx >= len(segments) - 1:
            continue

        wait_name = f"W_after_{seg_name}"
        wait_start = len(runtime_q)
        hold_q = dict(joint_waypoints[int(end) - 1])
        for _ in range(WAIT_WAYPOINTS_BETWEEN_SEGMENTS):
            runtime_q.append(dict(hold_q))
            runtime_segment_names.append(wait_name)
            runtime_is_hold.append(True)
        runtime_segments.append((wait_name, wait_start, len(runtime_q)))

    return runtime_q, runtime_segment_names, runtime_is_hold, runtime_segments


def _compute_velocity_hints(
    q_list: List[JointDict],
    dt_list: List[float],
    segment_names: List[str],
    is_hold: List[bool],
    segments: List[Tuple[str, int, int]],
    joint_names: Sequence[str],
    max_joint_vel_rad_s: float,
) -> List[JointDict]:
    n = len(q_list)
    dq = [_zero_joint_dict(joint_names) for _ in range(n)]
    if n < 3:
        return dq

    vmax = float(max_joint_vel_rad_s)

    def clamp(v: float) -> float:
        if v > vmax:
            return vmax
        if v < -vmax:
            return -vmax
        return v

    for k in range(1, n - 1):
        if is_hold[k]:
            continue
        if segment_names[k - 1] != segment_names[k] or segment_names[k] != segment_names[k + 1]:
            continue
        dt_prev = float(dt_list[k - 1])
        dt_next = float(dt_list[k])
        if dt_prev <= 0.0 or dt_next <= 0.0:
            raise ValueError(f"Invalid dt at indices {k - 1}, {k}")
        dt_sum = dt_prev + dt_next
        out = {}
        for name in joint_names:
            v = (float(q_list[k + 1][name]) - float(q_list[k - 1][name])) / dt_sum
            out[name] = clamp(float(v))
        dq[k] = out

    for _seg_name, start, end in segments:
        if end <= start:
            continue
        seg_is_hold = all(bool(is_hold[idx]) for idx in range(int(start), int(end)))
        if seg_is_hold:
            for idx in range(int(start), int(end)):
                dq[idx] = _zero_joint_dict(joint_names)
            continue
        for idx in range(int(start), int(end)):
            dist_to_edge = min(int(idx - start), int(end - 1 - idx))
            scale = min(1.0, float(dist_to_edge) / float(BOUNDARY_RAMP_POINTS))
            if scale >= 1.0:
                continue
            out = {}
            for name in joint_names:
                out[name] = float(dq[idx][name]) * float(scale)
            dq[idx] = out

    return dq


def _compute_tcp_positions_for_waypoints(
    sim,
    ee_link_index: int,
    q_list: List[JointDict],
) -> List[Vec3]:
    if not q_list:
        raise ValueError("q_list is empty")

    cur = sim.get_current_joint_positions()
    out: List[Vec3] = []
    try:
        for q_by_name in q_list:
            for js in sim.joints:
                if js.name not in q_by_name:
                    continue
                p.resetJointState(sim.robot_id, js.index, targetValue=float(q_by_name[js.name]), targetVelocity=0.0)
            pos, _quat = sim.get_tcp_pose(ee_link_index)
            out.append((float(pos[0]), float(pos[1]), float(pos[2])))
    finally:
        for js in sim.joints:
            if js.name not in cur:
                continue
            p.resetJointState(sim.robot_id, js.index, targetValue=float(cur[js.name]), targetVelocity=0.0)
    return out


def _speed_profile_at_distance(s_m: float, total_m: float, vmax_m_s: float, acc_m_s2: float) -> float:
    if total_m <= 0.0:
        return 0.0
    v_pos = math.sqrt(max(0.0, 2.0 * acc_m_s2 * s_m))
    v_neg = math.sqrt(max(0.0, 2.0 * acc_m_s2 * (total_m - s_m)))
    return min(float(vmax_m_s), float(v_pos), float(v_neg))


def _assign_segment_dt_by_speed(
    runtime_q: List[JointDict],
    runtime_segments: List[Tuple[str, int, int]],
    runtime_is_hold: List[bool],
    tcp_positions: List[Vec3],
    dynamics_scale: float,
) -> List[float]:
    if len(runtime_q) != len(tcp_positions):
        raise ValueError("runtime_q and tcp_positions length mismatch")
    if float(dynamics_scale) <= 0.0:
        raise ValueError("dynamics_scale must be > 0")

    dt_list = [float(HOLD_DT_S)] * len(runtime_q)

    for seg_name, start, end in runtime_segments:
        if end <= start:
            continue

        if all(bool(runtime_is_hold[idx]) for idx in range(int(start), int(end))):
            for idx in range(int(start), int(end)):
                dt_list[idx] = float(HOLD_DT_S)
            continue

        base_vmax = DEFAULT_SEGMENT_SPEED_M_S.get(seg_name)
        if base_vmax is None:
            raise ValueError(f"Missing segment max speed for {seg_name}")

        vmax = float(base_vmax) * float(dynamics_scale)
        acc = float(BASE_TRAJECTORY_ACC_M_S2) * float(dynamics_scale) * float(dynamics_scale)

        n_pts = int(end - start)
        if n_pts <= 1:
            dt_list[int(start)] = float(HOLD_DT_S)
            continue

        s_prefix = [0.0] * n_pts
        ds = [0.0] * (n_pts - 1)
        for i in range(n_pts - 1):
            a = tcp_positions[int(start) + i]
            b = tcp_positions[int(start) + i + 1]
            d = math.sqrt(
                (float(b[0]) - float(a[0])) ** 2
                + (float(b[1]) - float(a[1])) ** 2
                + (float(b[2]) - float(a[2])) ** 2
            )
            ds[i] = float(d)
            s_prefix[i + 1] = s_prefix[i] + float(d)

        total_len = float(s_prefix[-1])
        if total_len <= 1e-12:
            for idx in range(int(start), int(end)):
                dt_list[idx] = float(HOLD_DT_S)
            continue

        for i in range(n_pts - 1):
            if float(ds[i]) <= 1e-12:
                dt = float(HOLD_DT_S)
            else:
                s_mid = float(s_prefix[i]) + 0.5 * float(ds[i])
                v_mid = _speed_profile_at_distance(
                    s_m=float(s_mid),
                    total_m=float(total_len),
                    vmax_m_s=float(vmax),
                    acc_m_s2=float(acc),
                )
                dt = float(ds[i]) / max(float(v_mid), 1e-6)
            dt_list[int(start) + i] = max(float(dt), 1e-3)
        dt_list[int(end) - 1] = dt_list[int(end) - 2]

    return dt_list


def _compute_acceleration_hints(
    dq_list: List[JointDict],
    dt_list: List[float],
    segment_names: List[str],
    is_hold: List[bool],
    segments: List[Tuple[str, int, int]],
    joint_names: Sequence[str],
) -> List[JointDict]:
    n = len(dq_list)
    ddq = [_zero_joint_dict(joint_names) for _ in range(n)]
    if n < 3:
        return ddq

    for k in range(1, n - 1):
        if is_hold[k]:
            continue
        if segment_names[k - 1] != segment_names[k] or segment_names[k] != segment_names[k + 1]:
            continue
        dt_prev = float(dt_list[k - 1])
        dt_next = float(dt_list[k])
        if dt_prev <= 0.0 or dt_next <= 0.0:
            raise ValueError(f"Invalid dt at indices {k - 1}, {k}")
        dt_sum = dt_prev + dt_next
        out = {}
        for name in joint_names:
            a = (float(dq_list[k + 1][name]) - float(dq_list[k - 1][name])) / dt_sum
            out[name] = float(a)
        ddq[k] = out

    for seg_name, start, end in segments:
        if end <= start:
            continue
        seg_is_hold = all(bool(is_hold[idx]) for idx in range(int(start), int(end)))
        if seg_is_hold:
            for idx in range(int(start), int(end)):
                ddq[idx] = _zero_joint_dict(joint_names)
            continue
        for idx in range(int(start), int(end)):
            dist_to_edge = min(int(idx - start), int(end - 1 - idx))
            scale = min(1.0, float(dist_to_edge) / float(BOUNDARY_RAMP_POINTS))
            if scale >= 1.0:
                continue
            out = {}
            for name in joint_names:
                out[name] = float(ddq[idx][name]) * float(scale)
            ddq[idx] = out

    return ddq


def _check_trajectory_continuity(
    points: List[RuntimeTrajectoryPoint],
    joint_names: Sequence[str],
    max_joint_vel_rad_s: float,
) -> None:
    if not points:
        raise ValueError("Runtime trajectory is empty")

    vmax = float(max_joint_vel_rad_s)
    for idx, pt in enumerate(points):
        dt_s = float(pt.dt_s)
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError(f"Invalid dt at point {idx}: {dt_s}")
        vel_mag = _max_abs_joint_value(pt.dq_by_name, joint_names)
        if vel_mag > vmax + 1e-6:
            raise ValueError(f"Velocity exceeds clamp at point {idx}: {vel_mag:.6f} > {vmax:.6f}")
        if bool(pt.is_hold):
            if _max_abs_joint_value(pt.dq_by_name, joint_names) > EPS_CONTINUITY:
                raise ValueError(f"Hold point {idx} has non-zero velocity")
            if _max_abs_joint_value(pt.ddq_by_name, joint_names) > EPS_CONTINUITY:
                raise ValueError(f"Hold point {idx} has non-zero acceleration")

    for idx in range(1, len(points)):
        if points[idx - 1].segment_name == points[idx].segment_name:
            continue
        if _max_abs_joint_value(points[idx - 1].dq_by_name, joint_names) > EPS_CONTINUITY:
            raise ValueError(f"Segment boundary velocity not zero at point {idx - 1}")
        if _max_abs_joint_value(points[idx].dq_by_name, joint_names) > EPS_CONTINUITY:
            raise ValueError(f"Segment boundary velocity not zero at point {idx}")
        if _max_abs_joint_value(points[idx - 1].ddq_by_name, joint_names) > EPS_CONTINUITY:
            raise ValueError(f"Segment boundary acceleration not zero at point {idx - 1}")
        if _max_abs_joint_value(points[idx].ddq_by_name, joint_names) > EPS_CONTINUITY:
            raise ValueError(f"Segment boundary acceleration not zero at point {idx}")


def _max_abs_joint_delta(a: JointDict, b: JointDict, joint_names: Sequence[str]) -> float:
    out = 0.0
    for name in joint_names:
        dv = abs(float(b[name]) - float(a[name]))
        if dv > out:
            out = dv
    return out


def _joint_diff_vec(a: JointDict, b: JointDict, joint_names: Sequence[str]) -> List[float]:
    out: List[float] = []
    for name in joint_names:
        out.append(float(b[name]) - float(a[name]))
    return out


def _vec_norm(v: List[float]) -> float:
    s = 0.0
    for x in v:
        s += float(x) * float(x)
    return math.sqrt(s)


def _vec_dot(a: List[float], b: List[float]) -> float:
    s = 0.0
    for i in range(len(a)):
        s += float(a[i]) * float(b[i])
    return s


def _is_hold_segment_name(segment_name: str) -> bool:
    return str(segment_name).startswith("W_after_")


def _drop_hold_points(points: Sequence[RuntimeTrajectoryPoint]) -> List[RuntimeTrajectoryPoint]:
    out: List[RuntimeTrajectoryPoint] = []
    for pt in points:
        if _is_hold_segment_name(str(pt.segment_name)):
            continue
        out.append(pt)
    return out


def _drop_consecutive_duplicate_points(
    points: Sequence[RuntimeTrajectoryPoint],
    joint_names: Sequence[str],
    duplicate_tol_rad: float,
) -> List[RuntimeTrajectoryPoint]:
    if not points:
        return []
    out: List[RuntimeTrajectoryPoint] = [points[0]]
    for idx in range(1, len(points)):
        prev = out[-1]
        cur = points[idx]
        if _max_abs_joint_delta(prev.q_by_name, cur.q_by_name, joint_names) <= float(duplicate_tol_rad):
            continue
        out.append(cur)
    return out


def _is_segment_boundary(points: Sequence[RuntimeTrajectoryPoint], idx: int) -> bool:
    if idx <= 0 or idx >= len(points) - 1:
        return True
    prev_name = str(points[idx - 1].segment_name)
    cur_name = str(points[idx].segment_name)
    next_name = str(points[idx + 1].segment_name)
    return prev_name != cur_name or next_name != cur_name


def _is_turning_point(
    prev_pt: RuntimeTrajectoryPoint,
    cur_pt: RuntimeTrajectoryPoint,
    next_pt: RuntimeTrajectoryPoint,
    joint_names: Sequence[str],
    turning_cos_threshold: float,
) -> bool:
    v1 = _joint_diff_vec(prev_pt.q_by_name, cur_pt.q_by_name, joint_names)
    v2 = _joint_diff_vec(cur_pt.q_by_name, next_pt.q_by_name, joint_names)
    n1 = _vec_norm(v1)
    n2 = _vec_norm(v2)
    if n1 <= 1e-12 or n2 <= 1e-12:
        return False
    cos_theta = float(_vec_dot(v1, v2)) / float(n1 * n2)
    return cos_theta < float(turning_cos_threshold)


def _segment_groups(points: Sequence[RuntimeTrajectoryPoint]) -> List[Tuple[str, List[RuntimeTrajectoryPoint]]]:
    groups: List[Tuple[str, List[RuntimeTrajectoryPoint]]] = []
    if not points:
        return groups
    cur_name = str(points[0].segment_name)
    cur_group: List[RuntimeTrajectoryPoint] = [points[0]]
    for idx in range(1, len(points)):
        pt = points[idx]
        seg_name = str(pt.segment_name)
        if seg_name == cur_name:
            cur_group.append(pt)
            continue
        groups.append((cur_name, cur_group))
        cur_name = seg_name
        cur_group = [pt]
    groups.append((cur_name, cur_group))
    return groups


def split_runtime_points_into_motion_segments(
    points: Sequence[RuntimeTrajectoryPoint],
) -> List[Tuple[str, List[RuntimeTrajectoryPoint]]]:
    """Drop inter-segment hold points and return contiguous motion segments."""

    motion_points = _drop_hold_points(points)
    return _segment_groups(motion_points)


def _pick_first_mid_last(points: Sequence[RuntimeTrajectoryPoint]) -> List[RuntimeTrajectoryPoint]:
    n = len(points)
    if n <= 3:
        return list(points)
    mid = int(n // 2)
    return [points[0], points[mid], points[-1]]


def _clamp_dq_dict(
    dq: JointDict,
    joint_names: Sequence[str],
    max_joint_vel_rad_s: float,
) -> JointDict:
    vmax = float(max_joint_vel_rad_s)
    out: JointDict = {}
    for name in joint_names:
        v = float(dq[name])
        if v > vmax:
            v = vmax
        elif v < -vmax:
            v = -vmax
        out[name] = v
    return out


def _hw_from_runtime_no_hint(pt: RuntimeTrajectoryPoint) -> HardwareTrajectoryPoint:
    return HardwareTrajectoryPoint(q_by_name=dict(pt.q_by_name), dq_hint_by_name=None)


def _pick_first_mid_last_hw(points: Sequence[RuntimeTrajectoryPoint]) -> List[HardwareTrajectoryPoint]:
    picked = _pick_first_mid_last(points)
    return [_hw_from_runtime_no_hint(pt) for pt in picked]


def _drop_consecutive_duplicate_hardware(
    hws: Sequence[HardwareTrajectoryPoint],
    joint_names: Sequence[str],
    duplicate_tol_rad: float,
) -> List[HardwareTrajectoryPoint]:
    if not hws:
        return []
    out: List[HardwareTrajectoryPoint] = [hws[0]]
    for idx in range(1, len(hws)):
        prev = out[-1]
        cur = hws[idx]
        if _max_abs_joint_delta(prev.q_by_name, cur.q_by_name, joint_names) <= float(duplicate_tol_rad):
            continue
        out.append(cur)
    return out


def downsample_runtime_points_for_hardware(
    points: Sequence[RuntimeTrajectoryPoint],
    joint_names: Sequence[str],
    near_point_tol_rad: float,
    turning_cos_threshold: float,
    duplicate_tol_rad: float,
    max_joint_vel_rad_s: float,
) -> List[HardwareTrajectoryPoint]:
    """
    Build sparse hardware waypoints from runtime trajectory points.
    Rules:
    - remove all W_after_* hold points
    - remove consecutive duplicate points
    - raster segment keeps the final selected-point raster trajectory
    - other segments keep first/middle/last (position-only)
    - final pass drops very near points
    """
    _ = turning_cos_threshold
    _ = max_joint_vel_rad_s
    non_hold = _drop_hold_points(points)
    dedup = _drop_consecutive_duplicate_points(non_hold, joint_names, duplicate_tol_rad)
    if len(dedup) <= 2:
        return [_hw_from_runtime_no_hint(pt) for pt in dedup]

    groups = _segment_groups(dedup)
    if len(groups) == 1 and str(groups[0][0]) == "D_raster":
        raster_sparse = [_hw_from_runtime_no_hint(pt) for pt in groups[0][1]]
        return _drop_consecutive_duplicate_hardware(raster_sparse, joint_names, duplicate_tol_rad)

    sparse: List[HardwareTrajectoryPoint] = []
    for seg_name, seg_points in groups:
        if str(seg_name) == "D_raster":
            seg_sparse = [_hw_from_runtime_no_hint(pt) for pt in seg_points]
        else:
            seg_sparse = _pick_first_mid_last_hw(seg_points)
        sparse.extend(seg_sparse)

    sparse = _drop_consecutive_duplicate_hardware(sparse, joint_names, duplicate_tol_rad)
    if len(sparse) <= 2:
        return sparse

    final_points: List[HardwareTrajectoryPoint] = [sparse[0]]
    for idx in range(1, len(sparse) - 1):
        cur = sparse[idx]
        prev_kept = final_points[-1]
        if _max_abs_joint_delta(prev_kept.q_by_name, cur.q_by_name, joint_names) < float(near_point_tol_rad):
            continue
        final_points.append(cur)
    final_points.append(sparse[-1])
    final_points = _drop_consecutive_duplicate_hardware(final_points, joint_names, duplicate_tol_rad)
    if len(final_points) < 2:
        return [
            _hw_from_runtime_no_hint(dedup[0]),
            _hw_from_runtime_no_hint(dedup[-1]),
        ]
    return final_points


def compile_full_joint_trajectory(
    sim,
    ee_link_index: int,
    pose_waypoints: Sequence[object],
    q_current: JointDict,
    q_idle: JointDict,
    cfg: PipelineConfig,
    dynamics_scale: float,
    final_return_mode: str,
) -> CompiledTrajectory:
    """Compile the complete staged motion into a unified runtime trajectory."""

    if not pose_waypoints:
        raise ValueError("pose_waypoints is empty")

    joint_names = list(q_idle.keys())

    seg_ranges: List[Tuple[str, int, int]] = []
    Q: List[JointDict] = []

    def _mark(seg_name: str, start: int) -> int:
        return start

    def _close(seg_name: str, start: int) -> None:
        seg_ranges.append((seg_name, start, len(Q)))

    # Segment A: current -> idle (joint interpolation).
    # Keep sparse waypoint generation for compile efficiency.
    segment_a_interp_dt_s = float(SEGMENT_A_INTERP_DT_S)
    info(
        f"Segment A sparse interpolation dt_s={segment_a_interp_dt_s:.4f}"
    )
    s0 = _mark("A_current_to_idle", len(Q))
    Q.extend(
        interpolate_joint_segment(
            q_start=q_current,
            q_goal=q_idle,
            control_dt_s=segment_a_interp_dt_s,
            duration_s=cfg.timing.reset_current_to_idle_s,
            joint_names=joint_names,
            include_start=True,
        )
    )
    _close("A_current_to_idle", s0)

    # Segment B: idle -> parking (Cartesian -> IK)
    first_wp = pose_waypoints[0]
    parking_pose = _pose_above(first_wp, cfg.stages.parking_lift_m)
    idle_tcp_pose = _tcp_pose_at_joint(sim, ee_link_index, q_idle)
    b_poses = densify_by_max_step(
        [idle_tcp_pose, parking_pose],
        max_step_m=cfg.densify.max_step_m,
    )
    info(f"Segment B Cartesian waypoints after densify: {len(b_poses)}")
    s1 = _mark("B_idle_to_parking", len(Q))
    Q.extend(
        compile_cartesian_waypoints_to_joint_waypoints(
            sim=sim,
            ee_link_index=ee_link_index,
            pose_waypoints=b_poses,
            initial_rest_pose_by_name=q_idle,
            ik_check=cfg.ik_check,
            stage="B_idle_to_parking",
        )
    )
    _close("B_idle_to_parking", s1)

    # Segment C: parking -> first waypoint (Cartesian -> IK)
    c_start = parking_pose
    c_goal = _pose_from_wp(first_wp)
    c_poses = densify_by_max_step(
        [c_start, c_goal],
        max_step_m=cfg.densify.max_step_m,
    )
    info(f"Segment C Cartesian waypoints after densify: {len(c_poses)}")
    s2 = _mark("C_parking_to_first", len(Q))
    Q.extend(
        compile_cartesian_waypoints_to_joint_waypoints(
            sim=sim,
            ee_link_index=ee_link_index,
            pose_waypoints=c_poses,
            initial_rest_pose_by_name=Q[-1] if Q else q_idle,
            ik_check=cfg.ik_check,
            stage="C_parking_to_first",
        )
    )
    _close("C_parking_to_first", s2)

    # Segment D: raster (densify by max step, then IK)
    # The point cloud generator may already be dense; we enforce the 0.01 m constraint anyway.
    d_cart = [SimplePose(xyz=(float(w.xyz[0]), float(w.xyz[1]), float(w.xyz[2])), quat_xyzw=(float(w.quat_xyzw[0]), float(w.quat_xyzw[1]), float(w.quat_xyzw[2]), float(w.quat_xyzw[3]))) for w in pose_waypoints]
    d_cart = densify_by_max_step(d_cart, max_step_m=cfg.densify.max_step_m)
    info(f"Segment D Cartesian waypoints after densify: {len(d_cart)}")
    s3 = _mark("D_raster", len(Q))
    Q.extend(
        compile_cartesian_waypoints_to_joint_waypoints(
            sim=sim,
            ee_link_index=ee_link_index,
            pose_waypoints=d_cart,
            initial_rest_pose_by_name=Q[-1] if Q else q_idle,
            ik_check=cfg.ik_check,
            stage="D_raster",
        )
    )
    _close("D_raster", s3)

    # Segment E1: retract up from last waypoint
    last_wp = pose_waypoints[-1]
    retract_pose = _pose_above(last_wp, cfg.stages.retract_lift_m)
    e1_cart = densify_by_max_step(
        [SimplePose(xyz=(float(last_wp.xyz[0]), float(last_wp.xyz[1]), float(last_wp.xyz[2])), quat_xyzw=(float(last_wp.quat_xyzw[0]), float(last_wp.quat_xyzw[1]), float(last_wp.quat_xyzw[2]), float(last_wp.quat_xyzw[3]))), retract_pose],
        max_step_m=cfg.densify.max_step_m,
    )
    info(f"Segment E1 Cartesian waypoints after densify: {len(e1_cart)}")
    s4 = _mark("E_retract", len(Q))
    Q.extend(
        compile_cartesian_waypoints_to_joint_waypoints(
            sim=sim,
            ee_link_index=ee_link_index,
            pose_waypoints=e1_cart,
            initial_rest_pose_by_name=Q[-1] if Q else q_idle,
            ik_check=cfg.ik_check,
            stage="E_retract",
        )
    )
    _close("E_retract", s4)

    # Segment E2: retract -> configured final target (joint interpolation)
    final_return_segment_name, final_return_q_goal = _resolve_final_return_target(
        q_current=q_current,
        q_idle=q_idle,
        final_return_mode=str(final_return_mode),
    )
    info(
        "Final return mode applied: "
        f"mode={final_return_mode}, segment={final_return_segment_name}"
    )
    s5 = _mark(final_return_segment_name, len(Q))
    Q.extend(
        interpolate_joint_segment(
            q_start=Q[-1],
            q_goal=final_return_q_goal,
            control_dt_s=float(SEGMENT_E2_INTERP_DT_S),
            duration_s=cfg.timing.reset_end_to_idle_s,
            joint_names=joint_names,
            include_start=False,
        )
    )
    _close(final_return_segment_name, s5)

    runtime_q, runtime_seg_names, runtime_is_hold, runtime_segments = _build_runtime_sequence(
        joint_waypoints=Q,
        segments=seg_ranges,
    )
    tcp_positions = _compute_tcp_positions_for_waypoints(
        sim=sim,
        ee_link_index=ee_link_index,
        q_list=runtime_q,
    )
    runtime_dt_s = _assign_segment_dt_by_speed(
        runtime_q=runtime_q,
        runtime_segments=runtime_segments,
        runtime_is_hold=runtime_is_hold,
        tcp_positions=tcp_positions,
        dynamics_scale=float(dynamics_scale),
    )
    dq_list = _compute_velocity_hints(
        q_list=runtime_q,
        dt_list=runtime_dt_s,
        segment_names=runtime_seg_names,
        is_hold=runtime_is_hold,
        segments=runtime_segments,
        joint_names=joint_names,
        max_joint_vel_rad_s=float(cfg.franky.max_joint_vel_rad_s),
    )
    ddq_list = _compute_acceleration_hints(
        dq_list=dq_list,
        dt_list=runtime_dt_s,
        segment_names=runtime_seg_names,
        is_hold=runtime_is_hold,
        segments=runtime_segments,
        joint_names=joint_names,
    )

    points: List[RuntimeTrajectoryPoint] = []
    for idx in range(len(runtime_q)):
        points.append(
            RuntimeTrajectoryPoint(
                q_by_name=dict(runtime_q[idx]),
                dt_s=float(runtime_dt_s[idx]),
                dq_by_name=dict(dq_list[idx]),
                ddq_by_name=dict(ddq_list[idx]),
                segment_name=str(runtime_seg_names[idx]),
                is_hold=bool(runtime_is_hold[idx]),
            )
        )

    _check_trajectory_continuity(
        points=points,
        joint_names=joint_names,
        max_joint_vel_rad_s=float(cfg.franky.max_joint_vel_rad_s),
    )

    return CompiledTrajectory(
        points=points,
        segments=runtime_segments,
        segment_a_interp_dt_s=float(segment_a_interp_dt_s),
    )
