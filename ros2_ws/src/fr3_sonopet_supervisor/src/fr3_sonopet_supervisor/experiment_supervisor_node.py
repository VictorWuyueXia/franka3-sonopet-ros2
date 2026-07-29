from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import rclpy
from fr3_sonopet_interfaces.msg import RasterPlan
from fr3_sonopet_interfaces.msg import RunState as RunStateMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from fr3_sonopet_supervisor.run_state import RunState

ARTIFACT_PATH_TOPIC = "/sonopet/artifact_path"
MOTION_STATE_TOPIC = "/fr3/motion"
JOINT_STATES_TOPIC = "/joint_states"
RASTER_PLAN_TOPIC = "/sonopet/raster_plan"


@dataclass
class ActiveMotionTrace:
    """In-memory joint trace for one physical robot motion interval."""

    started_at: float
    raster_patch: dict | None
    seen_messages: int = 0
    joint_names: list[str] = field(default_factory=list)
    samples: list[tuple[int, int, tuple[float, ...]]] = field(default_factory=list)


class ExperimentSupervisorNode(Node):
    """Publishes the experiment lifecycle state for operator-facing tooling."""

    def __init__(self) -> None:
        super().__init__("experiment_supervisor_node")
        self.declare_parameter("fake_run", False)
        self.declare_parameter("joint_state_downsample")
        run_id = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        self._state = RunState(run_id=run_id, ready=True)
        self._joint_state_downsample = int(self.get_parameter("joint_state_downsample").value)
        self._artifact_path: Path | None = None
        self._active_motion: ActiveMotionTrace | None = None
        self._latest_raster_patch: dict | None = None
        self._state_pub = self.create_publisher(RunStateMsg, "/sonopet/run_state", 10)
        artifact_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        # The supervisor records physical joint motion into the recorder-owned artifact folder.
        self._artifact_path_sub = self.create_subscription(
            String,
            ARTIFACT_PATH_TOPIC,
            self._on_artifact_path,
            artifact_qos,
        )
        self._motion_state_sub = self.create_subscription(
            Bool,
            MOTION_STATE_TOPIC,
            self._on_motion_state,
            10,
        )
        self._joint_state_sub = self.create_subscription(
            JointState,
            JOINT_STATES_TOPIC,
            self._on_joint_state,
            10,
        )
        self._raster_plan_sub = self.create_subscription(
            RasterPlan,
            RASTER_PLAN_TOPIC,
            self._on_raster_plan,
            10,
        )
        self._state_timer = self.create_timer(1.0, self._publish_state)
        self.get_logger().info(f"Experiment supervisor ready: run_id={self._state.run_id}")

    def _on_artifact_path(self, msg: String) -> None:
        # The artifact root can arrive before or after the operator-selected raster patch.
        self._artifact_path = Path(msg.data)
        self._write_raster_patch_artifact()

    def _on_motion_state(self, msg: Bool) -> None:
        # Motion-state edges define one buffered joint trace interval.
        if msg.data and self._active_motion is None:
            if self._latest_raster_patch is None:
                self.get_logger().warning("Motion started before any raster patch was published.")
            self._active_motion = ActiveMotionTrace(
                started_at=time.time(),
                raster_patch=self._latest_raster_patch,
            )
        if not msg.data and self._active_motion is not None:
            stopped_at = time.time()
            active_motion = self._active_motion
            self._active_motion = None
            artifact_path = self._artifact_path
            if artifact_path is None:
                raise RuntimeError(f"Artifact path has not arrived on {ARTIFACT_PATH_TOPIC}")
            motion_dir = artifact_path / "motion"
            motion_dir.mkdir(parents=True, exist_ok=True)
            csv_path = motion_dir / "joint_states.csv"
            meta_path = motion_dir / "joint_states_meta.json"
            index = 1
            while csv_path.exists() or meta_path.exists():
                csv_path = motion_dir / f"joint_states_{index}.csv"
                meta_path = motion_dir / f"joint_states_meta_{index}.json"
                index += 1
            csv_file = csv_path.open("w", newline="", encoding="utf-8")
            writer = csv.writer(csv_file)
            writer.writerow(
                ["sample_index", "stamp_sec", "stamp_nanosec", *active_motion.joint_names]
            )
            for sample_index, sample in enumerate(active_motion.samples):
                stamp_sec, stamp_nanosec, positions = sample
                writer.writerow([sample_index, stamp_sec, stamp_nanosec, *positions])
            csv_file.close()
            duration_s = stopped_at - active_motion.started_at
            real_sample_rate_hz = len(active_motion.samples) / duration_s
            payload = {
                "started_at": active_motion.started_at,
                "started_at_local": datetime.fromtimestamp(
                    active_motion.started_at
                ).astimezone().strftime("%Y%m%d%H%M"),
                "stopped_at": stopped_at,
                "stopped_at_local": datetime.fromtimestamp(
                    stopped_at
                ).astimezone().strftime("%Y%m%d%H%M"),
                "sample_count": len(active_motion.samples),
                "downsample": self._joint_state_downsample,
                "real_sample_rate_hz": real_sample_rate_hz,
                "csv": str(csv_path.relative_to(artifact_path)),
            }
            if active_motion.raster_patch is not None:
                payload["raster_patch"] = active_motion.raster_patch
            meta_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def _on_raster_plan(self, plan: RasterPlan) -> None:
        # The raster patch is the operator-selected pointcloud area from RViz.
        patch = plan.patch
        center = patch.center
        half_side_m = float(patch.square_side_m) / 2.0
        self._latest_raster_patch = {
            "frame_id": patch.header.frame_id or plan.header.frame_id,
            "center_m": [float(center.x), float(center.y), float(center.z)],
            "xy_min_m": [float(center.x) - half_side_m, float(center.y) - half_side_m],
            "xy_max_m": [float(center.x) + half_side_m, float(center.y) + half_side_m],
            "square_side_m": float(patch.square_side_m),
            "line_spacing_m": float(patch.line_spacing_m),
            "pattern": str(patch.pattern),
        }
        self._write_raster_patch_artifact()

    def _write_raster_patch_artifact(self) -> None:
        # Persist the selected raster region independently of joint trace finalization.
        if self._artifact_path is None or self._latest_raster_patch is None:
            return
        patch_path = self._artifact_path / "raster_patch.json"
        patch_path.write_text(
            json.dumps(self._latest_raster_patch, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _on_joint_state(self, msg: JointState) -> None:
        # The first live sample anchors the trace, then every Nth sample is retained.
        if self._active_motion is None:
            return
        self._active_motion.seen_messages += 1
        if (self._active_motion.seen_messages - 1) % self._joint_state_downsample != 0:
            return
        if not self._active_motion.joint_names:
            self._active_motion.joint_names = list(msg.name)
        if list(msg.name) != self._active_motion.joint_names:
            raise ValueError("Joint-state names changed during motion recording")
        self._active_motion.samples.append(
            (
                int(msg.header.stamp.sec),
                int(msg.header.stamp.nanosec),
                tuple(float(position) for position in msg.position),
            )
        )

    def _publish_state(self) -> None:
        # A periodic state topic lets launch tests and operator tools observe liveness.
        msg = RunStateMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.run_id = self._state.run_id
        msg.phase = self._state.phase
        msg.ready = self._state.ready
        msg.blocking_reason = self._state.blocking_reason
        msg.last_error = ""
        self._state_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = ExperimentSupervisorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
