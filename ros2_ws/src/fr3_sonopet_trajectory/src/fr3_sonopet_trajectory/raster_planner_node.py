from __future__ import annotations

import numpy as np
import rclpy
from fr3_sonopet_interfaces.action import BuildRasterPlan
from fr3_sonopet_interfaces.msg import RasterPatch, RasterPlan
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray, Quaternion, Vector3
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from fr3_sonopet_trajectory.raster_pattern import RasterSpec
from fr3_sonopet_trajectory.raster_plan_builder import (
    RasterBuild,
    _resample_raster,
)


class RasterPlannerNode(Node):
    """Build and visualize local raster trajectories from RViz point selections."""

    def __init__(self) -> None:
        super().__init__("raster_planner_node")
        self.declare_parameter("target_frame", "fr3_link0")
        self.declare_parameter("planning_cloud_topic", "/sonopet/captured_planning_cloud")
        self.declare_parameter("planning_cloud_display_topic", "/sonopet/planning_cloud")
        self.declare_parameter("clicked_point_topic", "/clicked_point")
        self.declare_parameter("square_side_m", 0.02)
        self.declare_parameter("line_spacing_m", 0.002)
        self.declare_parameter("downsample_rate", 6)
        self.declare_parameter("pattern", "unidirectional_retract")

        self._target_frame = str(self.get_parameter("target_frame").value)
        self._cloud_topic = str(self.get_parameter("planning_cloud_topic").value)
        self._display_cloud_topic = str(self.get_parameter("planning_cloud_display_topic").value)
        self._clicked_topic = str(self.get_parameter("clicked_point_topic").value)
        self._spec = RasterSpec(
            square_side_m=float(self.get_parameter("square_side_m").value),
            line_spacing_m=float(self.get_parameter("line_spacing_m").value),
            downsample_rate=int(self.get_parameter("downsample_rate").value),
            pattern=str(self.get_parameter("pattern").value),
        )

        self._planning_cloud_points: np.ndarray | None = None
        self._selected_center_base: np.ndarray | None = None

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._display_cloud_pub = self.create_publisher(
            PointCloud2,
            self._display_cloud_topic,
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
            ),
        )
        self._plan_pub = self.create_publisher(RasterPlan, "/sonopet/raster_plan", 10)
        self._poses_pub = self.create_publisher(PoseArray, "/sonopet/raster_plan/poses", 10)
        self._markers_pub = self.create_publisher(MarkerArray, "/sonopet/raster_plan/markers", 10)
        self._cloud_sub = self.create_subscription(
            PointCloud2,
            self._cloud_topic,
            self._on_cloud,
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self._clicked_sub = self.create_subscription(
            PointStamped,
            self._clicked_topic,
            self._on_clicked_point,
            10,
        )
        self._build_server = ActionServer(
            self,
            BuildRasterPlan,
            "/fr3/build_raster_plan",
            self._execute_build_plan,
        )
        self.get_logger().info(
            "Raster planner ready: "
            f"target_frame={self._target_frame}, cloud={self._cloud_topic}, "
            f"display_cloud={self._display_cloud_topic}, clicked_point={self._clicked_topic}"
        )

    def _on_cloud(self, cloud_msg: PointCloud2) -> None:
        cloud_frame = cloud_msg.header.frame_id
        if not cloud_frame:
            raise RuntimeError("Captured point cloud does not carry a frame_id")
        if cloud_frame != self._target_frame:
            raise RuntimeError(
                f"Captured point cloud must be in {self._target_frame}, got {cloud_frame}"
            )
        xyz, rgb = _pointcloud2_xyz_rgb_arrays(cloud_msg)

        # Cache the latest captured cloud directly in the robot base frame used for raster geometry.
        self._planning_cloud_points = xyz
        self._display_cloud_pub.publish(
            _make_pointcloud2(
                xyz,
                rgb,
                self._target_frame,
                self.get_clock().now().to_msg(),
            )
        )
        self.get_logger().info(
            f"Updated planning cloud with {xyz.shape[0]} points; "
            f"published {self._display_cloud_topic} in {self._target_frame}"
            f" ({'colored' if rgb is not None else 'xyz-only'})"
        )

    def _on_clicked_point(self, point_msg: PointStamped) -> None:
        # RViz point publication is the operator-facing trigger for immediate planning.
        self._selected_center_base = self._transform_point(
            _point_to_array(point_msg.point),
            point_msg.header.frame_id,
            self._target_frame,
        )
        try:
            self._build_publish_plan()
        except (RuntimeError, ValueError) as exc:
            self.get_logger().warning(f"Ignored clicked point: {exc}")

    def _execute_build_plan(self, goal_handle):
        # The action path either updates the stored center or reuses it for re-sampling.
        feedback = BuildRasterPlan.Feedback()
        feedback.phase = "building_raster_from_latest_cloud"
        goal_handle.publish_feedback(feedback)

        result = BuildRasterPlan.Result()
        if goal_handle.request.update_selected_center:
            self._selected_center_base = _point_to_array(goal_handle.request.selected_center)
        try:
            plan = self._build_publish_plan(goal_handle.request.dig_depth_mm)
            goal_handle.succeed()
            result.success = True
            result.message = f"Published {len(plan.poses.poses)} raster poses."
            result.plan = plan
        except (RuntimeError, ValueError) as exc:
            goal_handle.abort()
            result.success = False
            result.message = str(exc)
        return result

    def _build_publish_plan(self, dig_depth_mm: float = 0.0) -> RasterPlan:
        # Geometry is computed and emitted in the robot base frame.
        cloud_points = self._require_cloud_points()
        selected_base = self._require_selected_center()
        target_build = _resample_raster(
            cloud_points,
            selected_base,
            self._spec,
            dig_depth_m=dig_depth_mm / 1000.0,
        )
        self._selected_center_base = target_build.center.copy()

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self._target_frame
        plan = self._make_plan(header, target_build)
        self._plan_pub.publish(plan)
        self._poses_pub.publish(plan.poses)
        self._markers_pub.publish(self._make_markers(header, target_build))
        self.get_logger().info(f"Published raster plan with {len(plan.poses.poses)} poses")
        return plan

    def _require_selected_center(self) -> np.ndarray:
        if self._selected_center_base is None:
            raise RuntimeError(f"No raster center has been received from {self._clicked_topic}")
        return self._selected_center_base

    def _require_cloud_points(self) -> np.ndarray:
        if self._planning_cloud_points is None:
            raise RuntimeError(f"No planning cloud has been captured from {self._cloud_topic}")
        return self._planning_cloud_points

    def _lookup_matrix(self, target_frame: str, source_frame: str) -> np.ndarray:
        if target_frame == source_frame:
            return np.eye(4, dtype=np.float64)
        transform = self._tf_buffer.lookup_transform(target_frame, source_frame, Time())
        return _matrix_from_transform(transform)

    def _transform_point(
        self,
        point: np.ndarray,
        source_frame: str,
        target_frame: str,
    ) -> np.ndarray:
        matrix = self._lookup_matrix(target_frame, source_frame)
        return _transform_points(point[None, :], matrix)[0]

    def _make_plan(self, header: Header, build: RasterBuild) -> RasterPlan:
        # RasterPlan remains the machine-readable contract consumed by motion nodes.
        patch = RasterPatch()
        patch.header = header
        patch.center = _array_to_point(build.center)
        patch.square_side_m = self._spec.square_side_m
        patch.line_spacing_m = self._spec.line_spacing_m
        patch.pattern = self._spec.pattern

        poses = PoseArray()
        poses.header = header
        poses.poses = [
            _pose_from_arrays(position, quaternion)
            for position, quaternion in zip(build.points, build.quaternions_xyzw, strict=True)
        ]

        plan = RasterPlan()
        plan.header = header
        plan.patch = patch
        plan.poses = poses
        plan.normals = [_array_to_vector(normal) for normal in build.normals]
        plan.segment_names = build.segment_names
        plan.config_hash = build.config_hash
        return plan

    def _make_markers(self, header: Header, build: RasterBuild) -> MarkerArray:
        # RViz stays sparse: only executable waypoints are shown.
        path = _base_marker(header, marker_id=0, marker_type=Marker.POINTS)
        path.ns = "raster_waypoints"
        path.scale.x = 0.0005
        path.scale.y = 0.0005
        path.color.g = 1.0
        path.color.a = 1.0
        path.points = [_array_to_point(point) for point in build.points]
        return MarkerArray(markers=[path])


def _pointcloud2_xyz_rgb_arrays(
    cloud_msg: PointCloud2,
) -> tuple[np.ndarray, np.ndarray | None]:
    # Read RGB only when the input cloud carries it, so the display preserves color.
    has_rgb = any(field.name == "rgb" for field in cloud_msg.fields)
    field_names = ("x", "y", "z", "rgb") if has_rgb else ("x", "y", "z")
    raw_points = point_cloud2.read_points(cloud_msg, field_names=field_names, skip_nans=True)
    raw_array = np.asarray(raw_points)
    if raw_array.dtype.names:
        xyz = np.column_stack(
            (raw_array["x"], raw_array["y"], raw_array["z"])
        ).astype(np.float64)
        rgb = np.asarray(raw_array["rgb"], dtype=np.float32) if has_rgb else None
        return xyz, rgb
    if raw_array.ndim == 0:
        raw_array = np.asarray(list(raw_points), dtype=np.float64)
    columns = 4 if has_rgb else 3
    matrix = np.asarray(raw_array, dtype=np.float64).reshape((-1, columns))
    xyz = matrix[:, :3]
    rgb = matrix[:, 3].astype(np.float32) if has_rgb else None
    return xyz, rgb


def _make_pointcloud2(
    points: np.ndarray, rgb: np.ndarray | None, frame_id: str, stamp
) -> PointCloud2:
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    xyz32 = np.asarray(points, dtype=np.float32).reshape((-1, 3))
    if rgb is None:
        return point_cloud2.create_cloud_xyz32(header, xyz32.tolist())

    # Pack x/y/z as floats and rgb as a 32-bit value at offset 12, the RViz-compatible layout.
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    rgb32 = np.asarray(rgb, dtype=np.float32).reshape((-1,))
    records = np.column_stack((xyz32, rgb32))
    return point_cloud2.create_cloud(header, fields, records.tolist())


def _matrix_from_transform(transform_msg) -> np.ndarray:
    translation = transform_msg.transform.translation
    rotation = transform_msg.transform.rotation
    rot = _rotation_matrix_from_quaternion(
        np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float64)
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rot
    matrix[:3, 3] = np.array([translation.x, translation.y, translation.z], dtype=np.float64)
    return matrix


def _rotation_matrix_from_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _point_to_array(point: Point) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=np.float64)


def _array_to_point(values: np.ndarray) -> Point:
    point = Point()
    point.x, point.y, point.z = map(float, values)
    return point


def _array_to_vector(values: np.ndarray) -> Vector3:
    vector = Vector3()
    vector.x, vector.y, vector.z = map(float, values)
    return vector


def _pose_from_arrays(position: np.ndarray, quaternion_xyzw: np.ndarray) -> Pose:
    pose = Pose()
    pose.position = _array_to_point(position)
    pose.orientation = Quaternion(
        x=float(quaternion_xyzw[0]),
        y=float(quaternion_xyzw[1]),
        z=float(quaternion_xyzw[2]),
        w=float(quaternion_xyzw[3]),
    )
    return pose


def _base_marker(header: Header, marker_id: int, marker_type: int) -> Marker:
    marker = Marker()
    marker.header = header
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    return marker


def main() -> None:
    rclpy.init()
    node = RasterPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
