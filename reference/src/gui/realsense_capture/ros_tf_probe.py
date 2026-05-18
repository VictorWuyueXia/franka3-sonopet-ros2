import argparse
import json
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lookup one ROS TF transform and print JSON."
    )
    parser.add_argument("--base-frame", type=str, required=True)
    parser.add_argument("--target-frame", type=str, required=True)
    parser.add_argument("--timeout-sec", type=float, required=True)
    return parser.parse_args()


def main():
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from tf2_ros import Buffer, TransformListener

    args = parse_args()
    rclpy.init(args=None)
    node = Node("realsense_ros_tf_probe")
    buffer = Buffer()
    listener = TransformListener(buffer, node, spin_thread=False)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    deadline_s = float(time.monotonic() + float(args.timeout_sec))
    payload = {
        "ok": False,
        "base_frame": str(args.base_frame),
        "target_frame": str(args.target_frame),
        "error": None,
        "translation_xyz": None,
        "quaternion_xyzw": None,
        "visible_topics": [],
        "required_topics_visible": {
            "/tf": False,
            "/tf_static": False,
            "/joint_states": False,
        },
        "required_topic_publishers": {
            "/tf": 0,
            "/tf_static": 0,
            "/joint_states": 0,
        },
        "visible_nodes": [],
    }
    try:
        last_error = None
        while time.monotonic() < deadline_s:
            executor.spin_once(timeout_sec=0.1)
            topic_names = [str(name) for name, _types in node.get_topic_names_and_types()]
            payload["visible_topics"] = sorted(topic_names)
            payload["required_topics_visible"] = {
                "/tf": "/tf" in topic_names,
                "/tf_static": "/tf_static" in topic_names,
                "/joint_states": "/joint_states" in topic_names,
            }
            payload["required_topic_publishers"] = {
                "/tf": int(len(node.get_publishers_info_by_topic("/tf"))),
                "/tf_static": int(len(node.get_publishers_info_by_topic("/tf_static"))),
                "/joint_states": int(len(node.get_publishers_info_by_topic("/joint_states"))),
            }
            payload["visible_nodes"] = sorted(
                [str(name) for name, _namespace in node.get_node_names_and_namespaces()]
            )
            try:
                transform = buffer.lookup_transform(
                    str(args.base_frame),
                    str(args.target_frame),
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2),
                )
                translation = transform.transform.translation
                rotation = transform.transform.rotation
                payload["ok"] = True
                payload["translation_xyz"] = [
                    float(translation.x),
                    float(translation.y),
                    float(translation.z),
                ]
                payload["quaternion_xyzw"] = [
                    float(rotation.x),
                    float(rotation.y),
                    float(rotation.z),
                    float(rotation.w),
                ]
                break
            except Exception as exc:
                last_error = str(exc)
        if not payload["ok"]:
            payload["error"] = str(last_error or "TF lookup timed out")
    finally:
        executor.remove_node(node)
        node.destroy_node()
        del listener
        rclpy.shutdown()
    print(json.dumps(payload, ensure_ascii=True))


if __name__ == "__main__":
    main()
