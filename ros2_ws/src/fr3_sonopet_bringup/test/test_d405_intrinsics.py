from pathlib import Path

from fr3_sonopet_bringup.camera_intrinsics import (
    CAMERA_ROLES,
    CAMERA_STREAMS,
    FRAME_ID,
    INTRINSICS_PATH,
    OUTPUT_CAMERA_INFO_TOPIC,
    camera_info_from_intrinsic,
    load_intrinsics,
    save_intrinsics,
)
from sensor_msgs.msg import CameraInfo


def _camera_info(frame_id: str) -> CameraInfo:
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = 848
    msg.height = 480
    msg.distortion_model = "plumb_bob"
    msg.d = [0.1, 0.2, 0.3, 0.4, 0.5]
    msg.k = [1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def test_d405_intrinsics_round_trip_to_dedicated_yaml(tmp_path):
    path = tmp_path / "d405_intrinsics.yaml"

    save_intrinsics(_camera_info("color"), _camera_info("depth"), path)
    intrinsics = load_intrinsics(path)

    assert path.exists()
    assert set(intrinsics) == {"color", "depth"}
    assert intrinsics["color"].width == 848
    assert intrinsics["depth"].k == (1.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 1.0)


def test_fixed_camera_info_reconstruction_for_both_roles(tmp_path):
    path = tmp_path / "d405_intrinsics.yaml"
    save_intrinsics(_camera_info("color"), _camera_info("depth"), path)
    intrinsics = load_intrinsics(path)

    for role in CAMERA_ROLES:
        for stream in CAMERA_STREAMS:
            msg = camera_info_from_intrinsic(intrinsics[stream], FRAME_ID[(role, stream)], None)
            assert msg.header.frame_id == FRAME_ID[(role, stream)]
            assert msg.width == 848
            assert msg.k[0] == 1.0
            assert OUTPUT_CAMERA_INFO_TOPIC.format(role=role, stream=stream).startswith(
                "/sonopet/d405_intrinsics/"
            )


def test_intrinsics_path_name_is_fixed():
    assert INTRINSICS_PATH.name == Path("d405_intrinsics.yaml").name
    assert INTRINSICS_PATH.parent.name == "intrinsics"
