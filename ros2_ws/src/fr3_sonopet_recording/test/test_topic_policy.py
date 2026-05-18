import pytest
from fr3_sonopet_recording.topic_policy import require_topics


def test_absolute_sensor_topics_are_valid():
    require_topics(("/microphone/audio", "/camera/color/image_raw"))


def test_relative_topics_are_rejected():
    with pytest.raises(ValueError):
        require_topics(("camera/color/image_raw",))
