import pytest

from fr3_sonopet_recording.topic_policy import DEFAULT_TOPICS, require_topics


def test_default_topics_are_valid():
    require_topics(DEFAULT_TOPICS)


def test_relative_topics_are_rejected():
    with pytest.raises(ValueError):
        require_topics(("joint_states",))

