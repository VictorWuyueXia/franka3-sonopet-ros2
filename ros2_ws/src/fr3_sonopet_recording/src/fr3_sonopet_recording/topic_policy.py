from __future__ import annotations


def require_topics(topics: tuple[str, ...]) -> None:
    if not topics:
        raise ValueError("Recording topic policy cannot be empty")
    if any(not topic.startswith("/") for topic in topics):
        raise ValueError("Recording topics must be absolute ROS topic names")
