from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SnapshotResult:
    label: str
    path: Path
    success: bool
    point_count: int = 0
    error: str = ""


class PointCloudSnapshotStateMachine:
    """Enable a camera pointcloud stream only for the duration of one artifact capture."""

    def __init__(
        self,
        set_enabled: Callable[[bool], None],
        receive_cloud: Callable[[float], Any],
        write_cloud: Callable[[Path, Any], int],
    ) -> None:
        self._set_enabled = set_enabled
        self._receive_cloud = receive_cloud
        self._write_cloud = write_cloud

    def capture(self, label: str, path: Path, timeout_sec: float) -> SnapshotResult:
        result: SnapshotResult | None = None
        try:
            self._set_enabled(True)
            cloud = self._receive_cloud(timeout_sec)
            point_count = self._write_cloud(path, cloud)
            result = SnapshotResult(label=label, path=path, success=True, point_count=point_count)
        except Exception as exc:
            result = SnapshotResult(label=label, path=path, success=False, error=str(exc))
        finally:
            try:
                self._set_enabled(False)
            except Exception as exc:
                if result is None or result.success:
                    result = SnapshotResult(label=label, path=path, success=False, error=str(exc))
                else:
                    result = SnapshotResult(
                        label=label,
                        path=path,
                        success=False,
                        error=f"{result.error}; disable failed: {exc}",
                    )

        return result
