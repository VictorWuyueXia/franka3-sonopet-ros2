from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PointCloudPathSet:
    """Centralized point-cloud data directories inside the repo."""

    root_dir: Path
    scans_dir: Path
    base_aligned_dir: Path
    stitched_dir: Path
    metadata_dir: Path


class PointCloudStorage:
    """Small helper that keeps point-cloud outputs in one dedicated location."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        root_dir = self.repo_root / "pointclouds"
        self.paths = PointCloudPathSet(
            root_dir=root_dir,
            scans_dir=root_dir / "scans",
            base_aligned_dir=root_dir / "base_aligned",
            stitched_dir=root_dir / "stitched",
            metadata_dir=root_dir / "metadata",
        )

    def ensure_directories(self) -> PointCloudPathSet:
        for path in (
            self.paths.root_dir,
            self.paths.scans_dir,
            self.paths.base_aligned_dir,
            self.paths.stitched_dir,
            self.paths.metadata_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self.paths

    def build_scan_path(self, camera_name: str, timestamp_s: int) -> Path:
        self.ensure_directories()
        return self.paths.scans_dir / f"{camera_name}_{int(timestamp_s)}.pcd"

    def build_stitched_path(self, timestamp_s: int) -> Path:
        self.ensure_directories()
        return self.paths.stitched_dir / f"merged_{int(timestamp_s)}.pcd"

    def build_base_aligned_path(self, camera_name: str, timestamp_s: int) -> Path:
        self.ensure_directories()
        return self.paths.base_aligned_dir / f"{camera_name}_{int(timestamp_s)}.pcd"

    def build_metadata_path(self, timestamp_s: int) -> Path:
        self.ensure_directories()
        return self.paths.metadata_dir / f"capture_{int(timestamp_s)}.json"

    def relative_to_repo(self, path: Path) -> str:
        return str(Path(path).resolve().relative_to(self.repo_root))
