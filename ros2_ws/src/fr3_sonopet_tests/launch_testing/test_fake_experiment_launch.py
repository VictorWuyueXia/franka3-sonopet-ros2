from pathlib import Path


def test_fake_experiment_launch_exists():
    repo_root = Path(__file__).resolve().parents[4]
    launch_file = (
        repo_root
        / "ros2_ws"
        / "src"
        / "fr3_sonopet_bringup"
        / "launch"
        / "fake_experiment.launch.py"
    )
    assert launch_file.exists()

