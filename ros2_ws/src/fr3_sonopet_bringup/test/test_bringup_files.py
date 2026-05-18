from pathlib import Path


def test_expected_launch_files_exist():
    package_root = Path(__file__).resolve().parents[1]
    expected = {
        "experiment.launch.py",
        "fake_experiment.launch.py",
        "franka.launch.py",
        "cameras.launch.py",
        "microphone.launch.py",
        "recording.launch.py",
    }
    actual = {path.name for path in (package_root / "launch").glob("*.launch.py")}
    assert expected <= actual

