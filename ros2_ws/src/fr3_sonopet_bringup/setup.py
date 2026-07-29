from glob import glob

from setuptools import find_packages, setup

package_name = "fr3_sonopet_bringup"

setup(
    name=package_name,
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/intrinsics", ["intrinsics/.gitkeep", *glob("intrinsics/*.yaml")]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Victor Xia",
    maintainer_email="victor@example.com",
    description="Launch and frozen configuration for the FR3 Sonopet experiment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "d405_intrinsics_capture = fr3_sonopet_bringup.camera_intrinsics:capture_main",
            "d405_intrinsics_publisher = fr3_sonopet_bringup.camera_intrinsics:publisher_main",
            "shutdown_manager_node = fr3_sonopet_bringup.shutdown_manager:main",
        ],
    },
)
