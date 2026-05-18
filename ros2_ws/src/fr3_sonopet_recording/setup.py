from setuptools import find_packages, setup

package_name = "fr3_sonopet_recording"

setup(
    name=package_name,
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Victor Xia",
    maintainer_email="victor@example.com",
    description="Recording policy and artifact manifests for the FR3 Sonopet experiment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "recording_node = fr3_sonopet_recording.recording_node:main",
        ],
    },
)
