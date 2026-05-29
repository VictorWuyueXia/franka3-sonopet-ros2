from setuptools import find_packages, setup

package_name = "sonopet"

setup(
    name=package_name,
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"lib/{package_name}", ["bin/sonopet_live_data"]),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Victor Xia",
    maintainer_email="victor@example.com",
    description="Sonopet DAQ and footpedal control for cutting-interval artifacts.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sonopet_node = sonopet.sonopet_node:main",
        ],
    },
)
