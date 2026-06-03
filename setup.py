from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="flatland_5agents",
    version="0.1.0",
    description="Multi-agent RL for railway traffic management with Flatland",
    author="flatland_5agents",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "flatland-train=train:main",
            "flatland-eval=evaluate:main",
        ]
    },
)
