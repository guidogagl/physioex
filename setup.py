from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="physioex",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "train=physioex.train.train:main",
            "preprocess=physioex.preprocess.main:main",
        ],
    },
)
