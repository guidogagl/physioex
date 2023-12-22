from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='reponame',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'train=train:main',
        ],
    },
)