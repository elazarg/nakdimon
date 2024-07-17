from setuptools import setup, find_packages

setup(
    name='nakdimon',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.1",
        "tensorflow==2.13.0",
        "wandb==0.17.0"
    ],
)