from setuptools import setup, find_packages

setup(
    name="epsilon_transformers",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "seaborn",
        "matplotlib",
        "wandb",
        "scikit-learn",
        "plotly",
        "transformer_lens",
        "pytest",
        "PyDrive",
    ],
)
