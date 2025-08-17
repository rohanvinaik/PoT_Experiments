#!/usr/bin/env python3
"""
Setup script for PoT Attack CLI.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="pot-attacks",
    version="1.0.0",
    author="PoT Team",
    description="Proof-of-Training Attack Resistance Evaluation Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pot/attacks",
    packages=find_packages(include=["pot", "pot.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "click>=8.0.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
        "matplotlib>=3.3.0",
        "plotly>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
        ],
        "dashboard": [
            "dash>=2.0.0",
            "dash-bootstrap-components>=1.0.0",
            "kaleido>=0.2.0",
        ],
        "advanced": [
            "torchattacks>=3.0.0",
            "foolbox>=3.3.0",
            "adversarial-robustness-toolbox>=1.10.0",
        ],
        "monitoring": [
            "wandb>=0.12.0",
            "tensorboard>=2.7.0",
            "mlflow>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pot-attack=pot.cli.attack_cli:main",
            "pot-benchmark=pot.eval.attack_benchmarks:main",
        ],
    },
    include_package_data=True,
    package_data={
        "pot": [
            "config/*.yaml",
            "config/*.json",
            "docs/*.md",
            "docs/*.rst",
        ],
    },
)