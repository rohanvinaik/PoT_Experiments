from setuptools import setup, find_packages

setup(
    name="pot",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ],
)