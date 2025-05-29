# setup.py
import setuptools

setuptools.setup(
    name="h12-controller",                 # PyPI-friendly name
    version="0.1.0",                               # bump on each release
    author="Your Name",
    author_email="you@example.com",
    description="Python implementation of H1-2 robot controller",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/H12Controller",
    packages=setuptools.find_packages(exclude=["tests*", "assets*", "data*"]),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19",
        "torch>=1.8",
        "open3d>=0.15",
        # add whatever else you need
    ],
    include_package_data=True,  # so MANIFEST.in is respected
    package_data={
        "h12_controller": [
            "../assets/*",         # adjust globs to your structure
            "../data/*.npy",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
