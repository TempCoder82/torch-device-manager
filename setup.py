from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="torch-device-manager",
    version="0.1.0",
    author="Ali B.M.",
    author_email="alimainabukar@gmail.com",
    description="A PyTorch device manager for automatic hardware detection and memory optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
        url="https://github.com/yourusername/torch-device-manager",
        packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.1",
    ],
    keywords="pytorch, cuda, mps, device, memory, optimization, machine learning",
    project_urls={
        "Bug Reports": "https://github.com/TempCoder82/torch-device-manager/issues",
        "Source": "https://github.com/TempCoder82/torch-device-manager",
    },
)