from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
setup(
    name="ppe-detection-framework",
    version="1.0.0",
    author="Pakin Thongraar",
    author_email="phakin.thongla-ar.external@autoliv.com",
    description="A comprehensive PPE detection framework using YOLO",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nnackpt/ppe-detection-framework",
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
    ],
    python_requires=">=3.8",
    install_requires=[
        "ultralytics>=8.0.0,<9.0.0",
        "opencv-python>=4.5.0,<5.0.0",
        "numpy>=1.21.0,<3.0.0",
        "torch>=2.0.0,<3.0.0",
        "torchvision>=0.15.0,<1.0.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.100.0,<1.0.0",
            "uvicorn>=0.23.0,<1.0.0",
            "python-multipart>=0.0.6",
        ],
        "database": [
            "pyodbc>=4.0.0,<6.0.0",
            "python-dotenv>=1.0.0,<2.0.0",
        ],
        "notifications": [
            "pygame>=2.0.0,<3.0.0",
            "pytz>=2023.3",
        ],
        "all": [
            "fastapi>=0.100.0,<1.0.0",
            "uvicorn>=0.23.0,<1.0.0",
            "python-multipart>=0.0.6",
            "pyodbc>=4.0.0,<6.0.0",
            "python-dotenv>=1.0.0,<2.0.0",
            "pygame>=2.0.0,<3.0.0",
            "pytz>=2023.3",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ppe-detect=ppe_framework.cli:main"
        ]
    }
)