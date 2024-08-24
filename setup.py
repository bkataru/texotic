# -*- encoding: utf-8 -*-
# @Author: bkataru
# @Contact: baalateja.k@gmail.com
import sys
from pathlib import Path
from typing import List

import setuptools
from get_pypi_latest_version import GetPyPiLatestVersion


def read_txt(txt_path: str) -> List:
    if not isinstance(txt_path, str):
        txt_path = str(txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        data = list(map(lambda x: x.rstrip("\n"), f))
    return data


def get_readme() -> str:
    root_dir = Path(__file__).resolve().parent
    readme_path = str(root_dir / "docs" / "doc_whl.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    return readme


MODULE_NAME = "texotic"
DESCRIPTION = "Python library to convert images of equations into LaTeX code based on the ONNXRuntime. Cythonized fork of RapidLatexOCR"
LONG_DESCRIPTION = get_readme()

obtainer = GetPyPiLatestVersion()
try:
    latest_version = obtainer(MODULE_NAME)
except ValueError:
    latest_version = "0.0.1"

VERSION_NUM = obtainer.version_add_one(latest_version)

if len(sys.argv) > 2:
    match_str = " ".join(sys.argv[2:])
    matched_versions = obtainer.extract_version(match_str)
    if matched_versions:
        VERSION_NUM = matched_versions
sys.argv = sys.argv[:2]

INSTALL_REQUIRES = read_txt("requirements.txt")

setuptools.setup(
    name=MODULE_NAME,
    version=VERSION_NUM,
    author="bkataru",
    author_email="<baalateja.k@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    platforms="Any",
    url="https://github.com/bkataru/texotic",
    download_url="https://github.com/bkataru/texotic",
    license="MIT",
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_namespace_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    keywords=["latex", "image to text", "ocr", "cython", "mathematics"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Framework :: NumPy",
        "Framework :: OpenCV",
        "Framework :: Onnx",
        "Framework :: PyTorch",
        "Framework :: Tokenizers",
        "Framework :: Pillow",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
    ],
    python_requires=">=3.6,<3.12",
    entry_points={
        "console_scripts": [f"{MODULE_NAME}={MODULE_NAME}.main:main"],
    },
)
