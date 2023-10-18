from __future__ import annotations

import os

import setuptools

# Release new version steps
# ToDo: crate a script to automate the following commands
# rm -rf airontools.egg-info/
# rm -rf build/
# rm -rf dist/
# rm requirements.txt
# poetry export --without-hashes --format=requirements.txt > requirements.txt
# poetry run python setup.py bdist_wheel
# poetry run python -m twine upload dist/*

PACKAGE_NAME = "airontools"
SUB_PACKAGES_NAMES = [
    "airontools.constructors",
    "airontools.constructors.models",
    "airontools.constructors.models.supervised",
    "airontools.constructors.models.unsupervised",
    "airontools.on_the_fly",
]
OPTIONS = {}
if os.uname().sysname.lower() == "darwin":
    OPTIONS.update({"bdist_wheel": {"plat_name": "macosx_13_0_arm64"}})

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()


setuptools.setup(
    name=PACKAGE_NAME,
    version="0.1.29",
    scripts=[],
    author="Claudi Ruiz Camps",
    author_email="claudi_ruiz@hotmail.com",
    description="Machine learning tools to complement the AIronSuit package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AtrejuArtax/airontools",
    packages=setuptools.find_packages(include=[PACKAGE_NAME] + SUB_PACKAGES_NAMES),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    license="BSD",
    options=OPTIONS,
)
