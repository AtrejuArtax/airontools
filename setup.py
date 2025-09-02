from __future__ import annotations

import os

import setuptools

# Release new version steps
# ToDo: crate a script to automate the following commands
# rm -rf *.egg-info/
# rm -rf build/
# rm -rf dist/
# rm requirements.txt
# poetry export --without-hashes --format=requirements.txt > requirements.txt
# poetry build
# run python -m twine upload dist/*.whl --config-file .pypirc

PACKAGE_NAME = "airontools"

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()


setuptools.setup(
    name=PACKAGE_NAME,
    version="0.1.63",
    scripts=[],
    author="Claudi",
    author_email="claudi_ruiz@hotmail.com",
    description="Machine learning tools to complement the AIronSuit package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AtrejuArtax/airontools",
    packages=setuptools.find_packages(include=[f"{PACKAGE_NAME}*"]),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    license="BSD",
)
