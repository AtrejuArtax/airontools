from __future__ import annotations

import setuptools

PACKAGE_NAME = "airontools"
SUB_PACKAGES_NAMES = [
    "airontools.constructors",
    "airontools.constructors.models",
    "airontools.constructors.models.supervised",
    "airontools.constructors.models.unsupervised",
]

with open("README.md") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()


setuptools.setup(
    name=PACKAGE_NAME,
    version="0.1.23",
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
)
