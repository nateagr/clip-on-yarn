#!/usr/bin/env python3

import os
from typing import List

import setuptools


def _read_reqs(relpath: str) -> List[str]:
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines() if (s.strip() and not s.startswith("#"))]


setuptools.setup(
    name="clip_on_yarn",
    version="0.1",
    install_requires=_read_reqs("requirements.txt"),
    # tests_require=_read_reqs("tests-requirements.txt"),
    packages=setuptools.find_packages(),
    # zip_safe=False,
    include_package_data=True,
)
