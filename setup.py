#!/usr/bin/env python
# Licensed under an MIT license - see LICENSE

import os
from setuptools import setup

setup(
    use_scm_version={
        "write_to": os.path.join("totoro", "version.py"),
        "write_to_template": '__version__ = "{version}"\n',
    }
)
