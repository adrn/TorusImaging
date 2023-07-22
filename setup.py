import os

from setuptools import setup

VERSION_TEMPLATE = """
try:
    from setuptools_scm import get_version
    __version__ = get_version(root='..', relative_to=__file__)
except Exception:
    __version__ = '{version}'
""".lstrip()

setup(
    use_scm_version={
        "write_to": os.path.join("torusimaging", "version.py"),
        "write_to_template": VERSION_TEMPLATE,
    }
)
