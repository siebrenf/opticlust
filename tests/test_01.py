"""
run from the command line with:
    pytest --disable-pytest-warnings -vvv
"""

import subprocess as sp
from os.path import dirname, join


def test_lint():
    base = dirname(dirname(__file__))
    sp.check_output(
        "black " + f"{join(base, 'opticlust')} {join(base, 'tests')}",
        shell=True,
    )
    sp.check_output(
        "isort --overwrite-in-place --profile black --conda-env requirements.yaml "
        + f"{join(base, 'opticlust')} {join(base, 'tests')}",
        shell=True,
    )
