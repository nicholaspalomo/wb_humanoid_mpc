"""pytest configuration shared by the remote_control tests.

Most of these tests import the package by the name colcon installs it under, `remote_control`, which resolves only when
humanoid_nmpc/remote_control is on the import path. setup_env.sh puts it on PYTHONPATH, so the suite used to pass in a
sourced shell and fail to collect in one that set PYTHONPATH itself (`PYTHONPATH=$PWD python3 -m pytest
humanoid_nmpc/remote_control/test` found no module named `remote_control`). Putting the source tree first here makes the
suite independent of the caller's environment, and makes sure it tests these sources rather than an installed copy of
an older build that happens to be on the path (test_package_import_path.py pins that).
"""

import os
import sys

# humanoid_nmpc/remote_control: the directory that holds the `remote_control` package.
PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PACKAGE_PARENT in sys.path:
    sys.path.remove(PACKAGE_PARENT)
sys.path.insert(0, PACKAGE_PARENT)
