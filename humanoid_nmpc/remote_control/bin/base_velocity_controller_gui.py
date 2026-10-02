"""The entry point of //humanoid_nmpc/remote_control:base_velocity_controller_gui: the operator GUI.

A file of its own outside the package, so that remote_control/base_velocity_controller_gui.py
runs as a module of the package rather than as __main__.
"""

import sys

from remote_control.base_velocity_controller_gui import main

if __name__ == "__main__":
    sys.exit(main())
