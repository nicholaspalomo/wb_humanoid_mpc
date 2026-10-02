"""The entry point of //humanoid_nmpc/remote_control:keyboard_velocity_publisher: the keyboard's walking commands.

A file of its own outside the package, so that remote_control/keyboard_walking_command_publisher.py
runs as a module of the package rather than as __main__.
"""

import sys

from remote_control.keyboard_walking_command_publisher import main

if __name__ == "__main__":
    sys.exit(main())
