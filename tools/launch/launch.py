"""Starts the processes of a launch file and stops them together: the launcher of the distributed runtime.

    bazel run //tools/launch -- <launch file> [--machine robot|laptop] [--set name=value] [--dry_run]
    .bazel/bin/tools/launch/launch <launch file> ...     # the built binary, e.g. on the robot computer

It runs through Bazel because launch files are textprotos, read with the module Bazel generates from
proto/launch_file.proto.

See README.md for the launch file format and launch_cli.py for the options. The command line lives in launch_cli.py
so that tests and other tools can import it by a name of its own: a module named `launch` is shadowed by, or shadows,
the `launch` package of ROS 2 wherever both are on the path.
"""

import sys

import launch_cli

if __name__ == "__main__":
    sys.exit(launch_cli.main())
