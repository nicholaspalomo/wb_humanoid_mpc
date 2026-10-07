"""What the robot model packages (robot_models/<robot>/<package>/BUILD.bazel) leave out of their filegroups."""

# Side files that sit beside a robot's configuration without being part of it: the save backups of the tuning GUI
# (`<file>.bak`), and the live copies (`<file>.live.yaml`) that the GUI and the MPC parameter updater of the YAML
# configuration wrote while tuning, which a working tree may still hold. Git ignores them; without this exclude every test
# that lists a robot package as data shipped whichever of them the working tree held
# (robot_models/tests/test_robot_packages_ship_no_side_files.py).
# LINT.IfChange(robot_file_excludes)
ROBOT_FILE_EXCLUDES = [
    "**/*.bak",
    "**/*.live*",
]
# LINT.ThenChange(//.gitignore:robot_file_excludes, //robot_models/tests/test_robot_packages_ship_no_side_files.py:robot_file_excludes)
