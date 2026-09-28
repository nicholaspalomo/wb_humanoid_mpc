"""The remote_control tests exercise the sources next to them, whatever the caller's PYTHONPATH.

conftest.py puts humanoid_nmpc/remote_control first on sys.path. Without it the suite depended on setup_env.sh having
been sourced (it failed to collect under `PYTHONPATH=$PWD`), and in a shell whose path also held a colcon install of the
package, `import remote_control` could load that installed copy, so a test would pass or fail against code that is not
the code in this checkout.
"""

import os
import unittest

# First, so that the module also runs under `python3 -m unittest`, which does not load conftest.py by itself.
import conftest
import remote_control
from remote_control.tk_app import yaml_param_tree

SOURCE_PACKAGE = os.path.realpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "remote_control")
)


def _package_dir(module):
    return os.path.dirname(os.path.realpath(module.__file__))


class TestPackageImportPath(unittest.TestCase):
    def test_conftest_names_the_directory_that_holds_the_package(self):
        self.assertEqual(
            os.path.realpath(os.path.join(conftest.PACKAGE_PARENT, "remote_control")),
            SOURCE_PACKAGE,
        )
        self.assertTrue(os.path.isfile(os.path.join(SOURCE_PACKAGE, "__init__.py")))

    def test_the_package_is_imported_from_this_source_tree(self):
        self.assertEqual(_package_dir(remote_control), SOURCE_PACKAGE)

    def test_a_subpackage_is_imported_from_this_source_tree(self):
        # tk_app is a package of its own in setup.py, so an install could provide it separately.
        self.assertEqual(
            _package_dir(yaml_param_tree), os.path.join(SOURCE_PACKAGE, "tk_app")
        )

    def test_positive_control_a_module_from_elsewhere_is_told_apart(self):
        # The comparison above can fail: a module loaded from anywhere else does not match the source tree.
        self.assertNotEqual(_package_dir(unittest), SOURCE_PACKAGE)
        self.assertNotEqual(
            _package_dir(unittest), os.path.join(SOURCE_PACKAGE, "tk_app")
        )


if __name__ == "__main__":
    unittest.main()
