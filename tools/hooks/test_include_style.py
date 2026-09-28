"""Tests for include_style.py: Abseil headers must be included with quotes, which is how Bazel exposes them."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import include_style  # noqa: E402


def _repository_file(relative_path):
    """A repository file, from the Bazel runfiles when run by Bazel and from the source tree otherwise."""
    roots = []
    if "TEST_SRCDIR" in os.environ:
        roots.append(os.path.join(os.environ["TEST_SRCDIR"], "_main"))
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    for root in roots:
        candidate = os.path.join(root, relative_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(relative_path)


class IncludeStyleTest(unittest.TestCase):
    def test_an_angle_absl_include_is_flagged_with_its_line(self):
        source = "#include <vector>\n#include <absl/log/log.h>\n  #  include <absl/status/status.h>\n"
        self.assertEqual(
            [(v.line, v.header) for v in include_style.check_source(source)],
            [(2, "absl/log/log.h"), (3, "absl/status/status.h")],
        )

    def test_quoted_absl_and_other_angle_includes_are_fine(self):
        source = '#include "absl/log/log.h"\n#include <Eigen/Dense>\n#include <abslish/x.h>\n// #include <absl/x.h>\n'
        self.assertEqual(include_style.check_source(source), [])

    def test_the_message_says_what_to_write(self):
        violation = include_style.check_source("#include <absl/log/log.h>\n", "a.cpp")[
            0
        ]
        self.assertIn('#include "absl/log/log.h"', str(violation))

    def test_no_system_abseil_is_installed(self):
        # A second Abseil on the compiler's default include path is what let the angle includes compile in the dev
        # container and fail in CI, and it still lets a quoted include whose Bazel dependency is missing compile against
        # the wrong release. Bazel's @abseil-cpp is the only one the build may see: neither the dev image builds one
        # into /usr/local nor the dependency list installs Ubuntu's into /usr/include.
        with open(_repository_file("docker/Dockerfile")) as f:
            dockerfile = f.read()
        self.assertNotIn("abseil-cpp.git", dockerfile)
        self.assertNotIn("libabsl_base", dockerfile)
        with open(_repository_file("dependencies.txt")) as f:
            packages = [line.split("#")[0].strip() for line in f]
        self.assertFalse(
            [p for p in packages if p.startswith("libabsl")],
            "dependencies.txt installs a system Abseil",
        )

    def test_the_linter_runs_it(self):
        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "lint_code.py")
        ) as f:
            lint = f.read()
        self.assertIn("include_style.check_files(cpp_files, REPO_ROOT)", lint)


if __name__ == "__main__":
    unittest.main()
