"""CI's format check must run the formatters the dev container and its pre-commit hook run.

The job used to take `ubuntu-latest`'s clang-format and the newest black on PyPI. Either moving on - ubuntu-latest
changes Ubuntu release on 2026-10-19, black changes its stable style every year - makes CI reject code that the hook
formatted a minute earlier. So the job names its versions, and its clang-format is the dev image's.
"""

import os
import re
import unittest


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


def _read(relative_path):
    with open(_repository_file(relative_path)) as f:
        return f.read()


class CiFormatterVersionsTest(unittest.TestCase):
    def setUp(self):
        self.workflow = _read(".github/workflows/format_test.yml")
        self.dockerfile = _read("docker/Dockerfile")

    def test_the_runner_image_is_a_fixed_release(self):
        runs_on = re.search(r"^\s*runs-on:\s*(\S+)", self.workflow, re.MULTILINE)
        self.assertIsNotNone(runs_on)
        self.assertRegex(runs_on.group(1), r"^ubuntu-\d+\.\d+$")

    def test_clang_format_is_the_dev_images_major_version(self):
        image = re.search(
            r"^ARG CLANG_VERSION=(\d+)\s*$", self.dockerfile, re.MULTILINE
        )
        self.assertIsNotNone(image, "docker/Dockerfile no longer sets CLANG_VERSION")
        installed = re.findall(r"apt-get install -y clang-format(\S*)", self.workflow)
        self.assertEqual(installed, [f"-{image.group(1)}"])

    def test_black_is_pinned(self):
        installed = re.findall(r"pip install black(\S*)", self.workflow)
        self.assertEqual(len(installed), 1)
        self.assertRegex(installed[0], r"^==\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()
