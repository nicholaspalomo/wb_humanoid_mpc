"""Tests for worktree_state.sh, the worktree state the closed-loop runs and the benchmark record as provenance."""

import os
import shutil
import subprocess
import tempfile
import unittest


def _script():
    """worktree_state.sh, from the Bazel runfiles when run by Bazel and from the source tree otherwise."""
    roots = []
    if "TEST_SRCDIR" in os.environ:
        roots.append(
            os.path.join(
                os.environ["TEST_SRCDIR"],
                "_main",
                "humanoid_nmpc",
                "humanoid_mpc_validation",
                "tools",
            )
        )
    roots.append(os.path.dirname(os.path.abspath(__file__)))
    for root in roots:
        candidate = os.path.join(root, "worktree_state.sh")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError("worktree_state.sh")


def _git(directory, *arguments):
    subprocess.run(["git", *arguments], cwd=directory, check=True, capture_output=True)


@unittest.skipIf(shutil.which("git") is None, "git is not installed")
class WorktreeStateTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
        _git(self.directory, "init", "-q")
        _git(self.directory, "config", "user.email", "test@example.com")
        _git(self.directory, "config", "user.name", "test")
        self._write("tracked.txt", "a\n")
        _git(self.directory, "add", "tracked.txt")
        _git(self.directory, "commit", "-q", "-m", "initial")

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _write(self, relative_path, contents):
        path = os.path.join(self.directory, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(contents)

    def _state(self, directory=None):
        result = subprocess.run(
            ["sh", _script()],
            cwd=directory or self.directory,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_a_committed_worktree_is_clean(self):
        self.assertEqual(self._state(), "clean")

    def test_outside_a_worktree_the_state_is_unknown(self):
        outside = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
        try:
            self.assertEqual(self._state(outside), "unknown")
        finally:
            shutil.rmtree(outside, ignore_errors=True)

    def test_the_contents_of_an_untracked_file_change_the_hash(self):
        # A new directory: git status --porcelain alone would print one "?? new/" line whatever the files hold.
        self._write("new/deeper/source.cpp", "int a = 1;\n")
        first = self._state()
        self.assertTrue(first.startswith("1 changed paths, diff sha256 "), first)
        self._write("new/deeper/source.cpp", "int a = 2;\n")
        self.assertNotEqual(self._state(), first)
        self._write("new/deeper/source.cpp", "int a = 1;\n")
        self.assertEqual(self._state(), first, "the same contents give the same hash")

    def test_every_untracked_file_is_counted_and_ignored_files_are_not(self):
        self._write("new/one.txt", "1\n")
        self._write("new/two.txt", "2\n")
        self._write(".gitignore", "*.log\n")
        self.assertTrue(self._state().startswith("3 changed paths"), self._state())
        before = self._state()
        self._write("build.log", "noise\n")
        self.assertEqual(self._state(), before)

    def test_an_edit_of_a_tracked_file_changes_the_hash(self):
        self._write("tracked.txt", "b\n")
        first = self._state()
        self._write("tracked.txt", "c\n")
        self.assertNotEqual(self._state(), first)


if __name__ == "__main__":
    unittest.main()
