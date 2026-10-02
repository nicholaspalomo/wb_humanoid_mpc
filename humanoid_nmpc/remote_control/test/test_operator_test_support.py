"""The Tk display rule of operator_test_support: a display, else an Xvfb of the tests' own, else a skip - but not on CI.

A Tk test that is skipped passes, so a CI without a display would pass every Tk test without running one.
"""

import os
import shutil
import unittest
from unittest import mock

import operator_test_support


class _Probe(unittest.TestCase):
    def test_nothing(self) -> None:
        pass


class RequiresDisplayTest(unittest.TestCase):
    def test_with_a_display_the_tests_run(self) -> None:
        with mock.patch.object(
            operator_test_support, "display_available", return_value=True
        ):
            decorated = operator_test_support.requires_display(_Probe)
        self.assertIs(decorated, _Probe)
        self.assertFalse(getattr(decorated, "__unittest_skip__", False))

    def test_without_one_they_are_skipped_off_ci(self) -> None:
        with mock.patch.object(
            operator_test_support, "display_available", return_value=False
        ), mock.patch.dict(os.environ, {"CI": ""}):

            class Skipped(unittest.TestCase):
                def test_nothing(self) -> None:
                    pass

            decorated = operator_test_support.requires_display(Skipped)
        self.assertTrue(decorated.__unittest_skip__)
        self.assertIn("Xvfb", decorated.__unittest_skip_why__)

    def test_without_one_they_fail_on_ci(self) -> None:
        with mock.patch.object(
            operator_test_support, "display_available", return_value=False
        ), mock.patch.dict(os.environ, {"CI": "true"}):
            with self.assertRaises(RuntimeError) as raised:
                operator_test_support.requires_display(_Probe)
        self.assertIn("xvfb", str(raised.exception))


class ReadDisplayNumberTest(unittest.TestCase):
    def _pipe(self):
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        return read_fd, write_fd

    def test_the_number_xvfb_writes_is_read(self) -> None:
        read_fd, write_fd = self._pipe()
        os.write(write_fd, b"42\n")
        os.close(write_fd)
        self.assertEqual(
            operator_test_support._read_display_number(read_fd, timeout=5.0), 42
        )

    def test_an_xvfb_that_exits_before_writing_gives_none(self) -> None:
        read_fd, write_fd = self._pipe()
        os.close(write_fd)
        self.assertIsNone(
            operator_test_support._read_display_number(read_fd, timeout=5.0)
        )

    def test_a_silent_xvfb_times_out(self) -> None:
        read_fd, write_fd = self._pipe()
        self.addCleanup(os.close, write_fd)
        self.assertIsNone(
            operator_test_support._read_display_number(read_fd, timeout=0.05)
        )


@unittest.skipIf(shutil.which("Xvfb") is None, "Xvfb is not installed here")
class PrivateDisplayTest(unittest.TestCase):
    def test_an_xvfb_of_its_own_gives_tk_a_display(self) -> None:
        with mock.patch.dict(os.environ, {"DISPLAY": ""}):
            self.assertTrue(operator_test_support._start_private_display())
            self.assertRegex(os.environ["DISPLAY"], r"^:\d+$")
            self.assertTrue(operator_test_support._tk_opens_a_window())


if __name__ == "__main__":
    unittest.main()
