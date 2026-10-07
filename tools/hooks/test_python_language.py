# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Tests for python_language.py: the Python language rules pylint does not check."""

import os
import stat
import tempfile
import unittest
from unittest import mock

from tools.hooks import check_test_support
from tools.hooks import python_language as language


def _count(check, source: str, path: str = "src/a.py") -> int:
    return len(check(source, path))


class LanguageTest(unittest.TestCase):
    def assert_cases(self, check, flagged: list[str], clean: list[str]) -> None:
        """Asserts one finding for each source of `flagged` and none for each of `clean`."""
        for source in flagged:
            with self.subTest(flagged=source):
                self.assertEqual(_count(check, source), 1)
        for source in clean:
            with self.subTest(clean=source):
                self.assertEqual(_count(check, source), 0)

    def test_staticmethod(self):
        self.assert_cases(
            language.check_staticmethod,
            ["class A:\n    @staticmethod\n    def f():\n        pass\n"],
            [
                "class A:\n    @classmethod\n    def f(cls):\n        pass\n",
                "def f():\n    pass\n",
            ],
        )

    def test_complex_comprehension(self):
        self.assert_cases(
            language.check_complex_comprehension,
            [
                "x = [a for row in rows for a in row]\n",
                "x = [a for a in b if a if not c]\n",
            ],
            ["x = [a for a in b if a]\n", "x = {k: v for k, v in d.items()}\n"],
        )

    def test_long_ternary(self):
        self.assert_cases(
            language.check_long_ternary,
            ["x = (\n    f(a,\n      b)\n    if c\n    else d\n)\n"],
            ["x = a if c else d\n", "x = (\n    a\n    if c\n    else d\n)\n"],
        )

    def test_long_lambda(self):
        self.assert_cases(
            language.check_long_lambda,
            [
                "f = lambda: (\n    1\n)\n",
                "f = lambda x: " + " + ".join(["x"] * 30) + "\n",
            ],
            ["f = lambda x: x + 1\n"],
        )

    def test_length_test(self):
        self.assert_cases(
            language.check_length_test,
            [
                "if len(x) == 0:\n    pass\n",
                "while len(x) > 0:\n    pass\n",
                "y = a if len(x) != 0 else b\n",
            ],
            [
                "if not x:\n    pass\n",
                "n = len(x) == 0\n",
                "if len(x) == 3:\n    pass\n",
            ],
        )

    def test_assert(self):
        self.assert_cases(
            language.check_assert,
            ["assert x is not None\n"],
            ["if x is None:\n    raise ValueError(x)\n"],
        )
        check = check_test_support.assert_registered(self, "py-assert")
        self.assertFalse(check.applies_to("tools/test_x.py"))

    def test_exception_name(self):
        self.assert_cases(
            language.check_exception_name,
            [
                "class MalformedMessage(ValueError):\n    pass\n",
                "class Failure(Exception):\n    pass\n",
            ],
            [
                "class MalformedMessageError(ValueError):\n    pass\n",
                "class Widget(Base):\n    pass\n",
            ],
        )

    def test_power_feature(self):
        self.assert_cases(
            language.check_power_feature,
            [
                "class A:\n    def __del__(self):\n        pass\n",
                "class A(metaclass=M):\n    pass\n",
                "exec(code)\n",
                "eval(s)\n",
            ],
            [
                "class A:\n    def close(self):\n        pass\n",
                "x = executor.submit(f)\n",
            ],
        )

    def test_backslash(self):
        self.assert_cases(
            language.check_backslash,
            ["x = 1 + \\\n    2\n"],
            [
                'x = (1 +\n     2)\ns = "a\\\nb"\n',
                "# a comment \\\nx = 1\n",
                's = """\\\nline\n"""\n',
            ],
        )

    def test_type_comment(self):
        self.assert_cases(
            language.check_type_comment,
            ["x = []  # type: List[int]\n", "x = f()  # type: ignore\n"],
            ["x = f()  # type: ignore[attr-defined]\n", "x: list[int] = []\n"],
        )

    def test_pragma_reason(self):
        self.assert_cases(
            language.check_pragma_reason,
            [
                "x = f()  # type: ignore[attr-defined]\n",
                "x = f()  # type: ignore[attr-defined]  # NOLINT(py-type-comment)\n",
                "except Exception:  # pylint: disable=broad-exception-caught\n",
                "import h5py  # pylint: disable=import-outside-toplevel\n",
                "\n# pylint: disable=wrong-import-position\nimport a\n",
                "# pylint: disable-next=consider-using-with\nproc = subprocess.Popen(cmd)\n",
                "# pylint: skip-file\n",
                "# pylint: skip-file  # A generated file.\n",
            ],
            [
                "x = f()  # type: ignore[attr-defined]  # The stub lacks it.\n",
                "except Exception:  # pylint: disable=broad-exception-caught  # A Tk callback must not raise.\n",
                "# pylint: disable=wrong-import-position  # JAX_PLATFORMS is set above.\nimport a\n",
                "# JAX_PLATFORMS is set above, before JAX is imported.\n# pylint: disable=wrong-import-position\nimport a\n",
                # The black-split form: the pragma above a statement black wrapped onto several lines.
                "# pylint: disable-next=consider-using-with  # tearDown() deletes it.\nproc = subprocess.Popen(\n    cmd,\n)\n",
                "x = 1  # pylint: enable=invalid-name\n",
                's = "# type: ignore[x] in a string"\n',
                "x = f()  # type: ignore[attr-defined, arg-type]  # Both are the stub's.\n",
                "x = 1  # pylint: disable=invalid-name,protected-access  # The paper's notation.\n",
            ],
        )
        # A pragma is not the reason of the pragma below it.
        self.assertEqual(
            _count(
                language.check_pragma_reason,
                "# pylint: disable-next=protected-access\n# pylint: disable-next=invalid-name\nx = a._b\n",
            ),
            2,
        )

    def test_main_guard(self):
        self.assert_cases(
            language.check_main_guard,
            ['if __name__ == "__main__":\n    run()\n', "print('at import')\n"],
            [
                'def main():\n    pass\n\n\nif __name__ == "__main__":\n    main()\n',
                'if __name__ == "__main__":\n    sys.exit(main())\n',
                'if __name__ == "__main__":\n    unittest.main()\n',
                'if __name__ == "__main__":\n    app.run(main)\n',
                "import sys\nsys.path.insert(0, 'x')\n",
            ],
        )


class ShebangTest(unittest.TestCase):
    def setUp(self):
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self.directory = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(language, "ROOT", self.directory.name)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.directory.cleanup()

    def write(self, name: str, text: str, executable: bool) -> str:
        """Writes a file below ROOT, executable or not, and returns its relative path."""
        path = os.path.join(self.directory.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        mode = os.stat(path).st_mode
        os.chmod(
            path,
            (
                mode | stat.S_IXUSR
                if executable
                else mode & ~stat.S_IXUSR & ~stat.S_IXGRP & ~stat.S_IXOTH
            ),
        )
        return name

    def test_a_shebang_exactly_on_executables(self):
        self.assertEqual(
            _count(
                language.check_shebang, "x = 1\n", self.write("a.py", "x = 1\n", False)
            ),
            0,
        )
        good = "#!/usr/bin/env python3\nx = 1\n"
        self.assertEqual(
            _count(language.check_shebang, good, self.write("b.py", good, True)), 0
        )
        self.assertEqual(
            _count(language.check_shebang, good, self.write("c.py", good, False)), 1
        )
        self.assertEqual(
            _count(
                language.check_shebang, "x = 1\n", self.write("d.py", "x = 1\n", True)
            ),
            1,
        )
        other = "#!/usr/bin/python3\nx = 1\n"
        self.assertEqual(
            _count(language.check_shebang, other, self.write("e.py", other, True)), 1
        )
        self.assertEqual(
            _count(language.check_shebang, good, "missing.py"),
            0,
            "an unknown file is not judged",
        )

    def test_the_registry_runs_it(self):
        check_test_support.assert_registered(self, "py-shebang")


class RegistryTest(unittest.TestCase):
    def test_every_check_behaves(self):
        for name, flagged in [
            (
                "py-staticmethod",
                "class A:\n    @staticmethod\n    def f():\n        pass\n",
            ),
            ("py-complex-comprehension", "x = [a for row in rows for a in row]\n"),
            ("py-long-lambda", "f = lambda x: " + " + ".join(["x"] * 30) + "\n"),
            ("py-length-test", "if len(x) == 0:\n    pass\n"),
            ("py-assert", "assert x\n"),
            ("py-exception-name", "class Failure(Exception):\n    pass\n"),
            ("py-power-feature", "eval(s)\n"),
            ("py-type-comment", "x = []  # type: List[int]\n"),
            ("py-pragma-reason", "x = f()  # type: ignore[attr-defined]\n"),
            ("py-main-guard", "print('at import')\n"),
        ]:
            with self.subTest(check=name):
                check_test_support.assert_check_behaves(self, name, flagged, "src/a.py")

    def test_a_long_ternary_behaves(self):
        check_test_support.assert_check_behaves(
            self,
            "py-long-ternary",
            "x = (\n    f(a,\n      b)\n    if c\n    else d\n)\n",
            "src/a.py",
        )

    def test_the_backslash_check_is_registered(self):
        # A marker cannot follow a continuation backslash on its line, so only the registration is shared.
        check = check_test_support.assert_registered(self, "py-backslash")
        self.assertTrue(check.applies_to("src/a.py"))
        self.assertEqual(
            check_test_support.findings(
                "py-backslash", "x = 1 + \\\n    2\n", "src/a.py"
            )[0].check,
            "py-backslash",
        )


if __name__ == "__main__":
    unittest.main()
