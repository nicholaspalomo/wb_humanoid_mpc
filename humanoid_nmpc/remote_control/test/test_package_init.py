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

"""The remote_control package re-exports nothing, so that importing it loads none of its modules.

Its modules are imported by name (`from remote_control import operator_bus`, Python style 2.2.4). An `__init__.py`
that imported them would load Tk, pygame and the bus for anyone who imports any one module, and would put a second
spelling of every name next to the module's own.
"""

import os
import subprocess
import sys
import unittest

import remote_control

_LOADED_SUBMODULES = (
    "import sys, remote_control, remote_control.tk_app; "
    "print(sorted(m for m in sys.modules if m.startswith('remote_control.') and m != 'remote_control.tk_app'))"
)


class TestPackageInit(unittest.TestCase):
    def test_importing_the_package_loads_none_of_its_modules(self):
        # A fresh interpreter: this one has loaded what the other tests import.
        result = subprocess.run(
            [sys.executable, "-c", _LOADED_SUBMODULES],
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(result.stdout.strip(), "[]", result.stderr)

    def test_the_package_has_no_attributes_but_its_own(self):
        public = [name for name in vars(remote_control) if not name.startswith("_")]
        # Submodules other tests imported become attributes; nothing else may be one.
        for name in public:
            with self.subTest(name=name):
                self.assertTrue(
                    os.path.exists(
                        os.path.join(os.path.dirname(remote_control.__file__), name)
                    )
                    or os.path.exists(
                        os.path.join(
                            os.path.dirname(remote_control.__file__), name + ".py"
                        )
                    ),
                    f"remote_control.{name} is not a module of the package",
                )


if __name__ == "__main__":
    unittest.main()
