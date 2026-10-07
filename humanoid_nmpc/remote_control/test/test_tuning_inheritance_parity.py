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

"""The tuning options pass down a schema alike in C++ and in Python.

The parameter updater and the reload-class coverage test read the (humanoid_mpc_config.tuning) options of a field with
configLeaves() (humanoid_common_mpc/config/ConfigReload.h); the tuning GUI with config_schema.schema_fields(). Each
implements how a block's options pass down to its fields (reload, consumer and formulations inherited, the registry a
field's own) and which fields the GUI renders. This test compares the two walks field by field, on every configuration
file, through a C++ binary that prints what configLeaves() finds (test/ConfigLeavesMain.cpp).
"""

import os
import subprocess
import unittest

from humanoid_mpc_config import contact_planning_file_pb2
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2
from humanoid_mpc_config import tuning_options_pb2

from remote_control import config_schema

CONFIG_LEAVES_BINARY = os.environ["CONFIG_LEAVES_BINARY"]

# The file messages ConfigLeavesMain.cpp prints, in its order.
# LINT.IfChange(file_messages)
_FILE_MESSAGES = (
    task_file_pb2.TaskFile,
    reference_file_pb2.ReferenceFile,
    contact_planning_file_pb2.ContactPlanningFile,
    joint_pd_gains_file_pb2.JointPdGainsFile,
)
# LINT.ThenChange(//humanoid_nmpc/remote_control/test/ConfigLeavesMain.cpp:file_messages)


def _python_leaves() -> list[str]:
    """The lines ConfigLeavesMain.cpp would print, from config_schema's walk."""
    lines = []
    for message in _FILE_MESSAGES:
        for field in config_schema.schema_fields(message.DESCRIPTOR):
            reload = tuning_options_pb2.TuningOptions.Reload.Name(field.tuning.reload)
            formulations = ",".join(field.tuning.formulations)
            tunable = 1 if field.renders else 0
            lines.append(
                f"{message.DESCRIPTOR.full_name}\t{field.path}\t{reload}"
                f"\t{field.tuning.consumer}\t{formulations}\t{tunable}"
            )
    return lines


class TuningInheritanceParityTest(unittest.TestCase):
    """configLeaves() and config_schema.schema_fields() read the same options."""

    def test_both_walks_find_the_same_fields_with_the_same_options(self) -> None:
        printed = subprocess.run(
            [CONFIG_LEAVES_BINARY], check=True, capture_output=True, text=True
        ).stdout
        cpp = printed.splitlines()
        python = _python_leaves()
        self.assertGreater(len(cpp), 300, "the C++ walk found too few fields")
        self.assertEqual(
            sorted(set(cpp) - set(python)), [], "fields the C++ walk reads otherwise"
        )
        self.assertEqual(
            sorted(set(python) - set(cpp)), [], "fields the Python walk reads otherwise"
        )
        self.assertEqual(cpp, python, "the two walks visit the fields in another order")

    def test_the_walks_see_the_inherited_options(self) -> None:
        # Positive control: the comparison covers a field that inherits each option from its block.
        python = _python_leaves()
        self.assertIn(
            "humanoid_mpc_config.TaskFile\tcom_weights.x\tRELOAD_HOT\t\tcentroidal\t1",
            python,
        )
        self.assertTrue(
            any(line.split("\t")[3] == "robot" for line in python),
            "no field names a consumer",
        )


if __name__ == "__main__":
    unittest.main()
