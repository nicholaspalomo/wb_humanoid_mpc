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


"""The configuration schemas as a package, and their retired fields answered by the strict Python parser.

A module per file, the imports, the struct names, every message reachable from a file message, snake_case fields.
"""

import glob
import importlib
import os
import re
from typing import Any
import unittest

from google.protobuf import descriptor as descriptor_module
from google.protobuf import descriptor_pb2
from humanoid_mpc_config import config_registries_pb2
from humanoid_mpc_config import contact_planning_file_pb2
from humanoid_mpc_config import gait_file_pb2
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import joint_value_pb2
from humanoid_mpc_config import mpc_parameter_update_pb2
from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2
from humanoid_mpc_config import tuning_group_options_pb2
from humanoid_mpc_config import tuning_options_pb2
from humanoid_mpc_config import xyz_pb2
from humanoid_mpc_config import yaw_pitch_roll_pb2
from nproto import options_pb2

import nproto_textproto

PACKAGE = "humanoid_mpc_config"
PROTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STRUCT_NAMESPACE = "ocs2::humanoid::mpc_config::"
# The messages a configuration file or the bus carries, the registries' names the GUI reads, the option messages no
# file holds, and the primitives the blocks share.
ROOTS = (
    task_file_pb2.TaskFile,
    reference_file_pb2.ReferenceFile,
    joint_pd_gains_file_pb2.JointPdGainsFile,
    contact_planning_file_pb2.ContactPlanningFile,
    gait_file_pb2.GaitFile,
    mpc_parameter_update_pb2.MpcParameterUpdate,
    config_registries_pb2.ConfigRegistries,
    tuning_options_pb2.TuningOptions,
    tuning_group_options_pb2.TuningGroupOptions,
    xyz_pb2.Xyz,
    yaw_pitch_roll_pb2.YawPitchRoll,
    joint_value_pb2.JointValue,
)
FILE_MESSAGES = ROOTS[:5]
ALLOWED_IMPORTS = re.compile(
    r"humanoid_mpc_config/[a-z0-9_]+\.proto|nproto/options\.proto|nproto/retired_field_options\.proto|"
    r"google/protobuf/descriptor\.proto"
)
SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(_[a-z0-9]+)*")


def proto_stems() -> list[str]:
    """The file names of the package's .proto files, without the extension."""
    return sorted(
        os.path.basename(path)[: -len(".proto")]
        for path in glob.glob(os.path.join(PROTO_DIR, "*.proto"))
    )


def file_descriptor(stem: str) -> descriptor_module.FileDescriptor:
    """The descriptor of humanoid_mpc_config/<stem>.proto, from its generated module."""
    return importlib.import_module(f"{PACKAGE}.{stem}_pb2").DESCRIPTOR


def all_messages(
    descriptor: descriptor_module.Descriptor,
) -> list[descriptor_module.Descriptor]:
    """`descriptor` and every message nested in it, map entries excluded."""
    found = [descriptor]
    for nested in descriptor.nested_types:
        if not nested.GetOptions().map_entry:
            found += all_messages(nested)
    return found


def reachable(roots: tuple[Any, ...]) -> set[str]:
    """The full names of every message the fields of `roots` reach, the roots included."""
    seen: set[str] = set()
    pending = [root.DESCRIPTOR for root in roots]
    while pending:
        descriptor = pending.pop()
        if descriptor.full_name in seen:
            continue
        seen.add(descriptor.full_name)
        for field in descriptor.fields:
            if field.message_type is not None:
                pending.append(field.message_type)
    return seen


class PackageLayoutTest(unittest.TestCase):
    def test_every_proto_file_has_a_module(self):
        stems = proto_stems()
        self.assertIn("task_file", stems)
        for stem in stems:
            with self.subTest(file=stem):
                self.assertEqual(file_descriptor(stem).package, PACKAGE)

    def test_every_import_is_of_the_package_or_the_options(self):
        for stem in proto_stems():
            for dependency in file_descriptor(stem).dependencies:
                with self.subTest(file=stem, imports=dependency.name):
                    self.assertRegex(dependency.name, ALLOWED_IMPORTS)

    def test_every_message_names_its_struct_in_the_namespace(self):
        for stem in proto_stems():
            for name, message in file_descriptor(stem).message_types_by_name.items():
                with self.subTest(message=name):
                    options = descriptor_pb2.MessageOptions.FromString(
                        message.GetOptions().SerializeToString()
                    )
                    self.assertEqual(
                        options.Extensions[options_pb2.generate_struct],
                        STRUCT_NAMESPACE + name,
                    )

    def test_every_message_is_reachable_from_a_file_message_or_is_an_option(self):
        reached = reachable(ROOTS)
        for stem in proto_stems():
            for message in file_descriptor(stem).message_types_by_name.values():
                for descriptor in all_messages(message):
                    with self.subTest(message=descriptor.full_name):
                        self.assertIn(
                            descriptor.full_name,
                            reached,
                            "no file message holds it; add it to one or drop it",
                        )

    def test_field_names_are_snake_case(self):
        for stem in proto_stems():
            for message in file_descriptor(stem).message_types_by_name.values():
                for descriptor in all_messages(message):
                    for field in descriptor.fields:
                        with self.subTest(field=field.full_name):
                            self.assertRegex(field.name, SNAKE_CASE)


class FileMessagesTest(unittest.TestCase):
    def test_a_file_of_only_its_header_parses(self):
        for message_class in FILE_MESSAGES:
            full_name = message_class.DESCRIPTOR.full_name
            with self.subTest(message=full_name):
                message = message_class()
                text = f"# proto-file: humanoid_nmpc/humanoid_mpc_config/x.proto\n# proto-message: {full_name}\n"
                nproto_textproto.parse_textproto(text, message, "header_only.textproto")
                self.assertEqual(message.ListFields(), [])

    def test_the_retired_keys_name_their_replacement(self):
        retired = dict(
            nproto_textproto.retired_fields(task_file_pb2.TaskFile.DESCRIPTOR)
        )
        for name in (
            "use_com_and_acom_tracking",
            "use_dcm_terminal_cost",
            "use_contact_planning",
            "enable_telemetry",
        ):
            self.assertIn(name, retired)
        with self.assertRaises(nproto_textproto.TextprotoError) as caught:
            nproto_textproto.parse_textproto(
                "useDcmTerminalCost: true\n", task_file_pb2.TaskFile(), "task.textproto"
            )
        self.assertTrue(
            str(caught.exception).startswith(
                "task.textproto:1:1: 'useDcmTerminalCost' is retired: "
            )
        )
        self.assertIn("list dcm_terminal_cost under costs", str(caught.exception))

    def test_the_formulation_switches_that_became_names_are_answered(self):
        cases = (
            (
                "model_settings {\n  foot_constraint {\n    constrain_orientation: true\n  }\n}\n",
                task_file_pb2.TaskFile(),
                "task.textproto:3:5: 'constrain_orientation' is retired: ",
                '"position_and_tilt" where it was true',
            ),
            (
                "centroidal_model_type: 0\n",
                task_file_pb2.TaskFile(),
                "task.textproto:1:1: 'centroidal_model_type' is retired: ",
                'centroidal_model: "full_centroidal_dynamics"',
            ),
            (
                "planner {\n  runInBackgroundThread: true\n}\n",
                contact_planning_file_pb2.ContactPlanningFile(),
                "planning.textproto:2:3: 'runInBackgroundThread' is retired: ",
                'threading: "background_thread"',
            ),
            (
                "multiple_shooting {\n  inequality_constraint_delta: 5.0\n}\n",
                task_file_pb2.TaskFile(),
                "task.textproto:2:3: 'inequality_constraint_delta' is retired: ",
                "nothing in the SQP solver reads it",
            ),
        )
        for text, message, position, replacement in cases:
            with self.subTest(text=text):
                path = position.split(":", maxsplit=1)[0]
                with self.assertRaises(nproto_textproto.TextprotoError) as caught:
                    nproto_textproto.parse_textproto(text, message, path)
                self.assertTrue(str(caught.exception).startswith(position))
                self.assertIn(replacement, str(caught.exception))

    def test_no_retired_name_is_a_live_field(self):
        for stem in proto_stems():
            for message in file_descriptor(stem).message_types_by_name.values():
                for descriptor in all_messages(message):
                    live = {
                        nproto_textproto.snake_case(field.name)
                        for field in descriptor.fields
                    }
                    for name, replacement in nproto_textproto.retired_fields(
                        descriptor
                    ):
                        with self.subTest(message=descriptor.full_name, retired=name):
                            self.assertNotIn(nproto_textproto.snake_case(name), live)
                            self.assertTrue(replacement)


if __name__ == "__main__":
    unittest.main()
