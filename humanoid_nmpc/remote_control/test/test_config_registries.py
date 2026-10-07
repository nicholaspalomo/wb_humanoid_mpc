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


"""The registries' names in the tuning GUI (humanoid_nmpc/humanoid_mpc_config/config_registries.textproto).

The file lists every registry the schemas name, and nothing else; every name a shipped file gives a registry string is
one the registry offers; and with the file, the tabs show a registry string as a drop-down of its registry's names.
That the file is what the registries themselves accept is //tools/config_registries:config_registries_test.
"""

import os
import unittest

from humanoid_mpc_config import config_registries_pb2
from humanoid_mpc_config import contact_planning_file_pb2
from humanoid_mpc_config import gait_file_pb2
from humanoid_mpc_config import joint_pd_gains_file_pb2
from humanoid_mpc_config import reference_file_pb2
from humanoid_mpc_config import task_file_pb2

import nproto_textproto
import operator_test_support
from remote_control import config_files
from remote_control import config_schema
from remote_control import tuned_file

FILE_SCHEMAS = (
    task_file_pb2.TaskFile,
    reference_file_pb2.ReferenceFile,
    joint_pd_gains_file_pb2.JointPdGainsFile,
    contact_planning_file_pb2.ContactPlanningFile,
    gait_file_pb2.GaitFile,
)
# The files of a robot's config/ directory, with their schemas.
ROBOT_FILE_SCHEMAS = (
    (os.path.join("mpc", "task.textproto"), task_file_pb2.TaskFile),
    (
        os.path.join("mpc", "contact_planning.textproto"),
        contact_planning_file_pb2.ContactPlanningFile,
    ),
    (os.path.join("command", "reference.textproto"), reference_file_pb2.ReferenceFile),
    (
        os.path.join("controller", "joint_pd_gains.textproto"),
        joint_pd_gains_file_pb2.JointPdGainsFile,
    ),
)
REGISTRIES_FILE = operator_test_support.repo_path(config_files.REGISTRIES_FILE)
REGISTRY_KINDS = (config_schema.Kind.CHOICE, config_schema.Kind.NAME_LIST)


def _registries() -> dict[str, tuple[str, ...]]:
    return config_files.load_registries(REGISTRIES_FILE)


def _shipped_files() -> list[tuned_file.TunedFile]:
    """Every robot's configuration files that exist, with the registries' names."""
    files = []
    for config in sorted(operator_test_support.ROBOT_CONFIGS.values()):
        for relative, message_class in ROBOT_FILE_SCHEMAS:
            path = os.path.join(config, relative)
            if os.path.exists(path):
                files.append(tuned_file.TunedFile(path, message_class, _registries()))
    return files


class TestTheFileNamesTheSchemasRegistries(unittest.TestCase):
    def test_the_file_parses_strictly_into_its_schema(self):
        registries = nproto_textproto.load_textproto(
            REGISTRIES_FILE, config_registries_pb2.ConfigRegistries
        )
        self.assertTrue(registries.registries)

    def test_every_registry_a_schema_names_is_listed_and_nothing_else(self):
        named: set[str] = set()
        for message_class in FILE_SCHEMAS:
            fields = config_schema.schema_fields(message_class.DESCRIPTOR)
            named.update(
                field.tuning.registry for field in fields if field.tuning.registry
            )
        self.assertTrue(named)
        self.assertEqual(set(_registries()), named)

    def test_every_name_a_shipped_file_gives_a_registry_string_is_offered(self):
        registries = _registries()
        files = _shipped_files()
        self.assertTrue(files)
        for file in files:
            for spec in file.parameters():
                if not spec.tuning.registry or spec.value is None:
                    continue
                with self.subTest(file=file.path, path=spec.path):
                    values = (
                        spec.value if isinstance(spec.value, tuple) else (spec.value,)
                    )
                    for value in values:
                        if not value:
                            continue  # An empty name selects none (sim_projectile: no ball).
                        self.assertIn(value, registries[spec.tuning.registry])


class TestARegistryStringIsADropDown(unittest.TestCase):
    def test_its_choices_are_the_registrys_names(self):
        registries = _registries()
        for file in _shipped_files():
            for spec in file.parameters():
                if not spec.tuning.registry:
                    continue
                with self.subTest(file=file.path, path=spec.path):
                    self.assertIn(spec.kind, REGISTRY_KINDS)
                    self.assertEqual(spec.choices, registries[spec.tuning.registry])

    def test_the_contact_estimator_is_a_choice_of_every_estimator(self):
        task = tuned_file.TunedFile(
            operator_test_support.ATLAS_TASK_FILE, task_file_pb2.TaskFile, _registries()
        )
        spec = task.spec("contact_estimator")
        self.assertEqual(spec.kind, config_schema.Kind.CHOICE)
        self.assertIn("cheater_sim", spec.choices or ())
        self.assertIn(spec.value, spec.choices or ())

    def test_without_the_file_there_are_no_choices_and_nothing_fails(self):
        self.assertEqual(
            config_files.load_registries(os.path.join("no", "such", "file.textproto")),
            {},
        )
        task = tuned_file.TunedFile(
            operator_test_support.ATLAS_TASK_FILE, task_file_pb2.TaskFile, {}
        )
        self.assertIsNone(task.spec("contact_estimator").choices)


if __name__ == "__main__":
    unittest.main()
