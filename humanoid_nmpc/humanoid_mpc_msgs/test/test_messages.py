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

"""The generated Python messages import with the operator tools' protobuf runtime and round-trip.

Every .proto file of the package has a generated module that loads, which also checks that its gencode version
matches the protobuf wheel of @operator_deps (the generated code refuses an incompatible runtime at import). The
messages travel over ZeroMQ in the bus's three-frame layout (humanoid_nmpc/docs/distributed_runtime/README.md), and
the Rerun SDK of the viewer bridge loads next to them.
"""

import glob
import importlib
import os
import re
import types
import unittest

import google.protobuf
from google.protobuf.internal import api_implementation
from humanoid_mpc_msgs import controller_type_pb2
from humanoid_mpc_msgs import mpc_policy_pb2
from humanoid_mpc_msgs import target_contact_patch_pb2
from nproto import options_pb2
import rerun
import zmq

PACKAGE = "humanoid_mpc_msgs"
# The one import from outside the package: the options that name each message's nproto struct.
NPROTO_OPTIONS = "nproto/options.proto"
STRUCT_NAMESPACE = "ocs2::humanoid::msgs"
# The .proto files are data of this test, next to its directory in the runfiles tree.
PROTO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIME_NODES = 4
STATE_DIM = 3


def proto_stems() -> list[str]:
    return sorted(
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(PROTO_DIR, "*.proto"))
    )


def snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def make_policy() -> mpc_policy_pb2.MpcPolicy:
    """A policy that sets a field of every kind MpcPolicy has."""
    policy = mpc_policy_pb2.MpcPolicy(resets_served=3, full_resets_served=1)
    policy.solver_status.healthy = True
    policy.solver_status.solve_time_ms = 4.5
    policy.init_observation.time = 1.25
    policy.init_observation.state.append(0.5)
    policy.target_trajectories.time.append(1.25)
    for node in range(TIME_NODES):
        policy.time_trajectory.append(1.25 + 0.01 * node)
        policy.state_trajectory.add().data.extend(
            float(node * STATE_DIM + i) for i in range(STATE_DIM)
        )
        policy.input_trajectory.add().data.append(-float(node))
        policy.controller_data.add().data.append(node + 0.5)
    policy.post_event_indices.append(2)
    policy.mode_schedule.event_times.append(1.27)
    policy.mode_schedule.mode_sequence.extend([3, 1])
    policy.controller_type = controller_type_pb2.CONTROLLER_TYPE_LINEAR
    policy.performance.merit = 10.0
    patch = policy.annotations.target_contact_patches.add()
    patch.valid = True
    patch.kind = target_contact_patch_pb2.TargetContactPatch.KIND_NEXT_SWING
    return policy


class GeneratedModulesTest(unittest.TestCase):

    def setUp(self) -> None:
        self.stems = proto_stems()
        self.assertTrue(self.stems, f"no .proto files found in {PROTO_DIR}")
        self.modules: dict[str, types.ModuleType] = {
            stem: importlib.import_module(f"{PACKAGE}.{stem}_pb2")
            for stem in self.stems
        }

    def test_every_proto_file_has_a_module_named_after_its_one_definition(self) -> None:
        for stem, module in self.modules.items():
            with self.subTest(proto=stem):
                file_descriptor = module.DESCRIPTOR
                self.assertEqual(file_descriptor.name, f"{PACKAGE}/{stem}.proto")
                self.assertEqual(file_descriptor.package, PACKAGE)
                definitions = list(file_descriptor.message_types_by_name) + list(
                    file_descriptor.enum_types_by_name
                )
                self.assertEqual(len(definitions), 1, definitions)
                self.assertEqual(snake_case(definitions[0]), stem)

    def test_every_import_is_a_file_of_the_package_or_the_nproto_options(self) -> None:
        files = {f"{PACKAGE}/{stem}.proto" for stem in self.stems} | {NPROTO_OPTIONS}
        for stem, module in self.modules.items():
            for dependency in module.DESCRIPTOR.dependencies:
                with self.subTest(proto=stem, imports=dependency.name):
                    self.assertIn(dependency.name, files)

    def test_every_definition_names_its_nproto_struct(self) -> None:
        # tools/nproto/README.md: the C++ type nproto generates, in the namespace of the package's structs.
        for stem, module in self.modules.items():
            file_descriptor = module.DESCRIPTOR
            for name, message in file_descriptor.message_types_by_name.items():
                with self.subTest(proto=stem, message=name):
                    self.assertEqual(
                        message.GetOptions().Extensions[options_pb2.generate_struct],
                        f"{STRUCT_NAMESPACE}::{name}",
                    )
            for name, enum in file_descriptor.enum_types_by_name.items():
                with self.subTest(proto=stem, enum=name):
                    self.assertEqual(
                        enum.GetOptions().Extensions[options_pb2.generate_enum],
                        f"{STRUCT_NAMESPACE}::{name}",
                    )


class RuntimeTest(unittest.TestCase):

    def test_runtime_is_the_upb_extension_of_the_operator_wheel(self) -> None:
        # //tools/python:python_proto_toolchain links the gencode against @operator_deps' protobuf, not protobuf's
        # pure-Python runtime built from source.
        self.assertEqual(api_implementation.Type(), "upb")
        self.assertIn("operator_deps", google.protobuf.__file__)


class MpcPolicyTest(unittest.TestCase):

    def test_serialize_then_parse_reproduces_the_message(self) -> None:
        policy = make_policy()
        payload = policy.SerializeToString()
        self.assertTrue(payload)
        parsed = mpc_policy_pb2.MpcPolicy.FromString(payload)
        self.assertEqual(parsed, policy)
        self.assertEqual(len(parsed.state_trajectory), TIME_NODES)
        self.assertEqual(len(parsed.state_trajectory[0].data), STATE_DIM)

    def test_type_name_is_the_full_protobuf_name(self) -> None:
        # The second frame of every bus message; tools/ipc decodes topics by it.
        self.assertEqual(
            mpc_policy_pb2.MpcPolicy.DESCRIPTOR.full_name, "humanoid_mpc_msgs.MpcPolicy"
        )


class ZeroMqTest(unittest.TestCase):

    def test_carries_a_policy_in_three_frames(self) -> None:
        context = zmq.Context()
        receiver = context.socket(zmq.PAIR)
        sender = context.socket(zmq.PAIR)
        try:
            for socket in (receiver, sender):
                socket.setsockopt(zmq.LINGER, 0)
            receiver.setsockopt(zmq.RCVTIMEO, 5000)
            receiver.bind("inproc://humanoid_mpc_msgs_test")
            sender.connect("inproc://humanoid_mpc_msgs_test")

            policy = make_policy()
            type_name = policy.DESCRIPTOR.full_name.encode()
            sender.send_multipart(
                [b"mpc/policy", type_name, policy.SerializeToString()]
            )
            frames = receiver.recv_multipart()
        finally:
            sender.close()
            receiver.close()
            context.term()

        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0], b"mpc/policy")
        self.assertEqual(frames[1], type_name)
        self.assertEqual(mpc_policy_pb2.MpcPolicy.FromString(frames[2]), policy)


class RerunTest(unittest.TestCase):

    def test_rerun_sdk_loads_with_the_archetypes_the_bridge_maps_onto(self) -> None:
        for archetype in ("Arrows3D", "LineStrips3D", "Points3D", "Transform3D"):
            with self.subTest(archetype=archetype):
                self.assertTrue(hasattr(rerun, archetype))


if __name__ == "__main__":
    unittest.main()
