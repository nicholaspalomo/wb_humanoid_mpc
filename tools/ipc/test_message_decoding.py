"""Tests for message_decoding.py: decoding by type name, field paths and the printed formats."""

import json
import unittest
from typing import Type

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf import json_format
from google.protobuf import message
from google.protobuf import message_factory
from google.protobuf import text_format

import message_decoding
from humanoid_mpc_msgs import controller_type_pb2
from humanoid_mpc_msgs import joint_targets_pb2
from humanoid_mpc_msgs import mpc_policy_pb2
from humanoid_mpc_msgs import mpc_solver_status_pb2
from humanoid_mpc_msgs import mpc_status_pb2


def make_policy() -> mpc_policy_pb2.MpcPolicy:
    policy = mpc_policy_pb2.MpcPolicy(resets_served=3)
    policy.solver_status.healthy = True
    policy.solver_status.solve_time_ms = 4.5
    for node in range(3):
        policy.time_trajectory.append(0.01 * node)
        policy.state_trajectory.add().data.extend([float(node), float(node) + 0.5])
    policy.controller_type = controller_type_pb2.CONTROLLER_TYPE_LINEAR
    return policy


def make_choice_class() -> Type[message.Message]:
    """A message with a oneof and a proto3 `optional` field, which humanoid_mpc_msgs does not have (yet)."""
    file_proto = descriptor_pb2.FileDescriptorProto(
        name="test_choice.proto", package="test_choice", syntax="proto3"
    )
    choice = file_proto.message_type.add(name="Choice")
    choice.oneof_decl.add(name="kind")
    # A proto3 `optional` field is the only member of a synthetic oneof, declared after the real ones.
    choice.oneof_decl.add(name="_count")
    field_proto = descriptor_pb2.FieldDescriptorProto
    choice.field.add(
        name="label",
        number=1,
        type=field_proto.TYPE_STRING,
        label=field_proto.LABEL_OPTIONAL,
        oneof_index=0,
    )
    choice.field.add(
        name="value",
        number=2,
        type=field_proto.TYPE_DOUBLE,
        label=field_proto.LABEL_OPTIONAL,
        oneof_index=0,
    )
    choice.field.add(
        name="count",
        number=3,
        type=field_proto.TYPE_UINT32,
        label=field_proto.LABEL_OPTIONAL,
        oneof_index=1,
        proto3_optional=True,
    )
    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName("test_choice.Choice")
    )


def parse_text(text: str, cls: Type[message.Message]) -> message.Message:
    """The text parsed strictly, as the textproto it is meant to be."""
    return text_format.Parse(text, cls())


class ImportMessageModulesTest(unittest.TestCase):

    def test_every_message_of_every_module_resolves_by_its_full_name(self) -> None:
        modules = message_decoding.import_message_modules()
        self.assertIn("humanoid_mpc_msgs.mpc_policy_pb2", modules)
        self.assertTrue(all(name.endswith("_pb2") for name in modules))
        for module_name in modules:
            module = __import__(module_name, fromlist=["DESCRIPTOR"])
            for descriptor in module.DESCRIPTOR.message_types_by_name.values():
                with self.subTest(message=descriptor.full_name):
                    cls = message_decoding.message_class(descriptor.full_name)
                    self.assertEqual(cls.DESCRIPTOR.full_name, descriptor.full_name)


class DecodeTest(unittest.TestCase):

    def test_a_payload_decodes_as_the_type_its_frame_names(self) -> None:
        status = mpc_status_pb2.MpcStatus(observation_time=1.5, resets_served=2)
        decoded = message_decoding.decode(
            "humanoid_mpc_msgs.MpcStatus", status.SerializeToString()
        )
        self.assertEqual(decoded, status)

    def test_an_unknown_type_name_raises(self) -> None:
        with self.assertRaises(message_decoding.UnknownMessageTypeError):
            message_decoding.decode("humanoid_mpc_msgs.NoSuchMessage", b"")


class MessageToPythonTest(unittest.TestCase):

    def test_fields_at_their_default_are_included(self) -> None:
        value = message_decoding.message_to_python(mpc_status_pb2.MpcStatus())
        self.assertEqual(value["observation_time"], 0.0)
        self.assertEqual(value["solver_status"]["healthy"], False)
        self.assertEqual(
            list(value), [f.name for f in mpc_status_pb2.MpcStatus.DESCRIPTOR.fields]
        )

    def test_enums_print_by_name_and_repeated_fields_as_lists(self) -> None:
        value = message_decoding.message_to_python(make_policy())
        self.assertEqual(value["controller_type"], "CONTROLLER_TYPE_LINEAR")
        self.assertEqual(value["state_trajectory"][1], {"data": [1.0, 1.5]})
        self.assertEqual(value["time_trajectory"], [0.0, 0.01, 0.02])


class SelectFieldTest(unittest.TestCase):

    def setUp(self) -> None:
        self.policy = make_policy()

    def test_paths_reach_nested_scalars_repeated_elements_and_whole_fields(
        self,
    ) -> None:
        cases = {
            "solver_status.healthy": True,
            "solver_status.solve_time_ms": 4.5,
            "state_trajectory.0.data": [0.0, 0.5],
            "state_trajectory.-1.data.1": 2.5,
            "time_trajectory": [0.0, 0.01, 0.02],
            "controller_type": "CONTROLLER_TYPE_LINEAR",
            "solver_status": message_decoding.message_to_python(
                self.policy.solver_status
            ),
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                self.assertEqual(
                    message_decoding.select_field(self.policy, path), expected
                )

    def test_bad_paths_say_what_is_wrong(self) -> None:
        cases = {
            "solver_status.no_such_field": "has no field 'no_such_field'",
            "state_trajectory.3": "out of range",
            "state_trajectory.first": "expected an index",
            "resets_served.value": "is a scalar",
            "": "not a dotted field path",
            "solver_status..healthy": "not a dotted field path",
        }
        for path, expected in cases.items():
            with self.subTest(path=path):
                with self.assertRaises(message_decoding.FieldPathError) as raised:
                    message_decoding.select_field(self.policy, path)
                self.assertIn(expected, str(raised.exception))

    def test_an_unknown_field_lists_the_fields_of_the_message(self) -> None:
        with self.assertRaises(message_decoding.FieldPathError) as raised:
            message_decoding.select_field(self.policy, "solver_status.healty")
        self.assertIn("healthy, consecutive_failures", str(raised.exception))


class TextFormatTest(unittest.TestCase):

    def test_text_is_a_textproto_of_every_field(self) -> None:
        policy = make_policy()
        text = message_decoding.format_fields(policy, [], "text")
        parsed = parse_text(text, mpc_policy_pb2.MpcPolicy)
        # Every field is printed, so unset submessages come back set to their defaults: the values are equal, the
        # presence bits are not.
        self.assertEqual(
            message_decoding.message_to_python(parsed),
            message_decoding.message_to_python(policy),
        )
        self.assertEqual(parsed.solver_status, policy.solver_status)
        self.assertEqual(parsed.state_trajectory, policy.state_trajectory)

    def test_fields_at_their_default_are_printed(self) -> None:
        lines = message_decoding.message_to_text(
            mpc_status_pb2.MpcStatus()
        ).splitlines()
        self.assertIn("observation_time: 0.0", lines)
        self.assertIn("observations_skipped: 0", lines)
        self.assertIn("solver_status {", lines)
        self.assertIn("  healthy: false", lines)
        self.assertIn('  last_error: ""', lines)
        empty_policy = message_decoding.message_to_text(mpc_policy_pb2.MpcPolicy())
        self.assertIn("time_trajectory: []", empty_policy.splitlines())
        self.assertIn("state_trajectory: []", empty_policy.splitlines())
        parse_text(empty_policy, mpc_policy_pb2.MpcPolicy)

    def test_repeated_scalars_are_one_list_and_enums_print_by_name(self) -> None:
        lines = message_decoding.message_to_text(make_policy()).splitlines()
        self.assertIn("time_trajectory: [0.0, 0.01, 0.02]", lines)
        self.assertIn("controller_type: CONTROLLER_TYPE_LINEAR", lines)
        self.assertIn("  data: [1.0, 1.5]", lines)

    def test_strings_and_maps_round_trip(self) -> None:
        status = mpc_solver_status_pb2.MpcSolverStatus(
            last_error='solver said "no"\n\tat node 3, \u00fcber',
            solve_time_ms=float("inf"),
        )
        self.assertEqual(
            parse_text(
                message_decoding.message_to_text(status),
                mpc_solver_status_pb2.MpcSolverStatus,
            ),
            status,
        )
        targets = joint_targets_pb2.JointTargets(
            positions={"right_knee": -0.5, "left_knee": 0.25}
        )
        text = message_decoding.message_to_text(targets)
        self.assertEqual(parse_text(text, joint_targets_pb2.JointTargets), targets)
        self.assertLess(text.index('"left_knee"'), text.index('"right_knee"'))
        empty = message_decoding.message_to_text(joint_targets_pb2.JointTargets())
        self.assertEqual(empty, "positions: []")
        self.assertEqual(
            parse_text(empty, joint_targets_pb2.JointTargets),
            joint_targets_pb2.JointTargets(),
        )

    def test_oneof_members_are_printed_only_when_set(self) -> None:
        choice_class = make_choice_class()
        choice = choice_class(value=0.0)
        text = message_decoding.message_to_text(choice)
        self.assertEqual(
            text.splitlines(), ["# label: not set", "value: 0.0", "# count: not set"]
        )
        parsed = parse_text(text, choice_class)
        self.assertEqual(parsed, choice)
        self.assertEqual(parsed.WhichOneof("kind"), "value")
        self.assertFalse(parsed.HasField("count"))

    def test_selected_fields_print_as_a_pruned_textproto(self) -> None:
        text = message_decoding.format_fields(
            make_policy(), ["solver_status.healthy", "resets_served"], "text"
        )
        expected = mpc_policy_pb2.MpcPolicy(resets_served=3)
        expected.solver_status.healthy = True
        self.assertEqual(parse_text(text, mpc_policy_pb2.MpcPolicy), expected)

    def test_selected_elements_carry_their_indices(self) -> None:
        text = message_decoding.format_fields(
            make_policy(),
            ["state_trajectory.-1.data.1", "time_trajectory.0", "time_trajectory.2"],
            "text",
        )
        lines = text.splitlines()
        self.assertIn("time_trajectory: [0.0, 0.02]  # [0, 2] of 3", lines)
        self.assertIn("state_trajectory {  # [2] of 3", lines)
        self.assertIn("  data: [2.5]  # [1] of 2", lines)
        parsed = parse_text(text, mpc_policy_pb2.MpcPolicy)
        self.assertEqual(list(parsed.time_trajectory), [0.0, 0.02])
        self.assertEqual(list(parsed.state_trajectory[0].data), [2.5])

    def test_a_field_selected_whole_wins_over_paths_into_it(self) -> None:
        policy = make_policy()
        whole = message_decoding.format_fields(policy, ["solver_status"], "text")
        for paths in (
            ["solver_status.healthy", "solver_status"],
            ["solver_status", "solver_status.healthy"],
        ):
            with self.subTest(paths=paths):
                self.assertEqual(
                    message_decoding.format_fields(policy, paths, "text"), whole
                )

    def test_bad_paths_raise_in_every_format(self) -> None:
        for output_format in message_decoding.OUTPUT_FORMATS:
            with self.subTest(output_format=output_format):
                with self.assertRaises(message_decoding.FieldPathError):
                    message_decoding.format_fields(
                        make_policy(), ["solver_status.healty"], output_format
                    )


class JsonFormatTest(unittest.TestCase):

    def test_json_parses_back_into_a_message_with_the_same_values(self) -> None:
        policy = make_policy()
        text = message_decoding.format_fields(policy, [], "json")
        parsed = json_format.ParseDict(json.loads(text), mpc_policy_pb2.MpcPolicy())
        # Every field is printed, so unset submessages come back set to their defaults: the values are equal, the
        # presence bits are not.
        self.assertEqual(
            message_decoding.message_to_python(parsed),
            message_decoding.message_to_python(policy),
        )
        self.assertEqual(parsed.solver_status, policy.solver_status)
        self.assertEqual(parsed.state_trajectory, policy.state_trajectory)

    def test_selected_fields_are_keyed_by_their_path(self) -> None:
        text = message_decoding.format_fields(
            make_policy(), ["solver_status.healthy", "resets_served"], "json"
        )
        self.assertEqual(
            json.loads(text),
            {"solver_status.healthy": True, "resets_served": 3},
        )


class OutputFormatsTest(unittest.TestCase):

    def test_unknown_formats_are_rejected_with_the_known_ones(self) -> None:
        with self.assertRaises(ValueError) as raised:
            message_decoding.format_fields(make_policy(), [], "xml")
        self.assertIn("text, json", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
