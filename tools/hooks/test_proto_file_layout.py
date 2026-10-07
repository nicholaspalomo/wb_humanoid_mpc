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

"""Tests for proto_file_layout.py: one top-level proto definition per file, named after it, with its nproto option."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import proto_file_layout

IMPORT = 'import "nproto/options.proto";\n'


def struct(name):
    """The first statement of a message's body: its nproto option."""
    return f'  option (nproto.generate_struct) = "ocs2::humanoid::msgs::{name}";\n'


def enum(name):
    return f'  option (nproto.generate_enum) = "ocs2::humanoid::msgs::{name}";\n'


def message(name, body=""):
    return f"message {name} {{\n{struct(name)}{body}}}\n"


def lines(source, path):
    return [v.line for v in proto_file_layout.check_source(source, path)]


def texts(source, path):
    return [v.text for v in proto_file_layout.check_source(source, path)]


class SnakeCaseTest(unittest.TestCase):
    def test_names(self):
        cases = {
            "MpcPolicy": "mpc_policy",
            "Vector3": "vector3",
            "Vector": "vector",
            "JointPdGainsFile": "joint_pd_gains_file",
            "HTTPRequest": "http_request",
            "RobotStateSample": "robot_state_sample",
        }
        for name, expected in cases.items():
            self.assertEqual(proto_file_layout.snake_case(name), expected, name)


class OneDefinitionPerFileTest(unittest.TestCase):
    def test_one_message_named_after_its_file(self):
        source = f"""syntax = "proto3";
package humanoid_mpc_msgs;
import "humanoid_mpc_msgs/vector.proto";
{IMPORT}
// Next ID: 2
message MpcPolicy {{
{struct("MpcPolicy")}
  reserved 2 to max;

  // Next ID: 2
  message Nested {{
    reserved 2 to max;
    int32 a = 1;
  }}
  enum Kind {{
    KIND_UNKNOWN = 0;
  }}
  Nested nested = 1;
}}
"""
        self.assertEqual(lines(source, "humanoid_mpc_msgs/mpc_policy.proto"), [])

    def test_an_enum_or_a_service_counts_as_the_one_definition(self):
        source = (
            IMPORT
            + "enum ControllerType {\n"
            + enum("ControllerType")
            + "  A = 0;\n}\n"
        )
        self.assertEqual(lines(source, "controller_type.proto"), [])
        # A service generates no struct, so it carries no nproto option.
        self.assertEqual(
            lines(
                "service Solver {\n  rpc Solve(A) returns (A) {}\n}\n", "solver.proto"
            ),
            [],
        )

    def test_a_second_message_is_flagged_where_it_is(self):
        source = IMPORT + message("Vector") + message("Vector3")
        found = proto_file_layout.check_source(source, "vector.proto")
        self.assertEqual([v.line for v in found if "further top-level" in v.text], [5])
        self.assertIn("vector3.proto", found[0].text)

    def test_a_top_level_enum_next_to_a_message_is_flagged(self):
        source = (
            IMPORT
            + "enum ControllerType {\n"
            + enum("ControllerType")
            + "  A = 0;\n}\n"
            + message("MpcPolicy")
        )
        found = proto_file_layout.check_source(source, "controller_type.proto")
        self.assertIn(6, [v.line for v in found if "further top-level" in v.text])

    def test_a_file_not_named_after_its_definition(self):
        found = texts(IMPORT + message("MpcPolicy"), "mpc.proto")
        self.assertEqual(len(found), 1)
        self.assertIn("belongs in mpc_policy.proto", found[0])

    def test_a_file_without_a_definition(self):
        self.assertEqual(lines('syntax = "proto3";\n', "empty.proto"), [1])

    def test_comments_and_strings_do_not_count(self):
        source = f"""// message Old {{}}
/* enum Older {{}} */
{IMPORT}message Point {{
{struct("Point")}  string doc = 1 [json_name = "message Fake {{"];
}}
"""
        self.assertEqual(lines(source, "point.proto"), [])


class OptionsFileTest(unittest.TestCase):
    def test_a_file_that_only_extends_options_is_accepted(self):
        source = """syntax = "proto3";
package nproto;
import "google/protobuf/descriptor.proto";
extend google.protobuf.MessageOptions {
  string generate_struct = 52001;
}
extend google.protobuf.EnumOptions {
  string generate_enum = 52002;
}
"""
        self.assertEqual(lines(source, "options.proto"), [])

    def test_an_extend_block_next_to_a_message_is_a_further_definition(self):
        source = (
            IMPORT
            + message("Point")
            + "extend google.protobuf.MessageOptions {\n  string x = 50000;\n}\n"
        )
        found = texts(source, "point.proto")
        self.assertEqual(len(found), 1)
        self.assertIn("further top-level definition", found[0])


class NprotoOptionTest(unittest.TestCase):
    def test_a_message_without_the_option(self):
        found = texts(IMPORT + "message Point {\n  double x = 1;\n}\n", "point.proto")
        self.assertEqual(len(found), 1)
        self.assertIn("option (nproto.generate_struct)", found[0])
        self.assertIn("first statement of its body", found[0])

    def test_an_enum_without_the_option(self):
        found = texts("enum Kind {\n  KIND_A = 0;\n}\n", "kind.proto")
        self.assertEqual(len(found), 1)
        self.assertIn("nproto.generate_enum", found[0])

    def test_the_option_at_file_scope_is_moved_into_the_message(self):
        source = (
            IMPORT
            + 'option (nproto.generate_struct) = "ocs2::humanoid::msgs::Point";\n'
            + "message Point {}\n"
        )
        found = texts(source, "point.proto")
        self.assertEqual(len(found), 1)
        self.assertIn("declared at file scope; move it into message Point", found[0])

    def test_the_option_must_be_the_first_statement(self):
        source = IMPORT + "message Point {\n  double x = 1;\n" + struct("Point") + "}\n"
        found = texts(source, "point.proto")
        self.assertEqual(len(found), 1)
        self.assertIn("must be the first statement", found[0])
        source = (
            IMPORT
            + "enum Kind {\n  option allow_alias = true;\n"
            + enum("Kind")
            + "  KIND_A = 0;\n}\n"
        )
        self.assertEqual(len(texts(source, "kind.proto")), 1)

    def test_a_message_takes_generate_struct_and_an_enum_generate_enum(self):
        found = texts(
            IMPORT + "message Point {\n" + enum("Point") + "}\n", "point.proto"
        )
        self.assertEqual(len(found), 1)
        self.assertIn("takes (nproto.generate_struct)", found[0])
        found = texts(
            IMPORT + "enum Kind {\n" + struct("Kind") + "  KIND_A = 0;\n}\n",
            "kind.proto",
        )
        self.assertEqual(len(found), 1)
        self.assertIn("takes (nproto.generate_enum)", found[0])

    def test_nested_types_carry_no_option(self):
        source = (
            IMPORT
            + "message Outer {\n"
            + struct("Outer")
            + "  message Inner {\n  "
            + struct("Inner")
            + "  }\n}\n"
        )
        found = texts(source, "outer.proto")
        self.assertEqual(len(found), 1)
        self.assertIn("nested type Outer.Inner", found[0])

    def test_the_struct_must_be_named_after_the_definition(self):
        found = texts(
            IMPORT + "message Point {\n" + struct("Other") + "}\n", "point.proto"
        )
        self.assertEqual(len(found), 1)
        self.assertIn("ending in ::Point", found[0])

    def test_the_value_must_be_a_cpp_qualified_name(self):
        for value in ("ocs2.humanoid.Point", "ocs2::::Point", "1ocs2::Point", ""):
            source = (
                IMPORT
                + f'message Point {{\n  option (nproto.generate_struct) = "{value}";\n}}\n'
            )
            self.assertEqual(len(texts(source, "point.proto")), 1, value)

    def test_unqualified_and_globally_qualified_names_are_accepted(self):
        for value in ("Point", "::ocs2::Point"):
            source = (
                IMPORT
                + f'message Point {{\n  option (nproto.generate_struct) = "{value}";\n}}\n'
            )
            self.assertEqual(texts(source, "point.proto"), [], value)

    def test_the_options_file_must_be_imported(self):
        found = texts(message("Point"), "point.proto")
        self.assertEqual(len(found), 1)
        self.assertIn("nproto/options.proto", found[0])

    def test_a_second_option_in_the_message(self):
        source = (
            IMPORT + "message Point {\n" + struct("Point") + struct("Point") + "}\n"
        )
        found = texts(source, "point.proto")
        self.assertEqual(len(found), 1)
        self.assertIn("a second nproto option", found[0])


class ReservedNamesTest(unittest.TestCase):
    def test_reserved_numbers_are_fine(self):
        source = IMPORT + message(
            "Point", "  reserved 2, 4 to 6;\n  reserved 9 to max;\n  double x = 1;\n"
        )
        self.assertEqual(texts(source, "point.proto"), [])

    def test_a_quoted_reserved_name_is_refused(self):
        source = IMPORT + message(
            "Point", '  reserved 2;\n  reserved "old_x", "old_y";\n  double x = 1;\n'
        )
        found = proto_file_layout.check_source(source, "point.proto")
        self.assertEqual([v.line for v in found], [5, 5])
        self.assertIn("reserves the field name 'old_x'", found[0].text)
        self.assertIn("(nproto.retired_field)", found[0].text)
        self.assertIn("silently skips", found[0].text)

    def test_an_edition_2023_reserved_name_is_refused(self):
        source = (
            'edition = "2023";\n'
            + IMPORT
            + message("Point", "  reserved old_x;\n  double x = 1;\n")
        )
        self.assertEqual(
            [name for _, name in proto_file_layout.reserved_names(source)], ["old_x"]
        )
        self.assertEqual(len(texts(source, "point.proto")), 1)

    def test_nested_messages_are_checked_and_enums_are_not(self):
        source = (
            IMPORT
            + "message Outer {\n"
            + struct("Outer")
            + '  message Inner {\n    reserved "gone";\n  }\n'
            + '  enum Kind {\n    reserved "KIND_OLD";\n    KIND_A = 0;\n  }\n}\n'
        )
        self.assertEqual(proto_file_layout.reserved_names(source), [(5, "gone")])

    def test_strings_elsewhere_are_not_reserved_names(self):
        source = IMPORT + message(
            "Point",
            '  option (x) = "reserved y";\n  string doc = 1 [json_name = "reserved"];\n',
        )
        self.assertEqual(proto_file_layout.reserved_names(source), [])


class WiringTest(unittest.TestCase):
    def test_the_pre_commit_hook_and_the_linter_run_the_check(self):
        check_test_support.assert_hook_runs_the_linter(self)
        check_test_support.assert_check_behaves(
            self,
            "proto-file-layout",
            'syntax = "proto3";\nmessage Point {\n}\n',
            "msgs/point.proto",
        )


if __name__ == "__main__":
    unittest.main()
