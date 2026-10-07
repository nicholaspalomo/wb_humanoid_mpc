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

"""Tests for proto_next_id.py: the `// Next ID: N` comment and the opening `reserved N to max;` of every message."""

import unittest

from tools.hooks import check_test_support
from tools.hooks import proto_next_id


def violations(source):
    """(message name, line) of every violation in `source`."""
    return [(v.message_name, v.line) for v in proto_next_id.check_source(source)]


def texts(source):
    return [v.text for v in proto_next_id.check_source(source)]


class AcceptsCorrectMessagesTest(unittest.TestCase):
    def test_a_message_with_its_comment_and_reservation(self):
        source = """syntax = "proto3";
// A point.
// Next ID: 3
message Point {
  reserved 3 to max;

  double x = 1;
  double y = 2;
}
"""
        self.assertEqual(violations(source), [])

    def test_a_number_above_the_largest_in_use_is_fine(self):
        # A field was deleted: its number must not come back, so the comment and the reservation keep announcing the one
        # after it.
        source = """// Next ID: 7
message Point {
  reserved 7 to max;

  double x = 1;
}
"""
        self.assertEqual(violations(source), [])

    def test_an_empty_message_takes_one(self):
        self.assertEqual(
            violations("// Next ID: 1\nmessage Empty {\n  reserved 1 to max;\n}\n"),
            [],
        )


class FlagsMissingAndStaleCommentsTest(unittest.TestCase):
    def test_a_missing_comment(self):
        self.assertEqual(
            violations("message Point {\n  reserved 2 to max;\n  double x = 1;\n}\n"),
            [("Point", 1)],
        )

    def test_a_comment_that_is_not_directly_above_the_message(self):
        source = """// Next ID: 2

message Point {
  reserved 2 to max;
  double x = 1;
}
"""
        self.assertEqual(violations(source), [("Point", 3)])

    def test_a_field_added_with_the_announced_number_and_the_comment_left_behind(self):
        # The slip the comment exists to prevent (protoc would also refuse field z here, its number is reserved).
        source = """// Next ID: 3
message Point {
  reserved 3 to max;
  double x = 1;
  double y = 2;
  double z = 3;
}
"""
        found = proto_next_id.check_source(source, "point.proto")
        self.assertEqual([(v.message_name, v.line) for v in found], [("Point", 2)])
        self.assertIn("field z = 3", found[0].text)
        self.assertIn("next free number is 4", found[0].text)
        self.assertTrue(str(found[0]).startswith("point.proto:2: message Point:"))

    def test_a_malformed_comment_is_reported_as_such(self):
        found = texts(
            "// next id 2\nmessage Point {\n  reserved 2 to max;\n  double x = 1;\n}\n"
        )
        self.assertEqual(len(found), 1)
        self.assertIn("malformed", found[0])
        self.assertEqual(
            len(
                texts(
                    "// Next ID: two\nmessage Point {\n  reserved 2 to max;\n  double x = 1;\n}\n"
                )
            ),
            1,
        )


class OpeningReservationTest(unittest.TestCase):
    def test_a_missing_reservation(self):
        found = texts("// Next ID: 2\nmessage Point {\n  double x = 1;\n}\n")
        self.assertEqual(len(found), 1)
        self.assertIn("does not open with 'reserved 2 to max;'", found[0])

    def test_a_reservation_that_does_not_match_the_comment(self):
        source = """// Next ID: 4
message Point {
  reserved 3 to max;
  double x = 1;
}
"""
        found = proto_next_id.check_source(source)
        self.assertEqual([(v.message_name, v.line) for v in found], [("Point", 3)])
        self.assertIn("write 'reserved 4 to max;'", found[0].text)

    def test_the_reservation_must_be_the_first_statement_after_the_options(self):
        source = """// Next ID: 2
message Point {
  double x = 1;
  reserved 2 to max;
}
"""
        # Not first: it is an ordinary reservation then, so it also takes the number the comment announces.
        self.assertEqual(len(texts(source)), 1)

    def test_the_options_of_the_body_come_before_the_reservation(self):
        source = """// Next ID: 4
message Vector3 {
  option (nproto.generate_struct) = "ocs2::humanoid::msgs::Vector3";

  reserved 4 to max;

  double x = 1;
  double y = 2;
  double z = 3;
}
"""
        self.assertEqual(violations(source), [])
        two_options = source.replace(
            '"ocs2::humanoid::msgs::Vector3";\n',
            '"ocs2::humanoid::msgs::Vector3";\n  option deprecated = true;\n',
        )
        self.assertEqual(violations(two_options), [])

    def test_options_with_aggregate_values_are_leading_options_too(self):
        source = """// Next ID: 2
message TaskFile {
  option (nproto.generate_struct) = "ocs2::humanoid::mpc_config::TaskFile";
  option (nproto.retired_field) = { name: "use_old" replacement: "write new" };
  option (nproto.retired_field) = {
    name: "legacy"
    replacement: "drop it"
  };

  reserved 2 to max;

  double new_value = 1;
}
"""
        self.assertEqual(violations(source), [])

    def test_nested_messages_need_their_own(self):
        source = """// Next ID: 2
message Outer {
  reserved 2 to max;

  // Next ID: 2
  message Inner {
    int32 a = 1;
  }
  Inner inner = 1;
}
"""
        self.assertEqual(violations(source), [("Inner", 6)])

    def test_a_nested_message_before_the_reservation(self):
        source = """// Next ID: 2
message Outer {
  // Next ID: 1
  message Inner {
    reserved 1 to max;
  }
  reserved 2 to max;
  Inner inner = 1;
}
"""
        self.assertEqual(violations(source), [("Outer", 2)])


class CountsEveryNumberAMessageUsesTest(unittest.TestCase):
    def test_reserved_numbers_and_ranges(self):
        source = """// Next ID: 3
message Point {
  reserved 3 to max;
  double x = 1;
  reserved 2, 5 to 9;
  reserved "old_name";
}
"""
        found = texts(source)
        self.assertEqual(len(found), 1)
        self.assertIn("next free number is 10", found[0])

    def test_extensions_ranges(self):
        source = """// Next ID: 2
message Extendable {
  reserved 2 to max;
  int32 a = 1;
  extensions 100 to 199;
}
"""
        self.assertIn("next free number is 200", texts(source)[0])

    def test_oneof_map_repeated_and_optioned_fields(self):
        source = """// Next ID: 6
message Mixed {
  reserved 6 to max;

  map<string, double> positions = 1;
  repeated double values = 2 [packed = true];
  optional string name = 3;
  oneof choice {
    int32 a = 4;
    string b = 5;
  }
}
"""
        self.assertEqual(violations(source), [])
        stale = source.replace("Next ID: 6", "Next ID: 5").replace(
            "reserved 6", "reserved 5"
        )
        self.assertEqual(violations(stale), [("Mixed", 2)])

    def test_fields_with_aggregate_options_keep_their_numbers(self):
        source = """// Next ID: 4
message Weights {
  reserved 4 to max;

  double x = 1 [(tuning) = { unit: "m" slider_max: 2 }];
  double y = 2;
  oneof choice {
    double z = 3 [(tuning) = { unit: "m" }];
  }
}
"""
        self.assertEqual(violations(source), [])
        stale = source.replace("Next ID: 4", "Next ID: 3").replace(
            "reserved 4", "reserved 3"
        )
        self.assertEqual(violations(stale), [("Weights", 2)])

    def test_nested_numbers_do_not_count_for_the_outer_message(self):
        source = """// Next ID: 2
message Outer {
  reserved 2 to max;

  // Next ID: 10
  message Inner {
    reserved 10 to max;
    int32 a = 9;
  }
  int32 b = 1;
}
"""
        self.assertEqual(violations(source), [])

    def test_enum_values_and_options_are_not_fields(self):
        source = """// Next ID: 2
message WithEnum {
  reserved 2 to max;

  enum Kind {
    KIND_UNKNOWN = 0;
    KIND_BIG = 7;
  }
  option deprecated = true;
  Kind kind = 1;
}
enum TopLevel {
  TOP_LEVEL_UNKNOWN = 0;
}
"""
        self.assertEqual(violations(source), [])

    def test_comments_and_strings_cannot_fake_a_field(self):
        source = """// Next ID: 2
message Point {
  reserved 2 to max;
  // double y = 7;
  /* double z = 8; */
  double x = 1 [json_name = "q = 9;"];
}
"""
        self.assertEqual(violations(source), [])

    def test_services_are_ignored(self):
        source = """// Next ID: 2
message Request {
  reserved 2 to max;
  int32 a = 1;
}
service Solver {
  rpc Solve(Request) returns (Request) {}
}
"""
        self.assertEqual(violations(source), [])


class FixTest(unittest.TestCase):
    def test_inserts_missing_comments_and_reservations(self):
        source = """syntax = "proto3";

// A point.
message Point {
  double x = 1;
  // An offset.
  message Offset {
    double dx = 1;
    double dy = 2;
  }
}
"""
        fixed = proto_next_id.fix_source(source)
        self.assertIn(
            "// A point.\n// Next ID: 2\nmessage Point {\n  reserved 2 to max;\n\n  double x = 1;",
            fixed,
        )
        self.assertIn(
            "  // An offset.\n  // Next ID: 3\n  message Offset {\n    reserved 3 to max;\n\n    double dx = 1;",
            fixed,
        )
        self.assertEqual(proto_next_id.check_source(fixed), [])

    def test_inserts_the_reservation_after_the_nproto_option(self):
        source = """message Vector3 {
  option (nproto.generate_struct) = "ocs2::humanoid::msgs::Vector3";

  double x = 1;
  double y = 2;
  double z = 3;
}
"""
        fixed = proto_next_id.fix_source(source)
        self.assertEqual(
            fixed,
            """// Next ID: 4
message Vector3 {
  option (nproto.generate_struct) = "ocs2::humanoid::msgs::Vector3";

  reserved 4 to max;

  double x = 1;
  double y = 2;
  double z = 3;
}
""",
        )
        self.assertEqual(proto_next_id.check_source(fixed), [])
        # Without the blank line after the option, the fix adds it.
        tight = source.replace('Vector3";\n\n', 'Vector3";\n')
        self.assertEqual(proto_next_id.fix_source(tight), fixed)

    def test_opens_up_a_one_line_empty_message(self):
        fixed = proto_next_id.fix_source("message Empty {}\n")
        self.assertEqual(
            fixed, "// Next ID: 1\nmessage Empty {\n  reserved 1 to max;\n}\n"
        )
        self.assertEqual(proto_next_id.check_source(fixed), [])

    def test_raises_stale_ones_and_keeps_larger_ones(self):
        source = """// Next ID: 2
message Point {
  reserved 2 to max;
  double x = 1;
  double y = 2;
}
// Next ID: 9
message Kept {
  double x = 1;
}
"""
        fixed = proto_next_id.fix_source(source)
        self.assertIn("// Next ID: 3\nmessage Point {\n  reserved 3 to max;\n", fixed)
        self.assertIn("// Next ID: 9\nmessage Kept {\n  reserved 9 to max;\n", fixed)
        self.assertEqual(proto_next_id.check_source(fixed), [])
        self.assertEqual(proto_next_id.fix_source(fixed), fixed)

    def test_a_reservation_that_disagrees_with_the_comment_follows_the_comment(self):
        source = """// Next ID: 5
message Point {
  reserved 2 to max;
  double x = 1;
}
"""
        fixed = proto_next_id.fix_source(source)
        self.assertIn("  reserved 5 to max;\n", fixed)
        self.assertEqual(proto_next_id.check_source(fixed), [])


class WiringTest(unittest.TestCase):
    def test_the_pre_commit_hook_and_the_linter_run_the_check(self):
        check_test_support.assert_hook_runs_the_linter(self)
        check_test_support.assert_check_behaves(
            self,
            "proto-next-id",
            'syntax = "proto3";\nmessage Point {\n  int32 x = 1;\n}\n',
            "msgs/point.proto",
        )


if __name__ == "__main__":
    unittest.main()
