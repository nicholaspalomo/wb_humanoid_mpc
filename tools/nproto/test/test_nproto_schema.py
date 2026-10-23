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

"""nproto_schema: the schema fingerprint, computed as nproto::SchemaFingerprint() computes it (testSchema.cpp)."""

import importlib
import unittest

import nproto_schema

# The test protos' modules, which exist only in the Bazel runfiles (py_proto_library), so the linters cannot see them.
outer_pb2 = importlib.import_module("tools.nproto.test.outer_pb2")
scalars_pb2 = importlib.import_module("tools.nproto.test.scalars_pb2")

# The fingerprint of nproto_test.Outer, which testSchema.cpp pins alike.
# LINT.IfChange(outer_fingerprint)
OUTER_FINGERPRINT_TEXT = (
    "message nproto_test.Outer\n"
    "field 1 inner 11 singular presence nproto_test.Outer.Inner\n"
    "field 2 inners 11 repeated no_presence nproto_test.Outer.Inner\n"
    "field 3 kind 14 singular no_presence nproto_test.Outer.Kind\n"
    "field 4 leaf 11 singular presence nproto_test.Outer.Inner.Leaf\n"
    "field 5 inner_by_name 11 repeated no_presence nproto_test.Outer.InnerByNameEntry\n"
    "message nproto_test.Outer.Inner\n"
    "field 1 kind 14 singular no_presence nproto_test.Outer.Kind\n"
    "field 2 leaf 11 singular presence nproto_test.Outer.Inner.Leaf\n"
    "field 3 later 11 singular presence nproto_test.Outer.Later\n"
    "field 4 weights 1 repeated no_presence -\n"
    "message nproto_test.Outer.Inner.Leaf\n"
    "field 1 label 9 singular no_presence -\n"
    "field 2 kind 14 singular no_presence nproto_test.Outer.Kind\n"
    "message nproto_test.Outer.InnerByNameEntry\n"
    "field 1 key 9 singular no_presence -\n"
    "field 2 value 11 singular presence nproto_test.Outer.Inner\n"
    "message nproto_test.Outer.Later\n"
    "field 1 value 1 singular no_presence -\n"
    "enum nproto_test.Outer.Kind\n"
    "value 0 KIND_UNSPECIFIED\n"
    "value 1 KIND_STANCE\n"
    "value 2 KIND_SWING\n"
)
OUTER_FINGERPRINT = "f97c11d0be6d2bfc"
# LINT.ThenChange(//tools/nproto/test/testSchema.cpp:outer_fingerprint)


class TestSchemaFingerprint(unittest.TestCase):
    def test_it_is_the_hash_of_every_reachable_message_and_enum(self):
        self.assertEqual(
            nproto_schema.schema_fingerprint_text(outer_pb2.Outer.DESCRIPTOR),
            OUTER_FINGERPRINT_TEXT,
        )
        self.assertEqual(
            nproto_schema.schema_fingerprint(outer_pb2.Outer.DESCRIPTOR),
            OUTER_FINGERPRINT,
        )

    def test_it_differs_between_schemas(self):
        self.assertNotEqual(
            nproto_schema.schema_fingerprint(outer_pb2.Outer.DESCRIPTOR),
            nproto_schema.schema_fingerprint(scalars_pb2.Scalars.DESCRIPTOR),
        )


if __name__ == "__main__":
    unittest.main()
