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

"""The nproto generator on requests built here: what it generates, and the files it refuses with a clear error.

The requests hold FileDescriptorProtos written as text, as protoc would send them, so that the cases nproto must
reject - which the repository's lint keeps out of real .proto files - need no .proto file of their own.
"""

from collections.abc import Sequence
import os
import subprocess
import sys
import unittest

from google.protobuf import descriptor_pb2
from google.protobuf import text_format
from google.protobuf.compiler import plugin_pb2
from nproto import options_pb2
from nproto import retired_field_options_pb2
from nproto import retired_field_pb2

import nproto_generator

PLUGIN = os.path.join(
    os.environ.get("TEST_SRCDIR", ""),
    "_main",
    "tools",
    "nproto",
    "protoc-gen-nproto",
)


def file_proto(text: str) -> descriptor_pb2.FileDescriptorProto:
    return text_format.Parse(text, descriptor_pb2.FileDescriptorProto())


def library_files() -> list[descriptor_pb2.FileDescriptorProto]:
    """descriptor.proto and nproto's option files, which every request carries."""
    files = []
    for module in (
        descriptor_pb2,
        options_pb2,
        retired_field_pb2,
        retired_field_options_pb2,
    ):
        proto = descriptor_pb2.FileDescriptorProto()
        module.DESCRIPTOR.CopyToProto(proto)
        files.append(proto)
    return files


def request(
    *files: str, generate: Sequence[str] = ()
) -> plugin_pb2.CodeGeneratorRequest:
    """A request for `files` (text FileDescriptorProtos, in import order), generating `generate` or the last one."""
    protos = [file_proto(text) for text in files]
    result = plugin_pb2.CodeGeneratorRequest()
    result.proto_file.extend(library_files() + protos)
    result.file_to_generate.extend(generate or [protos[-1].name])
    return result


FIELD_X = (
    'field { name: "x" number: 1 label: LABEL_OPTIONAL type: TYPE_DOUBLE }'
)


def message_file(
    name: str = "pkg/node.proto",
    package: str = "pkg",
    option: str = "pkg::nproto::Node",
    message: str = "Node",
    members: str = FIELD_X,
    dependencies: Sequence[str] = ("nproto/options.proto",),
    message_options: str = "",
) -> str:
    """A file of one message, `members` its fields and nested types, its option `option` (none when empty).

    Args:
      name: The file name.
      package: The proto package.
      option: The value of (nproto.generate_struct); empty for none.
      message: The message name.
      members: The fields and nested types of the message, as text.
      dependencies: The files the file imports.
      message_options: Further message options inside the options block, as text (with `option` only).

    Returns:
      The FileDescriptorProto as text.
    """
    options = (
        f'options {{ [nproto.generate_struct]: "{option}" {message_options}}} '
        if option
        else ""
    )
    return raw_file(
        name,
        package,
        f'message_type {{ name: "{message}" {options}{members} }}',
        dependencies,
    )


def enum_file(
    name: str = "pkg/color.proto",
    package: str = "pkg",
    option: str = "pkg::nproto::Color",
    enum: str = "Color",
    values: str = 'value { name: "COLOR_UNSPECIFIED" number: 0 } value { name: "COLOR_DARK_RED" number: 3 }',
) -> str:
    """A file of one enum, with the option `option` (none when empty)."""
    options = (
        f'options {{ [nproto.generate_enum]: "{option}" }} ' if option else ""
    )
    return raw_file(
        name, package, f'enum_type {{ name: "{enum}" {options}{values} }}'
    )


def raw_file(
    name: str,
    package: str,
    body: str,
    dependencies: Sequence[str] = ("nproto/options.proto",),
) -> str:
    imports = "".join(
        f'dependency: "{dependency}" ' for dependency in dependencies
    )
    return (
        f'name: "{name}" package: "{package}" syntax: "proto3" {imports} {body}'
    )


def generated(response: plugin_pb2.CodeGeneratorResponse) -> dict[str, str]:
    return {file.name: file.content for file in response.file}


class GeneratorTestCase(unittest.TestCase):
    """The assertions on a CodeGeneratorResponse that the generator tests share."""

    def assertError(
        self, response: plugin_pb2.CodeGeneratorResponse, *fragments: str
    ) -> None:
        self.assertTrue(response.error, "expected an error, got files")
        self.assertFalse(response.file, "an error generates no files")
        for fragment in fragments:
            self.assertIn(fragment, response.error)

    def assertGenerates(
        self, response: plugin_pb2.CodeGeneratorResponse
    ) -> dict[str, str]:
        self.assertEqual(response.error, "")
        return generated(response)


class GeneratedFilesTest(GeneratorTestCase):
    def test_a_message_gets_a_struct_header_and_conversions(self):
        files = self.assertGenerates(
            nproto_generator.generate(request(message_file()))
        )
        self.assertEqual(
            sorted(files),
            [
                "pkg/node.nproto.h",
                "pkg/node.nproto.pb.cc",
                "pkg/node.nproto.pb.h",
            ],
        )
        header = files["pkg/node.nproto.h"]
        self.assertIn("namespace pkg::nproto {", header)
        self.assertIn("struct Node {", header)
        self.assertIn("double x = 0.0;", header)
        self.assertIn(
            "inline bool operator==(const Node& lhs, const Node& rhs)", header
        )
        # The struct header knows nothing of protobuf.
        includes = [
            line for line in header.splitlines() if line.startswith("#include")
        ]
        self.assertFalse(
            [
                line
                for line in includes
                if "google/protobuf" in line or ".pb.h" in line
            ],
            includes,
        )
        conversions = files["pkg/node.nproto.pb.h"]
        self.assertIn('#include "pkg/node.pb.h"', conversions)
        self.assertIn(
            "void ToProto(const Node& value, ::pkg::Node* absl_nonnull proto);",
            conversions,
        )
        self.assertIn(
            "absl::Status FromProto(const ::pkg::Node& proto, Node*"
            " absl_nonnull value);",
            conversions,
        )

    def test_the_namespace_is_the_one_the_option_names(self):
        files = self.assertGenerates(
            nproto_generator.generate(
                request(message_file(option="::robot::deep::name_space::Node"))
            )
        )
        self.assertIn(
            "namespace robot::deep::name_space {", files["pkg/node.nproto.h"]
        )
        self.assertIn(
            "}  // namespace robot::deep::name_space",
            files["pkg/node.nproto.pb.cc"],
        )

    def test_nested_types_are_nested_in_the_struct_and_need_no_option(self):
        members = (
            'field { name: "kind" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_ENUM type_name: ".pkg.Node.Kind" } field { name: "child"'
            " number: 2 label: LABEL_OPTIONAL type: TYPE_MESSAGE type_name:"
            ' ".pkg.Node.Child" } nested_type { name: "Child" '
            + FIELD_X
            + ' } enum_type { name: "Kind" value { name: "KIND_STANCE" number:'
            ' 0 } value { name: "KIND_SWING" number: 1 } }'
        )
        files = self.assertGenerates(
            nproto_generator.generate(request(message_file(members=members)))
        )
        header = files["pkg/node.nproto.h"]
        self.assertIn(
            "struct Node {\n  // pkg.Node.Kind\n  enum class Kind : int32_t {",
            header,
        )
        self.assertIn("  struct Child {\n    double x = 0.0;\n  };", header)
        self.assertIn("  Kind kind = Kind::kStance;\n  Child child;\n", header)
        conversions = files["pkg/node.nproto.pb.h"]
        self.assertIn(
            "void ToProto(const Node::Child& value, ::pkg::Node_Child*"
            " absl_nonnull proto);",
            conversions,
        )
        self.assertIn(
            "void ToProto(Node::Kind value, ::pkg::Node_Kind* absl_nonnull"
            " proto);",
            conversions,
        )

    def test_the_response_declares_proto3_optional_and_editions(self):
        response = nproto_generator.generate(request(message_file()))
        features = plugin_pb2.CodeGeneratorResponse
        self.assertTrue(
            response.supported_features & features.FEATURE_PROTO3_OPTIONAL
        )
        self.assertTrue(
            response.supported_features & features.FEATURE_SUPPORTS_EDITIONS
        )
        self.assertEqual(
            response.minimum_edition, descriptor_pb2.EDITION_PROTO2
        )
        self.assertEqual(response.maximum_edition, descriptor_pb2.EDITION_2023)

    def test_an_enum_file_gets_an_enum_class(self):
        files = self.assertGenerates(
            nproto_generator.generate(request(enum_file()))
        )
        header = files["pkg/color.nproto.h"]
        self.assertIn("namespace pkg::nproto {", header)
        self.assertIn("enum class Color : int32_t {", header)
        self.assertIn("kUnspecified = 0,", header)
        self.assertIn("kDarkRed = 3,", header)
        self.assertIn(
            "absl::Status FromProto(::pkg::Color proto, Color* absl_nonnull"
            " value);",
            files["pkg/color.nproto.pb.h"],
        )

    def test_from_proto_names_each_enum_number_by_its_enumerator(self):
        # The magic-number rule (tools/hooks/magic_number.py): `case static_cast<int32_t>(Color::kDarkRed):`, never
        # `case 3:`. An alias (allow_alias) has no label of its own: the first name of its number wins.
        body = (
            'enum_type { name: "Color" options { [nproto.generate_enum]:'
            ' "pkg::nproto::Color" allow_alias: true } value { name:'
            ' "COLOR_UNSPECIFIED" number: 0 } value { name: "COLOR_DARK_RED"'
            ' number: 3 } value { name: "COLOR_CRIMSON" number: 3 } }'
        )
        source = self.assertGenerates(
            nproto_generator.generate(
                request(raw_file("pkg/color.proto", "pkg", body))
            )
        )["pkg/color.nproto.pb.cc"]
        self.assertIn(
            "  switch (static_cast<int32_t>(proto)) {\n"
            "    case static_cast<int32_t>(Color::kUnspecified):\n"
            "      *value = Color::kUnspecified;\n"
            "      return absl::OkStatus();\n"
            "    case static_cast<int32_t>(Color::kDarkRed):\n"
            "      *value = Color::kDarkRed;\n"
            "      return absl::OkStatus();\n"
            "    default:\n",
            source,
        )
        self.assertNotIn("kCrimson", source)
        self.assertNotRegex(source, r"case -?[0-9]")

    def test_the_output_parameters_are_nonnull_in_declarations_and_definitions(
        self,
    ):
        # A message, a nested enum and a nested message without fields, whose parameters have no names.
        members = (
            'field { name: "kind" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_ENUM type_name: ".pkg.Node.Kind" } nested_type { name:'
            ' "Empty" } enum_type { name: "Kind" value { name: "KIND_STANCE"'
            " number: 0 } }"
        )
        files = self.assertGenerates(
            nproto_generator.generate(request(message_file(members=members)))
        )
        for name in ("pkg/node.nproto.pb.h", "pkg/node.nproto.pb.cc"):
            self.assertIn(
                '#include "absl/base/nullability.h"', files[name], name
            )
        source = files["pkg/node.nproto.pb.cc"]
        for definition in (
            (
                "void ToProto(const Node& value, ::pkg::Node* absl_nonnull"
                " proto) {"
            ),
            (
                "absl::Status FromProto(const ::pkg::Node& proto, Node*"
                " absl_nonnull value) {"
            ),
            (
                "void ToProto(Node::Kind value, ::pkg::Node_Kind* absl_nonnull"
                " proto) {"
            ),
            (
                "absl::Status FromProto(::pkg::Node_Kind proto, Node::Kind*"
                " absl_nonnull value) {"
            ),
            (
                "void ToProto(const Node::Empty& /*value*/,\n"
                "             ::pkg::Node_Empty* absl_nonnull /*proto*/) {}"
            ),
            (
                "absl::Status FromProto(const ::pkg::Node_Empty& /*proto*/,\n"
                "                       Node::Empty* absl_nonnull /*value*/) {"
            ),
        ):
            self.assertIn(definition, source)
        # The struct header has no pointers, and stays free of Abseil.
        self.assertNotIn("absl", files["pkg/node.nproto.h"])

    def test_integer_types_are_spelled_without_std(self):
        # Every integer type in every place the generated code writes one: members, a oneof's index constants and
        # alternative, an enum's underlying type and its FromProto() switch, and the minimum as a default.
        members = (
            'field { name: "a" number: 1 label: LABEL_OPTIONAL type: TYPE_INT32'
            ' } field { name: "b" number: 2 label: LABEL_OPTIONAL type:'
            ' TYPE_SINT64 } field { name: "c" number: 3 label: LABEL_OPTIONAL'
            ' type: TYPE_UINT32 } field { name: "d" number: 4 label:'
            ' LABEL_OPTIONAL type: TYPE_FIXED64 } field { name: "e" number: 5'
            ' label: LABEL_REPEATED type: TYPE_SFIXED32 } field { name: "f"'
            " number: 6 label: LABEL_OPTIONAL type: TYPE_INT64 oneof_index: 0 }"
            ' field { name: "g" number: 7 label: LABEL_OPTIONAL type: TYPE_ENUM'
            ' type_name: ".pkg.Node.Kind" } oneof_decl { name: "choice" }'
            ' enum_type { name: "Kind" value { name: "KIND_STANCE" number:'
            " 0 } }"
        )
        proto2 = raw_file(
            "pkg/limits.proto",
            "pkg",
            'message_type { name: "Limits" options { [nproto.generate_struct]:'
            ' "pkg::nproto::Limits" } field { name: "lowest" number: 1 label:'
            " LABEL_OPTIONAL type: TYPE_INT64 default_value:"
            ' "-9223372036854775808" } field { name: "low" number: 2 label:'
            ' LABEL_OPTIONAL type: TYPE_INT32 default_value: "-2147483648" } }',
        ).replace('syntax: "proto3"', 'syntax: "proto2"')
        files = self.assertGenerates(
            nproto_generator.generate(request(message_file(members=members)))
        )
        files.update(
            self.assertGenerates(nproto_generator.generate(request(proto2)))
        )
        header = files["pkg/node.nproto.h"]
        self.assertIn("  int32_t a = 0;\n", header)
        self.assertIn("  static constexpr size_t kFIndex = 1;\n", header)
        self.assertIn(
            "std::numeric_limits<int64_t>::min()", files["pkg/limits.nproto.h"]
        )
        self.assertIn(
            "  static constexpr int32_t kDefaultLow ="
            " std::numeric_limits<int32_t>::min();\n",
            files["pkg/limits.nproto.h"],
        )
        for name, content in files.items():
            for qualified in ("std::int", "std::uint", "std::size_t"):
                self.assertNotIn(qualified, content, name)

    def test_a_signature_longer_than_a_line_keeps_its_annotation(self):
        # Too long to align the parameters after the open parenthesis, so clang-format breaks after it and puts them
        # one per line (BinPackParameters: false).
        name = "AMessageNamedLongEnoughToWrapItsSignature"
        option = f"pkg::nproto::{name}"
        files = self.assertGenerates(
            nproto_generator.generate(
                request(message_file(option=option, message=name))
            )
        )
        conversions = files["pkg/node.nproto.pb.h"]
        self.assertIn(
            "void ToProto(\n"
            f"    const {name}& value,\n"
            f"    ::pkg::{name}* absl_nonnull proto);",
            conversions,
        )
        self.assertIn(
            "absl::Status FromProto(\n"
            f"    const ::pkg::{name}& proto,\n"
            f"    {name}* absl_nonnull value);",
            conversions,
        )
        for line in conversions.splitlines():
            self.assertLessEqual(len(line), nproto_generator.COLUMN_LIMIT, line)

    def test_a_file_without_definitions_gets_empty_files(self):
        options_only = (
            'name: "pkg/extensions.proto" package: "pkg" syntax: "proto3"'
        )
        files = self.assertGenerates(
            nproto_generator.generate(request(options_only))
        )
        self.assertEqual(len(files), 3)
        for content in files.values():
            self.assertNotIn("struct", content)

    def test_an_import_with_a_struct_is_included(self):
        point = message_file(
            name="pkg/point.proto", option="pkg::nproto::Point", message="Point"
        )
        members = (
            'field { name: "point" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_MESSAGE type_name: ".pkg.Point" }'
        )
        node = message_file(
            members=members,
            dependencies=("nproto/options.proto", "pkg/point.proto"),
        )
        files = self.assertGenerates(
            nproto_generator.generate(request(point, node))
        )
        self.assertIn(
            '#include "pkg/point.nproto.h"', files["pkg/node.nproto.h"]
        )
        self.assertNotIn("nproto/options.nproto.h", files["pkg/node.nproto.h"])
        self.assertIn(
            '#include "pkg/point.nproto.pb.h"', files["pkg/node.nproto.pb.cc"]
        )
        self.assertIn("  Point point;", files["pkg/node.nproto.h"])

    def test_the_plugin_binary_speaks_protoc_protocol(self):
        # The binary parses the request itself, as protoc sends it: the options arrive as bytes.
        completed = subprocess.run(
            [PLUGIN],
            input=request(
                message_file(),
                enum_file(),
                generate=["pkg/node.proto", "pkg/color.proto"],
            ).SerializeToString(),
            capture_output=True,
            check=True,
        )
        response = plugin_pb2.CodeGeneratorResponse.FromString(completed.stdout)
        self.assertEqual(response.error, "")
        self.assertEqual(len(response.file), 6)
        self.assertIn(
            "namespace pkg::nproto {", generated(response)["pkg/color.nproto.h"]
        )

    def test_options_parsed_before_the_extensions_were_registered_are_read(
        self,
    ):
        # A process that parses the request before it imports nproto's options_pb2 holds the options as unknown fields.
        script = (
            "import sys\n"
            "from google.protobuf.compiler import plugin_pb2\n"
            "request ="
            " plugin_pb2.CodeGeneratorRequest.FromString(sys.stdin.buffer.read())\n"
            "assert 'nproto.options_pb2' not in sys.modules\n"
            "import nproto_generator\n"
            "sys.stdout.buffer.write(nproto_generator.generate(request).SerializeToString())\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            input=request(
                message_file(),
                enum_file(),
                generate=["pkg/node.proto", "pkg/color.proto"],
            ).SerializeToString(),
            capture_output=True,
            check=True,
            env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        )
        response = plugin_pb2.CodeGeneratorResponse.FromString(completed.stdout)
        self.assertEqual(response.error, "")
        files = generated(response)
        self.assertIn("namespace pkg::nproto {", files["pkg/node.nproto.h"])
        self.assertIn(
            "enum class Color : int32_t {", files["pkg/color.nproto.h"]
        )

    def test_the_plugin_binary_reports_errors_in_the_response(self):
        completed = subprocess.run(
            [PLUGIN],
            input=request(message_file(option="")).SerializeToString(),
            capture_output=True,
            check=True,
        )
        response = plugin_pb2.CodeGeneratorResponse.FromString(completed.stdout)
        self.assertIn("message Node does not name the C++ type", response.error)


class RejectedFilesTest(GeneratorTestCase):
    def test_a_message_without_the_option(self):
        self.assertError(
            nproto_generator.generate(request(message_file(option=""))),
            "pkg/node.proto: message Node does not name the C++ type nproto"
            " generates for it",
            'option (nproto.generate_struct) = "<namespace>::Node";',
            "as the first statement of its body",
            'import "nproto/options.proto"',
        )

    def test_an_enum_without_the_option(self):
        self.assertError(
            nproto_generator.generate(request(enum_file(option=""))),
            "pkg/color.proto: enum Color does not name the C++ type nproto"
            " generates for it",
            'option (nproto.generate_enum) = "<namespace>::Color";',
        )

    def test_a_nested_message_with_an_option_of_its_own(self):
        members = (
            'nested_type { name: "Child" options { [nproto.generate_struct]:'
            ' "pkg::nproto::Child" } ' + FIELD_X + " }"
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "pkg/node.proto: nested message pkg.Node.Child sets option"
            " (nproto.generate_struct)",
            "nested C++ types of their parent's struct"
            " (pkg::nproto::Node::Child)",
            "carry no option of their own",
        )

    def test_a_nested_enum_with_an_option_of_its_own(self):
        members = (
            'nested_type { name: "Child" enum_type { name: "Kind" options {'
            ' [nproto.generate_enum]: "pkg::nproto::Kind" } value { name:'
            ' "KIND_A" number: 0 } } }'
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "nested enum pkg.Node.Child.Kind sets option"
            " (nproto.generate_enum)",
            "(pkg::nproto::Node::Child::Kind)",
        )

    def test_a_recursive_message(self):
        members = (
            'field { name: "child" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_MESSAGE type_name: ".pkg.Node" }'
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "pkg/node.proto",
            "message pkg.Node contains itself (pkg.Node -> pkg.Node",
            "by value",
        )

    def test_a_message_recursive_through_a_repeated_nested_message(self):
        members = (
            'field { name: "branch" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_MESSAGE type_name: ".pkg.Node.Branch" } nested_type { name:'
            ' "Branch" field { name: "nodes" number: 1 label: LABEL_REPEATED'
            ' type: TYPE_MESSAGE type_name: ".pkg.Node" } }'
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "pkg.Node -> pkg.Node.Branch -> pkg.Node",
            "field pkg.Node.Branch.nodes",
        )

    def test_an_option_naming_another_message(self):
        self.assertError(
            nproto_generator.generate(
                request(message_file(option="pkg::nproto::Other"))
            ),
            "message Node: option (nproto.generate_struct) ="
            ' "pkg::nproto::Other" names a type Other',
            'named after the message: "pkg::nproto::Node"',
        )

    def test_an_option_naming_another_enum(self):
        self.assertError(
            nproto_generator.generate(
                request(enum_file(option="pkg::nproto::Hue"))
            ),
            'enum Color: option (nproto.generate_enum) = "pkg::nproto::Hue"'
            " names a type Hue",
            '"pkg::nproto::Color"',
        )

    def test_an_option_that_is_not_a_cpp_name(self):
        self.assertError(
            nproto_generator.generate(
                request(message_file(option="pkg.nproto.Node"))
            ),
            "is not a fully qualified C++ name",
        )

    def test_an_option_with_a_cpp_keyword(self):
        self.assertError(
            nproto_generator.generate(
                request(message_file(option="pkg::namespace::Node"))
            ),
            '"namespace" is a C++ keyword',
        )

    def test_an_option_with_a_namespace_named_like_an_integer_type(self):
        self.assertError(
            nproto_generator.generate(
                request(message_file(option="pkg::int32_t::Node"))
            ),
            '"int32_t" is a type the generated code names without std::',
        )

    def test_an_option_in_the_global_namespace(self):
        self.assertError(
            nproto_generator.generate(request(message_file(option="Node"))),
            "global namespace",
        )

    def test_an_option_naming_the_protobuf_class(self):
        self.assertError(
            nproto_generator.generate(
                request(message_file(option="pkg::Node"))
            ),
            "is the C++ name protobuf already gives pkg.Node",
        )

    def test_two_top_level_definitions(self):
        body = (
            'message_type { name: "Node" options { [nproto.generate_struct]:'
            ' "pkg::nproto::Node" } } enum_type { name: "Kind" options {'
            ' [nproto.generate_enum]: "pkg::nproto::Kind" } value { name:'
            ' "KIND_A" number: 0 } }'
        )
        self.assertError(
            nproto_generator.generate(
                request(raw_file("pkg/node.proto", "pkg", body))
            ),
            "defines 2 top-level types (message Node, enum Kind)",
        )

    def test_a_field_of_a_type_without_a_struct(self):
        stamp = raw_file(
            "pkg/stamp.proto",
            "pkg",
            'message_type { name: "Stamp" field { name: "seconds" number: 1'
            " label: LABEL_OPTIONAL type: TYPE_INT64 } }",
            dependencies=(),
        )
        members = (
            'field { name: "stamp" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_MESSAGE type_name: ".pkg.Stamp" }'
        )
        node = message_file(
            members=members,
            dependencies=("nproto/options.proto", "pkg/stamp.proto"),
        )
        self.assertError(
            nproto_generator.generate(request(stamp, node)),
            "field pkg.Node.stamp has type pkg.Stamp, which has no nproto"
            " struct",
            "message pkg.Stamp (pkg/stamp.proto) sets no option"
            " (nproto.generate_struct)",
        )

    def test_a_field_of_a_nested_type_of_a_message_without_a_struct(self):
        stamp = raw_file(
            "pkg/stamp.proto",
            "pkg",
            'message_type { name: "Stamp" enum_type { name: "Unit" value {'
            ' name: "UNIT_S" number: 0 } } }',
            dependencies=(),
        )
        members = (
            'field { name: "unit" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_ENUM type_name: ".pkg.Stamp.Unit" }'
        )
        node = message_file(
            members=members,
            dependencies=("nproto/options.proto", "pkg/stamp.proto"),
        )
        self.assertError(
            nproto_generator.generate(request(stamp, node)),
            "field pkg.Node.unit has type pkg.Stamp.Unit, which has no nproto"
            " struct: message pkg.Stamp",
        )

    def test_a_field_named_like_a_cpp_keyword(self):
        members = (
            'field { name: "class" number: 1 label: LABEL_OPTIONAL type:'
            " TYPE_INT32 }"
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "field pkg.Node.class is named 'class', a C++ keyword",
        )

    def test_a_field_or_oneof_named_like_an_integer_type(self):
        # The generated code writes int32_t and size_t without std::, which a member of that name would hide.
        field = (
            'field { name: "uint64_t" number: 1 label: LABEL_OPTIONAL type:'
            " TYPE_INT32 }"
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=field))),
            "field pkg.Node.uint64_t is named 'uint64_t', a type the generated"
            " code names without std::",
        )
        oneof = (
            'field { name: "x" number: 1 label: LABEL_OPTIONAL type: TYPE_INT32'
            ' oneof_index: 0 } oneof_decl { name: "size_t" }'
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=oneof))),
            "oneof pkg.Node.size_t is named 'size_t'",
        )
        nested = f'nested_type {{ name: "int64_t" {FIELD_X} }}'
        self.assertError(
            nproto_generator.generate(request(message_file(members=nested))),
            "message pkg.Node.int64_t is named 'int64_t'",
        )

    def test_a_nested_message_named_like_a_cpp_keyword(self):
        members = f'nested_type {{ name: "union" {FIELD_X} }}'
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "message pkg.Node.union is named 'union', a C++ keyword",
        )

    def test_a_nested_type_and_a_field_of_one_name(self):
        members = (
            'field { name: "child" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_MESSAGE type_name: ".pkg.Node.child" } nested_type { name:'
            ' "child" ' + FIELD_X + " }"
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "pkg.Node has a nested type and a field both named child",
        )

    def test_two_oneof_fields_that_become_one_index_constant(self):
        # The two names would share a JSON name, which protobuf refuses; explicit JSON names set them apart.
        members = (
            'field { name: "a_b" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_INT32 oneof_index: 0 json_name: "first" } field { name: "aB"'
            " number: 2 label: LABEL_OPTIONAL type: TYPE_INT32 oneof_index: 0"
            ' json_name: "second" } oneof_decl { name: "choice" }'
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "fields a_b and aB of pkg.Node both name the oneof index constant"
            " kABIndex",
        )

    def test_two_fields_that_become_one_default_constant(self):
        members = (
            'field { name: "a_b" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_INT32 json_name: "first" default_value: "2" } field { name:'
            ' "aB" number: 2 label: LABEL_OPTIONAL type: TYPE_INT32 json_name:'
            ' "second" default_value: "3" }'
        )
        text = message_file(members=members).replace(
            'syntax: "proto3"', 'syntax: "proto2"'
        )
        self.assertError(
            nproto_generator.generate(request(text)),
            "fields a_b and aB of pkg.Node both name the default constant"
            " kDefaultAB",
        )

    def test_a_default_constant_and_an_index_constant_of_one_name(self):
        members = (
            'field { name: "default_x" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_INT32 oneof_index: 0 } field { name: "x_index" number: 2'
            ' label: LABEL_OPTIONAL type: TYPE_INT32 default_value: "2" }'
            ' oneof_decl { name: "choice" }'
        )
        text = message_file(members=members).replace(
            'syntax: "proto3"', 'syntax: "proto2"'
        )
        self.assertError(
            nproto_generator.generate(request(text)),
            "fields default_x and x_index of pkg.Node both name the default"
            " constant kDefaultXIndex",
        )

    def test_a_default_constant_named_like_a_member(self):
        members = (
            'field { name: "kDefaultX" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_INT32 } field { name: "x" number: 2 label: LABEL_OPTIONAL'
            ' type: TYPE_INT32 default_value: "2" }'
        )
        text = message_file(members=members).replace(
            'syntax: "proto3"', 'syntax: "proto2"'
        )
        self.assertError(
            nproto_generator.generate(request(text)),
            "pkg.Node has a member or nested type named kDefaultX, which is"
            " also the default constant of field x",
        )

    def test_an_enumerator_too_long_for_a_line(self):
        long_value = f'COLOR_{"VERY_" * 25}DARK'
        enum = enum_file(
            values=(
                'value { name: "COLOR_UNSPECIFIED" number: 0 } value {'
                f' name: "{long_value}" number: 1 }}'
            )
        )
        self.assertError(
            nproto_generator.generate(request(enum)),
            "pkg/color.proto: a name is too long for nproto's code",
            "of pkg/color.nproto.h, `kVeryVery",
            "more than the 80 of a line",
        )

    def test_a_type_name_too_long_for_a_line_even_wrapped(self):
        # `const Node::<name>& lhs,` of operator== does not fit a line even with every break clang-format can make.
        nested = f'A{"VeryLong" * 9}Name'
        self.assertError(
            nproto_generator.generate(
                request(
                    message_file(members=f'nested_type {{ name: "{nested}" }}')
                )
            ),
            "pkg/node.proto: a name is too long for nproto's code",
            nested,
            "even broken as clang-format breaks it",
        )

    def test_a_file_the_descriptor_api_rejects(self):
        members = (
            'field { name: "a_b" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_INT32 } field { name: "aB" number: 2 label: LABEL_OPTIONAL'
            " type: TYPE_INT32 }"
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "pkg/node.proto: not a valid .proto file",
        )

    def test_enum_values_that_become_one_enumerator(self):
        enum = enum_file(
            name="pkg/kind.proto",
            option="pkg::nproto::Kind",
            enum="Kind",
            values=(
                'value { name: "KIND_A_B" number: 0 } value { name: "KIND_A__B"'
                " number: 1 }"
            ),
        )
        self.assertError(
            nproto_generator.generate(request(enum)),
            "enum values KIND_A_B and KIND_A__B of pkg.Kind both become the"
            " enumerator kAB",
        )


class GeneratedLayoutTest(GeneratorTestCase):
    def test_lint_directives_of_the_proto_are_not_copied(self):
        text = (
            message_file()
            + " source_code_info { location { path: [4, 0, 2, 0] span: [1, 1,"
            ' 1] leading_comments: " The weight.\\n LINT.IfChange(weight)\\n"'
            ' trailing_comments: " LINT.ThenChange(//a/b.cc:weight)\\n" } }'
        )
        header = self.assertGenerates(nproto_generator.generate(request(text)))[
            "pkg/node.nproto.h"
        ]
        self.assertIn("  // The weight.\n  double x = 0.0;", header)
        self.assertNotIn("LINT.", header)

    def test_nolint_markers_of_the_proto_are_not_copied(self):
        text = (
            message_file()
            + " source_code_info { location { path: [4, 0, 2, 0] span: [1, 1,"
            ' 1] leading_comments: " The weight.\\n NOLINTNEXTLINE(frame-naming):'
            ' a reason\\n" trailing_comments: " NOLINT(frame-naming): a'
            ' reason\\n" } }'
        )
        header = self.assertGenerates(nproto_generator.generate(request(text)))[
            "pkg/node.nproto.h"
        ]
        self.assertIn("  // The weight.\n  double x = 0.0;", header)
        self.assertNotIn("NOLINT", header)

    def test_a_statement_longer_than_a_line_is_wrapped_as_clang_format_wraps_it(
        self,
    ):
        name = f'a_{"very_" * 7}long_weight'
        members = (
            f'field {{ name: "{name}" number: 1 label: LABEL_OPTIONAL type:'
            " TYPE_DOUBLE proto3_optional: true oneof_index: 0 } oneof_decl"
            f' {{ name: "_{name}" }}'
        )
        source = self.assertGenerates(
            nproto_generator.generate(request(message_file(members=members)))
        )["pkg/node.nproto.pb.cc"]
        lines = source.splitlines()
        self.assertEqual(
            [
                line
                for line in lines
                if len(line) > nproto_generator.COLUMN_LIMIT
            ],
            [],
        )
        # The arguments do not fit after the open parenthesis, so clang-format breaks after it; two of them do not fit
        # one line at the continuation indent either, so each gets one.
        self.assertIn(
            "  ::nproto::internal::CopyOptionalFromProto(\n"
            f"      proto.has_{name}(),\n"
            f"      proto.{name}(),\n"
            f"      &value->{name});\n",
            source,
        )
        # A single argument that does not fit after the call moves to the next line.
        self.assertIn(
            f"    proto->set_{name}(\n        *value.{name});\n", source
        )

    def test_a_default_in_exponent_form_gets_a_radix_point(self):
        # protoc sends the default as it prints it: 1.0e-8 arrives as `1e-08`, 2.5e3 as `2500`.
        members = (
            'field { name: "tolerance" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_DOUBLE default_value: "1e-08" } field { name: "gain" number:'
            ' 2 label: LABEL_OPTIONAL type: TYPE_FLOAT default_value: "-2e+20"'
            ' } field { name: "ratio" number: 3 label: LABEL_OPTIONAL type:'
            ' TYPE_DOUBLE default_value: "0.25" } field { name: "count" number:'
            ' 4 label: LABEL_OPTIONAL type: TYPE_DOUBLE default_value: "3" }'
        )
        text = message_file(members=members).replace(
            'syntax: "proto3"', 'syntax: "proto2"'
        )
        header = self.assertGenerates(nproto_generator.generate(request(text)))[
            "pkg/node.nproto.h"
        ]
        self.assertIn(
            "static constexpr double kDefaultTolerance = 1.0e-08;", header
        )
        self.assertIn(
            "static constexpr float kDefaultGain = -2.0e+20f;", header
        )
        self.assertIn("static constexpr double kDefaultRatio = 0.25;", header)
        self.assertIn("static constexpr double kDefaultCount = 3.0;", header)

    def test_an_explicit_default_of_a_number_is_a_named_constant(self):
        # The magic-number rule (tools/hooks/magic_number.py): a number default is a constant of the struct and the
        # member's initializer; a bool, string or enum default, and the implicit 0 of a field without one, stay as they
        # are.
        members = (
            'field { name: "count" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_INT32 default_value: "5" } field { name: "gain" number: 2'
            ' label: LABEL_OPTIONAL type: TYPE_DOUBLE default_value: "-1.5" }'
            ' field { name: "ratio" number: 3 label: LABEL_OPTIONAL type:'
            ' TYPE_FLOAT default_value: "1" } field { name: "enabled" number: 4'
            ' label: LABEL_OPTIONAL type: TYPE_BOOL default_value: "true" }'
            ' field { name: "title" number: 5 label: LABEL_OPTIONAL type:'
            ' TYPE_STRING default_value: "hi" } field { name: "level" number: 6'
            " label: LABEL_OPTIONAL type: TYPE_ENUM type_name:"
            ' ".pkg.Node.Level" default_value: "LEVEL_HIGH" } field { name:'
            ' "required_count" number: 7 label: LABEL_REQUIRED type: TYPE_INT32'
            ' } enum_type { name: "Level" value { name: "LEVEL_LOW" number: 1 }'
            ' value { name: "LEVEL_HIGH" number: 2 } }'
        )
        text = message_file(members=members).replace(
            'syntax: "proto3"', 'syntax: "proto2"'
        )
        header = self.assertGenerates(nproto_generator.generate(request(text)))[
            "pkg/node.nproto.h"
        ]
        self.assertIn(
            "  // The explicit defaults of the .proto file ([default = ...]),"
            " the\n  // initializers of the members.\n  static constexpr"
            " int32_t kDefaultCount = 5;\n  static constexpr double"
            " kDefaultGain = -1.5;\n  static constexpr float kDefaultRatio ="
            " 1.0f;\n\n  int32_t count = kDefaultCount;\n  double gain ="
            " kDefaultGain;\n  float ratio = kDefaultRatio;\n  bool enabled ="
            ' true;\n  std::string title = "hi";\n  Level level ='
            " Level::kHigh;\n  int32_t required_count = 0;\n};",
            header,
        )
        # Equality compares the members only.
        self.assertNotIn("lhs.kDefault", header)
        proto3 = self.assertGenerates(
            nproto_generator.generate(request(message_file()))
        )
        self.assertNotIn("kDefault", proto3["pkg/node.nproto.h"])

    def test_a_default_constant_longer_than_a_line_is_wrapped_as_clang_format_wraps_it(
        self,
    ):
        name = f'a_{"very_" * 10}long_tolerance'
        constant = f'kDefaultA{"Very" * 10}LongTolerance'
        members = (
            f'field {{ name: "{name}" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_DOUBLE default_value: "1e-06" }'
        )
        text = message_file(members=members).replace(
            'syntax: "proto3"', 'syntax: "proto2"'
        )
        header = self.assertGenerates(nproto_generator.generate(request(text)))[
            "pkg/node.nproto.h"
        ]
        self.assertEqual(
            [
                line
                for line in header.splitlines()
                if len(line) > nproto_generator.COLUMN_LIMIT
            ],
            [],
        )
        # The constant does not fit before its `=`, so clang-format breaks before its name; the member breaks after
        # its `=`.
        self.assertIn(
            f"  static constexpr double\n      {constant} = 1.0e-06;\n",
            header,
        )
        self.assertIn(f"  double {name} =\n      {constant};\n", header)
        # operator== breaks its comparison after the operator, the second operand aligned with the first.
        self.assertIn(f"  if (lhs.{name} !=\n      rhs.{name}) {{\n", header)

    def test_a_comparison_longer_than_a_line_is_wrapped_as_clang_format_wraps_it(
        self,
    ):
        nested = f'A{"VeryLong" * 6}Name'
        header = self.assertGenerates(
            nproto_generator.generate(
                request(
                    message_file(
                        members=f'nested_type {{ name: "{nested}" {FIELD_X} }}'
                    )
                )
            )
        )["pkg/node.nproto.h"]
        for operator in ("==", "!="):
            self.assertIn(
                f"inline bool operator{operator}(\n"
                f"    const Node::{nested}& lhs,\n"
                f"    const Node::{nested}& rhs) {{\n",
                header,
            )
        # Without fields, the parameters are unnamed comments. An empty ToProto() whose `{}` does not fit after the
        # last parameter closes its body on the next line.
        nested = f'A{"VeryLong" * 4}Name'
        files = self.assertGenerates(
            nproto_generator.generate(
                request(
                    message_file(members=f'nested_type {{ name: "{nested}" }}')
                )
            )
        )
        for content in files.values():
            self.assertEqual(
                [
                    line
                    for line in content.splitlines()
                    if len(line) > nproto_generator.COLUMN_LIMIT
                ],
                [],
            )
        self.assertIn(
            f"inline bool operator==(\n    const Node::{nested}& /*lhs*/,\n",
            files["pkg/node.nproto.h"],
        )
        self.assertIn(
            "void ToProto(\n"
            f"    const Node::{nested}& /*value*/,\n"
            f"    ::pkg::Node_{nested}* absl_nonnull /*proto*/) {{\n"
            "}\n",
            files["pkg/node.nproto.pb.cc"],
        )

    def test_an_enum_conversion_longer_than_a_line_is_wrapped_as_clang_format_wraps_it(
        self,
    ):
        long_value = f'COLOR_{"DEEP_" * 12}RED'
        enum = enum_file(
            values=(
                'value { name: "COLOR_UNSPECIFIED" number: 0 } value {'
                f' name: "{long_value}" number: 1 }}'
            )
        )
        source = self.assertGenerates(nproto_generator.generate(request(enum)))[
            "pkg/color.nproto.pb.cc"
        ]
        enumerator = f'Color::kDeep{"Deep" * 11}Red'
        self.assertIn(
            "    case static_cast<int32_t>(\n"
            f"        {enumerator}):\n"
            f"      *value = {enumerator};\n",
            source,
        )
        # The error of an unknown number never fits one line after `return` at its indentation.
        self.assertIn(
            "      return ::nproto::internal::UnknownEnumValueError(\n"
            '          static_cast<int32_t>(proto), "pkg.Color");\n',
            source,
        )

    def test_template_arguments_and_nested_calls_wrap_as_clang_format_wraps_them(
        self,
    ):
        enum = "ASurfaceContactPlanningKind"
        members = (
            'field { name: "kinds" number: 1 label: LABEL_REPEATED type:'
            f' TYPE_ENUM type_name: ".pkg.Node.{enum}" }} field {{ name:'
            ' "planned_swing_trajectory" number: 2 label: LABEL_OPTIONAL type:'
            ' TYPE_MESSAGE type_name: ".pkg.Node.SwingTrajectoryParameters"'
            " oneof_index: 0 } field { name:"
            ' "terminal_swing_height_in_meters" number: 3 label:'
            " LABEL_OPTIONAL type: TYPE_DOUBLE oneof_index: 0 } field {"
            ' name: "kind_of_the_planned_contact" number: 4 label:'
            f' LABEL_OPTIONAL type: TYPE_ENUM type_name: ".pkg.Node.{enum}"'
            ' oneof_index: 0 } oneof_decl { name: "swing" } nested_type {'
            f' name: "SwingTrajectoryParameters" {FIELD_X} }} enum_type {{'
            f' name: "{enum}" value {{ name:'
            ' "A_SURFACE_CONTACT_PLANNING_KIND_FLAT" number: 0 } }'
        )
        files = self.assertGenerates(
            nproto_generator.generate(request(message_file(members=members)))
        )
        # A declaration's template arguments, which do not fit a line, go one per line (BinPackParameters: false).
        self.assertIn(
            "  std::variant<std::monostate,\n"
            "               SwingTrajectoryParameters,\n"
            "               double,\n"
            f"               {enum}>\n"
            "      swing;\n",
            files["pkg/node.nproto.h"],
        )
        source = files["pkg/node.nproto.pb.cc"]
        # A call's template argument that does not fit breaks after its `<`.
        self.assertIn(
            "  status = ::nproto::internal::EnumsFromProto<\n"
            f"      ::pkg::Node_{enum}>(proto.kinds(), &value->kinds);\n",
            source,
        )
        self.assertIn(
            "    ::nproto::internal::CopyAlternativeFromProto<\n"
            "        Node::kTerminalSwingHeightInMetersIndex>(\n"
            "        proto.terminal_swing_height_in_meters(),"
            " &value->swing);\n",
            source,
        )
        # A nested call breaks inside its own parentheses when the outer call's do not help.
        self.assertIn(
            "      proto->set_kind_of_the_planned_contact(\n         "
            f" static_cast<::pkg::Node_{enum}>(\n             "
            " std::get<Node::kKindOfThePlannedContactIndex>(value.swing)));\n",
            source,
        )
        # The arguments of a call with two of them align after its open parenthesis.
        self.assertIn(
            "      return"
            ' ::nproto::internal::AnnotateError("planned_swing_trajectory",\n  '
            "                                             status);\n",
            source,
        )
        for name, content in files.items():
            self.assertEqual(
                [
                    line
                    for line in content.splitlines()
                    if len(line) > nproto_generator.COLUMN_LIMIT
                ],
                [],
                name,
            )


# A nested message Child of Node, for the fields that hold one.
CHILD = f'nested_type {{ name: "Child" {FIELD_X} }} '
OPTIONAL_MESSAGE = "options { [nproto.optional_message]: true }"
RETIRED_DEPENDENCIES = (
    "nproto/options.proto",
    "nproto/retired_field_options.proto",
)


class OptionalMessageTest(GeneratorTestCase):
    def test_an_optional_message_field_is_a_std_optional(self):
        members = (
            'field { name: "child" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_MESSAGE type_name: ".pkg.Node.Child" '
            + OPTIONAL_MESSAGE
            + " } "
            + CHILD
        )
        files = self.assertGenerates(
            nproto_generator.generate(request(message_file(members=members)))
        )
        header = files["pkg/node.nproto.h"]
        self.assertIn("#include <optional>", header)
        self.assertIn("  std::optional<Child> child;\n", header)
        source = files["pkg/node.nproto.pb.cc"]
        self.assertIn("  if (value.child.has_value()) {", source)
        self.assertIn(
            "    ToProto(*value.child, proto->mutable_child());", source
        )
        self.assertIn("    proto->clear_child();", source)
        self.assertIn(
            "  status = ::nproto::internal::ConvertOptionalFromProto(\n"
            "      proto.has_child(), proto.child(), &value->child);\n",
            source,
        )

    def test_a_scalar_field_cannot_carry_it(self):
        members = (
            'field { name: "x" number: 1 label: LABEL_OPTIONAL type:'
            " TYPE_DOUBLE " + OPTIONAL_MESSAGE + " }"
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "field pkg.Node.x sets option (nproto.optional_message)",
            "only a singular message field",
        )

    def test_a_repeated_message_field_cannot_carry_it(self):
        members = (
            'field { name: "children" number: 1 label: LABEL_REPEATED type:'
            ' TYPE_MESSAGE type_name: ".pkg.Node.Child" '
            + OPTIONAL_MESSAGE
            + " } "
            + CHILD
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "field pkg.Node.children sets option (nproto.optional_message)",
        )

    def test_a_oneof_alternative_cannot_carry_it(self):
        members = (
            'field { name: "child" number: 1 label: LABEL_OPTIONAL type:'
            ' TYPE_MESSAGE type_name: ".pkg.Node.Child" oneof_index: 0 '
            + OPTIONAL_MESSAGE
            + ' } field { name: "x" number: 2 label: LABEL_OPTIONAL type:'
            ' TYPE_DOUBLE oneof_index: 0 } oneof_decl { name: "choice" } '
            + CHILD
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "field pkg.Node.child sets option (nproto.optional_message)",
            "oneof choice",
        )


def retired_entries(*entries: tuple[str, str]) -> str:
    """The (nproto.retired_field) options of `entries`, (name, replacement) each, as text."""
    return " ".join(
        f'[nproto.retired_field] {{ name: "{name}" replacement:'
        f' "{replacement}" }}'
        for name, replacement in entries
    )


class RetiredFieldTest(GeneratorTestCase):
    def generate_retired(
        self, *entries: tuple[str, str], members: str = FIELD_X
    ) -> plugin_pb2.CodeGeneratorResponse:
        return nproto_generator.generate(
            request(
                message_file(
                    members=members,
                    message_options=retired_entries(*entries),
                    dependencies=RETIRED_DEPENDENCIES,
                )
            )
        )

    def test_retired_names_beside_the_live_fields_generate(self):
        files = self.assertGenerates(
            self.generate_retired(("old_x", "write x"), ("legacyY", "drop it"))
        )
        # The retired names are options only: the struct holds the live fields.
        self.assertNotIn("old_x", files["pkg/node.nproto.h"])

    def test_a_retired_name_that_is_a_live_field(self):
        self.assertError(
            self.generate_retired(("x", "write x")),
            "pkg/node.proto: message pkg.Node: option (nproto.retired_field)"
            " retires 'x', which is the live field x",
        )

    def test_a_camel_case_spelling_of_a_live_field(self):
        members = (
            'field { name: "step_width" number: 1 label: LABEL_OPTIONAL type:'
            " TYPE_DOUBLE }"
        )
        self.assertError(
            self.generate_retired(
                ("stepWidth", "write step_width"), members=members
            ),
            "retires 'stepWidth', which is the live field step_width",
        )

    def test_a_name_listed_twice(self):
        self.assertError(
            self.generate_retired(("useOld", "a"), ("use_old", "b")),
            "retires 'use_old' and 'useOld', which are one name",
        )

    def test_an_entry_without_a_replacement(self):
        self.assertError(
            self.generate_retired(("old", "")), "needs a name and a replacement"
        )

    def test_nested_messages_are_checked_too(self):
        members = (
            'nested_type { name: "Child" options { '
            + retired_entries(("x", "write x"))
            + " } "
            + FIELD_X
            + " }"
        )
        self.assertError(
            nproto_generator.generate(
                request(
                    message_file(
                        members=members, dependencies=RETIRED_DEPENDENCIES
                    )
                )
            ),
            "message pkg.Node.Child: option (nproto.retired_field) retires 'x'",
        )

    def test_the_comparison_form_is_snake_case_in_lower_case(self):
        self.assertEqual(
            nproto_generator.retired_key("useDcmTerminalCost"),
            "use_dcm_terminal_cost",
        )
        self.assertEqual(nproto_generator.retired_key("Q_final"), "q_final")


class NamingTest(unittest.TestCase):
    def test_enumerators_drop_the_enum_prefix(self):
        self.assertEqual(
            nproto_generator.enumerator_name("KIND_SWING_IN_FLIGHT", "KIND_"),
            "kSwingInFlight",
        )
        self.assertEqual(
            nproto_generator.enumerator_name(
                "CONTROLLER_TYPE_LINEAR", "CONTROLLER_TYPE_"
            ),
            "kLinear",
        )
        self.assertEqual(
            nproto_generator.enumerator_name("LEVEL_10", "LEVEL_"), "k10"
        )
        self.assertEqual(
            nproto_generator.enumerator_name("lowerCase", ""), "kLowerCase"
        )

    def test_camel_case_follows_protobuf(self):
        self.assertEqual(
            nproto_generator.camel_case("swing_in_flight"), "SwingInFlight"
        )
        self.assertEqual(nproto_generator.camel_case("level10x"), "Level10X")

    def test_string_literals_escape_every_byte_that_needs_it(self):
        self.assertEqual(
            nproto_generator.cpp_string_literal(b'a"b\\c\n\x01\xff?'),
            '"a\\"b\\\\c\\n\\001\\377\\077"',
        )


class CommentFittingTest(unittest.TestCase):
    def test_a_long_line_breaks_into_the_next_line_of_its_paragraph(self):
        words = " ".join(["word"] * 20)
        fitted = nproto_generator._fit_comments(
            f"  // {words}\n  // tail\nint x;\n"
        ).split("\n")
        for line in fitted:
            self.assertLessEqual(len(line), nproto_generator.COLUMN_LIMIT, line)
        self.assertEqual(
            " ".join(line[len("  // ") :] for line in fitted[:-2]),
            f"{words} tail",
        )
        self.assertEqual(fitted[-2:], ["int x;", ""])

    def test_a_list_item_or_another_indentation_starts_a_line_of_its_own(self):
        long_line = "// " + " ".join(["word"] * 20)
        for following in ("// - item", "  // indented", "//", "int x;"):
            fitted = nproto_generator._fit_comments(f"{long_line}\n{following}")
            self.assertEqual(fitted.rsplit("\n", maxsplit=1)[-1], following)

    def test_short_lines_code_and_unbreakable_words_are_left_alone(self):
        source = "// short\n" + "x" * 100 + "\n// " + "y" * 100 + "\n"
        self.assertEqual(nproto_generator._fit_comments(source), source)


if __name__ == "__main__":
    unittest.main()
