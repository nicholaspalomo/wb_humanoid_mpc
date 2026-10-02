"""The nproto generator on requests built here: what it generates, and the files it refuses with a clear error.

The requests hold FileDescriptorProtos written as text, as protoc would send them, so that the cases nproto must
reject - which the repository's lint keeps out of real .proto files - need no .proto file of their own.
"""

import os
import subprocess
import sys
import unittest
from typing import Dict, List, Sequence

from google.protobuf import descriptor_pb2
from google.protobuf import text_format
from google.protobuf.compiler import plugin_pb2

from nproto import options_pb2
import nproto_generator

PLUGIN = os.path.join(
    os.environ.get("TEST_SRCDIR", ""), "_main", "tools", "nproto", "protoc-gen-nproto"
)


def file_proto(text: str) -> descriptor_pb2.FileDescriptorProto:
    return text_format.Parse(text, descriptor_pb2.FileDescriptorProto())


def library_files() -> List[descriptor_pb2.FileDescriptorProto]:
    """descriptor.proto and nproto/options.proto, which every request carries."""
    files = []
    for module in (descriptor_pb2, options_pb2):
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


FIELD_X = 'field { name: "x" number: 1 label: LABEL_OPTIONAL type: TYPE_DOUBLE }'


def message_file(
    name: str = "pkg/node.proto",
    package: str = "pkg",
    option: str = "pkg::nproto::Node",
    message: str = "Node",
    members: str = FIELD_X,
    dependencies: Sequence[str] = ("nproto/options.proto",),
) -> str:
    """A file of one message, `members` its fields and nested types, its option `option` (none when empty)."""
    options = f'options {{ [nproto.generate_struct]: "{option}" }} ' if option else ""
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
    options = f'options {{ [nproto.generate_enum]: "{option}" }} ' if option else ""
    return raw_file(name, package, f'enum_type {{ name: "{enum}" {options}{values} }}')


def raw_file(
    name: str,
    package: str,
    body: str,
    dependencies: Sequence[str] = ("nproto/options.proto",),
) -> str:
    imports = "".join(f'dependency: "{dependency}" ' for dependency in dependencies)
    return f'name: "{name}" package: "{package}" syntax: "proto3" {imports} {body}'


def generated(response: plugin_pb2.CodeGeneratorResponse) -> Dict[str, str]:
    return {file.name: file.content for file in response.file}


class GeneratorTestCase(unittest.TestCase):
    def assertError(
        self, response: plugin_pb2.CodeGeneratorResponse, *fragments: str
    ) -> None:
        self.assertTrue(response.error, "expected an error, got files")
        self.assertFalse(response.file, "an error generates no files")
        for fragment in fragments:
            self.assertIn(fragment, response.error)

    def assertGenerates(
        self, response: plugin_pb2.CodeGeneratorResponse
    ) -> Dict[str, str]:
        self.assertEqual(response.error, "")
        return generated(response)


class GeneratedFilesTest(GeneratorTestCase):
    def test_a_message_gets_a_struct_header_and_conversions(self):
        files = self.assertGenerates(nproto_generator.generate(request(message_file())))
        self.assertEqual(
            sorted(files),
            ["pkg/node.nproto.h", "pkg/node.nproto.pb.cc", "pkg/node.nproto.pb.h"],
        )
        header = files["pkg/node.nproto.h"]
        self.assertIn("namespace pkg::nproto {", header)
        self.assertIn("struct Node {", header)
        self.assertIn("double x = 0.0;", header)
        self.assertIn(
            "inline bool operator==(const Node& lhs, const Node& rhs)", header
        )
        # The struct header knows nothing of protobuf.
        includes = [line for line in header.splitlines() if line.startswith("#include")]
        self.assertFalse(
            [line for line in includes if "google/protobuf" in line or ".pb.h" in line],
            includes,
        )
        conversions = files["pkg/node.nproto.pb.h"]
        self.assertIn('#include "pkg/node.pb.h"', conversions)
        self.assertIn(
            "void ToProto(const Node& value, ::pkg::Node* proto);", conversions
        )
        self.assertIn(
            "absl::Status FromProto(const ::pkg::Node& proto, Node* value);",
            conversions,
        )

    def test_the_namespace_is_the_one_the_option_names(self):
        files = self.assertGenerates(
            nproto_generator.generate(
                request(message_file(option="::robot::deep::name_space::Node"))
            )
        )
        self.assertIn("namespace robot::deep::name_space {", files["pkg/node.nproto.h"])
        self.assertIn(
            "}  // namespace robot::deep::name_space", files["pkg/node.nproto.pb.cc"]
        )

    def test_nested_types_are_nested_in_the_struct_and_need_no_option(self):
        members = (
            'field { name: "kind" number: 1 label: LABEL_OPTIONAL type: TYPE_ENUM type_name: ".pkg.Node.Kind" } '
            'field { name: "child" number: 2 label: LABEL_OPTIONAL type: TYPE_MESSAGE type_name: ".pkg.Node.Child" } '
            'nested_type { name: "Child" ' + FIELD_X + " } "
            'enum_type { name: "Kind" value { name: "KIND_STANCE" number: 0 } value { name: "KIND_SWING" number: 1 } }'
        )
        files = self.assertGenerates(
            nproto_generator.generate(request(message_file(members=members)))
        )
        header = files["pkg/node.nproto.h"]
        self.assertIn(
            "struct Node {\n  // pkg.Node.Kind\n  enum class Kind : std::int32_t {",
            header,
        )
        self.assertIn("  struct Child {\n    double x = 0.0;\n  };", header)
        self.assertIn("  Kind kind = Kind::kStance;\n  Child child;\n", header)
        conversions = files["pkg/node.nproto.pb.h"]
        self.assertIn(
            "void ToProto(const Node::Child& value, ::pkg::Node_Child* proto);",
            conversions,
        )
        self.assertIn(
            "void ToProto(Node::Kind value, ::pkg::Node_Kind* proto);", conversions
        )

    def test_the_response_declares_proto3_optional_and_editions(self):
        response = nproto_generator.generate(request(message_file()))
        features = plugin_pb2.CodeGeneratorResponse
        self.assertTrue(response.supported_features & features.FEATURE_PROTO3_OPTIONAL)
        self.assertTrue(
            response.supported_features & features.FEATURE_SUPPORTS_EDITIONS
        )
        self.assertEqual(response.minimum_edition, descriptor_pb2.EDITION_PROTO2)
        self.assertEqual(response.maximum_edition, descriptor_pb2.EDITION_2023)

    def test_an_enum_file_gets_an_enum_class(self):
        files = self.assertGenerates(nproto_generator.generate(request(enum_file())))
        header = files["pkg/color.nproto.h"]
        self.assertIn("namespace pkg::nproto {", header)
        self.assertIn("enum class Color : std::int32_t {", header)
        self.assertIn("kUnspecified = 0,", header)
        self.assertIn("kDarkRed = 3,", header)
        self.assertIn(
            "absl::Status FromProto(::pkg::Color proto, Color* value);",
            files["pkg/color.nproto.pb.h"],
        )

    def test_a_file_without_definitions_gets_empty_files(self):
        options_only = 'name: "pkg/extensions.proto" package: "pkg" syntax: "proto3"'
        files = self.assertGenerates(nproto_generator.generate(request(options_only)))
        self.assertEqual(len(files), 3)
        for content in files.values():
            self.assertNotIn("struct", content)

    def test_an_import_with_a_struct_is_included(self):
        point = message_file(
            name="pkg/point.proto", option="pkg::nproto::Point", message="Point"
        )
        members = 'field { name: "point" number: 1 label: LABEL_OPTIONAL type: TYPE_MESSAGE type_name: ".pkg.Point" }'
        node = message_file(
            members=members, dependencies=("nproto/options.proto", "pkg/point.proto")
        )
        files = self.assertGenerates(nproto_generator.generate(request(point, node)))
        self.assertIn('#include "pkg/point.nproto.h"', files["pkg/node.nproto.h"])
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

    def test_options_parsed_before_the_extensions_were_registered_are_read(self):
        # A process that parses the request before it imports nproto's options_pb2 holds the options as unknown fields.
        script = (
            "import sys\n"
            "from google.protobuf.compiler import plugin_pb2\n"
            "request = plugin_pb2.CodeGeneratorRequest.FromString(sys.stdin.buffer.read())\n"
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
        self.assertIn("enum class Color : std::int32_t {", files["pkg/color.nproto.h"])

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
            "pkg/node.proto: message Node does not name the C++ type nproto generates for it",
            'option (nproto.generate_struct) = "<namespace>::Node";',
            "as the first statement of its body",
            'import "nproto/options.proto"',
        )

    def test_an_enum_without_the_option(self):
        self.assertError(
            nproto_generator.generate(request(enum_file(option=""))),
            "pkg/color.proto: enum Color does not name the C++ type nproto generates for it",
            'option (nproto.generate_enum) = "<namespace>::Color";',
        )

    def test_a_nested_message_with_an_option_of_its_own(self):
        members = (
            'nested_type { name: "Child" options { [nproto.generate_struct]: "pkg::nproto::Child" } '
            + FIELD_X
            + " }"
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "pkg/node.proto: nested message pkg.Node.Child sets option (nproto.generate_struct)",
            "nested C++ types of their parent's struct (pkg::nproto::Node::Child)",
            "carry no option of their own",
        )

    def test_a_nested_enum_with_an_option_of_its_own(self):
        members = (
            'nested_type { name: "Child" enum_type { name: "Kind" '
            'options { [nproto.generate_enum]: "pkg::nproto::Kind" } value { name: "KIND_A" number: 0 } } }'
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "nested enum pkg.Node.Child.Kind sets option (nproto.generate_enum)",
            "(pkg::nproto::Node::Child::Kind)",
        )

    def test_a_recursive_message(self):
        members = 'field { name: "child" number: 1 label: LABEL_OPTIONAL type: TYPE_MESSAGE type_name: ".pkg.Node" }'
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "pkg/node.proto",
            "message pkg.Node contains itself (pkg.Node -> pkg.Node",
            "by value",
        )

    def test_a_message_recursive_through_a_repeated_nested_message(self):
        members = (
            'field { name: "branch" number: 1 label: LABEL_OPTIONAL type: TYPE_MESSAGE type_name: ".pkg.Node.Branch" } '
            'nested_type { name: "Branch" field { name: "nodes" number: 1 label: LABEL_REPEATED type: TYPE_MESSAGE '
            'type_name: ".pkg.Node" } }'
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
            'message Node: option (nproto.generate_struct) = "pkg::nproto::Other" names a type Other',
            'named after the message: "pkg::nproto::Node"',
        )

    def test_an_option_naming_another_enum(self):
        self.assertError(
            nproto_generator.generate(request(enum_file(option="pkg::nproto::Hue"))),
            'enum Color: option (nproto.generate_enum) = "pkg::nproto::Hue" names a type Hue',
            '"pkg::nproto::Color"',
        )

    def test_an_option_that_is_not_a_cpp_name(self):
        self.assertError(
            nproto_generator.generate(request(message_file(option="pkg.nproto.Node"))),
            "is not a fully qualified C++ name",
        )

    def test_an_option_with_a_cpp_keyword(self):
        self.assertError(
            nproto_generator.generate(
                request(message_file(option="pkg::namespace::Node"))
            ),
            '"namespace" is a C++ keyword',
        )

    def test_an_option_in_the_global_namespace(self):
        self.assertError(
            nproto_generator.generate(request(message_file(option="Node"))),
            "global namespace",
        )

    def test_an_option_naming_the_protobuf_class(self):
        self.assertError(
            nproto_generator.generate(request(message_file(option="pkg::Node"))),
            "is the C++ name protobuf already gives pkg.Node",
        )

    def test_two_top_level_definitions(self):
        body = (
            'message_type { name: "Node" options { [nproto.generate_struct]: "pkg::nproto::Node" } } '
            'enum_type { name: "Kind" options { [nproto.generate_enum]: "pkg::nproto::Kind" } '
            'value { name: "KIND_A" number: 0 } }'
        )
        self.assertError(
            nproto_generator.generate(request(raw_file("pkg/node.proto", "pkg", body))),
            "defines 2 top-level types (message Node, enum Kind)",
        )

    def test_a_field_of_a_type_without_a_struct(self):
        stamp = raw_file(
            "pkg/stamp.proto",
            "pkg",
            'message_type { name: "Stamp" field { name: "seconds" number: 1 label: LABEL_OPTIONAL type: TYPE_INT64 } }',
            dependencies=(),
        )
        members = 'field { name: "stamp" number: 1 label: LABEL_OPTIONAL type: TYPE_MESSAGE type_name: ".pkg.Stamp" }'
        node = message_file(
            members=members, dependencies=("nproto/options.proto", "pkg/stamp.proto")
        )
        self.assertError(
            nproto_generator.generate(request(stamp, node)),
            "field pkg.Node.stamp has type pkg.Stamp, which has no nproto struct",
            "message pkg.Stamp (pkg/stamp.proto) sets no option (nproto.generate_struct)",
        )

    def test_a_field_of_a_nested_type_of_a_message_without_a_struct(self):
        stamp = raw_file(
            "pkg/stamp.proto",
            "pkg",
            'message_type { name: "Stamp" enum_type { name: "Unit" value { name: "UNIT_S" number: 0 } } }',
            dependencies=(),
        )
        members = 'field { name: "unit" number: 1 label: LABEL_OPTIONAL type: TYPE_ENUM type_name: ".pkg.Stamp.Unit" }'
        node = message_file(
            members=members, dependencies=("nproto/options.proto", "pkg/stamp.proto")
        )
        self.assertError(
            nproto_generator.generate(request(stamp, node)),
            "field pkg.Node.unit has type pkg.Stamp.Unit, which has no nproto struct: message pkg.Stamp",
        )

    def test_a_field_named_like_a_cpp_keyword(self):
        members = (
            'field { name: "class" number: 1 label: LABEL_OPTIONAL type: TYPE_INT32 }'
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "field pkg.Node.class is named 'class', a C++ keyword",
        )

    def test_a_nested_message_named_like_a_cpp_keyword(self):
        members = 'nested_type { name: "union" ' + FIELD_X + " }"
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "message pkg.Node.union is named 'union', a C++ keyword",
        )

    def test_a_nested_type_and_a_field_of_one_name(self):
        members = (
            'field { name: "child" number: 1 label: LABEL_OPTIONAL type: TYPE_MESSAGE type_name: ".pkg.Node.child" } '
            'nested_type { name: "child" ' + FIELD_X + " }"
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "pkg.Node has a nested type and a field both named child",
        )

    def test_two_oneof_fields_that_become_one_index_constant(self):
        # The two names would share a JSON name, which protobuf refuses; explicit JSON names set them apart.
        members = (
            'field { name: "a_b" number: 1 label: LABEL_OPTIONAL type: TYPE_INT32 oneof_index: 0 json_name: "first" } '
            'field { name: "aB" number: 2 label: LABEL_OPTIONAL type: TYPE_INT32 oneof_index: 0 json_name: "second" } '
            'oneof_decl { name: "choice" }'
        )
        self.assertError(
            nproto_generator.generate(request(message_file(members=members))),
            "fields a_b and aB of pkg.Node both name the oneof index constant kABIndex",
        )

    def test_a_file_the_descriptor_api_rejects(self):
        members = (
            'field { name: "a_b" number: 1 label: LABEL_OPTIONAL type: TYPE_INT32 } '
            'field { name: "aB" number: 2 label: LABEL_OPTIONAL type: TYPE_INT32 }'
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
            values='value { name: "KIND_A_B" number: 0 } value { name: "KIND_A__B" number: 1 }',
        )
        self.assertError(
            nproto_generator.generate(request(enum)),
            "enum values KIND_A_B and KIND_A__B of pkg.Kind both become the enumerator kAB",
        )


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
        self.assertEqual(nproto_generator.enumerator_name("LEVEL_10", "LEVEL_"), "k10")
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


if __name__ == "__main__":
    unittest.main()
