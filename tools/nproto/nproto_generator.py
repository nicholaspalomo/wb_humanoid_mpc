"""nproto's code generator: plain C++ structs and their protobuf conversions from .proto files.

Every .proto file of the repository defines one top-level message or enum, and that definition names the C++ type
nproto generates for it with an option, the first statement of its body (tools/nproto/README.md):

    message Vector3 {
      option (nproto.generate_struct) = "ocs2::humanoid::msgs::Vector3";
      ...
    }

    enum ControllerType {
      option (nproto.generate_enum) = "ocs2::humanoid::msgs::ControllerType";
      ...
    }

Its nested messages and enums become nested C++ types of that struct (TargetContactPatch::Kind) and carry no option of
their own. For every file protoc asks it to generate, the generator writes three files next to protoc's own output:

- `<file>.nproto.h`: the struct (or the `enum class` for an enum file) in the namespace of the option, with Eigen and
  standard types only and no protobuf include, plus `operator==` / `operator!=`;
- `<file>.nproto.pb.h` and `<file>.nproto.pb.cc`: `ToProto()` and `FromProto()` between the struct and the message of
  the protobuf C++ code, written so that converting into objects of the same shape does not allocate.

The generator is plain Python on the descriptor API, so that the protoc plugin (protoc_gen_nproto.py) and the tests
share it: `generate(request)` turns a CodeGeneratorRequest into a CodeGeneratorResponse, and a file nproto cannot map
is an error of the response that names the file and the definition.

protoc hands a plugin the custom options of a definition as extension fields of its MessageOptions / EnumOptions. They
are only readable as extensions when nproto/options.proto's extensions are registered when the request is parsed,
which importing `options_pb2` here does; `definition_option()` also parses the options again, in case they were parsed
before (then they are unknown fields).
"""

import dataclasses
import re
import textwrap
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from google.protobuf import descriptor as descriptor_module
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor_pool
from google.protobuf.compiler import plugin_pb2

from nproto import options_pb2

FieldDescriptor = descriptor_module.FieldDescriptor
Descriptor = descriptor_module.Descriptor
EnumDescriptor = descriptor_module.EnumDescriptor
FileDescriptor = descriptor_module.FileDescriptor

# The files generated for each .proto file, which nproto_cc_library declares as the outputs of the plugin.
# LINT.IfChange(generated_suffixes)
STRUCT_HEADER_SUFFIX = ".nproto.h"
CONVERSIONS_HEADER_SUFFIX = ".nproto.pb.h"
CONVERSIONS_SOURCE_SUFFIX = ".nproto.pb.cc"
# LINT.ThenChange(//tools/nproto/nproto.bzl:generated_suffixes)
# The option a top-level definition names its C++ type with, by the keyword of the definition, as error messages write
# it; the generator reads the options themselves through options_pb2.
# LINT.IfChange(option_names)
STRUCT_OPTION = "(nproto.generate_struct)"
ENUM_OPTION = "(nproto.generate_enum)"
OPTION_OF = {"message": STRUCT_OPTION, "enum": ENUM_OPTION}
OPTIONS_IMPORT = "nproto/options.proto"
# LINT.ThenChange(//tools/nproto/options.proto:nproto_options)
README = "tools/nproto/README.md"
RUNTIME_HEADER = "nproto/Conversions.h"

# The repository's .clang-format.
COLUMN_LIMIT = 140
INDENT = "  "
# The width the generator wraps its own comments to, so that they stay clear of the column limit when indented.
COMMENT_WIDTH = 116

# The editions the generator understands: proto2, proto3 and edition 2023 (descriptor.proto's Edition numbers).
MINIMUM_EDITION = descriptor_pb2.EDITION_PROTO2
MAXIMUM_EDITION = descriptor_pb2.EDITION_2023

# Source code info paths (descriptor.proto field numbers).
_FILE_MESSAGE_TYPE = 4
_FILE_ENUM_TYPE = 5
_MESSAGE_FIELD = 2
_MESSAGE_NESTED_TYPE = 3
_MESSAGE_ENUM_TYPE = 4
_MESSAGE_ONEOF = 8
_ENUM_VALUE = 2

_NEXT_ID = re.compile(r"Next ID: \d+")
_QUALIFIED_NAME = re.compile(r"(::)?[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_][A-Za-z0-9_]*)*")

_CPP_KEYWORDS = frozenset(
    """alignas alignof and and_eq asm atomic_cancel atomic_commit atomic_noexcept auto bitand bitor bool break case
    catch char char8_t char16_t char32_t class compl concept const consteval constexpr constinit const_cast continue
    co_await co_return co_yield decltype default delete do double dynamic_cast else enum explicit export extern false
    float for friend goto if inline int long mutable namespace new noexcept not not_eq nullptr operator or or_eq
    private protected public reflexpr register reinterpret_cast requires return short signed sizeof static
    static_assert static_cast struct switch synchronized template this thread_local throw true try typedef typeid
    typename union unsigned using virtual void volatile wchar_t while xor xor_eq""".split()
)

_INTEGER_TYPES = {
    FieldDescriptor.TYPE_INT64: "std::int64_t",
    FieldDescriptor.TYPE_SINT64: "std::int64_t",
    FieldDescriptor.TYPE_SFIXED64: "std::int64_t",
    FieldDescriptor.TYPE_UINT64: "std::uint64_t",
    FieldDescriptor.TYPE_FIXED64: "std::uint64_t",
    FieldDescriptor.TYPE_INT32: "std::int32_t",
    FieldDescriptor.TYPE_SINT32: "std::int32_t",
    FieldDescriptor.TYPE_SFIXED32: "std::int32_t",
    FieldDescriptor.TYPE_UINT32: "std::uint32_t",
    FieldDescriptor.TYPE_FIXED32: "std::uint32_t",
}
_FLOATING_TYPES = {
    FieldDescriptor.TYPE_DOUBLE: "double",
    FieldDescriptor.TYPE_FLOAT: "float",
}
_STRING_TYPES = (FieldDescriptor.TYPE_STRING, FieldDescriptor.TYPE_BYTES)
_MESSAGE_TYPES = (FieldDescriptor.TYPE_MESSAGE, FieldDescriptor.TYPE_GROUP)
# repeated double and repeated float are dense Eigen vectors; every other repeated field is a std::vector.
_EIGEN_VECTORS = {
    FieldDescriptor.TYPE_DOUBLE: "Eigen::VectorXd",
    FieldDescriptor.TYPE_FLOAT: "Eigen::VectorXf",
}


class GenerationError(Exception):
    """A .proto file nproto cannot generate a struct for. The message names the file and the definition."""


def snake_case(name: str) -> str:
    """`MpcPolicy` -> `mpc_policy`, `Vector3` -> `vector3` (as tools/hooks/proto_file_layout.py names files)."""
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    words = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words)
    return words.lower()


def camel_case(name: str) -> str:
    """protobuf's UnderscoresToCamelCase with the first letter capitalized: `swing_in_flight` -> `SwingInFlight`."""
    result = []
    capitalize = True
    for character in name:
        if "a" <= character <= "z":
            result.append(character.upper() if capitalize else character)
            capitalize = False
        elif "A" <= character <= "Z":
            result.append(character)
            capitalize = False
        elif "0" <= character <= "9":
            result.append(character)
            capitalize = True
        else:
            capitalize = True
    return "".join(result)


def enumerator_name(value_name: str, prefix: str) -> str:
    """The Google-style enumerator of a proto enum value: `KIND_SWING_IN_FLIGHT` with prefix `KIND_` -> `kSwingInFlight`."""
    stem = value_name[len(prefix) :]
    if re.fullmatch(r"[A-Z0-9_]+", stem):
        words = [word for word in stem.split("_") if word]
        return "k" + "".join(word[0] + word[1:].lower() for word in words)
    return "k" + camel_case(stem)


def output_stem(proto_file_name: str) -> str:
    """`humanoid_mpc_msgs/mpc_policy.proto` -> `humanoid_mpc_msgs/mpc_policy`, the stem of every generated file."""
    if proto_file_name.endswith(".proto"):
        return proto_file_name[: -len(".proto")]
    return proto_file_name


def cpp_string_literal(data: bytes) -> str:
    """A C++ string literal holding exactly `data`; non-printable bytes become three-digit octal escapes."""
    pieces = ['"']
    for byte in data:
        character = chr(byte)
        if character in ('"', "\\"):
            pieces.append("\\" + character)
        elif character == "\n":
            pieces.append("\\n")
        elif 0x20 <= byte < 0x7F and character != "?":
            pieces.append(character)
        else:
            pieces.append("\\%03o" % byte)
    pieces.append('"')
    return "".join(pieces)


@dataclasses.dataclass(frozen=True)
class Definition:
    """A message or enum of a .proto file, as the descriptor protoc sends describes it."""

    keyword: str  # "message" or "enum"
    proto: Union[descriptor_pb2.DescriptorProto, descriptor_pb2.EnumDescriptorProto]

    @property
    def name(self) -> str:
        return self.proto.name

    @property
    def option(self) -> str:
        """The option this kind of definition names its C++ type with: (nproto.generate_struct) or _enum."""
        return OPTION_OF[self.keyword]


def top_level_definitions(
    file_proto: descriptor_pb2.FileDescriptorProto,
) -> List[Definition]:
    """The file's top-level messages, then its top-level enums."""
    return [Definition("message", message) for message in file_proto.message_type] + [
        Definition("enum", enum) for enum in file_proto.enum_type
    ]


def _extension_value(options, extension) -> Optional[str]:
    """The value of `extension` in `options`, or None when it is not set.

    When the options were parsed before nproto/options.proto's extensions were registered, the option is an unknown
    field of them; parsing their bytes again, now that the extensions are registered, makes it an extension.
    """
    if options.HasExtension(extension):
        return options.Extensions[extension]
    reparsed = type(options).FromString(options.SerializeToString())
    if reparsed.HasExtension(extension):
        return reparsed.Extensions[extension]
    return None


def definition_option(definition: Definition) -> Optional[str]:
    """The value of the definition's `option (nproto.generate_struct)` (enums: generate_enum), or None."""
    if definition.keyword == "message":
        return _extension_value(definition.proto.options, options_pb2.generate_struct)
    return _extension_value(definition.proto.options, options_pb2.generate_enum)


def has_struct(file_proto: descriptor_pb2.FileDescriptorProto) -> bool:
    """Whether nproto generates a struct or enum class for the file: one of its definitions names one."""
    return any(
        definition_option(definition) is not None
        for definition in top_level_definitions(file_proto)
    )


def _wrap_comment(text: str, width: int = COMMENT_WIDTH) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def _comment_lines(text: str) -> List[str]:
    """`// ...` lines of a comment taken from the .proto file, one per line of the original."""
    lines = text.rstrip("\n").split("\n")
    result: List[str] = []
    for line in lines:
        stripped = line.rstrip()
        if stripped.startswith(" "):
            stripped = stripped[1:]
        if _NEXT_ID.fullmatch(stripped):
            continue  # tools/hooks/proto_next_id.py's bookkeeping, which means nothing to the struct
        result.append(("// " + stripped) if stripped else "//")
    # A blank line closing a comment would leave a lone "//"; drop it, as clang-format leaves it alone but it adds noise.
    while result and result[-1] == "//":
        result.pop()
    while result and result[0] == "//":
        result.pop(0)
    return result


@dataclasses.dataclass(frozen=True)
class TypeInfo:
    """Where the struct of one message or enum lives and what protobuf calls it in C++."""

    # ("ocs2", "humanoid", "msgs", "TargetContactPatch", "Kind")
    struct_path: Tuple[str, ...]
    proto_class: str  # "::humanoid_mpc_msgs::TargetContactPatch_Kind"
    is_enum: bool
    file_name: str


class TypeTable:
    """The structs of every message and enum of the request's files that names one, by protobuf full name."""

    def __init__(self, file_protos: Sequence[descriptor_pb2.FileDescriptorProto]):
        self.types: Dict[str, TypeInfo] = {}
        self.files: Dict[str, descriptor_pb2.FileDescriptorProto] = {}
        for file_proto in file_protos:
            self.files[file_proto.name] = file_proto
            package = file_proto.package.split(".") if file_proto.package else []
            for definition in top_level_definitions(file_proto):
                value = definition_option(definition)
                if value is None:
                    continue  # its types cannot be used: lookup() says so
                path = validate_definition_option(file_proto, definition, value)
                if definition.keyword == "message":
                    self._add_message(
                        definition.proto,
                        path,
                        package,
                        [definition.name],
                        file_proto.name,
                    )
                else:
                    self._add_enum(
                        definition.proto,
                        path,
                        package,
                        [definition.name],
                        file_proto.name,
                    )

    def _full_name(self, package: List[str], nesting: List[str]) -> str:
        return ".".join(package + nesting)

    def _proto_class(self, package: List[str], nesting: List[str]) -> str:
        return "::" + "::".join(package + ["_".join(nesting)])

    def _add_message(self, message, struct_path, package, nesting, file_name):
        if message.options.map_entry:
            return
        full_name = self._full_name(package, nesting)
        self.types[full_name] = TypeInfo(
            struct_path, self._proto_class(package, nesting), False, file_name
        )
        for nested in message.nested_type:
            self._add_message(
                nested,
                struct_path + (nested.name,),
                package,
                nesting + [nested.name],
                file_name,
            )
        for enum in message.enum_type:
            self._add_enum(
                enum,
                struct_path + (enum.name,),
                package,
                nesting + [enum.name],
                file_name,
            )

    def _add_enum(self, enum, struct_path, package, nesting, file_name):
        full_name = self._full_name(package, nesting)
        self.types[full_name] = TypeInfo(
            struct_path, self._proto_class(package, nesting), True, file_name
        )

    def lookup(self, full_name: str, user: str) -> TypeInfo:
        """The struct of the message or enum `full_name`, which the field `user` refers to."""
        if full_name in self.types:
            return self.types[full_name]
        # The top-level definition that holds the type, to name what lacks the option.
        owner = None
        for file_proto in self.files.values():
            prefix = file_proto.package + "." if file_proto.package else ""
            if not full_name.startswith(prefix):
                continue
            top = full_name[len(prefix) :].split(".")[0]
            for definition in top_level_definitions(file_proto):
                if definition.name == top:
                    owner = (file_proto.name, definition, prefix + top)
                    break
            if owner:
                break
        if owner is None:
            raise GenerationError(
                f"{user} has type {full_name}, which nproto cannot find in the request."
            )
        file_name, definition, top_name = owner
        raise GenerationError(
            f"{user} has type {full_name}, which has no nproto struct: {definition.keyword} {top_name} "
            f"({file_name}) sets no option {definition.option} (well-known types such as google.protobuf.Timestamp "
            f"have none either). Give it the option, 'option {definition.option} = \"<namespace>::{definition.name}\";' "
            "as the first statement of its body, or use another type."
        )


def validate_definition_option(
    file_proto: descriptor_pb2.FileDescriptorProto, definition: Definition, value: str
) -> Tuple[str, ...]:
    """The C++ path the option `value` of the top-level `definition` names; raises GenerationError when nproto cannot use it."""
    name = file_proto.name
    keyword = definition.keyword
    option = definition.option
    what = f"{keyword} {definition.name}"
    if not _QUALIFIED_NAME.fullmatch(value):
        raise GenerationError(
            f'{name}: {what}: option {option} = "{value}" is not a fully qualified C++ name such as '
            f'"my::project::{definition.name}".'
        )
    path = tuple(value.lstrip(":").split("::"))
    for component in path:
        if component in _CPP_KEYWORDS:
            raise GenerationError(
                f'{name}: {what}: option {option} = "{value}": "{component}" is a C++ keyword.'
            )
    if path[-1] != definition.name:
        expected = (
            "::".join(path[:-1] + (definition.name,))
            if len(path) > 1
            else definition.name
        )
        raise GenerationError(
            f'{name}: {what}: option {option} = "{value}" names a type {path[-1]}, but the C++ type is named after '
            f'the {keyword}: "{expected}".'
        )
    if len(path) < 2:
        raise GenerationError(
            f'{name}: {what}: option {option} = "{value}" puts the type in the global namespace; name a namespace, '
            f'e.g. "my::project::{definition.name}".'
        )
    package = file_proto.package.split(".") if file_proto.package else []
    if list(path) == package + [definition.name]:
        raise GenerationError(
            f'{name}: {what}: option {option} = "{value}" is the C++ name protobuf already gives '
            f"{'.'.join(package + [definition.name])}; put the type in a namespace of its own, e.g. "
            f"\"{'::'.join(package)}::nproto::{definition.name}\"."
        )
    return path


def _nested_options(definition: Definition, scope: str) -> List[Tuple[str, str, str]]:
    """(keyword, full name, option) of every type nested in `definition` that sets an nproto option of its own."""
    found: List[Tuple[str, str, str]] = []
    if definition.keyword != "message":
        return found
    for nested in definition.proto.nested_type:
        if nested.options.map_entry:
            continue
        child = Definition("message", nested)
        child_name = f"{scope}.{nested.name}"
        if definition_option(child) is not None:
            found.append(("message", child_name, child.option))
        found.extend(_nested_options(child, child_name))
    for enum in definition.proto.enum_type:
        child = Definition("enum", enum)
        if definition_option(child) is not None:
            found.append(("enum", f"{scope}.{enum.name}", child.option))
    return found


def validate_option(file_proto: descriptor_pb2.FileDescriptorProto) -> Tuple[str, ...]:
    """The C++ path of the file's one top-level definition; raises GenerationError when nproto cannot generate it."""
    name = file_proto.name
    definitions = top_level_definitions(file_proto)
    if not definitions:
        return ()
    if len(definitions) > 1:
        listed = ", ".join(
            f"{definition.keyword} {definition.name}" for definition in definitions
        )
        raise GenerationError(
            f"{name}: defines {len(definitions)} top-level types ({listed}); a file holds exactly one message or enum "
            "(tools/hooks/proto_file_layout.py), whose option names the one struct or enum class nproto generates "
            "for the file."
        )
    definition = definitions[0]
    value = definition_option(definition)
    if value is None:
        raise GenerationError(
            f"{name}: {definition.keyword} {definition.name} does not name the C++ type nproto generates for it. "
            f"Write 'option {definition.option} = \"<namespace>::{definition.name}\";' as the first statement of its "
            f"body, with 'import \"{OPTIONS_IMPORT}\";' ({README})."
        )
    path = validate_definition_option(file_proto, definition, value)
    package = file_proto.package + "." if file_proto.package else ""
    nested = _nested_options(definition, package + definition.name)
    if nested:
        keyword, full_name, option = nested[0]
        cpp_name = "::".join(path[:-1] + tuple(full_name[len(package) :].split(".")))
        raise GenerationError(
            f"{name}: nested {keyword} {full_name} sets option {option}. Nested types are generated as nested C++ "
            f"types of their parent's struct ({cpp_name}) and carry no option of their own; remove it."
        )
    return path


@dataclasses.dataclass
class _Field:
    """One field of a message as the struct holds it."""

    descriptor: FieldDescriptor
    raw: descriptor_pb2.FieldDescriptorProto
    name: str  # the struct member: the proto field name
    accessor: str  # protobuf's C++ accessor stem: the field name in lower case
    kind: str  # one of the _KIND_* constants
    optional: bool  # std::optional<T>
    oneof: Optional[str]  # the real oneof the field belongs to
    comment: List[str]


_KIND_SCALAR = "scalar"  # numbers and bool
_KIND_STRING = "string"  # string and bytes
_KIND_ENUM = "enum"
_KIND_MESSAGE = "message"
_KIND_EIGEN = "eigen"  # repeated double / float
_KIND_SCALARS = "scalars"  # other repeated numbers and bool
_KIND_STRINGS = "strings"
_KIND_ENUMS = "enums"
_KIND_MESSAGES = "messages"
_KIND_MAP = "map"


class _Includes:
    """The #include lines of one generated file, in the blocks clang-format keeps apart."""

    def __init__(self):
        self.standard: Set[str] = set()
        self.eigen = False
        self.absl: Set[str] = set()
        self.project: Set[str] = set()

    def lines(self) -> List[str]:
        blocks: List[List[str]] = []
        if self.standard:
            blocks.append([f"#include <{header}>" for header in sorted(self.standard)])
        if self.eigen:
            blocks.append(["#include <Eigen/Core>"])
        if self.absl:
            blocks.append([f'#include "{header}"' for header in sorted(self.absl)])
        if self.project:
            blocks.append([f'#include "{header}"' for header in sorted(self.project)])
        lines: List[str] = []
        for block in blocks:
            if lines:
                lines.append("")
            lines.extend(block)
        return lines


def _sort_by_dependencies(
    nodes: List[str], depends_on: Dict[str, Set[str]]
) -> List[str]:
    """`nodes` in declaration order, except that each comes after the nodes it depends on."""
    ordered: List[str] = []
    placed: Set[str] = set()

    def place(node: str, visiting: Set[str]) -> None:
        if node in placed:
            return
        visiting.add(node)
        for dependency in nodes:
            if (
                dependency in depends_on.get(node, set())
                and dependency not in placed
                and dependency not in visiting
            ):
                place(dependency, visiting)
        visiting.discard(node)
        placed.add(node)
        ordered.append(node)

    for node in nodes:
        place(node, set())
    return ordered


class _FileGenerator:
    """Generates the three files of one .proto file."""

    def __init__(
        self,
        file_descriptor: FileDescriptor,
        file_proto: descriptor_pb2.FileDescriptorProto,
        table: TypeTable,
    ):
        self.file = file_descriptor
        self.proto = file_proto
        self.table = table
        self.name = file_proto.name
        self.stem = output_stem(file_proto.name)
        self.path = validate_option(file_proto)
        self.namespace = self.path[:-1]
        self.comments: Dict[Tuple[int, ...], str] = {}
        for location in file_proto.source_code_info.location:
            text = location.leading_comments
            if location.trailing_comments:
                text = (text + "\n" if text else "") + location.trailing_comments
            if text.strip():
                self.comments[tuple(location.path)] = text
        self.raw_fields: Dict[str, descriptor_pb2.FieldDescriptorProto] = {}
        self.synthetic_oneofs: Dict[str, Set[str]] = {}
        package = [file_proto.package] if file_proto.package else []
        for message in file_proto.message_type:
            self._index_raw(message, ".".join(package + [message.name]))
        self.fields: Dict[str, List[_Field]] = {}
        self.uses_limits = False

    # ----------------------------------------------------------------------------------------------------------------
    # Descriptors
    # ----------------------------------------------------------------------------------------------------------------

    def _index_raw(
        self, message: descriptor_pb2.DescriptorProto, full_name: str
    ) -> None:
        synthetic = set()
        for field in message.field:
            self.raw_fields[full_name + "." + field.name] = field
            if field.proto3_optional:
                synthetic.add(message.oneof_decl[field.oneof_index].name)
        self.synthetic_oneofs[full_name] = synthetic
        for nested in message.nested_type:
            self._index_raw(nested, full_name + "." + nested.name)

    def _comment(self, path: Tuple[int, ...]) -> List[str]:
        text = self.comments.get(path)
        return _comment_lines(text) if text else []

    def _check_name(self, name: str, what: str) -> None:
        if name in _CPP_KEYWORDS:
            raise GenerationError(
                f"{self.name}: {what} is named '{name}', a C++ keyword, which cannot name a struct member; rename it."
            )

    def _fields(self, message: Descriptor) -> List[_Field]:
        if message.full_name in self.fields:
            return self.fields[message.full_name]
        path = self._message_path(message)
        synthetic = self.synthetic_oneofs.get(message.full_name, set())
        fields: List[_Field] = []
        for index, field in enumerate(message.fields):
            self._check_name(field.name, f"field {field.full_name}")
            raw = self.raw_fields[field.full_name]
            oneof = field.containing_oneof
            real_oneof = (
                oneof.name
                if oneof is not None and oneof.name not in synthetic
                else None
            )
            kind = self._kind(field)
            explicit = raw.proto3_optional
            if (
                not explicit
                and real_oneof is None
                and not field.is_repeated
                and field.type not in _MESSAGE_TYPES
            ):
                # proto2 `optional` and edition 2023's EXPLICIT presence without a default: absent has no value.
                explicit = (
                    field.has_presence
                    and not field.has_default_value
                    and not field.is_required
                )
            fields.append(
                _Field(
                    descriptor=field,
                    raw=raw,
                    name=field.name,
                    accessor=field.name.lower(),
                    kind=kind,
                    optional=explicit,
                    oneof=real_oneof,
                    comment=self._comment(path + (_MESSAGE_FIELD, index)),
                )
            )
        self.fields[message.full_name] = fields
        return fields

    @staticmethod
    def _is_map(field: FieldDescriptor) -> bool:
        return (
            field.is_repeated
            and field.type == FieldDescriptor.TYPE_MESSAGE
            and field.message_type.GetOptions().map_entry
        )

    def _kind(self, field: FieldDescriptor) -> str:
        if self._is_map(field):
            return _KIND_MAP
        if field.is_repeated:
            if field.type in _EIGEN_VECTORS:
                return _KIND_EIGEN
            if field.type in _STRING_TYPES:
                return _KIND_STRINGS
            if field.type == FieldDescriptor.TYPE_ENUM:
                return _KIND_ENUMS
            if field.type in _MESSAGE_TYPES:
                return _KIND_MESSAGES
            return _KIND_SCALARS
        if field.type in _STRING_TYPES:
            return _KIND_STRING
        if field.type == FieldDescriptor.TYPE_ENUM:
            return _KIND_ENUM
        if field.type in _MESSAGE_TYPES:
            return _KIND_MESSAGE
        return _KIND_SCALAR

    def _messages(self) -> List[Descriptor]:
        """Every message of the file, nested ones included, map entries excluded."""
        result: List[Descriptor] = []

        def visit(message: Descriptor) -> None:
            if message.GetOptions().map_entry:
                return
            result.append(message)
            for nested in message.nested_types:
                visit(nested)

        for message in self.file.message_types_by_name.values():
            visit(message)
        return result

    def _referenced_messages(
        self, message: Descriptor
    ) -> List[Tuple[FieldDescriptor, Descriptor]]:
        """The message types the fields of `message` hold, with the field that holds each (map values included)."""
        references = []
        for field in message.fields:
            if self._is_map(field):
                value = field.message_type.fields_by_name["value"]
                if value.type in _MESSAGE_TYPES:
                    references.append((field, value.message_type))
            elif field.type in _MESSAGE_TYPES:
                references.append((field, field.message_type))
        return references

    def check_recursion(self) -> None:
        """A message that holds itself, directly or through other messages, has no struct: fields are held by value."""
        messages = {message.full_name: message for message in self._messages()}
        state: Dict[str, int] = {}  # 1: on the stack, 2: done
        stack: List[str] = []

        def visit(name: str) -> None:
            state[name] = 1
            stack.append(name)
            for field, target in self._referenced_messages(messages[name]):
                if target.full_name not in messages:
                    continue  # another file: imports cannot form a cycle
                if state.get(target.full_name) == 1:
                    cycle = stack[stack.index(target.full_name) :] + [target.full_name]
                    raise GenerationError(
                        f"{self.name}: message {target.full_name} contains itself ({' -> '.join(cycle)}, through "
                        f"field {field.full_name}). nproto structs hold their fields by value, so a recursive "
                        "message has no struct; break the cycle (e.g. with an index into a repeated field)."
                    )
                if state.get(target.full_name) is None:
                    visit(target.full_name)
            stack.pop()
            state[name] = 2

        for name in messages:
            if state.get(name) is None:
                visit(name)

    # ----------------------------------------------------------------------------------------------------------------
    # Names
    # ----------------------------------------------------------------------------------------------------------------

    def _declared_names(self, message: Descriptor) -> Set[str]:
        """Every name the struct of `message` declares: nested types, members and oneof index constants."""
        names: Set[str] = set()
        for nested in message.nested_types:
            if not nested.GetOptions().map_entry:
                names.add(nested.name)
        for enum in message.enum_types:
            names.add(enum.name)
        for field in self._fields(message):
            names.add(field.oneof if field.oneof else field.name)
            if field.oneof:
                names.add(self._index_constant(field))
        return names

    @staticmethod
    def _index_constant(field: _Field) -> str:
        return "k" + camel_case(field.name) + "Index"

    def render(self, target: Tuple[str, ...], scope: Sequence[Descriptor]) -> str:
        """The C++ name of the struct or enum at `target` as code inside the structs `scope` (outermost first) writes it.

        Names in this file's namespace are written relative to it, unless a member or nested type of an enclosing
        struct would hide them; everything else is fully qualified.
        """
        namespace = self.namespace
        qualified = "::" + "::".join(target)
        if target[: len(namespace)] != namespace:
            return qualified
        remainder = target[len(namespace) :]
        chain = [message.name for message in scope]
        common = 0
        while (
            common < len(chain)
            and common < len(remainder)
            and chain[common] == remainder[common]
        ):
            common += 1
        if common == len(remainder):
            # An enclosing struct itself.
            relative: Tuple[str, ...] = (remainder[-1],)
            hiding_scopes = scope[common:]
        else:
            relative = remainder[common:]
            hiding_scopes = scope[common:]
        first = relative[0]
        for message in hiding_scopes:
            if first in self._declared_names(message):
                return qualified
        return "::".join(relative)

    def _type_info(self, full_name: str, user: str) -> TypeInfo:
        return self.table.lookup(full_name, user)

    def _value_type(
        self, field: FieldDescriptor, scope: Sequence[Descriptor], user: str
    ) -> str:
        """The C++ type of one value of `field` (an element, for a repeated field)."""
        if field.type in _FLOATING_TYPES:
            return _FLOATING_TYPES[field.type]
        if field.type in _INTEGER_TYPES:
            return _INTEGER_TYPES[field.type]
        if field.type == FieldDescriptor.TYPE_BOOL:
            return "bool"
        if field.type in _STRING_TYPES:
            return "std::string"
        if field.type == FieldDescriptor.TYPE_ENUM:
            return self.render(
                self._type_info(field.enum_type.full_name, user).struct_path, scope
            )
        return self.render(
            self._type_info(field.message_type.full_name, user).struct_path, scope
        )

    def _proto_value_class(self, field: FieldDescriptor, user: str) -> str:
        if field.type == FieldDescriptor.TYPE_ENUM:
            return self._type_info(field.enum_type.full_name, user).proto_class
        return self._type_info(field.message_type.full_name, user).proto_class

    def _member_type(
        self, field: _Field, scope: Sequence[Descriptor], includes: _Includes
    ) -> str:
        descriptor = field.descriptor
        user = f"field {descriptor.full_name}"
        if field.kind == _KIND_MAP:
            key = descriptor.message_type.fields_by_name["key"]
            value = descriptor.message_type.fields_by_name["value"]
            includes.standard.add("map")
            self._note_value_includes(key, includes)
            self._note_value_includes(value, includes)
            return f"std::map<{self._value_type(key, scope, user)}, {self._value_type(value, scope, user)}>"
        if field.kind == _KIND_EIGEN:
            includes.eigen = True
            return _EIGEN_VECTORS[descriptor.type]
        element = self._value_type(descriptor, scope, user)
        self._note_value_includes(descriptor, includes)
        if descriptor.is_repeated:
            includes.standard.add("vector")
            return f"std::vector<{element}>"
        if field.optional:
            includes.standard.add("optional")
            return f"std::optional<{element}>"
        return element

    @staticmethod
    def _note_value_includes(field: FieldDescriptor, includes: _Includes) -> None:
        if field.type in _INTEGER_TYPES:
            includes.standard.add("cstdint")
        elif field.type in _STRING_TYPES:
            includes.standard.add("string")

    def _enum_prefix(self, enum: EnumDescriptor) -> str:
        prefix = snake_case(enum.name).upper() + "_"
        if all(
            value.name.startswith(prefix) and len(value.name) > len(prefix)
            for value in enum.values
        ):
            return prefix
        return ""

    def _enumerators(self, enum: EnumDescriptor) -> List[Tuple[str, int, int]]:
        """(enumerator, number, index) of every value of `enum`; raises when two values would share an enumerator."""
        prefix = self._enum_prefix(enum)
        result = []
        seen: Dict[str, str] = {}
        for index, value in enumerate(enum.values):
            name = enumerator_name(value.name, prefix)
            if name in seen:
                raise GenerationError(
                    f"{self.name}: enum values {seen[name]} and {value.name} of {enum.full_name} both become the "
                    f"enumerator {name}; rename one of them."
                )
            seen[name] = value.name
            result.append((name, value.number, index))
        return result

    def _enumerator_for(self, enum: EnumDescriptor, number: int) -> str:
        for name, value_number, _ in self._enumerators(enum):
            if value_number == number:
                return name
        raise GenerationError(f"{self.name}: {enum.full_name} has no value {number}.")

    # ----------------------------------------------------------------------------------------------------------------
    # Defaults
    # ----------------------------------------------------------------------------------------------------------------

    def _float_literal(self, text: str, cpp_type: str) -> str:
        lowered = text.strip().lower()
        if lowered in ("inf", "infinity", "+inf"):
            self.uses_limits = True
            return f"std::numeric_limits<{cpp_type}>::infinity()"
        if lowered in ("-inf", "-infinity"):
            self.uses_limits = True
            return f"-std::numeric_limits<{cpp_type}>::infinity()"
        if lowered in ("nan", "-nan"):
            self.uses_limits = True
            return f"std::numeric_limits<{cpp_type}>::quiet_NaN()"
        literal = text.strip()
        if not any(character in literal for character in ".eE"):
            literal += ".0"
        return literal + ("f" if cpp_type == "float" else "")

    def _integer_literal(self, value: int, cpp_type: str) -> str:
        if cpp_type == "std::int64_t" and value == -(2**63):
            self.uses_limits = True
            return "std::numeric_limits<std::int64_t>::min()"
        if cpp_type == "std::int32_t" and value == -(2**31):
            self.uses_limits = True
            return "std::numeric_limits<std::int32_t>::min()"
        if cpp_type.startswith("std::uint") and value >= 2**31:
            return f"{value}u"
        return str(value)

    def _default(self, field: _Field, scope: Sequence[Descriptor]) -> Optional[str]:
        """The default member initializer of a singular, non-optional field outside a oneof, or None."""
        descriptor = field.descriptor
        explicit = descriptor.has_default_value
        if field.kind == _KIND_SCALAR:
            if descriptor.type == FieldDescriptor.TYPE_BOOL:
                return "true" if descriptor.default_value else "false"
            if descriptor.type in _FLOATING_TYPES:
                cpp_type = _FLOATING_TYPES[descriptor.type]
                text = (
                    field.raw.default_value
                    if explicit and field.raw.default_value
                    else "0"
                )
                return self._float_literal(text, cpp_type)
            return self._integer_literal(
                int(descriptor.default_value), _INTEGER_TYPES[descriptor.type]
            )
        if field.kind == _KIND_STRING:
            if not explicit:
                return None
            value = descriptor.default_value
            data = value if isinstance(value, bytes) else value.encode("utf-8")
            return cpp_string_literal(data) if data else None
        if field.kind == _KIND_ENUM:
            enum_type = self.render(
                self._type_info(
                    descriptor.enum_type.full_name, f"field {descriptor.full_name}"
                ).struct_path,
                scope,
            )
            return f"{enum_type}::{self._enumerator_for(descriptor.enum_type, int(descriptor.default_value))}"
        return None

    # ----------------------------------------------------------------------------------------------------------------
    # The struct header
    # ----------------------------------------------------------------------------------------------------------------

    def _enum_lines(
        self, enum: EnumDescriptor, path: Tuple[int, ...], indent: str
    ) -> List[str]:
        lines = [indent + line for line in self._comment(path)]
        if not lines:
            lines.append(f"{indent}// {enum.full_name}")
        lines.append(f"{indent}enum class {enum.name} : std::int32_t {{")
        for name, number, index in self._enumerators(enum):
            for comment in self._comment(path + (_ENUM_VALUE, index)):
                lines.append(indent + INDENT + comment)
            lines.append(f"{indent}{INDENT}{name} = {number},")
        lines.append(f"{indent}}};")
        return lines

    def _subtree_references(self, message: Descriptor) -> Set[str]:
        """The full names of every message and enum the fields of `message` and of its nested types refer to."""
        references: Set[str] = set()
        for field in message.fields:
            fields = [field]
            if self._is_map(field):
                fields = [
                    field.message_type.fields_by_name["key"],
                    field.message_type.fields_by_name["value"],
                ]
            for value in fields:
                if value.type in _MESSAGE_TYPES:
                    references.add(value.message_type.full_name)
                elif value.type == FieldDescriptor.TYPE_ENUM:
                    references.add(value.enum_type.full_name)
        for nested in message.nested_types:
            if not nested.GetOptions().map_entry:
                references |= self._subtree_references(nested)
        return references

    def _nested_messages_in_order(
        self, message: Descriptor, path: Tuple[int, ...]
    ) -> List[Tuple[Descriptor, Tuple[int, ...]]]:
        nested = [
            (child, path + (_MESSAGE_NESTED_TYPE, index))
            for index, child in enumerate(message.nested_types)
            if not child.GetOptions().map_entry
        ]
        by_name = {child.full_name: (child, child_path) for child, child_path in nested}
        depends_on: Dict[str, Set[str]] = {}
        for child, _ in nested:
            references = self._subtree_references(child)
            depends_on[child.full_name] = {
                other
                for other in by_name
                if other != child.full_name
                and any(
                    reference == other or reference.startswith(other + ".")
                    for reference in references
                )
            }
        order = _sort_by_dependencies(
            [child.full_name for child, _ in nested], depends_on
        )
        return [by_name[name] for name in order]

    def _struct_lines(
        self,
        message: Descriptor,
        path: Tuple[int, ...],
        scope: List[Descriptor],
        includes: _Includes,
    ) -> List[str]:
        indent = INDENT * len(scope)
        inner_scope = scope + [message]
        inner = INDENT * len(inner_scope)
        lines = [indent + line for line in self._comment(path)]
        if not lines:
            lines.append(f"{indent}// {message.full_name}")
        blocks: List[List[str]] = []
        for index, enum in enumerate(message.enum_types):
            includes.standard.add("cstdint")
            blocks.append(
                self._enum_lines(enum, path + (_MESSAGE_ENUM_TYPE, index), inner)
            )
        for nested, nested_path in self._nested_messages_in_order(message, path):
            blocks.append(
                self._struct_lines(nested, nested_path, inner_scope, includes)
            )

        fields = self._fields(message)
        constants: List[str] = []
        for field in fields:
            if field.oneof:
                includes.standard.add("cstddef")
                position = [f.name for f in fields if f.oneof == field.oneof].index(
                    field.name
                ) + 1
                constants.append(
                    f"{inner}static constexpr std::size_t {self._index_constant(field)} = {position};"
                )
        if constants:
            constants.insert(
                0,
                f"{inner}// The alternative of each oneof field in its std::variant (index 0 is std::monostate: none is set).",
            )
            blocks.append(constants)

        members: List[str] = []
        emitted_oneofs: Set[str] = set()
        for field in fields:
            if field.oneof:
                if field.oneof in emitted_oneofs:
                    continue
                emitted_oneofs.add(field.oneof)
                members.extend(
                    self._oneof_lines(
                        message, fields, field.oneof, inner_scope, includes
                    )
                )
                continue
            for comment in field.comment:
                members.append(inner + comment)
            member_type = self._member_type(field, inner_scope, includes)
            default = None
            if not field.optional and not field.descriptor.is_repeated:
                default = self._default(field, inner_scope)
            declaration = f"{inner}{member_type} {field.name}" + (
                f" = {default};" if default else ";"
            )
            if len(declaration) > COLUMN_LIMIT and default:
                members.append(f"{inner}{member_type} {field.name} =")
                members.append(f"{inner}{INDENT * 2}{default};")
            else:
                members.append(declaration)
        if members:
            blocks.append(members)

        if not blocks:
            lines.append(f"{indent}struct {message.name} {{}};")
            return lines
        lines.append(f"{indent}struct {message.name} {{")
        for number, block in enumerate(blocks):
            if number:
                lines.append("")
            lines.extend(block)
        lines.append(f"{indent}}};")
        return lines

    def _oneof_lines(
        self,
        message: Descriptor,
        fields: List[_Field],
        oneof: str,
        scope: Sequence[Descriptor],
        includes: _Includes,
    ) -> List[str]:
        inner = INDENT * len(scope)
        members = [field for field in fields if field.oneof == oneof]
        includes.standard.add("variant")
        oneof_descriptor = message.oneofs_by_name[oneof]
        lines: List[str] = []
        oneof_index = [o.name for o in message.oneofs].index(oneof)
        lines.extend(
            inner + comment
            for comment in self._comment(
                self._message_path(message) + (_MESSAGE_ONEOF, oneof_index)
            )
        )
        alternatives = ", ".join(
            f"{field.name} ({self._index_constant(field)})" for field in members
        )
        lines.extend(
            inner + "// " + line
            for line in _wrap_comment(f"oneof {oneof_descriptor.name}: {alternatives}.")
        )
        for field in members:
            for comment in field.comment:
                lines.append(
                    f"{inner}// {field.name}: {comment[3:]}"
                    if comment != "//"
                    else inner + comment
                )
        types = ["std::monostate"] + [
            self._member_type(field, scope, includes) for field in members
        ]
        declaration = f"{inner}std::variant<{', '.join(types)}> {oneof};"
        if len(declaration) > COLUMN_LIMIT:
            raise GenerationError(
                f"{self.name}: the std::variant of oneof {oneof_descriptor.full_name} is longer than "
                f"{COLUMN_LIMIT} columns; shorten the names of its alternatives' types."
            )
        lines.append(declaration)
        return lines

    def _message_path(self, message: Descriptor) -> Tuple[int, ...]:
        chain: List[Descriptor] = []
        current: Optional[Descriptor] = message
        while current is not None:
            chain.insert(0, current)
            current = current.containing_type
        top = list(self.file.message_types_by_name.values()).index(chain[0])
        path: Tuple[int, ...] = (_FILE_MESSAGE_TYPE, top)
        for parent, child in zip(chain, chain[1:]):
            path += (_MESSAGE_NESTED_TYPE, list(parent.nested_types).index(child))
        return path

    def _equality_lines(self, message: Descriptor, struct_name: str) -> List[str]:
        """operator== and operator!= of the struct of `message`, at namespace scope."""
        fields = self._fields(message)
        if not fields:
            return [
                f"inline bool operator==(const {struct_name}& /*lhs*/, const {struct_name}& /*rhs*/) {{",
                f"{INDENT}return true;",
                "}",
                "",
                f"inline bool operator!=(const {struct_name}& /*lhs*/, const {struct_name}& /*rhs*/) {{",
                f"{INDENT}return false;",
                "}",
            ]
        lines = [
            f"inline bool operator==(const {struct_name}& lhs, const {struct_name}& rhs) {{"
        ]
        members: List[Tuple[str, bool]] = []
        for field in fields:
            member = field.oneof if field.oneof else field.name
            if all(name != member for name, _ in members):
                members.append((member, field.kind == _KIND_EIGEN and not field.oneof))
        for member, is_eigen in members:
            # Eigen compares vectors of one size only, so the sizes are compared first.
            conditions = (
                [f"lhs.{member}.size() != rhs.{member}.size()"] if is_eigen else []
            )
            conditions.append(f"lhs.{member} != rhs.{member}")
            for condition in conditions:
                lines.append(f"{INDENT}if ({condition}) {{")
                lines.append(f"{INDENT * 2}return false;")
                lines.append(f"{INDENT}}}")
        lines.append(f"{INDENT}return true;")
        lines.append("}")
        lines.append("")
        lines.append(
            f"inline bool operator!=(const {struct_name}& lhs, const {struct_name}& rhs) {{"
        )
        lines.append(f"{INDENT}return !(lhs == rhs);")
        lines.append("}")
        return lines

    def _all_messages_post_order(self) -> List[Tuple[Descriptor, Tuple[int, ...]]]:
        """Every message of the file in the order its struct is complete: nested ones before their parents."""
        result: List[Tuple[Descriptor, Tuple[int, ...]]] = []

        def visit(message: Descriptor, path: Tuple[int, ...]) -> None:
            for nested, nested_path in self._nested_messages_in_order(message, path):
                visit(nested, nested_path)
            result.append((message, path))

        for index, message in enumerate(self.file.message_types_by_name.values()):
            visit(message, (_FILE_MESSAGE_TYPE, index))
        return result

    def _all_enums(self) -> List[EnumDescriptor]:
        enums = list(self.file.enum_types_by_name.values())
        for message, _ in self._all_messages_post_order():
            enums.extend(message.enum_types)
        return enums

    def _struct_name(self, descriptor) -> str:
        """The name of a struct or enum of this file at namespace scope: `TargetContactPatch::Kind`."""
        return self.render(self.table.types[descriptor.full_name].struct_path, [])

    def _header_comment(self, lines: List[str]) -> List[str]:
        return ["// " + line if line else "//" for line in lines]

    def _imports_with_structs(self) -> List[str]:
        stems = []
        for dependency in self.proto.dependency:
            dependency_proto = self.table.files.get(dependency)
            if dependency_proto is None or not has_struct(dependency_proto):
                continue
            stems.append(output_stem(dependency))
        return stems

    def struct_header(self) -> str:
        includes = _Includes()
        body: List[str] = []
        definition: Optional[str] = None
        if self.proto.message_type:
            top = self.file.message_types_by_name[self.proto.message_type[0].name]
            body.extend(self._struct_lines(top, (_FILE_MESSAGE_TYPE, 0), [], includes))
            for message, _ in self._all_messages_post_order():
                body.append("")
                body.extend(self._equality_lines(message, self._struct_name(message)))
            definition = f"struct {'::'.join(self.path)}"
        else:
            enum = self.file.enum_types_by_name[self.proto.enum_type[0].name]
            includes.standard.add("cstdint")
            body.extend(self._enum_lines(enum, (_FILE_ENUM_TYPE, 0), ""))
            definition = f"enum class {'::'.join(self.path)}"
        if self.uses_limits:
            includes.standard.add("limits")
        for stem in self._imports_with_structs():
            includes.project.add(stem + STRUCT_HEADER_SUFFIX)
        header = self._header_comment(
            _wrap_comment(
                f"Generated by nproto ({README}) from {self.name}. Do not edit.\n\n"
                f"The plain C++ {definition} of {self._top_full_name()}. Its conversions to and from the protobuf "
                f"{'message' if self.proto.message_type else 'enum'} are in {self.stem}{CONVERSIONS_HEADER_SUFFIX}."
            )
        )
        lines = header + ["", "#pragma once", ""]
        include_lines = includes.lines()
        if include_lines:
            lines += include_lines + [""]
        lines += self._namespace_open() + [""] + body + [""] + self._namespace_close()
        return "\n".join(lines) + "\n"

    def _top_full_name(self) -> str:
        package = self.proto.package + "." if self.proto.package else ""
        if self.proto.message_type:
            return package + self.proto.message_type[0].name
        return package + self.proto.enum_type[0].name

    def _namespace_open(self) -> List[str]:
        return [f"namespace {'::'.join(self.namespace)} {{"]

    def _namespace_close(self) -> List[str]:
        return [f"}}  // namespace {'::'.join(self.namespace)}"]

    # ----------------------------------------------------------------------------------------------------------------
    # The conversions
    # ----------------------------------------------------------------------------------------------------------------

    def _declarations(self) -> List[Tuple[str, str, str]]:
        """(comment, ToProto signature, FromProto signature) of every struct and enum of the file."""
        result = []
        for enum in self._all_enums():
            struct = self._struct_name(enum)
            proto = self.table.types[enum.full_name].proto_class
            result.append(
                (
                    enum.full_name,
                    f"void ToProto({struct} value, {proto}* proto)",
                    f"absl::Status FromProto({proto} proto, {struct}* value)",
                )
            )
        for message, _ in self._all_messages_post_order():
            struct = self._struct_name(message)
            proto = self.table.types[message.full_name].proto_class
            result.append(
                (
                    message.full_name,
                    f"void ToProto(const {struct}& value, {proto}* proto)",
                    f"absl::Status FromProto(const {proto}& proto, {struct}* value)",
                )
            )
        return result

    @staticmethod
    def _signature_lines(signature: str, suffix: str) -> List[str]:
        """A function signature, with its parameters one per line under the open parenthesis when it is too long."""
        line = signature + suffix
        if len(line) <= COLUMN_LIMIT:
            return [line]
        open_paren = signature.index("(")
        parameters = signature[open_paren + 1 : -1].split(", ")
        lines = [signature[: open_paren + 1] + parameters[0] + ","]
        for parameter in parameters[1:-1]:
            lines.append(" " * (open_paren + 1) + parameter + ",")
        lines.append(" " * (open_paren + 1) + parameters[-1] + ")" + suffix)
        return lines

    def conversions_header(self) -> str:
        includes = _Includes()
        includes.absl.add("absl/status/status.h")
        includes.project.add(self.stem + STRUCT_HEADER_SUFFIX)
        includes.project.add(self.stem + ".pb.h")
        body: List[str] = []
        for comment, to_proto, from_proto in self._declarations():
            if body:
                body.append("")
            body.append(f"// {comment}")
            body.extend(self._signature_lines(to_proto, ";"))
            body.extend(self._signature_lines(from_proto, ";"))
        header = self._header_comment(
            _wrap_comment(
                f"Generated by nproto ({README}) from {self.name}. Do not edit.\n\n"
                f"ToProto() and FromProto() between the struct {'::'.join(self.path)} "
                f"({self.stem}{STRUCT_HEADER_SUFFIX}) and the protobuf message {self._top_full_name()} "
                f"({self.stem}.pb.h). Converting into an object that already has the shape of the value (the same "
                "sizes, the same oneof alternatives, the same map keys) does not allocate; the README lists what does. "
                "FromProto() fails only on a value the struct cannot hold, an enum number the enum does not define, and "
                "leaves the struct partially written then."
            )
        )
        lines = header + ["", "#pragma once", ""] + includes.lines() + [""]
        lines += self._namespace_open() + [""] + body + [""] + self._namespace_close()
        return "\n".join(lines) + "\n"

    def conversions_source(self) -> str:
        includes = _Includes()
        includes.absl.add("absl/status/status.h")
        includes.project.add(RUNTIME_HEADER)
        for stem in self._imports_with_structs():
            includes.project.add(stem + CONVERSIONS_HEADER_SUFFIX)
        body: List[str] = []
        for enum in self._all_enums():
            body.extend(self._enum_conversion_lines(enum, includes))
            body.append("")
        for message, _ in self._all_messages_post_order():
            body.extend(self._to_proto_lines(message, includes))
            body.append("")
            body.extend(self._from_proto_lines(message, includes))
            body.append("")
        header = self._header_comment(
            _wrap_comment(
                f"Generated by nproto ({README}) from {self.name}. Do not edit."
            )
        )
        lines = header + ["", f'#include "{self.stem}{CONVERSIONS_HEADER_SUFFIX}"', ""]
        lines += includes.lines() + [""]
        lines += self._namespace_open() + [""] + body + self._namespace_close()
        return "\n".join(lines) + "\n"

    def _enum_conversion_lines(
        self, enum: EnumDescriptor, includes: _Includes
    ) -> List[str]:
        includes.standard.add("cstdint")
        struct = self._struct_name(enum)
        proto = self.table.types[enum.full_name].proto_class
        lines = self._signature_lines(
            f"void ToProto({struct} value, {proto}* proto)", " {"
        )
        lines.append(f"{INDENT}*proto = static_cast<{proto}>(value);")
        lines.append("}")
        lines.append("")
        lines.extend(
            self._signature_lines(
                f"absl::Status FromProto({proto} proto, {struct}* value)", " {"
            )
        )
        lines.append(f"{INDENT}switch (static_cast<std::int32_t>(proto)) {{")
        seen: Set[int] = set()
        for name, number, _ in self._enumerators(enum):
            if number in seen:
                continue  # an alias (allow_alias): the first name of the number wins
            seen.add(number)
            lines.append(f"{INDENT * 2}case {number}:")
            lines.append(f"{INDENT * 3}*value = {struct}::{name};")
            lines.append(f"{INDENT * 3}return absl::OkStatus();")
        lines.append(f"{INDENT * 2}default:")
        error = f'::nproto::internal::UnknownEnumValueError(static_cast<std::int32_t>(proto), "{enum.full_name}")'
        statement = f"{INDENT * 3}return {error};"
        if len(statement) > COLUMN_LIMIT:
            raise GenerationError(
                f"{self.name}: the name of enum {enum.full_name} is too long for nproto's code."
            )
        lines.append(statement)
        lines.append(f"{INDENT}}}")
        lines.append("}")
        return lines

    def _proto_enum(self, field: FieldDescriptor) -> str:
        return self._type_info(
            field.enum_type.full_name, f"field {field.full_name}"
        ).proto_class

    def _to_proto_lines(self, message: Descriptor, includes: _Includes) -> List[str]:
        struct = self._struct_name(message)
        proto_class = self.table.types[message.full_name].proto_class
        fields = self._fields(message)
        parameter = "value" if fields else "/*value*/"
        proto_parameter = "proto" if fields else "/*proto*/"
        signature = f"void ToProto(const {struct}& {parameter}, {proto_class}* {proto_parameter})"
        if not fields:
            return self._signature_lines(signature, " {}")
        lines = self._signature_lines(signature, " {")
        emitted: Set[str] = set()
        for field in fields:
            if field.oneof:
                if field.oneof in emitted:
                    continue
                emitted.add(field.oneof)
                lines.extend(self._oneof_to_proto_lines(message, fields, field.oneof))
                continue
            lines.extend(self._field_to_proto_lines(field))
        lines.append("}")
        return lines

    def _field_to_proto_lines(self, field: _Field) -> List[str]:
        member = f"value.{field.name}"
        accessor = field.accessor
        kind = field.kind
        i1, i2 = INDENT, INDENT * 2
        if field.optional:
            if kind == _KIND_MESSAGE:
                assign = f"ToProto(*{member}, proto->mutable_{accessor}());"
            elif kind == _KIND_ENUM:
                assign = f"proto->set_{accessor}(static_cast<{self._proto_enum(field.descriptor)}>(*{member}));"
            else:
                assign = f"proto->set_{accessor}(*{member});"
            return [
                f"{i1}if ({member}.has_value()) {{",
                f"{i2}{assign}",
                f"{i1}}} else {{",
                f"{i2}proto->clear_{accessor}();",
                f"{i1}}}",
            ]
        if kind in (_KIND_SCALAR, _KIND_STRING):
            return [f"{i1}proto->set_{accessor}({member});"]
        if kind == _KIND_ENUM:
            return [
                f"{i1}proto->set_{accessor}(static_cast<{self._proto_enum(field.descriptor)}>({member}));"
            ]
        if kind == _KIND_MESSAGE:
            return [f"{i1}ToProto({member}, proto->mutable_{accessor}());"]
        helper = {
            _KIND_EIGEN: "VectorToProto",
            _KIND_SCALARS: "ScalarsToProto",
            _KIND_STRINGS: "StringsToProto",
            _KIND_ENUMS: "EnumsToProto",
            _KIND_MESSAGES: "MessagesToProto",
            _KIND_MAP: "MapToProto",
        }[kind]
        return [
            f"{i1}::nproto::internal::{helper}({member}, proto->mutable_{accessor}());"
        ]

    def _oneof_to_proto_lines(
        self, message: Descriptor, fields: List[_Field], oneof: str
    ) -> List[str]:
        struct = self._struct_name(message)
        i1, i2, i3 = INDENT, INDENT * 2, INDENT * 3
        lines = [f"{i1}switch (value.{oneof}.index()) {{"]
        for field in fields:
            if field.oneof != oneof:
                continue
            constant = f"{struct}::{self._index_constant(field)}"
            alternative = f"std::get<{constant}>(value.{oneof})"
            if field.kind == _KIND_MESSAGE:
                statement = (
                    f"ToProto({alternative}, proto->mutable_{field.accessor}());"
                )
            elif field.kind == _KIND_ENUM:
                statement = f"proto->set_{field.accessor}(static_cast<{self._proto_enum(field.descriptor)}>({alternative}));"
            else:
                statement = f"proto->set_{field.accessor}({alternative});"
            lines.append(f"{i2}case {constant}:")
            lines.append(f"{i3}{statement}")
            lines.append(f"{i3}break;")
        lines.append(f"{i2}default:")
        lines.append(f"{i3}proto->clear_{oneof}();")
        lines.append(f"{i3}break;")
        lines.append(f"{i1}}}")
        return lines

    @staticmethod
    def _can_fail(field: _Field) -> bool:
        """Whether converting the field from protobuf can fail: only an enum number can, so enums and messages."""
        # MapFromProto returns a status for every map, also one of scalars, which cannot fail.
        return field.kind in (
            _KIND_ENUM,
            _KIND_MESSAGE,
            _KIND_ENUMS,
            _KIND_MESSAGES,
            _KIND_MAP,
        )

    def _from_proto_lines(self, message: Descriptor, includes: _Includes) -> List[str]:
        struct = self._struct_name(message)
        proto_class = self.table.types[message.full_name].proto_class
        fields = self._fields(message)
        parameter = "proto" if fields else "/*proto*/"
        value_parameter = "value" if fields else "/*value*/"
        lines = self._signature_lines(
            f"absl::Status FromProto(const {proto_class}& {parameter}, {struct}* {value_parameter})",
            " {",
        )
        if any(self._can_fail(field) for field in fields):
            lines.append(f"{INDENT}absl::Status status;")
        emitted: Set[str] = set()
        for field in fields:
            if field.oneof:
                if field.oneof in emitted:
                    continue
                emitted.add(field.oneof)
                lines.extend(self._oneof_from_proto_lines(message, fields, field.oneof))
                continue
            lines.extend(self._field_from_proto_lines(field))
        lines.append(f"{INDENT}return absl::OkStatus();")
        lines.append("}")
        return lines

    def _check_status(self, name: str, indent: str) -> List[str]:
        return [
            f"{indent}if (!status.ok()) {{",
            f'{indent}{INDENT}return ::nproto::internal::AnnotateError("{name}", status);',
            f"{indent}}}",
        ]

    def _field_from_proto_lines(self, field: _Field) -> List[str]:
        member = f"value->{field.name}"
        getter = f"proto.{field.accessor}()"
        kind = field.kind
        i1 = INDENT
        if field.optional:
            presence = f"proto.has_{field.accessor}()"
            if kind in (_KIND_ENUM, _KIND_MESSAGE):
                return [
                    f"{i1}status = ::nproto::internal::ConvertOptionalFromProto({presence}, {getter}, &{member});"
                ] + self._check_status(field.name, i1)
            return [
                f"{i1}::nproto::internal::CopyOptionalFromProto({presence}, {getter}, &{member});"
            ]
        if kind == _KIND_SCALAR:
            return [f"{i1}{member} = {getter};"]
        if kind == _KIND_STRING:
            return [f"{i1}{member}.assign({getter});"]
        if kind in (_KIND_ENUM, _KIND_MESSAGE):
            return [
                f"{i1}status = FromProto({getter}, &{member});"
            ] + self._check_status(field.name, i1)
        if kind == _KIND_ENUMS:
            proto_enum = self._proto_enum(field.descriptor)
            return [
                f"{i1}status = ::nproto::internal::EnumsFromProto<{proto_enum}>({getter}, &{member});"
            ] + self._check_status(field.name, i1)
        helper = {
            _KIND_EIGEN: "VectorFromProto",
            _KIND_SCALARS: "ScalarsFromProto",
            _KIND_STRINGS: "StringsFromProto",
            _KIND_MESSAGES: "MessagesFromProto",
            _KIND_MAP: "MapFromProto",
        }[kind]
        call = f"::nproto::internal::{helper}({getter}, &{member});"
        if self._can_fail(field):
            return [f"{i1}status = {call}"] + self._check_status(field.name, i1)
        return [f"{i1}{call}"]

    def _oneof_from_proto_lines(
        self, message: Descriptor, fields: List[_Field], oneof: str
    ) -> List[str]:
        struct = self._struct_name(message)
        i1, i2 = INDENT, INDENT * 2
        lines: List[str] = []
        members = [field for field in fields if field.oneof == oneof]
        for number, field in enumerate(members):
            keyword = "if" if number == 0 else "} else if"
            lines.append(f"{i1}{keyword} (proto.has_{field.accessor}()) {{")
            constant = f"{struct}::{self._index_constant(field)}"
            getter = f"proto.{field.accessor}()"
            if field.kind in (_KIND_ENUM, _KIND_MESSAGE):
                lines.append(
                    f"{i2}status = ::nproto::internal::ConvertAlternativeFromProto<{constant}>({getter}, &value->{oneof});"
                )
                lines.extend(self._check_status(field.name, i2))
            else:
                lines.append(
                    f"{i2}::nproto::internal::CopyAlternativeFromProto<{constant}>({getter}, &value->{oneof});"
                )
        lines.append(f"{i1}}} else {{")
        lines.append(f"{i2}value->{oneof}.emplace<std::monostate>();")
        lines.append(f"{i1}}}")
        return lines

    # ----------------------------------------------------------------------------------------------------------------

    def generate(self) -> List[plugin_pb2.CodeGeneratorResponse.File]:
        self.check_recursion()
        for message in self._messages():
            self._check_name(message.name, f"message {message.full_name}")
            nested_names = {nested.name for nested in message.nested_types} | {
                enum.name for enum in message.enum_types
            }
            names: Dict[str, str] = {}
            for enum in message.enum_types:
                self._check_name(enum.name, f"enum {enum.full_name}")
            for field in self._fields(message):
                member = field.oneof or field.name
                if member in nested_names:
                    raise GenerationError(
                        f"{self.name}: {message.full_name} has a nested type and a "
                        f"{'oneof' if field.oneof else 'field'} both named {member}, which one struct cannot hold."
                    )
                if field.oneof:
                    self._check_name(
                        field.oneof, f"oneof {message.full_name}.{field.oneof}"
                    )
                    constant = self._index_constant(field)
                    if constant in names and names[constant] != field.name:
                        raise GenerationError(
                            f"{self.name}: fields {names[constant]} and {field.name} of {message.full_name} both "
                            f"name the oneof index constant {constant}; rename one of them."
                        )
                    names[constant] = field.name
        files = []
        for suffix, content in (
            (STRUCT_HEADER_SUFFIX, self.struct_header()),
            (CONVERSIONS_HEADER_SUFFIX, self.conversions_header()),
            (CONVERSIONS_SOURCE_SUFFIX, self.conversions_source()),
        ):
            files.append(
                plugin_pb2.CodeGeneratorResponse.File(
                    name=self.stem + suffix, content=content
                )
            )
        return files


def _empty_files(
    file_proto: descriptor_pb2.FileDescriptorProto,
) -> List[plugin_pb2.CodeGeneratorResponse.File]:
    """The three files of a .proto file that defines no message or enum (only options, extensions or services)."""
    stem = output_stem(file_proto.name)
    note = f"// Generated by nproto ({README}) from {file_proto.name}, which defines no message or enum.\n"
    return [
        plugin_pb2.CodeGeneratorResponse.File(
            name=stem + STRUCT_HEADER_SUFFIX, content=note + "\n#pragma once\n"
        ),
        plugin_pb2.CodeGeneratorResponse.File(
            name=stem + CONVERSIONS_HEADER_SUFFIX, content=note + "\n#pragma once\n"
        ),
        plugin_pb2.CodeGeneratorResponse.File(
            name=stem + CONVERSIONS_SOURCE_SUFFIX, content=note
        ),
    ]


def generate_files(
    request: plugin_pb2.CodeGeneratorRequest,
) -> List[plugin_pb2.CodeGeneratorResponse.File]:
    """The generated files of every file of `request.file_to_generate`; raises GenerationError."""
    pool = descriptor_pool.DescriptorPool()
    for file_proto in request.proto_file:
        try:
            pool.Add(file_proto)
        except (TypeError, ValueError) as error:
            # protoc only sends files it has checked; a request built by hand may hold one the descriptor API rejects.
            raise GenerationError(
                f"{file_proto.name}: not a valid .proto file: {error}"
            ) from error
    table = TypeTable(request.proto_file)
    by_name = {file_proto.name: file_proto for file_proto in request.proto_file}
    files: List[plugin_pb2.CodeGeneratorResponse.File] = []
    for name in request.file_to_generate:
        file_proto = by_name[name]
        if not file_proto.message_type and not file_proto.enum_type:
            files.extend(_empty_files(file_proto))
            continue
        generator = _FileGenerator(pool.FindFileByName(name), file_proto, table)
        files.extend(generator.generate())
    return files


def generate(
    request: plugin_pb2.CodeGeneratorRequest,
) -> plugin_pb2.CodeGeneratorResponse:
    """The CodeGeneratorResponse protoc expects; a file nproto cannot map is the response's error."""
    response = plugin_pb2.CodeGeneratorResponse()
    response.supported_features = (
        plugin_pb2.CodeGeneratorResponse.FEATURE_PROTO3_OPTIONAL
        | plugin_pb2.CodeGeneratorResponse.FEATURE_SUPPORTS_EDITIONS
    )
    response.minimum_edition = MINIMUM_EDITION
    response.maximum_edition = MAXIMUM_EDITION
    try:
        response.file.extend(generate_files(request))
    except GenerationError as error:
        response.error = str(error)
    return response
