# nproto

nproto generates a **plain C++ struct** for every protobuf message of the repository, with the conversions between the
struct and the protobuf message, and parses the repository's **textproto configuration files** strictly. The robot's
realtime code, the MPC and the tools then work with Eigen vectors, `std::vector`s and `enum class`es, and protobuf
stays at the edge: the bus serializes the protobuf message, and the code around it fills or reads the struct.

```text
  humanoid_mpc_msgs/mpc_policy.proto
        |  protoc + protoc-gen-nproto (nproto_cc_library)
        v
  mpc_policy.nproto.h      struct ocs2::humanoid::msgs::MpcPolicy: Eigen and std types, no protobuf include
  mpc_policy.nproto.pb.h   ToProto(const MpcPolicy&, humanoid_mpc_msgs::MpcPolicy*)
  mpc_policy.nproto.pb.cc  FromProto(const humanoid_mpc_msgs::MpcPolicy&, MpcPolicy*) -> absl::Status
```

The generated conversions are realtime-safe: converting into a struct or a message that already has the value's
shape allocates nothing (see [What allocates](#what-allocates)).

## Naming the struct

Every `.proto` file holds one top-level message or enum (`tools/hooks/proto_file_layout.py`). The **first statement**
of its body names the C++ type nproto generates for it, and the file imports `nproto/options.proto`
(`:options_proto`, which the `proto_library` depends on):

```proto
import "nproto/options.proto";

// Next ID: 4
message Vector3 {
  option (nproto.generate_struct) = "ocs2::humanoid::msgs::Vector3";

  reserved 4 to max;

  double x = 1;
  double y = 2;
  double z = 3;
}

enum ControllerType {
  option (nproto.generate_enum) = "ocs2::humanoid::msgs::ControllerType";

  CONTROLLER_TYPE_UNKNOWN = 0;
  ...
}
```

- `nproto.generate_struct` is a `MessageOptions` extension (number 52001), `nproto.generate_enum` an `EnumOptions`
  extension (52002); a file-level option does not exist.
- The value is a fully qualified C++ name. Everything before the last `::` is the namespace of the generated code; the
  last component must be the definition's own name.
- The struct needs a namespace of its own: a name equal to protobuf's own C++ class (`humanoid_mpc_msgs::Vector3` for
  package `humanoid_mpc_msgs`) is refused, as it would collide.
- **Nested types carry no option.** A message's nested messages and enums become nested C++ types of its struct:
  `humanoid_mpc_msgs.TargetContactPatch.Kind` is `ocs2::humanoid::msgs::TargetContactPatch::Kind`.

## Using it from Bazel

```starlark
load("//tools/nproto:nproto.bzl", "nproto_cc_library")

proto_library(
    name = "messages_proto",
    srcs = glob(["*.proto"]),
    strip_import_prefix = "/humanoid_nmpc",
    deps = ["//tools/nproto:options_proto"],
)

nproto_cc_library(
    name = "messages_nproto",
    deps = [":messages_proto"],
)

cc_library(
    name = "robot_link",
    srcs = ["RobotLink.cpp"],
    deps = [":messages_nproto"],
)
```

`nproto_cc_library(name, deps)` takes `proto_library` targets. An aspect walks the `proto_library` graph below them,
runs the plugin once per `proto_library` over its own `.proto` files and compiles the generated code against that
library's `cc_proto_library` code, the nproto code of the libraries it imports, Eigen, Abseil and `:nproto_runtime`.
Each proto's struct is therefore generated and compiled once however many `nproto_cc_library` targets reach it, and
imports across Bazel packages and proto packages work as they do for `cc_proto_library`. The rule provides a
`CcInfo`, so it is a dependency of any `cc_library`, `cc_binary` or `cc_test`. Its output group `nproto_files` holds
every generated file of the graph, for tests that inspect the generated code.

The include paths follow the proto import paths:

| Include | What it declares | Depends on |
|---|---|---|
| `"humanoid_mpc_msgs/mpc_policy.nproto.h"` | the struct, its nested types and `operator==` / `operator!=` | Eigen and the standard library only |
| `"humanoid_mpc_msgs/mpc_policy.nproto.pb.h"` | `ToProto()` / `FromProto()` of the struct and of every nested type | the protobuf code, Abseil |

The protos of protobuf itself (descriptor.proto, the well-known types) and `options.proto` get no structs (the
`denylisted_protos` of `:nproto_toolchain`), so a message cannot hold a `google.protobuf.Timestamp`; the plugin says
so.

The generated code is clang-format clean, uses no `auto`, and compiles warning-free under `-Wall -Wextra -Werror`;
the struct headers also under `-Wpedantic` (`//tools/nproto/test:test_struct_headers`, `:test_generated_code`).

## The type mapping

| Proto field | Struct member | Default member initializer |
|---|---|---|
| `double`, `float` | `double`, `float` | `0.0`, `0.0f` (or the explicit default) |
| `int32`, `sint32`, `sfixed32` | `std::int32_t` | `0` |
| `int64`, `sint64`, `sfixed64` | `std::int64_t` | `0` |
| `uint32`, `fixed32` | `std::uint32_t` | `0` |
| `uint64`, `fixed64` | `std::uint64_t` | `0` |
| `bool` | `bool` | `false` |
| `string`, `bytes` | `std::string` | empty |
| an enum | its `enum class` (nested enums: a nested `enum class`) | the enum's first value |
| a message | its struct, **by value** | the default struct |
| `optional` scalar, string, enum or message (explicit presence) | `std::optional<T>` | `std::nullopt` |
| `repeated double` / `repeated float` | `Eigen::VectorXd` / `Eigen::VectorXf` | size 0 |
| any other `repeated` scalar, enum or string | `std::vector<T>` (`repeated bool`: `std::vector<bool>`) | empty |
| `repeated` message | `std::vector<Struct>` | empty |
| `map<K, V>` | `std::map<K, V or its struct>`, in key order | empty |
| `oneof o { A a = 1; B b = 2; }` | `std::variant<std::monostate, A, B> o;` and `kAIndex = 1`, `kBIndex = 2` | `std::monostate` (none set) |

Choices behind the table:

- **Member names are the proto field names**, snake_case as in the `.proto` file: Google style names struct data
  members in snake_case without a trailing underscore, and `policy.time_trajectory` reads like
  `policy.time_trajectory()`. A field named like a C++ keyword is refused.
- **Enumerators** are Google-style constants without the enum's prefix: `CONTROLLER_TYPE_LINEAR` is
  `ControllerType::kLinear`, `KIND_SWING_IN_FLIGHT` is `TargetContactPatch::Kind::kSwingInFlight`. The prefix is the
  enum's name in upper snake case and is dropped only when every value has it. The underlying type is `std::int32_t`,
  as for protobuf enums. An alias (`allow_alias`) is a second enumerator of the same number.
- **Presence.** A field with explicit presence and no explicit default becomes `std::optional`: proto3 `optional`,
  proto2 `optional` without `[default = ...]`, and edition 2023 fields that are not `IMPLICIT`. A message field without
  `optional` is held by value and keeps no presence: `FromProto()` of an absent submessage gives the default struct,
  and `ToProto()` always sets the submessage. Write `optional Foo foo = 1;` where "absent" must be told apart from
  "default". A proto2 `required` field is a plain member.
- **Oneofs** are a `std::variant` named after the oneof, whose index 0 (`std::monostate`) means that no alternative is
  set. Alternatives are told apart by index, because two of them may share a type (`double radius`, `double side`):
  the struct declares `static constexpr std::size_t k<Field>Index` for each, so `value.shape.index() ==
  Oneofs::kSideIndex` and `std::get<Oneofs::kSideIndex>(value.shape)`.
- **Recursive messages are refused.** Fields are held by value, so a message that contains itself, directly or through
  other messages (a repeated field included), has no struct. Break the cycle, for instance with an index into a
  repeated field.
- **Nested types come first.** A nested message referred to by an earlier sibling is declared before it.
- **Equality** compares every member exactly as `==` does for its type: `NaN != NaN`, `-0.0 == 0.0`, Eigen vectors of
  different sizes are unequal (sizes are compared first), oneofs with the same value in different alternatives are
  unequal, and presence takes part.

## The conversions

For the top-level type and every nested type, `<file>.nproto.pb.h` declares, in the struct's namespace:

```cpp
void ToProto(const Struct& value, ProtoMessage* proto);
absl::Status FromProto(const ProtoMessage& proto, Struct* value);
void ToProto(Enum value, ProtoEnum* proto);
absl::Status FromProto(ProtoEnum proto, Enum* value);
```

- `ToProto()` writes every field of the message: unset optional members clear their field, `std::monostate` clears
  the oneof, and repeated fields and maps end up exactly the value's.
- `FromProto()` overwrites every member of the struct. It fails only where the struct cannot hold the value: an enum
  number its enum does not define (open enums accept any number on the wire). The error is InvalidArgument and names
  the field path: `annotations.target_contact_patches[1].kind: 9 is not a value of
  humanoid_mpc_msgs.TargetContactPatch.Kind`; map keys appear as `[7]` or `["name"]`. On error the struct is partially
  written. Building the error message allocates, which only an error does.

```cpp
#include "humanoid_mpc_msgs/mpc_policy.nproto.pb.h"

// Kept by the communication thread, so that their buffers survive from one policy to the next.
humanoid_mpc_msgs::MpcPolicy message;
ocs2::humanoid::msgs::MpcPolicy policy;

if (message.ParseFromString(payload)) {
  const absl::Status status = FromProto(message, &policy);  // no allocation once sized
  if (!status.ok()) LOG_EVERY_N_SEC(WARNING, 1) << "dropping a policy: " << status;
}
```

## What allocates

The conversions reuse what the target object holds: Eigen and `std::vector` resize to the size they already have,
which does nothing; `std::string` assignment reuses its capacity; repeated message fields are resized in place, and
`RepeatedPtrField` keeps the elements a shorter value removed, cleared, for the next longer one; a map whose key set
is the value's is updated in place; a `std::optional` or `std::variant` that already holds the alternative is assigned
in place. No conversion takes a lock, does I/O or logs.

So `FromProto()` into a struct, and `ToProto()` into a message, that already has the value's **shape** make no heap
allocation. The shape is: the same size of every repeated field, map and vector, strings no longer than before, the
same map keys, the same oneof alternatives and the same optional members present. These allocate:

| What | Why |
|---|---|
| the first conversion into a default-constructed struct or an empty message | every buffer is created |
| a repeated field, vector or string that grows beyond its capacity | Eigen and protobuf's `RepeatedField` reallocate to grow; a vector or `RepeatedPtrField` of strings or messages also creates the new elements |
| a map whose key set changes | `std::map` and `google::protobuf::Map` allocate a node per key, so a map with other keys is rebuilt; prefer a repeated field or an Eigen vector in data the realtime loop converts |
| a oneof that switches to an alternative holding a string, a vector or a message | the variant (and the message) destroy the old alternative and create the new one |
| an optional message member or field that becomes present | the struct or the submessage is created |
| an error of `FromProto()` | the error message is built |

`ExpectConversionsIntoSizedObjectsDoNotAllocate<Struct, Proto>()` (`//tools/nproto/test:proto_allocations`) checks the
property for a message type with the malloc-counting `//robot_runtime/robot_realtime:allocation_counter`: it converts
a message of test values into a struct and back, then converts a second message of the same shape into the same
objects and counts the allocations of that second `FromProto()` and `ToProto()`. `:test_nproto_allocations` runs it
over the test protos, and `//humanoid_nmpc/humanoid_mpc_msgs:messages_nproto_allocations_test` over the messages of the
realtime loop and the MPC link, a 60-node policy included. Parsing and serializing the protobuf message itself are
protobuf's and are not covered here.

## Textproto configuration files

Every configuration file the repository reads is a `.textproto` of a message with a schema, one message per file. The
file names its schema in its first lines:

```textproto
# proto-file: humanoid_nmpc/humanoid_mpc_msgs/vector3.proto
# proto-message: humanoid_mpc_msgs.Vector3

x: 1.5
y: -0.25
```

`:nproto_runtime` (`#include "nproto/Textproto.h"`, namespace `nproto`) parses them strictly:

```cpp
absl::Status ParseTextprotoInto(absl::string_view text, absl::string_view sourceName, google::protobuf::Message* message);
template <typename Message>
absl::StatusOr<Message> ParseTextproto(absl::string_view text, absl::string_view sourceName);
template <typename Message>
absl::StatusOr<Message> ParseTextprotoFile(absl::string_view path);
template <typename Struct, typename Message>
absl::StatusOr<Struct> LoadTextprotoFile(absl::string_view path);  // ParseTextprotoFile, then FromProto
absl::StatusOr<std::string> ReadTextFile(absl::string_view path);
std::string WriteTextproto(const google::protobuf::Message& message);  // for tools that write configuration files
```

- Parsing uses `google::protobuf::TextFormat::Parser` with an error collector. A syntax error, an unknown field or
  extension, an unknown enum value name, a value of the wrong type or out of range, a non-repeated field given twice,
  two alternatives of one oneof, a field number instead of a name, a deprecated field and a missing proto2 required
  field are all errors; nothing is silently ignored. The error is InvalidArgument with one line per problem,
  `<file>:<line>:<column>: <problem>`, 1-based, at the token at fault (the parser itself reports an unknown field
  name, a bad value and a repeated field at the token after it; nproto moves the position back to it).
- A file that cannot be opened is NotFound (or the error the OS reports) and names the path. A `FromProto()` error of
  `LoadTextprotoFile()` is prefixed with the path: `config/x.textproto: kind: 7 is not a value of ...`.
- `WriteTextproto()` writes one field per line, repeated numbers on one line (`gains: [1, 2, 3]`), map entries in key
  order and UTF-8 strings unescaped; `ParseTextproto()` reads it back to the same message.

```cpp
#include "my_package/my_config.nproto.pb.h"  // message my_package.MyConfig, struct my::project::MyConfig
#include "nproto/Textproto.h"

const absl::StatusOr<my::project::MyConfig> config =
    nproto::LoadTextprotoFile<my::project::MyConfig, my_package::MyConfig>("config/my_config.textproto");
if (!config.ok()) return config.status();  // "config/my_config.textproto:3:5: Message type ... has no field ..."
```

Parsing allocates and reads files: it is for start-up and tools, never for a realtime thread.

## What the plugin refuses

The plugin (`:protoc-gen-nproto`, `protoc_gen_nproto.py` on `nproto_generator.py`) stops the build with an error that
names the file and the definition when:

- a top-level message or enum sets no option, or a file defines more than one top-level type;
- the option is not a fully qualified C++ name, puts the type in the global namespace, contains a C++ keyword, does
  not end in the definition's name, or is protobuf's own C++ name of it;
- a nested message or enum sets an option of its own;
- a message contains itself;
- a field refers to a type that has no struct (its definition sets no option, or it is one of protobuf's own types);
- a name cannot be a C++ name: a field, oneof or message named like a C++ keyword, a nested type and a member of one
  name, two enum values that give one enumerator, two oneof fields that give one index constant.

protoc passes custom options to a plugin as extension fields of `MessageOptions` / `EnumOptions`; the plugin imports
`options_pb2` (`:options_py_proto`) before it parses the request, so that they are readable as extensions.

## Files and tests

| Path | What |
|---|---|
| `options.proto` | the two options (`:options_proto`, `:options_py_proto`) |
| `nproto_generator.py`, `protoc_gen_nproto.py` | the generator and the protoc plugin (`:nproto_generator`, `:protoc-gen-nproto`) |
| `nproto.bzl` | `nproto_cc_library` and its aspect; `:nproto_toolchain` is how it runs the plugin |
| `include/nproto/Conversions.h`, `src/Conversions.cpp` | the helpers the generated code calls (`nproto::internal`, not an API) |
| `include/nproto/Textproto.h`, `src/Textproto.cpp` | the textproto helpers |
| `test/` | test protos for every mapping (one message or enum per file; `test/other/` is another proto and Bazel package) and the tests |

```bash
bazel test //tools/nproto/... //humanoid_nmpc/humanoid_mpc_msgs:messages_nproto_test \
    //humanoid_nmpc/humanoid_mpc_msgs:messages_nproto_allocations_test
```

- `:test_nproto`: the member type of every kind of field, defaults (proto3, proto2 and edition 2023), equality,
  round trips of every test message both ways, every oneof alternative, presence, maps, the errors of `FromProto()`.
- `:test_nproto_allocations`: allocation-freedom of the conversions into objects of the value's shape.
- `:test_struct_headers`: the struct headers alone compile under `-Wpedantic -Werror` without protobuf or Abseil.
- `:test_textproto`: a good file, unknown fields, type errors and other mistakes with their line and column, a missing
  file, a conversion error, and `WriteTextproto()` round trips.
- `:test_nproto_generator`: the generator's output and every error above, on requests built in the test, and the
  plugin binary speaking protoc's protocol.
- `:test_generated_code`: the generated code of the test protos and of `humanoid_mpc_msgs` is clang-format clean,
  uses no `auto`, and its struct headers include neither protobuf nor Abseil.
- `ExpectRoundTrips<Struct, Proto>()` (`:proto_test_values`) is the round-trip check any package can run over its own
  messages, as `//humanoid_nmpc/humanoid_mpc_msgs:messages_nproto_test` does for every message there.
