"""nproto_cc_library: plain C++ structs generated from proto_library targets (tools/nproto/README.md).

    load("//tools/nproto:nproto.bzl", "nproto_cc_library")

    nproto_cc_library(
        name = "messages_nproto",
        deps = [":messages_proto"],
    )

An aspect walks the proto_library graph below `deps`. For every proto_library it runs the nproto plugin
(:protoc-gen-nproto) over the library's own .proto files once, and compiles the generated code against the C++
protobuf code of the same library (protobuf's cc_proto_aspect), the nproto code of the libraries it depends on,
Eigen, Abseil and :nproto_runtime. Each proto's struct is therefore generated and compiled once however many
nproto_cc_library targets reach it, and proto dependencies across packages and repositories work as they do for
cc_proto_library. The protos of protobuf itself (the well-known types, descriptor.proto) and nproto's own
options.proto get no structs (the :nproto_toolchain's denylisted_protos).

The rule provides a CcInfo, so it is a dependency of cc_library, cc_binary and cc_test like any cc_library. Its
headers follow the proto import path: "humanoid_mpc_msgs/mpc_policy.nproto.h" for the struct and
"humanoid_mpc_msgs/mpc_policy.nproto.pb.h" for the conversions. The output group `nproto_files` holds every generated
file of the graph, for tests that inspect the generated code.
"""

load("@protobuf//bazel/common:proto_common.bzl", "proto_common")
load("@protobuf//bazel/common:proto_info.bzl", "ProtoInfo")
load("@protobuf//bazel/common:proto_lang_toolchain_info.bzl", "ProtoLangToolchainInfo")

# protobuf's cc_proto_aspect, at the path the Bazel autoloader loads it from, which protobuf keeps stable.
load("@protobuf//bazel/private:bazel_cc_proto_library.bzl", "cc_proto_aspect")  # buildifier: disable=bzl-visibility
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//bazel:copts.bzl", "FIRST_PARTY_COPTS")

NprotoInfo = provider(
    doc = "The nproto code of a proto_library and of every proto_library it depends on.",
    fields = {
        "cc_info": "CcInfo: the compiled nproto code of the library and of its dependencies, with their protobuf code.",
        "direct_files": "depset[File]: the files generated for the library's own .proto files.",
        "files": "depset[File]: the files generated for the library and every library it depends on.",
    },
)

# One header with the struct, one header and one source with the conversions, per .proto file: the files the plugin
# writes.
# LINT.IfChange(generated_suffixes)
STRUCT_HEADER_SUFFIX = ".nproto.h"
CONVERSIONS_HEADER_SUFFIX = ".nproto.pb.h"
CONVERSIONS_SOURCE_SUFFIX = ".nproto.pb.cc"
# LINT.ThenChange(//tools/nproto/nproto_generator.py:generated_suffixes)

# The generated code compiles with the flags of every first-party target, and -Werror keeps it warning-free under them.
# The Abseil and protobuf headers it includes are system headers (.bazelrc's external_include_paths), so they warn about
# nothing; the struct headers are held to the same flags on their own (//tools/nproto/test:test_struct_headers).
_COPTS = FIRST_PARTY_COPTS + [
    "-Werror",
]

def _strip_include_prefix(ctx, proto_info):
    """The include prefix that makes the generated headers' include paths their protos' import paths.

    The same computation as protobuf's cc_proto_library: a proto_library with strip_import_prefix or import_prefix
    has its .proto files under _virtual_imports/<name>/, and the generated headers next to them.
    """
    proto_root = proto_info.proto_source_root
    if proto_root == "." or proto_root == ctx.label.workspace_root:
        return ""
    if proto_root.startswith(ctx.bin_dir.path):
        proto_root = proto_root[len(ctx.bin_dir.path) + 1:]
    elif proto_root.startswith(ctx.genfiles_dir.path):
        proto_root = proto_root[len(ctx.genfiles_dir.path) + 1:]
    if proto_root.startswith(ctx.label.workspace_root):
        proto_root = proto_root[len(ctx.label.workspace_root):]
    return "//" + proto_root

def _nproto_aspect_impl(target, ctx):
    proto_info = target[ProtoInfo]
    toolchain = ctx.attr._nproto_toolchain[ProtoLangToolchainInfo]
    deps = [dep[NprotoInfo] for dep in getattr(ctx.rule.attr, "deps", []) if NprotoInfo in dep]
    dep_cc_infos = [dep.cc_info for dep in deps]
    transitive_files = [dep.files for dep in deps]

    if not proto_common.experimental_should_generate_code(proto_info, toolchain, "nproto_cc_library", target.label):
        # A library of protobuf's own protos or of options.proto: nothing to generate, only the dependencies' code.
        return [NprotoInfo(
            cc_info = cc_common.merge_cc_infos(cc_infos = dep_cc_infos),
            direct_files = depset(),
            files = depset(transitive = transitive_files),
        )]

    headers = []
    for suffix in [STRUCT_HEADER_SUFFIX, CONVERSIONS_HEADER_SUFFIX]:
        headers.extend(proto_common.declare_generated_files(ctx.actions, proto_info, suffix))
    sources = proto_common.declare_generated_files(ctx.actions, proto_info, CONVERSIONS_SOURCE_SUFFIX)
    proto_common.compile(
        actions = ctx.actions,
        proto_info = proto_info,
        proto_lang_toolchain_info = toolchain,
        generated_files = headers + sources,
        experimental_output_files = "multiple",
    )

    # The library's own protobuf C++ code (cc_proto_aspect, which this aspect requires), the runtime and the
    # dependencies' nproto code.
    compilation_cc_infos = [target[CcInfo], toolchain.runtime[CcInfo]] + dep_cc_infos
    cc_toolchain = find_cc_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features + ["parse_headers", "layering_check", "header_modules"],
    )
    name = ctx.label.name + "_nproto"
    compilation_context, compilation_outputs = cc_common.compile(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = sources,
        public_hdrs = headers,
        compilation_contexts = [info.compilation_context for info in compilation_cc_infos],
        name = name,
        strip_include_prefix = _strip_include_prefix(ctx, proto_info),
        user_compile_flags = _COPTS,
    )
    linking_context, _ = cc_common.create_linking_context_from_compilation_outputs(
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        compilation_outputs = compilation_outputs,
        linking_contexts = [info.linking_context for info in compilation_cc_infos],
        name = name,
        disallow_dynamic_library = not cc_common.is_enabled(
            feature_name = "supports_dynamic_linker",
            feature_configuration = feature_configuration,
        ),
    )
    direct_files = depset(headers + sources)
    return [NprotoInfo(
        cc_info = CcInfo(compilation_context = compilation_context, linking_context = linking_context),
        direct_files = direct_files,
        files = depset(transitive = [direct_files] + transitive_files),
    )]

nproto_aspect = aspect(
    implementation = _nproto_aspect_impl,
    attr_aspects = ["deps"],
    # The protobuf C++ code of each library, which its nproto conversions compile against: target[CcInfo].
    requires = [cc_proto_aspect],
    required_providers = [ProtoInfo],
    provides = [NprotoInfo],
    fragments = ["cpp"],
    attrs = {
        "_nproto_toolchain": attr.label(
            default = Label("//tools/nproto:nproto_toolchain"),
            providers = [ProtoLangToolchainInfo],
        ),
    },
    toolchains = use_cc_toolchain(),
)

def _nproto_cc_library_impl(ctx):
    infos = [dep[NprotoInfo] for dep in ctx.attr.deps]
    return [
        cc_common.merge_cc_infos(direct_cc_infos = [info.cc_info for info in infos]),
        DefaultInfo(files = depset(transitive = [info.direct_files for info in infos])),
        OutputGroupInfo(nproto_files = depset(transitive = [info.files for info in infos])),
    ]

nproto_cc_library = rule(
    implementation = _nproto_cc_library_impl,
    doc = """The nproto structs and conversions of the given proto_library targets and of everything they import.

A dependency of cc_library, cc_binary and cc_test: #include "<proto import path stem>.nproto.h" for a struct and
"<stem>.nproto.pb.h" for its ToProto() / FromProto(). See tools/nproto/README.md.
""",
    attrs = {
        "deps": attr.label_list(
            aspects = [nproto_aspect],
            providers = [ProtoInfo],
            allow_files = False,
            doc = "The proto_library targets whose .proto files, and imports, get structs.",
        ),
    },
    provides = [CcInfo],
)
