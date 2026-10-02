"""clang-tidy as a Bazel aspect (tools/clang_tidy/README.md).

    bazel build --config=clang-tidy //humanoid_nmpc/...        # .clang-tidy, the enforced checks (make lint-tidy)
    bazel build --config=clang-tidy-sweep //humanoid_nmpc/...  # sweep.clang-tidy, every candidate (make lint-tidy-sweep)

clang-tidy parses C++ as a compiler does, so it runs only as Bazel actions: inside .bazelrc's RAM-bounded --jobs and
tools/bazel's machine lock, never by hand (AGENTS.md, "Builds share one machine's memory").

The aspect does not propagate: `//...` already names every first-party target, and no external repository is visited.
For each cc_library, cc_binary and cc_test outside EXCLUDED_PACKAGES it runs clang-tidy once per source file: each .cpp
as a translation unit, and each header as a main file of its own (`-x c++-header`), which also proves that the header
is self-contained. Generated files are never linted, but they are inputs, so that includes resolve.

Each action writes its findings (output group `clang_tidy`) and its fix-its (`clang_tidy_fixes`) and succeeds either
way, so that Bazel caches it and a re-run lints only what changed. tools/clang_tidy/clang_tidy_report.py turns the
reports of a build into the result, and tools/clang_tidy/clang_tidy_apply.py applies the fix-its.

Header diagnostics. clang-tidy files a diagnostic under the file it is located in, and some checks locate a diagnostic in
a header from a .cpp's context: readability-identifier-naming at a declaration (with the fix-its of every use),
modernize-use-default-member-init at a field whose constructor is in the .cpp, and
readability-inconsistent-declaration-parameter-name at the other declaration. So:

- every first-party include directory is passed as -I, never -isystem: clang-tidy drops a system header's diagnostics
  before any filter. rules_cc passes `includes = [...]` as -I today, but as -isystem once its system_include_paths
  feature is on; the aspect keeps first-party headers visible either way. External and vendored directories keep what
  the build gives them;
- a .cpp action shows the diagnostics of every first-party header (FIRST_PARTY_HEADER_FILTER) except the generated
  ones (GENERATED_FILES); a header action shows only its own (`^$`).

A finding in a header is therefore reported by the header's action and by every .cpp that includes it; the collector
and the fix applier deduplicate.
"""

load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:find_cc_toolchain.bzl", "find_cc_toolchain", "use_cc_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

# Packages the aspect never lints. Vendored code is left as it comes, and its include directories stay -isystem; the
# fixtures are deliberately bad code, linted only by the self-test (scope = "selftest").
# LINT.IfChange(excluded_packages)
VENDORED_PACKAGES = [
    "lib/",
    "tools/ifttt-lint/",
]
FIXTURE_PACKAGES = ["tools/clang_tidy/testdata/"]
# LINT.ThenChange(//tools/hooks/lint_files.py:vendored_dirs, //tools/hooks/lint_files.py:fixture_dirs)
EXCLUDED_PACKAGES = VENDORED_PACKAGES + FIXTURE_PACKAGES

# Generated sources checked into the tree. Their generator is held to the rules instead.
# LINT.IfChange(generated_files)
GENERATED_FILES = [
    "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/acom/AcomSirenWeightsAtlas.h",
    "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/acom/AcomSirenWeightsG1.h",
    "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/acom/AcomSirenWeightsSa01.h",
]
# LINT.ThenChange(//tools/hooks/lint_files.py:generated_files)

# The first-party headers whose diagnostics a .cpp action shows: every top-level directory that holds first-party C++
# (tools/clang_tidy/test_clang_tidy_config.py checks that against tools/hooks/lint_files.py:cpp_top_level_dirs). It is
# anchored, so that generated headers under bazel-out/, external repositories and lib/ never match. A header found
# through `-iquote .` is named `./<path>`.
# LINT.IfChange(header_filter)
FIRST_PARTY_HEADER_FILTER = "^(\\./)?(bazel|humanoid_nmpc|humanoid_state_estimation|robot_models|robot_runtime|tools)/"
# LINT.ThenChange(//tools/clang_tidy/test_clang_tidy_config.py:header_filter)
OWN_FILE_ONLY_HEADER_FILTER = "^$"

# The compiler warnings that .clang-tidy and sweep.clang-tidy enable as `clang-diagnostic-<name>` checks. Every GCC
# warning flag of the build is dropped (_DROPPED_FLAG_PREFIXES), so each such check needs its flag here.
# LINT.IfChange(clang_diagnostics)
CLANG_DIAGNOSTIC_FLAGS = [
    "-Wctad-maybe-unsupported",
    "-Wnonnull",
    "-Wnullability",
]
# LINT.ThenChange(//.clang-tidy:checks, //tools/clang_tidy/sweep.clang-tidy:checks)

# GCC's warnings would become `clang-diagnostic-*` noise (and its -Werror would turn them into errors); the diagnostics
# come from the configuration instead. The other flags are GCC-only or need what clang has not got (omp.h).
_DROPPED_FLAG_PREFIXES = ["-W", "-pedantic"]
_DROPPED_FLAGS = [
    "-fno-canonical-system-headers",
    "-fopenmp",
    "-fstack-usage",
    "-pass-exit-codes",
]

_SOURCE_EXTENSIONS = ["c", "cc", "cpp", "cxx"]
_HEADER_EXTENSIONS = ["h", "hh", "hpp", "hxx"]
_RULE_KINDS = ["cc_binary", "cc_library", "cc_test"]

ClangTidyInfo = provider(
    doc = "The clang-tidy actions of one target: the self-test and the argument test read them.",
    fields = {
        "files": "list of struct(source, report, fixes, arguments): one per linted file, `arguments` being the file " +
                 "the action's clang-tidy arguments are written to (output group clang_tidy_args).",
    },
)

def _is_under(path, prefixes):
    for prefix in prefixes:
        if path.startswith(prefix):
            return True
    return False

def _source_tree_path(path):
    """`path` without a `bazel-out/<configuration>/bin/` prefix: where a generated directory's sources would be."""
    if path.startswith("bazel-out/"):
        parts = path.split("/", 3)
        if len(parts) == 4 and parts[2] == "bin":
            return parts[3]
    return path

def _is_first_party_directory(path):
    """True for an include directory of this repository's own code (not external, not vendored)."""
    path = _source_tree_path(path)
    return not (path.startswith("external/") or _is_under(path, VENDORED_PACKAGES))

def _excluded(package, scope):
    if scope == "selftest":
        return _is_under(package + "/", VENDORED_PACKAGES)
    return _is_under(package + "/", EXCLUDED_PACKAGES)

def _gcc_install_dir(cc_toolchain):
    """GCC's /usr/lib/gcc/<triple>/<version>, so that clang picks the same libstdc++ as the build's GCC."""
    for directory in cc_toolchain.built_in_include_directories:
        parts = directory.split("/")
        if len(parts) >= 3 and parts[-1] == "include" and parts[-4:-3] == ["gcc"]:
            return "/".join(parts[:-1])
    return None

def _kept_flags(flags):
    kept = []
    for flag in flags:
        if flag in _DROPPED_FLAGS or _is_under(flag, _DROPPED_FLAG_PREFIXES):
            continue
        kept.append(flag)
    return kept

def _compiler_flags(ctx, target, cc_toolchain):
    """The compile command line of the target's C++ files, as clang-tidy should see it (without the source file)."""
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    # The command-line copts and cxxopts carry -std=c++20 (.bazelrc); without them the code would parse as the
    # toolchain's C++17.
    user_compile_flags = ctx.fragments.cpp.copts + ctx.fragments.cpp.cxxopts + getattr(ctx.rule.attr, "copts", [])
    variables = cc_common.create_compile_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        user_compile_flags = user_compile_flags,
    )
    flags = _kept_flags(cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = ACTION_NAMES.cpp_compile,
        variables = variables,
    ))

    # Not the toolchain's built-in include directories: they would put GCC's own stddef.h and intrinsics headers in
    # front of clang's. Clang finds libstdc++ through GCC's installation directory instead.
    gcc_install_dir = _gcc_install_dir(cc_toolchain)
    if gcc_install_dir:
        flags.append("--gcc-install-dir=" + gcc_install_dir)

    compilation_context = target[CcInfo].compilation_context
    for define in compilation_context.defines.to_list() + getattr(ctx.rule.attr, "local_defines", []):
        flags.append("-D" + define)
    for directory in compilation_context.quote_includes.to_list():
        flags += ["-iquote", directory]
    system_includes = (
        compilation_context.system_includes.to_list() +
        getattr(compilation_context, "external_includes", depset()).to_list()
    )
    for directory in compilation_context.includes.to_list():
        flags += ["-I", directory]
    for directory in system_includes:
        if _is_first_party_directory(directory):
            flags += ["-I", directory]
    for directory in system_includes:
        if not _is_first_party_directory(directory):
            flags += ["-isystem", directory]
    for directory in compilation_context.framework_includes.to_list():
        flags += ["-F", directory]
    return flags

def _generated_files_filter():
    escaped = [path.replace(".", "\\.") for path in GENERATED_FILES]
    return "^(\\./)?(" + "|".join(escaped) + ")$"

def _linted_files(ctx, scope):
    files = []
    for attribute in ("srcs", "hdrs"):
        for file in getattr(ctx.rule.files, attribute, []):
            if not file.is_source or file.extension not in _SOURCE_EXTENSIONS + _HEADER_EXTENSIONS:
                continue
            if file.path in GENERATED_FILES:
                continue
            if scope != "selftest" and _is_under(file.path, EXCLUDED_PACKAGES):
                continue
            files.append(file)
    return files

def _clang_tidy_aspect_impl(target, ctx):
    scope = ctx.attr.scope
    if ctx.rule.kind not in _RULE_KINDS or CcInfo not in target:
        return []
    if _excluded(ctx.label.package, scope) or "no-clang-tidy" in getattr(ctx.rule.attr, "tags", []):
        return []
    files = _linted_files(ctx, scope)
    if not files:
        return []

    cc_toolchain = find_cc_toolchain(ctx)
    compiler_flags = _compiler_flags(ctx, target, cc_toolchain)
    config = ctx.file._config
    inputs = depset(
        [config, ctx.file._clang_format],
        transitive = [target[CcInfo].compilation_context.headers],
    )

    entries = []
    for file in files:
        is_header = file.extension in _HEADER_EXTENSIONS
        prefix = "{}.{}/{}".format(ctx.label.name, ctx.attr._output_directory, file.short_path)
        report = ctx.actions.declare_file(prefix + ".txt")
        fixes = ctx.actions.declare_file(prefix + ".yaml")
        arguments_file = ctx.actions.declare_file(prefix + ".args")

        arguments = ctx.actions.args()
        arguments.set_param_file_format("multiline")
        arguments.add("--quiet")
        arguments.add("--allow-no-checks")
        arguments.add("--config-file=" + config.path)
        if is_header:
            arguments.add("--header-filter=" + OWN_FILE_ONLY_HEADER_FILTER)
        else:
            arguments.add("--header-filter=" + FIRST_PARTY_HEADER_FILTER)
            arguments.add("--exclude-header-filter=" + _generated_files_filter())
        arguments.add(file.path)
        arguments.add("--")
        arguments.add_all(compiler_flags)
        if is_header:
            arguments.add_all(["-x", "c++-header"])
        arguments.add_all(CLANG_DIAGNOSTIC_FLAGS)

        outputs = ctx.actions.args()
        outputs.add(report.path)
        outputs.add(fixes.path)
        ctx.actions.run(
            executable = ctx.executable._wrapper,
            arguments = [outputs, arguments],
            inputs = depset([file], transitive = [inputs]),
            outputs = [report, fixes],
            mnemonic = "ClangTidy",
            progress_message = "Running clang-tidy on %{label}: " + file.short_path,
            # PATH, for the clang-tidy the wrapper runs (.bazelrc passes --action_env=PATH).
            use_default_shell_env = True,
        )
        ctx.actions.write(output = arguments_file, content = arguments)
        entries.append(struct(source = file, report = report, fixes = fixes, arguments = arguments_file))

    return [
        ClangTidyInfo(files = entries),
        OutputGroupInfo(
            clang_tidy = depset([entry.report for entry in entries]),
            clang_tidy_fixes = depset([entry.fixes for entry in entries]),
            clang_tidy_args = depset([entry.arguments for entry in entries]),
        ),
    ]

def _make_aspect(config, output_directory):
    return aspect(
        implementation = _clang_tidy_aspect_impl,
        attr_aspects = [],
        fragments = ["cpp"],
        attrs = {
            # "selftest" lints the fixtures of FIXTURE_PACKAGES; only the rules below set it.
            "scope": attr.string(default = "repository", values = ["repository", "selftest"]),
            "_clang_format": attr.label(default = Label("//:.clang-format"), allow_single_file = True),
            "_config": attr.label(default = config, allow_single_file = True),
            "_output_directory": attr.string(default = output_directory),
            "_wrapper": attr.label(
                default = Label("//tools/clang_tidy:run_clang_tidy.sh"),
                allow_single_file = True,
                executable = True,
                cfg = "exec",
            ),
        },
        toolchains = use_cc_toolchain(),
    )

# The enforced checks (//:.clang-tidy): `make lint-tidy` and CI.
clang_tidy_aspect = _make_aspect(Label("//:.clang-tidy"), "clang_tidy")

# Every candidate check (sweep.clang-tidy), while the sweep removes their findings: `make lint-tidy-sweep` and
# `make lint-tidy-fix`. The two aspects differ only in a private attribute, so switching between them keeps the build
# configuration, and with it the analysis cache and the generated files.
clang_tidy_sweep_aspect = _make_aspect(Label("//tools/clang_tidy:sweep.clang-tidy"), "clang_tidy_sweep")

def _clang_tidy_args_impl(ctx):
    files = []
    manifest = []
    for fixture in ctx.attr.targets:
        if ClangTidyInfo not in fixture:
            continue
        for entry in fixture[ClangTidyInfo].files:
            files.append(entry.arguments)
            manifest.append({
                "arguments": entry.arguments.short_path,
                "source": entry.source.short_path,
                "target": str(fixture.label),
            })
    manifest_file = ctx.actions.declare_file(ctx.label.name + ".json")
    ctx.actions.write(manifest_file, json.encode_indent(manifest))
    return [DefaultInfo(
        files = depset([manifest_file]),
        runfiles = ctx.runfiles(files = files + [manifest_file]),
    )]

# The clang-tidy arguments of every file the aspect lints in `targets`, with a JSON manifest, for
# test_clang_tidy_aspect.py. It writes the arguments and runs no clang-tidy.
clang_tidy_args = rule(
    implementation = _clang_tidy_args_impl,
    attrs = {
        "scope": attr.string(default = "selftest", values = ["repository", "selftest"]),
        "targets": attr.label_list(aspects = [clang_tidy_aspect]),
    },
)

def _clang_tidy_selftest_impl(ctx):
    fixtures = []
    inputs = []
    for fixture in ctx.attr.fixtures:
        entries = fixture[ClangTidyInfo].files if ClangTidyInfo in fixture else []
        fixtures.append({
            "fixes": [entry.fixes.path for entry in entries],
            "name": fixture.label.name,
            "reports": [entry.report.path for entry in entries],
            "sources": [entry.source.path for entry in entries],
        })
        for entry in entries:
            inputs += [entry.report, entry.fixes, entry.source]
    manifest = {
        "configs": [config.path for config in ctx.files.configs],
        "expected": [expected.path for expected in ctx.files.expected],
        "fixtures": fixtures,
        "golden": [golden.path for golden in ctx.files.golden],
        "golden_prefix": ctx.attr.golden_prefix,
        "renamed_prefix": ctx.attr.renamed_prefix,
        "wrapper": ctx.executable._wrapper.path,
    }
    manifest_file = ctx.actions.declare_file(ctx.label.name + ".manifest.json")
    ctx.actions.write(manifest_file, json.encode_indent(manifest))
    stamp = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.run(
        executable = ctx.executable._selftest,
        arguments = [manifest_file.path, stamp.path],
        inputs = depset(inputs + ctx.files.configs + ctx.files.expected + ctx.files.golden + [manifest_file]),
        outputs = [stamp],
        tools = [ctx.attr._wrapper[DefaultInfo].files_to_run],
        mnemonic = "ClangTidySelftest",
        progress_message = "Checking the clang-tidy aspect against its fixtures",
        use_default_shell_env = True,
    )

    # The stamp is empty when the self-test passes. It is in the `clang_tidy` output group, which is what
    # `make lint-tidy` builds, so that the collector reads it like any report.
    return [
        DefaultInfo(files = depset([stamp])),
        OutputGroupInfo(clang_tidy = depset([stamp])),
    ]

# Lints the fixtures with every candidate check and fails unless each fixture's findings, deduplicated, are its
# `<name>.expected` file, and unless the fix-its of the `renamed_prefix` fixture, applied all or nothing, give the golden
# copy. Needs clang-tidy, so it is `manual` and built by `make lint-tidy`.
clang_tidy_selftest = rule(
    implementation = _clang_tidy_selftest_impl,
    attrs = {
        "configs": attr.label_list(allow_files = True, doc = "Configurations whose checks and options must exist."),
        "expected": attr.label_list(allow_files = [".expected"]),
        "fixtures": attr.label_list(aspects = [clang_tidy_sweep_aspect]),
        "golden": attr.label_list(allow_files = True),
        "golden_prefix": attr.string(mandatory = True),
        "renamed_prefix": attr.string(mandatory = True),
        "scope": attr.string(default = "selftest", values = ["repository", "selftest"]),
        "_selftest": attr.label(
            default = Label("//tools/clang_tidy:clang_tidy_selftest"),
            executable = True,
            cfg = "exec",
        ),
        "_wrapper": attr.label(
            default = Label("//tools/clang_tidy:run_clang_tidy.sh"),
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
    },
)
