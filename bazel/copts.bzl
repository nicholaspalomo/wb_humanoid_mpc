"""The C++ warnings every first-party target compiles with (AGENTS.md "Style guides").

    load("//bazel:copts.bzl", "FIRST_PARTY_COPTS")

    cc_library(
        name = "foo",
        copts = FIRST_PARTY_COPTS,
        ...
    )
"""

# Each flag names the guide section (G:) or tip (ToTW #) it enforces. GCC reports these in the .cpp files and in the
# headers they include that are not system headers, and the third-party ones are system headers: external repositories
# (Abseil, protobuf, googletest, the system_libs wrappers of Eigen and Pinocchio) arrive as -isystem through
# .bazelrc's external_include_paths feature, and the vendored lib/ocs2 and CppAD directories through the
# system_include_paths feature of lib/ocs2/BUILD.bazel. First-party `includes = [...]` stay -I, so their headers are
# warned about. //bazel:first_party_copts_probe compiles every third-party header family with these flags and -Werror,
# which proves in every build that none of them trips one. Casts, nullptr and override are left to the clang-tidy aspect
# (tools/clang_tidy), which filters by header. A flag that a third-party header trips is dropped here with a comment
# naming the header, never suppressed per file. No -std=c++20: .bazelrc sets the standard.
# LINT.IfChange(first_party_copts)
FIRST_PARTY_COPTS = [
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror=return-type",
    "-Werror=switch",  # ToTW #147 (with tools/hooks totw-enum-switch-default)
    "-Werror=implicit-fallthrough",  # G: Switch statements ([[fallthrough]])
    "-Werror=unused-result",  # ToTW #76: absl::Status / StatusOr are [[nodiscard]]
    "-Werror=range-loop-construct",  # ToTW #232 and the no-auto rule: std::pair<const K, V>
    "-Werror=pessimizing-move",  # ToTW #77, #166
    "-Werror=redundant-move",  # ToTW #77, #166
    "-Werror=self-move",  # ToTW #77
    "-Werror=uninitialized",  # ToTW #146, #182
    # GCC 13 makes -Wmaybe-uninitialized an error under -Werror=uninitialized too; it stays a warning, because Eigen's
    # selfadjoint and triangular matrix-vector products trip it at -O2.
    "-Wno-error=maybe-uninitialized",
    "-Werror=dangling-pointer",  # ToTW #5
    "-Werror=missing-declarations",  # ToTW #186; G: Internal linkage
    "-Werror=vla",  # G: Nonstandard extensions
    "-Werror=unused-parameter",  # G: Function declarations and definitions (an unused parameter's name is commented out)
]
# LINT.ThenChange(//AGENTS.md:cpp_style, //AGENTS.md:totw_rules)
