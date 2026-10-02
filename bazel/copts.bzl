"""The C++ warnings every first-party target compiles with (AGENTS.md "Style guides").

    load("//bazel:copts.bzl", "FIRST_PARTY_COPTS")

    cc_library(
        name = "foo",
        copts = FIRST_PARTY_COPTS,
        ...
    )
"""

# Each flag names the guide section (G:) or tip (ToTW #) it enforces. GCC reports these in the .cpp files and in every
# header they include that is not a system header, and here almost none is: rules_cc passes `includes = [...]` as -I
# (its system_include_paths feature is off), first-party and vendored (lib/ocs2, CppAD) directories as well as the
# system_libs wrappers of Eigen, Pinocchio and yaml-cpp, and Abseil, protobuf and googletest arrive as -iquote / -I
# (external_include_paths is off). So a flag here must be clean on every third-party header family first-party code
# includes, which //bazel:first_party_copts_probe proves in every build. Casts, nullptr and override are left to the
# clang-tidy aspect (tools/clang_tidy), which filters by header; protobuf, gtest and Abseil headers would trip them under
# GCC. A flag that fails on the probe is dropped here with a comment naming the header, never suppressed per file. No
# -std=c++20: .bazelrc sets the standard.
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
    "-Werror=uninitialized",  # ToTW #146, #182 (-Wmaybe-uninitialized stays a warning: Eigen at -O2)
    "-Werror=dangling-pointer",  # ToTW #5
    "-Werror=missing-declarations",  # ToTW #186; G: Internal linkage
    "-Werror=vla",  # G: Nonstandard extensions
]
# LINT.ThenChange(//AGENTS.md:cpp_style, //AGENTS.md:totw_rules)
