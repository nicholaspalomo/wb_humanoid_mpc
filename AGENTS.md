# Coding Agent Style Guidelines & Rule Directives

#### Co-dependent changes: use IFTTT directives (`LINT.IfChange` / `LINT.ThenChange`)

When code or configuration in one place must stay in sync with code elsewhere - but DRY cannot eliminate the duplication (for example, package registrations, launch targets, system dependencies, or Bazel repository rules) - mark the dependency with `LINT.IfChange` / `LINT.ThenChange` directives so changes to one side prompt review of the other. Add these directives proactively when creating new co-dependent content, not just when maintaining existing pairs.

Keep the guarded block as small as possible. Prefer several small labeled source->target pairs over one large catch-all block unless the whole region genuinely needs to change together. Directives are enforced via `ifttt-lint`.

<details>
<summary>Example</summary>

```bash
# LINT.IfChange(registered_packages)
_setup_package "my_robot_description" "${SCRIPT_DIR}/robot_models/my_robot/my_robot_description"
# LINT.ThenChange(//Makefile:launch_targets, //.devcontainer/README.md:launch_targets)
```

</details>

#### Builds share one machine's memory

A build here once exhausted a 32 GB workstation: twelve parallel compiles of Pinocchio / CppAD / OCS2 translation
units (2-3 GB each) next to the simulator, the 3D visualizer and out-of-tree compiles pushed it into swap until it had
to be power-cycled. Two guards exist, and changes must not weaken them:

<!-- LINT.IfChange(build_memory) -->
- `.bazelrc` bounds Bazel's parallelism by RAM (`--jobs=HOST_RAM*...`, `--local_test_jobs` half of that) rather than by
  cores. Do not pass a larger `--jobs` or `--local_test_jobs` on the command line; a machine with more headroom raises
  it in a git-ignored `user.bazelrc`.
- The dev container is capped at 85% of the host's RAM with no swap beyond the cap
  (`tools/resource_limits/set_container_memory_limit.sh`, `docker-compose.yaml`), so an overcommit is ended inside the
  container instead of freezing the host.
- Bazel can only budget what it schedules. Run one `bazel` command at a time, and never run compilers outside Bazel
  (an out-of-tree build for a mutation check, a hand-invoked `g++`) while a build or tests are running: run them
  afterwards, one at a time, each under a `ulimit -v` of a few GB.
<!-- LINT.ThenChange(//.bazelrc:bazel_memory_bound) -->

<!-- LINT.IfChange(clangd_memory) -->
The editor's clangd is a compiler front end outside that budget too: `tools/ide/clangd_in_dev_container.sh` runs its
background index with two workers at low priority, and changes must not raise them (`tools/ide/README.md`).
<!-- LINT.ThenChange(//tools/ide/clangd_in_dev_container.sh:clangd_flags) -->

Separate containers each size their builds to the whole machine, so `tools/bazel` (which Bazelisk runs in place of
Bazel in the dev container, in CI and in `make ci-local`) makes the `build`, `test` and `coverage` commands of every
container take turns through a lock file in the shared checkout. A second build waits and says why. Do not bypass it
(`BAZELISK_SKIP_WRAPPER`), and keep that script POSIX `sh`: the dev container's `BASH_ENV` makes every bash script
source the shell setup first, and a bash wrapper around Bazel once recursed through it until the container ran out of
memory.

#### American English throughout

Write American spelling everywhere in the repository: identifiers, comments, strings, log messages, YAML, Python and
documentation - color, behavior, center, meter, initialize, normalize, analyze, modeling, labeled, canceled, defense,
gray, program. `make lint` checks it (`american-spelling`, tools/hooks/american_spelling.py; `python3 -m
tools.hooks.american_spelling --fix <files>` fixes them); a line that must keep a British spelling, such as the title of
a cited paper or the name of the license file, is marked `NOLINT(american-spelling): <reason>`. Vendored code under `lib/` and robot model files are left as they come.

#### Style guides: Google C++, Google Python and the Abseil Tips of the Week

C++ follows the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and the
[Abseil C++ Tips of the Week](https://abseil.io/tips/) (ToTW); Python follows the
[Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). This file outranks them: its own rules
(American spelling, IFTTT directives, and the departures below, such as no `auto` and argument comments) are stricter
than the guides or orthogonal to them. Where a tip and the style guide disagree, the tip is the more specific advice and
wins. The sections below list every place this repository departs from the guides; everything not listed applies as
written.

The rules cover first-party code. Vendored code under `lib/` is left as it comes, with two exceptions for the OCS2
sources in `lib/ocs2` (not the CppAD and iit copies in `lib/ocs2/thirdparty`): they stay Boost-free, and their raw
pointers carry nullability annotations. Generated code is held to the rules through its generator
(`tools/nproto/nproto_generator.py`) rather than by linting `bazel-bin`; the generated ACOM weight headers are listed in
`tools/hooks/lint_files.py` and skipped. Every C++ and Python file starts with the BSD-3 license comment block of
the repository's license file: the holder line, the three conditions and the disclaimer (`license-header`, `py-license-header`; cpplint
`legal/copyright` only looks for the word). A new file's holder line is
`Copyright (c) <year>, Nicholas Palomo. All rights reserved.`; an existing holder line is kept, and a holder line
without its holder gets the one `git log` names.

A rule that a tool checks is a hard failure: there is no baseline file and no grandfathered finding. Fix the finding.
Where a rule is genuinely wrong for one line, say why on that line: `// NOLINT(<check>): <reason>` (`#` in Python,
shell and Starlark; or `NOLINTNEXTLINE`, or a `NOLINTBEGIN` / `NOLINTEND` pair around a block), for pylint's own
messages `# pylint: disable=<message>  # <reason>` (a block pragma alone on its line may say why on the comment line
above it), and for mypy `# type: ignore[<code>]  # <reason>`. A marker without a check name or without a reason fails
the lint (`nolint-category`, `nolint-reason`, `py-pragma-reason`), and so does a marker that names no known check
(`nolint-unknown`), one for the repository's own checks that suppresses nothing (`nolint-unused`), an unclosed
`NOLINTBEGIN` (`nolint-unbalanced`), a bare `# type: ignore` (`py-type-comment`) and `# pylint: skip-file`
(`py-pragma-reason`). The agent rules below are the ones no tool can check. They apply to new code and to the code a change touches; they are not
a reason to restyle unrelated code in the same change.

#### C++: where this repository departs from the Google C++ Style Guide

<!-- LINT.IfChange(cpp_style) -->
- **`auto` is not written**, except as `auto x = std::make_unique<T>(...)` or `auto x = std::make_shared<T>(...)`, where
  the type is already on the line. This is stricter than the guide's "Type deduction" and also rules out range-for
  `auto`, structured bindings, generic lambdas, lambdas stored in an `auto` variable, deduced return types and trailing
  return types. Spell map elements `std::pair<const K, V>`; without the `const` each element is copied, which
  `-Werror=range-loop-construct` rejects. On the realtime path, pass a lambda straight to a template or
  `absl::FunctionRef` parameter instead of storing it in a `std::function`, which can allocate.
- **Class template argument deduction** is not written either: `std::array<int, 3> a = {1, 2, 3};`, not
  `std::array a = ...`, although the guide presumes the `std` templates opted in. The one exception is `absl::Cleanup`,
  whose template argument is a lambda type that cannot be spelled.
- **`absl::Status` and `absl::StatusOr` are never dropped.** Handle the value, return it, or call `.IgnoreError()` with
  a comment saying why. A `(void)` cast does not count.
- **Literal arguments** carry `/*parameterName=*/` comments, with the name copied from the callee's declaration
  (`tools/hooks/argument_comments.py`; clang-tidy's `bugprone-argument-comment` checks the names) - not only as a last
  resort. A literal that the function name already explains (`std::pow(x, 2)` or `v.resize(3)` in a test, where
  literals need no name) needs none.
- **Magic numbers** are named constants, in production C++ and lib/ocs2: a literal is a
  `static constexpr T kName = ...;` in its class, a `constexpr` in the .cpp's unnamed namespace or an `inline constexpr`
  in a shared header, named for what it is (`kPositionDim`, `kDefaultSwingHeight`, never `kThree`) and of the
  literal's type, so the arithmetic is unchanged, with a short comment when its meaning or unit is not obvious. Exempt:
  0 and 1, a floating 0.5 or 2.0 as a factor of `*` or `/` (the integer 2 never is), and literals in a constant's
  definition or an enumerator; tests, test support and test-only libraries, benchmarks, the generated ACoM weight
  headers, the clang-tidy fixtures and lib/ocs2/thirdparty are not read. A literal that must stay carries
  `// NOLINT(magic-number): <reason>`. The generic dimensions are in `lib/ocs2/core/include/ocs2_core/Dimensions.h`.
- **Naming.** Types, constants and enumerators (`kName`), namespaces, macros and the trailing `_` of class data members
  follow the guide. Functions, methods, variables and parameters are camelCase as in OCS2, whose virtual functions we
  override (the guide has PascalCase functions and snake_case variables); match the file you are in. Files are
  `PascalCase.h` / `PascalCase.cpp` and tests `testThing.cpp`. OCS2's lowercase type aliases (`scalar_t`, `vector_t`)
  and `SCALAR_T` template parameters stay.
- **Header guards** are `#pragma once`, not `#define` guards.
- **Namespaces.** First-party code lives in `ocs2::humanoid` and `robot::`, not under a project-owned top-level name.
  Never name a nested namespace `testing`, `std`, `absl` or `util` (ToTW #130), nor `Eigen` or `pinocchio`.
  Implementation details may live in a nested `internal` namespace, as `nproto::internal` does.
- **Exceptions.** First-party code does not throw; it returns `absl::Status` / `absl::StatusOr`. Two boundaries keep
  exceptions, each with a `NOLINT(exceptions): <reason>`: an override of an OCS2 virtual function that has no status
  channel may throw, and a call into a library that throws (Pinocchio, OCS2's solvers) is wrapped in
  one `try` / `catch` that converts the exception to an `absl::Status` at once. Nothing on the realtime thread throws.
- **`<filesystem>`** is used, although the guide bans it: no Abseil equivalent exists. Use the `std::error_code`
  overloads, which do not throw, and convert failures to `absl::Status`.
- **OCS2 class hierarchies** keep their protected data members, `std::shared_ptr` parameters and owning
  `T* absl_nonnull clone() const` returns, because the OCS2 base classes define them. New class hierarchies have private
  data and hand over ownership with `std::unique_ptr`.
- **Status macros.** `RETURN_IF_ERROR` and `ASSIGN_OR_RETURN` (`humanoid_common_mpc/common/StatusMacros.h`) keep the
  names Google uses for them. Every other macro defined in a header starts with its library's prefix (`ROBOT_IPC_`,
  `NPROTO_`, `HUMANOID_`, `WBMPC_`), and a macro defined in a `.cpp` file is `#undef`ined at the end of it.
- **Braces.** A controlled statement without braces fits on the line of its condition; the guide's two-line braceless
  form is not used, and `if ... else` always has braces.
- **Includes.** Only C and C++ standard library headers and POSIX/Linux system headers use angle brackets. Everything
  else, third-party libraries included, uses quotes and its full path. clang-format orders the include blocks,
  `"pinocchio/fwd.hpp"` first. A first-party `cc_library` exports its headers with `includes = [...]`, never
  `strip_include_prefix` / `include_prefix`, so that the tools see each header at its source path.
- **Switch fallthrough** is written `[[fallthrough]];` (`-Werror=implicit-fallthrough`).
- **Warnings.** Every first-party C++ target compiles with `FIRST_PARTY_COPTS` from `bazel/copts.bzl` instead of a
  package-local flag list. A target may add `-Werror`, and a `-Wno-...` with a reason on its line, and no other warning
  flag; never `-w`, `-fpermissive`, `-Wno-error` or a `-Wno-...` of a flag the constant makes an error. Third-party
  headers are system headers (.bazelrc's `external_include_paths`, lib/ocs2's `system_include_paths`), so the warnings
  are about first-party code; a flag that a third-party header still trips is dropped from the constant, never silenced
  per file.
<!-- LINT.ThenChange(//tools/hooks/checks.py:registry, //.clang-tidy:checks, //bazel/copts.bzl:first_party_copts) -->

Enforced by:
- clang-format (layout, include order);
- cpplint (`CPPLINT.cfg`);
- clang-tidy (`.clang-tidy`, run by `make lint-tidy`), which also checks that move constructors are `noexcept`
  (`performance-noexcept-move-constructor`) and that default arguments go only on non-virtual functions
  (`google-default-arguments`);
- the checks in `tools/hooks`, which cover:
  - floating-point literals with a radix point (`float-literal`);
  - named constants instead of magic numbers (`magic-number`);
  - prefix `++` (`postfix-increment`), and `int64_t` / `size_t` without `std::` (`std-integer-type`);
  - no `auto` and no class template argument deduction (`no-auto`, `ctad`);
  - argument comments on literal arguments, `{}`, `""` and `std::nullopt` included (`argument-comment`);
  - exceptions and RTTI, `std::dynamic_pointer_cast` and `std::any_cast` included, only at their boundaries
    (`exceptions`, `rtti`);
  - no `assert`, `ABSL_DCHECK` or `DCHECK` outside tests, which every `-c opt` build compiles out (`debug-only-check`);
  - header macros, and `#pragma once` instead of include guards (`macro-naming`);
  - static storage, Eigen, OCS2 and `absl::Status` types included (`static-storage`);
  - quoted non-system includes (`include-style`);
  - class comments (`class-comment`);
  - braces around `if ... else` (`if-else-braces`);
  - the `TODO` format (`todo-format`);
  - inclusive language (`inclusive-language`);
  - the license block (`license-header`);
  - the forbidden constructs (`long double` and its `1.0L` literals, user-defined literals, inline namespaces,
    `decltype(auto)`, coroutines, modules, `<ratio>`, `<cfenv>`, `alloca`, GNU extensions such as `x ?: y`)
    (`forbidden-construct`);
- the GCC warnings of `bazel/copts.bzl`.

Protocol Buffers and dependencies are enforced by the checks in `tools/hooks` too (humanoid_mpc_msgs/README.md):
<!-- LINT.IfChange(protobuf_rules) -->
- the project is Boost-free: only the `@pinocchio` target depends on Boost (`boost`);
- the project is YAML-free: no `#include` of a `yaml-cpp/` header and no `@yaml_cpp` label (`yaml-cpp`), and no `.yaml`
  or `.yml` file under `robot_models/` or `humanoid_nmpc/`, whose configuration files are typed textprotos
  (`config-yaml`, humanoid_nmpc/humanoid_mpc_config/README.md);
- one top-level message or enum per `.proto` file, named after it, with its nproto option, and no message reserves a
  field name - `reserved "old";` or `reserved old;` - which protobuf's C++ text parser would silently skip: a retired
  field is listed in `(nproto.retired_field)` instead (`proto-file-layout`, tools/nproto/README.md);
- every message names its next free field number (`// Next ID: N`) and opens with `reserved N to max;`
  (`proto-next-id`);
- every `.textproto` names its schema in its leading comment block, `# proto-file: <path>` and
  `# proto-message: <package>.<Message>`, and the schema exists (`textproto-header`).
<!-- LINT.ThenChange(//tools/hooks/checks.py:registry) -->

The base orientation is a unit quaternion (humanoid_nmpc/docs/quaternion_base_orientation/README.md), and the checks in
`tools/hooks` keep its Euler parameterization at the boundary of the formulation:
<!-- LINT.IfChange(orientation_rules) -->
- no `SphericalZYX` root joint in the C++ of `humanoid_nmpc/`, `robot_runtime/` and
  `lib/ocs2/pinocchio/centroidal_model/` (`spherical-zyx`);
- no Euler API - an identifier containing `EulerZyx`, `EulerAnglesZyx` or `ZyxEulerAngles` in any capitalization, or an
  `#include` of `RotationDerivativesTransforms.h` or `orientation/EulerBoundary.h` - in the formulation directories of
  `humanoid_nmpc/` (`dynamics`, `cost`, `constraint`, `end_effector`, `acom`, `parameter_update`, `pinocchio_model` and
  `contact` below `src/` and `include/<package>/`), in the formulation files (`HumanoidPreComputation`,
  `WBMpcPreComputation`, `orientation/BaseOrientation` and lib/ocs2's `PinocchioEndEffectorKinematicsCppAd`), in
  `lib/ocs2/pinocchio/centroidal_model/` and in the robot-model headers, whose Euler boundary accessors carry
  `NOLINT(euler-boundary): Euler boundary accessor (design section 4.2)` (`euler-boundary`; the full list is in
  `tools/hooks/euler_boundary.py`);
- no linear interpolation of a state trajectory under `humanoid_nmpc/` outside the robot models: no one-argument
  `getDesiredState(t)` and no `LinearInterpolation::interpolate` of anything but inputs (`state-interpolation`);
- no quaternion built from four scalars under `humanoid_nmpc/` outside `orientation/`, since Eigen's constructor takes
  `(w, x, y, z)` and the state stores `(x, y, z, w)` (`quaternion-scalar-constructor`).
<!-- LINT.ThenChange(//tools/hooks/checks.py:registry) -->

Agent rules (no tool checks these):
- Write short functions; past about 40 lines, look for a split. That is a prompt, not a limit.
- Constructors do no work that can fail and call no virtual functions. Fallible construction is a
  `static absl::StatusOr<std::unique_ptr<T>> Create(...)` factory, not an `Init()` method (ToTW #42).
- Make copy and move semantics explicit in the public section unless a base class makes them obvious. A `struct` is
  passive data with no invariants; anything else is a `class` with private data. Prefer a struct with named fields to
  `std::pair` / `std::tuple`.
- Prefer composition to inheritance. Inheritance is public and "is-a"; there is no multiple implementation
  inheritance.
- Overload operators only with their built-in meaning. Binary operators are non-members in the type's namespace.
- An object has one owner and changes hands as a `std::unique_ptr`. Use `std::shared_ptr` only where an OCS2 or
  third-party API demands it, and say so.
- Prefer return values to output parameters (ToTW #176). A required output is a `T&` and an optional output a
  `T* absl_nullable`. An optional input is a `std::optional<T>` or a `const T* absl_nullable`. Inputs come before
  outputs. Two kinds of code may keep output parameters: preallocated buffers on the allocation-free realtime path, and
  OCS2 virtual signatures.
- Overload only when every overload has the same meaning, and document the set with one comment. Default arguments
  have constant values.
- A lambda that can outlive its scope (a thread, a stored callback) lists its captures. `[&]` only when the lambda
  obviously dies first; `[=]` never captures `this`.
- Avoid template metaprogramming. Constrain templates with `requires` and the standard concepts; add no new public
  concepts.
- Choose libraries in this order: what the code base already uses, Abseil, the standard library, then first-party code.
  Abseil first means `absl::Status` / `StatusOr` with `RETURN_IF_ERROR` / `ASSIGN_OR_RETURN`, `absl::StrCat` /
  `StrJoin` / `StrFormat`, `absl::flat_hash_map` / `flat_hash_set`, Abseil logging and `absl::string_view`.
- Names are as descriptive as their scope is wide, with no letter-dropping abbreviations. A new identifier writes an
  acronym as one word (`IcpCost`, not `ICPCost`).
- Comments:
  - A class comment says what the class is for, how to use it and whether it is thread-safe.
  - A function comment is a verb phrase. It covers inputs and outputs, which pointer and reference arguments are
    retained, which may be null, what happens to output arguments, and costs.
  - Members document their invariants and sentinels.
  - Never restate the code.
- Use `int64_t` for anything that may reach 2^31. Do not use unsigned types to say "non-negative".
- Do not format with streams: use `absl::StrCat` / `StrFormat` / `StreamFormat` and Abseil logging. Overload `<<` only
  for value types; prefer `AbslStringify` (ToTW #215).
- The `const` methods of a class are safe to call concurrently, or the class says it is not thread-safe.
- Do not use gendered pronouns for unspecified people. Software is "it".

#### Raw pointers carry nullability (`absl_nonnull` / `absl_nullable`)

<!-- LINT.IfChange(nullability) -->
Every raw-pointer declarator in first-party code and in `lib/ocs2` (not `lib/ocs2/thirdparty`) carries `absl_nonnull`
or `absl_nullable` directly after its `*`, the way `const` qualifies a const pointer:
- parameters, returns, members, locals, template arguments, lambda parameters, range-for and condition variables, and
  aliases: `const Foo* absl_nonnull`, `std::vector<Foo* absl_nonnull>`, `Foo* absl_nonnull const p`;
- every level of a multi-level pointer: `char* absl_nonnull* absl_nonnull argv`;
- function and member pointers: `void (*absl_nonnull fn)(int)`, `int Foo::*absl_nullable m`;
- array parameters: `const double p[absl_nonnull 3]`.

A file that uses an annotation includes `"absl/base/nullability.h"`, and its target depends on
`@abseil-cpp//absl/base:nullability`. Casts, `sizeof`, `alignof`, `typeid`, `decltype`, `new T*[n]` and type-trait
arguments stay unannotated: an annotated cast is an assertion that silences the analyzer, not documentation. Do not use
`absl_nonnull Foo* p`, which qualifies `Foo` rather than the pointer, nor `ABSL_POINTERS_DEFAULT_NONNULL`, the removed
`absl::Nonnull<>` / `absl::Nullable<>` templates, or raw `_Nonnull`. `absl_nullability_unknown` is only for a pointer
whose contract genuinely cannot be decided (an undocumented C API), with `// NOLINT(pointer-nullability): <why>`.
String constants are not pointers: `inline constexpr char kName[] = "...";` (ToTW #140).

GCC expands the macros to nothing, so the build checks nothing. `tools/hooks/pointer_nullability.py`
(`pointer-nullability`) checks that every pointer is annotated, one declarator per declaration, no annotation in a cast,
and no null literal given to an `absl_nonnull` pointer (as its initializer or default, `{}`, a branch of `?:`, in a
member initializer list, by an assignment, or returned). clang-tidy (`make lint-tidy`, clang 21) sees the annotations:
clang reports a misplaced one, `nullptr` passed to or returned from an `absl_nonnull` pointer
(`clang-diagnostic-nonnull`), and a redeclaration that contradicts the first (`clang-diagnostic-nullability`); the
self-test's `nullability` fixture fails when they stop firing. Neither can tell whether the choice is right, and a wrong one is
silent. Choose from the implementation and every caller, not from the type:
- `absl_nonnull` when:
  - the pointee is dereferenced without a check;
  - the pointer was taken from a reference, `this`, `new` or the `get()` of an owner that outlives it;
  - it is a required output, a `clone()` result, a C-API handle passed in, or `operator->`.
- `absl_nullable` when:
  - any path compares the pointer with `nullptr`, defaults it to `nullptr` or returns `nullptr`;
  - it is a lookup result (`find`, `getenv`, `dynamic_cast`), an optional observer or sink, a member set after
    construction, or the result of a C-API constructor that can fail.
- An override repeats its base's annotations, and an out-of-line definition repeats its declaration. Nothing compares an
  override with its base: the owner of a base class changes all its overrides in the same change.
- `main` takes `char* absl_nonnull* absl_nonnull argv`. Its elements are non-null below `argc`, and the code never reads
  `argv[argc]`.

A nullable value becomes a nonnull one only after a check. Where it becomes an invariant (a constructor storing an
injected pointer, a lookup or `dynamic_cast` that must succeed), check it once with `ABSL_CHECK(p != nullptr)` or
`ABSL_DIE_IF_NULL(p)`. Never use `ABSL_DCHECK` or `assert`: every build is `-c opt` (`debug-only-check`). Never check per tick in the realtime
loop; establish the invariant at construction, and give the realtime path unchecked accessors with a documented
precondition (`IDMapBase::operator[]`, `RobotState::getJointPosition()`). A C-API constructor that can fail reports an `absl::Status`. Escape:
`// NOLINT(pointer-nullability): <reason>`.
<!-- LINT.ThenChange(//tools/hooks/pointer_nullability.py:rules, //.clang-format:attribute_macros) -->

#### C++: the Abseil Tips of the Week

<!-- LINT.IfChange(totw_rules) -->
Checked by a tool (tip: what is enforced, by what):

| Tip | Rule | Checked by |
|---|---|---|
| #1, #93, #234 | views and spans by value, never `const absl::string_view&` / `const absl::Span<...>&` | `totw-view-param` |
| #3 | `absl::StrCat` / `StrAppend`, no `+` chains; `absl::StrContains` / `StartsWith` | clang-tidy `performance-inefficient-string-concatenation`, `abseil-str-cat-append`, `abseil-redundant-strcat-calls`, `abseil-string-find-str-contains`, `abseil-string-find-startswith` |
| #5 | no `c_str()` of a temporary stored in a pointer | `totw-c-str-stored`, `-Werror=dangling-pointer` |
| #10 | `absl::StrSplit` with a character delimiter; no `strtok` | `abseil-faster-strsplit-delimiter`, `forbidden-construct` |
| #45, #103 | `ABSL_FLAG` only in `*Main.cpp`, `*AppFlags.{h,cpp}` and test drivers | `totw-flag-location` |
| #55, #77 | no use after `std::move`; no pessimizing, redundant or self moves | `bugprone-use-after-move`, `performance-move-const-arg`, `-Werror=pessimizing-move,redundant-move,self-move` |
| #61, #146, #182 | every scalar and pointer initialized; default member initializers | `cppcoreguidelines-init-variables`, `cppcoreguidelines-pro-type-member-init`, `modernize-use-default-member-init`, `-Werror=uninitialized` |
| #64 | raw string literals instead of escapes | `modernize-raw-string-literal` |
| #76 | `absl::Status`, not exceptions; a returned Status is never dropped | `exceptions`, `-Werror=unused-result`, `bugprone-unused-return-value` |
| #86 | `enum class` | `totw-unscoped-enum` |
| #88 | `=` for values, not `bool x_{false};` | `totw-brace-literal-init` (fixed by `make format`) |
| #99, #152, #218 | nothing added to `namespace std`, no `std::hash` specializations | `totw-std-specialization`, `cert-dcl58-cpp` |
| #101, #107, #180 | no references or views to temporaries; no view-typed members | `totw-view-member`, `bugprone-dangling-handle`, `bugprone-return-const-ref-from-parameter`, `readability-reference-to-constructed-temporary`, `performance-unnecessary-copy-initialization` |
| #108 | lambdas, not `std::bind` | `modernize-avoid-bind` |
| #109 | no top-level `const` on parameters in declarations | `readability-avoid-const-params-in-decls` |
| #112 | `emplace_back` only to construct in place | `modernize-use-emplace` |
| #119, #153 | no `using namespace`; no namespace-scope using-declarations or aliases in headers | cpplint `build/namespaces`, `google-build-using-namespace`, `google-global-names-in-headers`, `misc-unused-using-decls`, `misc-unused-alias-decls` |
| #120 | `absl::Cleanup` without template arguments | `abseil-cleanup-ctad` |
| #124 | `absl::StrFormat`, not `printf` or iomanip | `totw-printf` |
| #126, #134 | `std::make_unique` / `make_shared`; `absl::WrapUnique(new T)` only for private constructors | `totw-raw-new`, `modernize-make-unique`, `modernize-make-shared` |
| #130 | no `testing` / `std` / `absl` / `util` sub-namespaces | `totw-namespace-name`, `abseil-no-namespace` |
| #131, #143 | `= default`; public `= delete` | `modernize-use-equals-default`, `modernize-use-equals-delete` |
| #135 | no `FRIEND_TEST`, no befriended test fixtures | `totw-friend-test` |
| #136 | `absl::flat_hash_map` / `flat_hash_set`, not `std::unordered_*` | `totw-unordered-container` |
| #140, #168 | header constants `inline constexpr`; string constants as `char[]` | `totw-header-constant`, `totw-string-constant`, `misc-definitions-in-headers` |
| #141 | no implicit bool conversion of a `bool*` | `bugprone-bool-pointer-implicit-conversion` |
| #142 | `explicit` single-argument constructors | `google-explicit-constructor`, cpplint `runtime/explicit` |
| #147 | no `default:` in a switch over an owned `enum class` | `totw-enum-switch-default`, `-Werror=switch` |
| #158 | `contains()` | `readability-container-contains` |
| #163 | no `const std::optional<T>&` parameters | `totw-optional-ref-param`, `bugprone-optional-value-conversion` |
| #166 | no `const` local that blocks the implicit move on return | `performance-no-automatic-move` |
| #172 | designated initializers for aggregates | `modernize-use-designated-initializers` |
| #181, #224 | no `.at()` outside tests; checked optional access | `totw-at`, `bugprone-unchecked-optional-access` |
| #186 | file-local functions in an unnamed namespace | `-Werror=missing-declarations`, `misc-use-anonymous-namespace`, `misc-use-internal-linkage` |
| #188 | no `const std::unique_ptr<T>&` / `const std::shared_ptr<T>&` parameters | `totw-smart-ptr-ref-param` |
| #197 | no reader locks | `totw-reader-lock` |
| #227 | `i + 1 < v.size()`, never `i < v.size() - 1` | `totw-size-minus` |
| #231 | `std::min` / `std::max`, not a hand-written comparison | `readability-use-std-min-max` |
| #232 | no `auto` (this file's rule, which is stricter); no copying range-for | `no-auto`, `-Werror=range-loop-construct`, `performance-for-range-copy`, `performance-implicit-conversion-in-loop` |
| #234 | numbers, enums and views by value | `totw-small-by-const-ref`, `performance-unnecessary-value-param` |
<!-- LINT.ThenChange(//tools/hooks/checks.py:registry, //.clang-tidy:checks, //bazel/copts.bzl:first_party_copts) -->

Agent rules:
- Take read-only strings as `absl::string_view` by value; store a `std::string` when you keep the data (#1).
- Format messages with `absl::Substitute` or `StrFormat`, and join with `absl::StrJoin` (#18, #36). Under GCC, a
  `StrFormat` mismatch is not caught at compile time, so test formatted messages.
- Return by value. Do not add output parameters or `std::move` "for speed" (#166, #176).
- Use a static factory returning `absl::StatusOr` rather than an `Init()` method (#42). Delegate constructors rather
  than sharing an init helper (#74).
- Define operators, `AbslHashValue` and `AbslStringify` next to their type, never in tests or on generated protos (#49,
  #99, #215).
- Take read-only sequences as `absl::Span<const T>` by value (#93). Eigen's counterpart is `const Eigen::Ref<const ...>&`:
  a const `Ref` may own a temporary copy of its argument, so copying it is not cheap (`performance-unnecessary-value-param`).
- Replace a `bool` parameter with an `enum class` or a registry name (#94).
- A function that keeps a pointer or reference to an argument takes it as `T* absl_nonnull` and documents how long it
  must live (#116). Pass sink parameters by value and move them (#117).
- Never touch a returned local after `return`, including from `absl::Cleanup` or a destructor (#120).
- In tests prefer `TEST` with local objects and free helpers to fixture state, and use a `*Peer` rather than `friend`
  (#122, #135).
- Hold a member by value first, then as `std::optional`, and use `std::unique_ptr` only for ownership or polymorphism
  (#123, #187).
- Look a key up once: use `operator[]`, `try_emplace`, `insert().second` or `contains()`. Use heterogeneous lookup, not
  `std::string(view)` temporaries (#132, #144).
- Never let hash-map iteration order reach numerics, golden data, logs or serialized output; iterate a `std::map`,
  `absl::btree_map` or sorted keys (#136).
- Mark a multi-argument constructor `explicit` unless its arguments are the value (#142).
- Do not write an exhaustive switch over a proto enum, because proto enums are open. After a switch over an owned enum,
  handle the impossible value (#147).
- Overload only when every overload means the same. Do not `= delete` rvalue overloads to enforce lifetimes (#148,
  #149).
- Name a local only when the name documents, simplifies, removes repetition or extends a lifetime. Scope it with
  `if (init; cond)` (#161, #165).
- Use `std::optional` or `StatusOr` instead of sentinels such as `-1` or NaN (#171).
- Use designated initializers, and an options struct for functions with many or confusable parameters (#172, #173).
- Use digit separators in long literals (`1'000'000`) (#175).
- Keep const and reference members out of copyable value types (#177).
- Check `.ok()`, then use `*` or `->`; move a value out with `*std::move(s)`. Avoid `.value()`. Use `operator[]` where
  an index is valid by construction; otherwise check it and return a Status (#181, #224).
- Use `std::clamp`, `midpoint` and `lerp`, and never bind a reference to their result (#231).
- Pass Eigen fixed-size types and CppAD `SCALAR_T` by `const&`, never by value (#234; Eigen alignment).

Not applied:
- Obsolete or superseded: #11 and #24, which predate C++17 guaranteed copy elision and are superseded by #166; #65,
  superseded by #112; the removed spellings `absl::make_unique`, `absl::Nullable<>` and `absl::optional`; #90 (retired
  flags). #117's sink-parameter advice still applies (above); only its copy-elision background is superseded.
- Not used here: #59, #198, #229.
- Only partly applicable: #130's "one top-level namespace per project"; the code lives in `ocs2::humanoid`.

#### Python: where this repository departs from the Google Python Style Guide

<!-- LINT.IfChange(python_style) -->
- **Line length.** black formats code at the guide's 80 columns, as clang-format does C++. Comments and docstrings may
  run to 140 (pylint, `py-docstring-summary`). Indentation is 4 spaces, as in Google's published pylintrc.
- **Type checking** uses mypy (`mypy.ini`), since pytype has reached end of life. Every non-test function is fully
  annotated. Code must run on Python 3.11, Bazel's hermetic interpreter, so:
  - no `type X = ...` statements or `class C[T]` generics;
  - `typing_extensions.override`, not `typing.override`;
  - no reuse of the enclosing quote character inside an f-string.
- **Imports.**
  - Import modules, not classes or functions: `from remote_control import operator_bus`, then
    `operator_bus.TopicPublisher`. `typing`, `collections.abc` and `typing_extensions` are exempt.
  - First-party modules are imported by their path below a Bazel `imports` root, or below the repository root for
    code without one (`from tools.hooks import checks`). A script that imports first-party modules is run as a module
    (`python3 -m tools.hooks.lint_code`), never as `python3 tools/hooks/lint_code.py`.
  - No relative imports, and no `sys.path` changes outside tests.
  - isort (`--profile google`) orders them.
- **Negated float comparisons are kept.** `not value >= 0.0` rejects NaN, which `value < 0.0` accepts, so pylint's
  `unnecessary-negation` is disabled; each such validator has a NaN unit test.
- **Tests** are `unittest.TestCase` classes collected by pytest, and they may call protected members of the unit under
  test.
- **Unused arguments** are deleted at the top of the body (`del event  # Unused.`), not renamed.
- **Exceptions.** `except Exception` is allowed only at a thread's outermost loop or another documented isolation
  point, with `# pylint: disable=broad-exception-caught  # <reason>`.
- **The license** is a `#` comment block above the module docstring, never the docstring itself.
<!-- LINT.ThenChange(//.pylintrc:repository_changes, //tools/hooks/checks.py:registry) -->

Enforced by:
- black and isort (`make format`), each at the version the lint pins (a missing or other version fails with CI=true);
- pylint (`.pylintrc`; Google's pylintrc plus the docstring and typing extensions);
- mypy;
- the checks in `tools/hooks`, which cover:
  - module-only, absolute and unaliased imports (aliases only from the standard list: `np`, `tk`, `rr`, ...) and no
    `sys.path` changes (`py-import-modules`, `py-relative-import`, `py-import-alias`, `py-sys-path`);
  - the license header, a comment and not the docstring (`py-license-header`, `py-license-docstring`);
  - the docstring summary line, its `Args:` / `Returns:` / `Yields:` sections, and property docstrings
    (`py-docstring-summary`, `py-docstring-sections`, `py-property-docstring`);
  - no `@staticmethod` (`py-staticmethod`);
  - comprehensions with one `for` and one condition at most (`py-complex-comprehension`);
  - one-line lambdas and conditional expressions (`py-long-lambda`, `py-long-ternary`);
  - no `len(x) == 0` (`py-length-test`);
  - no `assert` outside tests (`py-assert`);
  - exception class names ending in `Error` (`py-exception-name`);
  - no `__del__`, metaclasses, `exec` or `eval` outside tests (`py-power-feature`);
  - no backslash continuations or `# type:` comments (`py-backslash`, `py-type-comment`);
  - a reason on every pylint pragma and `# type: ignore[<code>]` (`py-pragma-reason`);
  - a shebang exactly on executable files (`py-shebang`);
  - a `main()` called from the `__main__` guard (`py-main-guard`);
  - the `TODO` format (`todo-format`).

Agent rules:
- Keep `try` blocks small, and clean up in `finally`. Raise built-in exceptions for API misuse; a custom exception's
  docstring says what it represents.
- Avoid mutable global state. Where it is unavoidable, give it a `_` name and a comment saying why.
- Nest a function only to close over a local value. Hide a helper by giving it a module-level `_name`.
- Use properties only for cheap, unsurprising computations, and never for a property that merely wraps an attribute.
  Use `get_` / `set_` only for costly operations or ones that invalidate state.
- Threads communicate through `queue.Queue`, otherwise `threading.Condition` and locks. Never rely on the atomicity of
  built-in types.
- No reflection as control flow (`hasattr(self, "_x")` caches: initialize the attribute in `__init__`), no `exec` or
  `eval`, and no import hacks. A test that executes a shipped notebook may `exec` it, with a pragma saying so.
- `subprocess.Popen(preexec_fn=...)` is used only to set the parent-death signal (`PR_SET_PDEATHSIG`), which
  `start_new_session` cannot do, and carries `# pylint: disable=subprocess-popen-preexec-fn  # <that reason>`.
- Build strings with f-strings, `%` or `.format`, never `+`. Accumulate with `"".join` or `io.StringIO`. An error
  message matches the actual condition, shows interpolated values recognizably and stays greppable.
- Docstrings:
  - say what callers need: side effects, mutated arguments, the behavior of a decorated function;
  - a class summary describes an instance, and public attributes go under `Attributes:`;
  - use one summary style (descriptive or imperative) per file.
- Use single-character names only for counters, `e`, `f`, `_T`, and mathematical notation from a cited paper, with a
  scoped `pylint: disable=invalid-name`.
- Past about 40 lines, consider splitting a function.
- Use `X | None`, built-in generics and `collections.abc` parameter types. Name aliases `CapWords: TypeAlias`. Use
  `TYPE_CHECKING` imports only as a last resort.

#### Frame-dependent quantities are named `<quantity>_<frame1>_rt_<frame2>_ewrt_<frame3>`

Every physical quantity that depends on frames is named by them: `rt` is "(with) respect to" and `ewrt` "expressed
with respect to", so `pos_lfoot_rt_world_ewrt_base` is the left foot's position relative to the world, in base axes.
humanoid_nmpc/docs/frame_naming/README.md is the convention: the grammar and its spelling per kind of identifier. A
new name keeps the case style of the identifier it replaces, with the same words: a camelCase C++ local, parameter,
member or function stays camelCase (`comToBasePositionNominal` becomes `nomPosBaseRtComEwrtBase`, `getBasePosition`
becomes `getPosBaseRtWorldEwrtWorld`, members keep the trailing `_`), a constant is `kNomPosBaseRtComEwrtBase`, and a
snake_case name (a proto field, Python, a C++ field that mirrors a proto) stays snake_case
(`get_pos_base_rt_world_ewrt_world`, Python constants UPPER_SNAKE). Never mix them (`getPos_base_rt_world_ewrt_world`).
The README lists the closed vocabularies of quantities (`pos`, `linvel`, `angvel`, `quat`, `rotmat`, `force`,
`wrench`, ...), frames (`world`, `base`, `com`, `heading`, `lfoot`, `contact`, ...), qualifiers (`des`, `ref`, `meas`,
`est`, `err`, ...) and suffixes, the exemptions (joint-space quantities, dimensions and indices, types, third-party
names), and the rename table. Orientations relate two frames and name no expressed-in frame (`quat_base_rt_world`); forces name the point they
act at and no relative frame (`wrench_rfoot_ewrt_world`). Where the code does not make the frames clear, read the math;
never guess, and never leave a `TODO(frame-naming)`: split a model-dependent value, add a missing word, or fix the math. The vocabulary lives in one place,
`tools/hooks/frame_naming.py`.

Enforced by the checks in `tools/hooks`:
<!-- LINT.IfChange(frame_naming_rules) -->
- an identifier with an `rt` or `ewrt` word follows the grammar with the vocabulary's words, spelled in one case style
  (never mixed; snake_case in Python and `.proto`, UPPER_SNAKE for a Python constant), and a C++, Python or `.proto`
  identifier with a legacy quantity word (`position`, `velocity`, `orientation`, `quaternion`, `rotation`,
  `acceleration`, `force`, `torque`, `wrench`, `twist`, `momentum`, `pose`, their abbreviations and plurals) and no
  exemption is a legacy name (`frame-naming`). While the rename runs, the check is opt-in: only
  `python3 -m tools.hooks.lint_code --only frame-naming --paths <dir>` runs it, and `make lint` does not until the
  rename's Verify phase deletes its `opt_in=True` (README section 9).
<!-- LINT.ThenChange(//tools/hooks/checks.py:registry, //tools/hooks/frame_naming.py:quantities) -->

#### Running the style checks

<!-- LINT.IfChange(style_commands) -->
- `make format` applies clang-format, black, isort and the rewrites that cannot change behavior: floating-point radix
  points, `=` initialization, prefix `++`, unqualified integer types and quoted include spelling (the `fix_source` of
  each enforced check that has one).
- `make fix` applies every automatic fix - the registry's `fix_source` rewrites, `make format`'s, and with
  `TIDY_CHECKS=<glob>` clang-tidy's fix-its (`make lint-tidy-fix`) - and then runs `make lint`, whose findings are the
  ones left to fix by hand. Some of its rewrites can change behavior (`american-spelling` renames identifiers), so build
  and test afterwards.
- `make lint` (`python3 -m tools.hooks.lint_code`) runs every check that needs no compiler: IFTTT, whitespace,
  clang-format, black, isort, the token checks of `tools/hooks/checks.py`, cpplint, pylint and mypy. It runs them all
  and then fails, so one run shows every finding. `--only <check>` and `--paths <dir>` narrow a run.
  - cpplint, pylint and mypy take turns through a lint lock (`.lint_machine.lock` in the checkout), because together
    they need hundreds of MB next to a build that is already sized to the machine. A second run waits and says why. Do
    not bypass it. The token checks need no lock; narrow a run with `--only` and `--paths` while you work.
- The pre-commit hook fixes, formats and lints the files the commit touches: it applies every automatic fix
  (`python3 -m tools.hooks.lint_code --fix-only --git-staged`, the behavior-changing ones included), runs `make format`,
  stages the changes of the staged files and then runs `python3 -m tools.hooks.lint_code --git-staged`. A finding left
  after that has no automatic fix and blocks the commit until it is fixed by hand. A partially staged file is never
  re-staged; when a fix or the formatter changes it, the commit stops for the user to stage the changes.
- `make lint-tidy [PKG=//humanoid_nmpc/...]` runs clang-tidy. CI runs it after the tests.
  - clang-tidy is a compiler front end. It runs only through `make lint-tidy`, a Bazel aspect inside the machine lock
    and the RAM-bounded `--jobs`.
  - The image contains a clang compiler because the clang-tidy packages depend on it. Never run `clang`, `clang++`,
    `clang-tidy`, `clang-check`, `run-clang-tidy` or a `compile_commands.json` loop by hand, and never while another
    build runs.
  - `make lint-tidy-fix PKG=... CHECKS=<glob>` applies clang-tidy's fix-its; build afterwards.
  - Each run reads its reports from a build event file of its own, so two runs at once never read each other's
    reports; pass `CLANG_TIDY_BEP=<path>` to keep the file of a run.
- Every check is enforced: no check waits in a list until its findings are fixed, `mypy.ini` has no per-module
  section, `.clang-tidy` is the only clang-tidy configuration, and every first-party C++ target compiles with
  `FIRST_PARTY_COPTS` (`tools/hooks/test_lint_enforcement.py`, `test_first_party_build_files.py`). A new check lands
  with its last finding fixed; while you fix them, narrow the runs with
  `python3 -m tools.hooks.lint_code --only <check> --paths <dir>` and `make lint-tidy PKG=...`.
<!-- LINT.ThenChange(//tools/hooks/lint_code.py:lint_lock, //Makefile:lint_tidy) -->
- To add a rule:
  - add a check to the registry in `tools/hooks/checks.py`, with its own module and `test_<module>.py`
    (`tools/hooks/README.md`), or add a check to `.clang-tidy`, `CPPLINT.cfg` or `.pylintrc`;
  - fix every finding in the same change;
  - list the rule here.

  A rule the user states is meant to be machine-checked where it can be.

# Avoiding "Claudisms"

<!-- LINT.IfChange(claudisms_rules) -->
Avoid writing using "Claudisms" at all times. Documentation, both in Markdown files and in source code, must be concise, yet technically precise such that it reads like it was written by a human for other humans to read. `make lint` checks the machine-detectable entries below (`claudisms`, `tools/hooks/claudisms.py`): em dashes, emojis, bare `---` section dividers, bold-lead bullets (`**Term**:`, `**Term:**`, `**Term.**`, `**Term** -`) in Markdown, and the banned phrases and vocabulary. A line that must keep a flagged phrase, such as the title of a cited paper, is marked `NOLINT(claudisms): <reason>`.

## Claudisms

AI models overuse the same words. This is a running catalog of them: the phrases, verbs and sentence shapes that give machine-written text away, each with a short note on why it fails and what to write instead.

### Confirmed Claudisms

- **"sit with" / "worth sitting with"** - reflective-pose filler. Sounds thoughtful; doesn't say anything.
- **"worth talking about"** - value-claim filler. Tells the reader something deserves attention instead of letting the substance demonstrate it. Same family as "worth sitting with." The broader rule it breaks: never tell people what to think of something - cool people don't need to tell people they are cool.
- **"arriving at" / "keep arriving at"** - pseudo-reflective verb. Reaches for false depth where stating the thing directly or "the question that won't leave" would work. Often pairs with "the question I keep…" - same family as "sit with." Note: "keep coming back to" was previously suggested as a replacement here; that's now banned in its own right (see "coming back to" entry below).
- **"the question that kept coming up" / "the question that keeps coming up" / "the question I keep coming back to"** - pseudo-reflective cleft. Frames a thought as a recurring question to add weight, when stating the question directly is stronger. Same family as "arriving at." Note: "What keeps coming up" as a sentence opener (no "the question" cleft) is fine. The tic is the cleft construction.
- **"coming back to" / "keep coming back to" / "keeps coming back to" / "I keep coming back to"** - pseudo-reflective verb on its own, separate from the cleft form above. Reaches for false depth in any framing - "the word I keep coming back to," "the idea I keep coming back to," etc. Same family as "arriving at" and "sit with."
- **"the thread I didn't plan but can't unsee" / "the thing I didn't plan but keep noticing" / "I didn't set out to X, but" / "I didn't plan it, but" / "what I didn't expect to find"** - discovery-arc framing. Stages a little narrative of stumbling onto an insight (didn't plan it -> can't unsee it now) to lend the observation weight it should carry on its own. Same family as "the question I keep coming back to" - both manufacture a journey to dramatize a point - and it also breaks the no-narrative-arc rule: a set of pieces isn't a story about the author discovering something. The observation should stand without the origin story. Cut the setup and state the thing: "Every one of these turns on the same pivot." not "The thread I didn't plan but can't unsee: every one of these…"
- **"the one that surprised me most" / "what struck me hardest" / "the part that stuck with me most" / "because it surprised me most" / "if I had to pick one"** - false-singularity framing. Collapses a set into a single dramatic peak (the most, the one, the hardest) to manufacture emphasis - but the ranking is usually invented, not a deliberate ranking, and it reads unlike natural speech. Same family as the totalizing-superlative entry ("the whole game"), but pointed at singularity/ranking rather than magnitude, and a cousin of the "the question I keep coming back to" cleft - both fabricate a focal point for drama. Just present the example or point plainly without crowning it: "One example." not "the example that surprised me most."
- **"the best thing you can do is" / "the one that pays off most" / "where AI helps most" / "helps most" / "the most X you can get" / "the one that [verb]"** - crowned-superlative filler. Ranks one tactic, tool, or idea as the best or highest-value in a set, which is binary thinking dressed as a recommendation and quietly demotes everything else. Same family as the false-singularity entry ("the one that surprised me most") and the totalizing-superlative entry ("the whole game") - all manufacture a single peak. Present each thing on its merits; if one genuinely comes first, give the reason, not the crown. Note: a text search only catches the literal phrases listed here - the same move with a novel noun or verb in the slot reads exactly the same and has to be caught by eye.
- **"physics" / "different physics" / "rules of physics" / "the physics of"** - pseudo-scientific metaphor reaching for grandeur. Means "how something works" or "the conditions" - say that instead. Same family as "shape."
- **"names"** (as verb, e.g., "names the move," "names the question," "names what's at stake") - pseudo-elevated diction. Sounds writerly; means "says directly" or "spells out." Use the plain verb.
- **"naming" / "naming that" / "naming it"** - the gerund form of the same tic, used as a sentence-opener to lend false weight to the act of labeling something ("Naming that changes what you expect," "Naming the phase is the first step"). Means "calling it what it is" or just "recognizing" - or cut the framing and state the consequence directly. Same family as "names" as a verb.
- **"quieter" / "louder"** (used metaphorically about writing or argument, e.g., "quieter version of the same thing," "the louder claim") - writerly metaphor that's opaque to readers. Means "less explicit" or "more direct" - say that.
- **"bumped into"** - colloquial-feeling verb that sounds casual but lands as a writerly tic when reaching for spontaneity. "Three things I ran into" is sharper than "three things I bumped into." "Ran into" is the natural choice.
- **"shape"** (as in "the shape of the experience," "shape of the time," "the shape of the trend") - used constantly. When Claude reaches for "shape" it almost always means "the way something feels," "the structure of something," or "the direction of something." Say that instead.
- **"lives"** (as metaphorical placement of abstract concepts, e.g., "the risk lives somewhere specific," "the work lives at the operator level," "the tension lives in the gap") - pseudo-anthropomorphic placement metaphor. Same family as "shape" - Claude reaches for a vivid placement verb where "is," "happens," or "shows up" would land plainer.
- **"the engine"** (as metaphor for what drives or generates value, e.g., "the challenge function is the engine," "tension is the engine of growth," "that dynamic is the engine") - mechanical metaphor reaching for compactness and motion-feel. Means "what works," "what makes companies sharp," "the productive part," or just cut the sentence and let surrounding lines do the work. Same family as "shape" and "lives" - Claude reaches for a vivid metaphor where direct statement is sharper.
- **"hits hardest" / "hits the hardest" / "lands hardest" / "lands the hardest" / "strikes hardest" / "strikes the hardest"** (as in "where commoditization hits hardest," "when this hits hardest," "the layer where AI lands hardest") - superlative intensification reaching for dramatic-impact framing. Means "is most exposed," "is most affected," "shows up first," or "is most replaceable" - say that. Same family as "the engine" and "shape" - Claude reaches for a vivid intensification verb where direct statement is sharper.
- **"surface"** (as verb) - "surface the insight," "surface a pattern." Corporate-adjacent, vague.
- **"the tell" / "that's the tell" / "the tell is"** - poker/diagnosis diction reaching for knowing-insider weight. Frames an observation as a secret giveaway the writer has cleverly spotted - same announce-your-own-cleverness move as the cool-people rule. It is not natural phrasing. Say plainly what the thing shows or means, or just state the observation: "The song is the odd one out, and that's worth looking at" -> just say why it's different.
- **"load-bearing"** - engineering metaphor that's become a tic.
- **"the point is"** - throat-clearing. Get to the point without announcing it.
- **"Here's where it gets interesting"** - lecture-hall framing. Skips ahead without earning the turn.
- **"Here's the analogy that clicked for me" / "Here's the moment that hit me" / "Here's the thing that landed" / "Here's the part that stuck"** - section-announcer plus performative-epiphany combo. Same family as "Here's where it gets interesting" - telegraphs incoming insight instead of delivering it. The "that clicked / hit / landed / stuck (for me)" suffix adds self-narration on top. Just open with the analogy/moment/thing directly.
- **"because it matters" / "this matters" / "and that matters"** - presumptuous. Tells the reader what to value before they've decided. Same family as "Here's where it gets interesting." Just say the thing without prefacing why it deserves attention.
- **"the right time" / "the right question" / "the right way" / "the right answer" / "the right tool"** - assertion-of-rightness filler. Variants on telling the reader what's correct. Same family as "this matters" - value-claim filler dressed up as conclusion. Specific instance of the cool-people rule (see "worth talking about"). Note: questions that contain "right" (e.g., a question paraphrased as "are these the right ones") are exempt - those are inquiries, not assertions.
- **"I can't stop thinking about"** - false-intimacy cliche.
- **"at the end of the day"** - corporate filler.
- **"double-click on"** - corporate-speak.
- **"unpack"** - overused reflective verb.
- **"leverage"** (as verb) - corporate-speak.
- **"lean into" / "lean out"** - overused.
- **"come along"** - too soft for definitive moments.
- **"throughline"** - buzzy and off-key.
- **"dispatches from…" / "field notes from the frontier"** - frontier-dispatch tropes. Performative.
- **"best operators" / "top practitioners"** - hierarchy language to avoid; there are many valid approaches, not a single elite tier.
- **"first wave of a multi-year transition"** - grandiose. Overclaiming.
- **"lessons learned"** - too corporate.
- **"stakes of their seat"** - retired.
- **"seat" as metaphor generally** - spent.
- **"paradigm shift"** - tired.
- **Em dashes (—)** - banned outright. This style uses dashes heavily (parentheticals, interruptions, asides), but they're regular hyphens with spaces (` - `), not em dashes. The em dash itself is the formatting Claudism. Use ` - ` instead in all drafts. Note: already-published work may contain em dashes; there's no need to go back and change it, but don't introduce new ones.
- **Emojis** - banned outright in all output: articles, newsletters, post blurbs, share text, headlines, everything. Emojis are a social-media tic to avoid and read as AI-generated filler, especially the call-to-action arrow/pointing-hand under a "read more" line. Don't add them even where LinkedIn convention invites them. The writing carries the energy; a glyph telling the reader where to look or how to feel is the same move as the cool-people rule. No exceptions.
- **"the whole game" / "that's the whole game" / "the whole ballgame" / "the only thing that matters" / "that's the entire point"** - totalizing-superlative filler. Collapses a real point into a single all-or-nothing claim - nothing is ever literally "the whole game." A specific instance of the broader aversion to absolute framing (always, never, every time): the superlative inflates one true thing into the only thing. Say what the thing actually does or unlocks and let it stand at its real size. Same family as the cool-people rule - it's announcing magnitude instead of demonstrating it. **The ban is the construction, not just these nouns:** any "the whole [X]" / "the entire [X]" / "the only [X] that matters" / "the whole point of [Y]" formulation is the same tic regardless of which noun fills the slot - "the whole lesson," "the whole story," "the whole point," "the whole job," "the whole thing." A text search only matches the literal phrases listed above; novel nouns slip past, so this one has to be caught by eye on every pass.
- **"the only thing that changed" / "the only thing that mattered" / "the only thing that happened" / "the only X that [verb]"** (the construction beyond "the only X that *matters*" already covered above) - binary-dramatization construction. Same family as the totalizing-superlative entry above and the always/never/every-time aversion. It inflates a point by claiming nothing else was involved ("the only thing that changed was how it talked") - often false on its face (other things plainly changed too), and even when literally true it crowns a single factor for drama instead of just naming it. Say what differed or what counted without the singular crown: "what differed was the tone," not "the only thing that changed was the tone." A text search won't catch novel verbs in the slot - this one has to be caught by eye.
- **"hold" / "holding"** (a thought, a line, a tension, an idea, two things at once - e.g. "holding them at the same time," "the line it holds," "hold that thought") - pseudo-weighty placement verb applied to abstractions. Reaches for gravitas by treating an idea as a physical thing gripped in the hand. Same family as "sit with," "lives," and "shape." Say the plain thing: "believing both at once," "the rule it keeps," "both are true." Note: literal physical holding is fine; the tic is the metaphorical application to thoughts/principles/tensions.
- **"carry" / "carries" / "carrying"** (an idea, an argument, a cost, a section - e.g. "that carries two costs," "the sentence carries the weight," "what I'd ask you to carry") - pseudo-weighty transport metaphor applied to things nobody picks up. Same placement-metaphor family as "hold" and "lives." Two forms, both banned: the reader-instruction ("carry this with you," "the thing to carry forward"), which just means "remember" or "keep in mind"; and the abstract-agency form ("that carries two costs," "the framing carries the argument"), which means "there are two costs" or "the argument only works because of the framing." Say the plain version. Note: literal carrying is fine; the tic is every metaphorical use.
- **"moves" / "moving"** (abstract motion - "the class's attention moves onto you," "the argument moves from one thing to the next," "the conversation moved somewhere else") - filler motion verb applied to things that do not travel. It borrows the feel of physical movement so a plain relationship sounds dynamic. Say what actually happens: "the whole class stops listening to the teacher and looks at you," "the next section is about X." Same family as "lives," "shape," and "rides along." Note: literal movement is fine - a kid moving to the back of the room, a class moving at one speed. The tic is the metaphorical one.
- **"points" / "pointing"** (as in "both costs point the same direction," "the evidence points toward," "everything is pointing at the same conclusion") - directional-gesture metaphor standing in for a plain statement of what follows. Nothing in a sentence is literally pointing anywhere, and the gesture lets the writer imply a conclusion without stating it. State the conclusion: "both of them make a kid less likely to ask." Same family as "moves" and "lives." Note: a person literally pointing, and "a point" as a noun, are fine.
- **"essays"** - always "articles." The pieces are articles, not essays.
- **"stays yours" / "stay yours" / "what stays yours" / "that part stays yours" / "the rest is yours" / "yours to [keep/own/decide]"** - ownership-retention filler. Reaches for a warm, proprietary glow by telling the reader which part still belongs to them instead of just saying what they do. Note: it turns up often in marketing and bio copy ("what it changes, what it doesn't, and what stays yours"), which is exactly why it leaks into prose - that register may use it; an article shouldn't. Same family as the cool-people rule and the placement-metaphor verbs ("hold," "carry," "lives"): it announces value/possession rather than demonstrating it. Say the plain thing - "you decide what done means," "you make the call," "you set the standard," "the review is on you." Note: literal possession ("that laptop is yours") is fine; the tic is the abstract "the judgment stays yours" register.
- **"worth asking" / "worth a look" / "worth it" / "worth the"** and any other "worth [X]" construction - the entire "worth ___" family is banned, not just the gerund forms enumerated below. If a sentence uses "worth" to tell the reader something deserves their attention/effort/consideration, cut it and let the thing demonstrate its own value. Same cool-people move every time.
- **"worth drawing" / "worth making" / "worth examining" / "worth considering" / "worth exploring"** - the "worth [gerund]" construction beyond the already-banned "worth noting," "worth talking about," "worth sitting with." All reach for the same move: telling the reader something deserves attention instead of demonstrating it. Same family, same ban.
- **"mature" / "more mature" / "mature setup" / "mature version"** - subjective grading that implies the current state is deficient without discussion. Same family as the cool-people rule - don't grade things without earning it. When describing a future or improved state, describe what it does differently, not how developed it is.
- **"sit with" / "sitting with" / "a hard thing to sit with" / "sit with that"** - reflective-pose filler (verb form, distinct from the noun "worth sitting with" already listed). Sounds thoughtful, says nothing. Replace with the plain thing: "hard to shake," "hard to get past," or just state the feeling.
- **"which brings me back to" / "brings me back to" / "that brings us back to"** - same pseudo-reflective return-framing family as "coming back to" and "the question I keep coming back to." Narrates circling back instead of just making the next point. Cut it and state the point.
- **"leave"** in the reader-direction sense - **"what I'd leave you with" / "I'll leave you with" / "leaving you with" / "what I want to leave you with"** - section-closer framing. Same family as "carry this with you." Don't announce the parting thought; just say it.
- **"in the same chest" / "in my chest" / "in the chest"** and other chest/gut/ribcage placement of emotions - pseudo-physical placement metaphor for feeling. Same family as "lives" and "hold." Just name the feeling: "both at once," "I felt both."
- **"here is where I landed" / "where I landed" / "here's where I come down"** - pseudo-reflective journey-framing. Narrates the act of reaching a conclusion instead of just stating the conclusion. Same family as "arriving at" and the section-announcer tics. Cut it and state the point.
- **"we've seen this movie before" / "we have seen this movie before" / "seen this one before"** and variations - tired cliche dressed as wry recognition. Also the historical-precedent framing more broadly: **"we have stood here before" / "we've been here before" / "we keep meeting it"** and variations. Reaches for knowing-narrator gravitas. Say the precedent plainly or just give the example.
- **"compounds" / "compound"** (in any context - "the advantage compounds," "value compounds," "it compounds over time") - finance-adjacent growth-metaphor tic. Banned in all contexts. Say "builds," "adds up," "grows," or state the actual mechanism.
- **"honest" / "honestly"** - qualifier-on-own-statements tic ("the honest version," "an honest snapshot," "an honest breakdown"). Implies the inverse - that other statements might not be honest. Operate on the assumption that everything you write is honest by default unless explicitly stated otherwise; flagging a particular claim as "honest" suspiciously implies the others aren't. Same family as the cool-people rule. Exceptions worth defending: when "honest" describes external state or a third party's behavior ("honest accounting" of someone else's reporting, "be honest with yourself") rather than qualifying your own statements.
- **"turns" / "turns on"** (as a pivot verb - "the sentence turns on the word," "it all turns on trust") - pseudo-elevated pivot verb reaching for weight where a plain verb lands harder. Means "depends on," "comes down to," or "is about." Reserve "turns" for navigation. Same borrowed-domain family as "the physics of" and "names" - a verb lifted from another field to make a plain relationship sound weightier; it also self-announces when used to point at the writer's own framing ("that sentence turns on X").
- **"compounds" / "compounding"** (metaphorical - "trust compounds," "the value compounds," "compounds into something") - finance/growth-metaphor reach for accumulation-over-time. Means "builds," "adds up," or "grows." Note: legitimate when describing real compounding, including human trust genuinely accreting over time - the tic is reaching for it as a weighty synonym for "builds." Same borrowed-domain family as "the engine" and "physics."
- **"useful" / "the useful thing" / "what's useful"** (announcing usefulness as a frame - "the useful thing she does is," "here's the useful part") - cool-people-rule violation: labels the value instead of letting the substance show it. Same family as "worth noting." Cut the label and state the thing; if it's useful the reader sees it. Note: "useful" describing an external object plainly ("a useful tool") is fine; the tic is announcing usefulness.
- **"the part that stayed mine" / "the part that stayed X"** - false-ownership / false-intimacy framing to avoid. Manufactures a sentimental "this bit was truly mine" beat over plain fact. Same family as "I can't stop thinking about." Say what you actually did or kept; don't narrate which part felt like yours.
- **"what I keep wanting people to see" / "what I want you to see" / "what I want people to take away"** - author-desire framing that tells the reader what the author wishes they would notice. Readers don't care what the author wants them to see - the substance shows it or it doesn't. Same family as "this matters" and the cool-people rule. Cut it and just make the point.
- **"hit a nerve" / "struck a nerve" / "stuck with me" / "stayed with me" / "struck a chord" / "it really landed for me"** - false emotional-resonance clichés. They borrow a generic emotional reaction instead of naming the actual one, and several ("hit a nerve," "struck a chord") smuggle in a wounded/negative connotation that may be wrong. When you react to something, say precisely what the reaction was - "interesting, relevant, and something I could act on" - not "it hit a nerve." Same family as "I can't stop thinking about" and "the part that stayed mine."
- **"wreck" / "wrecked" / "destroy" / "shatter" / "obliterate" / "blow up"** (as catastrophizing verbs for a measured effect) - over-dramatization. Match the verb to what actually happened: a one-point drop on a seven-point scale "made the work worse," it didn't "wreck" it. Note: literal destruction is fine; the tic is inflating a real-but-bounded effect for impact.
- **"real" / "a real X" / "the real thing" / "the real problem" / "real data"** - banned in nearly all contexts. "Real" is best avoided in almost any context. It's binary - its opposite, "fake," carries insult and can disparage perfectly legitimate things (calling one approach "real" implies the others aren't, e.g. saying a study's design was "the real way to prove it" slights everything else). It also usually adds nothing. Say the specific thing instead: "measured," "actual," "the deeper point," "the work itself," or just drop it. Note: "really" as an intensifier is a different word and not the target; the ban is the adjective "real."
- **"the thing that got me" / "what got me was" / "the thing that got to me" / "the thing that hooked me"** - fabricated-singularity plus invented-reaction. Two tics in one: it crowns a single item ("the thing") when there's rarely just one, and "got me" invents an emotional reaction the writer never stated. Same family as "the question I keep coming back to" (manufactured focal point) and "hit a nerve" / "stuck with me" (borrowed generic reaction). Say plainly what the reaction actually was - liked it, appreciated it, was pleasantly surprised by it - and don't anoint one item as the singular thing.
- **"reaching for" / "reach for" / "reaches for" / "reaching"** (metaphorical - grasping for an effect, a word, weight, gravitas, false depth, e.g. "reaching for gravitas," "reaches for a vivid metaphor") - grasping-metaphor tic. Ironically the exact verb this very list leans on over and over to describe other tics - fine inside these editor notes, but in article prose it's the same pseudo-writerly grasp it names. Say the plain thing: "wants," "goes for," "tries for," "grabs at," or just state what's happening. Note: literal reaching (for an object) and ordinary senses ("audience reach," "reach out," "reach parity," "reach a conclusion") are fine; the tic is the metaphorical grasp-for-an-effect.
- **"doing the work" / "the word was doing the work" / "X is doing the work here" / "doing the heavy lifting" / "carrying the argument" / "holding the sentence up" / "the noun is doing all the lifting"** - abstract-agency metaphor: a word, sentence, framing, or idea is personified as an employee performing labor. Reads as writerly shorthand and says nothing a reader can picture. The construction is a cousin of "lives," "holds," "carries," and "the engine" (vivid placement/agency verb standing in for a plain statement), but it earns its own entry because it gets used constantly when writing about language, arguments, or framing. Say what actually happened in plain terms: "the sentence only sounded smart because of the word," "take the word out and there's nothing left." The test: could a reader draw what the sentence claims is happening? If a word is "doing work," no.
- **"most people" / "most people I've talked to" / "the people I've watched" / "everyone I've worked with" / "in my experience, most teams" / "a lot of folks" / "nobody I know"** - invented-observation framing, the writerly version of "people say." Two failures stacked: it attributes a claim to a vague mass that can't be checked, and it fabricates a first-person history - watching people, asking people, noticing a pattern across people - that was never described. Unless the observation was actually made or is in the source material, it does not exist and can't be salvaged by softening it ("many people," "a lot of the time"). Cut the sentence, or make the point without a population behind it. Same failure as any unsourced first-person specific: if it didn't happen, it can't go in.
- **"quietly"** (as an adverb implying either an unnoticed, creeping change - "removes that step quietly," "shifts quietly," "erodes quietly" - or an understated deliberate act - "what they quietly refuse to use," "quietly dropped support") - reaches for a hidden-drama effect. In the first sense it asserts that nobody noticed, without evidence that anyone failed to. In the second it credits an ordinary decision with a discretion nobody claimed, and flatters the writer for having spotted it. Same family as "shape" and "lives" - a vivid modifier standing in for a plain statement of what happened. Say what actually happened, or cut the adverb.
- **"settled" / "isn't settled" / "not settled yet" / "nobody has settled" / "unsettled" / "a settled question" / "a question we thought was settled" / "reopens a settled question" / "back into an open one"** - banned as a word, in every form. Two moves hide behind it, and both manufacture drama. The first claims a prior agreement has come undone; the agreement is never stated, so the reversal is invented. The second turns up constantly in AI writing about AI - asserting that a question is open and nobody has worked it out yet ("nobody has settled this," "it isn't settled," "that part is still unsettled"). It flatters the writer for standing at the edge of a big unresolved thing and tells the reader how unresolved to feel. Say the plain version: "it raises a new question," "we have not worked out X yet," "people disagree about X." Same family as the cool-people rule - it announces significance instead of demonstrating it.
- **"rides along" / "rides on" / "riding along"** (metaphorical, e.g. "a summary rides along," "the risk rides on the decision") - filler placement/motion verb dressing up "comes with" or "is attached." Say the plain thing: "with a summary attached," "comes with it." Note: literal riding is fine; the tic is the metaphorical "X rides along with Y."
- **"hands" / "hands you" / "hands it back"** - (as in "the moment the AI hands you something," "what the model hands back") - borrowed physical-transfer verb standing in for a plain one. It stages the output as an object passed across a table, which adds a little scene where "gives you," "produces," "returns," or "comes back with" says it flatter and truer. Same family as "rides along," "lives," and "shape" - a vivid verb lifted in to make an ordinary relationship feel weightier. Literal handing is fine; the tic is the metaphorical one.
- **"nobody handed" / "nobody handed you that" / "nobody handed it to you" / "no one handed you" / "nobody gave you that"** - false-origin credit framing. Denies help that nobody claimed, so the achievement sounds bigger for having had none. It invents an alternative version of events for the only purpose of knocking it down, and it flatters the person in the same breath. Same family as the discovery-arc framing ("I didn't set out to X, but") and "the part that stayed mine." Say what the person actually built and let the work show where it came from.
- **"hand-waving" / "hand-wave it" / "a wave of the hand" / "waves it away"** - dismissal-by-gesture metaphor, and it hides the work. Saying someone hand-waves an objection, or gets past it "with a wave of the hand," borrows a physical gesture instead of saying what was skipped and why skipping it was wrong. Every form is the same tic: "hand-waving," "hand-waved it," "a wave of the hand," "waves the question away." State what went unanswered - "they never say how," "the strongest argument against it goes unanswered," "the number is asserted and never sourced." Same borrowed-gesture family as "points" and "names," and it breaks drama-from-facts, since the image stands in for the consequence. Note: a person literally waving is fine.

### Structural / framing tics

- **Announcing the structure before delivering it** - "I'm going to make three points" instead of just making them. Trust the reader to follow.
- **Judging what's interesting or not** - "the most interesting part," "the least interesting thing." Haughty. Let the reader decide.
- **Hyping / over-dramatizing** - reaching for stakes, metaphors, or cosmic framing when plain is stronger.
- **False-modest asides** - "I want to be careful here," "with appropriate caveats." Usually signals lean-out, not care.
- **Stacked dashes in a single sentence** - rhythmic tic. This style uses regular hyphens with spaces for parentheticals and pivots; more than one in a single sentence reads as a tic. (Em dashes themselves are banned outright - see the Confirmed Claudisms entry.)
- **Stacked staccato sentences** - short declaratives in a row of four or more without variation. Short sentences work as punctuation; they need longer sentences around them to land. Mix lengths.
- **Ending with a "one-liner close" that restates the thesis** - over-formula. Sometimes the piece just ends.
- **Vague stakes-gesturing** - "the question is coming," "the conversation is going to happen," "the reckoning will come." Deferred-reference without naming the question/conversation/reckoning. The reader can't grip stakes that the article keeps gesturing at. Always name the specific question or event being deferred to.
- **Visibility framings where stakes-framings are stronger** - "the line will be harder to ignore," "the gap will become more visible," "the difference will be clearer." When describing how AI changes a landscape, outcome-language (who survives, what dies, what builds up) carries weight that visibility-language only points at. Use outcome language when stakes are real.
- **"what I keep running into" / "what I keep noticing" / "where I keep landing"** - pseudo-reflective cleft, the same move as "the question I keep coming back to" with a different verb dropped in. Framing an observation as one that recurs adds weight the observation should carry by itself, and it stages a history of noticing that was never described. The verb is not the problem - "I ran into a wall" is fine, and the "bumped into" entry endorses "ran into" as the plain choice - the cleft is. Cut the frame and state the thing: "There's no shortcut to production quality" not "What I keep running into is that there's no shortcut to production quality." Same family as "arriving at," "coming back to," and "the thread I didn't plan but can't unsee."
- **"what moved is" / "what actually moved" / "what changed is the"** - (as in "what moved is the assembly," "what actually moved is the timeline") - a cleft that announces a summary verdict on a change, using a physical-motion verb for something that did not physically move. Two faults in one: it borrows weight from a movement image the sentence has not earned, and it compresses a many-part change into a single crowned item, which is almost always narrower than the truth. Name the change plainly and at its real size - "you no longer start at a blank page, and it keeps you inside the method while you work" - rather than crowning one part of it as the thing that moved. Same family as "the shift is," "the unlock is," and the crowned-superlative entries.

### Register tics to avoid

- **Finance-bro shorthand** - "$X," "the forward case," "price-to-value ratio." Say what you mean in English.
- **Consultant phrasing** - "pressure-test," "right-size," "north star," "true north," "strategic imperative."
- **McKinsey-adjacent structure** - "three things to know," "key takeaways," neat bullet summaries at the end.

### Spoken-word tells

- **Cross-voice echo** - two speakers stating the same substantive point back to back in different words ("So nothing here is theoretical." / "Very little of it is theoretical."). It reads as emphasis to the writer and as padding to the listener. One idea, one speaker, once. Short backchannels ("Right.") are fine; a rephrase that adds nothing is not.
- **Sprinkled disfluency** - seeding "um" / "uh" / "you know" / "right?" evenly across turns, especially short ones. Real fillers cluster at high-planning moments and are prosodically reduced; a typed one gets rendered at full energy, so it lands about twice as loud as a real one. A filler only belongs mid-passage in a turn of three-plus sentences or one long complex sentence. Never on a one-liner, a question, or a backchannel. Fewer is always better.
- **Invented human biography for a synthetic speaker** - an AI voice claiming a career ("I ran engineering organizations for twenty years"). Not a style problem, an honesty problem. Synthetic narrators disclose what they are, then may play a named role openly. First-person war stories get reframed as observations about the evidence.
- **Crowned superlatives, spoken form** - "my favorite detail in the entire pile," "the most interesting thing about this," "the best record on the panel." Same ban as in prose, and more conspicuous aloud because there is no page to skim past it. If order matters, give the reason, not the crown.
- **"the whole [X]" / "the entire [X]"** - "the whole point," "the entire diagnosis," "the whole metric." An intensifier doing no work. Cut it.
- **"the honest version is" / "honestly" (as a self-qualifier)** - "the honest version is..." / "honestly." Fine when describing third-party data; a tell when qualifying your own statement, because it implies the other statements weren't.

### Imported from external AI-writing references

Cross-validated against Will Francis's "How to Stop Claude Writing Like an AI" (willfrancis.com, March 2026), Wikipedia's "Signs of AI writing" page, and the Washington Post analysis of 328K ChatGPT messages. Items below are widely-flagged AI tells that weren't already in this banlist. Some are diction a careful writer would catch instinctively - listed here as defaults to avoid.

#### Vocabulary

- **"underscore" / "underscoring the importance of"** - pseudo-emphasis. Among the most-cited AI tics after "delve."
- **"foster"** - corporate-warm verb for "support" or "help."
- **"shed light on"** - pseudo-illuminative cliché.
- **"pave the way"** - cliché.
- **"pivotal" / "plays a pivotal role"** - empty intensifier.
- **"transformative" / "game-changing" / "groundbreaking" / "cutting-edge"** - hype adjectives. Empty.
- **"robust" / "seamless" / "intricate" / "comprehensive" / "holistic"** - corporate-tech-press adjectives. Look substantive, say nothing.
- **"navigate"** (figurative, e.g., "navigate the landscape") - AI tic.
- **"realm"** - fancy filler for "area" or "field."
- **"landscape"** (figurative) - cosmic-framing emptiness.
- **"testament" / "is a testament to"** - overused.
- **"delve" / "dive into"** - canonical AI tic since 2023.
- **"harness"** - pseudo-active corporate verb.

#### Phrases

- **"It's worth noting that" / "It's important to note that"** - throat-clearing.
- **"When it comes to…"** - soft opener that delays the point.
- **"At its core…"** - pseudo-fundamental framing.
- **"This is where X comes in"** - lecture-hall framing.
- **"Let's break it down"** - same family.
- **"It cannot be overstated"** - overclaim.
- **"Reflecting a broader trend toward" / "Marking a significant shift in"** - AI summary-speak.
- **"Great question" / "I love that you asked"** - preamble flattery.

#### Structural tics

- **`---` horizontal rules as section dividers** - AI tic. Multi-section articles use headers, not horizontal rules. A bare divider is invisible structure that leaves the reader without orientation. Banned in all article drafts.
- **Negative parallelism / cleft contrasts** ("It's not just X, it's Y," "Not only X, but Y," "This isn't about X. It's about Y.") - the most-cited AI tell anywhere. Wikipedia, Francis, and the WaPo analysis (6% of July 2025 ChatGPT messages) all flag this as the most recognizable AI structure. Banned outright. Note: this is a broader family than the "the question that…" cleft already in the list.
- **"No X. No Y. Just Z." staccato negation** - sub-form of the above.
- **"Bold term: explanation sentence" list format** - Francis flags this as the single most recognizable AI list pattern. Avoid even when bullets are appropriate.
- **Signposting transitions** ("Let's explore," "Now let's turn to," "Moving on to") - explicit examples extending the existing "announcing the structure" rule.
- **Restating the question before answering** - AI tic. Just answer.
- **Sweeping contextual opener** ("In today's rapidly evolving X…") - already cut in practice; explicit ban here.

### Usage

When drafting, check against this list before producing. If a candidate phrase appears here, rewrite in plainer English before showing the draft. This list grows - add new entries as they're flagged.

*A living list of AI-writing tells. Free to copy and adapt.*
<!-- LINT.ThenChange(//tools/hooks/checks.py:registry, //tools/hooks/claudisms.py:rules) -->
