# Coding Agent Style Guidelines & Rule Directives

#### Co-dependent changes: use IFTTT directives (`LINT.IfChange` / `LINT.ThenChange`)

When code or configuration in one place must stay in sync with code elsewhere — but DRY cannot eliminate the duplication (for example, package registrations, launch targets, system dependencies, or Bazel repository rules) — mark the dependency with `LINT.IfChange` / `LINT.ThenChange` directives so changes to one side prompt review of the other. Add these directives proactively when creating new co-dependent content, not just when maintaining existing pairs.

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

Separate containers each size their builds to the whole machine, so `tools/bazel` (which Bazelisk runs in place of
Bazel in the dev container, in CI and in `make ci-local`) makes the `build`, `test` and `coverage` commands of every
container take turns through a lock file in the shared checkout. A second build waits and says why. Do not bypass it
(`BAZELISK_SKIP_WRAPPER`), and keep that script POSIX `sh`: the dev container's `BASH_ENV` makes every bash script
source the shell setup first, and a bash wrapper around Bazel once recursed through it until the container ran out of
memory.

#### American English throughout

Write American spelling everywhere in the repository: identifiers, comments, strings, log messages, YAML, Python and
documentation - color, behavior, center, meter, initialize, normalize, analyze, modeling, labeled, canceled, defense,
gray, program. `make lint` checks it (`tools/hooks/american_spelling.py`; `python3 -m tools.hooks.american_spelling
--fix <files>` fixes them); a line that must keep a British spelling, such as the title of a cited paper, is marked
`NOLINT(american-spelling): <reason>`. Vendored code under `lib/` and robot model files are left as they come.

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
`tools/hooks/lint_files.py` and skipped. Every C++ and Python file starts with the BSD-3 license comment block
(cpplint `legal/copyright`, `py-license-header`). A new file's holder line is
`Copyright (c) <year>, Nicholas Palomo. All rights reserved.`; an existing holder line is kept.

A rule that a tool checks is a hard failure: there is no baseline file and no grandfathered finding. Fix the finding.
Where a rule is genuinely wrong for one line, say why on that line: `// NOLINT(<check>): <reason>` (`#` in Python,
shell and Starlark; or `NOLINTNEXTLINE`, or a `NOLINTBEGIN` / `NOLINTEND` pair around a block), and for pylint's own
messages `# pylint: disable=<message>  # <reason>`. A marker without a check name or without a reason fails the lint,
and so does a marker that names no known check, or one for the repository's own checks that suppresses nothing. The
agent rules below are the ones no tool can check. They apply to new code and to the code a change touches; they are not
a reason to restyle unrelated code in the same change.

#### C++: where this repository departs from the Google C++ Style Guide

<!-- LINT.IfChange(cpp_style) -->
- **Line length** is 140 columns (`.clang-format`), not 80.
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
  resort. A literal that the function name already explains (`std::pow(x, 2)`, `v.resize(3)`) needs none.
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
  channel may throw, and a call into a library that throws (yaml-cpp, cppzmq, OCS2's loaders and solvers) is wrapped in
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
  package-local flag list. A `-Wno-...` needs a reason on its line; a flag that a third-party header trips is dropped
  from the constant, never silenced per file.
<!-- LINT.ThenChange(//tools/hooks/checks.py:registry, //.clang-tidy:checks, //bazel/copts.bzl:first_party_copts) -->

Enforced by:
- clang-format (layout, include order);
- cpplint (`CPPLINT.cfg`);
- clang-tidy (`.clang-tidy`, run by `make lint-tidy`);
- the checks in `tools/hooks`, which cover:
  - floating-point literals with a radix point;
  - prefix `++`;
  - no `auto` and no class template argument deduction;
  - exceptions and RTTI only at their boundaries;
  - header macros;
  - static storage;
  - quoted non-system includes;
  - class comments;
  - braces around `if ... else`;
  - the `TODO` format;
  - inclusive language;
  - the forbidden constructs (`long double`, user-defined literals, inline namespaces, `decltype(auto)`, coroutines,
    modules, `<ratio>`, `<cfenv>`, `alloca`, GNU extensions);
- the GCC warnings of `bazel/copts.bzl`.

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
- Overload only when every overload has the same meaning, and document the set with one comment. Default arguments go
  only on non-virtual functions, with constant values.
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
- Move constructors are `noexcept`. The `const` methods of a class are safe to call concurrently, or the class says it
  is not thread-safe.
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

GCC expands the macros to nothing, so the build checks nothing. `tools/hooks/pointer_nullability.py` checks that every
pointer is annotated, and clang-tidy (`make lint-tidy`, clang 21) sees the annotations: clang reports a misplaced one,
`nullptr` passed to or returned from an `absl_nonnull` pointer (`clang-diagnostic-nonnull`), and a redeclaration that
contradicts the first (`clang-diagnostic-nullability`). Neither can tell whether the choice is right, and a wrong one is
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
`ABSL_DIE_IF_NULL(p)`. Never use `ABSL_DCHECK`: every build is `-c opt`. Never check per tick in the realtime loop;
establish the invariant at construction. A C-API constructor that can fail reports an `absl::Status`. Escape:
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
- Take read-only sequences as `absl::Span<const T>` (Eigen: `Eigen::Ref<const ...>`) by value (#93).
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
- **Line length.** black formats code at its default 88 columns. Comments, docstrings and strings may run to 140
  (pylint). The guide's 80 is not used. Indentation is 4 spaces, as in Google's published pylintrc.
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
- black and isort (`make format`);
- pylint (`.pylintrc`; Google's pylintrc plus the docstring and typing extensions);
- mypy;
- the checks in `tools/hooks`, which cover:
  - module-only, absolute and unaliased imports (aliases only from the standard list: `np`, `tk`, `rr`, ...) and no
    `sys.path` changes;
  - the license header;
  - the docstring summary line, its `Args:` / `Returns:` / `Yields:` sections, and property docstrings;
  - no `@staticmethod`;
  - comprehensions with one `for` and one condition at most;
  - one-line lambdas and conditional expressions;
  - no `len(x) == 0`;
  - no `assert` outside tests;
  - exception class names ending in `Error`;
  - no `__del__`, metaclasses, `exec` or `eval` outside tests;
  - no backslash continuations or `# type:` comments;
  - a shebang exactly on executable files;
  - a `main()` called from the `__main__` guard;
  - the `TODO` format.

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

#### Running the style checks

<!-- LINT.IfChange(style_commands) -->
- `make format` applies clang-format, black, isort and the rewrites that cannot change behavior: floating-point radix
  points, `=` initialization, prefix `++`, unqualified integer types and quoted include spelling (the `fix_source` of
  each enforced check that has one).
- `make lint` (`python3 -m tools.hooks.lint_code`) runs every check that needs no compiler: IFTTT, whitespace,
  clang-format, black, isort, the token checks of `tools/hooks/checks.py`, cpplint, pylint and mypy. It runs them all
  and then fails, so one run shows every finding. `--only <check>` and `--paths <dir>` narrow a run.
  - cpplint, pylint and mypy take turns through a lint lock (`.lint_machine.lock` in the checkout), because together
    they need hundreds of MB next to a build that is already sized to the machine. A second run waits and says why. Do
    not bypass it. The token checks need no lock; narrow a run with `--only` and `--paths` while you work.
- The pre-commit hook runs `make format` and then `python3 -m tools.hooks.lint_code --git-staged` on the files the
  commit touches.
- `make lint-tidy [PKG=//humanoid_nmpc/...]` runs clang-tidy. CI runs it after the tests.
  - clang-tidy is a compiler front end. It runs only through `make lint-tidy`, a Bazel aspect inside the machine lock
    and the RAM-bounded `--jobs`.
  - The image contains a clang compiler because the clang-tidy packages depend on it. Never run `clang`, `clang++`,
    `clang-tidy`, `clang-check`, `run-clang-tidy` or a `compile_commands.json` loop by hand, and never while another
    build runs.
  - `make lint-tidy-fix PKG=... CHECKS=<glob>` applies clang-tidy's fix-its; build afterwards.
- While the repository is being swept, a check that still has findings is pending: its name is in
  `tools/hooks/checks.py:PENDING`, its cpplint category or pylint message in the `SWEEP` block of `CPPLINT.cfg` or
  `.pylintrc`, its relaxed module in the `SWEEP` block of `mypy.ini`, or its clang-tidy check only in
  `tools/clang_tidy/sweep.clang-tidy`; and targets not yet switched keep their package's old copts. `make lint` and
  `make lint-tidy` skip pending checks, but the rules above apply to new code all the same: run
  `python3 -m tools.hooks.lint_code --only <check> --paths <dir>` and `make lint-tidy-sweep PKG=... PATHS=<dir>` on what
  you touch. A check leaves the pending set in the change that fixes its last finding, and the mechanism goes when none
  is left.
<!-- LINT.ThenChange(//tools/hooks/lint_code.py:lint_lock, //Makefile:lint_tidy) -->
- To add a rule:
  - add a check to the registry in `tools/hooks/checks.py`, with its own module and `test_<module>.py`
    (`tools/hooks/README.md`), or add a check to `.clang-tidy`, `CPPLINT.cfg` or `.pylintrc`;
  - fix every finding in the same change;
  - list the rule here.

  A rule the user states is meant to be machine-checked where it can be.
