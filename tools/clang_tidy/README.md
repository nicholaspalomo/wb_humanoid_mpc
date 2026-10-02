# clang-tidy as a Bazel aspect

clang-tidy parses C++ the way a compiler does, and a Pinocchio or CppAD translation unit costs it 1 to 2.5 GB. So it
runs only as Bazel actions, inside `.bazelrc`'s RAM-bounded `--jobs` and `tools/bazel`'s machine lock (AGENTS.md,
"Builds share one machine's memory"). Never run `clang-tidy`, `clang`, `run-clang-tidy` or a `compile_commands.json`
loop by hand.

| Command | Configuration | What it does |
|---|---|---|
| `make lint-tidy [PKG=//humanoid_nmpc/...]` | `//:.clang-tidy`, the enforced checks | Lints, runs the self-test, prints the findings and fails on any. CI's build job runs it after the tests. |
| `make lint-tidy-sweep [PKG=...] [PATHS="dir ..."] [SUMMARY=1]` | `sweep.clang-tidy`, every candidate check | The same with the checks not yet enforced. `PATHS` keeps the findings in those directories, and `SUMMARY=1` adds counts per check and per package. |
| `make lint-tidy-fix CHECKS='<glob>' [PKG=...] [PATHS=...]` | `sweep.clang-tidy` | Applies the fix-its of the matching checks and clang-formats the edited files. Build afterwards. |

clang-tidy 21 comes from apt.llvm.org (`docker/install_llvm_tools.sh tidy`); the dev image and CI's build job install it
with that script. `run_clang_tidy.sh` pins the major version, so a different clang-tidy is an error that says to rebuild
the dev container.

The rules are in `AGENTS.md` ("Style guides" and the sections after it), and each check in the configurations names the
Google C++ Style Guide section (`G:`) or the Abseil tip (`ToTW #`) it enforces. The token-level checks that need no
compiler are in `tools/hooks` (`tools/hooks/README.md`).

**Cost.** A cold run over `//...` is about 850 actions: on a 32 GB workstation (7 jobs) about 16 minutes with
`.clang-tidy` and 32 with `sweep.clang-tidy`, about twice that on CI's 4 jobs. The results are cached like any action,
so a re-run lints only the files whose inputs changed, and an unchanged tree takes seconds. A configuration file is an
input of every action of its aspect: any edit to `.clang-tidy` or `sweep.clang-tidy`, a comment included, re-lints the
whole tree.

## How it works

`clang_tidy.bzl` defines two aspects that differ only in their configuration file: `clang_tidy_aspect` and
`clang_tidy_sweep_aspect`. Switching between them, or adding either to a build, keeps the build configuration, so the
analysis cache and the generated headers are shared with ordinary builds.

The aspect does not propagate: `//...` already names every first-party target, and no external repository is
visited. For each `cc_library`, `cc_binary` and `cc_test` outside `lib/`, `tools/ifttt-lint/` and `testdata/` it runs
one action per source file.

- **Translation units.** Each `.cpp` is a translation unit. Its header filter shows the diagnostics of every first-party
  header (`FIRST_PARTY_HEADER_FILTER`) except the generated ones checked into the tree (`GENERATED_FILES`). Some checks
  locate a finding in a header from a `.cpp`'s context: `readability-identifier-naming` (at the declaration, with the
  fix-its of every use), `modernize-use-default-member-init` and `readability-inconsistent-declaration-parameter-name`.
- **Headers.** Each header is a main file of its own (`-x c++-header`) and shows only its own diagnostics, which also
  proves that it is self-contained.
- **Generated code.** Generated `.pb.h` and `.nproto.h` files are inputs, never linted. Their generators are held to
  the rules instead.

The compile flags are the build's, with these changes:

- The toolchain's C++ command line, the `--copt` / `--cxxopt` flags (`-std=c++20`), the rule's `copts`, defines and
  include directories are kept.
- Every GCC warning flag (`-W*`, `-pedantic*`) and the GCC-only flags are dropped. The diagnostics come from the
  configuration; a `clang-diagnostic-<name>` check gets its `-W<name>` from `CLANG_DIAGNOSTIC_FLAGS`.
- `--gcc-install-dir` points clang at the build's libstdc++, instead of the toolchain's built-in include directories,
  which would put GCC's own `stddef.h` in front of clang's.
- First-party include directories are always `-I`, because clang-tidy drops a system header's diagnostics.

Headers keep their source paths, so that the filter matches them. First-party BUILD files therefore use
`includes = [...]` and never `strip_include_prefix` / `include_prefix`; `tools/hooks/test_first_party_build_files.py`
checks this.

**Reports.** Each action writes its findings (output group `clang_tidy`) and its fix-its (`clang_tidy_fixes`). It
succeeds either way, so that Bazel caches it and the next run lints only what changed. The action fails only when
clang-tidy itself fails. A compile error that clang reports and GCC accepted is a finding (`clang-diagnostic-error`):
fix the code, it is a portability bug.

`clang_tidy_report.py` reads the reports of exactly this build from the build event file. It makes the paths
repository-relative and prints each finding once: a header's finding is reported by the header's own action and by
every `.cpp` that includes it.

**Applying fix-its.** `clang_tidy_apply.py` deduplicates replacements and applies each diagnostic all or nothing:

- it never edits a generated file or an external one;
- it applies no replacement of a diagnostic that some translation unit could not fix (the renamer's "inside a macro");
- with `--paths`, it applies no replacement that would edit a file outside the given directories;
- it skips a diagnostic whose edit overlaps another's.

It lists every diagnostic it left alone, for a manual fix.

## The rollout

`sweep.clang-tidy` lists every candidate check. `//:.clang-tidy` lists the ones with no finding left in the tree, with
the same options. A check moves from the sweep to `.clang-tidy` in the change that fixes its last finding. When none is
left, `sweep.clang-tidy`, `clang_tidy_sweep_aspect`, the `clang-tidy-sweep` configuration and `make lint-tidy-sweep`
are deleted, and `clang-tidy-fix` uses `clang_tidy_aspect`.

`test_clang_tidy_config.py` keeps the checks that contradict this repository's rules out of both files, for example
`modernize-use-auto` and `readability-implicit-bool-conversion`.

To silence one finding, write `// NOLINT(<check>): <reason>` on its line, or `NOLINTNEXTLINE(<check>): <reason>` on the
line above. A marker without a reason is itself a lint error (`tools/hooks/nolint.py`).

## Tests

| Test | Needs clang-tidy | Runs in |
|---|---|---|
| `test_clang_tidy_config.py` | no | `bazel test //...` |
| `test_clang_tidy_aspect.py` | no | `bazel test //...` |
| `test_clang_tidy_report.py` | no | `bazel test //...` |
| `//tools/clang_tidy:selftest` | yes | `make lint-tidy` |

- **`test_clang_tidy_config.py`** checks the configurations: each check is listed once, sorted and with a comment;
  findings are errors; the forbidden checks are absent; the enforced checks are a subset of the sweep with the same
  options. It also checks that the header filter covers every directory with first-party C++ and nothing else.
- **`test_clang_tidy_aspect.py`** reads the arguments the aspect writes for each fixture file (the `clang_tidy_args`
  rule) and pins the command line.
- **`test_clang_tidy_report.py`** tests the collector and the fix applier on synthetic reports, fix-its and build event
  files.
- **`//tools/clang_tidy:selftest`** is `manual`. It lints `testdata/` with every candidate check. It fails unless each
  fixture's deduplicated findings are its `<name>.expected` file, unless the rename fixture's fix-its give the golden
  copy `testdata/renamed/`, and unless `clang-tidy --verify-config` accepts both configurations. `testdata/renamed/` is
  a `cc_library` in `//...`, so every build proves that the applied fixes compile.

Every linter skips `testdata/`: the aspect outside the self-test (`excluded_packages`), and cpplint and the token checks
of `tools/hooks` (`lint_files.FIXTURE_DIRS`).
