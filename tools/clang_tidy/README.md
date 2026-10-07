# clang-tidy as a Bazel aspect

clang-tidy parses C++ the way a compiler does, and a Pinocchio or CppAD translation unit costs it 1 to 2.5 GB. So it
runs only as Bazel actions, inside `.bazelrc`'s RAM-bounded `--jobs` and `tools/bazel`'s machine lock (AGENTS.md,
"Builds share one machine's memory"). Never run `clang-tidy`, `clang`, `run-clang-tidy` or a `compile_commands.json`
loop by hand.

| Command | What it does |
|---|---|
| `make lint-tidy [PKG=//humanoid_nmpc/...]` | Lints with the checks of `//:.clang-tidy`, runs the self-test, prints the findings and fails on any. CI's build job runs it after the tests. |
| `make lint-tidy-fix CHECKS='<glob>' [PKG=...] [PATHS=...]` | Applies the fix-its of the matching checks and clang-formats the edited files. Build afterwards. |

`.clang-tidy` is the only configuration, and every check in it is enforced: there is no list of checks still being
adopted and no baseline of findings (`tools/hooks/test_lint_enforcement.py`). To see where the findings of a run are,
keep its build event file and ask the collector for counts, or for the findings of some directories only:

```bash
make lint-tidy PKG=//humanoid_nmpc/humanoid_wb_mpc/... CLANG_TIDY_BEP=.bazel/wb_mpc_tidy.json
python3 -m tools.clang_tidy.clang_tidy_report .bazel/wb_mpc_tidy.json --summary --paths humanoid_nmpc/humanoid_wb_mpc
```

clang-tidy 21 comes from apt.llvm.org (`docker/install_llvm_tools.sh tidy`); the dev image and CI's build job install it
with that script. `run_clang_tidy.sh` pins the major version, so a different clang-tidy is an error that says to rebuild
the dev container.

The rules are in `AGENTS.md` ("Style guides" and the sections after it), and each check in the configuration names the
Google C++ Style Guide section (`G:`) or the Abseil tip (`ToTW #`) it enforces. The token-level checks that need no
compiler are in `tools/hooks` (`tools/hooks/README.md`).

**Cost.** A cold run over `//...` is about 875 actions: on a 32 GB workstation (7 jobs) about 33 minutes, and about
twice that on CI's 4 jobs. The results are cached like any action, so a re-run lints only the files whose inputs
changed, and an unchanged tree takes seconds. The configuration file is an input of every action of the aspect: any
edit to `.clang-tidy`, a comment included, re-lints the whole tree.

## How it works

`clang_tidy.bzl` defines one aspect, `clang_tidy_aspect`, which reads `//:.clang-tidy`. Adding it to a build keeps the
build configuration, so the analysis cache and the generated headers are shared with ordinary builds. `make lint-tidy`
builds its `clang_tidy` output group (`--config=clang-tidy`), and `make lint-tidy-fix` also its fix-its
(`--config=clang-tidy-fix`).

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
every `.cpp` that includes it. The report runs after the build has released the machine lock, so each `make` run
writes its own build event file (`.bazel/clang_tidy_bep.<pid>.json`, deleted afterwards) and a run that starts
meanwhile cannot overwrite it; `CLANG_TIDY_BEP=<path>` keeps the file (`test_lint_tidy_targets.py`).

**Applying fix-its.** `clang_tidy_apply.py` deduplicates replacements and applies each diagnostic all or nothing:

- it never edits a generated file or an external one;
- it applies no replacement of a diagnostic that some translation unit could not fix (the renamer's "inside a macro");
- with `--paths`, it applies no replacement that would edit a file outside the given directories;
- it skips a diagnostic whose edit overlaps another's.

It lists every diagnostic it left alone, for a manual fix.

## Adding a check

Add the check to `.clang-tidy` with the guide section or tip it enforces, fix every finding it reports in the same
change (`make lint-tidy-fix CHECKS=<check>` where it has fix-its), and list it in `AGENTS.md`. A check that cannot be
enabled with its findings fixed stays out of the file, with a comment that says why (`misc-include-cleaner` is one).
`test_clang_tidy_config.py` keeps the checks that contradict this repository's rules out, for example
`modernize-use-auto` and `readability-implicit-bool-conversion`.

To silence one finding, write `// NOLINT(<check>): <reason>` on its line, or `NOLINTNEXTLINE(<check>): <reason>` on the
line above. A marker without a reason is itself a lint error (`tools/hooks/nolint.py`).

## Tests

| Test | Needs clang-tidy | Runs in |
|---|---|---|
| `test_clang_tidy_config.py` | no | `bazel test //...` |
| `test_clang_tidy_aspect.py` | no | `bazel test //...` |
| `test_clang_tidy_report.py` | no | `bazel test //...` |
| `test_lint_tidy_targets.py` | no | `bazel test //...` |
| `//tools/clang_tidy:selftest` | yes | `make lint-tidy` |

- **`test_clang_tidy_config.py`** checks the configuration: each check is listed once, sorted and with a comment;
  findings are errors; the forbidden checks are absent. It also checks that the header filter covers every directory
  with first-party C++ and nothing else.
- **`test_clang_tidy_aspect.py`** reads the arguments the aspect writes for each fixture file (the `clang_tidy_args`
  rule) and pins the command line.
- **`test_clang_tidy_report.py`** tests the collector and the fix applier on synthetic reports, fix-its and build event
  files.
- **`test_lint_tidy_targets.py`** runs the Makefile's `lint-tidy` and `lint-tidy-fix` recipes with stand-ins for Bazel
  and the collector: each run reads the build event file its own build wrote, two runs use different files, and a
  failure of either step fails the target.
- **`//tools/clang_tidy:selftest`** is `manual`. It lints `testdata/` with the checks of `//:.clang-tidy`. It fails
  unless each fixture's deduplicated findings are its `<name>.expected` file, unless the rename fixture's fix-its give
  the golden copy `testdata/renamed/`, and unless `clang-tidy --verify-config` accepts the configuration.
  `testdata/renamed/` is a `cc_library` in `//...`, so every build proves that the applied fixes compile. The
  `nullability` fixture proves that Abseil's `absl_nonnull` / `absl_nullable` reach clang: it expects a
  `clang-diagnostic-nonnull` and a `clang-diagnostic-nullability`, so a clang-tidy or an Abseil that expanded the macros
  to nothing would fail the self-test instead of passing every file silently.

Every linter skips `testdata/`: the aspect outside the self-test (`excluded_packages`), and cpplint and the token checks
of `tools/hooks` (`lint_files.FIXTURE_DIRS`).
