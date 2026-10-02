# The linters and the formatter (`tools/hooks`)

`AGENTS.md` holds the rules: "Style guides" and the sections after it list where this repository departs from the
Google C++ and Python style guides and the Abseil Tips of the Week, and which tool checks what. This directory holds
the machinery that checks them without a compiler: the formatter, the linter, the pre-commit hook and the token-level
checks. clang-tidy, which needs a compiler front end, runs as a Bazel aspect instead (`tools/clang_tidy/README.md`).

Run everything from the repository root, as modules: `python3 -m tools.hooks.<module>`, never
`python3 tools/hooks/<module>.py`.

| Command | What it does |
|---|---|
| `make format` (`python3 -m tools.hooks.format_code`) | Trailing whitespace and final newlines, the safe rewrites of the enforced checks, clang-format, isort (once its step is enforced) and black. |
| `make lint` (`python3 -m tools.hooks.lint_code`) | Every step below. It runs them all, then fails with a summary per step, so one run shows every finding. |
| `python3 -m tools.hooks.lint_code --only <check>[,<check>...]` | Only the named registry checks or steps, pending ones included. |
| `python3 -m tools.hooks.lint_code --paths <dir> ...` | Only the files under the given paths. |
| `python3 -m tools.hooks.lint_code --summary` | Counts per check and per Bazel package instead of the findings (token checks). |
| `python3 -m tools.hooks.lint_code --fix --only <check>` | Applies the check's `fix_source` to the working tree first. |
| `python3 -m tools.hooks.lint_code --git-staged` | The pre-commit hook's run: the token checks read the staged blobs, the tools the staged paths. |
| `make install-hooks` | Links `.git/hooks/pre-commit` to `pre-commit`: IFTTT, `make format`, restaging, then `--git-staged`. |

## The steps of `make lint`

1. IFTTT directives (`check_ifttt.py`).
2. Trailing whitespace and final newlines.
3. clang-format (`.clang-format`).
4. black.
5. isort (`.isort.cfg`, `--check-only`).
6. The token checks of the registry (`checks.py`).
7. cpplint (`CPPLINT.cfg`), in four parallel chunks.
8. pylint on the sources and 9. on the tests (`.pylintrc`; tests with `--disable=protected-access`).
10. mypy on the sources and 11. on the tests (`mypy.ini`; tests with `--allow-untyped-defs --allow-incomplete-defs`).

cpplint, pylint and mypy are the heavy steps. They run under the lint lock, an `flock` on `.lint_machine.lock` in the
checkout (`WB_LINT_MACHINE_LOCK` names another file, for the tests), so that only one of them runs on the machine at a
time next to a build that `tools/bazel` already sized to the whole machine. A second run prints that it is waiting and
blocks. The token checks take no lock: iterate with them (`--only`, `--paths`) and run the heavy steps once at the end.

pylint and mypy run with `PYTHONPATH` / `MYPYPATH` set to the Bazel import roots (`lint_files.python_import_roots()`)
in place of the container's own `PYTHONPATH`, so that they resolve imports the way Bazel does.

## The tools and their versions

cpplint, pylint (with astroid), isort and mypy are pinned in `lint_requirements.txt`, and `lint_requirements_lock.txt`
holds the hashed lock (`bazel run //tools/hooks:lint_requirements.update` regenerates it; run it on its own). The dev
image installs the lock into the venv `/opt/wb-lint`, CI's format job installs it with `pip --require-hashes`, and the
`@lint_deps` hub gives the real-tool tests the same versions under `bazel test`. `lint_code` compares each tool's
version with the lock: on a mismatch or a missing tool it fails when `CI=true`, and otherwise warns, skips the step and
says to rebuild the dev container. black and clang-format come from the image's apt packages, with CI's pins checked by
`test_ci_formatter_versions.py`.

## The registry

`checks.py` concatenates the `CHECKS` list of every check module into `REGISTRY`. A `Check` (`check_types.py`) has:

- `name`: kebab-case. It is the check's NOLINT category and its `--only` name.
- `languages`: the languages it reads (`CPP`, `PYTHON`, `PROTO`, `STARLARK`, `BAZELRC`, `SHELL`, or `TEXT` for every
  text file).
- `scope`: the file set (`lint_files.Scope`): `FIRST_PARTY`, `FIRST_PARTY_NON_TEST`, `FIRST_PARTY_AND_OCS2` (the
  nullability scope, which adds `lib/ocs2` without `lib/ocs2/thirdparty`), `TEXT` or `NOT_THIRDPARTY`.
- `check_source(source, path) -> list[Finding]`, a `description` (what it enforces and where the rule comes from), and a
  `hint` printed once under its findings.
- `fix_source(source, path) -> str` where a rewrite exists; with `fixed_by_format=True`, `make format` applies it once
  the check is enforced, so it must not be able to change behavior.

A finding prints as `path:line:column: message [check]`. `lint_files.py` is the one place that defines the file sets:
the vendored directories (`lib/`, `tools/ifttt-lint/`), the checked-in generated files, the fixture directories that
only the spelling checks read, test paths, and the Python import roots. `cpp_source.py` is the shared C++ tokenizer and
comment / string masker; use it rather than regular expressions over raw source.

The modules: `argument_comments`, `include_style`, `boost_usage`, `american_spelling`, `proto_file_layout`,
`proto_next_id`, `pointer_nullability`, `no_auto`, `cpp_google_style`, `cpp_totw`, `todo_format`,
`inclusive_language`, `python_imports`, `python_docstrings` and `python_language`. `python3 -m tools.hooks.lint_code
--only <name>` runs one check; the names and descriptions are in each module's `CHECKS`.

## NOLINT markers

`nolint.py` reads the markers from comments only (`//` and `/* */` in C++, `#` in Python, shell and Starlark):

- `NOLINT(<check>[, <check>...]): <reason>` on the line;
- `NOLINTNEXTLINE(<check>...): <reason>` on the line above;
- `NOLINTBEGIN(<check>...): <reason>` ... `NOLINTEND(<check>...)` around a block.

The engine applies them, and five checks judge the markers themselves:

| Check | Fails on |
|---|---|
| `nolint-category` | a bare `NOLINT`, which would silence every cpplint and clang-tidy check on the line |
| `nolint-reason` | a marker without a reason after the colon; such a marker suppresses nothing |
| `nolint-unknown` | a name that is no registry check, cpplint category or clang-tidy check of `.clang-tidy` (or, while it exists, of `tools/clang_tidy/sweep.clang-tidy`) |
| `nolint-unused` | a marker for a registry check, pending ones included, that suppresses no finding |
| `nolint-unbalanced` | a `NOLINTBEGIN` without its `NOLINTEND`, or the reverse |

The same markers serve cpplint and clang-tidy; `lint_code` drops cpplint's "Unknown NOLINT error category" complaints
about the registry's names. pylint's own messages use its pragma, `# pylint: disable=<message>  # <reason>`.

## Adding a check

1. Write the check in its own module (or in the module of its family): `check_source`, and `fix_source` if a rewrite
   cannot change behavior. List it in the module's `CHECKS`, and add a new module to `_MODULE_CHECKS` in `checks.py`.
2. Give the module a `py_library` and its `test_<module>.py` a `py_test` in `BUILD.bazel`. The test covers what the
   check detects and what it accepts, and calls `check_test_support.assert_check_behaves()`, which asserts the rest for
   every check alike: registration (enforced or pending), NOLINT and NOLINTNEXTLINE, a marker without a reason,
   `--git-staged` in a temporary git repository, the scope, and that a fix is a fixed point the check accepts.
3. Fix every finding in the same change. A check that cannot be fixed at once goes into `PENDING` (below) with the
   work that will fix it.
4. List the rule in `AGENTS.md` ("Enforced by" or the ToTW table). The `registry` IFTTT label of `checks.py` points
   there.

A stock-tool rule goes into `.clang-tidy`, `CPPLINT.cfg` or `.pylintrc` instead, with the same three steps after the
first.

## Pending checks and the sweeps

While the repository is being swept, a check that still has findings in the tree is pending:

- a token check is named in `checks.py:PENDING`;
- a cpplint category is filtered in the `SWEEP` block of `CPPLINT.cfg`, and a pylint message disabled in the
  `SWEEP_BEGIN` / `SWEEP_END` block of `.pylintrc`;
- a module with mypy errors has a relaxing section in the `SWEEP` block of `mypy.ini` (only `ignore_errors = True` or a
  relaxed `disallow_*`; `lint_code` rejects any other option and any section that names no module);
- a clang-tidy check is listed only in `tools/clang_tidy/sweep.clang-tidy`.

`make lint`, the pre-commit hook and CI skip pending checks; `--only` runs them. A sweep works like this:

```bash
python3 -m tools.hooks.lint_code --only no-auto,totw-at --paths humanoid_nmpc/humanoid_wb_mpc            # iterate
python3 -m tools.hooks.lint_code --only pointer-nullability --summary                                 # progress
python3 -m tools.hooks.lint_code --only float-literal --fix --paths robot_runtime                     # safe rewrites
make lint-tidy-sweep PKG=//humanoid_nmpc/humanoid_wb_mpc/... PATHS=humanoid_nmpc/humanoid_wb_mpc     # clang-tidy
```

A check leaves the pending set in the change that fixes its last finding: delete its name from `PENDING`, its line
from the `SWEEP` block, or move it from `sweep.clang-tidy` to `.clang-tidy`. The tests of each check assert whether it
is pending, so that change also flips `pending=True` in its test. When nothing is pending, `PENDING`, the `SWEEP`
blocks, `sweep.clang-tidy` and its Make target are deleted, and a test keeps them deleted.

## Tests

`bazel test //tools/hooks/...` runs every test here:

- one `test_<module>.py` per check module, and tests of the shared modules (`test_cpp_source.py`, `test_nolint.py`,
  `test_lint_files.py`, `test_checks.py`);
- `test_lint_code.py` (the steps, the run-all behavior, the lint lock) and `test_lint_entry_points.py` (the `-m` entry
  points from the repository root);
- `test_lint_tool_configs.py`, which runs the real cpplint, pylint, mypy, isort and black from `@lint_deps` against the
  repository's configurations, and `test_lint_tool_versions.py` / `test_ci_formatter_versions.py`, which keep the
  lock, the Dockerfile, CI and `MODULE.bazel` in step;
- `test_first_party_build_files.py`, which keeps `strip_include_prefix` / `include_prefix` out of first-party BUILD
  files.
