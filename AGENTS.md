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
units (2-3 GB each) next to the simulator, RViz and out-of-tree compiles pushed it into swap until it had to be
power-cycled. Two guards exist, and changes must not weaken them:

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
gray, program. `tools/hooks/lint_code.py` checks it (`tools/hooks/american_spelling.py`, which also fixes files with
`--fix`); a line that must keep a British spelling, such as the title of a cited paper, is marked
`NOLINT(american-spelling)`. Vendored code under `lib/` and robot model files are left as they come.
