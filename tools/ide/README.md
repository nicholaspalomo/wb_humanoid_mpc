# The editor's view of the build (`tools/ide`)

Go to definition, find references and diagnostics in C++ come from [clangd](https://clangd.llvm.org/), which reads
`compile_commands.json` at the repository root: one entry per source file with the command line Bazel compiles it
with. Python navigation comes from a Pyright-based language server, which reads `pyrightconfig.json`.

## compile_commands.json

`python3 -m tools.ide.compile_commands` (`compile_commands.py`) writes it in the dev container, where Bazel and its
output base are. `bazel query` lists the C++ rules of the checkout, manual ones included, and its `proto_library`
targets, whose `.pb.cc` files the C++ proto aspect compiles; `bazel aquery` lists their CppCompile actions. So the
database holds the flags, defines and include directories the build really uses, and only C++ targets are analyzed: a
refresh fetches no repository that only Python targets use, such as the RL wheels of `@rl_deps`. The query and the
analysis take about a second on a warm Bazel server. The first refresh after the server starts, and the first after a
command with other configuration flags (`--config=dbg`), analyze every C++ target anew and take as long as the analysis
of a build of all of them, since Bazel keeps the analysis of one configuration at a time. A refresh that finds the
database unchanged leaves the file alone, so clangd has nothing to reload. The generated headers (protobuf, nproto) have
to exist for clangd to resolve their includes, and a build makes them.

The query loads the packages with the build's own options: `.bazelrc` sets `--incompatible_autoload_externally` for
every command (`common`). A `bazel query` that loaded them otherwise would make the next build or aquery fetch
rules_cc, the pip hubs and the system libraries' CMake builds again, about 12 seconds each time.

It stays current by itself:

| When | How |
|---|---|
| The dev container starts | `docker-compose.yaml`'s command runs `--build_headers --quiet` in the background. It builds the compilation prerequisites of the C++ targets, which are the generated headers and the tools that generate them (protobuf's C++ generator, the nproto generator, MuJoCo's CMake build) but no first-party source, and then queries. On a new container's empty output base that takes a while, and a build started meanwhile waits for it (`tools/bazel`'s machine lock). The log is inside the container: `docker exec devcontainer-app-1 cat /tmp/compile_commands.log`. |
| A Makefile target builds | Every target that runs `bazel build`, `test` or `run` (the build, test, lint-tidy, validation, RL and launch targets, `deploy-robot`, `robot-images`) runs it through `python3 -m tools.ide.compile_commands --after bazel ...`. The runner refreshes the database after the command, passed or failed, with the command's configuration flags (`--config`, `-c`, `--platforms`, ...), and keeps its exit status; it does not refresh after a command that was interrupted, could not start or had a command line error. It says `compile_commands: refreshing compile_commands.json` before the refresh and how many source files the database holds after it, on standard error, where all its messages go. A refresh that fails or is interrupted (Ctrl-C) only warns, but Ctrl-C during the refresh of a `bazel run` also cancels the program: the runner exits with 130. |
| A Makefile target runs a program | `bazel run` becomes `bazel run --script_path`, which builds the program and writes its command line, its arguments included, to a script without running it. The database is refreshed, and the script then runs in the runner's place, so the program starts as it would from `bazel run`. Like `bazel run`, this does not wait for `tools/bazel`'s machine lock. |
| By hand | `make compile-commands`, from the host or the container. |

A `--keep_going` command that some targets fail in still refreshes: the build of the generated headers and the query
warn, and the database gets the rest. A result that leaves out targets that failed to load or to analyze is merged into
the database by file, which keeps its entries for the sources the result lacks, so a partial result never shrinks it. A
query that lists no target, a command line error, an interruption or an internal error of Bazel leaves the database as
it was.

Only a dev container writes it, and every container that mounts the checkout at `/wb_humanoid_mpc_ws` does: the last
one to refresh wins. Two such containers write the same paths, since each runs Bazel as the same user with the
checkout at the same path, and the output base's path (`~/.cache/bazel/_bazel_<user>/<the hash of the checkout's
path>`) follows from those two. A container whose builds should leave the database to another sets
`WB_COMPILE_COMMANDS=off` in its environment. Elsewhere the runner runs the Bazel command alone and says so in one
line: in CI (GitHub sets `CI`, and `tools/ci_local.sh` sets it for `make ci-local`) and wherever the checkout is not
below the dev container's mount (the container of `make ci-local`, the host). A Makefile target run on the host runs
the host's own Bazel, if the host has one, which is neither the dev container's nor under its memory cap.

`//tools/ide:test_ide_setup` fails when a Makefile recipe or a launch script calls Bazel without the refresh.

Each entry runs in Bazel's execution root, so `bazel-out/` and the checkout's directories resolve as in the build; its
`file` is the source's path in the checkout. clangd follows the execution root's links into the checkout, so a
first-party header is reported at its path in the checkout too. The paths into `external/` point at the output base
instead: the execution root links only the external repositories its last build used, so after `make ipc-list` its
`external/` has no Abseil, while the output base holds every repository Bazel fetched.

`.clangd` removes the few GCC flags clang does not take, and keeps the generated sources under `bazel-out/` (about a
fifth of the entries) out of the background index: clangd indexes one when it is opened, and the generated headers
with the first-party sources that include them.

## clangd from an editor on the host

The database's paths are the container's, so clangd runs in the container. `clangd_in_dev_container.sh` starts it
there through `docker exec` and maps the checkout's path on the host to its path in the container,
`/wb_humanoid_mpc_ws/<parent>/<checkout>` (`docker-compose.yaml` mounts the checkout's grandparent at
`/wb_humanoid_mpc_ws`), or the path `WB_DEV_CONTAINER_REPO` names. Point the editor's clangd extension
(`llvm-vs-code-extensions.vscode-clangd`) at it in the workspace's `.vscode/settings.json`:

```json
{
  "clangd.path": "${workspaceFolder}/tools/ide/clangd_in_dev_container.sh"
}
```

`WB_DEV_CONTAINER` names another container than `devcontainer-app-1`. In an editor attached to the container the same
script runs clangd directly.

<!-- LINT.IfChange(clangd_version) -->
The image's clangd is `clangd-21`, the build of the image's clang-tidy (`docker/install_llvm_tools.sh`). A container of
an older image without it runs the clangd that the extension downloaded on the host, through the mount.
<!-- LINT.ThenChange(//tools/ide/clangd_in_dev_container.sh:clangd_version) -->

<!-- LINT.IfChange(clangd_flags) -->
clangd's background index is a compiler front end running outside Bazel's RAM budget (AGENTS.md, "Builds share one
machine's memory"), so the script runs it with two workers (`-j=2`) at low priority and keeps preambles on disk
(`--pch-storage=disk`). The first index of the repository takes a while; later ones only re-index what changed. The index
is in `.cache/clangd/`, which git ignores. `--header-insertion=never` keeps clangd from adding includes in another
style than the repository's.

Two workers keep the editor responsive while it indexes. Each dev container's memory cap is 85% of the host's RAM on
its own, though, so two containers that each run a clangd (an editor attached to each, or this script pointed at both)
can still overcommit the machine together, next to a build: let one index finish before starting the other.
<!-- LINT.ThenChange(//tools/ide/clangd_in_dev_container.sh:clangd_flags) -->

What a host editor cannot open: a definition in an external library (Abseil, Eigen, Pinocchio, protobuf) or in a
generated header (the nproto structs) lives in the container's Bazel output base, which the host does not see. Go to
definition into those works from an editor attached to the container.

## Bazel on the host

An editor's Bazel extension on the host (`bazelbuild.vscode-bazel`) runs the host's Bazel for its workspace tree and
its queries: a Bazel server with an output base of its own, outside the dev container's memory cap and outside
`tools/bazel`'s machine lock, next to the container's builds. On the host, disable the extension for this workspace,
or keep its syntax highlighting and buildifier and stop its queries with `"bazel.pathsToIgnore": [".*"]` in the
workspace's `.vscode/settings.json`; in an editor attached to the container it runs the container's Bazel.

## Python

`pyrightconfig.json` names the Bazel import roots, the directories the BUILD files' `imports = [...]` put on the path
(`tools.hooks.lint_files.python_import_roots()`, which pylint and mypy use too), so `from remote_control import
operator_bus` resolves as it does in Bazel. `//tools/ide:test_ide_setup` keeps the two lists equal. Pylance (VS Code)
reads it; in a VS Code fork without Pylance (Antigravity, VSCodium), install basedpyright
(`detachhead.basedpyright`, from Open VSX), which reads it too. The Jedi server of the Python extension alone does
not. It sets no type checking: mypy is the repository's checker (`make lint`).
