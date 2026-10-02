#!/bin/sh
# Runs clang-tidy on one file for the clang-tidy aspect (tools/clang_tidy/clang_tidy.bzl, README.md). Each run is a
# Bazel action, so it is scheduled inside .bazelrc's RAM-bounded --jobs and tools/bazel's machine lock, like a compile.
#
# Usage: run_clang_tidy.sh <report> <fixes> <clang-tidy argument>...
#        run_clang_tidy.sh --verify-config <config>...
#
# The first form writes clang-tidy's findings to <report> and its fix-its to <fixes> (--export-fixes), and succeeds
# whether or not there are findings: Bazel caches a successful action, so a later run re-lints only the files that
# changed, and tools/clang_tidy/clang_tidy_report.py turns the reports into the result. It fails only when clang-tidy
# itself fails: a crash, or an exit status with no diagnostic to show for it. The second form checks that every check and
# option a configuration names exists (the self-test runs it).
#
# The system's clang-tidy is not part of an action's cache key, so this script pins its major version: a different
# clang-tidy is an error, and changing the pin changes this file, which re-runs every action.
#
# POSIX sh, not bash, for the reason tools/bazel gives: the dev container's BASH_ENV makes every bash script source the
# shell setup first.
set -eu

# LINT.IfChange(clang_tidy_version)
CLANG_TIDY_VERSION=21
# LINT.ThenChange(//docker/Dockerfile:clang_tidy_version, //docker/install_llvm_tools.sh:llvm_versions)

tidy="clang-tidy-${CLANG_TIDY_VERSION}"
if ! command -v "${tidy}" >/dev/null 2>&1; then
  echo "run_clang_tidy.sh: ${tidy} is not installed. Rebuild the dev container, or install it with" \
    "'sudo sh docker/install_llvm_tools.sh tidy'." >&2
  exit 1
fi
version="$("${tidy}" --version | sed -n 's/.*LLVM version \([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
if [ "${version}" != "${CLANG_TIDY_VERSION}" ]; then
  echo "run_clang_tidy.sh: ${tidy} reports LLVM version '${version}', expected ${CLANG_TIDY_VERSION}." \
    "Rebuild the dev container." >&2
  exit 1
fi

if [ "${1:-}" = "--verify-config" ]; then
  shift
  for config in "$@"; do
    # --verify-config reads no source file, so it compiles nothing.
    "${tidy}" --verify-config --config-file="${config}"
  done
  exit 0
fi

if [ "$#" -lt 3 ]; then
  echo "usage: run_clang_tidy.sh <report> <fixes> <clang-tidy argument>..." >&2
  exit 2
fi
report="$1"
fixes="$2"
shift 2

errors="$(mktemp)"
trap 'rm -f "${errors}"' EXIT
status=0
"${tidy}" --export-fixes="${fixes}" "$@" >"${report}" 2>"${errors}" || status=$?
# clang-tidy writes no fixes file for a file without findings; the action declares one either way.
[ -f "${fixes}" ] || : >"${fixes}"

# A signal (a crash) is never a finding, even when clang-tidy printed some before it.
if [ "${status}" -ge 128 ] || { [ "${status}" -ne 0 ] && ! grep -Eq ': (warning|error): ' "${report}"; }; then
  echo "run_clang_tidy.sh: clang-tidy failed with status ${status} and reported nothing it found:" >&2
  cat "${errors}" >&2
  exit 1
fi
exit 0
