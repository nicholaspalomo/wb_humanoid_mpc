#!/bin/sh
# The state of the git worktree in the current directory, for the provenance of a recorded run (Makefile:
# VALIDATION_WORKTREE_STATE): "clean", "unknown" outside a git worktree, or "<N> changed paths, diff sha256 <16 hex>".
#
# The hash covers the diff of the tracked files against HEAD, the status with every untracked file listed on its own,
# and the CONTENTS of every untracked file that is not ignored, so that editing a new, not yet committed file changes
# it as editing a tracked one does. POSIX sh: the dev container's BASH_ENV makes every bash script source the shell setup.
set -u

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo unknown
  exit 0
fi

count=$(git status --porcelain --untracked-files=all 2>/dev/null | wc -l | tr -d ' ')
if [ "$count" = "0" ]; then
  echo clean
  exit 0
fi

hash=$({
  git diff HEAD
  git status --porcelain --untracked-files=all
  git ls-files -z --others --exclude-standard | xargs -0 -r sha256sum --
} 2>/dev/null | sha256sum | cut -c1-16)
echo "$count changed paths, diff sha256 $hash"
