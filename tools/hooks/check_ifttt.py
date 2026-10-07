# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Validates the Google LINT.IfChange / LINT.ThenChange directives of the repository.

The pre-commit hook and `make lint` run it where the ifttt-lint Rust binary is not installed. It matches directives
alone on comment lines (shell, C++, Markdown, YAML, etc.).
"""

import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Directives must be alone on comment lines (e.g. '# LINT.IfChange(...)', '// LINT.IfChange(...)', '<!-- LINT.IfChange(...) -->')
IF_CHANGE_RE = re.compile(
    r"^\s*(?:#|//|<!--|--|;|\*)\s*LINT\.IfChange(?:\(([A-Za-z0-9_\-\.]+)\))?"
)
THEN_CHANGE_RE = re.compile(r"^\s*(?:#|//|<!--|--|;|\*)\s*LINT\.ThenChange\(([^)]+)\)")


def parse_targets(target_str: str) -> list[str]:
    """The targets of a ThenChange's comma-separated list, each without its leading `//`."""
    targets = []
    for item in target_str.split(","):
        item = item.strip()
        if not item:
            continue
        targets.append(item[2:] if item.startswith("//") else item)
    return targets


# The labels of each file read so far; one run reads a file once, however many directives point at it.
_LABEL_CACHE: dict[str, set[str]] = {}


def labels_in(rel_path: str) -> set[str]:
    """The IfChange labels a file declares, outside Markdown code blocks, so a ThenChange can be checked against them."""
    if rel_path in _LABEL_CACHE:
        return _LABEL_CACHE[rel_path]
    labels: set[str] = set()
    full_path = os.path.join(REPO_ROOT, rel_path)
    if os.path.isfile(full_path):
        in_code_block = False
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if rel_path.endswith(".md") and line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    continue
                match = IF_CHANGE_RE.search(line)
                if match and match.group(1):
                    labels.add(match.group(1))
    _LABEL_CACHE[rel_path] = labels
    return labels


def check_file(rel_path: str) -> list[str]:
    """The directive errors of one file: duplicate, nested or unmatched blocks, and targets that do not exist.

    Args:
        rel_path: The file, relative to the repository root.

    Returns:
        One `<path>:<line>: <problem>` message per error; none for a file that does not exist.
    """
    full_path = os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(full_path):
        return []

    errors = []
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if_stack: list[tuple[int, str | None]] = []
    seen_labels: set[str] = set()
    in_markdown_code_block = False

    for idx, line in enumerate(lines, 1):
        if rel_path.endswith(".md") and line.strip().startswith("```"):
            in_markdown_code_block = not in_markdown_code_block
            continue

        if in_markdown_code_block:
            continue

        if_match = IF_CHANGE_RE.search(line)
        then_match = THEN_CHANGE_RE.search(line)

        if if_match:
            label = if_match.group(1)
            if label:
                if label in seen_labels:
                    errors.append(
                        f"{rel_path}:{idx}: duplicate LINT.IfChange label '{label}'"
                    )
                seen_labels.add(label)
            # ifttt-lint, which the pre-commit hook runs, does not support nesting: it reports the outer block as
            # unmatched. Rejecting it here too keeps this fallback from passing what the real hook fails.
            if if_stack:
                open_idx, open_label = if_stack[-1]
                open_str = f"({open_label})" if open_label else ""
                errors.append(
                    f"{rel_path}:{idx}: LINT.IfChange inside the LINT.IfChange{open_str} opened at line "
                    f"{open_idx}; blocks cannot be nested - close the outer one first"
                )
            if_stack.append((idx, label))

        if then_match:
            if not if_stack:
                errors.append(
                    f"{rel_path}:{idx}: LINT.ThenChange without preceding LINT.IfChange"
                )
            else:
                if_stack.pop()

            targets = parse_targets(then_match.group(1))
            for target in targets:
                if ":" in target:
                    t_file, t_label = target.split(":", 1)
                else:
                    t_file, t_label = target, None

                if t_file:
                    target_full_path = os.path.join(REPO_ROOT, t_file)
                    if not os.path.exists(target_full_path):
                        errors.append(
                            f"{rel_path}:{idx}: target file not found '//{t_file}'"
                        )
                        continue
                # The label has to exist in the target too, or the directive guards nothing: a label renamed or
                # moved on one side silently leaves the other side pointing at nowhere.
                if t_label and t_label not in labels_in(t_file or rel_path):
                    errors.append(
                        f"{rel_path}:{idx}: target label not found '//{t_file}:{t_label}'"
                    )

    for unclosed_idx, unclosed_label in if_stack:
        lbl_str = f"({unclosed_label})" if unclosed_label else ""
        errors.append(
            f"{rel_path}:{unclosed_idx}: LINT.IfChange{lbl_str} without matching LINT.ThenChange"
        )

    return errors


def main() -> None:
    """Checks the files named on the command line, or every file git would ship; exits 1 on an error."""
    files = sys.argv[1:]
    if not files:
        # What git would ship - tracked files and untracked ones that are not ignored - which is what the pre-commit
        # hook sees. Walking the tree instead would lint editor backups and build output the hook never checks.
        try:
            listed = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
            files = [
                rel for rel in listed if os.path.isfile(os.path.join(REPO_ROOT, rel))
            ]
        except (OSError, subprocess.CalledProcessError):
            files = []
    if not files:
        for root, _, fnames in os.walk(REPO_ROOT):
            if ".git" in root or "bazel-" in root or "tools/ifttt-lint" in root:
                continue
            for fname in fnames:
                rel = os.path.relpath(os.path.join(root, fname), REPO_ROOT)
                files.append(rel)

    total_errors = []
    for rel_path in files:
        if rel_path.startswith("tools/ifttt-lint") or rel_path.startswith(".git"):
            continue
        errs = check_file(rel_path)
        total_errors.extend(errs)

    if total_errors:
        print("❌ IFTTT Directives Lint Errors:")
        for err in total_errors:
            print(f"  {err}")
        sys.exit(1)
    else:
        print("✅ IFTTT Directives verified cleanly across all files.")
        sys.exit(0)


if __name__ == "__main__":
    main()
