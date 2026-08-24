#!/usr/bin/env python3
"""
Python validator for Google LINT.IfChange / LINT.ThenChange directives.
Used in git pre-commit hooks when ifttt-lint Rust binary is not installed locally.
Matches directives appearing alone on comment lines (shell, C++, Markdown, YAML, etc.).
"""

import sys
import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Directives must be alone on comment lines (e.g. '# LINT.IfChange(...)', '// LINT.IfChange(...)', '<!-- LINT.IfChange(...) -->')
IF_CHANGE_RE = re.compile(
    r"^\s*(?:#|//|<!--|--|;|\*)\s*LINT\.IfChange(?:\(([A-Za-z0-9_\-\.]+)\))?"
)
THEN_CHANGE_RE = re.compile(
    r"^\s*(?:#|//|<!--|--|;|\*)\s*LINT\.ThenChange\(([^)]+)\)"
)


def parse_targets(target_str):
    targets = []
    for item in target_str.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("//"):
            targets.append(item[2:])
        elif item.startswith(":"):
            targets.append(item)
        else:
            targets.append(item)
    return targets


def check_file(rel_path):
    full_path = os.path.join(REPO_ROOT, rel_path)
    if not os.path.exists(full_path):
        return []

    errors = []
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    if_stack = []
    seen_labels = set()
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

    for unclosed_idx, unclosed_label in if_stack:
        lbl_str = f"({unclosed_label})" if unclosed_label else ""
        errors.append(
            f"{rel_path}:{unclosed_idx}: LINT.IfChange{lbl_str} without matching LINT.ThenChange"
        )

    return errors


def main():
    files = sys.argv[1:]
    if not files:
        # Scan all tracked files in repo
        for root, dirs, fnames in os.walk(REPO_ROOT):
            if ".git" in root or "bazel-" in root or "tools/ifttt-lint" in root:
                continue
            for fname in fnames:
                rel = os.path.relpath(os.path.join(root, fname), REPO_ROOT)
                files.append(rel)

    total_errors = []
    for rel_path in files:
        if rel_path.startswith("tools/ifttt-lint") or rel_path.startswith(
            ".git"
        ):
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
