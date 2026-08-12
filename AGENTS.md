# Coding Agent Style Guidelines & Rule Directives

#### Co-dependent changes: use IFTTT directives (`LINT.IfChange` / `LINT.ThenChange`)

When code or configuration in one place must stay in sync with code elsewhere — but DRY cannot eliminate the duplication (for example, package registrations, launch targets, system dependencies, or Bazel repository rules) — mark the dependency with `LINT.IfChange` / `LINT.ThenChange` directives so changes to one side prompt review of the other. Add these directives proactively when creating new co-dependent content, not just when maintaining existing pairs.

Keep the guarded block as small as possible. Prefer several small labeled source->target pairs over one large catch-all block unless the whole region genuinely needs to change together. Directives are enforced via `ifttt-lint`.

<details>
<summary>Example</summary>

```bash
# LINT.IfChange(registered_packages)
_setup_package "my_robot_description" "${SCRIPT_DIR}/robot_models/my_robot/my_robot_description"
# LINT.ThenChange(//Makefile:launch_targets, //.devcontainer/VISUALIZATION.md:launch_targets)
```

</details>
