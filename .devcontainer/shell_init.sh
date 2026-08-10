#!/bin/bash
# ==============================================================================
# Container shell setup — sourced by .bashrc inside the Docker container
# Provides: Bazel/make tab completion, ROS2+Bazel env, git completion
# ==============================================================================

# --- Source bash-completion framework (provides git, etc.) ---
if [ -f /usr/share/bash-completion/bash_completion ]; then
    source /usr/share/bash-completion/bash_completion
elif [ -f /etc/bash_completion ]; then
    source /etc/bash_completion
fi

# --- Git completion ---
if [ -f ~/.git-completion.bash ]; then
    source ~/.git-completion.bash
elif [ -f /usr/share/bash-completion/completions/git ]; then
    source /usr/share/bash-completion/completions/git
fi

# --- Bazel tab completion ---
if command -v bazel &>/dev/null; then
    source <(bazel completion bash 2>/dev/null) || true
fi

# --- Make target tab completion ---
_make_targets() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local makefile="Makefile"
    if [ -f "$makefile" ]; then
        local targets=$(grep -oE '^[a-zA-Z0-9_-]+:' "$makefile" | sed 's/://' | sort -u)
        COMPREPLY=($(compgen -W "$targets" -- "$cur"))
    fi
}
complete -F _make_targets make

# --- Auto-source Bazel+ROS2 environment ---
WORKSPACE_DIR="/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc"
if [ -f "${WORKSPACE_DIR}/setup_env.sh" ]; then
    pushd "${WORKSPACE_DIR}" >/dev/null
    source setup_env.sh
    popd >/dev/null
fi
