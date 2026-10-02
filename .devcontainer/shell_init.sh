#!/bin/bash
# ==============================================================================
# Container shell setup — sourced by .bashrc inside the Docker container
# Provides: Bazel/make tab completion, the Bazel environment (setup_env.sh), git completion
# ==============================================================================

# Tab completion is for a person at a prompt, so it is loaded in INTERACTIVE shells only. docker-compose.yaml points
# BASH_ENV at this file, so every non-interactive bash in the container sources it too - every Makefile recipe, every
# `docker exec ... bash -c`, every bash script. `bazel completion bash` there started a Bazel client per shell, and
# once `bazel` itself ran a bash script (tools/bazel) each call re-entered this file and called Bazel again, without
# end, until the container ran out of memory. BAZEL_REAL is set while Bazelisk runs tools/bazel: never recurse there.
case "$-" in
  *i*) _wb_interactive_shell=1 ;;
  *) _wb_interactive_shell=0 ;;
esac
if [ "${_wb_interactive_shell}" = 1 ] && [ -z "${BAZEL_REAL:-}" ]; then
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
fi
unset _wb_interactive_shell

# --- Auto-source the Bazel environment ---
# The setup_env.sh of the checkout the shell starts in, so that a second checkout of the repository (a git worktree
# next to the default one, with an environment of its own) is set up by its own script; outside every checkout, the
# default workspace's. Found by walking up from the working directory, without starting a process: every bash in the
# container runs this.
_wb_checkout=""
_wb_dir="${PWD:-}"
while [ -n "${_wb_dir}" ] && [ "${_wb_dir}" != "/" ]; do
    if [ -f "${_wb_dir}/setup_env.sh" ] && [ -f "${_wb_dir}/MODULE.bazel" ]; then
        _wb_checkout="${_wb_dir}"
        break
    fi
    _wb_dir="${_wb_dir%/*}"
done
WORKSPACE_DIR="${_wb_checkout:-/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc}"
unset _wb_checkout _wb_dir
if [ -f "${WORKSPACE_DIR}/setup_env.sh" ]; then
    pushd "${WORKSPACE_DIR}" >/dev/null
    source setup_env.sh
    popd >/dev/null
fi
