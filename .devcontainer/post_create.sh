#!/bin/bash
set -e

# Detect container user home directory
CONTAINER_HOME=$(getent passwd $(id -u) | cut -d: -f6)

# Setup SSH keys if mounted
if [ -d /tmp/host_ssh ]; then
  # Symlink ~/.ssh to /tmp/host_ssh if it isn't already a symlink pointing to /tmp/host_ssh
  if [ "$(readlink -f "${CONTAINER_HOME}/.ssh" 2>/dev/null)" != "/tmp/host_ssh" ]; then
    rm -rf "${CONTAINER_HOME}/.ssh"
    ln -s /tmp/host_ssh "${CONTAINER_HOME}/.ssh"
  fi
  chmod 700 /tmp/host_ssh
  chmod 600 /tmp/host_ssh/id_* 2>/dev/null || true
  echo "SSH keys mounted and permissions set"

  eval "$(ssh-agent -s)" > /dev/null
  find /tmp/host_ssh -type f -name "id_*" ! -name "*.pub" | xargs -I{} ssh-add {} 2>/dev/null
  echo "SSH agent started and keys added"
fi

# Copy and use host Git config if available
if [ -f /tmp/host.gitconfig ]; then
  cp /tmp/host.gitconfig "${CONTAINER_HOME}/.gitconfig"
  echo "Using Git configuration from host"
else
  # Only prompt for Git info if no config exists
  if [ ! -f "${CONTAINER_HOME}/.gitconfig" ]; then
    echo "Please enter your Git user name:"
    read -r GIT_USER_NAME
    echo "Please enter your Git email:"
    read -r GIT_USER_EMAIL

    git config --global user.name "$GIT_USER_NAME"
    git config --global user.email "$GIT_USER_EMAIL"
    echo "Git configuration saved. Name: $GIT_USER_NAME, Email: $GIT_USER_EMAIL"
  fi
fi

# Setup Git credentials for HTTPS access
if [ -f "${CONTAINER_HOME}/.git-credentials" ]; then
  git config --global credential.helper store
  echo "Git credentials file found - HTTPS authentication configured"
else
  git config --global credential.helper cache
  echo "Git credential helper configured to use cache"
fi

# Safe directories
WORKSPACE_DIR="/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc"
git config --global --add safe.directory "${WORKSPACE_DIR}"
git submodule foreach --recursive bash -c "git config --global --add safe.directory \$(realpath .)" 2>/dev/null || true

# Git completion
curl -sSLo "${CONTAINER_HOME}/.git-completion.bash" https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash
echo "source ${CONTAINER_HOME}/.git-completion.bash" >> "${CONTAINER_HOME}/.bashrc"

# Bazel tab completion
echo '# Bazel completion' >> "${CONTAINER_HOME}/.bashrc"
echo 'if command -v bazel &>/dev/null; then' >> "${CONTAINER_HOME}/.bashrc"
echo '  source <(bazel completion bash 2>/dev/null) || true' >> "${CONTAINER_HOME}/.bashrc"
echo 'fi' >> "${CONTAINER_HOME}/.bashrc"

# Make target tab completion
cat >> "${CONTAINER_HOME}/.bashrc" << 'MAKE_COMPLETION'

# Make target completion
_make_targets() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local makefile="Makefile"
    if [ -f "$makefile" ]; then
        local targets=$(grep -oE '^[a-zA-Z0-9_-]+:' "$makefile" | sed 's/://' | sort -u)
        COMPREPLY=($(compgen -W "$targets" -- "$cur"))
    fi
}
complete -F _make_targets make
MAKE_COMPLETION

# Auto-source Bazel+ROS2 environment
echo '# Bazel + ROS2 environment' >> "${CONTAINER_HOME}/.bashrc"
echo 'WORKSPACE_DIR="/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc"' >> "${CONTAINER_HOME}/.bashrc"
echo 'if [ -f "${WORKSPACE_DIR}/setup_env.sh" ]; then' >> "${CONTAINER_HOME}/.bashrc"
echo '  cd "${WORKSPACE_DIR}" && source setup_env.sh && cd - >/dev/null' >> "${CONTAINER_HOME}/.bashrc"
echo 'fi' >> "${CONTAINER_HOME}/.bashrc"

# SSH agent in bashrc
echo '# Start SSH agent on terminal startup' >> "${CONTAINER_HOME}/.bashrc"
echo 'if [ -d /tmp/host_ssh ]; then' >> "${CONTAINER_HOME}/.bashrc"
echo '  if [ "$(readlink -f ~/.ssh 2>/dev/null)" != "/tmp/host_ssh" ]; then rm -rf ~/.ssh && ln -s /tmp/host_ssh ~/.ssh; fi' >> "${CONTAINER_HOME}/.bashrc"
echo '  eval $(ssh-agent -s) > /dev/null' >> "${CONTAINER_HOME}/.bashrc"
echo '  find ~/.ssh -type f -name "id_*" ! -name "*.pub" | xargs -I{} ssh-add {} 2>/dev/null' >> "${CONTAINER_HOME}/.bashrc"
echo 'fi' >> "${CONTAINER_HOME}/.bashrc"

# Auto-install Git pre-commit hook
if [ -f "${WORKSPACE_DIR}/Makefile" ]; then
  cd "${WORKSPACE_DIR}" && make install-hooks 2>/dev/null || true
fi

# Register Jupyter kernel for VS Code / Antigravity
if command -v python3 &>/dev/null; then
  python3 -m ipykernel install --user --name "wb_humanoid_mpc" --display-name "Python 3 (Humanoid MPC)" 2>/dev/null || true
fi

echo "Container environment setup complete."
echo "- SSH authentication configured: $([ -d /tmp/host_ssh ] && echo "Yes" || echo "No")"
echo "- HTTPS credential helper: $(git config --global credential.helper)"
echo "- Git user: $(git config --global user.name)"
echo "- Git email: $(git config --global user.email)"
