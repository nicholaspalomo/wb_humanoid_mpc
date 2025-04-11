#!/bin/bash
set -e

# Setup SSH keys if mounted
if [ -d ~/.ssh ]; then
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/id_* 2>/dev/null || true
  echo "SSH keys mounted and permissions set"

  eval "$(ssh-agent -s)"
  find ~/.ssh -type f -name "id_*" ! -name "*.pub" | xargs -I{} ssh-add {} 2>/dev/null
  echo "SSH agent started and keys added"
fi

# Copy and use host Git config if available
if [ -f /tmp/host.gitconfig ]; then
  cp /tmp/host.gitconfig ~/.gitconfig
  echo "Using Git configuration from host"
else
  # Only prompt for Git info if no config exists
  if [ ! -f ~/.gitconfig ]; then
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
if [ -f ~/.git-credentials ]; then
  git config --global credential.helper store
  echo "Git credentials file found - HTTPS authentication configured"
else
  git config --global credential.helper cache
  echo "Git credential helper configured to use cache"
fi

# Safe directories
git config --global --add safe.directory /wb_humanoid_mpc_ws/src/wb_humanoid_mpc
git submodule foreach --recursive bash -c "git config --global --add safe.directory \$(realpath .)"

# Git completion
curl -sSLo ~/.git-completion.bash https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash
echo "source ~/.git-completion.bash" >> ~/.bashrc

# Colcon + ROS setup
echo '# Colcon settings' >> ~/.bashrc
echo 'export COLCON_HOME=/wb_humanoid_mpc_ws' >> ~/.bashrc
echo 'export COLCON_DEFAULTS_FILE=/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/colcon.defaults.yaml' >> ~/.bashrc
echo 'source /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc

# SSH agent in bashrc
echo '# Start SSH agent on terminal startup' >> ~/.bashrc
echo 'if [ -d ~/.ssh ]; then' >> ~/.bashrc
echo '  eval $(ssh-agent -s) > /dev/null' >> ~/.bashrc
echo '  find ~/.ssh -type f -name "id_*" ! -name "*.pub" | xargs -I{} ssh-add {} 2>/dev/null' >> ~/.bashrc
echo 'fi' >> ~/.bashrc

chmod +x "$0"

echo "Container environment setup complete."
echo "- SSH authentication configured: $([ -d ~/.ssh ] && echo "Yes" || echo "No")"
echo "- HTTPS credential helper: $(git config --global credential.helper)"
echo "- Git user: $(git config --global user.name)"
echo "- Git email: $(git config --global user.email)"
