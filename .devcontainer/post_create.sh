#!/bin/bash

# Configure Git user
git config --global user.name "$(git config --global user.name || echo Default User)"
git config --global user.email "$(git config --global user.email || echo user@example.com)"

# Add main repository to safe directories
git config --global --add safe.directory /wb_humanoid_mpc_ws/src/wb_humanoid_mpc

# Add all submodules to safe directories
git submodule foreach --recursive bash -c "git config --global --add safe.directory \$(realpath .)"

# Set up Git completion
curl -o ~/.git-completion.bash https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash
echo "source ~/.git-completion.bash" >> ~/.bashrc