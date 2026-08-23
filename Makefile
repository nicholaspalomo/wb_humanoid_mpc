SHELL := /bin/bash

############################################################
# Bazel Build System — Monorepo Makefile
############################################################

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_path := $(dir $(mkfile_path))

# Source the Bazel+ROS2 environment
define source_env
	source $(current_path)/setup_env.sh
endef

############################################################
# Build targets
############################################################
.PHONY: build-all build-debug build-release build-relwithdebinfo build \
        test-all test clean clean-all format ci-local \
        launch-g1-dummy-sim launch-g1-sim launch-wb-g1-dummy-sim launch-wb-g1-sim \
        launch-drc-atlas-dummy-sim launch-drc-atlas-sandbox test-pinocchio-model-atlas \
        launch-r1-dummy-sim launch-r1-sim launch-r1-sandbox test-pinocchio-model-r1 \
        start-vnc stop-vnc \
        launch-g1-dummy-sim-vnc launch-g1-sim-vnc launch-wb-g1-dummy-sim-vnc launch-wb-g1-sim-vnc \
        launch-drc-atlas-dummy-sim-vnc launch-drc-atlas-sandbox-vnc \
        launch-r1-dummy-sim-vnc launch-r1-sim-vnc launch-r1-sandbox-vnc \
        run-ocs2-tests run-mpc-tests test-rl train-rl train-bc export-rollouts lock-rl-deps echo-packages update-submodules git-lfs install-hooks

## Build everything
build-all:
	$(source_env) && bazel build //...

## Build a single target: make build PKG=//humanoid_nmpc/humanoid_common_mpc
build:
	@$(if $(PKG),bazel build $(PKG),@echo "Usage: make build PKG=//path/to:target")

## Debug build
build-debug:
	bazel build //... --config=dbg

## Release build
build-release:
	bazel build //...

## Release with debug info
build-relwithdebinfo:
	bazel build //... --config=relwithdebinfo

############################################################
# Test targets
############################################################

## Run all tests
test-all:
	$(source_env) && bazel test //...

## Run local CI emulator (matches GitHub Actions clean container setup)
ci-local:
	@tools/ci_local.sh

## Run a single test: make test PKG=//humanoid_nmpc/humanoid_centroidal_mpc_test:test_pinocchio_frame_conversions
test:
	@$(if $(PKG),bazel test $(PKG),@echo "Usage: make test PKG=//path/to:test_target")

## Run OCS2 library tests
run-ocs2-tests:
	bazel test //lib/ocs2:all

## Run MPC tests
run-mpc-tests:
	bazel test //humanoid_nmpc/...

## Run MuJoCo Playground RL tests
test-rl:
	bazel test //humanoid_learning/...

## Run MuJoCo Playground RL PPO Training
train-rl:
	bazel run //humanoid_learning/training:train_ppo

## Behavioral Cloning pretraining on MPC demos
train-bc:
	bazel run //humanoid_learning/training:bc_warmstart

## Export recorded MPC trajectories to HDF5 demos
export-rollouts:
	python3 humanoid_nmpc/humanoid_common_mpc_pyutils/humanoid_common_mpc_pyutils/export_rollouts.py

## Lock / Update RL pip dependencies
lock-rl-deps:
	bazel run //humanoid_learning:requirements.update

## Run Pinocchio Model Atlas test
test-pinocchio-model-atlas:
	@bazel build //... && \
	$(source_env) && ros2 run drc_atlas_centroidal_mpc test_pinocchio_model

## Run Pinocchio Model R1 test
test-pinocchio-model-r1:
	@bazel build //... && \
	$(source_env) && ros2 run unitree_r1_centroidal_mpc test_pinocchio_model

############################################################
# Utility targets
############################################################

## List all Bazel targets
echo-packages:
	@bazel query '//...' 2>/dev/null | sort

## Clean Bazel cache
clean:
	bazel clean

## Deep clean (remove entire Bazel cache + generated env)
clean-all:
	bazel clean --expunge
	rm -rf $(current_path)/.bazel_ros_install

## Format source code (C++, Python, trailing newlines, and whitespace)
format:
	@python3 tools/hooks/format_code.py

## Install git pre-commit hook
install-hooks:
	@chmod +x tools/hooks/pre-commit && \
	cp tools/hooks/pre-commit .git/hooks/pre-commit && \
	chmod +x .git/hooks/pre-commit && \
	echo "✅ Git pre-commit hook installed successfully."

## Update git submodules (mujoco)
update-submodules:
	git submodule update --init --recursive

## Pull git-lfs files
git-lfs:
	git lfs install && git lfs pull

############################################################
# Launch targets (Bazel build + ROS2 launch)
############################################################

# LINT.IfChange(launch_targets)
launch-g1-dummy-sim:
	@bazel build //humanoid_nmpc/humanoid_centroidal_mpc_ros2:all //humanoid_nmpc/humanoid_common_mpc_ros2:all && \
	$(source_env) && ros2 launch g1_centroidal_mpc dummy_sim.launch.py

launch-g1-sim:
	@bazel build //humanoid_nmpc/humanoid_centroidal_mpc_ros2:all //humanoid_nmpc/humanoid_common_mpc_ros2:all && \
	$(source_env) && ros2 launch g1_centroidal_mpc mujoco_sim.launch.py

launch-wb-g1-dummy-sim:
	@bazel build //humanoid_nmpc/humanoid_wb_mpc_ros2:all //humanoid_nmpc/humanoid_common_mpc_ros2:all && \
	$(source_env) && ros2 launch g1_wb_mpc dummy_sim.launch.py

launch-wb-g1-sim:
	@bazel build //humanoid_nmpc/humanoid_wb_mpc_ros2:all //humanoid_nmpc/humanoid_common_mpc_ros2:all && \
	$(source_env) && ros2 launch g1_wb_mpc mujoco_sim.launch.py

launch-drc-atlas-dummy-sim:
	@bazel build //... && \
	$(source_env) && ros2 launch drc_atlas_centroidal_mpc dummy_sim.launch.py

launch-drc-atlas-sim:
	@bazel build //... && \
	$(source_env) && ros2 launch drc_atlas_centroidal_mpc mujoco_sim.launch.py

launch-drc-atlas-sandbox:
	@bazel build //... && \
	$(source_env) && ros2 launch drc_atlas_description display.launch.py

launch-r1-dummy-sim:
	@bazel build //... && \
	$(source_env) && ros2 launch unitree_r1_centroidal_mpc dummy_sim.launch.py

launch-r1-sim:
	@bazel build //... && \
	$(source_env) && ros2 launch unitree_r1_centroidal_mpc mujoco_sim.launch.py

launch-r1-sandbox:
	@bazel build //... && \
	$(source_env) && ros2 launch unitree_r1_description display.launch.py

############################################################
# VNC visualization (for macOS host)
############################################################
start-vnc:
	@chmod +x $(current_path)/.devcontainer/start_vnc.sh && \
	$(current_path)/.devcontainer/start_vnc.sh $(RESOLUTION)

stop-vnc:
	@chmod +x $(current_path)/.devcontainer/start_vnc.sh && \
	$(current_path)/.devcontainer/start_vnc.sh stop

# Environment overrides for VNC display + Mesa software GLX
VNC_GL_ENV := export DISPLAY=:99 && \
	export LIBGL_ALWAYS_SOFTWARE=1 && \
	export LIBGL_ALWAYS_INDIRECT=0 && \
	export GALLIUM_DRIVER=llvmpipe && \
	export MESA_GL_VERSION_OVERRIDE=3.3 && \
	export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

launch-g1-dummy-sim-vnc: start-vnc
	@echo "🚀 Building targets and launching G1 Centroidal MPC Dummy Simulation..."
	@bazel build //humanoid_nmpc/humanoid_centroidal_mpc_ros2:all //humanoid_nmpc/humanoid_common_mpc_ros2:all && \
	$(source_env) && $(VNC_GL_ENV) && ros2 launch g1_centroidal_mpc dummy_sim.launch.py

launch-g1-sim-vnc: start-vnc
	@echo "🚀 Building targets and launching G1 Centroidal MPC MuJoCo Simulation..."
	@bazel build //humanoid_nmpc/humanoid_centroidal_mpc_ros2:all //humanoid_nmpc/humanoid_common_mpc_ros2:all && \
	$(source_env) && $(VNC_GL_ENV) && ros2 launch g1_centroidal_mpc mujoco_sim.launch.py

launch-wb-g1-dummy-sim-vnc: start-vnc
	@echo "🚀 Building targets and launching G1 Whole-Body MPC Dummy Simulation..."
	@bazel build //humanoid_nmpc/humanoid_wb_mpc_ros2:all //humanoid_nmpc/humanoid_common_mpc_ros2:all && \
	$(source_env) && $(VNC_GL_ENV) && ros2 launch g1_wb_mpc dummy_sim.launch.py

launch-wb-g1-sim-vnc: start-vnc
	@echo "🚀 Building targets and launching G1 Whole-Body MPC MuJoCo Simulation..."
	@bazel build //humanoid_nmpc/humanoid_wb_mpc_ros2:all //humanoid_nmpc/humanoid_common_mpc_ros2:all && \
	$(source_env) && $(VNC_GL_ENV) && ros2 launch g1_wb_mpc mujoco_sim.launch.py

launch-drc-atlas-dummy-sim-vnc: start-vnc
	@echo "🚀 Building targets and launching DRC Atlas Dummy Simulation..."
	@bazel build //... && \
	$(source_env) && $(VNC_GL_ENV) && ros2 launch drc_atlas_centroidal_mpc dummy_sim.launch.py

launch-drc-atlas-sim-vnc: start-vnc
	@echo "🚀 Building targets and launching DRC Atlas MuJoCo Simulation..."
	@bazel build //... && \
	$(source_env) && $(VNC_GL_ENV) && ros2 launch drc_atlas_centroidal_mpc mujoco_sim.launch.py

launch-drc-atlas-sandbox-vnc: start-vnc
	@echo "🚀 Building targets and launching DRC Atlas URDF Viewer..."
	@bazel build //... && \
	$(source_env) && $(VNC_GL_ENV) && ros2 launch drc_atlas_description display.launch.py

launch-r1-dummy-sim-vnc: start-vnc
	@echo "🚀 Building targets and launching Unitree R1 Centroidal MPC Dummy Simulation..."
	@bazel build //... && \
	$(source_env) && $(VNC_GL_ENV) && ros2 launch unitree_r1_centroidal_mpc dummy_sim.launch.py

launch-r1-sim-vnc: start-vnc
	@echo "🚀 Building targets and launching Unitree R1 Centroidal MPC MuJoCo Simulation..."
	@bazel build //... && \
	$(source_env) && $(VNC_GL_ENV) && ros2 launch unitree_r1_centroidal_mpc mujoco_sim.launch.py

launch-r1-sandbox-vnc: start-vnc
	@echo "🚀 Building targets and launching Unitree R1 URDF Viewer..."
	@bazel build //... && \
	$(source_env) && $(VNC_GL_ENV) && ros2 launch unitree_r1_description display.launch.py
# LINT.ThenChange(//setup_env.sh:registered_packages, //.devcontainer/VISUALIZATION.md:launch_targets)
