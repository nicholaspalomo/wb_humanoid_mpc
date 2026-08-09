SHELL := /bin/bash

############################################################
# Bazel Build System — drop-in replacement for colcon Makefile
############################################################

# --- Detect ROS2 distro for launch commands ---
ros_source_file := /bin/ros_setup.sh

ifeq ("$(wildcard /opt/ros/jazzy/setup.bash)","")
    ifeq ("$(wildcard $(ros_source_file))","")
        ros_source_file := /opt/ros/humble/setup.bash
    endif
else
    ifeq ("$(wildcard $(ros_source_file))","")
        ros_source_file := /opt/ros/jazzy/setup.bash
    endif
endif

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_path := $(dir $(mkfile_path))
build_dir ?= $(abspath $(lastword $(MAKEFILE_LIST))/../../..)

############################################################
# Build targets
############################################################
.PHONY: build-all build-debug build-release build-relwithdebinfo build \
        test-all test clean format \
        launch-g1-dummy-sim launch-g1-sim launch-wb-g1-dummy-sim launch-wb-g1-sim \
        start-vnc stop-vnc \
        launch-g1-dummy-sim-vnc launch-g1-sim-vnc launch-wb-g1-dummy-sim-vnc launch-wb-g1-sim-vnc \
        run-ocs2-tests run-mpc-tests echo-packages update-submodules git-lfs

## Build everything (optimized, equivalent to `make build-all`)
build-all:
	bazel build //...

## Build a single package: make build PKG=humanoid_common_mpc
build:
	@$(if $(PKG),bazel build //humanoid_nmpc/$(PKG) //robot_models/unitree_g1/$(PKG) //robot_runtime/$(PKG) //lib/ocs2_ros2:$(PKG) 2>/dev/null || \
		echo "Trying direct target..." && bazel build //$(PKG), \
		@echo "Please specify a package: make build PKG=package_name")

## Debug build
build-debug:
	bazel build //... --config=dbg

## Release build (no debug info, tests off)
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
	bazel test //...

## Run a single test package: make test PKG=humanoid_centroidal_mpc_test
test:
	@$(if $(PKG),bazel test //humanoid_nmpc/$(PKG)/... 2>/dev/null || \
		bazel test //$(PKG)/..., \
		@echo "Please specify a package: make test PKG=package_name")

## Run OCS2 library tests
run-ocs2-tests:
	bazel test //lib/ocs2_ros2:all

## Run MPC tests
run-mpc-tests:
	bazel test //humanoid_nmpc/...

############################################################
# Utility targets
############################################################

## List all Bazel targets
echo-packages:
	@bazel query '//...' 2>/dev/null | sort

## Clean Bazel cache
clean:
	bazel clean

## Deep clean (remove entire Bazel cache)
clean-ws:
	bazel clean --expunge

## Clean CppAD generated code
clean-cppad:
	rm -rf cppad_code_gen

## Format source code
format:
	find . -name "lib" -prune -o \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -print | xargs clang-format -i && \
	black . --exclude="lib/"

## Update git submodules
update-submodules:
	git submodule update --init --recursive

## Pull git-lfs files
git-lfs:
	git lfs install && git lfs pull

############################################################
# Launch targets (still use ROS2 launch — Bazel builds only)
############################################################

# Helper to source ROS2 and colcon install
define ros2_launch
	source $(ros_source_file) && \
	source $(build_dir)/install/setup.bash && \
	ros2 launch $(1) $(2)
endef

launch-g1-dummy-sim:
	@cd $(build_dir) && $(call ros2_launch,g1_centroidal_mpc,dummy_sim.launch.py)

launch-g1-sim:
	@cd $(build_dir) && $(call ros2_launch,g1_centroidal_mpc,mujoco_sim.launch.py)

launch-wb-g1-dummy-sim:
	@cd $(build_dir) && $(call ros2_launch,g1_wb_mpc,dummy_sim.launch.py)

launch-wb-g1-sim:
	@cd $(build_dir) && $(call ros2_launch,g1_wb_mpc,mujoco_sim.launch.py)

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
VNC_GL_ENV := export DISPLAY=:1 && \
	export LIBGL_ALWAYS_SOFTWARE=1 && \
	export LIBGL_ALWAYS_INDIRECT=0 && \
	export GALLIUM_DRIVER=llvmpipe && \
	export MESA_GL_VERSION_OVERRIDE=3.3 && \
	export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe

launch-g1-dummy-sim-vnc: start-vnc
	@cd $(build_dir) && $(VNC_GL_ENV) && $(call ros2_launch,g1_centroidal_mpc,dummy_sim.launch.py)

launch-g1-sim-vnc: start-vnc
	@cd $(build_dir) && $(VNC_GL_ENV) && $(call ros2_launch,g1_centroidal_mpc,mujoco_sim.launch.py)

launch-wb-g1-dummy-sim-vnc: start-vnc
	@cd $(build_dir) && $(VNC_GL_ENV) && $(call ros2_launch,g1_wb_mpc,dummy_sim.launch.py)

launch-wb-g1-sim-vnc: start-vnc
	@cd $(build_dir) && $(VNC_GL_ENV) && $(call ros2_launch,g1_wb_mpc,mujoco_sim.launch.py)
