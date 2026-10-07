SHELL := /bin/bash

############################################################
# Bazel Build System — Monorepo Makefile
############################################################

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_path := $(dir $(mkfile_path))

# Source the Bazel environment (setup_env.sh)
source_env := source $(current_path)/setup_env.sh

# The launch targets stop their processes themselves (tools/launch tears down every process group it started, and
# tools/deploy/session.sh the robot container), so an interrupted target leaves nothing running.

############################################################
# Build targets
############################################################
.PHONY: help build-all build-debug build-release build-relwithdebinfo build \
        test-all test test-heuristic-parameters derive-heuristic-parameters clean clean-all format lint ci-local \
        lint-tidy lint-tidy-fix \
        test-pinocchio-model-atlas test-pinocchio-model-r1 test-pinocchio-model-sa01 \
        start-vnc stop-vnc kill-sims check-zombies \
        deploy-robot robot-images rerun-viewer rerun-web ipc-list ipc-echo ipc-hz \
        closed-loop-metrics benchmark-mpc-solve \
        run-ocs2-tests run-mpc-tests test-rl train-rl train-cartpole train-cartpole-vnc train-bc export-rollouts lock-rl-deps echo-packages update-submodules install-hooks train-acom-jupyter
# The launch targets are declared .PHONY where the robot table below generates them.

## Launch ACoM SIREN training notebook in Jupyter Lab
train-acom-jupyter:
	@tools/launch_jupyter.sh

## Stop what a launch target left running: the laptop side's launcher (the PID it left in .deploy/ of this checkout,
## never a process of another checkout or test), the simulation's robot container, and a NETEM qdisc on lo.
## This cannot remove zombies: a zombie has already exited and only its parent can reap it. See check-zombies.
kill-sims:
	@echo "🧹 Cleaning up previous sim processes..."
	@$(session) stop
	@echo "✅ Cleanup done."
	@$(MAKE) --no-print-directory check-zombies WARN_ONLY=1

## Report zombie processes and which parent is holding them.
## A zombie has already exited and only its parent can reap it, so kill/pkill have no effect; that is why kill-sims
## never cleared them. Docker's init (tini) reaps orphans, which is why docker-compose.yaml sets `init: true`. Run
## this inside the dev container for a container-local view.
check-zombies:
	@z=$$(ps -eo stat --no-headers 2>/dev/null | awk '/^Z/ {n++} END {print n+0}'); \
	if [ "$$z" -eq 0 ]; then \
		[ -n "$(WARN_ONLY)" ] || echo "✅ No zombie processes."; \
	else \
		echo "⚠️  $$z zombie process(es). They cannot be killed; only their parent can reap them."; \
		echo "   Held by:"; \
		ps -eo ppid,stat --no-headers | awk '$$2 ~ /^Z/ {print $$1}' | sort | uniq -c | sort -rn | head -3 | \
		while read -r n ppid; do \
			echo "     $$n under PID $$ppid ($$(ps -p $$ppid -o comm= 2>/dev/null || echo '<gone>'))"; \
		done; \
		echo "   If the holder is the container's PID 1 (sleep/bash), it never calls wait() and they are permanent."; \
		echo "   docker-compose.yaml now sets 'init: true' so tini becomes PID 1 and reaps orphans."; \
		echo "   Apply it and clear the backlog: Dev Containers 'Rebuild Container', or on the host"; \
		echo "     tools/resource_limits/set_container_memory_limit.sh && docker compose -f docker-compose.yaml up -d --force-recreate"; \
	fi

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

## Train Cartpole RL example with visual progress rendering
train-cartpole:
	bazel run //humanoid_learning/examples:train_cartpole

## Train Cartpole RL with interactive 3D MuJoCo Viewer in VNC
train-cartpole-vnc: start-vnc
	$(VNC_GL_ENV) && bazel run //humanoid_learning/examples:train_cartpole -- --vnc

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
	bazel run //robot_models/drc_atlas/drc_atlas_centroidal_mpc:test_pinocchio_model

## Run Pinocchio Model R1 test
test-pinocchio-model-r1:
	bazel run //robot_models/unitree_r1/unitree_r1_centroidal_mpc:test_pinocchio_model

## Run Pinocchio Model EngineAI SA01 test
test-pinocchio-model-sa01:
	bazel run //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc:test_pinocchio_model

############################################################
# Utility targets
############################################################

## List all Bazel targets
echo-packages:
	@bazel query '//...' 2>/dev/null | sort

## Clean Bazel cache
clean:
	bazel clean

## Deep clean (remove the entire Bazel cache of this checkout)
clean-all:
	bazel clean --expunge

## Format source code (C++, Python, trailing newlines, and whitespace)
format:
	@python3 -m tools.hooks.format_code

## Lint repository: IFTTT, formatting, the token checks of tools/hooks/checks.py, cpplint, pylint and mypy
lint:
	@python3 -m tools.hooks.lint_code

# clang-tidy runs as a Bazel aspect (tools/clang_tidy/README.md), inside .bazelrc's RAM-bounded --jobs and tools/bazel's
# machine lock: never by hand. Each target prints the findings of the build's reports even when an action failed, and
# fails when either did. The build event file names exactly this build's reports. The report is read after the build
# has released the machine lock, so each run writes its own file (named after the shell's PID, and deleted afterwards)
# and a concurrent run cannot overwrite it; CLANG_TIDY_BEP=<path> keeps the file of a run.
# LINT.IfChange(lint_tidy)
CLANG_TIDY_BEP ?=
clang_tidy_bep := bep="$(or $(CLANG_TIDY_BEP),.bazel/clang_tidy_bep.$$$$.json)"
clang_tidy_bep_cleanup := $(if $(CLANG_TIDY_BEP),,rm -f "$$bep";)

## Lint C++ with clang-tidy as a Bazel aspect (RAM-bounded, machine lock), its self-test included:
## make lint-tidy [PKG=//humanoid_nmpc/...]
lint-tidy:
	@$(source_env) && mkdir -p .bazel && status=0 && $(clang_tidy_bep); \
	bazel build --config=clang-tidy --keep_going --build_event_json_file="$$bep" \
		$(or $(PKG),//...) //tools/clang_tidy:selftest || status=$$?; \
	python3 -m tools.clang_tidy.clang_tidy_report "$$bep" || status=$$?; \
	$(clang_tidy_bep_cleanup) exit $$status

## Apply clang-tidy's fix-its, all or nothing per diagnostic, and clang-format the edited files; build afterwards:
## make lint-tidy-fix CHECKS='modernize-use-override' [PKG=//humanoid_nmpc/...] [PATHS="dir ..."]
lint-tidy-fix:
	@$(if $(CHECKS),,$(error lint-tidy-fix needs CHECKS=<glob>, e.g. CHECKS=modernize-use-override))
	@$(source_env) && mkdir -p .bazel && status=0 && $(clang_tidy_bep); \
	bazel build --config=clang-tidy-fix --keep_going --build_event_json_file="$$bep" \
		$(or $(PKG),//...) || status=$$?; \
	python3 -m tools.clang_tidy.clang_tidy_apply "$$bep" --checks '$(CHECKS)' \
		$(if $(PATHS),--paths $(PATHS)) || status=$$?; \
	$(clang_tidy_bep_cleanup) exit $$status
# LINT.ThenChange(//.bazelrc:clang_tidy_config, //AGENTS.md:style_commands)

############################################################
# Closed-loop metrics and the solve benchmark (humanoid_nmpc/humanoid_mpc_validation/README.md)
############################################################
# Both run under `bazel test`, so that tools/bazel's machine lock keeps them from running next to another build: each
# run compiles CppAD libraries and simulates for minutes. One robot at a time. The commit and the worktree state go into
# every document; pass VALIDATION_GIT_COMMIT and VALIDATION_WORKTREE_STATE where git cannot see the checkout.
# LINT.IfChange(closed_loop_targets)
VALIDATION_PKG := humanoid_nmpc/humanoid_mpc_validation
# Both are computed in the recipe, by the bash that already runs it: a $(shell) starts a bash of its own, which in the dev
# container prints the environment banner (BASH_ENV) into the value, before any `| tail` can drop it. The worktree state
# hashes the diff, the status and the contents of every untracked file (tools/worktree_state.sh); POSIX sh prints none.
VALIDATION_GIT_COMMIT ?=
VALIDATION_WORKTREE_STATE ?=
validation_provenance = commit="$(VALIDATION_GIT_COMMIT)"; [ -n "$$commit" ] || commit=$$(git rev-parse HEAD 2>/dev/null || echo unknown); \
	state="$(VALIDATION_WORKTREE_STATE)"; [ -n "$$state" ] || state=$$(sh $(VALIDATION_PKG)/tools/worktree_state.sh);
validation_env = --test_env=WB_VALIDATION_GIT_COMMIT="$$commit" --test_env=WB_VALIDATION_WORKTREE_STATE="$$state" \
	$(if $(VALIDATION_MACHINE),--test_env=WB_VALIDATION_MACHINE="$(VALIDATION_MACHINE)") \
	--nozip_undeclared_test_outputs --cache_test_results=no --test_output=errors

## Closed-loop metrics of one robot: make closed-loop-metrics ROBOT=<robot> LABEL=M0 [SCENARIO=walk_0p5] [BASELINE=M0] [RECORD_STATES=1]
# ROBOT: drc_atlas, engineai_sa01, unitree_g1, unitree_r1 or unitree_g1_wb. Without SCENARIO every scenario runs. The
# metrics land in data/closed_loop/$(LABEL)/ with the time series of the turning scenarios, which the 720-degree turn
# exception of a later comparison reads; RECORD_STATES=1 also keeps the walking states for the solve benchmark.
closed-loop-metrics:
	@$(if $(and $(ROBOT),$(LABEL)),,$(error Usage: make closed-loop-metrics ROBOT=drc_atlas|engineai_sa01|unitree_g1|unitree_r1|unitree_g1_wb LABEL=<label> [SCENARIO=<scenario>] [BASELINE=<label>] [RECORD_STATES=1]))
	@$(validation_provenance) outputs=.bazel/testlogs/$(VALIDATION_PKG)/closed_loop_$(ROBOT)/test.outputs; \
	bazel test //$(VALIDATION_PKG):closed_loop_$(ROBOT) $(validation_env) --test_arg=--label=$(LABEL) \
		$(if $(SCENARIO),--test_filter='*/$(SCENARIO)') $(if $(BASELINE),--test_arg=--baseline=$(BASELINE)); status=$$?; \
	mkdir -p $(VALIDATION_PKG)/data/closed_loop/$(LABEL); \
	install -m 644 $$outputs/*.json $(VALIDATION_PKG)/data/closed_loop/$(LABEL)/ 2>/dev/null; \
	for series in $$outputs/*_turn_*_timeseries.txt $$outputs/*_arc_timeseries.txt; do \
		[ -f "$$series" ] && install -m 644 "$$series" $(VALIDATION_PKG)/data/closed_loop/$(LABEL)/; \
	done; \
	$(if $(RECORD_STATES),install -D -m 644 -t $(VALIDATION_PKG)/data/benchmark/states/ $$outputs/*_states.txt 2>/dev/null;) \
	echo "Metrics in $(VALIDATION_PKG)/data/closed_loop/$(LABEL)/, time series in $$outputs/."; exit $$status

## Solve benchmark of one robot: make benchmark-mpc-solve ROBOT=<robot> LABEL=B0 [STATES=<recorded states>] [BASELINE=B0]
# Replays data/benchmark/states/<robot>_<walking scenario>_states.txt by default: walk_0p5, or walk_0p3 for EngineAI SA01
# and Unitree R1 (RobotConfiguration::walkingScenario). The document lands in data/benchmark/$(LABEL)/; with BASELINE the
# run is held to the real-time gate of section 4.6 against data/benchmark/$(BASELINE)/.
benchmark-mpc-solve:
	@$(if $(and $(ROBOT),$(LABEL)),,$(error Usage: make benchmark-mpc-solve ROBOT=drc_atlas|engineai_sa01|unitree_g1|unitree_r1|unitree_g1_wb LABEL=<label> [STATES=<file>] [BASELINE=<label>]))
	@$(validation_provenance) outputs=.bazel/testlogs/$(VALIDATION_PKG)/benchmark_mpc_solve_$(ROBOT)/test.outputs; \
	bazel test //$(VALIDATION_PKG):benchmark_mpc_solve_$(ROBOT) $(validation_env) --test_arg=--label=$(LABEL) \
		$(if $(STATES),--test_arg=--states=$(STATES)) $(if $(BASELINE),--test_arg=--baseline=$(BASELINE)); status=$$?; \
	mkdir -p $(VALIDATION_PKG)/data/benchmark/$(LABEL); \
	install -m 644 $$outputs/*.json $(VALIDATION_PKG)/data/benchmark/$(LABEL)/ 2>/dev/null; \
	echo "Benchmark in $(VALIDATION_PKG)/data/benchmark/$(LABEL)/."; exit $$status
# LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/RobotConfiguration.h:robot_configurations, //humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/LockstepOutputs.h:output_names, //humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/io/RunProvenance.h:provenance_environment, //humanoid_nmpc/humanoid_mpc_validation/exe/benchmarkMpcSolveMain.cpp:benchmark_paths)

## Check the locomotion-heuristic coefficients against the robots' URDFs (humanoid_nmpc/docs/locomotion_heuristics)
# Not a bazel target: Pinocchio reaches Python through robotpkg's bindings in /opt/openrobots (setup_env.sh puts them on
# PYTHONPATH), which bazel's hermetic toolchain cannot see.
# The script reads the configuration textprotos with config_textproto (the standard library only, so the system Python
# of robotpkg's pinocchio can run it), which lives under its Bazel import root.
HEURISTICS_PYTHONPATH := $(CURDIR)/humanoid_nmpc/humanoid_mpc_config/python$(if $(PYTHONPATH),:$(PYTHONPATH))
test-heuristic-parameters:
	@PYTHONPATH="$(HEURISTICS_PYTHONPATH)" python3 -m unittest discover -s tools/locomotion_heuristics -p "test_*.py" -v

## Print the derived locomotion-heuristic block for one robot: make derive-heuristic-parameters ROBOT=drc_atlas
# LINT.IfChange(derive_heuristic_parameters_usage)
derive-heuristic-parameters:
	@$(if $(ROBOT),PYTHONPATH="$(HEURISTICS_PYTHONPATH)" python3 tools/locomotion_heuristics/derive_parameters.py --robot $(ROBOT),\
		@echo "Usage: make derive-heuristic-parameters ROBOT=drc_atlas|engineai_sa01|unitree_g1|unitree_r1")
# LINT.ThenChange(//tools/locomotion_heuristics/derive_parameters.py:derive_parameters_robots)

## Install git pre-commit hook
# A symlink rather than a copy, so that a change to tools/hooks/pre-commit takes effect without reinstalling.
install-hooks:
	@chmod +x tools/hooks/pre-commit && \
	ln -sfn ../../tools/hooks/pre-commit .git/hooks/pre-commit && \
	echo "✅ Git pre-commit hook installed successfully (a symlink to tools/hooks/pre-commit)."

## Update git submodules (tools/ifttt-lint)
update-submodules:
	git submodule update --init --recursive

############################################################
# VNC visualization (for macOS host)
############################################################
# LINT.IfChange(vnc_resolution)
RESOLUTION ?= $(VNC_RESOLUTION)
# LINT.ThenChange(//.devcontainer/start_vnc.sh:vnc_resolution, //docker-compose.yaml:vnc_resolution)
# In the dev container, also when make runs on the host: the VNC desktop is the dev container's.
start-vnc:
	@$(dev_run) 'chmod +x .devcontainer/start_vnc.sh && .devcontainer/start_vnc.sh $(RESOLUTION)'

stop-vnc:
	@$(dev_run) 'chmod +x .devcontainer/start_vnc.sh && .devcontainer/start_vnc.sh stop'

# LINT.IfChange(vnc_ports)
# Environment overrides for VNC display + Mesa software GLX
VNC_GL_ENV := export DISPLAY=:99 && \
	export LIBGL_ALWAYS_SOFTWARE=1 && \
	export LIBGL_ALWAYS_INDIRECT=0 && \
	export GALLIUM_DRIVER=llvmpipe && \
	export MESA_GL_VERSION_OVERRIDE=3.3 && \
	export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
# Printed right before a -vnc target launches, so the URLs are not buried under the build output.
vnc_urls := echo "🖥️  noVNC: http://localhost:6080/vnc.html (MuJoCo viewer & operator GUI)" && \
	echo "🌐 Rerun: http://localhost:9090/?url=rerun+http://localhost:9876/proxy (host browser; set RERUN_SINK=spawn to view in noVNC)"
# LINT.ThenChange(//.devcontainer/start_vnc.sh:vnc_ports, //docker-compose.bridge.yaml:vnc_ports, //.devcontainer/devcontainer.json:vnc_ports, //.devcontainer/README.md:vnc_ports)


############################################################
# Launch and deployment (humanoid_nmpc/docs/distributed_runtime/README.md, tools/deploy/README.md)
############################################################
# Simulation runs the hardware topology. The robot side - the robot process, with MuJoCo as its backend - runs in the
# robot-sim container: the robot's own image (robot-runtime) and compose file (docker-compose.robot.yaml), with its
# realtime settings, plus only the viewer's GL. The laptop side - the MPC node, the remote-control GUI and the Rerun
# bridge - runs in the dev container, over the remote MPC link on the bus. Nothing runs the MPC inside the robot
# process. The robot side needs the host's Docker, so `make launch-<robot>-sim` runs on the host (Linux), or in a dev
# container that can reach Docker; the laptop side and the Bazel builds then run in the dev container DEV_CONTAINER.
#
#   make launch-drc-atlas-sim                 # the viewer and the GUI on this shell's DISPLAY (default :99)
#   make launch-drc-atlas-sim-vnc             # ... on the VNC desktop :99, started first
#   make launch-drc-atlas-sim NETEM="delay 3ms 1ms loss 0.5%"    # the bus over a lossy, delayed link (tc netem)
#   make launch-drc-atlas-dummy-sim           # the MPC against the dummy simulator (the MPC model's rollout)
#   make deploy-robot ROBOT=drc_atlas HOST=<robot> NETWORK=<file> [SERVICE=enable]   # the robot's computer
#   make launch-drc-atlas-robot HOST=<robot>  # the deployed robot side, in the foreground over ssh
#   make launch-drc-atlas-mpc NETWORK=<file>  # the laptop side against it
#   make launch-drc-atlas-sandbox             # the URDF in Rerun, with a slider per joint
#
# A launch target's variables:
#   RERUN_SINK=serve_web|spawn|connect|save   the Rerun bridge's sink (default serve_web: the
#                                             web viewer at http://localhost:9090; spawn: the native viewer; save: to RRD_PATH=<file>.rrd)
#   NETEM="<netem parameters>"                -sim: tc netem on the bus's packets on lo (default: none)
#   HEADLESS=true                             -sim: the robot process without the MuJoCo viewer
#   NETWORK=<network file>                    -mpc: the network file with the robot's address (required); a path
#                                             inside the checkout, relative to it or absolute: deploy-robot reads it on
#                                             the host, the launch targets in the dev container, which sees the checkout
#                                             only
#   JOINT_SOURCE=sliders|nominal              -sandbox: where the joint positions come from (default sliders)
#   DEV_CONTAINER=<name>                      the dev container, when make runs on the host (default devcontainer-app-1)
#   PLOT_CONFIG=<file>                        the Rerun bridge's plot configuration (default:
#                                             humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto)
#
# The sandbox replaces RViz's display launch files: the model sandbox publishes the URDF's link poses at the nominal
# joint positions, or at those of its Tk window with a slider per joint (joint_state_publisher_gui's), and the Rerun
# bridge draws them. RViz's interactive markers and its other displays are not reproduced.

DEV_CONTAINER ?= devcontainer-app-1
export WB_DEV_CONTAINER := $(DEV_CONTAINER)
RERUN_SINK ?= serve_web
NETEM ?=
HEADLESS ?=
NETWORK ?=
JOINT_SOURCE ?= sliders
RRD_PATH ?=
PLOT_CONFIG ?= humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto
# The Rerun bridge's sink, and the recording of RERUN_SINK=save (relative to the checkout, where the bridge runs).
rerun_sets = --set rerun_sink=$(RERUN_SINK) $(if $(RRD_PATH),--set rrd_path=$(RRD_PATH)) $(if $(PLOT_CONFIG),--set plot_config=$(PLOT_CONFIG))
# deploy-robot and launch-<robot>-robot
ROBOT ?=
HOST ?= localhost
SERVICE ?= none
REGISTRY ?=
PLATFORM ?=
BACKEND ?=
MEMORY_LIMIT ?=
CPUSET ?=
REALTIME_PRIORITY ?=
REALTIME_CORES ?=
BACKEND_CORES ?=
CONFIG_SEED ?=
NETEM_INTERFACE ?= lo
DEPLOY_DIR ?= wb-humanoid-robot

# NETWORK= is a path relative to the checkout, where deploy_robot.sh (on the host) and the launcher (in the dev
# container) both resolve it, or an absolute path inside the checkout, which is made relative. One outside it is refused:
# the dev container sees nothing else of the host.
checkout_root := $(abspath $(current_path))
network_file = $(if $(filter /%,$(NETWORK)),$(if $(filter $(checkout_root)/%,$(NETWORK)),$(patsubst $(checkout_root)/%,%,$(NETWORK)),$(error NETWORK=$(NETWORK) is outside the checkout $(checkout_root): the dev container sees only the checkout, so copy the file into it, e.g. config/ipc/my_network.textproto)),$(NETWORK))

# A shell command in the dev container's checkout: here inside it, through docker exec from the host.
dev_run := $(current_path)tools/deploy/in_dev_container.sh
session := $(current_path)tools/deploy/session.sh
deploy_robot := $(current_path)tools/deploy/deploy_robot.sh
deploy_flags = --host $(HOST) --deploy_dir $(DEPLOY_DIR) \
	$(if $(NETWORK),--network $(network_file)) $(if $(REGISTRY),--registry $(REGISTRY)) \
	$(if $(PLATFORM),--platform $(PLATFORM)) $(if $(CPUSET),--cpuset $(CPUSET)) \
	$(if $(BACKEND),--backend $(BACKEND)) $(if $(MEMORY_LIMIT),--memory_limit $(MEMORY_LIMIT)) \
	$(if $(REALTIME_PRIORITY),--realtime_priority $(REALTIME_PRIORITY)) \
	$(if $(REALTIME_CORES),--realtime_cores $(REALTIME_CORES)) \
	$(if $(BACKEND_CORES),--backend_cores $(BACKEND_CORES)) $(if $(CONFIG_SEED),--config_seed $(CONFIG_SEED)) \
	$(if $(NETEM),--netem "$(NETEM)" --netem_interface $(NETEM_INTERFACE))

# The laptop side's binaries, per formulation, and the bridge and the GUI every launch file starts: the processes of the
# launch files (mpc.textproto, dummy_sim.textproto, sandbox.textproto), which the launch targets build first.
# LINT.IfChange(launch_binaries)
laptop_binaries := //humanoid_nmpc/remote_control:base_velocity_controller_gui //humanoid_nmpc/humanoid_rerun_viewer
mpc_binaries_centroidal := //humanoid_nmpc/humanoid_centroidal_mpc_app:humanoid_centroidal_mpc_node
dummy_binaries_centroidal := $(mpc_binaries_centroidal) //humanoid_nmpc/humanoid_centroidal_mpc_app:humanoid_centroidal_mpc_dummy_sim
mpc_binaries_wb := //humanoid_nmpc/humanoid_wb_mpc_app:humanoid_wb_mpc_node
dummy_binaries_wb := $(mpc_binaries_wb) //humanoid_nmpc/humanoid_wb_mpc_app:humanoid_wb_mpc_dummy_sim
sandbox_binaries := //humanoid_nmpc/humanoid_rerun_viewer:model_sandbox //humanoid_nmpc/humanoid_rerun_viewer
# LINT.ThenChange(//tools/deploy/test_deploy_files.py:launch_binaries)

# The robot configurations: <target name> <deployment name (ROBOT=)> <MPC package> <formulation> <description package>.
# LINT.IfChange(robot_configurations)
ROBOT_CONFIGURATIONS := \
	g1:unitree_g1:robot_models/unitree_g1/g1_centroidal_mpc:centroidal:robot_models/unitree_g1/g1_description \
	wb-g1:unitree_g1_wb:robot_models/unitree_g1/g1_wb_mpc:wb:robot_models/unitree_g1/g1_description \
	drc-atlas:drc_atlas:robot_models/drc_atlas/drc_atlas_centroidal_mpc:centroidal:robot_models/drc_atlas/drc_atlas_description \
	r1:unitree_r1:robot_models/unitree_r1/unitree_r1_centroidal_mpc:centroidal:robot_models/unitree_r1/unitree_r1_description \
	sa01:engineai_sa01:robot_models/engineai_sa01/engineai_sa01_centroidal_mpc:centroidal:robot_models/engineai_sa01/engineai_sa01_description
# The robots with a sandbox target: one per description (the whole-body G1 shares the G1's).
SANDBOX_ROBOTS := g1 drc-atlas r1 sa01
# LINT.ThenChange(//tools/deploy/BUILD.bazel:robot_configurations, //robot_models/tests/test_launch_files.py:robot_configurations, //.devcontainer/README.md:launch_targets, //README.md:launch_targets)

field = $(word $(2),$(subst :, ,$(1)))

# LINT.IfChange(launch_targets)
# The launch targets of one robot configuration ($(1) is its entry of ROBOT_CONFIGURATIONS): the launch files of its
# MPC package's launch/ directory (robot.textproto, mpc.textproto, dummy_sim.textproto).
define robot_launch_targets
.PHONY: launch-$(call field,$(1),1)-sim launch-$(call field,$(1),1)-sim-vnc launch-$(call field,$(1),1)-dummy-sim \
	launch-$(call field,$(1),1)-dummy-sim-vnc launch-$(call field,$(1),1)-robot launch-$(call field,$(1),1)-mpc

launch-$(call field,$(1),1)-sim: kill-sims
	@$(session) sim --robot $(call field,$(1),2) --launch_file $(call field,$(1),3)/launch/mpc.textproto \
		--build "$(mpc_binaries_$(call field,$(1),4)) $(laptop_binaries)" $$(rerun_sets) \
		$$(if $$(NETEM),--netem "$$(NETEM)") $$(if $$(filter true,$$(HEADLESS)),--headless)

launch-$(call field,$(1),1)-sim-vnc: kill-sims start-vnc
	@$$(vnc_urls)
	@DISPLAY=:99 $(session) sim --robot $(call field,$(1),2) --launch_file $(call field,$(1),3)/launch/mpc.textproto \
		--build "$(mpc_binaries_$(call field,$(1),4)) $(laptop_binaries)" $$(rerun_sets) \
		$$(if $$(NETEM),--netem "$$(NETEM)") $$(if $$(filter true,$$(HEADLESS)),--headless)

launch-$(call field,$(1),1)-dummy-sim: kill-sims
	@$(session) laptop --launch_file $(call field,$(1),3)/launch/dummy_sim.textproto \
		--build "$(dummy_binaries_$(call field,$(1),4)) $(laptop_binaries)" $$(rerun_sets)

launch-$(call field,$(1),1)-dummy-sim-vnc: kill-sims start-vnc
	@$$(vnc_urls)
	@DISPLAY=:99 $(session) laptop --launch_file $(call field,$(1),3)/launch/dummy_sim.textproto \
		--build "$(dummy_binaries_$(call field,$(1),4)) $(laptop_binaries)" $$(rerun_sets)

launch-$(call field,$(1),1)-robot:
	@$(deploy_robot) up --robot $(call field,$(1),2) --host $$(HOST) --deploy_dir $$(DEPLOY_DIR)

launch-$(call field,$(1),1)-mpc: kill-sims
	@$$(if $$(NETWORK),,$$(error launch-$(call field,$(1),1)-mpc needs NETWORK=<network file> with the robot's address, e.g. a copy of config/ipc/two_machine.example.textproto))
	@$(session) laptop --launch_file $(call field,$(1),3)/launch/mpc.textproto \
		--build "$(mpc_binaries_$(call field,$(1),4)) $(laptop_binaries)" $$(rerun_sets) \
		--set network_file=$$(network_file)
endef

# The sandbox of one robot description ($(1) is its entry of ROBOT_CONFIGURATIONS): launch/sandbox.textproto of it.
define robot_sandbox_targets
.PHONY: launch-$(call field,$(1),1)-sandbox launch-$(call field,$(1),1)-sandbox-vnc

launch-$(call field,$(1),1)-sandbox: kill-sims
	@$(session) laptop --launch_file $(call field,$(1),5)/launch/sandbox.textproto --build "$(sandbox_binaries)" \
		$$(rerun_sets) --set joint_source=$$(JOINT_SOURCE)

launch-$(call field,$(1),1)-sandbox-vnc: kill-sims start-vnc
	@$$(vnc_urls)
	@DISPLAY=:99 $(session) laptop --launch_file $(call field,$(1),5)/launch/sandbox.textproto \
		--build "$(sandbox_binaries)" $$(rerun_sets) --set joint_source=$$(JOINT_SOURCE)
endef

$(foreach configuration,$(ROBOT_CONFIGURATIONS),$(eval $(call robot_launch_targets,$(configuration))))
$(foreach configuration,$(filter $(addsuffix :%,$(SANDBOX_ROBOTS)),$(ROBOT_CONFIGURATIONS)),\
	$(eval $(call robot_sandbox_targets,$(configuration))))
# LINT.ThenChange(//.devcontainer/README.md:launch_targets, //README.md:launch_targets, //tools/deploy/test_deploy_files.py:launch_targets)

## Deploy the robot side to the robot's computer: make deploy-robot ROBOT=drc_atlas HOST=<ssh host> NETWORK=<file>
## [SERVICE=install|enable] [REGISTRY=<registry>] [PLATFORM=linux/arm64] [BACKEND=<backend name>] [MEMORY_LIMIT=4g]
## [CPUSET=2-5] [REALTIME_PRIORITY=80] [NETEM="delay 3ms"]. HOST=localhost deploys to this machine without ssh
## (tools/deploy/README.md). A deployment that runs already is restarted on the new image.
deploy-robot:
	@$(if $(ROBOT),,$(error deploy-robot needs ROBOT=<robot configuration>: drc_atlas, engineai_sa01, unitree_g1, unitree_g1_wb or unitree_r1))
	@$(deploy_robot) deploy --robot $(ROBOT) --service $(SERVICE) $(deploy_flags)

## Build the robot bundle and the robot-runtime and robot-sim images, without deploying them
robot-images:
	@$(deploy_robot) image --target robot-sim $(if $(PLATFORM),--platform $(PLATFORM))

## The Rerun bridge on its own, with the native viewer (rerun-viewer) or the web viewer at http://localhost:9090
## (rerun-web): make rerun-viewer [ROBOT=drc_atlas] [NETWORK=<file>]. Draws the robot only with ROBOT=.
rerun_urdf_drc_atlas := robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf
rerun_urdf_engineai_sa01 := robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf
rerun_urdf_unitree_g1 := robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf
rerun_urdf_unitree_g1_wb := $(rerun_urdf_unitree_g1)
rerun_urdf_unitree_r1 := robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf
rerun_flags = $(if $(ROBOT),--urdf=$(rerun_urdf_$(ROBOT))) $(if $(NETWORK),--network_config=$(network_file))
rerun-viewer:
	@$(dev_run) 'bazel run //humanoid_nmpc/humanoid_rerun_viewer -- $(rerun_flags) --rerun_sink=spawn'

rerun-web:
	@echo "🌐 Rerun web viewer: http://localhost:9090/?url=rerun+http://localhost:9876/proxy"
	@$(dev_run) 'bazel run //humanoid_nmpc/humanoid_rerun_viewer -- $(rerun_flags) --rerun_sink=serve_web'

## Inspect the bus (tools/ipc): make ipc-list, make ipc-echo TOPIC=mpc/status, make ipc-hz TOPIC=robot/mpc_observation
ipc_network = $(if $(NETWORK),--network_config=$(network_file))
ipc-list:
	@$(dev_run) 'bazel run //tools/ipc:ipc_tool -- list $(ipc_network)'

ipc-echo:
	@$(if $(TOPIC),,$(error ipc-echo needs TOPIC=<topic>, e.g. TOPIC=mpc/status))
	@$(dev_run) 'bazel run //tools/ipc:ipc_tool -- echo $(TOPIC) $(ipc_network)'

ipc-hz:
	@$(if $(TOPIC),,$(error ipc-hz needs TOPIC=<topic>, e.g. TOPIC=robot/mpc_observation))
	@$(dev_run) 'bazel run //tools/ipc:ipc_tool -- hz $(TOPIC) $(ipc_network)'

## The launch, deployment, inspection and style-check targets
help:
	@echo "Launch targets, for <robot> in $(foreach c,$(ROBOT_CONFIGURATIONS),$(call field,$(c),1)):"
	@echo "  launch-<robot>-sim[-vnc]        robot side in the robot-sim container + MPC, GUI and Rerun bridge in the dev"
	@echo "                                  container, over the bus (NETEM=, HEADLESS=true, RERUN_SINK=)"
	@echo "  launch-<robot>-dummy-sim[-vnc]  the MPC against the dummy simulator, with the GUI and the Rerun bridge"
	@echo "  launch-<robot>-robot HOST=      the deployed robot side, in the foreground (ssh to HOST, or localhost)"
	@echo "  launch-<robot>-mpc NETWORK=     the laptop side against the robot of the network file"
	@echo "  launch-<robot>-sandbox[-vnc]    for <robot> in $(SANDBOX_ROBOTS): the URDF in Rerun at its nominal pose or"
	@echo "                                  at a slider per joint (JOINT_SOURCE=sliders|nominal); RViz's interactive"
	@echo "                                  markers and other displays are not reproduced"
	@echo "Deployment: deploy-robot ROBOT= HOST= NETWORK= [SERVICE=install|enable] [REGISTRY=] [PLATFORM=] [BACKEND=]"
	@echo "            [MEMORY_LIMIT=], robot-images"
	@echo "Viewers and the bus: rerun-viewer, rerun-web [ROBOT=] [NETWORK=], ipc-list, ipc-echo TOPIC=, ipc-hz TOPIC="
	@echo "Validation (humanoid_nmpc/humanoid_mpc_validation): closed-loop-metrics ROBOT= LABEL= [SCENARIO=] [BASELINE=]"
	@echo "            [RECORD_STATES=1], benchmark-mpc-solve ROBOT= LABEL= [STATES=] [BASELINE=]"
	@echo "Clean-up: kill-sims, check-zombies"
	@echo "Style checks (AGENTS.md, \"Running the style checks\"; tools/hooks/README.md, tools/clang_tidy/README.md):"
	@echo "  format                          clang-format, black, isort and the enforced checks' safe rewrites"
	@echo "  lint                            every check without a compiler; one check or directory:"
	@echo "                                  python3 -m tools.hooks.lint_code --only <check> --paths <dir>"
	@echo "  lint-tidy [PKG=]                clang-tidy as a Bazel aspect, inside the machine lock"
	@echo "  lint-tidy-fix CHECKS= [PKG=] [PATHS=]        apply clang-tidy's fix-its, then build"
	@echo "  install-hooks                   the pre-commit hook: format, then lint the staged files"
