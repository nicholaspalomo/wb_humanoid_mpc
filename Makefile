SHELL := /bin/bash

############################################################
# Bazel Build System — Monorepo Makefile
############################################################

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_path := $(dir $(mkfile_path))

# Source the Bazel environment (setup_env.sh)
source_env := source $(current_path)/setup_env.sh

# Every target that builds runs Bazel through tools/ide/compile_commands.py --after, which refreshes compile_commands.json,
# the editor's view of the build, after the build (tools/ide/README.md): $(bazel_indexed) in a recipe, and
# $(bazel_indexed_in_repo) in a command for the dev container ($(dev_run)), which starts in the checkout. Only the dev
# container refreshes; in CI and on the host the runner runs the Bazel command alone.
# LINT.IfChange(bazel_indexed)
bazel_indexed_in_repo := python3 -m tools.ide.compile_commands --after bazel
bazel_indexed := cd $(current_path) && $(bazel_indexed_in_repo)
# LINT.ThenChange(//tools/ide/test_ide_setup.py:bazel_indexed)

# The launch targets stop their processes themselves (tools/launch tears down every process group it started, and
# tools/deploy/session.sh the robot container), so an interrupted target leaves nothing running.

############################################################
# Build targets
############################################################
.PHONY: help build-all build-debug build-release build-relwithdebinfo build compile-commands \
        test-all test test-heuristic-parameters derive-heuristic-parameters clean clean-all format lint ci-local \
        lint-tidy lint-tidy-fix \
        test-pinocchio-model-atlas test-pinocchio-model-r1 test-pinocchio-model-sa01 \
        start-vnc stop-vnc kill-sims check-zombies \
        deploy-robot robot-images udp-buffers rerun-viewer rerun-web ipc-list ipc-echo ipc-hz \
        closed-loop-metrics benchmark-mpc-solve compare-closed-loop \
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
	$(source_env) && $(bazel_indexed) build //...

## Build a single target: make build PKG=//humanoid_nmpc/humanoid_common_mpc
build:
	@$(if $(PKG),$(bazel_indexed) build $(PKG),@echo "Usage: make build PKG=//path/to:target")

## Debug build
build-debug:
	$(bazel_indexed) build //... --config=dbg

## Release build
build-release:
	$(bazel_indexed) build //...

## Release with debug info
build-relwithdebinfo:
	$(bazel_indexed) build //... --config=relwithdebinfo

## Refresh compile_commands.json for the editor's clangd, building the generated headers first (tools/ide/README.md).
## The build and test targets refresh it themselves; this is for a checkout nothing has built yet.
compile-commands:
	@$(dev_run) 'python3 -m tools.ide.compile_commands --build_headers'

############################################################
# Test targets
############################################################

## Run all tests, or a batch of them: make test-all TEST_TARGETS="//robot_runtime/... -//robot_runtime/robot_ipc/..."
## (CI runs its package batches this way: .github/workflows/build_test.yml)
TEST_TARGETS ?= //...
test-all:
	$(source_env) && $(bazel_indexed) test -- $(TEST_TARGETS)

## Run local CI emulator (matches GitHub Actions clean container setup)
ci-local:
	@tools/ci_local.sh

## Run a single test: make test PKG=//humanoid_nmpc/humanoid_centroidal_mpc_test:test_pinocchio_frame_conversions
test:
	@$(if $(PKG),$(bazel_indexed) test $(PKG),@echo "Usage: make test PKG=//path/to:test_target")

## Run OCS2 library tests
run-ocs2-tests:
	$(bazel_indexed) test //lib/ocs2:all

## Run MPC tests
run-mpc-tests:
	$(bazel_indexed) test //humanoid_nmpc/...

## Run MuJoCo Playground RL tests
test-rl:
	$(bazel_indexed) test //humanoid_learning/...

## Run MuJoCo Playground RL PPO Training
train-rl:
	$(bazel_indexed) run //humanoid_learning/training:train_ppo

## Train Cartpole RL example with visual progress rendering
train-cartpole:
	$(bazel_indexed) run //humanoid_learning/examples:train_cartpole

## Train Cartpole RL with interactive 3D MuJoCo Viewer in VNC
train-cartpole-vnc: start-vnc
	$(VNC_GL_ENV) && $(bazel_indexed) run //humanoid_learning/examples:train_cartpole -- --vnc

## Behavioral Cloning pretraining on MPC demos
train-bc:
	$(bazel_indexed) run //humanoid_learning/training:bc_warmstart

## Export recorded MPC trajectories to HDF5 demos
export-rollouts:
	python3 humanoid_nmpc/humanoid_common_mpc_pyutils/humanoid_common_mpc_pyutils/export_rollouts.py

## Lock / Update RL pip dependencies
lock-rl-deps:
	$(bazel_indexed) run //humanoid_learning:requirements.update

## Run Pinocchio Model Atlas test
test-pinocchio-model-atlas:
	$(bazel_indexed) run //robot_models/drc_atlas/drc_atlas_centroidal_mpc:test_pinocchio_model

## Run Pinocchio Model R1 test
test-pinocchio-model-r1:
	$(bazel_indexed) run //robot_models/unitree_r1/unitree_r1_centroidal_mpc:test_pinocchio_model

## Run Pinocchio Model EngineAI SA01 test
test-pinocchio-model-sa01:
	$(bazel_indexed) run //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc:test_pinocchio_model

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
	$(bazel_indexed) build --config=clang-tidy --keep_going --build_event_json_file="$$bep" \
		$(or $(PKG),//...) //tools/clang_tidy:selftest || status=$$?; \
	python3 -m tools.clang_tidy.clang_tidy_report "$$bep" || status=$$?; \
	$(clang_tidy_bep_cleanup) exit $$status

## Apply clang-tidy's fix-its, all or nothing per diagnostic, and clang-format the edited files; build afterwards:
## make lint-tidy-fix CHECKS='modernize-use-override' [PKG=//humanoid_nmpc/...] [PATHS="dir ..."]
lint-tidy-fix:
	@$(if $(CHECKS),,$(error lint-tidy-fix needs CHECKS=<glob>, e.g. CHECKS=modernize-use-override))
	@$(source_env) && mkdir -p .bazel && status=0 && $(clang_tidy_bep); \
	$(bazel_indexed) build --config=clang-tidy-fix --keep_going --build_event_json_file="$$bep" \
		$(or $(PKG),//...) || status=$$?; \
	python3 -m tools.clang_tidy.clang_tidy_apply "$$bep" --checks '$(CHECKS)' \
		$(if $(PATHS),--paths $(PATHS)) || status=$$?; \
	$(clang_tidy_bep_cleanup) exit $$status
# LINT.ThenChange(//.bazelrc:clang_tidy_config, //AGENTS.md:style_commands)

############################################################
# Closed-loop metrics and the solve benchmark (humanoid_nmpc/humanoid_mpc_validation/README.md)
############################################################
# Both run under `bazel test`, so that tools/bazel's machine lock keeps them from running next to another build: each
# run compiles CppAD libraries and simulates for minutes. One robot at a time. The commit, the worktree state, the
# machine and the toolchain go into every document; pass VALIDATION_GIT_COMMIT and VALIDATION_WORKTREE_STATE where git
# cannot see the checkout, and VALIDATION_CONTAINER_IMAGE (the container cannot see its image) from the host:
# VALIDATION_CONTAINER_IMAGE=$(docker inspect --format '{{.Image}}' <container>), else the documents say unknown.
# LINT.IfChange(closed_loop_targets)
VALIDATION_PKG := humanoid_nmpc/humanoid_mpc_validation
# Both are computed in the recipe, by the bash that already runs it: a $(shell) starts a bash of its own, which in the dev
# container prints the environment banner (BASH_ENV) into the value, before any `| tail` can drop it. The worktree state
# hashes the diff, the status and the contents of every untracked file ($(VALIDATION_PKG)/tools/worktree_state.sh); POSIX
# sh prints none.
VALIDATION_GIT_COMMIT ?=
VALIDATION_WORKTREE_STATE ?=
VALIDATION_CONTAINER_IMAGE ?=
# HPIPM's build target is read from its repository rule, the one place it is set (-DTARGET=, bazel/system_libs.bzl).
validation_provenance = commit="$(VALIDATION_GIT_COMMIT)"; [ -n "$$commit" ] || commit=$$(git rev-parse HEAD 2>/dev/null || echo unknown); \
	state="$(VALIDATION_WORKTREE_STATE)"; [ -n "$$state" ] || state=$$(sh $(VALIDATION_PKG)/tools/worktree_state.sh); \
	hpipm=$$(sed -n 's/^ *"-DTARGET=\([A-Za-z0-9_]*\)",.*/\1/p' bazel/system_libs.bzl | head -n 1); [ -n "$$hpipm" ] || hpipm=unknown;
validation_env = --test_env=WB_VALIDATION_GIT_COMMIT="$$commit" --test_env=WB_VALIDATION_WORKTREE_STATE="$$state" \
	$(if $(VALIDATION_MACHINE),--test_env=WB_VALIDATION_MACHINE="$(VALIDATION_MACHINE)") \
	--test_env=WB_VALIDATION_HPIPM_TARGET="$$hpipm" \
	$(if $(VALIDATION_CONTAINER_IMAGE),--test_env=WB_VALIDATION_CONTAINER_IMAGE="$(VALIDATION_CONTAINER_IMAGE)") \
	--nozip_undeclared_test_outputs --cache_test_results=no --test_output=errors

## Closed-loop metrics of one robot: make closed-loop-metrics ROBOT=<robot> LABEL=<label> [SCENARIO=walk_0p5] [BASELINE=<label>] [RECORD_STATES=1]
# ROBOT: drc_atlas, engineai_sa01, unitree_g1, unitree_r1 or unitree_g1_wb. Without SCENARIO every scenario runs. The
# metrics land in data/closed_loop/$(LABEL)/ and the time series stay in the test's outputs; RECORD_STATES=1 also keeps
# the walking states for the solve benchmark. With BASELINE each run's violations and improvements against it are printed
# at the end.
closed-loop-metrics:
	@$(if $(and $(ROBOT),$(LABEL)),,$(error Usage: make closed-loop-metrics ROBOT=drc_atlas|engineai_sa01|unitree_g1|unitree_r1|unitree_g1_wb LABEL=<label> [SCENARIO=<scenario>] [BASELINE=<label>] [RECORD_STATES=1]))
	@$(validation_provenance) outputs=.bazel/testlogs/$(VALIDATION_PKG)/closed_loop_$(ROBOT)/test.outputs; \
	$(bazel_indexed) test //$(VALIDATION_PKG):closed_loop_$(ROBOT) $(validation_env) --test_arg=--label=$(LABEL) \
		$(if $(SCENARIO),--test_filter='*/$(SCENARIO)') $(if $(BASELINE),--test_arg=--baseline=$(BASELINE)); status=$$?; \
	mkdir -p $(VALIDATION_PKG)/data/closed_loop/$(LABEL); \
	install -m 644 $$outputs/*.json $(VALIDATION_PKG)/data/closed_loop/$(LABEL)/ 2>/dev/null; \
	$(if $(RECORD_STATES),install -D -m 644 -t $(VALIDATION_PKG)/data/benchmark/states/ $$outputs/*_states.txt 2>/dev/null;) \
	for comparison in $$outputs/*_comparison.txt; do [ -f "$$comparison" ] && cat "$$comparison"; done; \
	echo "Metrics in $(VALIDATION_PKG)/data/closed_loop/$(LABEL)/, time series in $$outputs/."; exit $$status

## Solve benchmark of one robot: make benchmark-mpc-solve ROBOT=<robot> LABEL=<label> [STATES=<recorded states>] [BASELINE=<label>]
# Replays data/benchmark/states/<robot>_<walking scenario>_states.txt by default: walk_0p5, or walk_0p3 for EngineAI SA01
# and Unitree R1 (RobotConfiguration::walkingScenario). The document lands in data/benchmark/$(LABEL)/; with BASELINE the
# run is held to the real-time gate (SolveBenchmarkGate.h, $(VALIDATION_PKG)/README.md) against
# data/benchmark/$(BASELINE)/.
benchmark-mpc-solve:
	@$(if $(and $(ROBOT),$(LABEL)),,$(error Usage: make benchmark-mpc-solve ROBOT=drc_atlas|engineai_sa01|unitree_g1|unitree_r1|unitree_g1_wb LABEL=<label> [STATES=<file>] [BASELINE=<label>]))
	@$(validation_provenance) outputs=.bazel/testlogs/$(VALIDATION_PKG)/benchmark_mpc_solve_$(ROBOT)/test.outputs; \
	$(bazel_indexed) test //$(VALIDATION_PKG):benchmark_mpc_solve_$(ROBOT) $(validation_env) --test_arg=--label=$(LABEL) \
		$(if $(STATES),--test_arg=--states=$(STATES)) $(if $(BASELINE),--test_arg=--baseline=$(BASELINE)); status=$$?; \
	mkdir -p $(VALIDATION_PKG)/data/benchmark/$(LABEL); \
	install -m 644 $$outputs/*.json $(VALIDATION_PKG)/data/benchmark/$(LABEL)/ 2>/dev/null; \
	echo "Benchmark in $(VALIDATION_PKG)/data/benchmark/$(LABEL)/."; exit $$status

## Two recorded labels compared offline: make compare-closed-loop CANDIDATE=<label> BASELINE=<label> [ROBOT=<robot>]
# Every run of data/closed_loop/$(CANDIDATE)/ that data/closed_loop/$(BASELINE)/ has too, held to it as closed-loop-metrics
# with BASELINE holds a run (compareClosedLoopMetrics()), from the archived documents: no simulation. Prints each run's
# violations and improvements; fails when a run has a violation.
compare-closed-loop:
	@$(if $(and $(CANDIDATE),$(BASELINE)),,$(error Usage: make compare-closed-loop CANDIDATE=<label> BASELINE=<label> [ROBOT=drc_atlas|engineai_sa01|unitree_g1|unitree_r1|unitree_g1_wb]))
	@$(bazel_indexed) test //$(VALIDATION_PKG):compare_closed_loop_labels --test_arg=--candidate=$(CANDIDATE) --test_arg=--baseline=$(BASELINE) \
		$(if $(ROBOT),--test_arg=--robot=$(ROBOT)) --cache_test_results=no --test_output=all
# LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/RobotConfiguration.h:robot_configurations, //humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/closed_loop/LockstepOutputs.h:output_names, //humanoid_nmpc/humanoid_mpc_validation/include/humanoid_mpc_validation/io/RunProvenance.h:provenance_environment, //humanoid_nmpc/humanoid_mpc_validation/exe/benchmarkMpcSolveMain.cpp:benchmark_paths, //humanoid_nmpc/humanoid_mpc_validation/exe/compareClosedLoopMain.cpp:comparison_paths, //humanoid_nmpc/humanoid_mpc_validation/test/closedLoopScenarios.cpp:comparison_file)

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
# LINT.ThenChange(//.devcontainer/start_vnc.sh:vnc_ports, //docker-compose.bridge.yaml:vnc_ports, //.devcontainer/devcontainer.json:vnc_ports, //.devcontainer/README.md:vnc_ports, //tools/deploy/session.sh:views_urls)


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
#   make launch-drc-atlas-sim                 # three browser tabs: MuJoCo, Rerun and the GUI through noVNC (VIEWS)
#   make launch-drc-atlas-sim-vnc             # ... all three on the VNC desktop :99 instead (VIEWS=vnc)
#   make launch-drc-atlas-sim NETEM="delay 3ms 1ms loss 0.5%"    # the bus over a lossy, delayed link (tc netem)
#   make launch-drc-atlas-dummy-sim           # the MPC against the dummy simulator (the MPC model's rollout)
#   make deploy-robot ROBOT=drc_atlas HOST=<robot> NETWORK=<file> [SERVICE=enable]   # the robot's computer
#   make launch-drc-atlas-robot HOST=<robot>  # the deployed robot side, in the foreground over ssh
#   make launch-drc-atlas-mpc NETWORK=<file>  # the laptop side against it (launch-drc-atlas-mpc-vnc: VIEWS=vnc)
#   make launch-drc-atlas-sandbox             # the URDF in Rerun, with a slider per joint
#
# A launch target's variables:
#   VIEWS=browser|vnc|display                 where the three views go, by name (default browser); the URLs are printed
#                                             right before the laptop side starts (.devcontainer/README.md, "Ports"):
#                                             browser: browser tabs - MuJoCo's web viewer (http://localhost:6090), Rerun's
#                                               web viewer (http://localhost:9090) and the Tk operator GUI on the VNC
#                                               desktop :99 through noVNC (http://localhost:6080/vnc.html), started first;
#                                             vnc: all three on the VNC desktop :99 (MuJoCo glfw, Rerun's native viewer),
#                                               started first: one noVNC tab (the -vnc targets);
#                                             display: native windows on this shell's DISPLAY, which must answer (the
#                                               target fails with a hint otherwise)
#   RERUN_SINK=serve_web|spawn|connect|save   the Rerun bridge's sink (default: the view's - serve_web, the web viewer
#                                             at http://localhost:9090, or spawn, the native viewer; save: to RRD_PATH=<file>.rrd)
#   NETEM="<netem parameters>"                -sim: tc netem on the bus's packets on lo (default: none)
#   VIEWER=glfw|web|none                      -sim: the robot process's MuJoCo viewer by name (default: the view's when
#                                             VIEWS= is named, or a -vnc target's glfw; otherwise the mujoco_viewer of
#                                             PLOT_CONFIG, the shipped one's web: a page at http://localhost:6090, whose
#                                             address the target prints); glfw is the window on DISPLAY, and a list such
#                                             as web,none falls back in order
#   NETWORK=<network file>                    -mpc: the network file with the robot's address (required); a path
#                                             inside the checkout, relative to it or absolute: deploy-robot reads it on
#                                             the host, the launch targets in the dev container, which sees the checkout
#                                             only
#   JOINT_SOURCE=sliders|nominal              -sandbox: where the joint positions come from (default sliders)
#   DEV_CONTAINER=<name>                      the dev container, when make runs on the host (default devcontainer-app-1)
#   PLOT_CONFIG=<file>                        the plot configuration: the Rerun bridge's plots and, for -sim, the robot
#                                             process's MuJoCo viewer without VIEWER= (its mujoco_viewer); a file of
#                                             the checkout, relative to it or absolute inside it (default:
#                                             humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto)
#
# The sandbox: the model sandbox publishes the URDF's link poses at the nominal joint positions, or at those of its Tk
# window with a slider per joint, and the Rerun bridge draws them. It draws no TF frames, interactive markers or
# collision geometry.

DEV_CONTAINER ?= devcontainer-app-1
export WB_DEV_CONTAINER := $(DEV_CONTAINER)
# The Rerun bridge's sink; empty: the view's (views_rerun_sink_<view>).
RERUN_SINK ?=
NETEM ?=
# Where the views go, by name: the MuJoCo viewer and the Rerun sink of each view, which VIEWER= and RERUN_SINK= override.
# The VNC desktop's display and the URL banner are session.sh's (--views).
# LINT.IfChange(views)
VIEWS ?= browser
VIEWS_NAMES := browser vnc display
views_viewer_browser := web
views_viewer_vnc := glfw
views_viewer_display := glfw
views_rerun_sink_browser := serve_web
views_rerun_sink_vnc := spawn
views_rerun_sink_display := spawn
# LINT.ThenChange(//tools/deploy/session.sh:views, //tools/deploy/test_deploy_files.py:views, //.devcontainer/README.md:views, //README.md:views)
# The VNC desktop's resolution, for the views that start it (session.sh).
export WB_VNC_RESOLUTION := $(RESOLUTION)
# Whether VIEWS was named (on the command line, in the environment, or by a -vnc target): only then does its MuJoCo
# viewer override PLOT_CONFIG's mujoco_viewer. $(call views_checked,<view>) is the view, or an error listing them.
views_named = $(filter command line environment override,$(origin VIEWS))
views_checked = $(if $(filter $(VIEWS_NAMES),$(1)),$(1),$(error VIEWS=$(1) is not a view; the views are: $(VIEWS_NAMES)))
# The session's view and the bridge's sink, and for -sim the robot's MuJoCo viewer: VIEWER=, else the named view's.
view_sets = --views $(call views_checked,$(VIEWS)) --set rerun_sink=$(or $(RERUN_SINK),$(views_rerun_sink_$(VIEWS)))
sim_viewer = $(if $(or $(VIEWER),$(views_named)),--viewer "$(or $(VIEWER),$(views_viewer_$(VIEWS)))")
# LINT.IfChange(viewer_variable)
# The robot process's MuJoCo viewer, by name (robot_runtime/mujoco_sim_interface/README.md); empty: the mujoco_viewer of
# PLOT_CONFIG, web unless it names another. web serves http://localhost:6090 (session.sh prints it).
VIEWER ?=
# LINT.ThenChange(//robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/viewer/ViewerNames.h:viewer_names, //robot_runtime/web_viewer/include/web_viewer/WebViewerServerOptions.h:web_viewer_port)
NETWORK ?=
JOINT_SOURCE ?= sliders
RRD_PATH ?=
# LINT.IfChange(plot_config_variable)
PLOT_CONFIG ?= humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto
# LINT.ThenChange(//tools/deploy/session.sh:default_plot_config, //humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/cli.py:default_plot_config, //docker-compose.robot.sim.yaml:sim_plot_config_mount)
# The Rerun bridge's sink, and the recording of RERUN_SINK=save: relative to the checkout, where the bridge runs, or
# absolute inside it, which is made relative; an absolute path outside it is the dev container's own (/tmp/walk.rrd is
# in the dev container's filesystem when make runs on the host). PLOT_CONFIG is a file of the checkout (checkout_relative).
rerun_sets = $(view_sets) $(if $(RRD_PATH),--set rrd_path=$(call relative_inside_checkout,$(RRD_PATH))) \
	$(if $(PLOT_CONFIG),--set plot_config=$(call checkout_relative,PLOT_CONFIG,$(PLOT_CONFIG)))
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

# NETWORK= and PLOT_CONFIG= are paths relative to the checkout, where deploy_robot.sh (on the host) and the launcher (in
# the dev container) both resolve them, or absolute paths inside the checkout, which are made relative. One outside it
# is refused: the dev container sees nothing else of the host. $(call checkout_relative,<variable>,<path>).
checkout_root := $(abspath $(current_path))
# LINT.IfChange(checkout_relative)
relative_inside_checkout = $(if $(filter $(checkout_root)/%,$(1)),$(patsubst $(checkout_root)/%,%,$(1)),$(1))
checkout_relative = $(if $(filter /%,$(2)),$(if $(filter $(checkout_root)/%,$(2)),$(patsubst $(checkout_root)/%,%,$(2)),$(error $(1)=$(2) is outside the checkout $(checkout_root): the dev container sees only the checkout, so copy the file into it)),$(2))
# LINT.ThenChange(//tools/deploy/session.sh:checkout_relative_plot_config)
network_file = $(call checkout_relative,NETWORK,$(NETWORK))

# A shell command in the dev container's checkout: here inside it, through docker exec from the host.
dev_run := $(current_path)tools/deploy/in_dev_container.sh
session := $(current_path)tools/deploy/session.sh
deploy_robot := $(current_path)tools/deploy/deploy_robot.sh
# Raises a machine's net.core.rmem_max / wmem_max to the socket buffers its network file asks the DDS bus for, and keeps
# them in /etc/sysctl.d (sudo, only when a limit is below the request): $(call udp_buffers,<network file>,<host>).
# The launch and deployment targets run it first; it never fails them (tools/deploy/udp_buffers.sh).
shipped_network := config/ipc/network.textproto
udp_buffers = $(current_path)tools/deploy/udp_buffers.sh --network $(current_path)$(1) --host $(2)
deploy_flags = --host $(HOST) --deploy_dir $(DEPLOY_DIR) \
	$(if $(NETWORK),--network $(network_file)) $(if $(REGISTRY),--registry $(REGISTRY)) \
	$(if $(TAG),--tag $(TAG)) $(if $(IMAGE_SOURCE),--image_source $(IMAGE_SOURCE)) \
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
# MPC package's launch/ directory (robot.textproto, mpc.textproto, dummy_sim.textproto). A -vnc target is the target
# without the suffix with VIEWS=vnc (named, so its MuJoCo viewer is glfw): target-specific variables reach the
# prerequisite, whose recipe runs.
define robot_launch_targets
.PHONY: launch-$(call field,$(1),1)-sim launch-$(call field,$(1),1)-sim-vnc launch-$(call field,$(1),1)-dummy-sim \
	launch-$(call field,$(1),1)-dummy-sim-vnc launch-$(call field,$(1),1)-robot launch-$(call field,$(1),1)-mpc \
	launch-$(call field,$(1),1)-mpc-vnc

launch-$(call field,$(1),1)-sim: kill-sims
	@$(call udp_buffers,$(shipped_network),localhost)
	@$(session) sim --robot $(call field,$(1),2) --launch_file $(call field,$(1),3)/launch/mpc.textproto \
		--build "$(mpc_binaries_$(call field,$(1),4)) $(laptop_binaries)" $$(rerun_sets) \
		$$(if $$(NETEM),--netem "$$(NETEM)") $$(sim_viewer) \
		$$(if $$(PLOT_CONFIG),--plot_config "$$(PLOT_CONFIG)")

launch-$(call field,$(1),1)-dummy-sim: kill-sims
	@$(call udp_buffers,$(shipped_network),localhost)
	@$(session) laptop --launch_file $(call field,$(1),3)/launch/dummy_sim.textproto \
		--build "$(dummy_binaries_$(call field,$(1),4)) $(laptop_binaries)" $$(rerun_sets)

launch-$(call field,$(1),1)-robot:
	@$$(call udp_buffers,$$(or $$(network_file),$(shipped_network)),$$(HOST))
	@$(deploy_robot) up --robot $(call field,$(1),2) --host $$(HOST) --deploy_dir $$(DEPLOY_DIR)

launch-$(call field,$(1),1)-mpc: kill-sims
	@$$(if $$(NETWORK),,$$(error launch-$(call field,$(1),1)-mpc needs NETWORK=<network file> with the robot's address, e.g. a copy of config/ipc/two_machine.example.textproto))
	@$$(call udp_buffers,$$(network_file),localhost)
	@$(session) laptop --launch_file $(call field,$(1),3)/launch/mpc.textproto \
		--build "$(mpc_binaries_$(call field,$(1),4)) $(laptop_binaries)" $$(rerun_sets) \
		--set network_file=$$(network_file)

launch-$(call field,$(1),1)-sim-vnc launch-$(call field,$(1),1)-dummy-sim-vnc launch-$(call field,$(1),1)-mpc-vnc: override VIEWS = vnc
launch-$(call field,$(1),1)-sim-vnc: launch-$(call field,$(1),1)-sim
launch-$(call field,$(1),1)-dummy-sim-vnc: launch-$(call field,$(1),1)-dummy-sim
launch-$(call field,$(1),1)-mpc-vnc: launch-$(call field,$(1),1)-mpc
endef

# The sandbox of one robot description ($(1) is its entry of ROBOT_CONFIGURATIONS): launch/sandbox.textproto of it.
define robot_sandbox_targets
.PHONY: launch-$(call field,$(1),1)-sandbox launch-$(call field,$(1),1)-sandbox-vnc

launch-$(call field,$(1),1)-sandbox: kill-sims
	@$(session) laptop --launch_file $(call field,$(1),5)/launch/sandbox.textproto --build "$(sandbox_binaries)" \
		$$(rerun_sets) --set joint_source=$$(JOINT_SOURCE)

launch-$(call field,$(1),1)-sandbox-vnc: override VIEWS = vnc
launch-$(call field,$(1),1)-sandbox-vnc: launch-$(call field,$(1),1)-sandbox
endef

$(foreach configuration,$(ROBOT_CONFIGURATIONS),$(eval $(call robot_launch_targets,$(configuration))))
$(foreach configuration,$(filter $(addsuffix :%,$(SANDBOX_ROBOTS)),$(ROBOT_CONFIGURATIONS)),\
	$(eval $(call robot_sandbox_targets,$(configuration))))
# LINT.ThenChange(//.devcontainer/README.md:launch_targets, //README.md:launch_targets, //tools/deploy/test_deploy_files.py:launch_targets)

## Deploy the robot side to the robot's computer: make deploy-robot ROBOT=drc_atlas HOST=<ssh host> NETWORK=<file>
## [SERVICE=install|enable] [REGISTRY=<registry>] [PLATFORM=linux/arm64] [BACKEND=<backend name>] [MEMORY_LIMIT=4g]
## [CPUSET=2-5] [REALTIME_PRIORITY=80] [NETEM="delay 3ms"]. HOST=localhost deploys to this machine without ssh
## (tools/deploy/README.md). A deployment that runs already is restarted on the new image. IMAGE_SOURCE=registry
## REGISTRY=ghcr.io/<owner> TAG=<branch>-arm64 deploys CI's arm64 image instead of building one here.
deploy-robot:
	@$(if $(ROBOT),,$(error deploy-robot needs ROBOT=<robot configuration>: drc_atlas, engineai_sa01, unitree_g1, unitree_g1_wb or unitree_r1))
	@$(call udp_buffers,$(or $(network_file),$(shipped_network)),$(HOST))
	@$(deploy_robot) deploy --robot $(ROBOT) --service $(SERVICE) $(deploy_flags)

## Raise this machine's (or HOST='s, over ssh) kernel limits on UDP socket buffers to what the network file asks the DDS
## bus for, and keep them across reboots: make udp-buffers [NETWORK=<file>] [HOST=<ssh host>]. The launch and deploy
## targets do this first; sudo asks only while a limit is below the request (tools/deploy/udp_buffers.sh).
udp-buffers:
	@$(call udp_buffers,$(or $(network_file),$(shipped_network)),$(HOST))

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
	@$(dev_run) '$(bazel_indexed_in_repo) run //humanoid_nmpc/humanoid_rerun_viewer -- $(rerun_flags) --rerun_sink=spawn'

rerun-web:
	@echo "🌐 Rerun web viewer: http://localhost:9090/?url=rerun+http://localhost:9876/proxy"
	@$(dev_run) '$(bazel_indexed_in_repo) run //humanoid_nmpc/humanoid_rerun_viewer -- $(rerun_flags) --rerun_sink=serve_web'

## Inspect the bus (tools/ipc): make ipc-list, make ipc-echo TOPIC=mpc/status, make ipc-hz TOPIC=robot/mpc_observation
ipc_network = $(if $(NETWORK),--network_config=$(network_file))
ipc-list:
	@$(dev_run) '$(bazel_indexed_in_repo) run //tools/ipc:ipc_tool -- list $(ipc_network)'

ipc-echo:
	@$(if $(TOPIC),,$(error ipc-echo needs TOPIC=<topic>, e.g. TOPIC=mpc/status))
	@$(dev_run) '$(bazel_indexed_in_repo) run //tools/ipc:ipc_tool -- echo $(TOPIC) $(ipc_network)'

ipc-hz:
	@$(if $(TOPIC),,$(error ipc-hz needs TOPIC=<topic>, e.g. TOPIC=robot/mpc_observation))
	@$(dev_run) '$(bazel_indexed_in_repo) run //tools/ipc:ipc_tool -- hz $(TOPIC) $(ipc_network)'

## The launch, deployment, inspection and style-check targets
help:
	@echo "Launch targets, for <robot> in $(foreach c,$(ROBOT_CONFIGURATIONS),$(call field,$(c),1)):"
	@echo "  launch-<robot>-sim[-vnc]        robot side in the robot-sim container + MPC, GUI and Rerun bridge in the dev"
	@echo "                                  container, over the bus (NETEM=, VIEWER=glfw|web|none, RERUN_SINK=,"
	@echo "                                  PLOT_CONFIG=)"
	@echo "  launch-<robot>-dummy-sim[-vnc]  the MPC against the dummy simulator, with the GUI and the Rerun bridge"
	@echo "  launch-<robot>-robot HOST=      the deployed robot side, in the foreground (ssh to HOST, or localhost)"
	@echo "  launch-<robot>-mpc[-vnc] NETWORK=  the laptop side against the robot of the network file"
	@echo "  VIEWS=browser|vnc|display       where the views go (default browser): browser tabs for MuJoCo (:6090),"
	@echo "                                  Rerun (:9090) and the GUI through noVNC (:6080); vnc (the -vnc targets): all"
	@echo "                                  on the VNC desktop :99; display: native windows on DISPLAY, checked"
	@echo "  launch-<robot>-sandbox[-vnc]    for <robot> in $(SANDBOX_ROBOTS): the URDF in Rerun at its nominal pose or"
	@echo "                                  at a slider per joint (JOINT_SOURCE=sliders|nominal)"
	@echo "Deployment: deploy-robot ROBOT= HOST= NETWORK= [SERVICE=install|enable] [REGISTRY=] [PLATFORM=] [BACKEND=]"
	@echo "            [MEMORY_LIMIT=], robot-images"
	@echo "            udp-buffers [NETWORK=] [HOST=]: the kernel's UDP buffer limits the network file asks for (sudo,"
	@echo "            once per machine; the launch and deploy targets run it first)"
	@echo "Viewers and the bus: rerun-viewer, rerun-web [ROBOT=] [NETWORK=], ipc-list, ipc-echo TOPIC=, ipc-hz TOPIC="
	@echo "Validation (humanoid_nmpc/humanoid_mpc_validation): closed-loop-metrics ROBOT= LABEL= [SCENARIO=] [BASELINE=]"
	@echo "            [RECORD_STATES=1] [VALIDATION_CONTAINER_IMAGE=], benchmark-mpc-solve ROBOT= LABEL= [STATES=] [BASELINE=],"
	@echo "            compare-closed-loop CANDIDATE= BASELINE= [ROBOT=]"
	@echo "Clean-up: kill-sims, check-zombies"
	@echo "Style checks (AGENTS.md, \"Running the style checks\"; tools/hooks/README.md, tools/clang_tidy/README.md):"
	@echo "  format                          clang-format, black, isort and the enforced checks' safe rewrites"
	@echo "  lint                            every check without a compiler; one check or directory:"
	@echo "                                  python3 -m tools.hooks.lint_code --only <check> --paths <dir>"
	@echo "  lint-tidy [PKG=]                clang-tidy as a Bazel aspect, inside the machine lock"
	@echo "  lint-tidy-fix CHECKS= [PKG=] [PATHS=]        apply clang-tidy's fix-its, then build"
	@echo "  install-hooks                   the pre-commit hook: format, then lint the staged files"
