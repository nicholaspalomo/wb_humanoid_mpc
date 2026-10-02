# tools/deploy

The robot side of the distributed runtime
([`humanoid_nmpc/docs/distributed_runtime/README.md`](../../humanoid_nmpc/docs/distributed_runtime/README.md),
"Deployment") as something a robot computer runs: one Bazel bundle, a slim Docker image built from it, a compose file
with the realtime settings, and a systemd unit that starts it at boot. Simulation on the laptop runs the same image and
compose file, with the MuJoCo viewer added.

```text
bazel build //tools/deploy:robot_bundle   ->  robot_bundle.tar (binaries + runfiles, robot models, entry scripts)
docker build --target robot-runtime       ->  wb-humanoid-robot:latest       (the robot's computer)
docker build --target robot-sim           ->  wb-humanoid-robot-sim:latest   (robot-runtime + the viewer's GL)
docker compose -f docker-compose.robot.yaml up                             host network, SCHED_FIFO, mlockall
systemctl enable wb-humanoid-robot@<robot>                                 at boot
```

```bash
make deploy-robot ROBOT=drc_atlas HOST=robot.local NETWORK=my_network.textproto SERVICE=enable
make launch-drc-atlas-mpc NETWORK=my_network.textproto          # the laptop side against it
make launch-drc-atlas-sim                                       # both on this machine, the robot side in robot-sim
```

## The bundle

`//tools/deploy:robot_bundle` (`robot_bundle.bzl`, packed by `pack_bundle.py`) is one deterministic tar: the image has
no Bazel, so the binaries' runfiles are laid out as Bazel lays them out next to a built binary, and the repository's
files sit at their repository paths, where the launch files name them.

| Path | What |
|---|---|
| `bin/humanoid_nmpc/humanoid_centroidal_mpc_app/humanoid_centroidal_mpc_robot` (+ `.runfiles/`) | the centroidal robot binary; its runfiles hold `libmujoco` |
| `bin/humanoid_nmpc/humanoid_wb_mpc_app/humanoid_wb_mpc_robot` (+ `.runfiles/`) | the whole-body robot binary |
| `bin/robot_entrypoint.sh`, `bin/netem.sh`, `bin/collect_runtime_libraries.sh` | the image's entry point, NETEM, the library check |
| `launch/<robot>.sh` | the robot side of each robot configuration: its `launch/robot.textproto` exported with `//tools/launch:export_script` |
| `robot_models/...` | every robot's configuration and description (task, reference, PD gains, URDF, MJCF, meshes), without the GUI's `.bak` and `.live` side files |
| `config/ipc/network.textproto`, `config/ipc/two_machine.example.textproto` | the network files |

The image runs no Python, so it cannot run the launcher: `export_script` turns the launch file into a POSIX sh script
that runs the same command, with every variable of the launch file an environment variable `WB_ROBOT_<NAME>`
(`task_file` is `WB_ROBOT_TASK_FILE`). The launch file stays the one place the robot's command line is written.

## The images

<!-- LINT.IfChange(robot_images) -->
`docker/Dockerfile` builds them from the bundle, given as the build context `bundle` (a directory with
`robot_bundle.tar`), on the dev image's Ubuntu release (`BASE_IMAGE`), so the libraries are the ones the binaries were
linked against:

| Stage | What |
|---|---|
| `robot-pinocchio-amd64` | Pinocchio and coal from robotpkg (`docker/install_robotpkg.sh`), as in the dev image |
| `robot-pinocchio-arm64` | the same releases built from source (`docker/build_pinocchio_from_source.sh`), without Python |
| `robot-libraries` | the dev image's packages (`dependencies.txt`) and the bundle; `collect_runtime_libraries.sh collect` copies the libraries of `/opt/openrobots` the bundle links and lists the Ubuntu packages of the others |
| `robot-runtime` | Ubuntu, those packages, those libraries, `iproute2` (tc) and the bundle in `/opt/wb-humanoid-robot`; no compiler, Bazel, Python, VNC or Mesa. The build fails when a library is not found (`collect_runtime_libraries.sh check`). `WB_ROBOT_DEFAULT_HEADLESS=true` |
| `robot-sim` | `robot-runtime` plus Mesa's GL and the X libraries GLFW loads, for the MuJoCo viewer. `WB_ROBOT_DEFAULT_HEADLESS=false` |
<!-- LINT.ThenChange(//docker/Dockerfile:robot_images) -->

The binaries link GLFW, GLEW and libGL's dispatch library (the viewer is compiled in), so `robot-runtime` carries
those small client libraries; with no Mesa to create a context, the robot process runs with `--headless`.

`make robot-images` (`deploy_robot.sh image --target robot-sim`) builds both. The image carries the commit it was built
from (`org.opencontainers.image.revision`).

### aarch64

The robot computer's architecture is not decided. Everything is in place for an ARM one, but nothing has been built
for it yet:

- MuJoCo, which both robot binaries link (the `mujoco` backend), comes prebuilt: `lib/mujoco_vendor` downloads the
  release of the architecture it builds for, `linux-x86_64` or `linux-aarch64`, each with its published checksum.
- robotpkg publishes amd64 packages only, so on arm64 the dev image and the robot image build Pinocchio 4.1.0 and coal
  3.0.4 from source (`docker/build_pinocchio_from_source.sh`, the releases `install_robotpkg.sh` installs). The dev
  image then has no Pinocchio Python bindings (`tools/locomotion_heuristics` needs them), unless the script runs with
  `PINOCCHIO_PYTHON=ON`.
- The bundle must be built for the robot's architecture: in a dev container on an arm64 machine (natively, e.g. on the
  robot computer itself or an arm64 build machine, or under QEMU with `docker buildx`, which is very slow for this
  code base). `bazel/system_libs.bzl` finds yaml-cpp in the architecture's multiarch directory.
- Then `make deploy-robot ROBOT=... HOST=... DEV_CONTAINER=<the arm64 dev container>`. The image is built for the
  architecture the robot's Docker reports (`ssh <host> docker version`), or `PLATFORM=linux/arm64`; the script refuses a
  bundle whose robot binary is not of that architecture (its ELF header), and a `PLATFORM=` the robot does not run.

## The compose file

`docker-compose.robot.yaml` (repository root) runs the robot process of one robot configuration:

- `network_mode: host`: the bus binds and connects at the network file's addresses on the host's interfaces.
- `cap_add: [SYS_NICE, IPC_LOCK]` and `ulimits: {rtprio: 99, memlock: -1}`: `SCHED_FIFO` for the realtime thread and
  `mlockall` for the process, which `--realtime_priority` asks for. No privileged mode and no device; the one mount is
  the network file's (below).
- `cpuset: ${WB_ROBOT_CPUSET}`: the cores the container may use (all by default). Pair it with
  `WB_ROBOT_REALTIME_CORES` and `WB_ROBOT_BACKEND_CORES`; a core outside the cpuset is reported and skipped.
- `restart: unless-stopped`: a robot process that exits comes back, in ZERO_TORQUE.
- `init: true`: tini forwards `docker stop`'s SIGTERM to the robot process, which ends cleanly.
- The network file is at `/etc/wb-humanoid-robot/network.textproto` in the container (compose `configs`, from
  `WB_ROBOT_NETWORK_SOURCE`). Without swarm, docker compose implements a file config as a read-only bind mount of the
  file on the machine it runs on: the file must stay in the deployment directory, and the robot reads it when it
  starts (a new one takes effect at its next start, which a redeploy does).
- Logs: json-file, 3 x 10 MB.

<!-- LINT.IfChange(robot_memory_limit) -->
- `mem_limit` = `memswap_limit` = `WB_ROBOT_MEMORY_LIMIT` (default `4g`): a cap and no swap beyond it, the rule of
  AGENTS.md ("Builds share one machine's memory") for the robot. A leak ends inside the container instead of pushing
  the robot's computer into swap, which no realtime loop survives. The robot process locks its memory, so the cap
  bounds what it can lock too.
<!-- LINT.ThenChange(//docker-compose.robot.yaml:robot_memory_limit) -->

<!-- LINT.IfChange(robot_environment) -->
| Variable | Default | What |
|---|---|---|
| `ROBOT` | required | the robot configuration: `drc_atlas`, `engineai_sa01`, `unitree_g1`, `unitree_g1_wb`, `unitree_r1` (a script of `launch/`) |
| `WB_ROBOT_IMAGE` | `wb-humanoid-robot:latest` | the image |
| `WB_ROBOT_NETWORK_SOURCE` | `./config/ipc/network.textproto` | the network file mounted read-only into the container |
| `WB_ROBOT_CPUSET` | all | the container's cores, e.g. `2-5` |
| `WB_ROBOT_MEMORY_LIMIT` | `4g` | the memory cap, without swap |
| `WB_ROBOT_RESTART` | `unless-stopped` | the restart policy |
| `WB_ROBOT_BACKEND` | `mujoco` | the robot backend, by name |
| `WB_ROBOT_HEADLESS` | the image's: `true` in robot-runtime, `false` in robot-sim | the MuJoCo backend without its viewer |
| `WB_ROBOT_REALTIME_PRIORITY` | `80` | `SCHED_FIFO` priority of the realtime thread; 0 = off |
| `WB_ROBOT_REALTIME_CORES`, `WB_ROBOT_BACKEND_CORES` | `default` | cores of the realtime thread and of the backend's threads |
| `NETEM`, `NETEM_INTERFACE` | none, `lo` | tc-netem on the bus's packets (below) |

The `WB_ROBOT_*` variables of the robot process are the variables of `launch/robot.textproto`; empty means the launch
file's value.
<!-- LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/launch/robot.textproto:robot_variables, //docker-compose.robot.yaml:robot_environment) -->

Two overrides: `docker-compose.robot.sim.yaml` (the robot-sim image, `DISPLAY`, Mesa's software renderer, no
restart, and the checkout's `robot_models` mounted read-only, see "Simulation") and `docker-compose.robot.netem.yaml`
(`NET_ADMIN`, for NETEM only).

## Deploying

`make deploy-robot ROBOT=<robot> HOST=<ssh host> NETWORK=<network file>` (`deploy_robot.sh deploy`):

1. builds the bundle in the dev container (`in_dev_container.sh`: directly in a container, through `docker exec` into
   `DEV_CONTAINER` from the host) and copies it to `.deploy/context/`;
2. builds `robot-runtime` with the host's Docker, for the architecture the robot's Docker reports (`PLATFORM=`
   overrides it, and must match it);
3. ships it: `docker save | gzip | ssh <host> docker load`, or with `REGISTRY=<registry>` `docker push` and a
   `docker pull` on the host;
4. installs the deployment directory `~/wb-humanoid-robot` (`DEPLOY_DIR=`) on the host: `docker-compose.robot.yaml`, the
   NETEM override, `network.textproto` (`NETWORK=`), a `.env` with the settings (`deploy_robot.sh environment` prints
   it) and the unit `wb-humanoid-robot@.service`;
5. `SERVICE=install` installs the unit (sudo), `SERVICE=enable` also enables it at boot and (re)starts it;
6. restarts the robot when it runs already, so that a redeploy takes effect at once: `SERVICE=enable` restarts the
   unit (`enable --now` would leave an active oneshot unit, and the container it started, on the old image), and
   otherwise a running container is recreated with `docker compose up --detach` on the new image and `.env`.

One robot per machine: the deployment directory and its compose project hold the robot configuration deployed last.
`SERVICE=enable` disables and stops the unit of any other robot on the host, and an instance of the unit whose robot is
not the deployed one refuses to start (its `ExecStartPre` names the deployed one) rather than run the container under
its own name. `ROBOT=` must be one of the robot configurations (`deploy_robot.sh` refuses a misspelled one before it
builds anything, instead of installing a unit that restarts a failing container for ever).

`HOST=localhost` deploys to this machine without ssh, with the shipped localhost network unless `NETWORK=` is given; a
remote host needs `NETWORK=` (`config/ipc/two_machine.example.textproto`). Every option: `deploy_robot.sh --help`;
`--dry_run` prints the commands.

| Make variable | `deploy_robot.sh` | Default |
|---|---|---|
| `ROBOT` | `--robot` | required |
| `HOST` | `--host` | `localhost` |
| `NETWORK` | `--network` | the shipped localhost network, on localhost only. A path inside the checkout, relative to it or absolute: the launch targets read it in the dev container, which sees only the checkout |
| `SERVICE` | `--service` | `none` (`install`, `enable`) |
| `REGISTRY` | `--registry` | none: `docker save` over ssh |
| `PLATFORM` | `--platform` | the robot's Docker's architecture (asked over ssh), this machine's for localhost |
| `BACKEND`, `MEMORY_LIMIT` | `--backend`, `--memory_limit` | the launch file's (`mujoco`), the compose file's (`4g`) |
| `CPUSET`, `REALTIME_PRIORITY`, `REALTIME_CORES`, `BACKEND_CORES` | `--cpuset`, ... | the compose file's, the launch file's |
| `NETEM`, `NETEM_INTERFACE` | `--netem`, `--netem_interface` | none, `lo` |
| `DEPLOY_DIR` | `--deploy_dir` | `wb-humanoid-robot` in the host's home |
| `DEV_CONTAINER` | `--dev_container` | `devcontainer-app-1` |

`DEV_CONTAINER` must be a dev container of this checkout's image (`docker/Dockerfile`: Pinocchio in `/opt/openrobots`,
no ROS) on the host's network. A dev container created before the switch from ROS has neither; recreate it from the new
`docker-compose.yaml` (Dev Containers: Rebuild Container, or `docker compose up -d --build --force-recreate`) before
using these targets from the host. `in_dev_container.sh` checks for both and says so.

The robot computer needs Docker with the compose plugin (v2) and, for `SERVICE=`, systemd and sudo. Nothing else: no
checkout, no Bazel.

## Running it

- At boot: `SERVICE=enable`, then `sudo systemctl status|stop|start wb-humanoid-robot@<robot>`. The unit runs
  `docker compose up --detach` in the deployment directory once Docker and the network are up; Docker's restart policy
  brings the process back if it exits.
- By hand: `make launch-<robot>-robot HOST=<host>` runs `docker compose up` in the deployment directory over ssh (or here
  for `localhost`), in the foreground with its log; Ctrl-C stops it. On the robot itself: `cd ~/wb-humanoid-robot &&
  docker compose up`.
- Logs: `docker compose --project-directory ~/wb-humanoid-robot logs -f robot`. The entry point's first line says what
  the realtime loop may take there (`realtime priority limit 99, locked memory limit unlimited, cpus 0-19`), and the
  robot process logs every realtime setting it was refused.

The robot process starts in ZERO_TORQUE whatever the MPC does, and holds JOINT_PD while the MPC link is down; start
order does not matter (humanoid_nmpc/docs/distributed_runtime/README.md, "Deployment").

## NETEM

`NETEM="delay 3ms 1ms loss 0.5%"` (any `tc-netem` parameters) on `make launch-<robot>-sim` or `make deploy-robot` adds
`docker-compose.robot.netem.yaml` (`NET_ADMIN`) and makes the entry point run `netem.sh apply <NETEM_INTERFACE>` before
the robot process and `netem.sh clear` after it. `netem.sh` replaces the interface's root qdisc with a prio qdisc whose
fourth band is netem and filters only TCP packets (IPv4 and IPv6, matched on the IP protocol and the ports) from or to
the bus's ports (5600-5631) into it; everything else is untouched. With host networking the interface is the host's: on `lo` (the default, for simulation) every bus
connection on the machine is shaped in both directions; on a network interface only what the robot sends. A container
killed before it could clear it leaves the qdisc; `make kill-sims` removes it, or `tc qdisc del dev lo root`. Off by
default.

## Realtime in a container

- `SCHED_FIFO` needs `CAP_SYS_NICE` or an `rtprio` limit (both given), and a kernel without `CONFIG_RT_GROUP_SCHED`
  (Ubuntu's generic and lowlatency kernels have none) or with a realtime budget for Docker's cgroup: with
  `CONFIG_RT_GROUP_SCHED=y` (some PREEMPT_RT builds) and cgroup v2, a container's realtime threads get no runtime, so
  `sched_setscheduler` fails; boot with it off or give `/sys/fs/cgroup/.../cpu.rt_runtime_us` a budget.
- `mlockall` needs `CAP_IPC_LOCK` or `memlock: -1` (both given); the memory cap bounds it.
- A PREEMPT_RT kernel (Ubuntu Pro's `linux-realtime`, or a mainline kernel since 6.12 with `PREEMPT_RT=y`) is
  recommended on the robot; `isolcpus`/`nohz_full` for the realtime core and `WB_ROBOT_CPUSET` keep other load off it.
- `sched_rt_runtime_us` (default 950000 of 1000000) leaves the rest of the machine 5% even against a runaway realtime
  thread.

## Simulation

`make launch-<robot>-sim[-vnc]` (`session.sh sim`) runs the hardware topology on one machine: it builds the bundle,
`robot-runtime` and `robot-sim`, starts `docker-compose.robot.yaml` + `docker-compose.robot.sim.yaml` with the robot's
realtime settings and the shipped localhost network, follows its log as `[robot]`, and runs the laptop side
(`launch/mpc.textproto`: the MPC node, the GUI, the Rerun bridge) in the dev container.

<!-- LINT.IfChange(sim_project) -->
The compose project is this checkout's own, `wb-humanoid-robot-sim-<checkout directory>-<checksum of its path>`
(`session.sh project` prints it): apart from a deployment on this machine (`wb-humanoid-robot`) and from the simulation
of another checkout, so that `make kill-sims` stops only what this checkout started. The bus's ports are the host's, so
a session refuses to start while another robot side holds them - a robot deployed on this machine, another checkout's
simulation - and says how to stop it; `make launch-<robot>-dummy-sim` checks the same.
<!-- LINT.ThenChange(//tools/deploy/session.sh:sim_project) -->

One difference from the robot, for tuning: the checkout's `robot_models` is mounted read-only over the image's copy
(`docker-compose.robot.sim.yaml`). The robot process then reads the files the MPC node reads and the GUI's "Save to
YAML" and an editor write, and its watchers reload `joint_pd_gains.yaml` and the task file's controller-side keys
(`contactEstimator`, `contact_wrench_gate`) while it runs, as the ROS sims did. A deployed robot reads the copy in its
image; there, what the GUI publishes on the bus (`operator/pd_gains`, `operator/mpc_parameters`) reaches it, and a file
edited on the laptop takes effect with the next deploy. The MuJoCo viewer draws on `DISPLAY` through the X server's abstract socket on
the host's network: the VNC desktop's `:99` (`-vnc`), or the host's display (`xhost +SI:localuser:root` lets the
container's root draw there). `HEADLESS=true` runs it without the viewer. When the laptop side ends, the robot
container stops; `make kill-sims` (`session.sh stop`) ends what a session left.

The robot side needs the host's Docker: run the target on the host (Linux, host networking), or in a dev container
that can reach Docker (`session.sh` then mounts the checkout by its host path, from the dev container's mounts, or
`WB_HOST_CHECKOUT`). Docker Desktop's VM network (macOS) does not share the host's loopback between containers, so the
`-sim` targets do not run there; use `make launch-<robot>-dummy-sim[-vnc]` on macOS.

## Files

| File | What |
|---|---|
| `robot_bundle.bzl`, `pack_bundle.py` | the bundle rule and its packer |
| `BUILD.bazel` | the robot configurations, their exported scripts and the bundle |
| `robot_entrypoint.sh` | the images' entry point: `ROBOT`'s script, the image's viewer default, NETEM around the run |
| `netem.sh` | tc-netem on the bus's ports |
| `collect_runtime_libraries.sh` | the bundle's libraries: collected in the builder stage, checked in the image |
| `wb-humanoid-robot@.service` | the systemd unit template |
| `deploy_robot.sh` | bundle, image, deploy, up, down, environment |
| `session.sh` | the Makefile's launch sessions: `sim`, `laptop`, `stop` |
| `in_dev_container.sh` | a command in the dev container's checkout, from the host or inside it |

## Tests

`bazel test //tools/deploy/...`:

- `:test_pack_bundle`: sorted, root-owned, dated-0 entries, symlinks resolved, modes, the same bytes for the same inputs,
  malformed manifests.
- `:test_robot_bundle`: the bundle's layout and entries, no GUI side files, every library of the robot binaries found
  from inside it (`libmujoco` from its own runfiles), every robot's script starting its binary with files of the bundle,
  and the environment choosing the network file, the viewer and the priority.
- `:test_deploy_files`: `netem.sh` and the entry point against a recording `tc` (only the bus's ports, cleared after the
  run, SIGTERM forwarded, the exit status kept, unknown robots refused, the image's viewer default); the compose files
  (realtime settings, no privileged mode, memory cap without swap, the environment equal to the launch files'
  variables, NET_ADMIN only with NETEM); the unit; the Dockerfile's robot stages (nothing to build with in the runtime
  image, only GL in the sim image); the deployment script's commands (`--dry_run`) and `.env`; the Makefile's robot
  table against the dev container README.

`//robot_models/tests:test_launch_files` checks the launch files themselves, `//tools/launch:test_launch_script` the
export.
