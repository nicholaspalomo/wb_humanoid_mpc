# Copyright (c) 2026, Nicholas Palomo. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""The files around the robot images (tools/deploy/README.md), which the tests pin by property rather than by version.

The files are the entry point and netem.sh, run against a recording `tc`; the compose files; the systemd unit; the
Dockerfile's robot stages; the deployment script; and the Makefile's launch targets. The properties are the realtime
settings, what the images install, the environment the launch files read and the hardware topology of every
simulation.
"""

import configparser
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import unittest

import yaml

RUNFILES = os.path.join(os.environ["TEST_SRCDIR"], "_main")


def runfile(path: str) -> str:
    """The path of the runfile `path`, relative to the repository root."""
    return os.path.join(RUNFILES, path)


def read(path: str) -> str:
    """The text of the runfile `path`."""
    with open(runfile(path), "r", encoding="utf-8") as stream:
        return stream.read()


def search(pattern: str, text: str, flags: int = 0) -> re.Match[str]:
    """The first match of `pattern` in `text`; fails the test when there is none."""
    match = re.search(pattern, text, flags)
    assert match is not None, f"no match of {pattern!r}"
    return match


def without_comments(text: str) -> str:
    """`text` without its `#` comment lines."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


# The environment variables the robot images take from docker-compose.robot.yaml: launch-file variables of every robot.
# LINT.IfChange(robot_environment)
ROBOT_ENVIRONMENT = (
    "WB_ROBOT_NETWORK_FILE",
    "WB_ROBOT_BACKEND",
    "WB_ROBOT_VIEWER",
    "WB_ROBOT_PLOT_CONFIG",
    "WB_ROBOT_WEB_VIEWER_ADDRESS",
    "WB_ROBOT_WEB_VIEWER_PORT",
    "WB_ROBOT_WEB_VIEWER_ALLOWED_HOSTS",
    "WB_ROBOT_VIEWER_CORES",
    "WB_ROBOT_REALTIME_PRIORITY",
    "WB_ROBOT_REALTIME_CORES",
    "WB_ROBOT_BACKEND_CORES",
    "WB_ROBOT_CONFIG_STORE_DIR",
    "WB_ROBOT_CONFIG_SEED",
)
# LINT.ThenChange(//docker-compose.robot.yaml:robot_environment)

# Where the robot images keep the robot's stored configuration (docker-compose.robot.yaml), one directory per robot.
CONFIG_STORE_TARGET = "/var/lib/wb-humanoid-robot/config"
# Where the simulated robot reads its plot configuration (docker-compose.robot.sim.yaml).
SIM_PLOT_CONFIG_TARGET = "/etc/wb-humanoid-robot/plot_config.textproto"
# The shipped plot configuration, and the header that makes a file one the robot reads its viewer from.
SHIPPED_PLOT_CONFIG = (
    "humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto"
)
PLOT_CONFIG_HEADER = (
    "# proto-file:"
    " humanoid_nmpc/humanoid_rerun_viewer/proto/plot_config.proto\n#"
    " proto-message: humanoid_rerun_viewer_proto.PlotConfig\n"
)
# Stand-ins for what a sim session runs, which record their command lines to SESSION_LOG: Docker, with the network file,
# the viewer and the plot configuration it would give the robot container, has nothing running, answers `compose logs`
# with FAKE_ROBOT_LOG, and knows the mounts of one container, FAKE_CONTAINER_ID's; the dev container's runner and the
# image build do nothing.
# LINT.IfChange(sim_network)
FAKE_DOCKER = """#!/bin/sh
echo "docker $* | network=${WB_ROBOT_NETWORK_SOURCE-} viewer=${WB_ROBOT_VIEWER-} plot_config=${WB_ROBOT_SIM_PLOT_CONFIG-}" >> "$SESSION_LOG"
if [ "$1" = "inspect" ]; then
    for last; do :; done
    [ -n "${FAKE_CONTAINER_ID-}" ] && [ "$last" = "$FAKE_CONTAINER_ID" ] || exit 1
    echo "$FAKE_MOUNT_SOURCE $FAKE_MOUNT_DESTINATION"
fi
case " $* " in
    *" logs "*) [ -n "${FAKE_ROBOT_LOG-}" ] && printf '%s\n' "$FAKE_ROBOT_LOG" ;;
esac
exit 0
"""
# LINT.ThenChange(//tools/deploy/session.sh:sim_network)
FAKE_RECORDER = """#!/bin/sh
echo "$(basename "$0") $*" >> "$SESSION_LOG"
case "$*" in
    *"xset q"*) [ -z "${FAKE_NO_X_SERVER-}" ] || exit 1 ;;
esac
exit 0
"""
# The host's path of the fake checkout, as WB_HOST_CHECKOUT gives it.
HOST_CHECKOUT = "/host/checkout"

# A `tc` that records its arguments, one call per line, and answers `qdisc show` with the file TC_SHOW names.
FAKE_TC = """#!/bin/sh
echo "$*" >> "$TC_LOG"
if [ "$1" = "qdisc" ] && [ "$2" = "show" ]; then cat "$TC_SHOW" 2>/dev/null; fi
exit 0
"""
OUR_QDISC = (
    "qdisc prio 5600: root refcnt 2 bands 4 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1"
    " 1 1\n"
)


class FakeTc:
    """A directory with the recording `tc`, to put in front of PATH."""

    def __init__(self, directory: str) -> None:
        self.bin = os.path.join(directory, "fake_bin")
        os.makedirs(self.bin)
        with open(
            os.path.join(self.bin, "tc"), "w", encoding="utf-8"
        ) as stream:
            stream.write(FAKE_TC)
        os.chmod(os.path.join(self.bin, "tc"), 0o755)
        self.log = os.path.join(directory, "tc.log")
        self.show = os.path.join(directory, "tc_show")

    def environment(self, installed: bool) -> dict[str, str]:
        with open(self.show, "w", encoding="utf-8") as stream:
            stream.write(
                OUR_QDISC if installed else "qdisc noqueue 0: root refcnt 2\n"
            )
        return {
            "PATH": self.bin
            + os.pathsep
            + os.environ.get("PATH", "/usr/bin:/bin"),
            "TC_LOG": self.log,
            "TC_SHOW": self.show,
        }

    def calls(self) -> list[str]:
        if not os.path.exists(self.log):
            return []
        with open(self.log, encoding="utf-8") as stream:
            return stream.read().splitlines()


class NetemTest(unittest.TestCase):
    def setUp(self) -> None:
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self._directory = tempfile.TemporaryDirectory()
        self.tc = FakeTc(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def netem(
        self, *arguments: str, installed: bool = False
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sh", runfile("tools/deploy/netem.sh")] + list(arguments),
            capture_output=True,
            text=True,
            env=self.tc.environment(installed),
            check=False,  # The tests check the return code.
        )

    def test_apply_shapes_only_the_bus_ports(self) -> None:
        result = self.netem(
            "apply", "lo", "delay", "3ms", "1ms", "loss", "0.5%"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [call for call in self.tc.calls() if "show" not in call],
            [
                (
                    "qdisc replace dev lo root handle 5600: prio bands 4"
                    " priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1"
                ),
                (
                    "qdisc add dev lo parent 5600:4 handle 5601: netem delay"
                    " 3ms 1ms loss 0.5%"
                ),
                (
                    "filter add dev lo parent 5600: protocol ip prio 1 u32"
                    " match ip protocol 17 0xff match ip sport 7168 0xfe00"
                    " flowid 5600:4"
                ),
                (
                    "filter add dev lo parent 5600: protocol ipv6 prio 2 u32"
                    " match ip6 protocol 17 0xff match ip6 sport 7168 0xfe00"
                    " flowid 5600:4"
                ),
                (
                    "filter add dev lo parent 5600: protocol ip prio 1 u32"
                    " match ip protocol 17 0xff match ip dport 7168 0xfe00"
                    " flowid 5600:4"
                ),
                (
                    "filter add dev lo parent 5600: protocol ipv6 prio 2 u32"
                    " match ip6 protocol 17 0xff match ip6 dport 7168 0xfe00"
                    " flowid 5600:4"
                ),
            ],
        )

    def test_every_filter_matches_udp_only(self) -> None:
        self.netem("apply", "lo", "delay", "3ms")
        filters = [
            call for call in self.tc.calls() if call.startswith("filter add")
        ]
        self.assertEqual(len(filters), 4, "both directions, IPv4 and IPv6")
        for call in filters:
            with self.subTest(call=call):
                self.assertRegex(
                    call, r"\bmatch ip6? protocol 17 0xff match ip6? [sd]port "
                )

    def test_apply_replaces_a_qdisc_a_previous_run_left(self) -> None:
        self.netem("apply", "lo", "delay", "3ms", installed=True)
        self.assertIn("qdisc del dev lo root", self.tc.calls())

    def test_clear_removes_its_own_qdisc_and_nothing_else(self) -> None:
        self.assertEqual(
            self.netem("clear", "eth0", installed=True).returncode, 0
        )
        self.assertIn("qdisc del dev eth0 root", self.tc.calls())
        os.remove(self.tc.log)
        self.assertEqual(
            self.netem("clear", "eth0", installed=False).returncode, 0
        )
        self.assertEqual(
            [call for call in self.tc.calls() if "del" in call], []
        )

    def test_a_usage_error_changes_nothing(self) -> None:
        for arguments in (
            ("apply", "lo"),
            ("clear",),
            ("shape", "lo"),
            ("clear", "lo", "extra"),
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.netem(*arguments).returncode, 2)
        self.assertEqual(
            [call for call in self.tc.calls() if "show" not in call], []
        )

    def test_the_port_range_holds_the_dds_ports_of_the_network_files(
        self,
    ) -> None:
        # Every process's discovery and data port of the file's domain: 7400 + 250 * domain_id + 10 + 2 * index and
        # the port after it, for the 20 indices of one machine (robot_ipc/DomainConfig.h unicastPortRange()).
        script = read("tools/deploy/netem.sh")
        base = int(search(r"^BUS_PORT=(\d+)$", script, re.M).group(1))
        mask = int(
            search(r"^BUS_PORT_MASK=(0x[0-9a-f]+)$", script, re.M).group(1), 16
        )
        for network in (
            "config/ipc/network.textproto",
            "config/ipc/two_machine.example.textproto",
        ):
            domain = search(r"^domain_id:\s*(\d+)$", read(network), re.M)
            first = 7400 + 250 * int(domain.group(1)) + 10
            ports = range(first, first + 2 * 20)
            with self.subTest(network=network):
                self.assertEqual(
                    [port for port in ports if port & mask != base], []
                )


class EntrypointTest(unittest.TestCase):
    """robot_entrypoint.sh in a bundle of its own: two robot scripts that report what they were started with."""

    def setUp(self) -> None:
        # pylint: disable-next=consider-using-with  # tearDown() deletes it.
        self._directory = tempfile.TemporaryDirectory()
        directory = self._directory.name
        self.bundle = os.path.join(directory, "bundle")
        os.makedirs(os.path.join(self.bundle, "bin"))
        os.makedirs(os.path.join(self.bundle, "launch"))
        for script in ("robot_entrypoint.sh", "netem.sh"):
            shutil.copy(
                runfile(f"tools/deploy/{script}"),
                os.path.join(self.bundle, "bin", script),
            )
        self.write_robot(
            "fake",
            'echo "pid=$$"\necho "viewer=${WB_ROBOT_VIEWER:-}"\nexit'
            ' "${FAKE_STATUS:-0}"\n',
        )
        self.write_robot(
            "patient",
            "trap 'echo got-term; exit 0' TERM\necho ready\nwhile :; do sleep"
            " 0.05; done\n",
        )
        self.tc = FakeTc(directory)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def write_robot(self, name: str, body: str) -> None:
        path = os.path.join(self.bundle, "launch", f"{name}.sh")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(f"#!/bin/sh\n{body}")
        os.chmod(path, 0o755)

    def entrypoint(self) -> list[str]:
        return ["sh", os.path.join(self.bundle, "bin", "robot_entrypoint.sh")]

    def run_entrypoint(
        self, environment: dict[str, str]
    ) -> subprocess.CompletedProcess:
        env = self.tc.environment(installed=False)
        env.update(environment)
        return subprocess.run(
            self.entrypoint(),
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,  # The tests check the return code.
        )

    def test_robot_must_name_a_robot_script(self) -> None:
        for robot in (None, "nope", "../fake", "fa ke"):
            with self.subTest(robot=robot):
                result = self.run_entrypoint(
                    {} if robot is None else {"ROBOT": robot}
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("fake patient", result.stderr)

    def test_the_robot_script_replaces_the_entry_point(self) -> None:
        env = self.tc.environment(installed=False)
        env["ROBOT"] = "fake"
        with subprocess.Popen(
            self.entrypoint(), stdout=subprocess.PIPE, text=True, env=env
        ) as process:
            output, _ = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0)
        self.assertIn(f"pid={process.pid}", output)
        self.assertIn("realtime priority limit", output)
        self.assertEqual(self.tc.calls(), [], "no NETEM, no tc")

    def test_the_image_decides_the_viewer_unless_the_environment_does(
        self,
    ) -> None:
        cases: tuple[tuple[dict[str, str], str], ...] = (
            ({}, "viewer="),
            ({"WB_ROBOT_DEFAULT_VIEWER": "none"}, "viewer=none"),
            (
                {"WB_ROBOT_DEFAULT_VIEWER": "none", "WB_ROBOT_VIEWER": ""},
                "viewer=none",
            ),
            ({"WB_ROBOT_DEFAULT_VIEWER": "glfw"}, "viewer=glfw"),
            (
                {
                    "WB_ROBOT_DEFAULT_VIEWER": "none",
                    "WB_ROBOT_VIEWER": "web,none",
                },
                "viewer=web,none",
            ),
            # A plot configuration the environment names chooses instead of the image (robot-sim's mount).
            (
                {
                    "WB_ROBOT_DEFAULT_VIEWER": "none",
                    "WB_ROBOT_PLOT_CONFIG": (
                        "/etc/wb-humanoid-robot/plot_config.textproto"
                    ),
                },
                "viewer=",
            ),
            # robot-sim names no viewer of its own: the launch file's plot configuration chooses.
            ({"WB_ROBOT_DEFAULT_VIEWER": ""}, "viewer="),
        )
        for environment, expected in cases:
            with self.subTest(environment=environment):
                result = self.run_entrypoint(dict(environment, ROBOT="fake"))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"{expected}\n", result.stdout)

    def test_netem_is_applied_for_the_run_and_cleared_after(self) -> None:
        result = self.run_entrypoint(
            {
                "ROBOT": "fake",
                "NETEM": "delay 3ms 1ms loss 0.5%",
                "FAKE_STATUS": "3",
            }
        )
        self.assertEqual(
            result.returncode, 3, "the robot process's exit status"
        )
        calls = self.tc.calls()
        self.assertIn(
            "qdisc add dev lo parent 5600:4 handle 5601: netem delay 3ms 1ms"
            " loss 0.5%",
            calls,
        )
        self.assertEqual(calls[-1], "qdisc show dev lo", "cleared last")

    def test_sigterm_reaches_the_robot_process_and_netem_is_cleared(
        self,
    ) -> None:
        env = self.tc.environment(installed=False)
        env.update(
            {
                "ROBOT": "patient",
                "NETEM": "delay 3ms",
                "NETEM_INTERFACE": "eth9",
            }
        )
        # pylint: disable-next=consider-using-with  # The cleanup kills and reaps it, should the test fail first.
        process = subprocess.Popen(
            self.entrypoint(), stdout=subprocess.PIPE, text=True, env=env
        )
        self.addCleanup(kill_and_reap, process)
        stdout = process.stdout
        assert stdout is not None
        deadline = time.monotonic() + 10.0
        line = ""
        while "ready" not in line and time.monotonic() < deadline:
            line = stdout.readline()
        self.assertIn("ready", line)
        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=30)
        self.assertEqual(process.returncode, 0)
        self.assertIn("got-term", output)
        self.assertEqual(self.tc.calls()[-1], "qdisc show dev eth9")


def kill_and_reap(process: subprocess.Popen[str]) -> None:
    """Kills `process` unless it has exited, waits for it and closes its output pipe."""
    if process.poll() is None:
        process.kill()
    process.wait()
    if process.stdout is not None:
        process.stdout.close()


def load_yaml(path: str) -> dict:
    """The YAML document of the runfile `path`."""
    return yaml.safe_load(read(path))


class ComposeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = load_yaml("docker-compose.robot.yaml")["services"]["robot"]

    def test_the_robot_container_gets_the_realtime_settings_and_nothing_more(
        self,
    ) -> None:
        self.assertEqual(self.robot["network_mode"], "host")
        self.assertIs(self.robot["init"], True)
        self.assertEqual(
            sorted(self.robot["cap_add"]), ["IPC_LOCK", "SYS_NICE"]
        )
        self.assertEqual(self.robot["ulimits"], {"rtprio": 99, "memlock": -1})
        for key in ("privileged", "ports", "devices", "pid", "ipc"):
            self.assertNotIn(key, self.robot)
        # The one directory of the host the robot writes: its stored configuration.
        self.assertEqual(
            [volume["target"] for volume in self.robot["volumes"]],
            [CONFIG_STORE_TARGET],
        )
        self.assertIn("unless-stopped", self.robot["restart"])
        self.assertIn("WB_ROBOT_CPUSET", self.robot["cpuset"])

    def test_its_memory_is_capped_without_swap(self) -> None:
        self.assertEqual(self.robot["mem_limit"], self.robot["memswap_limit"])
        self.assertRegex(
            self.robot["mem_limit"], r"\$\{WB_ROBOT_MEMORY_LIMIT:-\d+[gm]\}"
        )

    def test_the_network_file_is_mounted_where_the_robot_reads_it(self) -> None:
        (config,) = self.robot["configs"]
        self.assertEqual(
            config["target"], self.robot["environment"]["WB_ROBOT_NETWORK_FILE"]
        )
        self.assertIn(
            "./config/ipc/network.textproto",
            load_yaml("docker-compose.robot.yaml")["configs"]["network"][
                "file"
            ],
        )

    def test_the_stored_configuration_is_a_host_directory_that_outlives_the_container(
        self,
    ) -> None:
        # LINT.IfChange(robot_config_store)
        (store,) = self.robot["volumes"]
        self.assertEqual(store["type"], "bind")
        self.assertEqual(store["target"], CONFIG_STORE_TARGET)
        self.assertNotIn("read_only", store)
        self.assertEqual(
            store["source"], "${WB_ROBOT_CONFIG_SOURCE:-./robot_config}"
        )
        # A bind of the long syntax creates a missing host directory only when it says so.
        self.assertIs(store["bind"]["create_host_path"], True)
        # One directory per robot configuration below it.
        self.assertEqual(
            self.robot["environment"]["WB_ROBOT_CONFIG_STORE_DIR"],
            f"{CONFIG_STORE_TARGET}/${{ROBOT}}",
        )
        self.assertEqual(
            self.robot["environment"]["WB_ROBOT_CONFIG_SEED"],
            "${WB_ROBOT_CONFIG_SEED:-}",
        )
        # The deployment creates it, as the deploy user, next to the compose file.
        script = read("tools/deploy/deploy_robot.sh")
        self.assertIn('ROBOT_CONFIG_DIR="robot_config"', script)
        self.assertIn(
            "mkdir -p '${remote_dir}' '${remote_dir}/${ROBOT_CONFIG_DIR}'",
            script,
        )
        # LINT.ThenChange(//docker-compose.robot.yaml:robot_config_store)

    def test_the_environment_is_the_launch_files_variables(self) -> None:
        launch = read(
            "robot_models/drc_atlas/drc_atlas_centroidal_mpc/launch/robot.textproto"
        )
        variables = {
            f"WB_ROBOT_{name.upper()}"
            for name in re.findall(r'variables \{ name: "(\w+)"', launch)
        }
        passed = {
            name
            for name in self.robot["environment"]
            if name.startswith("WB_ROBOT_")
        }
        self.assertEqual(passed, set(ROBOT_ENVIRONMENT))
        self.assertEqual(passed - variables, set())

    def test_the_sim_override_adds_the_viewer_and_the_checkouts_models_only(
        self,
    ) -> None:
        document = load_yaml("docker-compose.robot.sim.yaml")
        sim = document["services"]["robot"]
        self.assertEqual(
            set(sim), {"image", "restart", "environment", "volumes"}
        )
        self.assertIn("wb-humanoid-robot-sim", sim["image"])
        self.assertEqual(sim["restart"], "no")
        self.assertIn("DISPLAY", sim["environment"])
        mounts = {volume["target"]: volume for volume in sim["volumes"]}
        self.assertEqual(
            set(mounts),
            {
                "/opt/wb-humanoid-robot/robot_models",
                SIM_PLOT_CONFIG_TARGET,
                CONFIG_STORE_TARGET,
            },
        )
        # LINT.IfChange(sim_config_store)
        # The stored configuration is a volume of the compose project, at the base file's target, which merging the
        # files replaces by it: nothing of the host is written. The checkout's files seed it at every start.
        store = mounts[CONFIG_STORE_TARGET]
        self.assertEqual(store["type"], "volume")
        self.assertEqual(store["source"], "robot_config")
        self.assertIn("robot_config", document["volumes"])
        self.assertEqual(
            sim["environment"]["WB_ROBOT_CONFIG_SEED"],
            "${WB_ROBOT_CONFIG_SEED:-every_start}",
        )
        # LINT.ThenChange(//docker-compose.robot.sim.yaml:robot_config_store)
        # LINT.IfChange(sim_models_mount)
        # The checkout's robot models over the image's copy, read-only: the bundled files the robot seeds its stored
        # configuration from at every start are the ones the MPC node reads. An editor's save reaches a running robot at
        # its next start or through push_robot_config, the GUI's Save over the bus.
        mount = mounts["/opt/wb-humanoid-robot/robot_models"]
        self.assertEqual(mount["type"], "bind")
        self.assertEqual(mount["target"], "/opt/wb-humanoid-robot/robot_models")
        self.assertIs(mount["read_only"], True)
        self.assertIn("WB_ROBOT_SIM_MODELS", mount["source"])
        self.assertIn(
            'WB_ROBOT_SIM_MODELS="${WB_HOST_CHECKOUT:-$(host_checkout)}/robot_models"',
            read("tools/deploy/session.sh"),
        )
        # LINT.ThenChange(//docker-compose.robot.sim.yaml:sim_models_mount)
        # LINT.IfChange(sim_plot_config_mount)
        # The checkout's plot configuration (the Makefile's PLOT_CONFIG, by default the shipped one), read-only, whose
        # mujoco_viewer names the MuJoCo viewer unless WB_ROBOT_VIEWER does.
        mount = mounts[SIM_PLOT_CONFIG_TARGET]
        self.assertEqual(mount["type"], "bind")
        self.assertIs(mount["read_only"], True)
        self.assertEqual(
            mount["source"],
            "${WB_ROBOT_SIM_PLOT_CONFIG:-./humanoid_nmpc/humanoid_rerun_viewer/config/plot_config.textproto}",
        )
        self.assertEqual(
            sim["environment"]["WB_ROBOT_PLOT_CONFIG"], SIM_PLOT_CONFIG_TARGET
        )
        session = read("tools/deploy/session.sh")
        self.assertIn(
            'WB_ROBOT_SIM_PLOT_CONFIG="${WB_HOST_CHECKOUT:-$(host_checkout)}/${robot_plot_config}"',
            session,
        )
        self.assertIn(
            f'sim_plot_config_target="{SIM_PLOT_CONFIG_TARGET}"', session
        )
        # LINT.ThenChange(//docker-compose.robot.sim.yaml:sim_plot_config_mount)

    def test_net_admin_only_with_the_netem_override(self) -> None:
        netem = load_yaml("docker-compose.robot.netem.yaml")["services"][
            "robot"
        ]
        self.assertEqual(netem, {"cap_add": ["NET_ADMIN"]})
        self.assertNotIn("NET_ADMIN", self.robot["cap_add"])


class _CaseSensitiveConfigParser(configparser.ConfigParser):
    """A ConfigParser that keeps the case of option names, as systemd does (ExecStart, not execstart)."""

    def optionxform(self, optionstr: str) -> str:
        return optionstr


def read_unit() -> configparser.ConfigParser:
    """The systemd unit of the deployment, parsed."""
    unit = _CaseSensitiveConfigParser(interpolation=None)
    unit.read_string(read("tools/deploy/wb-humanoid-robot@.service"))
    return unit


class SystemdUnitTest(unittest.TestCase):
    def test_the_unit_starts_the_deployment_after_docker_and_at_boot(
        self,
    ) -> None:
        unit = read_unit()
        self.assertIn("docker.service", unit["Unit"]["Requires"])
        self.assertIn("network-online.target", unit["Unit"]["After"])
        service = unit["Service"]
        self.assertEqual(service["WorkingDirectory"], "@DEPLOY_DIR@")
        self.assertEqual(service["Environment"], "ROBOT=%i")
        self.assertRegex(service["ExecStart"], r"docker compose up --detach")
        self.assertRegex(service["ExecStop"], r"docker compose down")
        self.assertEqual(unit["Install"]["WantedBy"], "multi-user.target")

    def test_an_instance_of_another_robot_than_the_deployed_one_refuses_to_start(
        self,
    ) -> None:
        # The deployment directory holds one robot's .env; a unit of another robot left enabled must not run the
        # container under its own name (they share the compose project). systemd expands %i and turns $$ into $.
        unit = read_unit()
        guard = unit["Service"]["ExecStartPre"]
        with tempfile.TemporaryDirectory() as directory:
            with open(
                os.path.join(directory, ".env"), "w", encoding="utf-8"
            ) as stream:
                stream.write(
                    "COMPOSE_PROJECT_NAME=wb-humanoid-robot\nROBOT=unitree_g1_wb\n"
                )

            def start(instance: str) -> subprocess.CompletedProcess:
                command = guard.replace("%i", instance).replace("$$", "$")
                self.assertTrue(command.startswith("/bin/sh -c '"))
                return subprocess.run(
                    ["sh", "-c", command[len("/bin/sh -c '") : -1]],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    check=False,  # The test checks the return code.
                )

            self.assertEqual(start("unitree_g1_wb").returncode, 0)
            refused = start("drc_atlas")
            self.assertEqual(refused.returncode, 1)
            self.assertIn("unitree_g1_wb", refused.stderr)
            self.assertIn("make deploy-robot ROBOT=drc_atlas", refused.stderr)


def dockerfile_stages() -> dict[str, str]:
    """Stage name -> the Dockerfile text from its FROM to the next."""
    text = read("docker/Dockerfile")
    stages: dict[str, str] = {}
    matches = list(re.finditer(r"^FROM (\S+) AS (\S+)$", text, re.M))
    for index, match in enumerate(matches):
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        stages[match.group(2)] = text[match.start() : end]
    return stages


def installed_packages(stage: str) -> list[str]:
    """The packages the `apt-get install` commands of a Dockerfile stage install."""
    joined = without_comments(stage).replace("\\\n", " ")
    packages: list[str] = []
    for match in re.finditer(
        r"apt-get install([^\n]*?)(?:&&|<|$)", joined, re.M
    ):
        packages += [
            word for word in match.group(1).split() if not word.startswith("-")
        ]
    return packages


class DockerfileTest(unittest.TestCase):
    # LINT.IfChange(robot_images)
    def setUp(self) -> None:
        self.stages = dockerfile_stages()

    def test_the_robot_stages(self) -> None:
        for name in (
            "robot-pinocchio-amd64",
            "robot-pinocchio-arm64",
            "robot-libraries",
            "robot-runtime",
            "robot-sim",
        ):
            self.assertIn(name, self.stages)
        self.assertTrue(
            self.stages["robot-runtime"].startswith(
                "FROM ${BASE_IMAGE} AS robot-runtime"
            )
        )
        self.assertTrue(
            self.stages["robot-sim"].startswith(
                "FROM robot-runtime AS robot-sim"
            )
        )
        self.assertTrue(
            self.stages["robot-libraries"].startswith(
                "FROM robot-pinocchio-${TARGETARCH} AS robot-libraries"
            )
        )

    def test_the_runtime_image_has_nothing_to_build_or_view_with(self) -> None:
        runtime = without_comments(self.stages["robot-runtime"])
        self.assertEqual(
            installed_packages(self.stages["robot-runtime"]), ["iproute2"]
        )
        self.assertRegex(runtime, r"< /tmp/robot-packages.txt")
        self.assertNotRegex(
            runtime + without_comments(self.stages["robot-sim"]),
            r"build-essential|\bg\+\+|\bgcc\b|clang|cmake|bazel|novnc|x11vnc|xvfb|python3",
        )
        self.assertIn("COPY --from=robot-libraries", runtime)
        self.assertNotIn("--from=base", runtime)
        self.assertIn(
            'ENTRYPOINT ["/opt/wb-humanoid-robot/bin/robot_entrypoint.sh"]',
            runtime,
        )
        # No viewer by default: the image has no GL to draw with.
        self.assertIn("ENV WB_ROBOT_DEFAULT_VIEWER=none", runtime)
        self.assertNotRegex(runtime, r"\blibegl")
        self.assertIn("collect_runtime_libraries.sh check", runtime)

    def test_the_sim_image_adds_the_viewers_gl_only(self) -> None:
        packages = installed_packages(self.stages["robot-sim"])
        self.assertTrue(packages)
        self.assertEqual(
            [p for p in packages if not re.match(r"^lib(gl|glx|egl|x)", p)], []
        )
        # The glfw viewer's GLX and the web viewer's EGL, each with Mesa's vendor library.
        for package in ("libglx-mesa0", "libegl1", "libegl-mesa0"):
            self.assertIn(package, packages)
        # No viewer of the image's own, in place of robot-runtime's none: the plot configuration's chooses (web).
        self.assertRegex(
            self.stages["robot-sim"],
            re.compile(r"^ENV WB_ROBOT_DEFAULT_VIEWER=$", re.M),
        )

    def test_pinocchio_from_robotpkg_on_amd64_and_from_source_on_arm64(
        self,
    ) -> None:
        self.assertIn(
            "RUN sh /tmp/install_robotpkg.sh",
            self.stages["robot-pinocchio-amd64"],
        )
        self.assertIn(
            "build_pinocchio_from_source.sh",
            self.stages["robot-pinocchio-arm64"],
        )
        base = self.stages["base"]
        self.assertIn(
            'if [ "$(dpkg --print-architecture)" = "amd64" ]; then sh'
            " /tmp/install_robotpkg.sh",
            base,
        )
        self.assertIn("else sh /tmp/build_pinocchio_from_source.sh", base)

    # LINT.ThenChange(//docker/Dockerfile:robot_images)


class DevContainerPlatformTest(unittest.TestCase):
    def test_the_aarch64_instructions_name_the_dev_containers_platform_variable(
        self,
    ) -> None:
        # The dev container is amd64 unless its variable says otherwise: an arm64 bundle needs that variable set.
        compose = read("docker-compose.yaml")
        variable = search(
            r"^\s*platform: \$\{(\w+):-linux/amd64\}$", compose, re.M
        ).group(1)
        section = search(
            r"### aarch64\n(.*?)\n## ", read("tools/deploy/README.md"), re.S
        )
        self.assertIn(f"{variable}=linux/arm64", section.group(1))
        self.assertIn(
            f"{variable}=linux/", read("tools/deploy/deploy_robot.sh")
        )


class DeployScriptTest(unittest.TestCase):
    def deploy(self, *arguments: str) -> subprocess.CompletedProcess:
        env = dict(os.environ)
        env["HOME"] = tempfile.gettempdir()
        return subprocess.run(
            ["bash", runfile("tools/deploy/deploy_robot.sh")] + list(arguments),
            capture_output=True,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
            timeout=60,
            check=False,  # The tests check the return code.
        )

    def test_the_scripts_parse(self) -> None:
        for script in ("deploy_robot.sh", "session.sh", "in_dev_container.sh"):
            with self.subTest(script=script):
                subprocess.run(
                    ["bash", "-n", runfile(f"tools/deploy/{script}")],
                    check=True,
                )
        for script in ("robot_entrypoint.sh", "netem.sh"):
            subprocess.run(
                ["sh", "-n", runfile(f"tools/deploy/{script}")], check=True
            )

    def test_the_environment_of_a_deployment(self) -> None:
        result = self.deploy(
            "environment",
            "--robot",
            "drc_atlas",
            "--cpuset",
            "2-5",
            "--realtime_priority",
            "70",
            "--netem",
            "delay 3ms 1ms loss 0.5%",
            "--netem_interface",
            "eth0",
            "--config_seed",
            "never",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.splitlines()
        for expected in (
            "COMPOSE_FILE=docker-compose.robot.yaml:docker-compose.robot.netem.yaml",
            "ROBOT=drc_atlas",
            "WB_ROBOT_IMAGE=wb-humanoid-robot:latest",
            "WB_ROBOT_NETWORK_SOURCE=./network.textproto",
            "WB_ROBOT_CPUSET=2-5",
            "WB_ROBOT_REALTIME_PRIORITY=70",
            'NETEM="delay 3ms 1ms loss 0.5%"',
            "NETEM_INTERFACE=eth0",
            "WB_ROBOT_CONFIG_SEED=never",
        ):
            self.assertIn(expected, lines)
        plain = self.deploy(
            "environment", "--robot", "drc_atlas"
        ).stdout.splitlines()
        self.assertIn("COMPOSE_FILE=docker-compose.robot.yaml", plain)
        self.assertEqual([line for line in plain if "REALTIME" in line], [])
        # Without --config_seed the launch file's policy applies (when_bundle_changes).
        self.assertEqual([line for line in plain if "CONFIG_SEED" in line], [])

    def test_a_remote_deployment_ships_the_image_and_installs_the_unit(
        self,
    ) -> None:
        result = self.deploy(
            "deploy",
            "--dry_run",
            "--skip_bundle",
            "--robot",
            "drc_atlas",
            "--host",
            "robot.local",
            "--network",
            "config/ipc/two_machine.example.textproto",
            "--service",
            "enable",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout
        self.assertRegex(
            output,
            r"docker build .*--target robot-runtime -t"
            r" wb-humanoid-robot:latest \.",
        )
        self.assertIn(
            "docker save wb-humanoid-robot:latest | gzip -1 | ssh robot.local",
            output,
        )
        for file in (
            "docker-compose.robot.yaml",
            "docker-compose.robot.netem.yaml",
            "network.textproto",
            ".env",
        ):
            self.assertRegex(
                output,
                rf"scp -q \S+ robot\.local:wb-humanoid-robot/{re.escape(file)}",
            )
        self.assertIn("wb-humanoid-robot@drc_atlas.service", output)
        # The stored configuration's directory, which a deploy never empties.
        self.assertRegex(output, r"mkdir -p .*wb-humanoid-robot/robot_config")
        # A unit that is active already (oneshot, RemainAfterExit) would ignore `enable --now`: a redeploy restarts it,
        # so that the container comes up again on the image just shipped, and the units of other robots go off.
        self.assertNotIn("enable --now", output)
        self.assertRegex(
            output,
            r"systemctl enable '\S+' && sudo systemctl restart"
            r" '\S+@drc_atlas\.service'",
        )
        self.assertIn("systemctl disable --now", output)
        # The image is built for the robot's architecture, which its Docker is asked for.
        self.assertIn("ssh robot.local docker version --format", output)

    def test_a_redeploy_restarts_a_running_deployment_on_the_new_image(
        self,
    ) -> None:
        result = self.deploy(
            "deploy", "--dry_run", "--skip_bundle", "--robot", "drc_atlas"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(
            result.stdout,
            r"docker compose ps --quiet robot .*then\s+docker compose up"
            r" --detach --no-build",
        )

    def test_up_refuses_a_robot_other_than_the_deployed_one(self) -> None:
        # One robot per machine: `up` of another robot than the deployment's .env would recreate the deployment's
        # container (one compose project) running that robot, which the unit refuses too. And ROBOT comes from the
        # .env alone: a ROBOT exported by the caller, or by make from its command line, must not override it.
        with tempfile.TemporaryDirectory() as home:
            deployment = os.path.join(home, "wb-humanoid-robot")
            os.makedirs(os.path.join(home, "bin"))
            os.makedirs(deployment)
            with open(
                os.path.join(deployment, ".env"), "w", encoding="utf-8"
            ) as env:
                env.write(
                    "COMPOSE_PROJECT_NAME=wb-humanoid-robot\nROBOT=unitree_g1\n"
                )
            log = os.path.join(home, "docker.log")
            docker = os.path.join(home, "bin", "docker")
            with open(docker, "w", encoding="utf-8") as stream:
                stream.write(
                    '#!/bin/sh\necho "$* ROBOT=${ROBOT-unset} cwd=$(pwd)" >>'
                    ' "$DOCKER_LOG"\n'
                )
            os.chmod(docker, 0o755)

            def up(
                robot: str, caller_robot: str | None
            ) -> subprocess.CompletedProcess:
                env = dict(os.environ)
                env["HOME"] = home
                env["DOCKER_LOG"] = log
                env["PATH"] = f"{os.path.join(home, 'bin')}:{env['PATH']}"
                env.pop("ROBOT", None)
                if caller_robot is not None:
                    env["ROBOT"] = caller_robot
                return subprocess.run(
                    [
                        "bash",
                        runfile("tools/deploy/deploy_robot.sh"),
                        "up",
                        "--robot",
                        robot,
                        "--host",
                        "localhost",
                    ],
                    capture_output=True,
                    text=True,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    timeout=60,
                    check=False,  # The test checks the return code.
                )

            def logged() -> str:
                if not os.path.exists(log):
                    return ""
                with open(log, encoding="utf-8") as stream:
                    return stream.read()

            refused = up("drc_atlas", caller_robot=None)
            self.assertEqual(refused.returncode, 1, refused.stdout)
            self.assertIn("unitree_g1", refused.stderr)
            self.assertIn("make deploy-robot ROBOT=drc_atlas", refused.stderr)
            self.assertEqual(logged(), "")

            started = up("unitree_g1", caller_robot="drc_atlas")
            self.assertEqual(started.returncode, 0, started.stderr)
            self.assertEqual(
                logged().splitlines(),
                [f"compose up ROBOT=unset cwd={os.path.realpath(deployment)}"],
            )

    def test_an_unknown_robot_is_refused_before_anything_is_built(self) -> None:
        result = self.deploy("deploy", "--dry_run", "--robot", "drc_atlass")
        self.assertEqual(result.returncode, 1)
        self.assertIn("drc_atlas engineai_sa01", result.stderr)
        self.assertNotIn("docker build", result.stdout)

    def test_a_registry_replaces_the_copy(self) -> None:
        output = self.deploy(
            "deploy",
            "--dry_run",
            "--skip_bundle",
            "--robot",
            "drc_atlas",
            "--host",
            "robot.local",
            "--network",
            "config/ipc/two_machine.example.textproto",
            "--registry",
            "registry.local:5000",
        ).stdout
        self.assertIn(
            "docker push registry.local:5000/wb-humanoid-robot:latest", output
        )
        self.assertIn("docker pull", output)
        self.assertNotIn("docker save", output)
        self.assertNotIn("systemctl", output)

    def test_a_registry_deployment_pulls_the_image_and_builds_nothing(
        self,
    ) -> None:
        result = self.deploy(
            "deploy",
            "--dry_run",
            "--robot",
            "drc_atlas",
            "--host",
            "robot.local",
            "--network",
            "config/ipc/two_machine.example.textproto",
            "--image_source",
            "registry",
            "--registry",
            "ghcr.io/owner",
            "--tag",
            "main-arm64",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = result.stdout
        image = "ghcr.io/owner/wb-humanoid-robot:main-arm64"
        self.assertRegex(
            output, rf"\+ ssh robot\.local .*docker pull .*{re.escape(image)}"
        )
        self.assertIn(
            f"check that {image} on robot.local is of the host's architecture",
            output,
        )
        for built in (
            "docker build",
            "docker save",
            "docker push",
            "in_dev_container.sh",
        ):
            self.assertNotIn(built, output)
        # The deployment's compose file runs the pulled image.
        self.assertIn(
            f"deployed to robot.local:wb-humanoid-robot ({image})", output
        )

    def test_a_registry_deployment_needs_the_registry(self) -> None:
        for arguments in (
            ("--image_source", "registry"),
            ("--image_source", "cache", "--registry", "ghcr.io/owner"),
        ):
            with self.subTest(arguments=arguments):
                result = self.deploy(
                    "deploy", "--dry_run", "--robot", "drc_atlas", *arguments
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("docker pull", result.stdout)

    def test_the_makefile_passes_the_tag_and_the_image_source(self) -> None:
        makefile = read("Makefile")
        self.assertIn("$(if $(TAG),--tag $(TAG))", makefile)
        self.assertIn(
            "$(if $(IMAGE_SOURCE),--image_source $(IMAGE_SOURCE))", makefile
        )

    def test_what_a_deployment_cannot_do_is_refused(self) -> None:
        remote = self.deploy(
            "deploy",
            "--dry_run",
            "--skip_bundle",
            "--robot",
            "drc_atlas",
            "--host",
            "robot.local",
        )
        self.assertEqual(remote.returncode, 1)
        self.assertIn("two_machine.example.textproto", remote.stderr)
        for arguments in (
            (
                "deploy",
                "--dry_run",
                "--robot",
                "drc_atlas",
                "--service",
                "always",
            ),
            ("image", "--dry_run", "--target", "robot-dev"),
            ("deploy", "--dry_run", "--robot", "../x"),
            ("deploy", "--dry_run"),
            (
                "environment",
                "--robot",
                "drc_atlas",
                "--config_seed",
                "sometimes",
            ),
        ):
            with self.subTest(arguments=arguments):
                self.assertNotEqual(self.deploy(*arguments).returncode, 0)

    @unittest.skipUnless(
        os.path.exists("/.dockerenv"), "runs in place only inside a container"
    )
    def test_in_a_container_the_command_runs_in_the_checkout(self) -> None:
        result = subprocess.run(
            ["bash", runfile("tools/deploy/in_dev_container.sh"), "pwd"],
            capture_output=True,
            text=True,
            check=False,  # The assertion shows stderr.
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            os.path.realpath(result.stdout.strip()), os.path.realpath(RUNFILES)
        )
        self.assertEqual(
            subprocess.run(
                ["bash", runfile("tools/deploy/in_dev_container.sh")],
                capture_output=True,
                check=False,
            ).returncode,
            2,
        )


# The launch targets each robot configuration of the Makefile gets, and those of a robot with a sandbox.
# LINT.IfChange(launch_targets)
CONFIGURATION_TARGETS = (
    "sim",
    "sim-vnc",
    "dummy-sim",
    "dummy-sim-vnc",
    "robot",
    "mpc",
    "mpc-vnc",
)
# The targets whose -vnc twin is the target with VIEWS=vnc.
VNC_TWINS = ("sim", "dummy-sim", "mpc", "sandbox")
SANDBOX_TARGETS = ("sandbox", "sandbox-vnc")
# LINT.ThenChange(//Makefile:launch_targets)


# The views of the Makefile's VIEWS: each one's MuJoCo viewer and Rerun sink.
# LINT.IfChange(views)
VIEWS = {
    "browser": ("web", "serve_web"),
    "vnc": ("glfw", "spawn"),
    "display": ("glfw", "spawn"),
}
# LINT.ThenChange(//Makefile:views, //tools/deploy/session.sh:views)


def make_dry_run(*arguments: str) -> subprocess.CompletedProcess:
    """`make -n` of the checkout's Makefile with `arguments`: the commands a target would run, without running them."""
    return subprocess.run(
        ["make", "-n", "--no-print-directory", "-C", RUNFILES]
        + list(arguments),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,  # The tests check the return code.
    )


def session_command(output: str) -> str:
    """The session.sh line of a dry run's output that starts a session (kill-sims' `session.sh stop` is not one)."""
    lines = [
        line
        for line in output.splitlines()
        if "tools/deploy/session.sh" in line
        and not line.rstrip().endswith(" stop")
    ]
    assert len(lines) == 1, output
    return lines[0]


def sandbox_robots(makefile: str) -> list[str]:
    """The robots the Makefile's SANDBOX_ROBOTS gives a sandbox."""
    return search(r"^SANDBOX_ROBOTS := (.*)$", makefile, re.M).group(1).split()


def label(target: str) -> str:
    """//a/b as //a/b:b, the label a launch file's binary path names."""
    return target if ":" in target else f"{target}:{target.rsplit('/', 1)[1]}"


def launch_binaries(launch_file: str) -> set[str]:
    """The Bazel labels of the binaries a launch file's processes start ({bin_dir}/<package>/<name>)."""
    binaries = set()
    for path in re.findall(r'"\{bin_dir\}/([^"]+)"', read(launch_file)):
        package, name = path.rsplit("/", 1)
        binaries.add(f"//{package}:{name}")
    return binaries


class MakefileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.makefile = read("Makefile")
        block = search(
            r"^ROBOT_CONFIGURATIONS := \\\n((?:\t.*\n)+)", self.makefile, re.M
        )
        self.configurations = [
            line.strip().rstrip("\\").strip().split(":")
            for line in block.group(1).splitlines()
        ]
        self.targets: set[str] = set()
        for configuration in self.configurations:
            self.targets.update(
                f"launch-{configuration[0]}-{suffix}"
                for suffix in CONFIGURATION_TARGETS
            )
        for name in sandbox_robots(self.makefile):
            self.targets.update(
                f"launch-{name}-{suffix}" for suffix in SANDBOX_TARGETS
            )

    def test_every_configuration_names_a_robot_package_and_its_formulation(
        self,
    ) -> None:
        for (
            name,
            robot,
            package,
            formulation,
            description,
        ) in self.configurations:
            with self.subTest(name=name):
                self.assertRegex(robot, r"^[a-z0-9_]+$")
                self.assertTrue(
                    package.startswith("robot_models/")
                    and package.endswith("_mpc")
                )
                self.assertTrue(description.endswith("_description"))
                self.assertEqual(
                    formulation, "wb" if "_wb_" in package else "centroidal"
                )

    def test_the_dev_container_readme_lists_the_launch_targets(self) -> None:
        readme = read(".devcontainer/README.md")
        block = search(
            r"LINT\.IfChange\(launch_targets\) -->\n(.*?)<!-- LINT\.ThenChange",
            readme,
            re.S,
        )
        listed = set(re.findall(r"`make (launch-[a-z0-9-]+)", block.group(1)))
        self.assertTrue(listed)
        self.assertEqual(listed - self.targets, set())
        self.assertEqual(
            {t for t in self.targets if t.endswith("-vnc")} - listed, set()
        )

    def makefile_variable(self, name: str) -> list[str]:
        """The words of a `name := ...` line of the Makefile, with $(other) variables expanded."""
        match = search(rf"^{name} := (.*)$", self.makefile, re.M)
        words: list[str] = []
        for word in match.group(1).split():
            reference = re.fullmatch(r"\$\((\w+)\)", word)
            words += (
                self.makefile_variable(reference.group(1))
                if reference
                else [word]
            )
        return words

    # LINT.IfChange(launch_binaries)
    def test_every_launch_target_builds_every_binary_its_launch_file_starts(
        self,
    ) -> None:
        laptop = {label(t) for t in self.makefile_variable("laptop_binaries")}
        for name, _, package, formulation, description in self.configurations:
            for launch, built in (
                ("mpc", self.makefile_variable(f"mpc_binaries_{formulation}")),
                (
                    "dummy_sim",
                    self.makefile_variable(f"dummy_binaries_{formulation}"),
                ),
            ):
                started = launch_binaries(
                    f"{package}/launch/{launch}.textproto"
                )
                with self.subTest(target=f"launch-{name}", launch=launch):
                    self.assertTrue(started)
                    self.assertEqual(
                        started - laptop - {label(t) for t in built}, set()
                    )
        sandboxes = sandbox_robots(self.makefile)
        sandbox = {label(t) for t in self.makefile_variable("sandbox_binaries")}
        for name, _, _, _, description in self.configurations:
            if name not in sandboxes:
                continue
            started = launch_binaries(f"{description}/launch/sandbox.textproto")
            with self.subTest(target=f"launch-{name}-sandbox"):
                self.assertTrue(started)
                self.assertEqual(started - sandbox, set())

    # LINT.ThenChange(//Makefile:launch_binaries)

    # LINT.IfChange(robot_names)
    def test_the_deployment_script_knows_the_robots_of_the_makefile(
        self,
    ) -> None:
        robots = search(
            r'^ROBOTS="([^"]*)"$', read("tools/deploy/deploy_robot.sh"), re.M
        ).group(1)
        self.assertEqual(
            sorted(robots.split()),
            sorted({configuration[1] for configuration in self.configurations}),
        )

    # LINT.ThenChange(//tools/deploy/deploy_robot.sh:robot_names)

    @unittest.skipUnless(shutil.which("make"), "needs make")
    def test_deploy_robot_passes_the_backend_the_memory_limit_and_a_checkout_relative_network(
        self,
    ) -> None:
        def make(*arguments: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["make", "-n", "--no-print-directory", "-C", RUNFILES]
                + list(arguments),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,  # The test checks the return code.
            )

        network = os.path.join(
            RUNFILES, "config/ipc/two_machine.example.textproto"
        )
        result = make(
            "deploy-robot",
            "ROBOT=drc_atlas",
            "HOST=robot.local",
            "BACKEND=some_hardware",
            "MEMORY_LIMIT=6g",
            "CONFIG_SEED=every_start",
            f"NETWORK={network}",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--backend some_hardware", result.stdout)
        self.assertIn("--config_seed every_start", result.stdout)
        self.assertIn("--memory_limit 6g", result.stdout)
        self.assertIn(
            "--network config/ipc/two_machine.example.textproto", result.stdout
        )
        outside = make(
            "deploy-robot", "ROBOT=drc_atlas", "NETWORK=/etc/hostname"
        )
        self.assertNotEqual(outside.returncode, 0)
        self.assertIn("outside the checkout", outside.stderr)

    def test_the_launch_targets_pass_the_plots_and_the_recording_relative_to_the_checkout(
        self,
    ) -> None:
        # The dev container mounts the checkout elsewhere than the host keeps it: an absolute path of the host inside
        # the checkout reaches the bridge relative to the checkout, on every launch target, not only -sim.
        def make(*arguments: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["make", "-n", "--no-print-directory", "-C", RUNFILES]
                + list(arguments),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,  # The test checks the return code.
            )

        plots = os.path.join(RUNFILES, SHIPPED_PLOT_CONFIG)
        recording = os.path.join(RUNFILES, "walk.rrd")
        for target, extra in (
            ("launch-drc-atlas-dummy-sim", ()),
            ("launch-drc-atlas-sandbox", ()),
            ("launch-drc-atlas-mpc", ("NETWORK=config/ipc/network.textproto",)),
        ):
            with self.subTest(target=target):
                result = make(
                    target,
                    f"PLOT_CONFIG={plots}",
                    f"RRD_PATH={recording}",
                    *extra,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"--set plot_config={SHIPPED_PLOT_CONFIG}", result.stdout
                )
                self.assertIn("--set rrd_path=walk.rrd", result.stdout)
        outside = make(
            "launch-drc-atlas-dummy-sim", "PLOT_CONFIG=/etc/hostname"
        )
        self.assertNotEqual(outside.returncode, 0)
        self.assertIn(
            "PLOT_CONFIG=/etc/hostname is outside the checkout", outside.stderr
        )
        # A recording outside the checkout is the dev container's own path, kept as it is.
        elsewhere = make("launch-drc-atlas-dummy-sim", "RRD_PATH=/tmp/walk.rrd")
        self.assertEqual(elsewhere.returncode, 0, elsewhere.stderr)
        self.assertIn("--set rrd_path=/tmp/walk.rrd", elsewhere.stdout)

    def test_the_views_are_the_sessions(self) -> None:
        self.assertEqual(
            search(r"^VIEWS_NAMES := (.*)$", self.makefile, re.M)
            .group(1)
            .split(),
            list(VIEWS),
        )
        self.assertEqual(
            search(
                r'^VIEWS_NAMES="([^"]*)"$',
                read("tools/deploy/session.sh"),
                re.M,
            )
            .group(1)
            .split(),
            list(VIEWS),
        )
        self.assertRegex(
            self.makefile, re.compile(r"^VIEWS \?= browser$", re.M)
        )
        for view, (viewer, sink) in VIEWS.items():
            with self.subTest(view=view):
                self.assertEqual(
                    self.makefile_variable(f"views_viewer_{view}"), [viewer]
                )
                self.assertEqual(
                    self.makefile_variable(f"views_rerun_sink_{view}"), [sink]
                )

    def test_every_vnc_target_is_its_twin_with_the_vnc_view(self) -> None:
        for twin in VNC_TWINS:
            with self.subTest(target=twin):
                self.assertRegex(
                    self.makefile,
                    re.compile(
                        rf"^launch-\$\(call field,\$\(1\),1\)-{twin}-vnc: "
                        rf"launch-\$\(call field,\$\(1\),1\)-{twin}$",
                        re.M,
                    ),
                )
                self.assertRegex(
                    self.makefile,
                    re.compile(
                        r"^(?:\S+ )*launch-\$\(call"
                        rf" field,\$\(1\),1\)-{twin}-vnc[ :]"
                        r".*override VIEWS = vnc$",
                        re.M,
                    ),
                )

    @unittest.skipUnless(shutil.which("make"), "needs make")
    def test_a_view_picks_the_mujoco_viewer_and_the_rerun_sink(self) -> None:
        network = "NETWORK=config/ipc/network.textproto"
        cases = (
            # (arguments, the session's view, its sink, the robot's viewer or None for the plot configuration's)
            (("launch-drc-atlas-sim",), "browser", "serve_web", None),
            (
                ("launch-drc-atlas-sim", "VIEWS=browser"),
                "browser",
                "serve_web",
                "web",
            ),
            (("launch-drc-atlas-sim-vnc",), "vnc", "spawn", "glfw"),
            (
                ("launch-drc-atlas-sim-vnc", "VIEWS=browser"),
                "vnc",
                "spawn",
                "glfw",
            ),
            (
                ("launch-drc-atlas-sim", "VIEWS=display"),
                "display",
                "spawn",
                "glfw",
            ),
            (
                (
                    "launch-drc-atlas-sim-vnc",
                    "VIEWER=web",
                    "RERUN_SINK=serve_web",
                ),
                "vnc",
                "serve_web",
                "web",
            ),
            (("launch-drc-atlas-mpc-vnc", network), "vnc", "spawn", None),
            (
                ("launch-drc-atlas-mpc", network, "RERUN_SINK=save"),
                "browser",
                "save",
                None,
            ),
            (("launch-drc-atlas-dummy-sim-vnc",), "vnc", "spawn", None),
            (("launch-drc-atlas-sandbox-vnc",), "vnc", "spawn", None),
        )
        for arguments, view, sink, viewer in cases:
            with self.subTest(arguments=arguments):
                result = make_dry_run(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                command = session_command(result.stdout)
                self.assertIn(f"--views {view} ", command)
                self.assertEqual(
                    re.findall(r"--set rerun_sink=(\S+)", command), [sink]
                )
                if viewer is None:
                    self.assertNotIn("--viewer", command)
                else:
                    self.assertIn(f'--viewer "{viewer}"', command)
        unknown = make_dry_run("launch-drc-atlas-sim", "VIEWS=desktop")
        self.assertNotEqual(unknown.returncode, 0)
        self.assertIn("VIEWS=desktop is not a view", unknown.stderr)
        self.assertIn("browser vnc display", unknown.stderr)

    def test_no_ros_remains_and_no_launch_runs_the_mpc_in_the_robot_process(
        self,
    ) -> None:
        code = without_comments(self.makefile)
        self.assertNotRegex(code, r"ros2 |rviz|plotjuggler")
        self.assertNotIn("in_process", code)
        self.assertRegex(code, r"\$\(session\) sim --robot")

    def test_launch_and_deploy_targets_raise_the_udp_buffers_first(
        self,
    ) -> None:
        """Every target that starts the bus first raises the kernel's UDP buffer limits on the machine that runs it."""
        network = "config/ipc/two_machine.example.textproto"
        cases = (
            (
                ("launch-drc-atlas-sim",),
                "config/ipc/network.textproto",
                "localhost",
                "session.sh sim",
            ),
            (
                ("launch-drc-atlas-dummy-sim",),
                "config/ipc/network.textproto",
                "localhost",
                "session.sh laptop",
            ),
            (
                ("launch-drc-atlas-mpc", f"NETWORK={network}"),
                network,
                "localhost",
                "session.sh laptop",
            ),
            (
                ("launch-drc-atlas-robot", "HOST=robot"),
                "config/ipc/network.textproto",
                "robot",
                "deploy_robot.sh up",
            ),
            (
                (
                    "deploy-robot",
                    "ROBOT=drc_atlas",
                    "HOST=robot",
                    f"NETWORK={network}",
                ),
                network,
                "robot",
                "deploy_robot.sh deploy",
            ),
            (
                ("udp-buffers", "HOST=robot"),
                "config/ipc/network.textproto",
                "robot",
                None,
            ),
        )
        for arguments, expected_network, expected_host, starter in cases:
            with self.subTest(arguments=arguments):
                result = make_dry_run(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = result.stdout.splitlines()
                buffers = [
                    i
                    for i, line in enumerate(lines)
                    if "tools/deploy/udp_buffers.sh" in line
                ]
                self.assertEqual(len(buffers), 1, result.stdout)
                command = lines[buffers[0]]
                self.assertRegex(
                    command, rf"--network \S*/{re.escape(expected_network)} "
                )
                self.assertTrue(
                    command.rstrip().endswith(f"--host {expected_host}"),
                    command,
                )
                if starter is not None:
                    starts = [
                        i for i, line in enumerate(lines) if starter in line
                    ]
                    self.assertTrue(
                        starts and buffers[0] < starts[0], result.stdout
                    )


class SessionTest(unittest.TestCase):
    # LINT.IfChange(sim_project)
    def test_every_checkout_has_a_simulation_project_of_its_own(self) -> None:
        # `make kill-sims` in one checkout must not take down the robot container another checkout's session started.
        projects = []
        with tempfile.TemporaryDirectory() as directory:
            for checkout in ("wb_humanoid_mpc", "Other_Checkout"):
                deploy = os.path.join(directory, checkout, "tools", "deploy")
                os.makedirs(deploy)
                shutil.copy(runfile("tools/deploy/session.sh"), deploy)
                result = subprocess.run(
                    ["bash", os.path.join(deploy, "session.sh"), "project"],
                    capture_output=True,
                    text=True,
                    check=False,  # The assertion shows stderr.
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                projects.append(result.stdout.strip())
        self.assertNotEqual(projects[0], projects[1])
        for project in projects:
            self.assertTrue(
                project.startswith("wb-humanoid-robot-sim-"), project
            )
            self.assertRegex(
                project, r"^[a-z0-9][a-z0-9_-]*$", "a compose project name"
            )
            self.assertNotEqual(
                project, "wb-humanoid-robot", "the deployment's project"
            )

    # LINT.ThenChange(//tools/deploy/session.sh:sim_project)

    def session(self, *arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", runfile("tools/deploy/session.sh"), *arguments],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
            check=False,  # The tests check the return code.
        )

    def test_an_unknown_option_stops_the_session_at_its_usage(self) -> None:
        # The usage is the script's header comment, all of it and no code, printed before Docker is reached.
        unknown = self.session(
            "sim", "--robot", "drc_atlas", "--no_such_option"
        )
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unknown option --no_such_option", unknown.stderr)
        self.assertIn("Environment: DISPLAY", unknown.stderr)
        self.assertNotIn("set -euo pipefail", unknown.stderr)

    def test_the_viewer_is_chosen_by_name(self) -> None:
        # --viewer takes a name; without a launch file the session stops at its usage, before Docker.
        named = self.session("sim", "--robot", "drc_atlas", "--viewer", "web")
        self.assertEqual(named.returncode, 2)
        self.assertNotIn("unknown option", named.stderr)
        self.assertIn("--viewer glfw|web|none", named.stderr)
        # Without --viewer, the plot configuration's mujoco_viewer chooses: --plot_config names it.
        configured = self.session(
            "sim", "--robot", "drc_atlas", "--plot_config", "my.textproto"
        )
        self.assertEqual(configured.returncode, 2)
        self.assertNotIn("unknown option", configured.stderr)
        self.assertIn("[--plot_config <file>]", configured.stderr)
        # A file the robot container cannot see, or none, is refused before anything is built or stopped.
        launch = (
            "sim",
            "--robot",
            "drc_atlas",
            "--launch_file",
            "mpc.textproto",
        )
        outside = self.session(
            *launch, "--plot_config", "/elsewhere/plots.textproto"
        )
        self.assertEqual(outside.returncode, 2)
        self.assertIn("is outside the checkout", outside.stderr)
        missing = self.session(
            *launch, "--plot_config", "no_such_plots.textproto"
        )
        self.assertEqual(missing.returncode, 2)
        self.assertIn("no such file in the checkout", missing.stderr)

    def test_the_sim_targets_pass_the_plot_configuration_to_the_robot_side(
        self,
    ) -> None:
        makefile = read("Makefile")
        targets = re.findall(
            r"^launch-\$\(call field,\$\(1\),1\)-sim:.*\n((?:\t.*\n)+)",
            makefile,
            re.M,
        )
        self.assertEqual(len(targets), 1)
        self.assertIn('--plot_config "$$(PLOT_CONFIG)"', targets[0])
        self.assertIn("$$(sim_viewer)", targets[0])
        self.assertRegex(makefile, re.compile(r"^VIEWER \?=$", re.M))

    def fake_checkout(self, files: dict[str, str]) -> str:
        """A checkout with session.sh, the shipped plot configuration (web) and `files`, whose sessions run on stand-ins."""
        checkout = os.path.realpath(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, checkout)
        executables = {
            "tools/deploy/in_dev_container.sh": FAKE_RECORDER,
            "tools/deploy/deploy_robot.sh": FAKE_RECORDER,
            "fake_bin/docker": FAKE_DOCKER,
            "fake_bin/tc": "#!/bin/sh\n",
        }
        texts = {
            SHIPPED_PLOT_CONFIG: f'{PLOT_CONFIG_HEADER}mujoco_viewer: "web"\n',
            **files,
            **executables,
        }
        for name, text in texts.items():
            path = os.path.join(checkout, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(text)
            if name in executables:
                os.chmod(path, 0o755)
        shutil.copy(
            runfile("tools/deploy/session.sh"),
            os.path.join(checkout, "tools", "deploy"),
        )
        return checkout

    def run_sim(
        self,
        checkout: str,
        *arguments: str,
        environment: dict[str, str] | None = None,
        laptop: str | None = None,
    ) -> tuple[subprocess.CompletedProcess, list[str]]:
        """Runs `session.sh sim` with `arguments` in a fake checkout.

        Args:
            checkout: A checkout of fake_checkout().
            *arguments: The session's arguments after its launch file and build.
            environment: Replaces the session's WB_HOST_CHECKOUT (HOST_CHECKOUT) with these variables.
            laptop: A launch file: runs `session.sh laptop` of it instead.

        Returns:
            How it ended, and the command lines the stand-ins recorded, in order.
        """
        log = os.path.join(checkout, "session.log")
        if os.path.exists(log):
            os.remove(log)
        env = dict(os.environ)
        env.pop("WB_ROBOT_WEB_VIEWER_PORT", None)
        env.update(
            PATH=os.pathsep.join(
                (os.path.join(checkout, "fake_bin"), env["PATH"])
            ),
            SESSION_LOG=log,
            WB_HOST_CHECKOUT=HOST_CHECKOUT,
        )
        if environment is not None:
            env.pop("WB_HOST_CHECKOUT")
            env.update(environment)
        command = (
            ["sim", "--robot", "drc_atlas"] if laptop is None else ["laptop"]
        ) + ["--launch_file", laptop or "mpc.textproto"]
        result = subprocess.run(
            [
                "bash",
                os.path.join(checkout, "tools", "deploy", "session.sh"),
                *command,
                "--build",
                "//tools/launch",
                "--skip_build",
                *arguments,
            ],
            capture_output=True,
            text=True,
            env=env,
            stdin=subprocess.DEVNULL,
            timeout=60,
            check=False,  # The tests check the return code.
        )
        recorded: list[str] = []
        if os.path.exists(log):
            with open(log, encoding="utf-8") as stream:
                recorded = stream.read().splitlines()
        return result, recorded

    def laptop_plot_config(self, recorded: list[str]) -> str:
        """The plot_config the laptop side's launcher gets: its last --set of it."""
        launches = [line for line in recorded if "tools/launch/launch " in line]
        self.assertEqual(len(launches), 1, recorded)
        return re.findall(r"--set plot_config=(\S*)", launches[0])[-1]

    def robot_container(self, recorded: list[str]) -> tuple[str, str]:
        """The viewer and the plot configuration of the robot container the session started."""
        starts = [
            line
            for line in recorded
            if line.startswith("docker compose ") and " up " in line
        ]
        self.assertEqual(len(starts), 1, recorded)
        match = search(r" viewer=(\S*) plot_config=(\S*)$", starts[0])
        return match.group(1), match.group(2)

    def robot_network(self, recorded: list[str]) -> str:
        """The network file the robot container the session started mounts."""
        (start,) = [
            line
            for line in recorded
            if line.startswith("docker compose ") and " up " in line
        ]
        return search(r"\| network=(\S*) ", start).group(1)

    def test_the_robot_container_mounts_the_checkouts_network_file_by_its_host_path(
        self,
    ) -> None:
        # Compose resolves its default network file relative to where it runs: in a dev container, a path the host's
        # Docker does not have. The session names the host's path of the checkout's file.
        checkout = self.fake_checkout({})
        result, recorded = self.run_sim(checkout)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.robot_network(recorded),
            f"{HOST_CHECKOUT}/config/ipc/network.textproto",
        )

    @unittest.skipUnless(
        os.path.exists("/.dockerenv"), "finds its container only inside one"
    )
    def test_in_a_container_on_the_hosts_network_the_checkout_is_found_by_the_containers_id(
        self,
    ) -> None:
        # On the host's network a container's hostname is the host's, which names no container: the session reads its
        # own container's id from the mount of /etc/hostname, and asks Docker for that container's mounts.
        checkout = self.fake_checkout({})
        container_id = "abc0123456789def"
        mountinfo = os.path.join(checkout, "mountinfo")
        with open(mountinfo, "w", encoding="utf-8") as stream:
            stream.write(
                "1 0 259:8 / / rw - ext4 /dev/root rw\n2 1 259:8"
                f" /var/lib/docker/containers/{container_id}/hostname"
                " /etc/hostname rw,relatime - ext4 /dev/nvme0n1p5 rw\n"
            )
        result, recorded = self.run_sim(
            checkout,
            environment={
                "WB_MOUNTINFO": mountinfo,
                "FAKE_CONTAINER_ID": container_id,
                "FAKE_MOUNT_SOURCE": "/srv/host_checkout",
                "FAKE_MOUNT_DESTINATION": checkout,
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.robot_network(recorded),
            "/srv/host_checkout/config/ipc/network.textproto",
        )
        self.assertEqual(
            self.robot_container(recorded),
            ("", f"/srv/host_checkout/{SHIPPED_PLOT_CONFIG}"),
        )

    def test_an_absolute_plot_configuration_reaches_both_sides_relative_to_the_checkout(
        self,
    ) -> None:
        # The Makefile gives PLOT_CONFIG to the laptop side (--set) and to the session (--plot_config); an absolute path
        # of the host is not one of the dev container, which mounts the checkout elsewhere, so both sides get the path
        # relative to the checkout: the robot container its file of the host, the launcher the same file in the dev
        # container's checkout.
        checkout = self.fake_checkout(
            {
                "plots/mine.textproto": (
                    f'{PLOT_CONFIG_HEADER}mujoco_viewer: "none"\n'
                )
            }
        )
        absolute = os.path.join(checkout, "plots", "mine.textproto")
        result, recorded = self.run_sim(
            checkout,
            "--set",
            f"plot_config={absolute}",
            "--plot_config",
            absolute,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.laptop_plot_config(recorded), "plots/mine.textproto"
        )
        self.assertEqual(
            self.robot_container(recorded),
            ("", f"{HOST_CHECKOUT}/plots/mine.textproto"),
        )
        # The robot's errors name the file by the path the container sees: the session says which file that is.
        self.assertIn(
            "the robot reads its MuJoCo viewer from plots/mine.textproto,"
            f" which it sees as {SIM_PLOT_CONFIG_TARGET}",
            result.stdout,
        )
        self.assertNotIn("localhost:6090", result.stdout, "its viewer is none")

    def test_a_plot_configuration_the_robot_cannot_read_is_refused_before_anything_starts(
        self,
    ) -> None:
        # A plain list of signals, which the Rerun bridge reads, is no PlotConfig textproto: without a viewer named, the
        # robot would refuse it at start-up, after the build and the stop of the running simulation.
        checkout = self.fake_checkout(
            {"plots/signals.txt": "base_pose/*\njoints/*\n"}
        )
        refused, recorded = self.run_sim(
            checkout, "--plot_config", "plots/signals.txt"
        )
        self.assertEqual(refused.returncode, 2)
        self.assertIn(
            "does not declare '# proto-message:"
            " humanoid_rerun_viewer_proto.PlotConfig'",
            refused.stderr,
        )
        self.assertIn("VIEWER=web|glfw|none", refused.stderr)
        self.assertEqual(recorded, [], "nothing was built, stopped or started")
        # A viewer named on the command line is the robot's, which then does not read the file.
        named, recorded = self.run_sim(
            checkout, "--plot_config", "plots/signals.txt", "--viewer", "glfw"
        )
        self.assertEqual(named.returncode, 0, named.stderr)
        self.assertEqual(
            self.robot_container(recorded),
            ("glfw", f"{HOST_CHECKOUT}/plots/signals.txt"),
        )
        self.assertEqual(self.laptop_plot_config(recorded), "plots/signals.txt")
        self.assertNotIn("the robot reads its MuJoCo viewer from", named.stdout)

    def test_without_plots_the_robot_gets_the_shipped_plot_configuration(
        self,
    ) -> None:
        # The Rerun bridge's "none" turns its plots off; the robot reads the shipped file's viewer, web.
        checkout = self.fake_checkout({})
        result, recorded = self.run_sim(
            checkout,
            "--set",
            "plot_config=none",
            "--plot_config",
            "none",
            environment=self.robot_log(_SERVING_LOCALHOST, _VIEWER_WEB),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.laptop_plot_config(recorded), "none")
        self.assertEqual(
            self.robot_container(recorded),
            ("", f"{HOST_CHECKOUT}/{SHIPPED_PLOT_CONFIG}"),
        )
        self.assertIn("http://localhost:6090", result.stdout)

    def robot_log(self, *lines: str) -> dict[str, str]:
        """The session's environment with the robot container's log of `lines`.

        Args:
            *lines: The lines `compose logs` answers.

        Returns:
            The variables run_sim() gives the session.
        """
        return {
            "WB_HOST_CHECKOUT": HOST_CHECKOUT,
            "FAKE_ROBOT_LOG": "\n".join(lines),
        }

    def announcements(self, stdout: str) -> list[str]:
        """The session's lines that give the web viewer's address."""
        return [
            line
            for line in stdout.splitlines()
            if line.startswith("session.sh: the MuJoCo web viewer")
        ]

    def test_the_viewer_address_waits_for_the_robot_to_serve_it(self) -> None:
        # The address used to be printed as soon as the container was created, about a second before the robot bound
        # the port, so a browser opening it at once got "connection refused" and never retried.
        checkout = self.fake_checkout({})
        result, _ = self.run_sim(checkout, environment=self.robot_log())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.announcements(result.stdout), [])
        self.assertIn("once it serves it", result.stdout)

    def test_a_viewer_that_fell_back_announces_no_address(self) -> None:
        checkout = self.fake_checkout({})
        result, _ = self.run_sim(
            checkout,
            environment=self.robot_log(
                _SERVING_LOCALHOST,
                "[MujocoSimInterface] the 'web' viewer did not start (no EGL);"
                " trying 'none'",
                "[MujocoSimInterface] viewer: none",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.announcements(result.stdout), [])

    def test_the_announced_address_is_the_servers_own(self) -> None:
        # WB_ROBOT_WEB_VIEWER_ADDRESS binds an interface address, which localhost does not reach.
        checkout = self.fake_checkout({})
        result, _ = self.run_sim(
            checkout,
            environment=self.robot_log(
                "I1019 [WebViewerServer] serving the MuJoCo viewer at"
                " http://192.0.2.10:6090/",
                _VIEWER_WEB,
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        [announcement] = self.announcements(result.stdout)
        self.assertIn("http://192.0.2.10:6090/", announcement)
        self.assertIn("ssh -L 6090:192.0.2.10:6090", announcement)
        self.assertNotIn("http://localhost", announcement)

    def test_the_browser_view_starts_the_vnc_desktop_and_prints_three_tabs(
        self,
    ) -> None:
        checkout = self.fake_checkout({})
        result, recorded = self.run_sim(
            checkout,
            "--views",
            "browser",
            "--set",
            "rerun_sink=serve_web",
            environment=self.robot_log(_SERVING_LOCALHOST, _VIEWER_WEB),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        starts = [line for line in recorded if "start_vnc.sh" in line]
        self.assertEqual(len(starts), 1, recorded)
        # The desktop comes up before the robot container, whose glfw window would need it.
        compose_up = next(
            i
            for i, line in enumerate(recorded)
            if " up " in line and "docker" in line
        )
        self.assertLess(recorded.index(starts[0]), compose_up)
        for url in (
            "http://localhost:9090/?url=rerun+http://localhost:9876/proxy",
            "http://localhost:6090",
            "http://localhost:6080/vnc.html",
        ):
            self.assertIn(url, result.stdout)
        self.assertIn("(the VNC desktop :99)", result.stdout)

    def test_the_vnc_view_puts_everything_on_the_desktop(self) -> None:
        checkout = self.fake_checkout({})
        result, recorded = self.run_sim(
            checkout,
            "--views",
            "vnc",
            "--viewer",
            "glfw",
            "--set",
            "rerun_sink=spawn",
            laptop="dummy_sim.textproto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue([line for line in recorded if "start_vnc.sh" in line])
        self.assertIn("Everything is on the VNC desktop :99", result.stdout)
        self.assertIn("native viewer, on DISPLAY=:99", result.stdout)

    def test_the_display_view_fails_before_the_build_without_an_x_server(
        self,
    ) -> None:
        checkout = self.fake_checkout({})
        no_server, recorded = self.run_sim(
            checkout,
            "--views",
            "display",
            environment={
                "WB_HOST_CHECKOUT": HOST_CHECKOUT,
                "DISPLAY": ":42",
                "FAKE_NO_X_SERVER": "1",
            },
        )
        self.assertEqual(no_server.returncode, 2)
        self.assertIn("no X server answers on DISPLAY=:42", no_server.stderr)
        self.assertIn("VIEWS=browser", no_server.stderr)
        self.assertFalse(
            [
                line
                for line in recorded
                if "tools/launch" in line or " up " in line
            ],
            recorded,
        )
        answered, recorded = self.run_sim(
            checkout,
            "--views",
            "display",
            laptop="dummy_sim.textproto",
            environment={"WB_HOST_CHECKOUT": HOST_CHECKOUT, "DISPLAY": ":42"},
        )
        self.assertEqual(answered.returncode, 0, answered.stderr)
        self.assertFalse([line for line in recorded if "start_vnc.sh" in line])
        self.assertIn("a window on DISPLAY=:42", answered.stdout)

    def test_an_unknown_view_is_refused(self) -> None:
        checkout = self.fake_checkout({})
        result, _ = self.run_sim(checkout, "--views", "desktop")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--views desktop is not a view", result.stderr)


# The robot's log lines the session announces the web viewer from (WebViewerServer.cpp, MujocoSimInterface.cpp).
_SERVING_LOCALHOST = (
    "I1019 [WebViewerServer] serving the MuJoCo viewer at"
    " http://localhost:6090/"
)
_VIEWER_WEB = (
    "I1019 MujocoSimInterface.cpp:915] [MujocoSimInterface] viewer: web"
)


def _arm64_workflow() -> dict:
    """.github/workflows/robot_arm64.yml, parsed."""
    return yaml.safe_load(read(".github/workflows/robot_arm64.yml"))


def _step(job: dict, name: str) -> dict:
    """The step of a parsed workflow job with this name."""
    for step in job["steps"]:
        if step.get("name") == name:
            return step
    raise KeyError(name)


def _script_value(name: str) -> str:
    """The value of `NAME="..."` in deploy_robot.sh."""
    return search(
        rf'^{name}="([^"]+)"$', read("tools/deploy/deploy_robot.sh"), re.M
    ).group(1)


class Arm64WorkflowTest(unittest.TestCase):
    """CI builds the robot side on arm64 the way a deploy does, so that an x86-64 laptop deploys it to an ARM robot."""

    def setUp(self) -> None:
        self.jobs = _arm64_workflow()["jobs"]

    def test_both_jobs_run_on_the_arm64_runner(self) -> None:
        self.assertEqual(sorted(self.jobs), ["bundle", "image"])
        for name, job in self.jobs.items():
            with self.subTest(job=name):
                self.assertEqual(job["runs-on"], "ubuntu-24.04-arm")
        self.assertEqual(self.jobs["image"]["needs"], "bundle")
        bazelisk = _step(self.jobs["bundle"], "Install Bazel (via Bazelisk)")[
            "run"
        ]
        self.assertIn("bazelisk-linux-arm64", bazelisk)

    def test_the_bundle_job_builds_pinocchio_from_source(self) -> None:
        bundle = self.jobs["bundle"]
        build = _step(bundle, "Build Pinocchio from Source")["run"]
        self.assertIn("sh docker/build_pinocchio_from_source.sh", build)
        self.assertIn("PINOCCHIO_STEPS=dependencies", build)
        self.assertNotIn("install_robotpkg.sh", yaml.safe_dump(bundle))
        restore = _step(bundle, "Restore Pinocchio")
        self.assertEqual(restore["with"]["path"], "/opt/openrobots")
        self.assertIn(
            "hashFiles('docker/build_pinocchio_from_source.sh')",
            restore["with"]["key"],
        )

    def test_the_bundle_is_the_deploys_bundle_in_the_deploys_context(
        self,
    ) -> None:
        bundle = self.jobs["bundle"]
        target = _script_value("BUNDLE_TARGET")
        self.assertIn(
            f"bazel build {target}",
            _step(bundle, "Build the Robot Bundle")["run"],
        )
        keep = _step(bundle, "Keep the Bundle for the Image")["run"]
        self.assertIn(
            f"cp -L {_script_value('BUNDLE_OUTPUT')} bundle/robot_bundle.tar",
            keep,
        )
        upload = _step(bundle, "Upload the Bundle")
        download = _step(self.jobs["image"], "Download the Bundle")
        self.assertEqual(download["with"]["name"], upload["with"]["name"])
        self.assertEqual(download["with"]["path"], _script_value("CONTEXT_DIR"))

    def test_the_image_is_built_by_the_deploy_script_and_published_on_push_only(
        self,
    ) -> None:
        image = self.jobs["image"]
        self.assertEqual(image["permissions"]["packages"], "write")
        build = _step(image, "Build the Robot Image")["run"]
        self.assertIn(
            "tools/deploy/deploy_robot.sh image --skip_bundle --platform"
            " linux/arm64",
            build,
        )
        for name in (
            "Log in to the GitHub Container Registry",
            "Publish the Robot Image",
        ):
            with self.subTest(step=name):
                self.assertEqual(
                    _step(image, name)["if"], "github.event_name == 'push'"
                )
        publish = _step(image, "Publish the Robot Image")["run"]
        self.assertIn(f"/{_script_value('RUNTIME_IMAGE')}\"", publish)
        names = _step(image, "Name the Image")["run"]
        self.assertIn("commit_tag=sha-${GITHUB_SHA::12}-arm64", names)
        self.assertIn("branch_tag=${branch//\\//-}-arm64", names)

    def test_the_readme_deploys_the_published_image(self) -> None:
        section = search(
            r"### aarch64\n(.*?)\n## ", read("tools/deploy/README.md"), re.S
        ).group(1)
        self.assertIn(
            "IMAGE_SOURCE=registry REGISTRY=ghcr.io/<owner> TAG=main-arm64",
            section,
        )
        self.assertIn(".github/workflows/robot_arm64.yml", section)


class PinocchioSourceBuildTest(unittest.TestCase):
    """docker/build_pinocchio_from_source.sh, against a recording apt-get and a git that must not run."""

    def run_script(self, steps: str) -> tuple[subprocess.CompletedProcess, str]:
        """Runs the script with PINOCCHIO_STEPS=steps; returns its result and the commands it ran, one per line."""
        with tempfile.TemporaryDirectory() as directory:
            log = os.path.join(directory, "commands.log")
            for tool, body in (
                ("apt-get", 'echo "apt-get $*" >> "$LOG"'),
                ("git", 'echo "git $*" >> "$LOG"; exit 1'),
            ):
                path = os.path.join(directory, tool)
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write(f"#!/bin/sh\n{body}\n")
                os.chmod(path, 0o755)
            env = dict(
                os.environ,
                PATH=f"{directory}:{os.environ['PATH']}",
                LOG=log,
                PINOCCHIO_STEPS=steps,
            )
            result = subprocess.run(
                ["sh", runfile("docker/build_pinocchio_from_source.sh")],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
                check=False,  # The tests check the return code.
            )
            commands = ""
            if os.path.exists(log):
                with open(log, encoding="utf-8") as stream:
                    commands = stream.read()
            return result, commands

    def test_dependencies_installs_the_packages_and_builds_nothing(
        self,
    ) -> None:
        result, commands = self.run_script("dependencies")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("libboost-serialization-dev", commands)
        self.assertIn("liburdfdom-dev", commands)
        self.assertEqual(
            [line for line in commands.splitlines() if line.startswith("git ")],
            [],
        )

    def test_an_unknown_step_is_refused_before_anything_runs(self) -> None:
        result, commands = self.run_script("pinocchio")
        self.assertEqual(result.returncode, 2)
        self.assertIn("PINOCCHIO_STEPS", result.stderr)
        self.assertEqual(commands, "")


if __name__ == "__main__":
    unittest.main()
