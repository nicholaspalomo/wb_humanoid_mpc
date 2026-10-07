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
    "WB_ROBOT_HEADLESS",
    "WB_ROBOT_REALTIME_PRIORITY",
    "WB_ROBOT_REALTIME_CORES",
    "WB_ROBOT_BACKEND_CORES",
    "WB_ROBOT_CONFIG_STORE_DIR",
    "WB_ROBOT_CONFIG_SEED",
)
# LINT.ThenChange(//docker-compose.robot.yaml:robot_environment)

# Where the robot images keep the robot's stored configuration (docker-compose.robot.yaml), one directory per robot.
CONFIG_STORE_TARGET = "/var/lib/wb-humanoid-robot/config"

# A `tc` that records its arguments, one call per line, and answers `qdisc show` with the file TC_SHOW names.
FAKE_TC = """#!/bin/sh
echo "$*" >> "$TC_LOG"
if [ "$1" = "qdisc" ] && [ "$2" = "show" ]; then cat "$TC_SHOW" 2>/dev/null; fi
exit 0
"""
OUR_QDISC = (
    "qdisc prio 5600: root refcnt 2 bands 4 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1\n"
)


class FakeTc:
    """A directory with the recording `tc`, to put in front of PATH."""

    def __init__(self, directory: str) -> None:
        self.bin = os.path.join(directory, "fake_bin")
        os.makedirs(self.bin)
        with open(os.path.join(self.bin, "tc"), "w", encoding="utf-8") as stream:
            stream.write(FAKE_TC)
        os.chmod(os.path.join(self.bin, "tc"), 0o755)
        self.log = os.path.join(directory, "tc.log")
        self.show = os.path.join(directory, "tc_show")

    def environment(self, installed: bool) -> dict[str, str]:
        with open(self.show, "w", encoding="utf-8") as stream:
            stream.write(OUR_QDISC if installed else "qdisc noqueue 0: root refcnt 2\n")
        return {
            "PATH": self.bin + os.pathsep + os.environ.get("PATH", "/usr/bin:/bin"),
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
        result = self.netem("apply", "lo", "delay", "3ms", "1ms", "loss", "0.5%")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [call for call in self.tc.calls() if "show" not in call],
            [
                "qdisc replace dev lo root handle 5600: prio bands 4 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1",
                "qdisc add dev lo parent 5600:4 handle 5601: netem delay 3ms 1ms loss 0.5%",
                "filter add dev lo parent 5600: protocol ip prio 1 u32 match ip protocol 6 0xff match ip sport 5600 0xffe0 "
                "flowid 5600:4",
                "filter add dev lo parent 5600: protocol ipv6 prio 2 u32 match ip6 protocol 6 0xff match ip6 sport 5600 0xffe0 "
                "flowid 5600:4",
                "filter add dev lo parent 5600: protocol ip prio 1 u32 match ip protocol 6 0xff match ip dport 5600 0xffe0 "
                "flowid 5600:4",
                "filter add dev lo parent 5600: protocol ipv6 prio 2 u32 match ip6 protocol 6 0xff match ip6 dport 5600 0xffe0 "
                "flowid 5600:4",
            ],
        )

    def test_every_filter_matches_tcp_only(self) -> None:
        self.netem("apply", "lo", "delay", "3ms")
        filters = [call for call in self.tc.calls() if call.startswith("filter add")]
        self.assertEqual(len(filters), 4, "both directions, IPv4 and IPv6")
        for call in filters:
            with self.subTest(call=call):
                self.assertRegex(
                    call, r"\bmatch ip6? protocol 6 0xff match ip6? [sd]port "
                )

    def test_apply_replaces_a_qdisc_a_previous_run_left(self) -> None:
        self.netem("apply", "lo", "delay", "3ms", installed=True)
        self.assertIn("qdisc del dev lo root", self.tc.calls())

    def test_clear_removes_its_own_qdisc_and_nothing_else(self) -> None:
        self.assertEqual(self.netem("clear", "eth0", installed=True).returncode, 0)
        self.assertIn("qdisc del dev eth0 root", self.tc.calls())
        os.remove(self.tc.log)
        self.assertEqual(self.netem("clear", "eth0", installed=False).returncode, 0)
        self.assertEqual([call for call in self.tc.calls() if "del" in call], [])

    def test_a_usage_error_changes_nothing(self) -> None:
        for arguments in (
            ("apply", "lo"),
            ("clear",),
            ("shape", "lo"),
            ("clear", "lo", "extra"),
        ):
            with self.subTest(arguments=arguments):
                self.assertEqual(self.netem(*arguments).returncode, 2)
        self.assertEqual([call for call in self.tc.calls() if "show" not in call], [])

    def test_the_port_range_holds_every_node_of_the_network_files(self) -> None:
        script = read("tools/deploy/netem.sh")
        base = int(search(r"^BUS_PORT=(\d+)$", script, re.M).group(1))
        mask = int(search(r"^BUS_PORT_MASK=(0x[0-9a-f]+)$", script, re.M).group(1), 16)
        for network in (
            "config/ipc/network.textproto",
            "config/ipc/two_machine.example.textproto",
        ):
            ports = [
                int(port) for port in re.findall(r"\bport:\s*(\d+)", read(network))
            ]
            with self.subTest(network=network):
                self.assertTrue(ports)
                self.assertEqual([port for port in ports if port & mask != base], [])


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
            'echo "pid=$$"\necho "headless=${WB_ROBOT_HEADLESS:-}"\nexit "${FAKE_STATUS:-0}"\n',
        )
        self.write_robot(
            "patient",
            "trap 'echo got-term; exit 0' TERM\necho ready\nwhile :; do sleep 0.05; done\n",
        )
        self.tc = FakeTc(directory)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def write_robot(self, name: str, body: str) -> None:
        path = os.path.join(self.bundle, "launch", f"{name}.sh")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write("#!/bin/sh\n" + body)
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
                result = self.run_entrypoint({} if robot is None else {"ROBOT": robot})
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

    def test_the_image_decides_the_viewer_unless_the_environment_does(self) -> None:
        cases: tuple[tuple[dict[str, str], str], ...] = (
            ({}, "headless="),
            ({"WB_ROBOT_DEFAULT_HEADLESS": "true"}, "headless=true"),
            (
                {"WB_ROBOT_DEFAULT_HEADLESS": "true", "WB_ROBOT_HEADLESS": ""},
                "headless=true",
            ),
            (
                {"WB_ROBOT_DEFAULT_HEADLESS": "true", "WB_ROBOT_HEADLESS": "false"},
                "headless=false",
            ),
        )
        for environment, expected in cases:
            with self.subTest(environment=environment):
                result = self.run_entrypoint(dict(environment, ROBOT="fake"))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected + "\n", result.stdout)

    def test_netem_is_applied_for_the_run_and_cleared_after(self) -> None:
        result = self.run_entrypoint(
            {"ROBOT": "fake", "NETEM": "delay 3ms 1ms loss 0.5%", "FAKE_STATUS": "3"}
        )
        self.assertEqual(result.returncode, 3, "the robot process's exit status")
        calls = self.tc.calls()
        self.assertIn(
            "qdisc add dev lo parent 5600:4 handle 5601: netem delay 3ms 1ms loss 0.5%",
            calls,
        )
        self.assertEqual(calls[-1], "qdisc show dev lo", "cleared last")

    def test_sigterm_reaches_the_robot_process_and_netem_is_cleared(self) -> None:
        env = self.tc.environment(installed=False)
        env.update(
            {"ROBOT": "patient", "NETEM": "delay 3ms", "NETEM_INTERFACE": "eth9"}
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
        self.assertEqual(sorted(self.robot["cap_add"]), ["IPC_LOCK", "SYS_NICE"])
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
            load_yaml("docker-compose.robot.yaml")["configs"]["network"]["file"],
        )

    def test_the_stored_configuration_is_a_host_directory_that_outlives_the_container(
        self,
    ) -> None:
        # LINT.IfChange(robot_config_store)
        (store,) = self.robot["volumes"]
        self.assertEqual(store["type"], "bind")
        self.assertEqual(store["target"], CONFIG_STORE_TARGET)
        self.assertNotIn("read_only", store)
        self.assertEqual(store["source"], "${WB_ROBOT_CONFIG_SOURCE:-./robot_config}")
        # A bind of the long syntax creates a missing host directory only when it says so.
        self.assertIs(store["bind"]["create_host_path"], True)
        # One directory per robot configuration below it.
        self.assertEqual(
            self.robot["environment"]["WB_ROBOT_CONFIG_STORE_DIR"],
            CONFIG_STORE_TARGET + "/${ROBOT}",
        )
        self.assertEqual(
            self.robot["environment"]["WB_ROBOT_CONFIG_SEED"],
            "${WB_ROBOT_CONFIG_SEED:-}",
        )
        # The deployment creates it, as the deploy user, next to the compose file.
        script = read("tools/deploy/deploy_robot.sh")
        self.assertIn('ROBOT_CONFIG_DIR="robot_config"', script)
        self.assertIn(
            "mkdir -p '${remote_dir}' '${remote_dir}/${ROBOT_CONFIG_DIR}'", script
        )
        # LINT.ThenChange(//docker-compose.robot.yaml:robot_config_store)

    def test_the_environment_is_the_launch_files_variables(self) -> None:
        launch = read(
            "robot_models/drc_atlas/drc_atlas_centroidal_mpc/launch/robot.textproto"
        )
        variables = {
            "WB_ROBOT_" + name.upper()
            for name in re.findall(r'variables \{ name: "(\w+)"', launch)
        }
        passed = {
            name for name in self.robot["environment"] if name.startswith("WB_ROBOT_")
        }
        self.assertEqual(passed, set(ROBOT_ENVIRONMENT))
        self.assertEqual(passed - variables, set())

    def test_the_sim_override_adds_the_viewer_and_the_checkouts_models_only(
        self,
    ) -> None:
        document = load_yaml("docker-compose.robot.sim.yaml")
        sim = document["services"]["robot"]
        self.assertEqual(set(sim), {"image", "restart", "environment", "volumes"})
        self.assertIn("wb-humanoid-robot-sim", sim["image"])
        self.assertEqual(sim["restart"], "no")
        self.assertIn("DISPLAY", sim["environment"])
        mounts = {volume["target"]: volume for volume in sim["volumes"]}
        self.assertEqual(
            set(mounts), {"/opt/wb-humanoid-robot/robot_models", CONFIG_STORE_TARGET}
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

    def test_net_admin_only_with_the_netem_override(self) -> None:
        netem = load_yaml("docker-compose.robot.netem.yaml")["services"]["robot"]
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
    def test_the_unit_starts_the_deployment_after_docker_and_at_boot(self) -> None:
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
            with open(os.path.join(directory, ".env"), "w", encoding="utf-8") as stream:
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
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        stages[match.group(2)] = text[match.start() : end]
    return stages


def installed_packages(stage: str) -> list[str]:
    """The packages the `apt-get install` commands of a Dockerfile stage install."""
    joined = without_comments(stage).replace("\\\n", " ")
    packages: list[str] = []
    for match in re.finditer(r"apt-get install([^\n]*?)(?:&&|<|$)", joined, re.M):
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
            self.stages["robot-sim"].startswith("FROM robot-runtime AS robot-sim")
        )
        self.assertTrue(
            self.stages["robot-libraries"].startswith(
                "FROM robot-pinocchio-${TARGETARCH} AS robot-libraries"
            )
        )

    def test_the_runtime_image_has_nothing_to_build_or_view_with(self) -> None:
        runtime = without_comments(self.stages["robot-runtime"])
        self.assertEqual(installed_packages(self.stages["robot-runtime"]), ["iproute2"])
        self.assertRegex(runtime, r"< /tmp/robot-packages.txt")
        self.assertNotRegex(
            runtime + without_comments(self.stages["robot-sim"]),
            r"build-essential|\bg\+\+|\bgcc\b|clang|cmake|bazel|novnc|x11vnc|xvfb|python3",
        )
        self.assertIn("COPY --from=robot-libraries", runtime)
        self.assertNotIn("--from=base", runtime)
        self.assertIn(
            'ENTRYPOINT ["/opt/wb-humanoid-robot/bin/robot_entrypoint.sh"]', runtime
        )
        self.assertIn("ENV WB_ROBOT_DEFAULT_HEADLESS=true", runtime)
        self.assertIn("collect_runtime_libraries.sh check", runtime)

    def test_the_sim_image_adds_the_viewers_gl_only(self) -> None:
        packages = installed_packages(self.stages["robot-sim"])
        self.assertTrue(packages)
        self.assertEqual(
            [p for p in packages if not re.match(r"^lib(gl|glx|egl|x)", p)], []
        )
        self.assertIn("ENV WB_ROBOT_DEFAULT_HEADLESS=false", self.stages["robot-sim"])

    def test_pinocchio_from_robotpkg_on_amd64_and_from_source_on_arm64(self) -> None:
        self.assertIn(
            "RUN sh /tmp/install_robotpkg.sh", self.stages["robot-pinocchio-amd64"]
        )
        self.assertIn(
            "build_pinocchio_from_source.sh", self.stages["robot-pinocchio-arm64"]
        )
        base = self.stages["base"]
        self.assertIn(
            'if [ "$(dpkg --print-architecture)" = "amd64" ]; then sh /tmp/install_robotpkg.sh',
            base,
        )
        self.assertIn("else sh /tmp/build_pinocchio_from_source.sh", base)

    # LINT.ThenChange(//docker/Dockerfile:robot_images)


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
                    ["bash", "-n", runfile(f"tools/deploy/{script}")], check=True
                )
        for script in ("robot_entrypoint.sh", "netem.sh"):
            subprocess.run(["sh", "-n", runfile(f"tools/deploy/{script}")], check=True)

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
        plain = self.deploy("environment", "--robot", "drc_atlas").stdout.splitlines()
        self.assertIn("COMPOSE_FILE=docker-compose.robot.yaml", plain)
        self.assertEqual([line for line in plain if "REALTIME" in line], [])
        # Without --config_seed the launch file's policy applies (when_bundle_changes).
        self.assertEqual([line for line in plain if "CONFIG_SEED" in line], [])

    def test_a_remote_deployment_ships_the_image_and_installs_the_unit(self) -> None:
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
            r"docker build .*--target robot-runtime -t wb-humanoid-robot:latest \.",
        )
        self.assertIn(
            "docker save wb-humanoid-robot:latest | gzip -1 | ssh robot.local", output
        )
        for file in (
            "docker-compose.robot.yaml",
            "docker-compose.robot.netem.yaml",
            "network.textproto",
            ".env",
        ):
            self.assertRegex(
                output, rf"scp -q \S+ robot\.local:wb-humanoid-robot/{re.escape(file)}"
            )
        self.assertIn("wb-humanoid-robot@drc_atlas.service", output)
        # The stored configuration's directory, which a deploy never empties.
        self.assertRegex(output, r"mkdir -p .*wb-humanoid-robot/robot_config")
        # A unit that is active already (oneshot, RemainAfterExit) would ignore `enable --now`: a redeploy restarts it,
        # so that the container comes up again on the image just shipped, and the units of other robots go off.
        self.assertNotIn("enable --now", output)
        self.assertRegex(
            output,
            r"systemctl enable '\S+' && sudo systemctl restart '\S+@drc_atlas\.service'",
        )
        self.assertIn("systemctl disable --now", output)
        # The image is built for the robot's architecture, which its Docker is asked for.
        self.assertIn("ssh robot.local docker version --format", output)

    def test_a_redeploy_restarts_a_running_deployment_on_the_new_image(self) -> None:
        result = self.deploy(
            "deploy", "--dry_run", "--skip_bundle", "--robot", "drc_atlas"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(
            result.stdout,
            r"docker compose ps --quiet robot .*then\s+docker compose up --detach --no-build",
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
            ("deploy", "--dry_run", "--robot", "drc_atlas", "--service", "always"),
            ("image", "--dry_run", "--target", "robot-dev"),
            ("deploy", "--dry_run", "--robot", "../x"),
            ("deploy", "--dry_run"),
            ("environment", "--robot", "drc_atlas", "--config_seed", "sometimes"),
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
CONFIGURATION_TARGETS = ("sim", "sim-vnc", "dummy-sim", "dummy-sim-vnc", "robot", "mpc")
SANDBOX_TARGETS = ("sandbox", "sandbox-vnc")
# LINT.ThenChange(//Makefile:launch_targets)


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
            self.targets.update(f"launch-{name}-{suffix}" for suffix in SANDBOX_TARGETS)

    def test_every_configuration_names_a_robot_package_and_its_formulation(
        self,
    ) -> None:
        for name, robot, package, formulation, description in self.configurations:
            with self.subTest(name=name):
                self.assertRegex(robot, r"^[a-z0-9_]+$")
                self.assertTrue(
                    package.startswith("robot_models/") and package.endswith("_mpc")
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
            words += self.makefile_variable(reference.group(1)) if reference else [word]
        return words

    # LINT.IfChange(launch_binaries)
    def test_every_launch_target_builds_every_binary_its_launch_file_starts(
        self,
    ) -> None:
        laptop = {label(t) for t in self.makefile_variable("laptop_binaries")}
        for name, _, package, formulation, description in self.configurations:
            for launch, built in (
                ("mpc", self.makefile_variable(f"mpc_binaries_{formulation}")),
                ("dummy_sim", self.makefile_variable(f"dummy_binaries_{formulation}")),
            ):
                started = launch_binaries(f"{package}/launch/{launch}.textproto")
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
    def test_the_deployment_script_knows_the_robots_of_the_makefile(self) -> None:
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

        network = os.path.join(RUNFILES, "config/ipc/two_machine.example.textproto")
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
        outside = make("deploy-robot", "ROBOT=drc_atlas", "NETWORK=/etc/hostname")
        self.assertNotEqual(outside.returncode, 0)
        self.assertIn("outside the checkout", outside.stderr)

    def test_no_ros_remains_and_no_launch_runs_the_mpc_in_the_robot_process(
        self,
    ) -> None:
        code = without_comments(self.makefile)
        self.assertNotRegex(
            code, r"ros2 |rviz|plotjuggler|\.bazel_ros_install|DISPLAY=:100|6082"
        )
        self.assertNotIn("in_process", code)
        self.assertRegex(code, r"\$\(session\) sim --robot")


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
            self.assertTrue(project.startswith("wb-humanoid-robot-sim-"), project)
            self.assertRegex(
                project, r"^[a-z0-9][a-z0-9_-]*$", "a compose project name"
            )
            self.assertNotEqual(
                project, "wb-humanoid-robot", "the deployment's project"
            )

    # LINT.ThenChange(//tools/deploy/session.sh:sim_project)


if __name__ == "__main__":
    unittest.main()
