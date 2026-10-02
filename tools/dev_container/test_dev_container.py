"""Pins what the ROS-free dev container, its image and CI's copy of its setup provide.

The processes talk over ZeroMQ + Protocol Buffers and draw with Rerun (humanoid_nmpc/docs/distributed_runtime), so the
image is plain Ubuntu with Pinocchio from robotpkg and one VNC display for the MuJoCo viewer and the operator GUI. These
tests keep ROS from coming back through the image, CI or the container definition; keep the image, CI and the local CI
emulator on one base, one Pinocchio and one environment; and keep the ports the processes listen on published. They
assert properties (the copies agree, every node of the network file is published) rather than versions, so that moving
to a new release is an edit in one place and not a test failure.
"""

import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import tempfile
import types
import unittest
from typing import Dict, List, Tuple
from unittest import mock


def _runfile(relative_path: str) -> str:
    """A data file of this test, from the Bazel runfiles when run by Bazel and from the source tree otherwise."""
    roots = []
    if "TEST_SRCDIR" in os.environ:
        roots.append(os.path.join(os.environ["TEST_SRCDIR"], "_main"))
    roots.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    for root in roots:
        candidate = os.path.join(root, relative_path)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(relative_path)


def _read(relative_path: str) -> str:
    with open(_runfile(relative_path)) as f:
        return f.read()


def _without_comment_lines(text: str) -> str:
    """The text without its full-line `#` comments, which may name what a file deliberately does NOT do."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _installed_packages(text: str) -> List[str]:
    """The arguments of every `apt-get install` in a script, Dockerfile or workflow, without options or versions."""
    joined = _without_comment_lines(text).replace("\\\n", " ")
    packages = []
    for match in re.finditer(r"apt-get install([^\n]*)", joined):
        for word in match.group(1).split():
            if word in ("&&", "||", ";", "|"):
                break
            if not word.startswith("-") and not word.startswith("$"):
                packages.append(word.split("=")[0])
    return packages


def _dependency_list() -> List[str]:
    return [
        line.strip()
        for line in _without_comment_lines(_read("dependencies.txt")).splitlines()
        if line.strip()
    ]


# The files that define the dev container, its image and CI's copy of the image's setup.
_CONTAINER_FILES = (
    "dependencies.txt",
    "docker/Dockerfile",
    "docker/install_robotpkg.sh",
    "docker/image_build.bash",
    "docker/launch_wb_mpc.bash",
    "docker-compose.yaml",
    "docker-compose.bridge.yaml",
    "docker-compose.gpu.yaml",
    "setup_env.sh",
    ".devcontainer/devcontainer.json",
    ".devcontainer/post_create.sh",
    ".devcontainer/shell_init.sh",
    ".devcontainer/start_vnc.sh",
    ".devcontainer/openbox_layout.py",
    ".github/workflows/build_test.yml",
    "tools/ci_local.sh",
)

# What a ROS install leaves in such a file: its images and packages, its prefix, its environment and its tools.
_ROS_TRACES = re.compile(
    r"\bros:\w"
    r"|\bros-\$\{"
    r"|\bros-(?:jazzy|humble|iron|rolling)-"
    r"|/opt/ros\b"
    r"|\bROS_DISTRO\b"
    r"|\bROS_DOMAIN_ID\b"
    r"|\bRMW_IMPLEMENTATION\b"
    r"|\bros2 (?:launch|run)\b"
)

# The second VNC display that only PlotJuggler used, and the DDS discovery ports of ROS 2.
_REMOVED_PORTS = {5903, 6082, 7400, 7410, 7447}
_PLOTJUGGLER_DISPLAY_TRACES = re.compile(
    r"PLOTJUGGLER_DISPLAY|\bPJ_[A-Z_]+|Xvfb[^\n]*:100\b|DISPLAY=:100\b"
)


# What a contributor is asked for: the issue templates.
_ISSUE_TEMPLATES = (
    ".github/ISSUE_TEMPLATE/bug_report.md",
    ".github/ISSUE_TEMPLATE/feature_request.md",
)


class NoRosTest(unittest.TestCase):
    def test_no_issue_template_asks_for_ros(self):
        for path in _ISSUE_TEMPLATES:
            with self.subTest(path=path):
                self.assertNotRegex(
                    _read(path), r"\bROS\b|\bJazzy\b|\bHumble\b|\bcolcon\b"
                )

    def test_no_file_installs_or_configures_ros(self):
        for path in _CONTAINER_FILES:
            with self.subTest(path=path):
                found = _ROS_TRACES.findall(_read(path))
                self.assertEqual(found, [], f"{path} still refers to ROS")

    def test_nothing_starts_or_names_the_plotjuggler_display(self):
        for path in _CONTAINER_FILES:
            with self.subTest(path=path):
                self.assertIsNone(_PLOTJUGGLER_DISPLAY_TRACES.search(_read(path)))


class BaseImageTest(unittest.TestCase):
    def test_the_image_ci_and_the_emulator_share_one_ubuntu_base(self):
        dockerfile = _read("docker/Dockerfile")
        image = re.search(r"^ARG BASE_IMAGE=(\S+)$", dockerfile, re.MULTILINE)
        workflow = re.search(
            r"^    container: (\S+)$", _read(".github/workflows/build_test.yml"), re.M
        )
        emulator = re.search(r'^CI_IMAGE="(\S+)"$', _read("tools/ci_local.sh"), re.M)
        for match in (image, workflow, emulator):
            self.assertIsNotNone(match)
        self.assertRegex(image.group(1), r"^ubuntu:\d+\.\d+$")
        self.assertEqual(workflow.group(1), image.group(1))
        self.assertEqual(emulator.group(1), image.group(1))
        self.assertRegex(dockerfile, r"(?m)^FROM \$\{BASE_IMAGE\} AS base$")


class NoninteractiveAptTest(unittest.TestCase):
    """A plain Ubuntu image asks tzdata's debconf question, which python3-pip pulls in, and waits for an answer."""

    def test_the_image_ci_and_the_emulator_install_without_a_prompt(self):
        self.assertRegex(
            _read("docker/Dockerfile"), r"(?m)^ENV DEBIAN_FRONTEND=noninteractive"
        )
        self.assertRegex(
            _read(".github/workflows/build_test.yml"),
            r"(?m)^    env:\n(?:      \w+: \S+\n)*      DEBIAN_FRONTEND: noninteractive$",
        )
        self.assertRegex(
            _read("tools/ci_local.sh"),
            r"(?m)^CI_ENVIRONMENT=\(.*-e DEBIAN_FRONTEND=noninteractive.*\)$",
        )
        self.assertRegex(
            _read("tools/ci_local.sh"),
            r'docker run --rm [^\n]*"\$\{CI_ENVIRONMENT\[@\]\}"',
        )


class PinocchioTest(unittest.TestCase):
    def setUp(self):
        self.script = _read("docker/install_robotpkg.sh")
        self.dockerfile = _read("docker/Dockerfile")
        self.workflow = _read(".github/workflows/build_test.yml")
        self.emulator = _read("tools/ci_local.sh")

    def test_one_script_installs_it_for_the_image_ci_and_the_emulator(self):
        self.assertIn(
            "COPY docker/install_robotpkg.sh /tmp/install_robotpkg.sh", self.dockerfile
        )
        self.assertIn("RUN sh /tmp/install_robotpkg.sh", self.dockerfile)
        self.assertIn("sh docker/install_robotpkg.sh", self.workflow)
        self.assertIn("sh docker/install_robotpkg.sh", self.emulator)
        # And nothing installs a second Pinocchio beside it.
        for path in (
            "docker/Dockerfile",
            ".github/workflows/build_test.yml",
            "tools/ci_local.sh",
        ):
            with self.subTest(path=path):
                self.assertEqual(
                    [
                        p
                        for p in _installed_packages(_read(path))
                        if "pinocchio" in p or p.startswith("robotpkg")
                    ],
                    [],
                )
        self.assertEqual([p for p in _dependency_list() if "pinocchio" in p], [])

    def test_the_library_and_its_bindings_are_pinned_to_one_release(self):
        version = re.search(r'^PINOCCHIO_VERSION="(\d+\.\d+\.\d+)"$', self.script, re.M)
        self.assertIsNotNone(version, "PINOCCHIO_VERSION is not a full x.y.z release")
        packages = re.search(r'^PINOCCHIO_PACKAGES="([^"]+)"$', self.script, re.M)
        self.assertIsNotNone(packages)
        self.assertEqual(
            sorted(re.sub(r"py\d+", "pyXY", p) for p in packages.group(1).split()),
            [
                "robotpkg-pinocchio=${PINOCCHIO_VERSION}",
                "robotpkg-pyXY-pinocchio=${PINOCCHIO_VERSION}",
            ],
        )
        self.assertIn(
            "apt-get install -y --no-install-recommends ${PINOCCHIO_PACKAGES}",
            self.script,
        )

    def test_the_key_is_checked_and_trusted_for_the_robotpkg_source_only(self):
        fingerprint = re.search(
            r'^ROBOTPKG_KEY_FINGERPRINT="([0-9A-F]+)"$', self.script, re.M
        )
        self.assertIsNotNone(fingerprint)
        self.assertEqual(
            len(fingerprint.group(1)), 40, "not a full OpenPGP v4 fingerprint"
        )
        self.assertIn(
            'if [ "${fingerprint}" != "${ROBOTPKG_KEY_FINGERPRINT}" ]; then',
            self.script,
        )
        self.assertRegex(self.script, r'(?m)^KEYRING="/etc/apt/keyrings/[\w.-]+\.asc"$')
        self.assertIn("signed-by=${KEYRING}", self.script)
        # apt-key would trust the key for every source on the system.
        for text in (self.script, self.dockerfile):
            self.assertNotIn("apt-key", _without_comment_lines(text))

    def test_the_script_is_posix_sh_and_keeps_the_apt_lists_ci_needs(self):
        self.assertTrue(self.script.startswith("#!/bin/sh\n"))
        self.assertNotIn("[[", self.script)
        # CI installs dependencies.txt right after the script without another `apt-get update`.
        self.assertNotIn("/var/lib/apt/lists", _without_comment_lines(self.script))
        subprocess.run(["sh", "-n", _runfile("docker/install_robotpkg.sh")], check=True)

    def test_the_python_path_is_the_python_the_bindings_are_built_for(self):
        python = re.search(r"robotpkg-py(\d)(\d+)-pinocchio", self.script)
        self.assertIsNotNone(python)
        site_packages = (
            "/opt/openrobots/lib/python%s.%s/site-packages" % python.groups()
        )
        self.assertIn('PYTHONPATH="%s' % site_packages, self.dockerfile)
        self.assertIn('echo "PYTHONPATH=%s"' % site_packages, self.workflow)
        self.assertIn("export PYTHONPATH=%s" % site_packages, self.emulator)


def _dockerfile_robotpkg_environment() -> Dict[str, str]:
    """NAME -> the directory the image's robotpkg ENV puts first on it."""
    block = re.search(
        r"# LINT\.IfChange\(robotpkg_environment\)\n(.*?)# LINT\.ThenChange",
        _read("docker/Dockerfile"),
        re.S,
    )
    return dict(re.findall(r'(\w+)="([^":$]+)', block.group(1)))


class EnvironmentTest(unittest.TestCase):
    def test_the_image_ci_and_the_emulator_set_one_environment(self):
        image = _dockerfile_robotpkg_environment()
        self.assertEqual(
            sorted(image),
            ["CMAKE_PREFIX_PATH", "PATH", "PKG_CONFIG_PATH", "PYTHONPATH"],
        )
        workflow = _read(".github/workflows/build_test.yml")
        ci = dict(re.findall(r'echo "(\w+)=(\S+)" >> "\$GITHUB_ENV"', workflow))
        ci["PATH"] = re.search(r'echo "(\S+)" >> "\$GITHUB_PATH"', workflow).group(1)
        self.assertEqual(ci, image)
        emulator = dict(
            re.findall(r"^\s*export (\w+)=([^:\s]+)", _read("tools/ci_local.sh"), re.M)
        )
        self.assertEqual(emulator, image)

    def test_robotpkg_stays_off_the_library_path(self):
        # Nothing needs it, and robotpkg ships a libblasfeo.so of another version than the solver's: on
        # LD_LIBRARY_PATH it would outrank the RUNPATH of every binary started from that shell.
        for path in (
            "docker/Dockerfile",
            ".github/workflows/build_test.yml",
            "tools/ci_local.sh",
            "setup_env.sh",
        ):
            with self.subTest(path=path):
                self.assertNotRegex(
                    _without_comment_lines(_read(path)),
                    r"LD_LIBRARY_PATH\W+/opt/openrobots",
                )


class SystemPackagesTest(unittest.TestCase):
    def test_no_boost_package_is_installed_explicitly(self):
        # Boost reaches the system only as a dependency of Pinocchio and coal (docker/install_robotpkg.sh).
        installed = _dependency_list()
        for path in (
            "docker/Dockerfile",
            "docker/install_robotpkg.sh",
            ".github/workflows/build_test.yml",
        ):
            installed += _installed_packages(_read(path))
        self.assertIn("libeigen3-dev", installed, "the package lists were not read")
        self.assertEqual([p for p in installed if "boost" in p], [])

    def test_the_image_installs_every_program_start_vnc_calls(self):
        start_vnc = _read(".devcontainer/start_vnc.sh")
        installed = set(
            _installed_packages(_read("docker/Dockerfile")) + _dependency_list()
        )
        for program, package in (
            ("Xvfb", "xvfb"),
            ("x11vnc", "x11vnc"),
            ("websockify", "websockify"),
            ("openbox", "openbox"),
            ("xsetroot", "x11-xserver-utils"),
            ("fuser", "psmisc"),
            ("ip", "iproute2"),
        ):
            with self.subTest(program=program):
                self.assertRegex(start_vnc, r"\b%s\b" % program)
                self.assertIn(package, installed)
        self.assertIn("novnc", installed)

    def test_the_image_installs_the_dependency_list(self):
        self.assertIn(
            "grep -v '^\\s*#' /tmp/dependencies.txt | envsubst | xargs apt-get install",
            _read("docker/Dockerfile"),
        )


def _published_ports(compose: str) -> Dict[int, int]:
    """Container port -> host port of every `ports:` entry of a compose file, ranges expanded."""
    published = {}
    for host_first, host_last, first, last in re.findall(
        r'^\s+- "(\d+)(?:-(\d+))?:(\d+)(?:-(\d+))?(?:/(?:tcp|udp))?"$', compose, re.M
    ):
        host_ports = range(int(host_first), int(host_last or host_first) + 1)
        container_ports = range(int(first), int(last or first) + 1)
        published.update(zip(container_ports, host_ports))
    return published


def _devcontainer() -> dict:
    """devcontainer.json, which is JSON with full-line // comments."""
    text = _read(".devcontainer/devcontainer.json")
    return json.loads(
        "\n".join(
            line for line in text.splitlines() if not line.lstrip().startswith("//")
        )
    )


class HostNetworkTest(unittest.TestCase):
    """On Linux the dev container runs on the host's network, so the bus reaches other containers and machines at the
    network file's addresses; docker-compose.bridge.yaml is the Docker Desktop override that publishes the ports.
    """

    def test_the_dev_container_uses_the_host_network_and_publishes_nothing(self):
        compose = _without_comment_lines(_read("docker-compose.yaml"))
        self.assertRegex(compose, r"(?m)^\s+network_mode: host$")
        # Docker discards published ports on the host network and refuses a `dns` option there.
        self.assertNotRegex(compose, r"(?m)^\s+ports:")
        self.assertNotRegex(compose, r"(?m)^\s+dns:")
        self.assertEqual(_published_ports(compose), {})

    def test_the_bridge_override_gives_the_container_a_network_of_its_own(self):
        override = _without_comment_lines(_read("docker-compose.bridge.yaml"))
        self.assertRegex(override, r"(?m)^\s+network_mode: bridge$")
        self.assertRegex(override, r"(?m)^\s+dns:")


class PortsTest(unittest.TestCase):
    """What docker-compose.bridge.yaml publishes where the host network is not the computer's (Docker Desktop)."""

    def setUp(self):
        self.published = _published_ports(_read("docker-compose.bridge.yaml"))

    def test_every_node_of_the_network_file_is_published(self):
        ports = [
            int(p)
            for p in re.findall(
                r"\bport:\s*(\d+)", _read("config/ipc/network.textproto")
            )
        ]
        self.assertTrue(ports, "the network file names no node")
        self.assertEqual([p for p in ports if p not in self.published], [])

    def test_rerun_vnc_and_jupyter_are_published(self):
        # Rerun's gRPC server and web viewer, noVNC and VNC, Jupyter.
        for port in (9876, 9090, 6080, 5901, 8888):
            self.assertIn(port, self.published)

    def test_the_ports_of_the_plotjuggler_display_and_of_dds_are_gone(self):
        self.assertEqual(_REMOVED_PORTS & set(self.published), set())
        self.assertEqual(_REMOVED_PORTS & set(_devcontainer()["forwardPorts"]), set())

    def test_every_port_is_published_at_its_own_number(self):
        # The READMEs and the start-up banners name http://localhost:6080 and so on: a remapped port would break them.
        self.assertEqual({c: h for c, h in self.published.items() if c != h}, {})

    def test_the_ide_forwards_published_ports_and_labels_each(self):
        devcontainer = _devcontainer()
        forwarded = set(devcontainer["forwardPorts"])
        self.assertEqual(forwarded - set(self.published), set())
        self.assertEqual({int(p) for p in devcontainer["portsAttributes"]}, forwarded)


class DisplayTest(unittest.TestCase):
    def test_compose_the_ide_and_the_vnc_script_use_the_same_single_display(self):
        start_vnc = _read(".devcontainer/start_vnc.sh")
        display = re.search(
            r'^VNC_DISPLAY="\$\{VNC_DISPLAY:-(:\d+)\}"$', start_vnc, re.M
        )
        self.assertIsNotNone(display)
        self.assertEqual(len(re.findall(r"\bXvfb \$\{display\}", start_vnc)), 1)
        self.assertIn('DISPLAY: "%s"' % display.group(1), _read("docker-compose.yaml"))
        self.assertEqual(_devcontainer()["containerEnv"]["DISPLAY"], display.group(1))

    def test_a_stale_second_display_mode_is_refused_before_anything_starts(self):
        script = _runfile(".devcontainer/start_vnc.sh")
        for argument in ("plotjuggler", "main"):
            with self.subTest(argument=argument):
                result = subprocess.run(
                    ["bash", script, argument], capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("usage:", result.stderr)

    def test_the_shell_scripts_parse(self):
        for path in (
            ".devcontainer/start_vnc.sh",
            ".devcontainer/post_create.sh",
            ".devcontainer/shell_init.sh",
            "docker/image_build.bash",
            "docker/launch_wb_mpc.bash",
            "tools/ci_local.sh",
        ):
            with self.subTest(path=path):
                subprocess.run(["bash", "-n", _runfile(path)], check=True)


def _load_openbox_layout() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "openbox_layout", _runfile(".devcontainer/openbox_layout.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Rectangle = Tuple[int, int, int, int]


def _overlap(a: Rectangle, b: Rectangle) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


class OpenboxLayoutTest(unittest.TestCase):
    def setUp(self):
        self.layout = _load_openbox_layout()

    def _panes(self, width: int, height: int) -> Dict[str, Rectangle]:
        g = self.layout.layout(width, height)
        return {
            "viewer": (0, 0, g["left_w"], g["viewer_h"]),
            "gui": (0, g["gui_y"], g["left_w"], g["gui_h"]),
            "mujoco": (g["half_w"], 0, g["left_w"], g["pane_h"]),
        }

    def test_the_panes_lie_on_the_display_and_never_overlap(self):
        for width, height in (
            (1280, 720),
            (1920, 1080),
            (2560, 1440),
            (3840, 2160),
            (1280, 600),
        ):
            panes = self._panes(width, height)
            names = sorted(panes)
            for name in names:
                x, y, w, h = panes[name]
                with self.subTest(resolution=(width, height), pane=name):
                    self.assertGreater(w, 0)
                    self.assertGreater(h, 0)
                    self.assertLessEqual(x + w, width)
                    self.assertLessEqual(y + h, height)
            for i, first in enumerate(names):
                for second in names[i + 1 :]:
                    with self.subTest(
                        resolution=(width, height), panes=(first, second)
                    ):
                        self.assertFalse(_overlap(panes[first], panes[second]))

    def test_the_gui_keeps_its_minimum_height_and_the_viewer_gives_way(self):
        for width, height in ((1280, 600), (1920, 1080)):
            g = self.layout.layout(width, height)
            with self.subTest(resolution=(width, height)):
                self.assertGreaterEqual(g["gui_h"], self.layout.GUI_MIN_H)
                self.assertEqual(g["gui_y"] + g["gui_h"], g["pane_h"])

    def test_the_rules_place_the_three_windows_of_the_display(self):
        rules = self.layout.RULE_TEMPLATE.format(**self.layout.layout(1920, 1080))
        titles = re.findall(r'<application title="\*([^*"]+)\*"', rules)
        self.assertEqual(titles, ["Rerun", "Robot Base Controller", "Mujoco"])
        self.assertNotRegex(rules.lower(), r"rviz|plotjuggler")

    def test_a_malformed_resolution_falls_back_to_the_default(self):
        self.assertEqual(self.layout.parse_resolution("2560x1440"), (2560, 1440))
        default = (self.layout.DEFAULT_WIDTH, self.layout.DEFAULT_HEIGHT)
        for value in ("", "plotjuggler", "1920x1080x24", "x"):
            with self.subTest(value=value):
                self.assertEqual(self.layout.parse_resolution(value), default)

    def _run_main(self, rc_xml: str, resolution: str = "1920x1080") -> Tuple[int, str]:
        with tempfile.TemporaryDirectory() as home:
            rc_path = os.path.join(home, ".config", "openbox", "rc.xml")
            os.makedirs(os.path.dirname(rc_path))
            with open(rc_path, "w") as f:
                f.write(rc_xml)
            output = io.StringIO()
            with mock.patch.dict(os.environ, {"HOME": home, "RESOLUTION": resolution}):
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(
                    output
                ):
                    status = self.layout.main()
            with open(rc_path) as f:
                return status, f.read()

    def test_main_adds_the_rules_to_the_applications_section(self):
        status, written = self._run_main(
            "<openbox_config>\n<applications>\n</applications>\n</openbox_config>\n"
        )
        self.assertEqual(status, 0)
        self.assertEqual(written.count("</applications>"), 1)
        self.assertLess(
            written.index('title="*Rerun*"'), written.index("</applications>")
        )

    def test_main_leaves_a_file_without_an_applications_section_alone(self):
        original = "<openbox_config>\n</openbox_config>\n"
        status, written = self._run_main(original)
        self.assertEqual(status, 1)
        self.assertEqual(written, original)


if __name__ == "__main__":
    unittest.main()
