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

"""Tests of tools/deploy/udp_buffers.sh, run against a fake `sysctl` and `sudo`.

The script raises a kernel limit only below the network file's request, keeps it in sysctl.d, and never fails a launch.
"""

import os
import subprocess
import tempfile
import unittest

RUNFILES = os.path.join(os.environ["TEST_SRCDIR"], "_main")
SCRIPT = os.path.join(RUNFILES, "tools/deploy/udp_buffers.sh")
SHIPPED_NETWORKS = (
    "config/ipc/network.textproto",
    "config/ipc/two_machine.example.textproto",
)

# A sysctl that reads and writes one file per key in $FAKE_SYSCTL_STATE, and logs every write.
FAKE_SYSCTL = """#!/usr/bin/env bash
set -eu
if [ "$1" = "-n" ]; then
    cat "${FAKE_SYSCTL_STATE}/$2"
elif [ "$1" = "-w" ]; then
    key="${2%%=*}"
    [ -w "${FAKE_SYSCTL_STATE}" ] || exit 1
    echo "${2#*=}" > "${FAKE_SYSCTL_STATE}/${key}"
    echo "$2" >> "${FAKE_SYSCTL_STATE}/writes.log"
fi
"""
# A sudo that runs its command, or refuses when $FAKE_SUDO_REFUSES is set.
FAKE_SUDO = """#!/usr/bin/env bash
[ -z "${FAKE_SUDO_REFUSES:-}" ] || exit 1
exec "$@"
"""


class UdpBuffersTest(unittest.TestCase):
    """The script's decisions, on a temporary machine."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.state = os.path.join(self.directory, "state")
        self.sysctl_dir = os.path.join(self.directory, "sysctl.d")
        os.makedirs(self.state)
        os.makedirs(self.sysctl_dir)
        self.sysctl = self.executable("sysctl", FAKE_SYSCTL)
        self.sudo = self.executable("sudo", FAKE_SUDO)

    def executable(self, name: str, text: str) -> str:
        path = os.path.join(self.directory, name)
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.chmod(path, 0o755)
        return path

    def network(self, text: str) -> str:
        path = os.path.join(self.directory, "network.textproto")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(text)
        return path

    def set_limits(self, rmem: int, wmem: int) -> None:
        for key, value in (("net.core.rmem_max", rmem), ("net.core.wmem_max", wmem)):
            with open(os.path.join(self.state, key), "w", encoding="utf-8") as stream:
                stream.write(f"{value}\n")

    def limit(self, key: str) -> int:
        with open(os.path.join(self.state, key), encoding="utf-8") as stream:
            return int(stream.read())

    def writes(self) -> list[str]:
        path = os.path.join(self.state, "writes.log")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as stream:
            return stream.read().splitlines()

    def run_script(
        self, network: str, in_container: bool = False, sudo_refuses: bool = False
    ) -> subprocess.CompletedProcess[str]:
        """Runs the script on `network`, here or as if in a container, with a sudo that runs or refuses."""
        dockerenv = os.path.join(self.directory, "dockerenv")
        if in_container:
            with open(dockerenv, "w", encoding="utf-8"):
                pass
        environment = dict(
            os.environ,
            FAKE_SYSCTL_STATE=self.state,
            WB_SYSCTL=self.sysctl,
            WB_SUDO=self.sudo,
            WB_SYSCTL_DIR=self.sysctl_dir,
            WB_DOCKERENV=dockerenv,
        )
        if sudo_refuses:
            environment["FAKE_SUDO_REFUSES"] = "1"
        return subprocess.run(
            [SCRIPT, "--network", network],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_a_limit_below_the_request_is_raised_and_kept_across_reboots(self) -> None:
        self.set_limits(rmem=212992, wmem=212992)
        result = self.run_script(
            self.network(
                "transport { socket_receive_buffer_bytes: 8388608 socket_send_buffer_bytes: 4194304 }\n"
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.limit("net.core.rmem_max"), 8388608)
        self.assertEqual(self.limit("net.core.wmem_max"), 4194304)
        for key, value in (
            ("net.core.rmem_max", 8388608),
            ("net.core.wmem_max", 4194304),
        ):
            path = os.path.join(self.sysctl_dir, f"60-wb-humanoid-{key}.conf")
            with open(path, encoding="utf-8") as stream:
                self.assertIn(f"{key} = {value}", stream.read().splitlines())

    def test_a_limit_at_or_above_the_request_is_never_written_or_lowered(self) -> None:
        self.set_limits(rmem=8388608, wmem=33554432)
        result = self.run_script(
            self.network(
                "transport { socket_receive_buffer_bytes: 8388608 socket_send_buffer_bytes: 4194304 }\n"
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.writes(), [])
        self.assertEqual(self.limit("net.core.wmem_max"), 33554432)
        self.assertEqual(os.listdir(self.sysctl_dir), [])

    def test_an_absent_or_zero_request_or_a_commented_one_changes_nothing(self) -> None:
        self.set_limits(rmem=1, wmem=1)
        for text in (
            "domain_id: 0\n",
            "transport { socket_receive_buffer_bytes: 0 }\n",
            "# transport { socket_receive_buffer_bytes: 8388608 }\ndomain_id: 0\n",
        ):
            with self.subTest(text=text):
                result = self.run_script(self.network(text))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.writes(), [])

    def test_inside_a_container_it_says_what_to_run_on_the_host_and_changes_nothing(
        self,
    ) -> None:
        self.set_limits(rmem=212992, wmem=212992)
        result = self.run_script(
            self.network("transport { socket_receive_buffer_bytes: 8388608 }\n"),
            in_container=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.writes(), [])
        self.assertIn("sudo sysctl -w net.core.rmem_max=8388608", result.stderr)

    def test_a_refused_sudo_leaves_the_limit_and_does_not_fail_the_launch(self) -> None:
        self.set_limits(rmem=212992, wmem=212992)
        result = self.run_script(
            self.network("transport { socket_receive_buffer_bytes: 8388608 }\n"),
            sudo_refuses=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.limit("net.core.rmem_max"), 212992)
        self.assertIn("could not raise net.core.rmem_max", result.stderr)

    def test_a_missing_network_file_is_a_usage_error(self) -> None:
        result = self.run_script(os.path.join(self.directory, "missing.textproto"))
        self.assertEqual(result.returncode, 2)

    def test_the_shipped_network_files_ask_for_what_the_script_raises(self) -> None:
        for network in SHIPPED_NETWORKS:
            with self.subTest(network=network):
                self.set_limits(rmem=212992, wmem=212992)
                result = self.run_script(os.path.join(RUNFILES, network))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertGreaterEqual(self.limit("net.core.rmem_max"), 8388608)


if __name__ == "__main__":
    unittest.main()
