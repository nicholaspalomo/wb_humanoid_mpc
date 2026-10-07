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

"""What the remote_control tests share: the repository's files, a bus that records, and Tk without a display.

RecordingPublisher is the real operator_bus.TopicPublisher of an operator topic over a RecordingBus, so a tab given one
goes through the same type check as in the GUI, and the test asserts on the protobuf message that would have gone out.

The Tk tests need a display. `bazel test` hands the tests the caller's DISPLAY (env_inherit in BUILD.bazel): in the dev
container that is the VNC session's Xvfb. Where Tk cannot open a window on it, or there is none (CI), the tests start an
Xvfb of their own. Only where that is not installed either are they skipped, saying why; on CI (the CI variable set)
they fail instead, since a skipped test would pass there without having run.
"""

import atexit
from collections.abc import Callable
import functools
import os
import select
import shutil
import subprocess
import threading
import time
import tkinter as tk
from typing import Any
import unittest

from google.protobuf import message as protobuf_message

from remote_control import operator_bus
import robot_ipc

# The root of the files the tests read: the runfiles tree under Bazel, the checkout otherwise.
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)


def repo_path(*parts: str) -> str:
    """A path under REPO_ROOT."""
    return os.path.join(REPO_ROOT, *parts)


# The DRC Atlas configuration the tab tests load (and copy before they save).
ATLAS_CONFIG = repo_path(
    "robot_models", "drc_atlas", "drc_atlas_centroidal_mpc", "config"
)
ATLAS_TASK_FILE = os.path.join(ATLAS_CONFIG, "mpc", "task.textproto")
ATLAS_REFERENCE_FILE = os.path.join(ATLAS_CONFIG, "command", "reference.textproto")
ATLAS_PD_GAINS_FILE = os.path.join(
    ATLAS_CONFIG, "controller", "joint_pd_gains.textproto"
)

# The config/ directory of every robot package, by package (BUILD.bazel's _ROBOT_CONFIGS are their data).
ROBOT_CONFIGS = {
    package: repo_path("robot_models", robot, package, "config")
    for robot, package in (
        ("drc_atlas", "drc_atlas_centroidal_mpc"),
        ("engineai_sa01", "engineai_sa01_centroidal_mpc"),
        ("unitree_g1", "g1_centroidal_mpc"),
        ("unitree_g1", "g1_wb_mpc"),
        ("unitree_r1", "unitree_r1_centroidal_mpc"),
    )
}
# The configuration files of a robot's config/ directory.
CONFIG_TEXTPROTOS = (
    os.path.join("mpc", "task.textproto"),
    os.path.join("mpc", "contact_planning.textproto"),
    os.path.join("command", "reference.textproto"),
    os.path.join("controller", "joint_pd_gains.textproto"),
)


def copy_config(config_dir: str, destination: str) -> str:
    """Copies the configuration textprotos of a robot's config/ directory to `destination`/config, for a test that saves.

    Args:
        config_dir: The robot's config/ directory.
        destination: A directory of the test's own.

    Returns:
        The copy's config/ directory; a file the robot does not have (contact_planning) is not copied.
    """
    copied = os.path.join(destination, "config")
    for relative in CONFIG_TEXTPROTOS:
        source = os.path.join(config_dir, relative)
        if not os.path.exists(source):
            continue
        target = os.path.join(copied, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)
    return copied


class RecordingBus:
    """Stands in for robot_ipc.Bus: records what is published and subscribed, and delivers what a test hands it.

    Thread-safe like the bus, so that a test may publish from a helper thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._published: list[tuple[str, protobuf_message.Message]] = []
        self.subscriptions: dict[
            str,
            tuple[
                type[protobuf_message.Message],
                Callable[[Any], None],
                robot_ipc.Delivery,
            ],
        ] = {}
        self.started = False
        self.closed = False

    def publish(self, topic: str, message: protobuf_message.Message) -> bool:
        copy = type(message)()
        copy.CopyFrom(message)
        with self._lock:
            self._published.append((topic, copy))
        return True

    def subscribe(
        self,
        topic: str,
        message_class: type[protobuf_message.Message],
        callback: Callable[[Any], None],
        delivery: robot_ipc.Delivery = robot_ipc.Delivery.LATEST,
    ) -> None:
        if topic in self.subscriptions:
            raise ValueError(f"the topic '{topic}' already has a subscription")
        self.subscriptions[topic] = (
            message_class,
            callback,
            robot_ipc.parse_delivery(delivery),
        )

    def deliver(self, topic: str, message: protobuf_message.Message) -> None:
        """Hands `message` to the subscription of `topic`, as the receive thread would."""
        message_class, callback, _ = self.subscriptions[topic]
        if not isinstance(message, message_class):
            raise TypeError(f"'{topic}' is subscribed with {message_class.__name__}")
        callback(message)

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    @property
    def published(self) -> list[tuple[str, protobuf_message.Message]]:
        with self._lock:
            return list(self._published)

    def messages_on(self, topic: str) -> list[protobuf_message.Message]:
        return [
            message
            for published_topic, message in self.published
            if published_topic == topic
        ]


class RecordingPublisher(operator_bus.TopicPublisher):
    """The GUI's publisher of one operator topic, over a bus that records instead of sending."""

    def __init__(self, topic: str, bus: RecordingBus | None = None) -> None:
        row = operator_bus.operator_topic(topic)
        super().__init__(
            bus if bus is not None else RecordingBus(), row.topic, row.message_class
        )

    @property
    def messages(self) -> list[protobuf_message.Message]:
        return self.bus.messages_on(self.topic)

    @property
    def publish_count(self) -> int:
        return len(self.messages)

    @property
    def last_message(self) -> protobuf_message.Message | None:
        messages = self.messages
        return messages[-1] if messages else None


# How long an Xvfb of the tests' own may take to report its display [s].
_XVFB_START_TIMEOUT = 10.0


def _tk_opens_a_window() -> bool:
    try:
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    # pylint: disable-next=broad-exception-caught  # Any failure means no usable display.
    except Exception:
        return False


def _read_display_number(read_fd: int, timeout: float) -> int | None:
    """The display number Xvfb writes to its -displayfd, or None when it exits or is silent for `timeout` seconds."""
    deadline = time.monotonic() + timeout
    text = b""
    while not text.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([read_fd], [], [], remaining)[0]:
            return None
        chunk = os.read(read_fd, 64)
        if not chunk:
            return None
        text += chunk
    try:
        return int(text.strip())
    except ValueError:
        return None


def _stop_private_display(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _start_private_display() -> bool:
    """Starts an Xvfb for this test process and points DISPLAY at it; False when there is no Xvfb, or it fails.

    Xvfb picks a free display number itself (-displayfd), so test processes that start one at the same time never
    share one. It listens on no TCP port and stops when the process exits.

    Returns:
        True once DISPLAY names the new Xvfb; False, with DISPLAY untouched, when none could be started.
    """
    binary = shutil.which("Xvfb")
    if binary is None:
        return False
    read_fd, write_fd = os.pipe()
    try:
        # pylint: disable-next=consider-using-with  # The Xvfb outlives this call; atexit stops it.
        process = subprocess.Popen(
            [binary, "-displayfd", str(write_fd), "-nolisten", "tcp"]
            + ["-screen", "0", "1280x1024x24"],
            pass_fds=(write_fd,),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        os.close(read_fd)
        os.close(write_fd)
        return False
    os.close(write_fd)
    try:
        number = _read_display_number(read_fd, _XVFB_START_TIMEOUT)
    finally:
        os.close(read_fd)
    if number is None:
        _stop_private_display(process)
        return False
    atexit.register(_stop_private_display, process)
    os.environ["DISPLAY"] = f":{number}"
    return True


@functools.lru_cache(maxsize=None)
def display_available() -> bool:
    """Whether Tk can open a window here: on the caller's display, or else on an Xvfb started for this process."""
    if _tk_opens_a_window():
        return True
    return _start_private_display() and _tk_opens_a_window()


def requires_display(test_item: Any) -> Any:
    """Skips a test class or method when Tk cannot open a window here, not even on an Xvfb of its own.

    Args:
        test_item: The decorated test class or test method.

    Returns:
        `test_item` itself where Tk can open a window, and otherwise `test_item` wrapped by unittest.skip.

    Raises:
        RuntimeError: there is no display on CI (the CI variable set), where the skip would pass the test unrun.
    """
    if display_available():
        return test_item
    reason = "no usable Tk display, and no Xvfb to start one (the xvfb package)"
    # The variables the Tk tests read (DISPLAY in Tk itself), which BUILD.bazel must pass through.
    # LINT.IfChange(tk_environment)
    on_ci = bool(os.environ.get("CI"))
    # LINT.ThenChange(//humanoid_nmpc/remote_control/BUILD.bazel:tk_environment, //humanoid_nmpc/humanoid_rerun_viewer/BUILD.bazel:tk_environment)
    if on_ci:
        raise RuntimeError(
            f"{reason}; on CI the Tk tests must run, not be skipped: install xvfb (dependencies.txt)"
        )
    return unittest.skip(reason)(test_item)
