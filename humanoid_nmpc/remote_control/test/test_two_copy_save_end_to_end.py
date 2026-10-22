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

"""The GUI's two-copy Save against the robot process: the laptop's file and the robot's stored copy are the same bytes.

The robot side of the two-process simulation runs as its own process: humanoid_centroidal_mpc_robot on the DRC Atlas
files (the MuJoCo backend without a viewer), with a configuration store in the test's scratch space whose seeds are the
bundled files, as docker-compose.robot.sim.yaml runs it. The laptop side is the GUI's own code: the MPC Parameters,
Joint PD Gains and Command Limits tabs on a copy of the Atlas files below a robot_models/ directory of the test's
(so that the copy's configuration identity is the robot's), saving through one robot_config_save.RobotConfigSaver
on an operator_bus.OperatorBus. Both sides share a network file of this test's own DDS domain
(robot_ipc.isolated_test_network()).

What it holds: each tab's Save leaves exactly the laptop's bytes in the robot's store; a save the robot refuses (another
configuration, another robot, a text that does not parse, another schema) leaves the stored copy as it was; and the
robot started again runs the saved task file (its telemetry rate, a start-up field, changes from the bundle's to the
saved one, a fifth of it). The edits are made on what the shipped files hold, so that retuning them changes nothing
here.
"""

from collections.abc import Callable
import os
import shutil
import tempfile
import threading
import time
import tkinter as tk
from typing import Any
import unittest

from google.protobuf import message as protobuf_message
from humanoid_mpc_config import task_file_pb2
from humanoid_mpc_msgs import config_file_save_pb2
from humanoid_mpc_msgs import robot_state_sample_pb2

from humanoid_mpc_ipc import topics
import nproto_textproto
import operator_test_support
from remote_control import config_schema
from remote_control import operator_bus
from remote_control import robot_config_save
from remote_control import tuned_file
from remote_control.tk_app import command_limits_tab
from remote_control.tk_app import joint_pd_tab
from remote_control.tk_app import mpc_params_tab
import robot_ipc

ROBOT_BINARY = os.environ["ROBOT_BINARY"]
ATLAS_PACKAGE = operator_test_support.ATLAS_PACKAGE
ATLAS_TASK = os.path.join(ATLAS_PACKAGE, "config", "mpc", "task.textproto")
ATLAS_REFERENCE = os.path.join(
    ATLAS_PACKAGE, "config", "command", "reference.textproto"
)
G1_WB_PACKAGE = os.path.join("robot_models", "unitree_g1", "g1_wb_mpc")
# The robot's store layout (RobotConfigDirectory), whatever the bundled files are called.
STORED_TASK = os.path.join("mpc", "task.textproto")
STORED_REFERENCE = os.path.join("command", "reference.textproto")
STORED_GAINS = os.path.join("controller", "joint_pd_gains.textproto")
# How long the robot may take to start, and how long a save may wait for its answer [s].
START_TIMEOUT_S = 60.0
ANSWER_TIMEOUT_S = 20.0
# The saver's timeout in this test [s]: a save whose first message the bus loses (the robot's reader still being
# discovered) is saved again after it, as the operator would.
SAVER_TIMEOUT_S = 1.0
_Status = robot_config_save.SaveState


def bundled_telemetry_rate() -> float:
    """The telemetry rate of the shipped Atlas task file [Hz], which a save changes (a start-up field)."""
    task = nproto_textproto.load_textproto(ATLAS_TASK, task_file_pb2.TaskFile)
    if not task.HasField("telemetry_frequency"):
        raise AssertionError(f"{ATLAS_TASK} sets no telemetry rate to change")
    return float(task.telemetry_frequency)


def stored_telemetry_rate(path: str) -> float:
    """The telemetry rate of the task file at `path` [Hz]."""
    return float(
        nproto_textproto.load_textproto(
            path, task_file_pb2.TaskFile
        ).telemetry_frequency
    )


def write_loopback_network_file(path: str, nodes: tuple[str, ...]) -> None:
    """Writes a network file of `nodes` on this test's own DDS domain, with the QoS of the topics the test uses.

    The save and its status are one-off events: their writers exist from start-up (their publishers are listed), as in
    the shipped network file, so that the first save does not race DDS discovery.

    Args:
        path: The file to write.
        nodes: The nodes of the network.
    """
    isolated = robot_ipc.isolated_test_network(list(nodes))
    events = robot_ipc.QosProfile.RELIABLE_EVENTS
    network = robot_ipc.NetworkConfig(
        nodes=isolated.nodes,
        domain_id=isolated.domain_id,
        domain_tag=isolated.domain_tag,
        topics=(
            robot_ipc.TopicQos(topics.OPERATOR_CONFIG_SAVE, events, ("operator",)),
            robot_ipc.TopicQos(topics.ROBOT_CONFIG_SAVE_STATUS, events, ("robot",)),
            robot_ipc.TopicQos(
                topics.ROBOT_STATE, robot_ipc.QosProfile.LATEST, ("robot",)
            ),
        ),
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(robot_ipc.format_network_config(network))


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()


def first_positive_number(
    specs: list[config_schema.ParameterSpec],
) -> config_schema.ParameterSpec:
    """The first of a file's rendered parameters whose value is a positive float."""
    for spec in specs:
        if isinstance(spec.value, float) and spec.value > 0.0:
            return spec
    raise AssertionError("the file has no parameter with a positive value")


def copy_robot_config(package: str, laptop: str) -> str:
    """Copies a robot package's configuration files to the same path below `laptop`; returns the copy's config/."""
    return operator_test_support.copy_config(
        os.path.join(package, "config"), os.path.join(laptop, package)
    )


# A task file that names its schema and does not parse: the parser's refusal, not the missing header's.
_UNPARSABLE_TASK = (
    "# proto-message: humanoid_mpc_config.TaskFile\ntelemetry_frequency: {\n"
)


class _SkewedPublisher:
    """Publishes the saves of another build: the publisher of operator/config_save, with another schema fingerprint."""

    def __init__(self, publisher: robot_config_save.SavePublisher) -> None:
        self._publisher = publisher

    def publish(self, message: protobuf_message.Message) -> bool:
        skewed = config_file_save_pb2.ConfigFileSave()
        skewed.CopyFrom(message)
        skewed.schema_fingerprint = "a schema of another build"
        return self._publisher.publish(skewed)


@operator_test_support.requires_display
class TwoCopySaveEndToEndTest(unittest.TestCase):
    """Each test starts the robot process on a fresh store and the GUI's bus, and ends both."""

    def setUp(self):
        self.scratch = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
        self.addCleanup(shutil.rmtree, self.scratch, ignore_errors=True)
        self.logs = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", self.scratch)
        self.network_file = os.path.join(self.scratch, "network.textproto")
        write_loopback_network_file(self.network_file, ("robot", "operator"))
        self.store = os.path.join(self.scratch, "robot_config", "drc_atlas")
        self.laptop = os.path.join(self.scratch, "laptop")
        self.config = copy_robot_config(ATLAS_PACKAGE, self.laptop)
        self.root = tk.Tk()
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.lock = threading.Lock()
        self.sample_times: list[float] = []
        self.operator = operator_bus.OperatorBus.connect(self.network_file)
        self.operator.bus.subscribe(
            topics.ROBOT_STATE,
            robot_state_sample_pb2.RobotStateSample,
            self._on_sample,
            delivery=robot_ipc.Delivery.ALL,
        )
        self.operator.start()
        self.addCleanup(self.operator.close)
        self.saver = self._saver(self.operator.config_save)
        self.robot: operator_test_support.RobotProcess | None = None
        self.addCleanup(self._stop_robot)

    def _saver(
        self, publisher: robot_config_save.SavePublisher
    ) -> robot_config_save.RobotConfigSaver:
        return robot_config_save.RobotConfigSaver(
            publisher, self.operator.config_save_statuses, timeout=SAVER_TIMEOUT_S
        )

    def _on_sample(self, sample: robot_state_sample_pb2.RobotStateSample) -> None:
        with self.lock:
            self.sample_times.append(sample.time)

    def _start_robot(self) -> None:
        """Starts the robot process on the store and waits until its FSM state reaches the GUI's bus."""
        self.robot = operator_test_support.RobotProcess(
            ROBOT_BINARY,
            self.network_file,
            self.store,
            os.path.join(self.logs, "robot.log"),
        )

        def reported() -> bool:
            return bool(self.operator.take_fsm_states()) or not self._robot_running()

        self.assertTrue(
            self._pump_until(reported, START_TIMEOUT_S),
            "the robot process did not report its FSM state",
        )
        self.assertTrue(self._robot_running(), "the robot process exited on start-up")

    def _robot_running(self) -> bool:
        return self.robot is not None and self.robot.running()

    def _stop_robot(self) -> int | None:
        code = self.robot.stop() if self.robot is not None else None
        self.robot = None
        return code

    def _pump_until(self, condition: Callable[[], bool], timeout: float) -> bool:
        """Runs Tk's event loop (the tabs' polls of the saver) until `condition` holds; whether it did in `timeout`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.root.update()
            if condition():
                return True
            time.sleep(0.01)
        return condition()

    def _telemetry_rate(self, window: float) -> float:
        """[Hz] The rate of robot/state over `window` seconds, by the samples' robot time; 0 with fewer than two."""
        with self.lock:
            self.sample_times.clear()
        self._pump_until(lambda: False, window)
        with self.lock:
            times = list(self.sample_times)
        if len(times) < 2 or not times[-1] > times[0]:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def _save(self, tab: Any) -> robot_config_save.SaveOutcome:
        """Clicks the tab's Save until the robot answers; the answered outcome."""
        deadline = time.monotonic() + ANSWER_TIMEOUT_S
        outcome: robot_config_save.SaveOutcome | None = None
        while time.monotonic() < deadline:
            self.assertTrue(tab.save(), "the laptop's copy was not saved")
            self._pump_until(
                lambda: not tab.robot_save.outcome().waiting, SAVER_TIMEOUT_S + 1.0
            )
            outcome = tab.robot_save.outcome()
            assert outcome is not None
            if not outcome.waiting:
                return outcome
        self.fail(f"the robot did not answer the save: {outcome}")

    def _send(
        self, saver: robot_config_save.RobotConfigSaver, text: str, robot_name: str
    ) -> robot_config_save.SaveOutcome:
        """Sends `text` as the laptop's task file until the robot answers; the answered outcome."""
        task = os.path.join(self.config, "mpc", "task.textproto")
        deadline = time.monotonic() + ANSWER_TIMEOUT_S
        while time.monotonic() < deadline:
            sequence = saver.send(robot_config_save.KIND_TASK, task, text, robot_name)

            def answered(sequence: int = sequence) -> bool:
                saver.poll(saver.clock())
                outcome = saver.outcome(sequence)
                return outcome is not None and not outcome.waiting

            if self._pump_until(answered, SAVER_TIMEOUT_S + 1.0):
                outcome = saver.outcome(sequence)
                assert outcome is not None
                return outcome
        self.fail("the robot did not answer the save")

    def _stored(self, relative: str) -> str:
        return os.path.join(self.store, relative)

    def test_each_tab_leaves_the_laptops_bytes_on_the_robot_and_a_restart_runs_them(
        self,
    ):
        bundled_rate = bundled_telemetry_rate()
        saved_rate = bundled_rate / 5.0
        self._start_robot()
        self.assertEqual(read_bytes(self._stored(STORED_TASK)), read_bytes(ATLAS_TASK))
        self.assertAlmostEqual(
            self._telemetry_rate(2.0), bundled_rate, delta=0.2 * bundled_rate
        )

        mpc = mpc_params_tab.MpcParamsTab(
            self.root,
            task_file=os.path.join(self.config, "mpc", "task.textproto"),
            enable_online_tuning=True,
            robot_saver=self.saver,
        )
        assert mpc.task is not None
        mpc._on_any_change("telemetry_frequency", saved_rate)
        gains = joint_pd_tab.JointPdGainsTab(
            self.root,
            pd_gains_file=os.path.join(
                self.config, "controller", "joint_pd_gains.textproto"
            ),
            enable_online_tuning=True,
            robot_saver=self.saver,
        )
        assert gains.gains is not None
        gain = first_positive_number(gains.gains.rendered())
        gains._on_row_change(gain.path, gain.label, float(gain.value) + 1.0)
        limits = command_limits_tab.CommandLimitsTab(
            self.root,
            reference_file=os.path.join(self.config, "command", "reference.textproto"),
            robot_saver=self.saver,
        )
        assert limits.reference is not None
        limit = first_positive_number(limits.reference.rendered())
        limits._on_change(limit.path, limit.label, float(limit.value) * 0.9)

        for tab, laptop, stored, bundled in (
            (mpc, mpc.task.path, STORED_TASK, ATLAS_TASK),
            (
                gains,
                gains.gains.path,
                STORED_GAINS,
                os.path.join(ATLAS_PACKAGE, "config", STORED_GAINS),
            ),
            (limits, limits.reference.path, STORED_REFERENCE, ATLAS_REFERENCE),
        ):
            with self.subTest(stored=stored):
                outcome = self._save(tab)
                self.assertIn(
                    outcome.state, (_Status.SAVED, _Status.UNCHANGED), outcome
                )
                self.assertEqual(outcome.stored_path, self._stored(stored))
                self.assertEqual(read_bytes(self._stored(stored)), read_bytes(laptop))
                self.assertNotEqual(read_bytes(laptop), read_bytes(bundled))
        self.assertAlmostEqual(
            stored_telemetry_rate(self._stored(STORED_TASK)), saved_rate
        )
        # The telemetry rate is a start-up field: the running robot keeps the bundle's rate.
        self.assertAlmostEqual(
            self._telemetry_rate(1.0), bundled_rate, delta=0.2 * bundled_rate
        )

        # Started again, the robot runs the stored copies: the laptop's bytes, so its saved telemetry rate.
        self.assertEqual(self._stop_robot(), 0, "the robot process did not end cleanly")
        self._start_robot()
        self.assertAlmostEqual(
            self._telemetry_rate(2.0), saved_rate, delta=0.25 * saved_rate
        )
        for laptop, stored in (
            (mpc.task.path, STORED_TASK),
            (gains.gains.path, STORED_GAINS),
            (limits.reference.path, STORED_REFERENCE),
        ):
            self.assertEqual(
                read_bytes(self._stored(stored)), read_bytes(laptop), stored
            )

    def test_a_save_the_robot_refuses_leaves_its_copy_untouched(self):
        self._start_robot()
        stored_task = self._stored(STORED_TASK)
        bundled = read_bytes(ATLAS_TASK)
        self.assertEqual(read_bytes(stored_task), bundled)
        modified = os.stat(stored_task).st_mtime_ns
        laptop_task = os.path.join(self.config, "mpc", "task.textproto")
        # The laptop's task file with a fifth of its telemetry rate, as the editor of a tab writes it.
        editor = tuned_file.TunedFile(laptop_task, task_file_pb2.TaskFile)
        editor.set("telemetry_frequency", bundled_telemetry_rate() / 5.0)
        edited = editor.edited().text
        self.assertNotEqual(edited.encode("utf-8"), bundled, "the edit changed nothing")

        # A GUI opened on another configuration: the whole-body G1's (robot "g1", not "atlas").
        other_config = copy_robot_config(G1_WB_PACKAGE, self.laptop)
        other = mpc_params_tab.MpcParamsTab(
            self.root,
            task_file=os.path.join(other_config, "mpc", "task.textproto"),
            enable_online_tuning=True,
            robot_saver=self.saver,
        )
        skewed = self._saver(_SkewedPublisher(self.operator.config_save))
        # Each refusal, and a part of the robot's reason for it.
        refusals: list[tuple[str, Callable[[], robot_config_save.SaveOutcome], str]] = [
            (
                "another configuration",
                lambda: self._save(other),
                "another configuration",
            ),
            (
                "another robot",
                lambda: self._send(self.saver, edited, "g1"),
                "robot 'g1'",
            ),
            (
                "a text that does not parse",
                lambda: self._send(self.saver, _UNPARSABLE_TASK, "atlas"),
                "Expected double",
            ),
            (
                "another schema",
                lambda: self._send(skewed, edited, "atlas"),
                "schema fingerprint",
            ),
        ]
        for name, refused, reason in refusals:
            with self.subTest(refusal=name):
                outcome = refused()
                self.assertEqual(outcome.state, _Status.REFUSED, outcome)
                self.assertIn(reason, outcome.message)
                self.assertEqual(read_bytes(stored_task), bundled)
                self.assertEqual(os.stat(stored_task).st_mtime_ns, modified)

        # The same edit, of this configuration and robot, is stored.
        outcome = self._send(self.saver, edited, "atlas")
        self.assertEqual(outcome.state, _Status.SAVED, outcome)
        self.assertEqual(read_bytes(stored_task), edited.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
