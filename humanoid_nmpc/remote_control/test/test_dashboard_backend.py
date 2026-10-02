"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

"""The dashboard backend: the simulation process manager and the virtual joystick, without a bus or a simulator.

SimProcessManager runs targets of a test table (shell commands) instead of the Makefile's launch targets, and stops
only the process group it started: nothing here signals a process it did not start. VirtualJoystick publishes through
the GUI's walking command publisher over a bus that records.
"""

import os
import threading
import time
import unittest

import robot_ipc
from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import walking_velocity_command_pb2
from operator_test_support import RecordingPublisher
from remote_control.dashboard_backend import SimProcessManager, VirtualJoystick


def _wait_for(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers; it is gone for every purpose here once its state says so.
    try:
        with open(f"/proc/{pid}/stat", "r") as stat:
            return stat.read().split(") ", 1)[1][0] != "Z"
    except OSError:
        return False


class _TestTargets(SimProcessManager):
    """The manager over shell commands: a grandchild that waits, one that ignores SIGTERM, and one that prints."""

    TARGETS = {
        "sleeper": {
            "name": "Sleeper",
            "command": "sleep 60 & echo grandchild=$! ; wait",
            "type": "test",
            "robot": "none",
        },
        "stubborn": {
            "name": "Stubborn",
            "command": "trap '' TERM; sleep 60 & echo grandchild=$! ; wait",
            "type": "test",
            "robot": "none",
        },
        "talker": {
            "name": "Talker",
            "command": "echo hello from the target",
            "type": "test",
            "robot": "none",
        },
        # The leader exits at once and leaves its background child in the group, off the output pipe.
        "orphaner": {
            "name": "Orphaner",
            "command": "sleep 60 > /dev/null 2>&1 & echo grandchild=$!",
            "type": "test",
            "robot": "none",
        },
    }


class TestSimProcessManagerTargets(unittest.TestCase):
    def test_every_target_names_a_make_target_and_its_robot(self):
        self.assertIn("atlas_centroidal_sim", SimProcessManager.TARGETS)
        self.assertIn("g1_wb_dummy", SimProcessManager.TARGETS)
        for key, target in SimProcessManager.TARGETS.items():
            with self.subTest(target=key):
                self.assertEqual(set(target), {"name", "command", "type", "robot"})
                self.assertTrue(target["command"].startswith("make launch-"))
                self.assertIn(target["type"], ("dummy", "mujoco"))

    def test_a_new_manager_is_stopped(self):
        manager = SimProcessManager()
        self.assertEqual(manager.get_status()["status"], "STOPPED")
        # Stopping what never ran is a no-op.
        self.assertTrue(manager.stop())

    def test_an_unknown_target_is_refused_with_the_known_ones(self):
        with self.assertRaisesRegex(ValueError, "atlas_centroidal_sim"):
            SimProcessManager().launch("no_such_target")


class TestSimProcessManagerProcesses(unittest.TestCase):
    def setUp(self):
        self.manager = _TestTargets(stop_grace_period=0.5)
        self.addCleanup(self.manager.stop)
        self.output = []
        self._output_lock = threading.Lock()

    def _on_output(self, chunk):
        with self._output_lock:
            self.output.append(chunk)

    def _text(self):
        with self._output_lock:
            return "".join(self.output)

    def _grandchild_pid(self):
        # The target prints the pid of the sleep it started in the background. Other lines may come first: a shell's
        # startup files (BASH_ENV) can print.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            line = self.manager.log_queue.get(timeout=10.0).strip()
            if line.startswith("grandchild="):
                return int(line.split("=", 1)[1])
        self.fail("the target never printed its grandchild's pid")

    def test_launch_runs_and_stop_ends_the_whole_process_group(self):
        self.assertTrue(self.manager.launch("sleeper", on_output=self._on_output))
        status = self.manager.get_status()
        self.assertEqual(status["status"], "RUNNING")
        self.assertEqual(status["target"], "sleeper")
        grandchild = self._grandchild_pid()
        self.assertTrue(_alive(grandchild))

        self.manager.stop()
        self.assertEqual(self.manager.get_status()["status"], "STOPPED")
        self.assertTrue(_wait_for(lambda: not _alive(grandchild)))

    def test_a_target_that_ignores_sigterm_is_killed_after_the_grace_period(self):
        self.manager.launch("stubborn")
        grandchild = self._grandchild_pid()
        started = time.monotonic()
        self.manager.stop()
        self.assertTrue(_wait_for(lambda: not _alive(grandchild)))
        # Not before the grace period: SIGTERM came first.
        self.assertGreaterEqual(time.monotonic() - started, 0.4)

    def test_the_output_reaches_the_callback_and_the_status_follows_the_exit(self):
        self.manager.launch("talker", on_output=self._on_output)
        self.assertTrue(_wait_for(lambda: "hello from the target" in self._text()))
        self.assertTrue(
            _wait_for(lambda: self.manager.get_status()["status"] == "STOPPED")
        )

    def test_stop_ends_what_an_exited_leader_left_in_its_group(self):
        self.manager.launch("orphaner")
        grandchild = self._grandchild_pid()
        process = self.manager.process
        self.assertTrue(_wait_for(lambda: process.poll() is not None))
        self.assertTrue(
            _alive(grandchild), "positive control: the group outlived its leader"
        )

        self.manager.stop()
        self.assertTrue(_wait_for(lambda: not _alive(grandchild)))

    def test_launching_another_target_stops_the_first(self):
        self.manager.launch("sleeper")
        grandchild = self._grandchild_pid()
        self.manager.launch("talker")
        self.assertTrue(_wait_for(lambda: not _alive(grandchild)))


class TestVirtualJoystick(unittest.TestCase):
    def setUp(self):
        self.publisher = RecordingPublisher(topics.OPERATOR_WALKING_VELOCITY_COMMAND)
        self.joy = VirtualJoystick(
            robot_name="atlas", publisher=self.publisher, auto_stream=False
        )
        self.addCleanup(self.joy.shutdown)

    def test_set_velocity_publishes_the_command(self):
        self.joy.set_velocity(
            linear_x=0.5, linear_y=-0.2, angular_z=0.3, desired_height=0.7
        )
        self.assertEqual(self.publisher.publish_count, 1)
        command = self.publisher.last_message
        self.assertIsInstance(
            command, walking_velocity_command_pb2.WalkingVelocityCommand
        )
        self.assertAlmostEqual(command.linear_velocity_x, 0.5)
        self.assertAlmostEqual(command.linear_velocity_y, -0.2)
        self.assertAlmostEqual(command.angular_velocity_z, 0.3)
        self.assertAlmostEqual(command.desired_pelvis_height, 0.7)

    def test_steps_and_the_emergency_stop(self):
        self.joy.set_velocity(linear_x=0.5, linear_y=-0.2, angular_z=0.3)
        self.joy.step("forward", delta_v=0.2)
        self.assertAlmostEqual(self.publisher.last_message.linear_velocity_x, 0.7)
        # The steps saturate.
        for _ in range(10):
            self.joy.step("forward")
        self.assertAlmostEqual(self.publisher.last_message.linear_velocity_x, 1.0)

        self.joy.step("stop")
        command = self.publisher.last_message
        self.assertEqual(command.linear_velocity_x, 0.0)
        self.assertEqual(command.linear_velocity_y, 0.0)
        self.assertEqual(command.angular_velocity_z, 0.0)
        # The height is not a velocity: the stop keeps it.
        self.assertGreater(command.desired_pelvis_height, 0.0)

    def test_an_unknown_direction_is_refused(self):
        with self.assertRaisesRegex(ValueError, "turn_left"):
            self.joy.step("sideways")

    def test_the_initial_height_is_the_robots_nominal_one(self):
        self.joy.publish_now()
        self.assertAlmostEqual(
            self.publisher.last_message.desired_pelvis_height, self.joy.desired_height
        )
        self.assertGreater(self.joy.desired_height, 0.3)

    def test_streaming_publishes_at_the_rate_until_stopped(self):
        self.joy.publish_rate = 100.0
        self.joy.start_streaming()
        self.assertTrue(_wait_for(lambda: self.publisher.publish_count >= 5))
        self.joy.stop_streaming()
        count = self.publisher.publish_count
        time.sleep(0.1)
        self.assertEqual(self.publisher.publish_count, count)

    def test_a_bus_closed_under_the_stream_thread_stops_only_the_sending(self):
        # shutdown() waits a second for the stream thread and then closes the bus; a publish that comes after that is
        # refused (robot_ipc.BusError), which must not end the thread with a traceback or reach the caller.
        class ClosedBusPublisher(RecordingPublisher):
            def __init__(self):
                super().__init__(topics.OPERATOR_WALKING_VELOCITY_COMMAND)
                self.refused = 0

            def publish(self, message):
                self.refused += 1
                raise robot_ipc.BusError("cannot publish: the bus is closed")

        publisher = ClosedBusPublisher()
        joystick = VirtualJoystick(publisher=publisher, auto_stream=False)
        self.addCleanup(joystick.shutdown)
        uncaught = []
        previous_hook = threading.excepthook
        threading.excepthook = lambda arguments: uncaught.append(arguments.exc_value)
        self.addCleanup(setattr, threading, "excepthook", previous_hook)

        joystick.set_velocity(linear_x=0.5)
        joystick.publish_rate = 100.0
        joystick.start_streaming()
        self.assertTrue(_wait_for(lambda: publisher.refused >= 5))
        joystick.stop_streaming()
        self.assertEqual(uncaught, [])

    def test_a_publisher_of_another_topic_is_refused(self):
        with self.assertRaisesRegex(ValueError, "walking command"):
            VirtualJoystick(publisher=RecordingPublisher(topics.OPERATOR_FSM_COMMAND))

    def test_a_new_joystick_shuts_the_previous_one_down(self):
        self.joy.publish_rate = 100.0
        self.joy.start_streaming()
        replacement = VirtualJoystick(
            publisher=RecordingPublisher(topics.OPERATOR_WALKING_VELOCITY_COMMAND),
            auto_stream=False,
        )
        self.addCleanup(replacement.shutdown)
        count = self.publisher.publish_count
        time.sleep(0.1)
        self.assertEqual(self.publisher.publish_count, count)


if __name__ == "__main__":
    unittest.main()
