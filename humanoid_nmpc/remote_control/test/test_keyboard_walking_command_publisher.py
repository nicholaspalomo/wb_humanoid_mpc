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

"""The keyboard teleoperation, without a terminal or a bus: key decoding, the command of each key, and the publisher.

The regressions these pin: every key but `w` and `s` used to be undone in the same call that applied it (a trailing
`else` reset the velocities), so the arrow keys and `a`/`d` never moved the robot; the status line read an attribute
that was never set, so the first tick raised; and the pelvis height kept climbing after `w` was released.
"""

from collections.abc import Iterable
import io
import unittest

from humanoid_mpc_msgs import walking_velocity_command_pb2

from humanoid_mpc_ipc import topics
import operator_test_support
from remote_control import keyboard_walking_command_publisher


def _reader(text: str):
    """A read_char over `text`, then None: what a terminal with those characters typed returns."""
    characters = iter(text)

    def read_char() -> str | None:
        return next(characters, None)

    return read_char


class TestReadKey(unittest.TestCase):
    def test_the_arrow_keys_escape_sequences(self):
        for sequence, key in (
            ("\x1b[A", keyboard_walking_command_publisher.KEY_UP),
            ("\x1b[B", keyboard_walking_command_publisher.KEY_DOWN),
            ("\x1b[C", keyboard_walking_command_publisher.KEY_RIGHT),
            ("\x1b[D", keyboard_walking_command_publisher.KEY_LEFT),
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    keyboard_walking_command_publisher.read_key(_reader(sequence)), key
                )

    def test_a_plain_key_is_its_character_and_nothing_is_none(self):
        self.assertEqual(keyboard_walking_command_publisher.read_key(_reader("w")), "w")
        self.assertIsNone(keyboard_walking_command_publisher.read_key(_reader("")))

    def test_an_unknown_or_cut_escape_sequence_is_no_key(self):
        self.assertIsNone(
            keyboard_walking_command_publisher.read_key(_reader("\x1b[Z"))
        )
        self.assertIsNone(keyboard_walking_command_publisher.read_key(_reader("\x1bO")))
        self.assertIsNone(keyboard_walking_command_publisher.read_key(_reader("\x1b")))

    def test_one_key_is_read_at_a_time(self):
        read_char = _reader("\x1b[Aw")
        self.assertEqual(
            keyboard_walking_command_publisher.read_key(read_char),
            keyboard_walking_command_publisher.KEY_UP,
        )
        self.assertEqual(keyboard_walking_command_publisher.read_key(read_char), "w")


class TestKeyboardCommand(unittest.TestCase):
    def test_each_key_commands_its_axis(self):
        state = keyboard_walking_command_publisher.KeyboardCommand(
            max_vel_x=1.0, max_vel_y=0.5, max_vel_yaw=0.8
        )
        for key, expected in (
            (keyboard_walking_command_publisher.KEY_UP, (1.0, 0.0, 0.0)),
            (keyboard_walking_command_publisher.KEY_DOWN, (-1.0, 0.0, 0.0)),
            (keyboard_walking_command_publisher.KEY_LEFT, (0.0, 0.5, 0.0)),
            (keyboard_walking_command_publisher.KEY_RIGHT, (0.0, -0.5, 0.0)),
            ("a", (0.0, 0.0, 0.8)),
            ("d", (0.0, 0.0, -0.8)),
        ):
            with self.subTest(key=key):
                command = state.apply(key)
                self.assertEqual(
                    (
                        command.linear_velocity_x,
                        command.linear_velocity_y,
                        command.angular_velocity_z,
                    ),
                    expected,
                )

    def test_releasing_every_key_stops_the_robot(self):
        state = keyboard_walking_command_publisher.KeyboardCommand()
        state.apply(keyboard_walking_command_publisher.KEY_UP)
        for key in (None, "q"):
            with self.subTest(key=key):
                state.apply(keyboard_walking_command_publisher.KEY_UP)
                command = state.apply(key)
                self.assertEqual(command.linear_velocity_x, 0.0)
                self.assertEqual(command.angular_velocity_z, 0.0)

    def test_the_height_moves_one_step_per_tick_held_and_stays_when_released(self):
        state = keyboard_walking_command_publisher.KeyboardCommand(
            pelvis_height=0.8, pelvis_height_step=0.01
        )
        self.assertAlmostEqual(state.apply("w").desired_pelvis_height, 0.81)
        self.assertAlmostEqual(state.apply("w").desired_pelvis_height, 0.82)
        # Released: it stays where it was raised to, and stops rising.
        self.assertAlmostEqual(state.apply(None).desired_pelvis_height, 0.82)
        self.assertAlmostEqual(state.apply(None).desired_pelvis_height, 0.82)
        self.assertAlmostEqual(state.apply("s").desired_pelvis_height, 0.81)

    def test_the_height_stays_in_its_range(self):
        state = keyboard_walking_command_publisher.KeyboardCommand(
            pelvis_height=0.99, max_pelvis_height=1.0
        )
        for _ in range(5):
            command = state.apply("w")
        self.assertEqual(command.desired_pelvis_height, 1.0)
        low = keyboard_walking_command_publisher.KeyboardCommand(
            pelvis_height=0.0, min_pelvis_height=0.2
        )
        self.assertEqual(low.command().desired_pelvis_height, 0.2)

    def test_the_status_line_shows_the_command(self):
        state = keyboard_walking_command_publisher.KeyboardCommand()
        state.apply(keyboard_walking_command_publisher.KEY_UP)
        self.assertIn("x=+1.00", state.status_line())
        self.assertIn("pelvis height=0.80", state.status_line())


class TestKeyboardWalkingCommandPublisher(unittest.TestCase):
    def _node(self, typed: Iterable[str]):
        self.publisher = operator_test_support.RecordingPublisher(
            topics.OPERATOR_WALKING_VELOCITY_COMMAND
        )
        self.discards = 0

        def discard_pending():
            self.discards += 1

        self.output = io.StringIO()
        return keyboard_walking_command_publisher.KeyboardWalkingCommandPublisher(
            self.publisher,
            _reader("".join(typed)),
            discard_pending,
            output=self.output,
        )

    def test_every_tick_publishes_one_command(self):
        node = self._node(["\x1b[A"])
        node.tick()
        node.tick()
        self.assertEqual(self.publisher.publish_count, 2)
        first, second = self.publisher.messages
        self.assertIsInstance(
            first, walking_velocity_command_pb2.WalkingVelocityCommand
        )
        self.assertEqual(first.linear_velocity_x, 1.0)
        # No key on the second tick: the robot stops.
        self.assertEqual(second.linear_velocity_x, 0.0)

    def test_a_key_discards_its_auto_repeat_backlog(self):
        node = self._node(["a"])
        node.tick()
        node.tick()
        self.assertEqual(self.discards, 1)

    def test_the_status_line_is_written_every_tick(self):
        node = self._node([])
        node.tick()
        self.assertIn("Command:", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
