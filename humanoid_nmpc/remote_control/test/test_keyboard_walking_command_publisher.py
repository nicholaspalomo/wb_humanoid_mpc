"""The keyboard teleoperation, without a terminal or a bus: key decoding, the command of each key, and the publisher.

The regressions these pin: every key but `w` and `s` used to be undone in the same call that applied it (a trailing
`else` reset the velocities), so the arrow keys and `a`/`d` never moved the robot; the status line read an attribute
that was never set, so the first tick raised; and the pelvis height kept climbing after `w` was released.
"""

import io
import unittest
from typing import Iterable, Optional

from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import walking_velocity_command_pb2
from operator_test_support import RecordingPublisher
from remote_control.keyboard_walking_command_publisher import (
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
    KeyboardCommand,
    KeyboardWalkingCommandPublisher,
    read_key,
)


def _reader(text: str):
    """A read_char over `text`, then None: what a terminal with those characters typed returns."""
    characters = iter(text)

    def read_char() -> Optional[str]:
        return next(characters, None)

    return read_char


class TestReadKey(unittest.TestCase):
    def test_the_arrow_keys_escape_sequences(self):
        for sequence, key in (
            ("\x1b[A", KEY_UP),
            ("\x1b[B", KEY_DOWN),
            ("\x1b[C", KEY_RIGHT),
            ("\x1b[D", KEY_LEFT),
        ):
            with self.subTest(key=key):
                self.assertEqual(read_key(_reader(sequence)), key)

    def test_a_plain_key_is_its_character_and_nothing_is_none(self):
        self.assertEqual(read_key(_reader("w")), "w")
        self.assertIsNone(read_key(_reader("")))

    def test_an_unknown_or_cut_escape_sequence_is_no_key(self):
        self.assertIsNone(read_key(_reader("\x1b[Z")))
        self.assertIsNone(read_key(_reader("\x1bO")))
        self.assertIsNone(read_key(_reader("\x1b")))

    def test_one_key_is_read_at_a_time(self):
        read_char = _reader("\x1b[Aw")
        self.assertEqual(read_key(read_char), KEY_UP)
        self.assertEqual(read_key(read_char), "w")


class TestKeyboardCommand(unittest.TestCase):
    def test_each_key_commands_its_axis(self):
        state = KeyboardCommand(max_vel_x=1.0, max_vel_y=0.5, max_vel_yaw=0.8)
        for key, expected in (
            (KEY_UP, (1.0, 0.0, 0.0)),
            (KEY_DOWN, (-1.0, 0.0, 0.0)),
            (KEY_LEFT, (0.0, 0.5, 0.0)),
            (KEY_RIGHT, (0.0, -0.5, 0.0)),
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
        state = KeyboardCommand()
        state.apply(KEY_UP)
        for key in (None, "q"):
            with self.subTest(key=key):
                state.apply(KEY_UP)
                command = state.apply(key)
                self.assertEqual(command.linear_velocity_x, 0.0)
                self.assertEqual(command.angular_velocity_z, 0.0)

    def test_the_height_moves_one_step_per_tick_held_and_stays_when_released(self):
        state = KeyboardCommand(pelvis_height=0.8, pelvis_height_step=0.01)
        self.assertAlmostEqual(state.apply("w").desired_pelvis_height, 0.81)
        self.assertAlmostEqual(state.apply("w").desired_pelvis_height, 0.82)
        # Released: it stays where it was raised to, and stops rising.
        self.assertAlmostEqual(state.apply(None).desired_pelvis_height, 0.82)
        self.assertAlmostEqual(state.apply(None).desired_pelvis_height, 0.82)
        self.assertAlmostEqual(state.apply("s").desired_pelvis_height, 0.81)

    def test_the_height_stays_in_its_range(self):
        state = KeyboardCommand(pelvis_height=0.99, max_pelvis_height=1.0)
        for _ in range(5):
            command = state.apply("w")
        self.assertEqual(command.desired_pelvis_height, 1.0)
        low = KeyboardCommand(pelvis_height=0.0, min_pelvis_height=0.2)
        self.assertEqual(low.command().desired_pelvis_height, 0.2)

    def test_the_status_line_shows_the_command(self):
        state = KeyboardCommand()
        state.apply(KEY_UP)
        self.assertIn("x=+1.00", state.status_line())
        self.assertIn("pelvis height=0.80", state.status_line())


class TestKeyboardWalkingCommandPublisher(unittest.TestCase):
    def _node(self, typed: Iterable[str]):
        self.publisher = RecordingPublisher(topics.OPERATOR_WALKING_VELOCITY_COMMAND)
        self.discards = 0

        def discard_pending():
            self.discards += 1

        self.output = io.StringIO()
        return KeyboardWalkingCommandPublisher(
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
