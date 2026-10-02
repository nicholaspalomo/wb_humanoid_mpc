"""****************************************************************************
Copyright (c) 2024, 1X Technologies. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

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

"""The Xbox controller as a source of walking commands, with a fake joystick: no pygame, no controller, no bus.

Pins the stick shaping and its deadband, the two axis layouts (USB and Bluetooth), the pelvis height integration of
the triggers, what happens when the controller goes away, the poller's periodic rescans, and that the publisher
publishes the controller's command and nothing without one.
"""

import unittest
from typing import Dict, List, Optional

from humanoid_mpc_ipc import topics
from humanoid_mpc_msgs import walking_velocity_command_pb2
from operator_test_support import RecordingPublisher
from remote_control.xbox_controller_interface import (
    BLUETOOTH_LAYOUT,
    BLUETOOTH_NAME_MARKER,
    STICK_DEADBAND,
    USB_LAYOUT,
    GamepadPoller,
    XBoxControllerInterface,
    read_controller_input,
    shape_stick,
)
from remote_control.xbox_walking_command_publisher import XBoxWalkingCommandPublisher

RATE_HZ = 25.0


class FakeJoystick:
    def __init__(
        self, name: str = "Xbox 360 Controller", axes: Optional[Dict[int, float]] = None
    ) -> None:
        self.name = name
        # Released triggers read -1; everything else rests at 0.
        self.axes = {axis: 0.0 for axis in range(8)}
        for layout in (USB_LAYOUT, BLUETOOTH_LAYOUT):
            self.axes[layout.lt] = -1.0
            self.axes[layout.rt] = -1.0
        self.axes.update(axes or {})

    def get_name(self) -> str:
        return self.name

    def get_axis(self, axis: int) -> float:
        return self.axes[axis]


class FakeBackend:
    """Joysticks plugged in and out by the test."""

    def __init__(self, joystick: Optional[FakeJoystick] = None) -> None:
        self.joystick = joystick
        self.scans = 0
        self.pumps = 0
        self.initialized = False

    def init(self) -> None:
        self.initialized = True

    def open_first_joystick(self) -> Optional[FakeJoystick]:
        self.scans += 1
        return self.joystick

    def joystick_count(self) -> int:
        return 1 if self.joystick is not None else 0

    def pump(self) -> None:
        self.pumps += 1


class TestStickShaping(unittest.TestCase):
    def test_the_ends_and_the_center_are_kept(self):
        self.assertEqual(shape_stick(0.0), 0.0)
        self.assertAlmostEqual(shape_stick(1.0), 1.0)
        self.assertAlmostEqual(shape_stick(-1.0), -1.0)

    def test_it_is_odd_and_monotonic(self):
        samples = [i / 50.0 for i in range(-50, 51)]
        shaped = [shape_stick(value) for value in samples]
        for value in samples:
            self.assertAlmostEqual(shape_stick(-value), -shape_stick(value))
        for earlier, later in zip(shaped, shaped[1:]):
            self.assertLessEqual(earlier, later)

    def test_a_drift_inside_the_deadband_reads_as_zero(self):
        self.assertEqual(shape_stick(0.05), 0.0)
        self.assertLess(abs(0.2 * 0.05 + 0.8 * 0.05**3), STICK_DEADBAND)
        self.assertNotEqual(shape_stick(0.2), 0.0)

    def test_the_center_is_softer_than_linear(self):
        self.assertLess(shape_stick(0.5), 0.5)


class TestAxisLayouts(unittest.TestCase):
    def test_forward_and_left_are_positive_in_both_layouts(self):
        for layout in (USB_LAYOUT, BLUETOOTH_LAYOUT):
            with self.subTest(layout=layout):
                # pygame reports a stick pushed forward or left as negative.
                joystick = FakeJoystick(
                    axes={
                        layout.x_left: -1.0,
                        layout.y_left: -1.0,
                        layout.y_right: -1.0,
                    }
                )
                controller_input = read_controller_input(joystick, layout)
                self.assertAlmostEqual(controller_input.x_left, 1.0)
                self.assertAlmostEqual(controller_input.y_left, 1.0)
                self.assertAlmostEqual(controller_input.y_right, 1.0)

    def test_the_triggers_read_zero_released_and_one_pressed(self):
        for layout in (USB_LAYOUT, BLUETOOTH_LAYOUT):
            with self.subTest(layout=layout):
                released = read_controller_input(FakeJoystick(), layout)
                self.assertEqual((released.lt, released.rt), (0.0, 0.0))
                pressed = read_controller_input(
                    FakeJoystick(axes={layout.lt: 1.0, layout.rt: 1.0}), layout
                )
                self.assertEqual((pressed.lt, pressed.rt), (1.0, 1.0))

    def test_the_layouts_map_six_distinct_axes(self):
        for layout in (USB_LAYOUT, BLUETOOTH_LAYOUT):
            axes = [
                layout.x_left,
                layout.y_left,
                layout.x_right,
                layout.y_right,
                layout.lt,
                layout.rt,
            ]
            self.assertEqual(len(set(axes)), 6, layout)


class TestXBoxControllerInterface(unittest.TestCase):
    def test_no_controller_gives_no_command(self):
        backend = FakeBackend()
        controller = XBoxControllerInterface(RATE_HZ, backend=backend)
        self.assertTrue(backend.initialized)
        self.assertFalse(controller.joystick_connected)
        self.assertEqual(controller.get_walking_command_msg(), (False, None))

    def test_the_sticks_become_the_command(self):
        joystick = FakeJoystick(axes={USB_LAYOUT.x_left: -1.0, USB_LAYOUT.y_right: 1.0})
        controller = XBoxControllerInterface(RATE_HZ, backend=FakeBackend(joystick))
        success, command = controller.get_walking_command_msg()
        self.assertTrue(success)
        self.assertIsInstance(
            command, walking_velocity_command_pb2.WalkingVelocityCommand
        )
        self.assertAlmostEqual(command.linear_velocity_x, 1.0)
        self.assertEqual(command.linear_velocity_y, 0.0)
        self.assertAlmostEqual(command.angular_velocity_z, -1.0)

    def test_a_bluetooth_controller_is_read_in_its_layout(self):
        joystick = FakeJoystick(
            name=f"Xbox {BLUETOOTH_NAME_MARKER}",
            axes={BLUETOOTH_LAYOUT.y_right: -1.0, USB_LAYOUT.y_right: 0.0},
        )
        controller = XBoxControllerInterface(RATE_HZ, backend=FakeBackend(joystick))
        self.assertTrue(controller.bluetooth_connection)
        _, command = controller.get_walking_command_msg()
        self.assertAlmostEqual(command.angular_velocity_z, 1.0)

    def test_the_triggers_move_the_height_at_one_meter_per_second_within_its_range(
        self,
    ):
        joystick = FakeJoystick(axes={USB_LAYOUT.rt: 1.0})
        controller = XBoxControllerInterface(RATE_HZ, backend=FakeBackend(joystick))
        start = controller.current_pelvis_height_target
        _, command = controller.get_walking_command_msg()
        self.assertAlmostEqual(command.desired_pelvis_height - start, 1.0 / RATE_HZ)
        for _ in range(int(RATE_HZ) * 2):
            _, command = controller.get_walking_command_msg()
        self.assertEqual(command.desired_pelvis_height, controller.max_pelvis_height)
        joystick.axes[USB_LAYOUT.rt] = -1.0
        joystick.axes[USB_LAYOUT.lt] = 1.0
        for _ in range(int(RATE_HZ) * 2):
            _, command = controller.get_walking_command_msg()
        self.assertEqual(command.desired_pelvis_height, controller.min_pelvis_height)

    def test_a_controller_that_goes_away_is_marked_disconnected(self):
        backend = FakeBackend(FakeJoystick())
        controller = XBoxControllerInterface(RATE_HZ, backend=backend)
        self.assertTrue(controller.joystick_connected)
        backend.joystick = None
        self.assertEqual(controller.get_walking_command_msg(), (False, None))
        self.assertFalse(controller.joystick_connected)

    def test_a_controller_that_raises_is_marked_disconnected(self):
        class BrokenJoystick(FakeJoystick):
            def get_axis(self, axis: int) -> float:
                raise RuntimeError("the device went away")

        controller = XBoxControllerInterface(
            RATE_HZ, backend=FakeBackend(BrokenJoystick())
        )
        self.assertEqual(controller.get_walking_command_msg(), (False, None))
        self.assertFalse(controller.joystick_connected)


class TestGamepadPoller(unittest.TestCase):
    def test_without_a_controller_it_rescans_every_scan_period(self):
        backend = FakeBackend()
        controller = XBoxControllerInterface(RATE_HZ, backend=backend)
        poller = GamepadPoller(controller, RATE_HZ, scan_period=2.0)
        scans_before = backend.scans
        for _ in range(int(2 * RATE_HZ) - 1):
            self.assertIsNone(poller.tick())
        self.assertEqual(backend.scans, scans_before)
        self.assertIsNone(poller.tick())
        self.assertEqual(backend.scans, scans_before + 1)

    def test_a_controller_plugged_in_is_found_at_the_next_scan(self):
        backend = FakeBackend()
        controller = XBoxControllerInterface(RATE_HZ, backend=backend)
        poller = GamepadPoller(controller, RATE_HZ, scan_period=1.0 / RATE_HZ)
        backend.joystick = FakeJoystick(axes={USB_LAYOUT.x_left: -1.0})
        self.assertIsNone(poller.tick())
        self.assertTrue(poller.connected)
        command = poller.tick()
        self.assertAlmostEqual(command.linear_velocity_x, 1.0)

    def test_the_periods_must_be_positive(self):
        controller = XBoxControllerInterface(RATE_HZ, backend=FakeBackend())
        with self.assertRaises(ValueError):
            GamepadPoller(controller, 0.0)
        with self.assertRaises(ValueError):
            GamepadPoller(controller, RATE_HZ, scan_period=0.0)


class TestXBoxWalkingCommandPublisher(unittest.TestCase):
    def test_it_publishes_the_controllers_command_and_nothing_without_one(self):
        backend = FakeBackend(FakeJoystick(axes={USB_LAYOUT.y_left: -1.0}))
        controller = XBoxControllerInterface(RATE_HZ, backend=backend)
        publisher = RecordingPublisher(topics.OPERATOR_WALKING_VELOCITY_COMMAND)
        node = XBoxWalkingCommandPublisher(
            publisher, GamepadPoller(controller, RATE_HZ)
        )
        node.tick()
        self.assertEqual(publisher.publish_count, 1)
        self.assertAlmostEqual(publisher.last_message.linear_velocity_y, 1.0)

        backend.joystick = None
        for _ in range(5):
            node.tick()
        self.assertEqual(publisher.publish_count, 1)


if __name__ == "__main__":
    unittest.main()
