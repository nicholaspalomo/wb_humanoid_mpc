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

"""What the teleoperation publishers share (remote_control/teleop.py): the timed loop and the bus flags."""

import argparse
import math
import os
import tempfile
import threading
import unittest

from humanoid_mpc_config import reference_file_pb2

from humanoid_mpc_ipc import topics
import nproto_textproto
import operator_test_support
from remote_control import config_files
from remote_control import operator_bus
from remote_control import teleop


class _FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class TestRunPeriodically(unittest.TestCase):
    def test_it_ticks_on_absolute_deadlines(self):
        clock = _FakeClock()
        ticks = []

        def tick():
            ticks.append(clock.now)
            clock.now += 0.01  # Each tick takes 10 ms of the 40 ms period.

        count = teleop.run_periodically(
            25.0,
            tick,
            threading.Event(),
            clock=clock,
            sleep=clock.sleep,
            max_ticks=5,
        )
        self.assertEqual(count, 5)
        for earlier, later in zip(ticks, ticks[1:]):
            self.assertAlmostEqual(later - earlier, 0.04)

    def test_an_overrun_skips_the_missed_deadlines_instead_of_bursting(self):
        clock = _FakeClock()
        ticks = []

        def tick():
            ticks.append(clock.now)
            # The second tick takes 2.5 periods.
            clock.now += 0.1 if len(ticks) == 2 else 0.0

        teleop.run_periodically(
            25.0,
            tick,
            threading.Event(),
            clock=clock,
            sleep=clock.sleep,
            max_ticks=4,
        )
        gaps = [later - earlier for earlier, later in zip(ticks, ticks[1:])]
        # No gap shorter than a period: the loop never fires twice to catch up.
        for gap in gaps:
            self.assertGreaterEqual(gap, 0.04 - 1e-9)
        # And it is back on the original grid of deadlines.
        for time_of_tick in ticks:
            periods = (time_of_tick - 100.0) / 0.04
            self.assertAlmostEqual(periods, round(periods), places=6)

    def test_setting_stop_ends_the_loop(self):
        stop = threading.Event()
        ticks = []

        def tick():
            ticks.append(1)
            if len(ticks) == 3:
                stop.set()

        self.assertEqual(teleop.run_periodically(1000.0, tick, stop), 3)

    def test_the_rate_must_be_positive(self):
        for rate_hz in (0.0, -1.0, math.nan):
            with self.subTest(rate_hz=rate_hz):
                with self.assertRaisesRegex(
                    ValueError, "the rate must be positive"
                ):
                    teleop.run_periodically(
                        rate_hz, lambda: None, threading.Event()
                    )


class TestBusFlags(unittest.TestCase):
    def test_the_defaults_are_the_shipped_network_file_and_the_given_node(self):
        parser = argparse.ArgumentParser()
        teleop.add_bus_flags(parser, default_node=operator_bus.TELEOP_NODE)
        args = parser.parse_args([])
        self.assertEqual(
            args.network_config, operator_bus.DEFAULT_NETWORK_CONFIG
        )
        self.assertEqual(args.ipc_node, operator_bus.TELEOP_NODE)
        args = parser.parse_args(
            ["--network_config=/tmp/n.textproto", "--ipc_node=teleop2"]
        )
        self.assertEqual(
            (args.network_config, args.ipc_node),
            ("/tmp/n.textproto", "teleop2"),
        )


def _walking_publisher() -> operator_test_support.RecordingPublisher:
    return operator_test_support.RecordingPublisher(
        topics.OPERATOR_WALKING_VELOCITY_COMMAND
    )


class TestStopCommand(unittest.TestCase):
    def test_close_after_stop_sends_the_stop_command_before_it_closes(self):
        publisher = _walking_publisher()
        self.assertTrue(
            teleop.close_after_stop(
                publisher, pelvis_height=0.74, grace_period=0.0
            )
        )
        self.assertEqual(publisher.publish_count, 1)
        stop = publisher.last_message
        assert stop is not None
        self.assertEqual(stop.linear_velocity_x, 0.0)
        self.assertEqual(stop.linear_velocity_y, 0.0)
        self.assertEqual(stop.angular_velocity_z, 0.0)
        self.assertAlmostEqual(stop.desired_pelvis_height, 0.74)
        self.assertTrue(publisher.bus.closed)
        # The stop command went out before the bus closed.
        self.assertEqual(publisher.bus.published_at_close, 1)

    def test_close_after_stop_without_a_height_sends_nothing_and_closes(self):
        publisher = _walking_publisher()
        self.assertFalse(teleop.close_after_stop(publisher, pelvis_height=None))
        self.assertEqual(publisher.publish_count, 0)
        self.assertTrue(publisher.bus.closed)

    def test_the_loop_ends_with_the_stop_command(self):
        publisher = _walking_publisher()
        moving = operator_bus.walking_velocity_command(0.5, 0.0, 0.0, 0.81)

        def tick() -> None:
            publisher.publish(moving)

        ticks = teleop.publish_until_stopped(
            publisher,
            tick,
            threading.Event(),
            lambda: 0.81,
            max_ticks=2,
        )
        self.assertEqual(ticks, 2)
        self.assertEqual(publisher.publish_count, 3)
        stop = publisher.last_message
        assert stop is not None
        self.assertEqual(stop.linear_velocity_x, 0.0)
        self.assertAlmostEqual(stop.desired_pelvis_height, 0.81)
        self.assertEqual(publisher.bus.published_at_close, 3)

    def test_a_tick_that_raises_still_sends_the_stop_command(self):
        publisher = _walking_publisher()

        def tick() -> None:
            raise RuntimeError("the terminal went away")

        with self.assertRaises(RuntimeError):
            teleop.publish_until_stopped(
                publisher, tick, threading.Event(), lambda: 0.8
            )
        self.assertEqual(publisher.publish_count, 1)
        self.assertTrue(publisher.bus.closed)


class TestInitialPelvisHeight(unittest.TestCase):
    def test_it_is_the_reference_files_default_base_height(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "reference.textproto")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    nproto_textproto.file_text(
                        reference_file_pb2.ReferenceFile(
                            default_base_height=0.7405
                        )
                    )
                )
            self.assertAlmostEqual(teleop.initial_pelvis_height(path), 0.7405)

    def test_without_a_reference_file_it_is_the_fallback(self):
        self.assertEqual(
            teleop.initial_pelvis_height(""), config_files.DEFAULT_PELVIS_HEIGHT
        )

    def test_the_flag_defaults_to_no_file(self):
        parser = argparse.ArgumentParser()
        teleop.add_reference_file_flag(parser)
        self.assertEqual(parser.parse_args([]).reference_file, "")
        self.assertEqual(
            parser.parse_args(["--reference_file=r.textproto"]).reference_file,
            "r.textproto",
        )


if __name__ == "__main__":
    unittest.main()
