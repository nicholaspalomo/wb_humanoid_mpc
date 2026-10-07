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
import threading
import unittest

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
            25.0, tick, threading.Event(), clock=clock, sleep=clock.sleep, max_ticks=5
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
            25.0, tick, threading.Event(), clock=clock, sleep=clock.sleep, max_ticks=4
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
                with self.assertRaisesRegex(ValueError, "the rate must be positive"):
                    teleop.run_periodically(rate_hz, lambda: None, threading.Event())


class TestBusFlags(unittest.TestCase):
    def test_the_defaults_are_the_shipped_network_file_and_the_given_node(self):
        parser = argparse.ArgumentParser()
        teleop.add_bus_flags(parser, default_node=operator_bus.TELEOP_NODE)
        args = parser.parse_args([])
        self.assertEqual(args.network_config, operator_bus.DEFAULT_NETWORK_CONFIG)
        self.assertEqual(args.ipc_node, operator_bus.TELEOP_NODE)
        args = parser.parse_args(
            ["--network_config=/tmp/n.textproto", "--ipc_node=teleop2"]
        )
        self.assertEqual(
            (args.network_config, args.ipc_node), ("/tmp/n.textproto", "teleop2")
        )


if __name__ == "__main__":
    unittest.main()
