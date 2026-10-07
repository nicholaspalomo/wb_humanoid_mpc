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

"""The viewer's periods reject zero, negative and NaN values: the BusBridge options and the --flush_period flag.

The checks are written `not period > 0.0`: `period <= 0.0` is False for NaN, which would then reach the bus's loop.
"""

import contextlib
import io
import math
import unittest
from unittest import mock

from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import bus_bridge
from humanoid_rerun_viewer import cli
import robot_ipc

INVALID_PERIODS = (0.0, -1.0, math.nan)


def _network() -> robot_ipc.NetworkConfig:
    """A network of one loopback node, which the rejected options never reach."""
    return robot_ipc.NetworkConfig(
        nodes=(
            robot_ipc.NodeEndpoint("publisher", "127.0.0.1", robot_ipc.EPHEMERAL_PORT),
        )
    )


class BusBridgePeriodTest(unittest.TestCase):
    def test_the_periods_must_be_positive(self) -> None:
        the_bridge = mock.create_autospec(bridge.RerunBridge, instance=True)
        for period in INVALID_PERIODS:
            with self.subTest(option="flush_period", period=period):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    bus_bridge.BusBridge(the_bridge, _network(), flush_period=period)
            with self.subTest(option="report_period", period=period):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    bus_bridge.BusBridge(the_bridge, _network(), report_period=period)


class FlushPeriodFlagTest(unittest.TestCase):
    def test_the_flag_must_be_positive(self) -> None:
        for period in INVALID_PERIODS:
            with self.subTest(period=period):
                errors = io.StringIO()
                with contextlib.redirect_stderr(errors):
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(["--flush_period", str(period)])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--flush_period must be positive", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
