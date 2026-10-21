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


"""The bridge's watch of the viewer of spawn and connect: when it is gone, when it is back, and what cli.run then does.

The probe and the clock are fakes; test_end_to_end runs the binary against a viewer that goes away.
"""

import unittest

from humanoid_rerun_viewer import cli


class ManualClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class FakeProbe:
    """A probe of a viewer that is `up` or refuses the connection."""

    def __init__(self, up: bool) -> None:
        self.up = up
        self.calls = 0

    def __call__(self, endpoint: tuple[str, int], timeout: float) -> None:
        del endpoint, timeout  # Unused.
        self.calls += 1
        if not self.up:
            raise ConnectionRefusedError("refused")


ENDPOINT = ("127.0.0.1", 40123)


class ViewerWatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = ManualClock()

    def watch(self, up: bool, probe: FakeProbe) -> cli.ViewerWatch:
        return cli.ViewerWatch(ENDPOINT, up=up, probe=probe, clock=self.clock)

    def poll_every_period(self, watch: cli.ViewerWatch, polls: int) -> list:
        changes = []
        for _ in range(polls):
            self.clock.now += cli.VIEWER_PROBE_PERIOD_S
            changes.append(watch.poll())
        return changes

    def test_a_viewer_unreachable_for_the_grace_period_is_gone_once(self) -> None:
        probe = FakeProbe(up=False)
        watch = self.watch(up=True, probe=probe)
        polls = int(cli.VIEWER_GONE_AFTER_S / cli.VIEWER_PROBE_PERIOD_S) + 3
        changes = self.poll_every_period(watch, polls)
        self.assertEqual(changes.count(cli.ViewerChange.GONE), 1)
        gone_at = changes.index(cli.ViewerChange.GONE)
        # The first refused probe starts the grace period; the probe that is VIEWER_GONE_AFTER_S later ends it.
        self.assertEqual(
            gone_at, int(cli.VIEWER_GONE_AFTER_S / cli.VIEWER_PROBE_PERIOD_S)
        )
        self.assertFalse(watch.up)

    def test_a_short_outage_is_not_gone(self) -> None:
        probe = FakeProbe(up=False)
        watch = self.watch(up=True, probe=probe)
        changes = self.poll_every_period(watch, 2)
        probe.up = True
        changes += self.poll_every_period(watch, 1)
        probe.up = False
        changes += self.poll_every_period(watch, 2)
        self.assertEqual(changes, [None] * 5)

    def test_a_viewer_that_comes_back_is_back(self) -> None:
        probe = FakeProbe(up=False)
        watch = self.watch(up=False, probe=probe)
        self.assertEqual(self.poll_every_period(watch, 2), [None, None])
        probe.up = True
        self.assertEqual(
            self.poll_every_period(watch, 2), [cli.ViewerChange.BACK, None]
        )
        self.assertTrue(watch.up)

    def test_a_probe_waits_for_its_period(self) -> None:
        probe = FakeProbe(up=True)
        watch = self.watch(up=True, probe=probe)
        for _ in range(5):
            self.clock.now += cli.VIEWER_PROBE_PERIOD_S / 10.0
            watch.poll()
        self.assertEqual(probe.calls, 0)
        self.poll_every_period(watch, 1)
        self.assertEqual(probe.calls, 1)


class FakeSink:
    def __init__(self, owns_viewer: bool) -> None:
        self.owns_viewer = owns_viewer
        self.reattached: list[object] = []

    def reattach(self, recording: object) -> str:
        self.reattached.append(recording)
        return "reattached"


class FakeFeed:
    def __init__(self) -> None:
        self.paused_for: list[str] = []
        self.reconnects: list = []

    def pause(self, reason: str) -> None:
        self.paused_for.append(reason)

    def resume(self, reconnect) -> None:
        self.reconnects.append(reconnect)


class FakeBridge:
    def __init__(self) -> None:
        self.recording = object()
        self.static_sent = 0

    def resend_static(self) -> None:
        self.static_sent += 1


class FollowViewerTest(unittest.TestCase):
    def follow(self, sink: FakeSink, change: cli.ViewerChange | None):
        watch = cli.ViewerWatch(
            ENDPOINT, up=True, probe=FakeProbe(up=True), clock=ManualClock()
        )
        watch.poll = lambda: change  # type: ignore[method-assign]  # The change under test.
        feed = FakeFeed()
        the_bridge = FakeBridge()
        keep_running = cli._follow_viewer(watch, sink, feed, the_bridge)  # type: ignore[arg-type]  # Fakes.
        return keep_running, feed, the_bridge

    def test_a_spawned_viewer_that_is_gone_ends_the_run(self) -> None:
        # The GUI's button can then start another bridge and viewer: is_running() turns false.
        keep_running, feed, _ = self.follow(
            FakeSink(owns_viewer=True), cli.ViewerChange.GONE
        )
        self.assertFalse(keep_running)
        self.assertEqual(feed.paused_for, [])

    def test_a_connect_viewer_that_is_gone_pauses_the_feed(self) -> None:
        keep_running, feed, _ = self.follow(
            FakeSink(owns_viewer=False), cli.ViewerChange.GONE
        )
        self.assertTrue(keep_running)
        self.assertEqual(len(feed.paused_for), 1)
        reason = feed.paused_for[0]
        self.assertIn("127.0.0.1:40123", reason)

    def test_a_viewer_that_is_back_is_reattached_and_sent_the_static_data(self) -> None:
        sink = FakeSink(owns_viewer=False)
        keep_running, feed, the_bridge = self.follow(sink, cli.ViewerChange.BACK)
        self.assertTrue(keep_running)
        self.assertEqual(len(feed.reconnects), 1)
        reconnect = feed.reconnects[0]
        self.assertEqual(sink.reattached, [])
        reconnect()
        self.assertEqual(sink.reattached, [the_bridge.recording])
        self.assertEqual(the_bridge.static_sent, 1)

    def test_nothing_changed_does_nothing(self) -> None:
        keep_running, feed, _ = self.follow(FakeSink(owns_viewer=True), None)
        self.assertTrue(keep_running)
        self.assertEqual((feed.paused_for, feed.reconnects), ([], []))


if __name__ == "__main__":
    unittest.main()
