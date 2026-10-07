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

"""Measures what the bridge costs per message at the size of the contract.

The cost is process CPU time (every thread, the Rerun SDK's encoder and file writer included) while it writes an .rrd
file.

The test prints the measurements (bazel test --test_output=all) and fails only far above them, so that a slow machine
does not make it flaky while a change that makes the bridge many times slower still fails it.
"""

from collections.abc import Callable
import os
import shutil
import tempfile
import time
import unittest

from humanoid_mpc_msgs import telemetry_series_pb2

from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import telemetry_contract
from humanoid_rerun_viewer import urdf_model
import synthetic_messages

# A G1-sized robot: 29 joints, the four telemetryFrames of the shipped task files, a centroidal MPC's state and input.
JOINTS = tuple(f"joint_{index}" for index in range(29))
FRAMES = ("foot_l_contact", "foot_r_contact", "pelvis", "torso_link")
LINK_COUNT = 30
TELEMETRY_RATE_HZ = 100.0
FLUSH_PERIOD_S = 0.05
TELEMETRY_MESSAGES = 1000
SCENES = 200
# The bounds: about ten times what the development workstation measures.
MAX_TELEMETRY_CPU_S = 0.010
MAX_SCENE_CPU_S = 0.050


def cpu_seconds_per_call(
    function: Callable[[int], None], count: int
) -> tuple[float, float]:
    """(process CPU, wall) seconds per call of `function(index)` over `count` calls."""
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    for index in range(count):
        function(index)
    return (
        (time.process_time() - cpu_start) / count,
        (time.perf_counter() - wall_start) / count,
    )


def box_robot(links: int) -> urdf_model.RobotModel:
    """A model of `links` links with one box each: link poses cost the same as with meshes, the meshes are not timed."""
    text = '<robot name="boxes">' + "".join(
        f'<link name="link_{index}"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>'
        for index in range(links)
    )
    text += "".join(
        f'<joint name="joint_{index}" type="fixed"><parent link="link_0"/><child link="link_{index}"/></joint>'
        for index in range(1, links)
    )
    return urdf_model.parse_urdf(text + "</robot>", "boxes.urdf")


class BridgeCostTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.recording = bridge.new_recording("test_bridge_cost")
        self.recording.save(os.path.join(self.directory, "cost.rrd"))
        self.addCleanup(self.recording.disconnect)

    def test_telemetry(self) -> None:
        the_bridge = bridge.RerunBridge(self.recording, None)
        messages = [
            synthetic_messages.full_telemetry(index / TELEMETRY_RATE_HZ, JOINTS, FRAMES)
            for index in range(TELEMETRY_MESSAGES)
        ]
        groups = len(messages[0].groups)
        payloads = [message.SerializeToString() for message in messages]
        flush_every = max(1, round(FLUSH_PERIOD_S * TELEMETRY_RATE_HZ))

        def bridge_one(index: int) -> None:
            the_bridge.handle_telemetry(messages[index])
            if index % flush_every == flush_every - 1:
                the_bridge.flush()

        # Warm up: the first sight of every group logs its SeriesLines.
        for index in range(flush_every):
            bridge_one(index)
        the_bridge.flush()
        self.recording.flush()

        def parse_one(index: int) -> None:
            telemetry_series_pb2.TelemetrySeries.FromString(payloads[index])

        parse_cpu, _ = cpu_seconds_per_call(parse_one, TELEMETRY_MESSAGES)
        cpu, wall = cpu_seconds_per_call(bridge_one, TELEMETRY_MESSAGES)
        flush_start = time.process_time()
        self.recording.flush()
        drain = (time.process_time() - flush_start) / TELEMETRY_MESSAGES
        total = parse_cpu + cpu + drain
        print(
            f"\nviz/telemetry: {groups} groups, {len(payloads[0])} bytes; per message: parse {parse_cpu * 1e6:.0f} us, "
            f"bridge {cpu * 1e6:.0f} us CPU ({wall * 1e6:.0f} us wall on its thread), SDK drain {drain * 1e6:.0f} us; "
            f"total {total * 1e6:.0f} us CPU = {100.0 * total * TELEMETRY_RATE_HZ:.1f} % of a core at "
            f"{TELEMETRY_RATE_HZ:.0f} Hz, flushed every {FLUSH_PERIOD_S * 1000:.0f} ms"
        )
        self.assertEqual(groups, len(telemetry_contract.all_groups(FRAMES)))
        self.assertEqual(the_bridge.statistics.malformed_count(), 0)
        self.assertLess(total, MAX_TELEMETRY_CPU_S)

    def test_scene(self) -> None:
        model = box_robot(LINK_COUNT)
        the_bridge = bridge.RerunBridge(self.recording, model)
        the_bridge.log_static()
        scenes = [
            synthetic_messages.full_scene(0.03 * index, model.links)
            for index in range(SCENES)
        ]
        payload = scenes[0].SerializeToString()
        # Warm up at an earlier time, so that every measured scene moves every link.
        the_bridge.handle_scene(synthetic_messages.full_scene(-0.03, model.links))
        self.recording.flush()

        cpu, wall = cpu_seconds_per_call(
            lambda index: the_bridge.handle_scene(scenes[index]), SCENES
        )
        flush_start = time.process_time()
        self.recording.flush()
        drain = (time.process_time() - flush_start) / SCENES
        total = cpu + drain
        print(
            f"\nviz/scene: {len(scene_contract.ROBOT_INSTANCES)} instances x {LINK_COUNT} links and every marker, "
            f"{len(payload)} bytes; per scene: bridge {cpu * 1e3:.2f} ms CPU ({wall * 1e3:.2f} ms wall), SDK drain "
            f"{drain * 1e3:.2f} ms; total {total * 1e3:.2f} ms CPU = {100.0 * total * 30.0:.1f} % of a core at 30 Hz"
        )
        self.assertEqual(
            the_bridge.statistics.link_poses_logged,
            SCENES * LINK_COUNT * len(scene_contract.ROBOT_INSTANCES)
            + LINK_COUNT * len(scene_contract.ROBOT_INSTANCES),
        )
        self.assertLess(total, MAX_SCENE_CPU_S)


if __name__ == "__main__":
    unittest.main()
