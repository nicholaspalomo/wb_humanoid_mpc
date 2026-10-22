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

"""The bridge's handlers, driven directly with synthetic messages; the recording is read back from an .rrd file."""

import bisect
from collections.abc import Callable
import math
import os
import random
import shutil
import tempfile
import unittest

from humanoid_mpc_msgs import arrows_pb2
from humanoid_mpc_msgs import fsm_state_pb2
from humanoid_mpc_msgs import loop_timing_pb2
from humanoid_mpc_msgs import mpc_status_pb2
from humanoid_mpc_msgs import scalar_group_pb2
from humanoid_mpc_msgs import spheres_pb2
from humanoid_mpc_msgs import telemetry_series_pb2
from humanoid_mpc_msgs import visualization_scene_pb2
import rerun as rr

from humanoid_mpc_ipc import topics
from humanoid_rerun_viewer import bridge
from humanoid_rerun_viewer import palette
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import status_contract
from humanoid_rerun_viewer import telemetry_contract
from humanoid_rerun_viewer import urdf_model
import rrd_contents
import synthetic_messages

ROBOT_TIMELINE = scene_contract.ROBOT_TIMELINE
WALL_TIMELINE = scene_contract.WALL_TIMELINE

WALL_START = 1.7e9
LINKS = synthetic_messages.link_names_with_visuals()


def packed(rgba8) -> int:
    red, green, blue, alpha = rgba8
    return (red << 24) | (green << 16) | (blue << 8) | alpha


def rounded(times: list[float]) -> list[float]:
    """`times` to the microsecond: the recording keeps nanoseconds, which a float in seconds rounds."""
    return [round(time_s, 6) for time_s in times]


def one_group_telemetry(robot_time: float) -> telemetry_series_pb2.TelemetrySeries:
    """A TelemetrySeries of one group, for tests of many messages."""
    return telemetry_series_pb2.TelemetrySeries(
        time=robot_time,
        groups=[
            scalar_group_pb2.ScalarGroup(
                path="base_pose/position_x",
                names=list(telemetry_contract.MEASURED_VS_REFERENCE),
                values=[robot_time, 0.0],
            )
        ],
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = WALL_START

    def __call__(self) -> float:
        self.now += 0.01
        return self.now


class ManualClock:
    """A wall clock that only moves when the test sets it."""

    def __init__(self) -> None:
        self.now = WALL_START

    def __call__(self) -> float:
        return self.now


class BridgeTestCase(unittest.TestCase):
    """A bridge that records into an .rrd file, which contents() reads back; with the sample URDF if `with_model`."""

    with_model = True

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.rrd_path = os.path.join(self.directory, "bridge.rrd")
        self.recording = bridge.new_recording("test_bridge")
        self.recording.save(self.rrd_path)
        self.model: urdf_model.RobotModel | None = None
        if self.with_model:
            self.model = urdf_model.load_urdf(
                synthetic_messages.write_sample_package(self.directory)
            )
        self.bridge = bridge.RerunBridge(self.recording, self.model, clock=FakeClock())
        self._contents: rrd_contents.RrdContents | None = None

    def contents(self) -> rrd_contents.RrdContents:
        if self._contents is None:
            self.bridge.flush()
            self.recording.flush()
            self.recording.disconnect()
            self._contents = rrd_contents.RrdContents(self.rrd_path)
        return self._contents

    def malformed(self, topic: str) -> int:
        return self.bridge.statistics.malformed_count(topic)


class StaticDataTest(BridgeTestCase):
    def test_world_axes_and_every_instance(self) -> None:
        self.bridge.log_static()
        contents = self.contents()
        self.assertIn(
            "ViewCoordinates:xyz", contents.entities["world"].static_components
        )
        assert self.model is not None  # with_model
        for style in scene_contract.ROBOT_INSTANCES:
            for visual in self.model.visuals:
                self.assertIn(
                    scene_contract.visual_path(style.name, visual.link, visual.index),
                    contents.entities,
                )


class SceneTest(BridgeTestCase):
    def test_links_move_on_both_timelines(self) -> None:
        self.bridge.log_static()
        self.bridge.handle_scene(synthetic_messages.full_scene(1.0, LINKS))
        self.bridge.handle_scene(synthetic_messages.full_scene(1.1, LINKS))
        contents = self.contents()
        for style in scene_contract.ROBOT_INSTANCES:
            for link in LINKS:
                entity = contents.entities[scene_contract.link_path(style.name, link)]
                self.assertEqual(entity.static_components, set())
                self.assertIn("Transform3D:translation", entity.temporal_components)
                self.assertIn("Transform3D:quaternion", entity.temporal_components)
                self.assertEqual(entity.timelines, {ROBOT_TIMELINE, WALL_TIMELINE})
                self.assertEqual(entity.times(ROBOT_TIMELINE), [1.0, 1.1])
        # The scene's links without visuals have nothing to move.
        self.assertNotIn(
            scene_contract.link_path(scene_contract.MEASURED, "sensor_mount"),
            contents.entities,
        )
        self.assertEqual(self.bridge.statistics.malformed_count(), 0)

    def test_the_quaternion_is_hamilton_wxyz_on_the_wire_and_xyzw_in_rerun(
        self,
    ) -> None:
        scene = synthetic_messages.full_scene(1.0, LINKS)
        # A rotation of 0.7 rad about the axis (1, 2, 3): four distinct, non-zero components, so that no permutation of
        # them (an x <-> y swap included) passes. Twice that length on the wire: the bridge normalizes.
        half_angle = 0.35
        axis_norm = math.sqrt(14.0)
        expected = (
            math.sin(half_angle) * 1.0 / axis_norm,
            math.sin(half_angle) * 2.0 / axis_norm,
            math.sin(half_angle) * 3.0 / axis_norm,
            math.cos(half_angle),
        )
        self.assertEqual(len({round(component, 3) for component in expected}), 4)
        self.assertTrue(all(abs(component) > 0.05 for component in expected))
        pose = scene.robots[0].link_poses[1]
        scale = 2.0
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = (component * scale for component in expected)
        self.bridge.handle_scene(scene)
        entity = self.contents().entities[
            scene_contract.link_path(scene_contract.MEASURED, LINKS[1])
        ]
        ((quaternion,),) = entity.values("Transform3D:quaternion")
        self.assertEqual(len(quaternion), 4)
        for actual, wanted in zip(quaternion, expected):
            self.assertAlmostEqual(actual, wanted, places=6)
        ((translation,),) = entity.values("Transform3D:translation")
        self.assertAlmostEqual(translation[0], pose.position.x, places=6)

    def test_every_marker_kind_with_its_defaults(self) -> None:
        self.bridge.handle_scene(synthetic_messages.full_scene(1.0, LINKS))
        contents = self.contents()
        for marker in scene_contract.MARKERS:
            path = scene_contract.entity_path(scene_contract.WORLD_ROOT, marker.path)
            entity = contents.entities[path]
            with self.subTest(marker=marker.path):
                if marker.kind is scene_contract.MarkerKind.ARROWS:
                    self.assertEqual(len(entity.values("Arrows3D:vectors")[0]), 2)
                    radii = entity.values("Arrows3D:radii")[0]
                    colors = entity.values("Arrows3D:colors")[0]
                elif marker.kind is scene_contract.MarkerKind.SPHERES:
                    self.assertEqual(len(entity.values("Points3D:positions")[0]), 3)
                    radii = entity.values("Points3D:radii")[0]
                    colors = entity.values("Points3D:colors")[0]
                else:
                    self.assertEqual(len(entity.values("LineStrips3D:strips")[0]), 2)
                    radii = entity.values("LineStrips3D:radii")[0]
                    colors = entity.values("LineStrips3D:colors")[0]
                self.assertEqual(len(radii), 1)
                self.assertAlmostEqual(radii[0], marker.default_radius, places=6)
                self.assertEqual(
                    colors[0], packed(palette.to_rgba8(marker.default_colors[0]))
                )

    def test_the_message_radius_and_colors_win(self) -> None:
        scene = visualization_scene_pb2.VisualizationScene(time=1.0)
        arrows = scene.arrows.add(path="markers/contact_forces", radius=0.03)
        arrows.origins.add()
        arrows.vectors.add(z=1.0)
        arrows.colors.add(r=1.0, g=0.0, b=0.0, a=1.0)
        self.bridge.handle_scene(scene)
        entity = self.contents().entities["world/markers/contact_forces"]
        self.assertAlmostEqual(entity.values("Arrows3D:radii")[0][0], 0.03, places=6)
        self.assertEqual(
            entity.values("Arrows3D:colors")[0], [packed((255, 0, 0, 255))]
        )

    def test_empty_and_missing_markers_are_cleared(self) -> None:
        self.bridge.handle_scene(synthetic_messages.full_scene(1.0, LINKS))
        emptied = visualization_scene_pb2.VisualizationScene(time=1.1)
        emptied.arrows.add(path="markers/contact_forces")
        self.bridge.handle_scene(emptied)
        self.bridge.handle_scene(visualization_scene_pb2.VisualizationScene(time=1.2))
        contents = self.contents()
        self.assertEqual(
            [
                len(vectors)
                for vectors in contents.entities[
                    "world/markers/contact_forces"
                ].sorted_values("Arrows3D:vectors", ROBOT_TIMELINE)
            ],
            [2, 0, 0],
        )
        for marker in scene_contract.MARKERS:
            if marker.path == "markers/contact_forces":
                continue
            entity = contents.entities[
                scene_contract.entity_path(scene_contract.WORLD_ROOT, marker.path)
            ]
            component = {
                scene_contract.MarkerKind.ARROWS: "Arrows3D:vectors",
                scene_contract.MarkerKind.SPHERES: "Points3D:positions",
                scene_contract.MarkerKind.LINE_STRIPS: "LineStrips3D:strips",
            }[marker.kind]
            values = entity.sorted_values(component, ROBOT_TIMELINE)
            self.assertEqual(len(values), 2, marker.path)
            self.assertEqual(values[-1], [], marker.path)

    def test_unchanged_poses_are_skipped_until_the_clock_goes_back(self) -> None:
        links_logged = len(LINKS) * len(scene_contract.ROBOT_INSTANCES)
        scene = synthetic_messages.full_scene(1.0, LINKS)
        self.bridge.handle_scene(scene)
        scene.time = 1.1
        self.bridge.handle_scene(scene)
        statistics = self.bridge.statistics
        self.assertEqual(statistics.link_poses_logged, links_logged)
        self.assertEqual(statistics.link_poses_unchanged, links_logged)
        # The clock went back (by less than a new run's NEW_RUN_REWIND_S): the same poses at an earlier time are all
        # logged again.
        scene.time = 0.5
        self.bridge.handle_scene(scene)
        self.assertEqual(statistics.link_poses_logged, 2 * links_logged)

    def test_scenes_throttled_by_wall_clock(self) -> None:
        clock = FakeClock()
        throttled_bridge = bridge.RerunBridge(
            self.recording, self.model, clock=clock, max_scene_frequency=10.0
        )
        scene = synthetic_messages.full_scene(1.0, LINKS)
        throttled_bridge.handle_scene(scene)
        self.assertEqual(throttled_bridge.statistics.scenes_throttled, 0)

        # Subsequent scenes arriving within 0.1 s wall time are throttled.
        for i in range(5):
            scene.time = 1.0 + (i + 1) * 0.01
            throttled_bridge.handle_scene(scene)
        self.assertEqual(throttled_bridge.statistics.scenes_throttled, 5)

        # Advancing the clock past 0.1 s allows the next scene to be logged.
        clock.now += 0.1
        scene.time = 2.0
        throttled_bridge.handle_scene(scene)
        self.assertEqual(throttled_bridge.statistics.scenes_throttled, 5)

        # Restarted simulation: the clock reset allows the scene even if wall time has not advanced.
        scene.time = 0.5
        throttled_bridge.handle_scene(scene)
        self.assertEqual(throttled_bridge.statistics.scenes_throttled, 5)

    def throttled(self, clock: ManualClock) -> bridge.RerunBridge:
        return bridge.RerunBridge(
            self.recording, self.model, clock=clock, max_scene_frequency=12.5
        )

    def test_the_last_throttled_scene_is_drawn_after_the_interval(self) -> None:
        # The end of a stream (a robot that fell and stopped) must not leave the viewer on an older scene.
        clock = ManualClock()
        throttled_bridge = self.throttled(clock)
        throttled_bridge.handle_scene(synthetic_messages.full_scene(10.0, LINKS))
        clock.now += 0.033
        throttled_bridge.handle_scene(synthetic_messages.full_scene(10.033, LINKS))
        self.assertEqual(throttled_bridge.statistics.scenes_throttled, 1)
        clock.now += 0.1
        throttled_bridge.flush()
        link = scene_contract.link_path(scene_contract.MEASURED, "base_link")
        times = self.contents().entities[link].times(ROBOT_TIMELINE)
        self.assertEqual(len(times), 2)
        self.assertAlmostEqual(times[1], 10.033)

    def test_a_held_back_scene_waits_for_the_interval(self) -> None:
        clock = ManualClock()
        throttled_bridge = self.throttled(clock)
        throttled_bridge.handle_scene(synthetic_messages.full_scene(10.0, LINKS))
        clock.now += 0.033
        throttled_bridge.handle_scene(synthetic_messages.full_scene(10.033, LINKS))
        clock.now += 0.01
        throttled_bridge.flush()
        link = scene_contract.link_path(scene_contract.MEASURED, "base_link")
        self.assertEqual(self.contents().entities[link].times(ROBOT_TIMELINE), [10.0])

    def test_a_newer_scene_drawn_replaces_the_held_back_one(self) -> None:
        clock = ManualClock()
        throttled_bridge = self.throttled(clock)
        throttled_bridge.handle_scene(synthetic_messages.full_scene(10.0, LINKS))
        clock.now += 0.033
        throttled_bridge.handle_scene(synthetic_messages.full_scene(10.033, LINKS))
        clock.now += 0.1
        throttled_bridge.handle_scene(synthetic_messages.full_scene(10.133, LINKS))
        clock.now += 0.1
        throttled_bridge.flush()
        link = scene_contract.link_path(scene_contract.MEASURED, "base_link")
        times = self.contents().entities[link].times(ROBOT_TIMELINE)
        self.assertEqual(len(times), 2)
        self.assertAlmostEqual(times[1], 10.133)

    def test_a_clock_that_goes_back_drops_the_held_back_scene(self) -> None:
        clock = ManualClock()
        throttled_bridge = self.throttled(clock)
        throttled_bridge.handle_scene(synthetic_messages.full_scene(10.0, LINKS))
        clock.now += 0.033
        throttled_bridge.handle_scene(synthetic_messages.full_scene(10.033, LINKS))
        throttled_bridge.handle_scene(synthetic_messages.full_scene(9.5, LINKS))
        clock.now += 0.1
        throttled_bridge.flush()
        link = scene_contract.link_path(scene_contract.MEASURED, "base_link")
        self.assertEqual(
            self.contents().entities[link].times(ROBOT_TIMELINE), [9.5, 10.0]
        )

    def scene_with_links(
        self, links: list[str]
    ) -> visualization_scene_pb2.VisualizationScene:
        scene = visualization_scene_pb2.VisualizationScene(time=1.0)
        scene.robots.append(
            synthetic_messages.robot_instance(scene_contract.MEASURED, links, 1.0)
        )
        return scene

    def test_a_scene_that_names_links_the_urdf_lacks_is_counted(self) -> None:
        # Another robot's URDF (--urdf): its links stay at the world origin, which the operator must be told about.
        with self.assertLogs("humanoid_rerun_viewer", level="WARNING") as logs:
            self.bridge.handle_scene(self.scene_with_links(["pelvis", "torso_link"]))
            self.bridge.handle_scene(self.scene_with_links(["pelvis", "torso_link"]))
        self.assertEqual(self.malformed(topics.VIZ_SCENE), 2)
        (reason,) = {reason for _, reason in self.bridge.statistics.malformed}
        self.assertIn("'measured'", reason)
        self.assertIn("--urdf", reason)
        self.assertEqual(self.bridge.statistics.handler_error_count(), 0)
        unknown = [line for line in logs.output if "pelvis, torso_link" in line]
        self.assertEqual(len(unknown), 1, logs.output)

    def test_visual_links_a_scene_never_names_are_warned_of_once(self) -> None:
        with self.assertLogs("humanoid_rerun_viewer", level="WARNING") as logs:
            self.bridge.handle_scene(self.scene_with_links(["base_link", "ghost"]))
            self.bridge.handle_scene(self.scene_with_links(["base_link", "ghost"]))
        unposed = [line for line in logs.output if "no pose" in line]
        self.assertEqual(len(unposed), 1, logs.output)
        self.assertIn("foot, shin, thigh", unposed[0])
        self.assertIn(
            scene_contract.link_path(scene_contract.MEASURED, "base_link"),
            self.contents().entities,
        )

    def test_max_scene_frequency_must_be_non_negative_and_finite(self) -> None:
        for invalid in (-1.0, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError, "max_scene_frequency must be non-negative and finite"
                ):
                    bridge.RerunBridge(
                        self.recording, self.model, max_scene_frequency=invalid
                    )

    def test_allowed_instances_filters_unwanted_robots(self) -> None:
        filtered_bridge = bridge.RerunBridge(
            self.recording, self.model, allowed_instances=(scene_contract.MEASURED,)
        )
        scene = visualization_scene_pb2.VisualizationScene(time=1.0)
        scene.robots.append(synthetic_messages.robot_instance("measured", LINKS, 1.0))
        scene.robots.append(
            synthetic_messages.robot_instance("terminal_state", LINKS, 1.0)
        )
        filtered_bridge.handle_scene(scene)
        self.assertEqual(filtered_bridge.statistics.link_poses_logged, len(LINKS))

    def test_an_unknown_instance_is_drawn_with_the_default_style(self) -> None:
        scene = visualization_scene_pb2.VisualizationScene(time=1.0)
        scene.robots.append(synthetic_messages.robot_instance("ghost", LINKS, 1.0))
        self.bridge.handle_scene(scene)
        contents = self.contents()
        visual = contents.entities[scene_contract.visual_path("ghost", "base_link", 0)]
        self.assertIn("Asset3D:blob", visual.static_components)
        self.assertIn(scene_contract.link_path("ghost", "base_link"), contents.entities)

    def test_malformed_parts_are_counted_and_the_rest_is_drawn(self) -> None:
        scene = synthetic_messages.full_scene(1.0, LINKS)
        scene.robots[0].link_names.append("extra")  # names and poses differ in length
        scene.robots[1].link_poses[0].orientation.w = 0.0
        scene.robots[1].link_poses[0].orientation.z = 0.0  # a zero quaternion
        scene.arrows.add(path="../escape").vectors.add(z=1.0)
        scene.arrows.add(path="markers/contact_forces")  # the same path twice
        bad = scene.spheres.add(path="markers/odd")
        bad.centers.add()
        bad.radii.extend([0.1, 0.2])  # two radii for one sphere
        self.bridge.handle_scene(scene)
        self.assertEqual(self.malformed(topics.VIZ_SCENE), 5)
        self.assertEqual(self.bridge.statistics.handler_error_count(), 0)
        contents = self.contents()
        self.assertNotIn("world/../escape", contents.entities)
        self.assertNotIn(
            scene_contract.link_path(scene_contract.MEASURED, LINKS[0]),
            contents.entities,
        )
        # The third instance and the valid markers are drawn.
        self.assertIn(
            scene_contract.link_path(scene_contract.TERMINAL_TARGET, LINKS[0]),
            contents.entities,
        )
        self.assertEqual(
            len(contents.values("world/markers/contact_forces", "Arrows3D:vectors")[0]),
            2,
        )
        self.assertEqual(
            contents.values("world/markers/odd", "Points3D:positions"), [[]]
        )

    def test_non_finite_values_are_malformed(self) -> None:
        self.bridge.handle_scene(
            visualization_scene_pb2.VisualizationScene(time=math.nan)
        )
        scene = visualization_scene_pb2.VisualizationScene(time=1.0)
        arrows = scene.arrows.add(path="markers/contact_forces")
        arrows.origins.add(x=math.inf)
        arrows.vectors.add()
        spheres = scene.spheres.add(path="markers/center_of_pressure")
        spheres.centers.add()
        spheres.colors.add(r=1.0)
        spheres.colors.add(r=1.0)  # two colors for one sphere
        self.bridge.handle_scene(scene)
        self.assertEqual(self.malformed(topics.VIZ_SCENE), 3)

    def test_a_handler_never_raises(self) -> None:
        self.bridge.handle_scene(None)
        self.bridge.handle_telemetry(object())
        self.assertEqual(self.bridge.statistics.handler_errors[topics.VIZ_SCENE], 1)
        self.assertEqual(self.bridge.statistics.handler_errors[topics.VIZ_TELEMETRY], 1)


class SceneWithoutModelTest(BridgeTestCase):
    with_model = False

    def test_markers_are_drawn_without_a_robot(self) -> None:
        self.bridge.log_static()
        self.bridge.handle_scene(synthetic_messages.full_scene(1.0, LINKS))
        contents = self.contents()
        self.assertIn("world/markers/contact_forces", contents.entities)
        self.assertFalse(
            any(
                path.startswith(scene_contract.ROBOTS_ROOT)
                for path in contents.entities
            )
        )
        self.assertEqual(self.bridge.statistics.handler_error_count(), 0)


class TelemetryTest(BridgeTestCase):
    with_model = False

    def test_every_group_of_the_contract_is_plotted(self) -> None:
        times = [0.01 * step for step in range(5)]
        for time in times:
            self.bridge.handle_telemetry(synthetic_messages.full_telemetry(time))
        groups = telemetry_contract.all_groups(synthetic_messages.FRAMES)
        self.assertEqual(self.bridge.statistics.telemetry_groups, 5 * len(groups))
        self.assertEqual(self.bridge.telemetry_batcher.pending_frames, 5)
        contents = self.contents()
        for group in groups:
            path = f"{scene_contract.TELEMETRY_ROOT}/{group.path}"
            with self.subTest(group=group.path):
                entity = contents.entities[path]
                self.assertEqual(entity.rows["Scalars:scalars"], 5)
                for actual, expected in zip(entity.times(ROBOT_TIMELINE), times):
                    self.assertAlmostEqual(actual, expected)
                (names,) = entity.values("SeriesLines:names")
                self.assertEqual(
                    tuple(names),
                    group.names or synthetic_messages.robot_dependent_names(group),
                )
                if group.colors:
                    (colors,) = entity.values("SeriesLines:colors")
                    self.assertEqual(
                        colors,
                        [packed(palette.hex_to_rgba8(color)) for color in group.colors],
                    )
                else:
                    self.assertNotIn("SeriesLines:colors", entity.components)
        self.assertEqual(self.bridge.statistics.malformed_count(), 0)

    def group_message(self, time: float, path: str, names, values):
        return telemetry_series_pb2.TelemetrySeries(
            time=time,
            groups=[
                scalar_group_pb2.ScalarGroup(path=path, names=names, values=values)
            ],
        )

    def test_malformed_groups_are_skipped_and_counted(self) -> None:
        self.bridge.handle_telemetry(self.group_message(0.0, "a", ["x", "y"], [1.0]))
        self.bridge.handle_telemetry(self.group_message(0.0, "/a", ["x"], [1.0]))
        self.bridge.handle_telemetry(self.group_message(math.inf, "a", ["x"], [1.0]))
        self.assertEqual(self.malformed(topics.VIZ_TELEMETRY), 3)
        self.assertEqual(self.bridge.statistics.telemetry_groups, 0)
        self.assertNotIn("telemetry/a", self.contents().entities)

    def test_a_declared_group_whose_names_stop_matching_its_values_is_skipped(
        self,
    ) -> None:
        self.bridge.handle_telemetry(
            self.group_message(0.0, "a", ["x", "y"], [1.0, 2.0])
        )
        self.bridge.handle_telemetry(
            self.group_message(0.1, "a", ["x", "y", "z"], [3.0, 4.0])
        )
        self.assertEqual(self.malformed(topics.VIZ_TELEMETRY), 1)
        self.assertEqual(self.bridge.statistics.telemetry_groups, 1)
        # The two messages are one record batch, in which a frame without the group has a null row: no data.
        values = (
            self.contents()
            .entities["telemetry/a"]
            .sorted_values("Scalars:scalars", ROBOT_TIMELINE)
        )
        self.assertEqual([value for value in values if value is not None], [[1.0, 2.0]])

    def test_a_group_twice_in_one_message_is_plotted_once(self) -> None:
        message = self.group_message(0.0, "a", ["x"], [1.0])
        message.groups.append(message.groups[0])
        self.bridge.handle_telemetry(message)
        self.assertEqual(self.malformed(topics.VIZ_TELEMETRY), 1)
        self.assertEqual(
            self.contents().values("telemetry/a", "Scalars:scalars"), [[1.0]]
        )

    def test_a_group_that_changes_size_is_declared_again(self) -> None:
        self.bridge.handle_telemetry(self.group_message(0.0, "a", ["x"], [1.0]))
        self.bridge.handle_telemetry(
            self.group_message(0.1, "a", ["x", "y"], [1.0, 2.0])
        )
        self.bridge.handle_telemetry(
            self.group_message(0.2, "a", ["x", "y"], [3.0, 4.0])
        )
        entity = self.contents().entities["telemetry/a"]
        self.assertIn(["x", "y"], entity.values("SeriesLines:names"))
        self.assertEqual(
            entity.sorted_values("Scalars:scalars", ROBOT_TIMELINE),
            [[1.0], [1.0, 2.0], [3.0, 4.0]],
        )

    def test_a_not_a_number_value_is_plotted_as_a_gap(self) -> None:
        # The display's Euler rates are NaN near a pitch of +-90 degrees: a value, not a malformed group.
        self.bridge.handle_telemetry(
            self.group_message(
                0.0, "base_pose/pitch_rate", ["measured", "reference"], [math.nan, 0.1]
            )
        )
        self.assertEqual(self.bridge.statistics.malformed_count(), 0)
        ((measured, reference),) = self.contents().values(
            "telemetry/base_pose/pitch_rate", "Scalars:scalars"
        )
        self.assertTrue(math.isnan(measured))
        self.assertAlmostEqual(reference, 0.1)

    def test_a_group_outside_the_contract_is_plotted_anyway(self) -> None:
        self.bridge.handle_telemetry(self.group_message(0.0, "custom/x", ["v"], [1.0]))
        self.assertIn("telemetry/custom/x", self.contents().entities)
        self.assertEqual(self.bridge.statistics.malformed_count(), 0)


class StatusTest(BridgeTestCase):
    with_model = False

    def text_lines(self, path: str):
        """(text, level) of every line, in the order the bridge logged them (the fake wall clock only grows)."""
        entity = self.contents().entities[path]
        return list(
            zip(
                [
                    text[0]
                    for text in entity.sorted_values("TextLog:text", WALL_TIMELINE)
                ],
                [
                    level[0]
                    for level in entity.sorted_values("TextLog:level", WALL_TIMELINE)
                ],
            )
        )

    def test_the_fsm_state_is_logged_on_change(self) -> None:
        self.bridge.handle_scene(visualization_scene_pb2.VisualizationScene(time=2.0))
        state = fsm_state_pb2.FsmState(mode="STANCE", mpc_healthy=True)
        self.bridge.handle_fsm_state(state)
        self.bridge.handle_fsm_state(state)
        self.bridge.handle_fsm_state(
            fsm_state_pb2.FsmState(
                mode="WB_MPC", mpc_healthy=False, controller_resets=1
            )
        )
        lines = self.text_lines(status_contract.FSM_STATE_LOG)
        self.assertEqual(len(lines), 2)
        self.assertIn("STANCE", lines[0][0])
        self.assertEqual(lines[0][1], "INFO")
        self.assertIn("UNHEALTHY", lines[1][0])
        self.assertEqual(lines[1][1], "WARN")
        entity = self.contents().entities[status_contract.FSM_STATE_LOG]
        self.assertEqual(entity.times(ROBOT_TIMELINE), [2.0, 2.0])

    def test_the_mpc_status_is_plotted_and_its_changes_logged(self) -> None:
        for step, healthy in enumerate((True, True, False, True)):
            status = mpc_status_pb2.MpcStatus(
                observation_time=0.1 * step,
                resets_served=0 if step < 3 else 1,
                observations_received=4,
                observations_skipped=step,
            )
            status.solver_status.healthy = healthy
            status.solver_status.solve_time_ms = 5.0 + step
            status.solver_status.consecutive_failures = 0 if healthy else 3
            status.solver_status.last_error = "" if healthy else "line search failed"
            self.bridge.handle_mpc_status(status)
        contents = self.contents()
        solve_time = contents.entities[status_contract.MPC_SOLVE_TIME.path]
        self.assertEqual(
            solve_time.sorted_values("Scalars:scalars", ROBOT_TIMELINE),
            [[5.0], [6.0], [7.0], [8.0]],
        )
        for actual, expected in zip(
            solve_time.times(ROBOT_TIMELINE), (0, 0.1, 0.2, 0.3)
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            contents.entities[status_contract.MPC_OBSERVATIONS.path].sorted_values(
                "Scalars:scalars", ROBOT_TIMELINE
            )[-1],
            [4.0, 3.0],
        )
        lines = self.text_lines(status_contract.MPC_STATUS_LOG)
        self.assertEqual(
            [level for _, level in lines], ["INFO", "ERROR", "INFO", "INFO"]
        )
        self.assertIn("line search failed", lines[1][0])
        self.assertIn("served reset 1", lines[3][0])

    def test_the_loop_timing_is_plotted_with_a_gap_without_policy(self) -> None:
        self.bridge.handle_telemetry(synthetic_messages.full_telemetry(3.0))
        self.bridge.handle_loop_timing(
            loop_timing_pb2.LoopTiming(
                target_period_s=0.002,
                mean_period_s=0.002,
                max_period_s=0.0025,
                max_compute_time_s=0.001,
                policy_age_s=-1.0,
            )
        )
        self.bridge.handle_loop_timing(
            loop_timing_pb2.LoopTiming(
                target_period_s=0.002, overruns=2, policy_age_s=0.01
            )
        )
        contents = self.contents()
        period = contents.entities[status_contract.LOOP_PERIOD.path]
        self.assertAlmostEqual(
            period.sorted_values("Scalars:scalars", WALL_TIMELINE)[0][2], 2.5
        )
        self.assertEqual(period.times(ROBOT_TIMELINE), [3.0, 3.0])
        age = contents.entities[status_contract.POLICY_AGE.path].sorted_values(
            "Scalars:scalars", WALL_TIMELINE
        )
        self.assertTrue(math.isnan(age[0][0]))
        self.assertAlmostEqual(age[1][0], 0.01)
        lines = self.text_lines(status_contract.LOOP_TIMING_LOG)
        self.assertEqual([level for _, level in lines], ["DEBUG", "WARN"])
        self.assertIn("no policy", lines[0][0])
        self.assertIn("2 overruns", lines[1][0])

    def test_skipped_periods_and_late_wake_ups_are_plotted_and_warned_of(
        self,
    ) -> None:
        # A loop that loses periods to late wake-ups overruns nothing: only these show it.
        self.bridge.handle_telemetry(synthetic_messages.full_telemetry(3.0))
        self.bridge.handle_loop_timing(
            loop_timing_pb2.LoopTiming(
                target_period_s=0.002,
                missed_periods=3,
                max_lateness_s=0.004,
                policy_age_s=0.01,
            )
        )
        contents = self.contents()
        lateness = contents.entities[status_contract.LOOP_LATENESS.path]
        self.assertAlmostEqual(
            lateness.sorted_values("Scalars:scalars", WALL_TIMELINE)[0][0], 4.0
        )
        events = contents.entities[status_contract.LOOP_EVENTS.path].sorted_values(
            "Scalars:scalars", WALL_TIMELINE
        )
        missed = status_contract.LOOP_EVENTS.names.index("missed_periods")
        self.assertEqual(events[0][missed], 3.0)
        lines = self.text_lines(status_contract.LOOP_TIMING_LOG)
        self.assertEqual([level for _, level in lines], ["WARN"])
        self.assertIn("3 missed periods", lines[0][0])

    def mpc_status(
        self, observation_time: float, failures: int = 0, error: str = ""
    ) -> mpc_status_pb2.MpcStatus:
        status = mpc_status_pb2.MpcStatus(observation_time=observation_time)
        status.solver_status.healthy = True
        status.solver_status.consecutive_failures = failures
        status.solver_status.last_error = error
        return status

    def test_the_fsm_state_and_the_loop_timing_take_a_preceding_mpc_status_time(
        self,
    ) -> None:
        # MpcStatus.observation_time is a robot time the bridge has seen; before any scene, the rows take it.
        self.bridge.handle_mpc_status(self.mpc_status(3.5))
        self.bridge.handle_fsm_state(fsm_state_pb2.FsmState(mode="STANCE"))
        self.bridge.handle_loop_timing(loop_timing_pb2.LoopTiming(policy_age_s=0.01))
        contents = self.contents()
        for path in (status_contract.FSM_STATE_LOG, status_contract.LOOP_PERIOD.path):
            with self.subTest(path=path):
                entity = contents.entities[path]
                self.assertEqual(entity.timelines, {ROBOT_TIMELINE, WALL_TIMELINE})
                self.assertEqual(entity.times(ROBOT_TIMELINE), [3.5])

    def test_an_older_mpc_status_does_not_pull_the_latest_robot_time_back(
        self,
    ) -> None:
        # The observation a solve started from trails the scene by the solve.
        self.bridge.handle_scene(visualization_scene_pb2.VisualizationScene(time=4.0))
        self.bridge.handle_mpc_status(self.mpc_status(3.9))
        self.bridge.handle_loop_timing(loop_timing_pb2.LoopTiming(policy_age_s=0.01))
        entity = self.contents().entities[status_contract.LOOP_PERIOD.path]
        self.assertEqual(entity.times(ROBOT_TIMELINE), [4.0])

    def test_an_mpc_status_of_a_restarted_clock_is_followed(self) -> None:
        self.bridge.handle_mpc_status(self.mpc_status(100.0))
        self.bridge.handle_mpc_status(self.mpc_status(0.5))
        self.bridge.handle_fsm_state(fsm_state_pb2.FsmState(mode="STANCE"))
        entity = self.contents().entities[status_contract.FSM_STATE_LOG]
        self.assertEqual(entity.times(ROBOT_TIMELINE), [0.5])

    def test_a_failed_solve_while_still_healthy_is_warned_with_its_error(
        self,
    ) -> None:
        # The supervisor calls the solver unhealthy only after several consecutive failures; the ones before it are
        # failures all the same.
        for status in (
            self.mpc_status(0.0),
            self.mpc_status(0.1, failures=1, error="MPC solver crashed at t = 1: QP"),
            self.mpc_status(0.2),
            self.mpc_status(0.3),
        ):
            self.bridge.handle_mpc_status(status)
        lines = self.text_lines(status_contract.MPC_STATUS_LOG)
        self.assertEqual([level for _, level in lines], ["INFO", "WARN", "INFO"])
        self.assertIn("crashed at t = 1: QP", lines[1][0])
        self.assertIn("1 in a row", lines[1][0])

    def loop_timing(self, cycles: int, overruns: int = 0, stale: int = 0):
        return loop_timing_pb2.LoopTiming(
            cycles=cycles,
            overruns=overruns,
            stale_policies_dropped=stale,
            target_period_s=0.002,
            policy_age_s=0.01,
        )

    def test_only_new_loop_events_are_warned_of(self) -> None:
        # LoopTiming counts since the robot started; the last report is from a restarted robot process.
        for timing in (
            self.loop_timing(500, overruns=1),
            self.loop_timing(1000, overruns=1),
            self.loop_timing(1500, overruns=2),
            self.loop_timing(200, overruns=1),
        ):
            self.bridge.handle_loop_timing(timing)
        lines = self.text_lines(status_contract.LOOP_TIMING_LOG)
        self.assertEqual(
            [level for _, level in lines], ["WARN", "DEBUG", "WARN", "WARN"]
        )
        self.assertIn("+0 overruns", lines[1][0])
        self.assertIn("+1 overruns", lines[2][0])
        # The plot keeps the totals.
        events = (
            self.contents()
            .entities[status_contract.LOOP_EVENTS.path]
            .sorted_values("Scalars:scalars", WALL_TIMELINE)
        )
        overruns = status_contract.LOOP_EVENTS.names.index("overruns")
        self.assertEqual([row[overruns] for row in events], [1.0, 1.0, 2.0, 1.0])

    def test_a_stale_policy_is_warned_of_once(self) -> None:
        self.bridge.handle_loop_timing(self.loop_timing(500, stale=1))
        self.bridge.handle_loop_timing(self.loop_timing(1000, stale=1))
        lines = self.text_lines(status_contract.LOOP_TIMING_LOG)
        self.assertEqual([level for _, level in lines], ["WARN", "DEBUG"])

    def test_status_before_any_robot_time_has_the_wall_clock_only(self) -> None:
        self.bridge.handle_fsm_state(fsm_state_pb2.FsmState(mode="SAFETY"))
        entity = self.contents().entities[status_contract.FSM_STATE_LOG]
        self.assertEqual(entity.timelines, {WALL_TIMELINE})

    def test_reports_name_new_problems_once(self) -> None:
        self.assertIsNone(self.bridge.report())
        self.bridge.handle_telemetry(
            telemetry_series_pb2.TelemetrySeries(
                time=0.0,
                groups=[scalar_group_pb2.ScalarGroup(path="a", names=["x"], values=[])],
            )
        )
        text = self.bridge.report(bus_rejected=2)
        assert text is not None
        self.assertIn("1 malformed", text)
        self.assertIn("2 rejected by the bus", text)
        self.assertIsNone(self.bridge.report(bus_rejected=2))
        (line,) = self.text_lines(status_contract.BRIDGE_LOG)
        self.assertEqual(line[1], "WARN")


class RunTest(unittest.TestCase):
    """A robot clock that goes back is a new run of the robot, which the bridge records as a recording of its own."""

    def setUp(self) -> None:
        self.directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.directory)
        self.paths: list[str] = []
        self.recordings: list[rr.RecordingStream] = []
        self.runs_started = 0
        model = urdf_model.load_urdf(
            synthetic_messages.write_sample_package(self.directory)
        )
        self.bridge = bridge.RerunBridge(
            self.new_recording(), model, clock=FakeClock(), new_run=self.next_run
        )
        self._contents: list[rrd_contents.RrdContents] | None = None

    def new_recording(self) -> rr.RecordingStream:
        path = os.path.join(self.directory, f"run{len(self.paths) + 1}.rrd")
        recording = bridge.new_recording("test_bridge_runs")
        recording.save(path)
        self.paths.append(path)
        self.recordings.append(recording)
        return recording

    def next_run(self) -> rr.RecordingStream:
        self.runs_started += 1
        return self.new_recording()

    def contents(self) -> list[rrd_contents.RrdContents]:
        if self._contents is None:
            self.bridge.flush()
            for recording in self.recordings:
                recording.flush()
                recording.disconnect()
            self._contents = [rrd_contents.RrdContents(path) for path in self.paths]
        return self._contents

    def step(self, robot_time: float) -> None:
        self.bridge.handle_scene(synthetic_messages.full_scene(robot_time, LINKS))
        self.bridge.handle_telemetry(synthetic_messages.full_telemetry(robot_time))

    def test_a_clock_that_goes_back_starts_a_new_recording(self) -> None:
        self.bridge.log_static()
        # A restarted simulation: its clock starts over and goes on.
        for robot_time in (0.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 1.5):
            self.step(robot_time)
        self.assertEqual(self.runs_started, 1)
        self.assertEqual(self.bridge.statistics.new_runs, 1)
        self.assertEqual(self.bridge.statistics.late_messages, 0)
        self.assertIs(self.bridge.recording, self.recordings[1])
        first, second = self.contents()
        link = scene_contract.link_path(scene_contract.MEASURED, "base_link")
        plot = f"{scene_contract.TELEMETRY_ROOT}/base_pose/position_x"
        for entity in (link, plot):
            self.assertEqual(
                first.entities[entity].times(ROBOT_TIMELINE), [0.0, 1.0, 2.0, 3.0]
            )
            # The messages that waited for the clock to confirm the new run are in it, in order.
            self.assertEqual(
                second.entities[entity].times(ROBOT_TIMELINE), [0.5, 0.6, 0.7, 1.5]
            )
        # The new recording holds everything a viewer needs: the world's axes, every instance's meshes, the series.
        self.assertIn("ViewCoordinates:xyz", second.entities["world"].static_components)
        for style in scene_contract.ROBOT_INSTANCES:
            self.assertTrue(
                second.has(
                    scene_contract.visual_path(style.name, "base_link", 0),
                    "Asset3D:blob",
                )
            )
        self.assertTrue(second.has(plot, "SeriesLines:names"))
        texts = [
            row[0]
            for row in second.entities[status_contract.BRIDGE_LOG].values(
                "TextLog:text"
            )
        ]
        self.assertTrue(
            any("went back from 3.000 s to 0.500 s" in text for text in texts), texts
        )

    def test_a_small_step_back_stays_in_the_run(self) -> None:
        # The scene falls back from the sample's time to the older observation's when the samples stop.
        for robot_time in (2.0, 1.5, 2.1):
            self.step(robot_time)
        self.assertEqual(self.runs_started, 0)
        (only,) = self.contents()
        link = scene_contract.link_path(scene_contract.MEASURED, "base_link")
        self.assertEqual(only.entities[link].times(ROBOT_TIMELINE), [1.5, 2.0, 2.1])

    def test_late_messages_of_other_topics_never_start_a_run(self) -> None:
        # What the live session showed: an MpcStatus whose solve took longer than NEW_RUN_REWIND_S, and a scene of
        # "latest" delivery that overtook older telemetry, each behind the robot's clock by more than it.
        times = [index / 100 for index in range(301)]
        for robot_time in times:
            self.step(robot_time)
        self.bridge.handle_mpc_status(mpc_status_pb2.MpcStatus(observation_time=1.5))
        self.bridge.handle_scene(synthetic_messages.full_scene(1.6, LINKS))
        self.bridge.handle_mpc_status(mpc_status_pb2.MpcStatus(observation_time=1.7))
        self.assertEqual(self.bridge.statistics.handled[topics.MPC_STATUS], 2)
        self.step(3.01)
        self.assertEqual(self.runs_started, 0)
        self.assertEqual(self.bridge.statistics.new_runs, 0)
        self.assertEqual(self.bridge.statistics.late_messages, 3)
        self.assertEqual(self.bridge.statistics.previous_run_dropped, 0)
        # Each handled once, as it came.
        self.assertEqual(self.bridge.statistics.handled[topics.MPC_STATUS], 2)
        (only,) = self.contents()
        # Recorded at their own times, before the message that showed them late.
        self.assertEqual(
            rounded(
                only.entities[status_contract.MPC_SOLVE_TIME.path].times(ROBOT_TIMELINE)
            ),
            [1.5, 1.7],
        )
        link = scene_contract.link_path(scene_contract.MEASURED, "base_link")
        self.assertEqual(
            rounded(only.entities[link].times(ROBOT_TIMELINE)),
            rounded(sorted([*times, 1.6, 3.01])),
        )
        plot = f"{scene_contract.TELEMETRY_ROOT}/base_pose/position_x"
        self.assertEqual(
            rounded(only.entities[plot].times(ROBOT_TIMELINE)), rounded([*times, 3.01])
        )

    def test_fewer_clock_messages_than_the_confirmations_are_late(self) -> None:
        for robot_time in (0.0, 1.0, 2.0, 3.0):
            self.step(robot_time)
        late = [0.5 + index / 100 for index in range(bridge.NEW_RUN_CONFIRMATIONS - 1)]
        for robot_time in late:
            self.bridge.handle_telemetry(synthetic_messages.full_telemetry(robot_time))
        self.bridge.handle_telemetry(synthetic_messages.full_telemetry(3.01))
        self.assertEqual(self.runs_started, 0)
        self.assertEqual(self.bridge.statistics.late_messages, len(late))
        (only,) = self.contents()
        plot = f"{scene_contract.TELEMETRY_ROOT}/base_pose/position_x"
        self.assertEqual(
            rounded(only.entities[plot].times(ROBOT_TIMELINE)),
            rounded(sorted([0.0, 1.0, 2.0, 3.0, 3.01, *late])),
        )

    def test_the_scene_is_the_clock_without_telemetry(self) -> None:
        for robot_time in (0.0, 1.0, 2.0, 3.0, 0.5, 0.6, 0.7):
            self.bridge.handle_scene(synthetic_messages.full_scene(robot_time, LINKS))
        self.assertEqual(self.runs_started, 1)
        _, second = self.contents()
        link = scene_contract.link_path(scene_contract.MEASURED, "base_link")
        self.assertEqual(second.entities[link].times(ROBOT_TIMELINE), [0.5, 0.6, 0.7])

    def test_a_message_of_the_previous_run_after_the_new_one_began_is_dropped(
        self,
    ) -> None:
        for robot_time in (0.0, 1.0, 2.0, 3.0):
            self.step(robot_time)
        restarted = [0.5 + index / 100 for index in range(bridge.NEW_RUN_CONFIRMATIONS)]
        for robot_time in restarted:
            self.bridge.handle_telemetry(synthetic_messages.full_telemetry(robot_time))
        self.assertEqual(self.runs_started, 1)
        # Still on its way when the robot restarted: telemetry and a status of the previous run.
        self.bridge.handle_telemetry(synthetic_messages.full_telemetry(3.01))
        self.bridge.handle_mpc_status(mpc_status_pb2.MpcStatus(observation_time=2.99))
        self.bridge.handle_telemetry(synthetic_messages.full_telemetry(0.6))
        self.assertEqual(self.runs_started, 1)
        self.assertEqual(self.bridge.statistics.previous_run_dropped, 2)
        _, second = self.contents()
        plot = f"{scene_contract.TELEMETRY_ROOT}/base_pose/position_x"
        self.assertEqual(
            rounded(second.entities[plot].times(ROBOT_TIMELINE)),
            rounded([*restarted, 0.6]),
        )
        self.assertNotIn(status_contract.MPC_SOLVE_TIME.path, second.entities)

    def test_messages_wait_for_a_quiet_clock_at_most_the_clock_timeout(self) -> None:
        clock = ManualClock()
        the_bridge = bridge.RerunBridge(
            self.new_recording(), None, clock=clock, new_run=self.next_run
        )
        self.bridge = the_bridge
        the_bridge.handle_telemetry(synthetic_messages.full_telemetry(3.0))
        the_bridge.handle_mpc_status(mpc_status_pb2.MpcStatus(observation_time=1.0))
        the_bridge.flush()
        self.assertEqual(the_bridge.statistics.late_messages, 0)
        clock.now += bridge.CLOCK_TIMEOUT_S
        the_bridge.flush()
        self.assertEqual(the_bridge.statistics.late_messages, 1)
        self.assertEqual(self.runs_started, 0)
        self.assertEqual(
            rounded(
                self.contents()[1]
                .entities[status_contract.MPC_SOLVE_TIME.path]
                .times(ROBOT_TIMELINE)
            ),
            [1.0],
        )

    def test_the_messages_that_wait_are_bounded(self) -> None:
        # Messages behind a clock topic that no longer comes: no more than MAX_HELD_MESSAGES wait at any time.
        self.step(3.0)
        count = 10 * bridge.MAX_HELD_MESSAGES
        for index in range(count):
            self.bridge.handle_mpc_status(
                mpc_status_pb2.MpcStatus(observation_time=1.0 + 1e-3 * index)
            )
            self.assertLessEqual(
                len(self.bridge._held),
                bridge.MAX_HELD_MESSAGES,  # pylint: disable=protected-access  # The bound.
            )
        self.bridge.draw_pending_scene()
        self.assertEqual(self.bridge.statistics.late_messages, count)
        self.assertEqual(self.runs_started, 0)

    def test_periodic_late_messages_over_a_long_stream_leave_one_recording(
        self,
    ) -> None:
        # Ten minutes of 100 Hz telemetry with a status 1.5 s late every 35 s, the live session's rate of new
        # recordings before the fix: nothing accumulates.
        rate = 100
        duration = 600
        late = range(rate * 30, rate * duration, rate * 35)
        for index in range(rate * duration):
            robot_time = index / rate
            self.bridge.handle_telemetry(one_group_telemetry(robot_time))
            if index in late:
                self.bridge.handle_mpc_status(
                    mpc_status_pb2.MpcStatus(observation_time=robot_time - 1.5)
                )
            if index % 20 == 0:
                self.bridge.flush()
        self.assertEqual(self.runs_started, 0)
        self.assertEqual(len(self.recordings), 1)
        self.assertEqual(self.bridge.statistics.late_messages, len(late))
        self.assertEqual(self.bridge.statistics.previous_run_dropped, 0)
        self.assertEqual(
            len(self.bridge._held), 0
        )  # pylint: disable=protected-access  # Nothing left waiting.


def sample_times(source_rate: float, rate: float, seed: int) -> list[float]:
    """Three seconds of increasing sample times at `source_rate` with jitter, none within 1e-4 of an interval's start."""
    generator = random.Random(seed)
    times: list[float] = []
    for index in range(int(3 * source_rate)):
        while True:
            time_s = (index + generator.uniform(-0.3, 0.3)) / source_rate
            offset = time_s * rate - math.floor(time_s * rate)
            if 1e-4 < offset < 1.0 - 1e-4:
                break
        times.append(time_s)
    return times


class TelemetryRecordingTest(unittest.TestCase):
    """telemetry_rate records the first sample of every 1 / rate interval of robot time; telemetry_selection the groups."""

    PLOT = f"{scene_contract.TELEMETRY_ROOT}/base_pose/position_x"

    def record(
        self,
        messages: list[telemetry_series_pb2.TelemetrySeries],
        rate: float = 0.0,
        selection: Callable[[str], bool] | None = None,
    ) -> tuple[bridge.RerunBridge, rrd_contents.RrdContents]:
        """Feeds `messages` to a bridge of `rate` and `selection`; returns it and its recording, read back."""
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        path = os.path.join(directory, "telemetry.rrd")
        recording = bridge.new_recording("test_bridge_telemetry")
        recording.save(path)
        the_bridge = bridge.RerunBridge(
            recording,
            None,
            clock=FakeClock(),
            telemetry_rate=rate,
            telemetry_selection=selection,
        )
        for message in messages:
            the_bridge.handle_telemetry(message)
        the_bridge.flush()
        recording.flush()
        recording.disconnect()
        return the_bridge, rrd_contents.RrdContents(path)

    def test_every_interval_records_its_first_sample_and_only_that(self) -> None:
        for source_rate, rate in (
            (100.0, 50.0),
            (100.0, 30.0),
            (250.0, 7.5),
            (33.0, 50.0),
        ):
            with self.subTest(source_rate=source_rate, rate=rate):
                times = sample_times(source_rate, rate, seed=int(source_rate * rate))
                the_bridge, contents = self.record(
                    [one_group_telemetry(time_s) for time_s in times], rate=rate
                )
                # The oracle: the interval [k / rate, (k + 1) / rate) of each sample, by bisection on the boundaries.
                boundaries = [index / rate for index in range(int(4 * rate) + 2)]
                first_of_interval: dict[int, float] = {}
                for time_s in times:
                    interval = bisect.bisect_right(boundaries, time_s) - 1
                    first_of_interval.setdefault(interval, time_s)
                expected = sorted(first_of_interval.values())
                self.assertEqual(
                    rounded(contents.entities[self.PLOT].times(ROBOT_TIMELINE)),
                    rounded(expected),
                )
                self.assertEqual(
                    the_bridge.statistics.telemetry_decimated,
                    len(times) - len(expected),
                )
                # Never more than the rate, and every interval with a sample has one: the mean rate is the lesser.
                self.assertLessEqual(len(expected), math.ceil(3 * rate) + 1)

    def test_rate_zero_records_every_sample(self) -> None:
        times = [index / 100 for index in range(100)]
        the_bridge, contents = self.record(
            [one_group_telemetry(time_s) for time_s in times]
        )
        self.assertEqual(
            rounded(contents.entities[self.PLOT].times(ROBOT_TIMELINE)), rounded(times)
        )
        self.assertEqual(the_bridge.statistics.telemetry_decimated, 0)

    def test_a_rate_must_be_finite_and_not_negative(self) -> None:
        for rate in (-1.0, math.nan, math.inf):
            with self.subTest(rate=rate):
                with self.assertRaises(ValueError):
                    bridge.RerunBridge(
                        bridge.new_recording("test_bridge_rate"),
                        None,
                        telemetry_rate=rate,
                    )

    def test_only_the_selected_groups_are_recorded(self) -> None:
        asked: list[str] = []

        def selection(path: str) -> bool:
            asked.append(path)
            return path.startswith("base_pose/")

        messages = [
            synthetic_messages.full_telemetry(index / 100) for index in range(5)
        ]
        _, contents = self.record(messages, selection=selection)
        recorded = {
            entity
            for entity in contents.entities
            if entity.startswith(f"{scene_contract.TELEMETRY_ROOT}/")
        }
        expected = {
            f"{scene_contract.TELEMETRY_ROOT}/{group.path}"
            for group in messages[0].groups
            if group.path.startswith("base_pose/")
        }
        self.assertTrue(expected)
        self.assertEqual(recorded, expected)
        # Asked once per group, not once per message.
        self.assertEqual(len(asked), len(messages[0].groups))


class MarkerHelpersTest(unittest.TestCase):
    def test_default_colors_cycle_over_the_elements(self) -> None:
        colors = bridge._defaults_rgba8(
            [
                palette.with_alpha(palette.RED, 1.0),
                palette.with_alpha(palette.BLUE, 1.0),
            ],
            3,
        )
        self.assertEqual(colors.shape, (3, 4))
        self.assertEqual(list(colors[0]), list(colors[2]))

    def test_message_colors_are_clamped(self) -> None:
        arrows = arrows_pb2.Arrows()
        arrows.colors.add(r=2.0, g=-1.0, b=0.5, a=1.0)
        rgba = bridge._colors_rgba8(arrows.colors)
        self.assertEqual(list(rgba[0]), [255, 0, 128, 255])

    def test_spheres_take_one_radius_or_one_each(self) -> None:
        spheres = spheres_pb2.Spheres(path="plan/footholds")
        spheres.centers.add()
        spheres.centers.add()
        spheres.radii.append(0.1)
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        recording = bridge.new_recording("test_bridge_spheres")
        recording.save(os.path.join(directory, "spheres.rrd"))
        the_bridge = bridge.RerunBridge(recording, None)
        scene = visualization_scene_pb2.VisualizationScene(time=0.0)
        scene.spheres.append(spheres)
        the_bridge.handle_scene(scene)
        self.assertEqual(the_bridge.statistics.malformed_count(), 0)
        recording.disconnect()


if __name__ == "__main__":
    unittest.main()
