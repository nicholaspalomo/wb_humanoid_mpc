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

import math
import os
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


class FakeClock:
    def __init__(self) -> None:
        self.now = WALL_START

    def __call__(self) -> float:
        self.now += 0.01
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
        self.bridge.handle_scene(scene)
        pose = scene.robots[0].link_poses[1]
        entity = self.contents().entities[
            scene_contract.link_path(scene_contract.MEASURED, LINKS[1])
        ]
        ((quaternion,),) = entity.values("Transform3D:quaternion")
        for actual, expected in zip(
            quaternion,
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ),
        ):
            self.assertAlmostEqual(actual, expected, places=6)
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
        # A restarted simulation: the same poses at an earlier time are all logged again.
        scene.time = 0.1
        self.bridge.handle_scene(scene)
        self.assertEqual(statistics.link_poses_logged, 2 * links_logged)

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
