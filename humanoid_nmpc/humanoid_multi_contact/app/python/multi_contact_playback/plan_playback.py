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

"""Plays a multi-contact plan kinematically in Rerun, with no MPC (design section 8.2, "plan playback").

    bazel run //humanoid_nmpc/humanoid_multi_contact/app:plan_playback -- \\
        --scenario humanoid_nmpc/humanoid_multi_contact/scenarios/config/box_climb.textproto \\
        --plan <plan.binpb or plan.textproto> [--rerun_sink spawn|connect|serve_web|save] [--rrd_path <file.rrd>]

It draws, on the timeline `plan_time`:
- the scenario's terrain (world/terrain/<primitive>), statically;
- the robot at every knot (world/robots/plan/<link>), from the URDF's forward kinematics and the knot's base pose;
- the active pads (world/plan/active_pads) and the planned contact forces as arrows (world/plan/contact_forces);
- the CoM and its ground projection, and the static equilibrium polygon (world/plan/com, world/plan/support_polygon)
  and the margins (margins/...), from the certifier's report when the plan carries one (certify_plan writes it);
- the certifier's failures as text (plan/failures).

The plan's hashes are not checked here: certify_plan refuses a plan of another contact set or scene. Exit status: 0, or
2 for a usage or input error.
"""

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import logging
import sys
import threading

from humanoid_multi_contact.contact_set import contact_set_pb2
from humanoid_multi_contact.plan import multi_contact_plan_pb2
from humanoid_multi_contact.scenarios import scenario_pb2
from humanoid_multi_contact.stability import certifier_report_pb2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from humanoid_rerun_viewer import cli
from humanoid_rerun_viewer import palette
from humanoid_rerun_viewer import rerun_sinks
from humanoid_rerun_viewer import robot_meshes
from humanoid_rerun_viewer import scene_contract
from humanoid_rerun_viewer import terrain_drawing
from humanoid_rerun_viewer import urdf_kinematics
from humanoid_rerun_viewer import urdf_model
import nproto_textproto

_LOGGER = logging.getLogger("plan_playback")

EXIT_OK = 0
EXIT_USAGE = 2

PLAN_TIMELINE = "plan_time"
ROBOT_INSTANCE = "plan"
TERRAIN_ROOT = terrain_drawing.TERRAIN_ROOT
PLAN_ROOT = "world/plan"
MARGINS_ROOT = "margins"
FAILURES_PATH = "plan/failures"
# The suffix of a binary plan (PlanFile.h kPlanBinarySuffix).
# LINT.IfChange(plan_binary_suffix)
PLAN_BINARY_SUFFIX = ".binpb"
# LINT.ThenChange(//humanoid_nmpc/humanoid_multi_contact/plan/include/humanoid_multi_contact/plan/PlanFile.h:plan_binary_suffix)
# Pinocchio's name of the floating base's joint, which a contact candidate may name as its parent.
ROOT_JOINT = "root_joint"
# [m/N] The length of a force arrow per newton: a body weight of Atlas is about 1.6 m.
FORCE_ARROW_SCALE = 0.001
_PAD_RADIUS = 0.02
_COM_RADIUS = 0.03
_LINE_RADIUS = 0.005
_ROBOT_STYLE = scene_contract.RobotInstanceStyle(
    name=ROBOT_INSTANCE, tint=None, alpha=0.8, visible_by_default=True
)
# The viewer of `spawn` and `connect` keeps the data when the playback exits; `serve_web` serves it from this
# process, which therefore waits for Ctrl-C.
_SERVING_SINKS = frozenset({"serve_web"})


@dataclasses.dataclass
class PlaybackInputs:
    """What a playback draws.

    Attributes:
        scenario: The scenario the plan was made for.
        contact_set: The robot's contact candidates.
        plan: The plan, with or without the certifier's report.
        tree: The URDF's kinematics.
        model: The URDF's visual geometry.
    """

    scenario: scenario_pb2.Scenario
    contact_set: contact_set_pb2.ContactSet
    plan: multi_contact_plan_pb2.MultiContactPlan
    tree: urdf_kinematics.KinematicTree
    model: urdf_model.RobotModel


def load_plan(path: str) -> multi_contact_plan_pb2.MultiContactPlan:
    """Reads a plan: binary when the path ends in PLAN_BINARY_SUFFIX, else a textproto.

    Args:
        path: The plan file.

    Returns:
        The plan.

    Raises:
        OSError: the file cannot be read.
        ValueError: it does not parse.
    """
    if path.endswith(PLAN_BINARY_SUFFIX):
        plan = multi_contact_plan_pb2.MultiContactPlan()
        with open(path, "rb") as file:
            plan.ParseFromString(file.read())
        return plan
    return nproto_textproto.load_textproto(
        path, multi_contact_plan_pb2.MultiContactPlan
    )


def load_inputs(scenario_path: str, plan_path: str) -> PlaybackInputs:
    """Reads the scenario, its contact set and URDF, and the plan.

    Args:
        scenario_path: The scenario textproto.
        plan_path: The plan, binary or textproto.

    Returns:
        The inputs of a playback.

    Raises:
        OSError: a file cannot be read.
        ValueError: a file does not parse, or the plan names a contact the set does not have.
    """
    scenario = nproto_textproto.load_textproto(
        cli.resolve_input_path(scenario_path), scenario_pb2.Scenario
    )
    contact_set = nproto_textproto.load_textproto(
        cli.resolve_input_path(scenario.contact_set_file),
        contact_set_pb2.ContactSet,
    )
    urdf = cli.resolve_input_path(scenario.urdf_file)
    plan = load_plan(cli.resolve_input_path(plan_path))
    names = {candidate.name for candidate in contact_set.candidates}
    for knot in plan.knots:
        for wrench in knot.contact_wrenches:
            if wrench.contact not in names:
                raise ValueError(
                    f"plan knot at t={knot.time}: '{wrench.contact}' is no"
                    " candidate of the contact set"
                )
    return PlaybackInputs(
        scenario=scenario,
        contact_set=contact_set,
        plan=plan,
        tree=urdf_kinematics.load_kinematic_tree(urdf),
        model=cli.load_model(urdf, ()),
    )


def rotation_from_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    """The rotation matrix of a quaternion (x, y, z, w), normalized first.

    Args:
        quaternion: Four coefficients, not all zero.

    Returns:
        A 3 x 3 rotation matrix.
    """
    x, y, z, w = np.asarray(quaternion, dtype=float) / np.linalg.norm(
        quaternion
    )
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# The terrain drawing is the Rerun bridge's, so that a plan's playback and a dummy sim draw the same scene.
quaternion_between = terrain_drawing.quaternion_between
log_terrain = terrain_drawing.log_terrain


def _joint_children(tree: urdf_kinematics.KinematicTree) -> dict[str, str]:
    """The link each joint moves, and the base link for ROOT_JOINT."""
    children = {joint.name: joint.child for joint in tree.joints}
    children[ROOT_JOINT] = tree.root_link
    return children


def _world_link_poses(
    inputs: PlaybackInputs, knot: multi_contact_plan_pb2.MultiContactPlan.Knot
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The (translation, rotation) of every link in the world at `knot`."""
    positions: Mapping[str, float] = dict(
        zip(inputs.plan.joint_names, knot.joint_positions)
    )
    base_rotation = rotation_from_xyzw(knot.base_orientation_xyzw)
    base_position = np.asarray(knot.base_position, dtype=float)
    poses = {}
    for link, (translation, quaternion) in inputs.tree.link_poses(
        positions
    ).items():
        poses[link] = (
            base_position + base_rotation @ np.asarray(translation),
            base_rotation @ rotation_from_xyzw(quaternion),
        )
    return poses


def pad_positions(
    inputs: PlaybackInputs, knot: multi_contact_plan_pb2.MultiContactPlan.Knot
) -> dict[str, np.ndarray]:
    """The world position of every candidate's pad at `knot`.

    Args:
        inputs: The playback's inputs.
        knot: One knot of the plan.

    Returns:
        The pad centers, by candidate name.

    Raises:
        ValueError: a candidate's parent joint is not a joint of the URDF.
    """
    children = _joint_children(inputs.tree)
    poses = _world_link_poses(inputs, knot)
    pads = {}
    for candidate in inputs.contact_set.candidates:
        link = children.get(candidate.parent_joint)
        if link is None:
            raise ValueError(
                f"contact candidate '{candidate.name}': the URDF has no joint"
                f" '{candidate.parent_joint}'"
            )
        translation, rotation = poses[link]
        pads[candidate.name] = translation + rotation @ np.asarray(
            candidate.offset, dtype=float
        )
    return pads


def _log_knot(
    recording: rr.RecordingStream,
    inputs: PlaybackInputs,
    index: int,
) -> None:
    """Logs knot `index` of the plan at its time."""
    knot = inputs.plan.knots[index]
    recording.set_time(PLAN_TIMELINE, duration=knot.time)
    for link, (translation, rotation) in _world_link_poses(
        inputs, knot
    ).items():
        recording.log(
            scene_contract.link_path(ROBOT_INSTANCE, link),
            rr.Transform3D(
                translation=translation.tolist(),
                mat3x3=rotation.tolist(),
            ),
        )
    pads = pad_positions(inputs, knot)
    active = [
        candidate.name
        for bit, candidate in enumerate(inputs.contact_set.candidates)
        if knot.active_contacts >> bit & 1
    ]
    recording.log(
        f"{PLAN_ROOT}/active_pads",
        rr.Points3D(
            positions=[pads[name].tolist() for name in active],
            radii=[_PAD_RADIUS],
            labels=active,
            colors=[palette.to_rgba8(palette.with_alpha(palette.GREEN, 1.0))],
        ),
    )
    recording.log(
        f"{PLAN_ROOT}/contact_forces",
        rr.Arrows3D(
            origins=[pads[w.contact].tolist() for w in knot.contact_wrenches],
            vectors=[
                (FORCE_ARROW_SCALE * np.asarray(w.force, dtype=float)).tolist()
                for w in knot.contact_wrenches
            ],
            radii=[scene_contract.ARROW_RADIUS],
            colors=[palette.to_rgba8(palette.with_alpha(palette.ORANGE, 1.0))],
        ),
    )
    if inputs.plan.HasField("report") and index < len(inputs.plan.report.knots):
        floor = min((pads[name][2] for name in active), default=0.0)
        _log_report_knot(recording, inputs.plan.report.knots[index], floor)


def _log_report_knot(
    recording: rr.RecordingStream,
    report: certifier_report_pb2.CertifierReport.Knot,
    floor: float,
) -> None:
    """Logs the certifier's view of one knot: CoM, polygon, margins and failures."""
    com = list(report.com)
    if len(com) == 3:
        recording.log(
            f"{PLAN_ROOT}/com",
            rr.Points3D(
                positions=[com, [com[0], com[1], floor]],
                radii=[_COM_RADIUS],
                colors=[palette.to_rgba8(palette.with_alpha(palette.RED, 1.0))],
            ),
        )
    polygon = [[p.x, p.y, floor] for p in report.support_polygon]
    recording.log(
        f"{PLAN_ROOT}/support_polygon",
        rr.LineStrips3D(
            [polygon + polygon[:1]] if polygon else [],
            radii=[_LINE_RADIUS],
            colors=[palette.to_rgba8(palette.with_alpha(palette.BLUE, 1.0))],
        ),
    )
    if report.HasField("static_margin"):
        recording.log(
            f"{MARGINS_ROOT}/static_margin", rr.Scalars(report.static_margin)
        )
    if report.HasField("torque_utilization"):
        recording.log(
            f"{MARGINS_ROOT}/torque_utilization",
            rr.Scalars(report.torque_utilization),
        )
    recording.log(
        f"{MARGINS_ROOT}/momentum_rate_residual",
        rr.Scalars(report.momentum_rate_residual),
    )
    if report.failures:
        recording.log(
            FAILURES_PATH,
            rr.TextLog("; ".join(report.failures), level=rr.TextLogLevel.WARN),
        )


def log_playback(recording: rr.RecordingStream, inputs: PlaybackInputs) -> int:
    """Logs the terrain, the robot's geometry and every knot of the plan.

    Args:
        recording: Where it all goes.
        inputs: The playback's inputs.

    Returns:
        How many knots it logged.

    Raises:
        OSError: a mesh file can no longer be read.
        ValueError: a contact candidate names an unknown joint.
    """
    log_terrain(recording, inputs.scenario.terrain_scene)
    robot_meshes.RobotMeshes(inputs.model).log_instance(recording, _ROBOT_STYLE)
    for index in range(len(inputs.plan.knots)):
        _log_knot(recording, inputs, index)
    return len(inputs.plan.knots)


def blueprint() -> rrb.Blueprint:
    """The layout: the 3D scene next to the margins and the failures."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="Plan", origin=scene_contract.WORLD_ROOT),
            rrb.Vertical(
                rrb.TimeSeriesView(name="Margins", origin=MARGINS_ROOT),
                rrb.TextLogView(name="Failures", origin="plan"),
            ),
            column_shares=[3, 1],
        ),
        collapse_panels=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """The parser of the playback's command line."""
    parser = argparse.ArgumentParser(
        prog="plan_playback",
        description="Plays a multi-contact plan kinematically in Rerun.",
    )
    parser.add_argument(
        "--scenario", required=True, help="the scenario textproto"
    )
    parser.add_argument(
        "--plan", required=True, help="the plan, a .binpb or a textproto"
    )
    parser.add_argument(
        "--rerun_sink",
        default="spawn",
        choices=rerun_sinks.sink_names(),
        help="where the recording goes (default: %(default)s)",
    )
    parser.add_argument(
        "--rrd_path", default="", help="save: the file to write"
    )
    parser.add_argument(
        "--rerun_url",
        default=rerun_sinks.DEFAULT_CONNECT_URL,
        help="connect: the viewer's gRPC URL",
    )
    parser.add_argument(
        "--grpc_port", type=int, default=rerun_sinks.DEFAULT_GRPC_PORT
    )
    parser.add_argument(
        "--web_port", type=int, default=rerun_sinks.DEFAULT_WEB_PORT
    )
    parser.add_argument("--app_id", default="multi_contact_playback")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the playback; returns the exit status."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="plan_playback: %(levelname)s: %(message)s"
    )
    try:
        inputs = load_inputs(args.scenario, args.plan)
    except (OSError, ValueError, urdf_model.UrdfError) as error:
        _LOGGER.error("%s", error)
        return EXIT_USAGE
    recording = rr.RecordingStream(args.app_id)
    options = rerun_sinks.SinkOptions(
        url=args.rerun_url,
        rrd_path=(
            cli.resolve_output_path(args.rrd_path) if args.rrd_path else ""
        ),
        grpc_port=args.grpc_port,
        web_port=args.web_port,
    )
    try:
        sink = rerun_sinks.make_sink(args.rerun_sink, blueprint(), options)
        _LOGGER.info("%s", sink.attach(recording))
        knots = log_playback(recording, inputs)
    except (OSError, ValueError, RuntimeError) as error:
        _LOGGER.error("%s", error)
        return EXIT_USAGE
    _LOGGER.info(
        "played %d knots of '%s' (%s)",
        knots,
        inputs.plan.scenario,
        "certified" if inputs.plan.HasField("report") else "not certified",
    )
    if args.rerun_sink in _SERVING_SINKS:
        _LOGGER.info("serving until Ctrl-C")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    sink.disconnect_all()
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
