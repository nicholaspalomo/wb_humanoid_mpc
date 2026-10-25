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
"""Every robot is described twice: by a URDF that Pinocchio loads for the MPC, and by an MJCF that MuJoCo simulates.

Both must describe the same robot. Where they disagree, the controller computes its dynamics for one machine and
applies the result to a different one, and the error is silent - nothing crashes, the robot simply behaves as if it
were being pushed by a force nobody commanded. The aCOM network is trained on the MJCF (humanoid_learning/acom) and
graded against Pinocchio on the URDF, so a disagreement also becomes a fit error no training can remove.

This test was written after exactly that. The EngineAI SA01 MJCF carried the center of mass of each ankle-pitch link at
+/- 0.259 m along x, where the URDF has +/- 2.5946e-10 m: an exponent dropped in transcription, ``E-10`` written as
``e-1``, nine orders of magnitude. The two legs took opposite signs, so the robot was asymmetric as well as wrong. In
GRAVITY_COMP that became a constant unbalanced torque on the three pitch joints of each leg and about 13 degrees of
drift over 15 seconds. With it corrected the robot holds every joint to four decimal places over the same window. The
Atlas pelvis later turned out to carry a center of mass 45.7 mm off the URDF's, and the R1 knees and ankles to leave
out the 109 g ankle linkage the URDF fixes to them.

The comparison is of each body's full rigid-body inertia - mass, center of mass and the rotational inertia about it, in
the body frame - and the MJCF side is what MuJoCo itself compiles, so every way an MJCF can state an inertia (diagonal
plus quaternion, full inertia, inferred from geoms) is read as the simulator reads it. An MJCF legitimately LUMPS a URDF
link into its parent when the two are joined by a fixed joint, as Pinocchio does when it loads the URDF, so the URDF
side aggregates each link with every fixed-joint descendant that the MJCF does not model separately (the parallel-axis
theorem moves each part's inertia to the aggregate's center of mass), rather than comparing links one to one.
"""

import math
import os
import unittest
from xml.etree import ElementTree

import mujoco
import numpy as np

# Largest accepted difference between the two descriptions, once fixed-joint lumping is accounted for. They are set
# where the six significant digits the model files are written with leave the numbers, and far below any defect worth
# catching: the SA01 bug this test was written for was 259 mm, the Atlas pelvis 45.7 mm and the R1 knee linkage 109 g.
MASS_TOLERANCE_KG = 1e-4
COM_TOLERANCE_M = 1e-4
# Of the Frobenius norm of the difference of the two inertias, relative to that of the URDF's.
RELATIVE_INERTIA_TOLERANCE = 1e-3
# The two descriptions may model a different number of massless frames, but never a different robot weight.
TOTAL_MASS_TOLERANCE_KG = 1e-3


class _Inertia:
    """A rigid body's inertia, in the frame of the body it belongs to.

    Attributes:
        mass: The mass [kg].
        com: The center of mass [m], shape (3,).
        rotational: The rotational inertia about the center of mass [kg m^2], shape (3, 3).
    """

    def __init__(self, mass: float, com: np.ndarray, rotational: np.ndarray):
        self.mass = mass
        self.com = com
        self.rotational = rotational


def _find_model_pairs() -> list[tuple[str, str, str]]:
    """Every robot description shipping both a URDF and an MJCF of the same basename, as (robot, urdf, mjcf)."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    pairs = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".urdf"):
                continue
            urdf = os.path.join(dirpath, filename)
            mjcf = urdf[: -len(".urdf")] + ".xml"
            if os.path.exists(mjcf):
                pairs.append(
                    (os.path.relpath(urdf, root).split(os.sep)[0], urdf, mjcf)
                )
    return sorted(pairs)


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Returns the rotation of the URDF's fixed-axis roll-pitch-yaw convention, Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _origin(element: ElementTree.Element) -> tuple[np.ndarray, np.ndarray]:
    """Returns the (translation, rotation) of `element`'s <origin>, the identity when it has none."""
    origin = element.find("origin")
    xyz = origin.get("xyz") if origin is not None else None
    rpy = origin.get("rpy") if origin is not None else None
    translation = (
        np.array([float(v) for v in xyz.split()]) if xyz else np.zeros(3)
    )
    angles = [float(v) for v in rpy.split()] if rpy else [0.0, 0.0, 0.0]
    return translation, _rotation_from_rpy(*angles)


def _symmetric(
    ixx: float, iyy: float, izz: float, ixy: float, ixz: float, iyz: float
) -> np.ndarray:
    """Returns the symmetric inertia matrix of the six moments, in MJCF fullinertia's order."""
    return np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])


def _attribute(element: ElementTree.Element, tag: str, name: str) -> str:
    """Returns the attribute `name` of `element`'s child `tag`, failing with the element named when either is missing."""
    child = element.find(tag)
    value = None if child is None else child.get(name)
    if value is None:
        raise ValueError(
            f"<{element.tag} name='{element.get('name')}'> has no"
            f" <{tag} {name}=...>"
        )
    return value


def _parse_urdf(path: str) -> tuple[dict[str, _Inertia], dict[str, list]]:
    """Parses the link inertias and the fixed joints of a URDF.

    Args:
        path: The URDF file.

    Returns:
        (links, fixed_children): links maps a link's name to its inertia in the link frame (a link without an
        <inertial> has none), and fixed_children maps a parent link to the (child, translation, rotation) of each
        fixed joint below it.
    """
    root = ElementTree.parse(path).getroot()
    links: dict[str, _Inertia] = {}
    for link in root.iter("link"):
        name = link.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            links[name] = _Inertia(0.0, np.zeros(3), np.zeros((3, 3)))
            continue
        com, rotation = _origin(inertial)
        moments = {
            key: float(_attribute(inertial, "inertia", key))
            for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        }
        in_inertial_frame = _symmetric(
            moments["ixx"],
            moments["iyy"],
            moments["izz"],
            moments["ixy"],
            moments["ixz"],
            moments["iyz"],
        )
        links[name] = _Inertia(
            float(_attribute(inertial, "mass", "value")),
            com,
            rotation @ in_inertial_frame @ rotation.T,
        )
    fixed_children: dict[str, list] = {}
    for joint in root.iter("joint"):
        if joint.get("type") != "fixed":
            continue
        translation, rotation = _origin(joint)
        fixed_children.setdefault(
            _attribute(joint, "parent", "link"), []
        ).append((_attribute(joint, "child", "link"), translation, rotation))
    return links, fixed_children


def _compile_mjcf(path: str) -> dict[str, _Inertia]:
    """Returns the inertia of every body of an MJCF, in its body frame, as MuJoCo compiles it."""
    model = mujoco.MjModel.from_xml_path(path)
    bodies = {}
    for body in range(1, model.nbody):  # Body 0 is the world.
        principal_axes = np.zeros(9)
        mujoco.mju_quat2Mat(principal_axes, model.body_iquat[body])
        principal_axes = principal_axes.reshape(3, 3)
        bodies[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)] = (
            _Inertia(
                float(model.body_mass[body]),
                np.array(model.body_ipos[body]),
                principal_axes
                @ np.diag(model.body_inertia[body])
                @ principal_axes.T,
            )
        )
    return bodies


def _combine(parts: list[_Inertia]) -> _Inertia:
    """Returns the inertia of rigidly joined `parts`, all in one frame, about their common center of mass."""
    mass = sum(part.mass for part in parts)
    if mass <= 0.0:
        return _Inertia(0.0, np.zeros(3), np.zeros((3, 3)))
    com = sum(part.mass * part.com for part in parts) / mass
    rotational = np.zeros((3, 3))
    for part in parts:
        offset = part.com - com
        rotational += part.rotational + part.mass * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )
    return _Inertia(mass, com, rotational)


def _lumped_inertia(
    link: str,
    links: dict[str, _Inertia],
    fixed_children: dict[str, list],
    modeled_separately: set[str],
) -> _Inertia:
    """Returns the inertia of a URDF link once every fixed-joint descendant the MJCF does not model is absorbed.

    Expressed in that link's own frame - the body the MJCF actually simulates.

    Args:
        link: The URDF link.
        links: _parse_urdf's links.
        fixed_children: _parse_urdf's fixed_children.
        modeled_separately: The bodies the MJCF models itself, which are not absorbed.
    """
    parts = []
    stack = [(link, np.zeros(3), np.eye(3))]
    while stack:
        name, translation, rotation = stack.pop()
        part = links[name]
        parts.append(
            _Inertia(
                part.mass,
                translation + rotation @ part.com,
                rotation @ part.rotational @ rotation.T,
            )
        )
        for child, child_translation, child_rotation in fixed_children.get(
            name, []
        ):
            if child in modeled_separately or child not in links:
                continue
            stack.append(
                (
                    child,
                    translation + rotation @ child_translation,
                    rotation @ child_rotation,
                )
            )
    return _combine(parts)


class InertiaAlgebraTest(unittest.TestCase):
    """The lumping the comparison rests on, against an independent oracle: MuJoCo fusing the same bodies."""

    def test_lumping_matches_mujoco_fusing_a_welded_child(self):
        # A body with a welded child, rotated and offset; MuJoCo's fusestatic merges the child into the parent, so its
        # compiled inertia is the lumped one, computed by MuJoCo's own code.
        mjcf = """<mujoco><compiler angle="radian" fusestatic="true" inertiafromgeom="false"/><worldbody>
          <body name="parent"><freejoint/>
            <inertial pos="0.1 -0.02 0.03" mass="2.0" fullinertia="0.05 0.04 0.03 0.002 -0.001 0.0005"/>
            <body name="child" pos="0.2 0.1 -0.3" euler="0.3 -0.5 0.9">
              <inertial pos="0.01 0.02 -0.05" mass="0.7" fullinertia="0.005 0.006 0.004 0.0003 0.0002 -0.0001"/>
            </body>
          </body></worldbody></mujoco>"""
        model = mujoco.MjModel.from_xml_string(mjcf)
        self.assertEqual(model.nbody, 2, "the child must have been fused away")
        principal_axes = np.zeros(9)
        mujoco.mju_quat2Mat(principal_axes, model.body_iquat[1])
        principal_axes = principal_axes.reshape(3, 3)
        fused = (
            principal_axes @ np.diag(model.body_inertia[1]) @ principal_axes.T
        )

        child_rotation = np.zeros(9)
        child_quat = np.zeros(4)
        mujoco.mju_euler2Quat(child_quat, np.array([0.3, -0.5, 0.9]), "xyz")
        mujoco.mju_quat2Mat(child_rotation, child_quat)
        links = {
            "parent": _Inertia(
                2.0,
                np.array([0.1, -0.02, 0.03]),
                _symmetric(0.05, 0.04, 0.03, 0.002, -0.001, 0.0005),
            ),
            "child": _Inertia(
                0.7,
                np.array([0.01, 0.02, -0.05]),
                _symmetric(0.005, 0.006, 0.004, 0.0003, 0.0002, -0.0001),
            ),
        }
        fixed_children = {
            "parent": [
                (
                    "child",
                    np.array([0.2, 0.1, -0.3]),
                    child_rotation.reshape(3, 3),
                )
            ]
        }
        lumped = _lumped_inertia("parent", links, fixed_children, {"parent"})
        self.assertAlmostEqual(
            lumped.mass, float(model.body_mass[1]), places=12
        )
        np.testing.assert_allclose(lumped.com, model.body_ipos[1], atol=1e-12)
        # MuJoCo stores the fused inertia as principal moments and axes, which costs it the last few digits.
        np.testing.assert_allclose(lumped.rotational, fused, atol=1e-7)

    def test_urdf_rpy_is_the_fixed_axis_convention(self):
        # Rz(yaw) Ry(pitch) Rx(roll): a rotation about x by roll applied first, about the fixed axes.
        roll, pitch, yaw = 0.4, -0.7, 1.3
        expected = np.zeros(9)
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([roll, pitch, yaw]), "XYZ")
        mujoco.mju_quat2Mat(expected, quat)
        np.testing.assert_allclose(
            _rotation_from_rpy(roll, pitch, yaw),
            expected.reshape(3, 3),
            atol=1e-12,
        )


class UrdfMjcfInertialConsistencyTest(unittest.TestCase):
    """Each robot's MJCF bodies carry the inertia Pinocchio gives the same bodies from the URDF."""

    def test_model_pairs_are_discovered(self):
        """Guards the test itself: a walk that silently found nothing would pass every assertion below."""
        robots = [robot for robot, _, _ in _find_model_pairs()]
        for robot in ("drc_atlas", "engineai_sa01", "unitree_g1", "unitree_r1"):
            self.assertIn(
                robot,
                robots,
                "expected a URDF/MJCF pair for each shipped robot",
            )

    def test_bodies_agree_on_mass_center_of_mass_and_inertia(self):
        for robot, urdf, mjcf in _find_model_pairs():
            links, fixed_children = _parse_urdf(urdf)
            bodies = _compile_mjcf(mjcf)
            shared = sorted(set(links) & set(bodies))
            self.assertGreater(
                len(shared),
                0,
                f"{robot}: the two descriptions share no body names",
            )
            for body in shared:
                with self.subTest(robot=robot, body=body):
                    self._assert_agree(
                        robot,
                        body,
                        _lumped_inertia(
                            body, links, fixed_children, set(bodies)
                        ),
                        bodies[body],
                    )

    def _assert_agree(
        self, robot: str, body: str, urdf: _Inertia, mjcf: _Inertia
    ) -> None:
        """Fails, naming both sets of numbers, when one body's two inertias differ."""
        mass_error = abs(urdf.mass - mjcf.mass)
        com_error = float(np.max(np.abs(urdf.com - mjcf.com)))
        inertia_error = float(
            np.linalg.norm(urdf.rotational - mjcf.rotational)
        ) / max(float(np.linalg.norm(urdf.rotational)), 1e-12)
        self.assertTrue(
            mass_error <= MASS_TOLERANCE_KG
            and com_error <= COM_TOLERANCE_M
            and inertia_error <= RELATIVE_INERTIA_TOLERANCE,
            f"{robot}: body '{body}' differs between the URDF and the MJCF\n "
            f" mass {urdf.mass:.6f} vs {mjcf.mass:.6f} kg (difference"
            f" {mass_error:.3e})\n  com  {urdf.com} vs {mjcf.com} m (largest"
            f" difference {com_error:.3e})\n  inertia about the"
            f" com\n{urdf.rotational}\n  vs\n{mjcf.rotational}\n  (relative"
            f" difference {inertia_error:.3e})\n(URDF figures include every"
            " fixed-joint descendant the MJCF does not model separately.)\nThe"
            " MPC computes its dynamics from the URDF and MuJoCo simulates the"
            " MJCF, so a difference here becomes an uncommanded force on the"
            " robot. Correct whichever file is wrong; do not widen the"
            " tolerance.",
        )

    def test_total_mass_agrees(self):
        for robot, urdf, mjcf in _find_model_pairs():
            links, _ = _parse_urdf(urdf)
            urdf_total = sum(link.mass for link in links.values())
            mjcf_total = sum(body.mass for body in _compile_mjcf(mjcf).values())
            self.assertAlmostEqual(
                urdf_total,
                mjcf_total,
                delta=TOTAL_MASS_TOLERANCE_KG,
                msg=(
                    f"{robot}: total mass {urdf_total:.4f} kg (URDF) vs"
                    f" {mjcf_total:.4f} kg (MJCF)"
                ),
            )


if __name__ == "__main__":
    unittest.main()
