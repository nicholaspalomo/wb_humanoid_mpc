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

"""Every wheel of a shipped robot spins freely and rolls on one cylinder: the rolling contact's geometry.

A wheel is a URDF `continuous` joint, so Pinocchio and the robot runtime give it no position bound, and in the robot's
MJCF an unlimited hinge of the same name. Its body carries exactly one collision geom, a cylinder coaxial with the hinge
with `condim="3"`: the MPC's rolling contact is the lowest point of that cylinder, wheel
center minus radius times the ground normal, which holds only for an upright cylinder on the wheel's own axis.
"""

import os
import unittest
from xml.etree import ElementTree

import mujoco
import numpy as np

# The robots with wheels and their wheel joints; the walk below must find exactly these.
EXPECTED_WHEELS = {
    "anybotics_anymal_c_wheels": (
        "LF_wheel",
        "LH_wheel",
        "RF_wheel",
        "RH_wheel",
    ),
    "limx_tron1": ("wheel_L_Joint", "wheel_R_Joint"),
}
# Largest angle between a wheel cylinder's axis and its hinge's axis [rad], and largest distance of the cylinder's
# center from the hinge's axis [m]: round-off of the six or nine digits the MJCF is written with.
AXIS_TOLERANCE_RAD = 1e-6
CENTER_TOLERANCE_M = 1e-6
# The MJCF's contact dimension of a point contact with sliding friction only.
POINT_CONTACT_CONDIM = 3

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _find_wheels() -> dict[str, tuple[str, str, list[str]]]:
    """Returns (urdf, mjcf, continuous joint names) of every robot description whose URDF has a continuous joint."""
    robots = {}
    for dirpath, _, filenames in os.walk(_ROOT):
        for filename in filenames:
            if not filename.endswith(".urdf"):
                continue
            urdf = os.path.join(dirpath, filename)
            continuous = [
                joint.get("name", "")
                for joint in ElementTree.parse(urdf).getroot().iter("joint")
                if joint.get("type") == "continuous"
            ]
            if continuous:
                robot = os.path.relpath(urdf, _ROOT).split(os.sep)[0]
                robots[robot] = (
                    urdf,
                    urdf[: -len(".urdf")] + ".xml",
                    continuous,
                )
    return robots


class WheelTest(unittest.TestCase):
    """The URDF and MJCF wheels of every wheeled robot."""

    def test_the_wheeled_robots_are_discovered(self):
        robots = _find_wheels()
        self.assertEqual(
            {
                robot: tuple(sorted(wheels))
                for robot, (_, _, wheels) in robots.items()
            },
            EXPECTED_WHEELS,
        )

    def test_the_urdf_wheels_have_no_position_bound(self):
        for robot, (urdf, _, wheels) in _find_wheels().items():
            joints = {
                joint.get("name"): joint
                for joint in ElementTree.parse(urdf).getroot().iter("joint")
            }
            for wheel in wheels:
                with self.subTest(robot=robot, wheel=wheel):
                    limit = joints[wheel].find("limit")
                    if limit is not None:
                        self.assertIsNone(limit.get("lower"))
                        self.assertIsNone(limit.get("upper"))

    def test_each_mjcf_wheel_is_an_unlimited_hinge_with_one_coaxial_cylinder(
        self,
    ):
        for robot, (_, mjcf, wheels) in _find_wheels().items():
            model = mujoco.MjModel.from_xml_path(mjcf)
            data = mujoco.MjData(model)
            mujoco.mj_kinematics(model, data)
            for wheel in wheels:
                with self.subTest(robot=robot, wheel=wheel):
                    joint = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, wheel
                    )
                    self.assertGreaterEqual(joint, 0)
                    self.assertEqual(
                        model.jnt_type[joint], mujoco.mjtJoint.mjJNT_HINGE
                    )
                    self.assertFalse(model.jnt_limited[joint])
                    body = model.jnt_bodyid[joint]
                    collisions = [
                        geom
                        for geom in range(model.ngeom)
                        if model.geom_bodyid[geom] == body
                        and (
                            model.geom_contype[geom]
                            or model.geom_conaffinity[geom]
                        )
                    ]
                    self.assertEqual(len(collisions), 1)
                    geom = collisions[0]
                    self.assertEqual(
                        model.geom_type[geom], mujoco.mjtGeom.mjGEOM_CYLINDER
                    )
                    self.assertEqual(
                        model.geom_condim[geom], POINT_CONTACT_CONDIM
                    )
                    hinge_axis = data.xaxis[joint]
                    rotmat_geom_rt_world = data.geom_xmat[geom].reshape(3, 3)
                    cosine = abs(float(hinge_axis @ rotmat_geom_rt_world[:, 2]))
                    self.assertLess(
                        np.arccos(min(cosine, 1.0)), AXIS_TOLERANCE_RAD
                    )
                    center_from_anchor = (
                        data.geom_xpos[geom] - data.xanchor[joint]
                    )
                    off_axis = center_from_anchor - hinge_axis * float(
                        hinge_axis @ center_from_anchor
                    )
                    self.assertLess(
                        np.linalg.norm(off_axis), CENTER_TOLERANCE_M
                    )


if __name__ == "__main__":
    unittest.main()
