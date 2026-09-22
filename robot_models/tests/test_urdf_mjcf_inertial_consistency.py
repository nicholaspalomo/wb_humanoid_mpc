"""Every robot is described twice: by a URDF that Pinocchio loads for the MPC, and by an MJCF that MuJoCo simulates.

Both must describe the same robot. Where they disagree, the controller computes its dynamics for one machine and
applies the result to a different one, and the error is silent - nothing crashes, the robot simply behaves as if it
were being pushed by a force nobody commanded.

This test was written after exactly that. The EngineAI SA01 MJCF carried the centre of mass of each ankle-pitch link at
+/- 0.259 m along x, where the URDF has +/- 2.5946e-10 m: an exponent dropped in transcription, ``E-10`` written as
``e-1``, nine orders of magnitude. The two legs took opposite signs, so the robot was asymmetric as well as wrong. In
GRAVITY_COMP that became a constant unbalanced torque on the three pitch joints of each leg and about 13 degrees of
drift over 15 seconds. With it corrected the robot holds every joint to four decimal places over the same window.

An MJCF legitimately LUMPS a URDF link into its parent when the two are joined by a fixed joint, which is why the
comparison below aggregates each URDF link with every fixed-joint descendant that the MJCF does not model separately,
rather than comparing link masses one to one. Without that, the Atlas cameras and the G1 head and rubber hands look
like defects when they are only a different, and equally valid, choice of where to draw a body boundary.
"""

import math
import os
import unittest
import xml.etree.ElementTree as ET

# Maximum accepted difference between the two descriptions, once fixed-joint lumping is accounted for. Loose enough to
# tolerate a massless placeholder link one description rounds away (the G1 pelvis differs by a gram and 20 micrometres
# on that account), and far tighter than any defect worth catching: the SA01 bug this test was written for was 259 mm,
# and the Atlas pelvis deviation below is 45.7 mm.
MASS_TOLERANCE_KG = 2e-3
COM_TOLERANCE_M = 1e-3
# The two descriptions may model a different number of massless frames, but never a different robot weight.
TOTAL_MASS_TOLERANCE_KG = 0.5

# Known, deliberately unresolved disagreements, as (robot, body) -> reason. Each is a real defect left uncorrected
# because the authoritative source is not established and changing a model would invalidate that robot's tuning.
# Entries are removed by fixing the model, never by widening a tolerance.
KNOWN_DEVIATIONS = {
    # Each of these is a genuine disagreement, not fixed-joint lumping, left uncorrected because the authoritative
    # source is not established and changing either model would invalidate that robot's existing tuning. They are
    # recorded here so the set cannot silently grow. Remove an entry by fixing the model, never by widening a tolerance.
    ("drc_atlas", "pelvis"): (
        "COM differs by 45.7 mm in x (URDF +0.0111, MJCF -0.0346) at identical mass. The pelvis is the floating-base "
        "root, so it contributes nothing to the joint gravity torques and does not affect GRAVITY_COMP, but it does "
        "move the whole-body COM and ZMP the centroidal MPC reasons about relative to the ones MuJoCo simulates."
    ),
    (
        "unitree_r1",
        "left_knee_link",
    ): "Mass differs by 108.9 g, 4.59% of the link - the largest deviation in the fleet.",
    (
        "unitree_r1",
        "right_knee_link",
    ): "Mass differs by 108.9 g, 4.59% of the link - the largest deviation in the fleet.",
    (
        "unitree_r1",
        "left_ankle_roll_link",
    ): "Mass differs by 8.6 g, 1.79% of the link, and the COM by 0.43 mm.",
    (
        "unitree_r1",
        "right_ankle_roll_link",
    ): "Mass differs by 8.6 g, 1.79% of the link, and the COM by 0.43 mm.",
    (
        "unitree_r1",
        "pelvis_link",
    ): "Mass differs by 3.2 g, 0.14% of the link. The root link, so no joint gravity effect.",
}


def _find_model_pairs():
    """Every robot description shipping both a URDF and an MJCF of the same basename."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    pairs = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".urdf"):
                continue
            urdf = os.path.join(dirpath, filename)
            mjcf = urdf[: -len(".urdf")] + ".xml"
            if os.path.exists(mjcf):
                pairs.append((os.path.relpath(urdf, root).split(os.sep)[0], urdf, mjcf))
    return sorted(pairs)


def _rotation_from_rpy(roll, pitch, yaw):
    """The URDF fixed-axis roll-pitch-yaw convention, as a row-major 3x3."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _transform(rotation, translation, point):
    return [
        translation[axis] + sum(rotation[axis][k] * point[k] for k in range(3))
        for axis in range(3)
    ]


def _floats(text, default=(0.0, 0.0, 0.0)):
    return [float(value) for value in text.split()] if text else list(default)


def _parse_urdf(path):
    """Returns (links, fixed_children) where links maps name -> (mass, com) and fixed_children maps parent -> list of
    (child, translation, rotation) for fixed joints only."""
    root = ET.parse(path).getroot()
    links = {}
    for link in root.iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            links[link.get("name")] = (0.0, [0.0, 0.0, 0.0])
            continue
        origin = inertial.find("origin")
        links[link.get("name")] = (
            float(inertial.find("mass").get("value")),
            _floats(origin.get("xyz") if origin is not None else None),
        )
    fixed_children = {}
    for joint in root.iter("joint"):
        if joint.get("type") != "fixed":
            continue
        origin = joint.find("origin")
        translation = _floats(origin.get("xyz") if origin is not None else None)
        rotation = _rotation_from_rpy(
            *_floats(origin.get("rpy") if origin is not None else None)
        )
        fixed_children.setdefault(joint.find("parent").get("link"), []).append(
            (joint.find("child").get("link"), translation, rotation)
        )
    return links, fixed_children


def _parse_mjcf(path):
    bodies = {}
    for body in ET.parse(path).getroot().iter("body"):
        inertial = body.find("inertial")
        if inertial is not None:
            bodies[body.get("name")] = (
                float(inertial.get("mass")),
                _floats(inertial.get("pos")),
            )
    return bodies


def _lumped_inertial(link, links, fixed_children, modelled_separately):
    """Mass and COM of a URDF link once every fixed-joint descendant the MJCF does not model separately is absorbed
    into it, expressed in that link's own frame - the body the MJCF actually simulates.
    """
    mass, com = links[link]
    total_mass = mass
    weighted = [mass * com[axis] for axis in range(3)]

    stack = [
        (child, translation, rotation)
        for child, translation, rotation in fixed_children.get(link, [])
    ]
    while stack:
        child, translation, rotation = stack.pop()
        if child in modelled_separately or child not in links:
            continue
        child_mass, child_com = links[child]
        position = _transform(rotation, translation, child_com)
        total_mass += child_mass
        for axis in range(3):
            weighted[axis] += child_mass * position[axis]
        for grandchild, child_translation, child_rotation in fixed_children.get(
            child, []
        ):
            composed_rotation = [
                [
                    sum(rotation[i][k] * child_rotation[k][j] for k in range(3))
                    for j in range(3)
                ]
                for i in range(3)
            ]
            stack.append(
                (
                    grandchild,
                    _transform(rotation, translation, child_translation),
                    composed_rotation,
                )
            )

    if total_mass <= 0.0:
        return 0.0, [0.0, 0.0, 0.0]
    return total_mass, [weighted[axis] / total_mass for axis in range(3)]


class UrdfMjcfInertialConsistencyTest(unittest.TestCase):
    def test_model_pairs_are_discovered(self):
        """Guards the test itself: a walk that silently found nothing would pass every assertion below."""
        self.assertGreaterEqual(
            len(_find_model_pairs()),
            4,
            "expected a URDF/MJCF pair for each shipped robot",
        )

    def test_bodies_agree_on_mass_and_centre_of_mass(self):
        for robot, urdf, mjcf in _find_model_pairs():
            links, fixed_children = _parse_urdf(urdf)
            bodies = _parse_mjcf(mjcf)
            shared = sorted(set(links) & set(bodies))
            self.assertGreater(
                len(shared), 0, "%s: the two descriptions share no body names" % robot
            )
            modelled_separately = set(bodies)

            for body in shared:
                urdf_mass, urdf_com = _lumped_inertial(
                    body, links, fixed_children, modelled_separately
                )
                mjcf_mass, mjcf_com = bodies[body]
                mass_error = abs(urdf_mass - mjcf_mass)
                com_error = max(abs(a - b) for a, b in zip(urdf_com, mjcf_com))
                agrees = (
                    mass_error <= MASS_TOLERANCE_KG and com_error <= COM_TOLERANCE_M
                )

                if (robot, body) in KNOWN_DEVIATIONS:
                    self.assertFalse(
                        agrees,
                        "%s/%s is listed in KNOWN_DEVIATIONS but the models now agree; delete the entry."
                        % (robot, body),
                    )
                    continue

                self.assertTrue(
                    agrees,
                    "%s: body '%s' differs between\n  %s\n  %s\n"
                    "  mass %.6f vs %.6f (difference %.3e kg)\n"
                    "  com  %s vs %s (largest difference %.3e m)\n"
                    "(URDF figures include every fixed-joint descendant the MJCF does not model separately.)\n"
                    "The MPC computes its dynamics from the URDF and MuJoCo simulates the MJCF, so a difference here "
                    "becomes an uncommanded force on the robot. Correct whichever file is wrong; do not widen the "
                    "tolerance."
                    % (
                        robot,
                        body,
                        urdf,
                        mjcf,
                        urdf_mass,
                        mjcf_mass,
                        mass_error,
                        urdf_com,
                        mjcf_com,
                        com_error,
                    ),
                )

    def test_total_mass_agrees(self):
        for robot, urdf, mjcf in _find_model_pairs():
            links, _ = _parse_urdf(urdf)
            urdf_total = sum(mass for mass, _ in links.values())
            mjcf_total = sum(mass for mass, _ in _parse_mjcf(mjcf).values())
            self.assertAlmostEqual(
                urdf_total,
                mjcf_total,
                delta=TOTAL_MASS_TOLERANCE_KG,
                msg="%s: total mass %.4f kg (URDF) vs %.4f kg (MJCF)"
                % (robot, urdf_total, mjcf_total),
            )


if __name__ == "__main__":
    unittest.main()
