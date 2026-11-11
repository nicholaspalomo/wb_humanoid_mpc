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
"""Every robot's joint PD gains are stable in its MuJoCo scene at the simulator's step.

The simulator (MujocoSimInterface::simulationStep()) computes each joint's PD torque kp (q_des - q) + kd (qd_des - qd)
from the state of the previous step and hands it to mj_step as a constant actuator force: the PD is explicit, at the
physics step h. MuJoCo's Euler integrator treats only the scene's joint damping d implicitly. Linearized about a pose
with the base free and no contact, one step is

    v+ = v + h (M + h D)^-1 (-Kd v - D v - Kp q),    q+ = q + h v+,

with M the joint-space inertia. The step map of (q, v) has eigenvalues of modulus at most 1 exactly when no mode grows
from one step to the next; the free base and the joints without a PD give eigenvalues of modulus 1. A light link under
a large kd fails first: its mode flips sign every step, at half the step rate, which the 80 Hz telemetry of
the closed-loop runs shows aliased, and the torque limits turn it into a chatter between the joint's limits. The G1's
ankle-roll links (3.2e-4 kg m^2 about their axes, kd = 6) did exactly that with the scene's damping of zero after the
simulator stopped giving every joint 10 N m s/rad (082971a6), and the robot fell 12 s into walk_0p5.

The pose is the scene's qpos0, the base lifted clear of the ground, which is where M is the swing leg's.
"""

import os
import unittest

from humanoid_mpc_config import joint_pd_gains_file_pb2
import mujoco
import numpy as np

import nproto_textproto

# LINT.IfChange(simulator_time_step)
# [s] The simulator's physics step, MujocoSimConfig::kDefaultTimestep, which every robot configuration runs at.
SIM_TIME_STEP_S = 0.0005
# LINT.ThenChange(//robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/MujocoSimInterface.h:simulator_time_step)
# The largest modulus of an eigenvalue of the step map that counts as stable: 1 up to round-off.
MAX_STABLE_EIGENVALUE_MODULUS = 1.0 + 1e-9
# [m] How far the base is lifted above its qpos0, clear of the ground.
LIFT_BASE_M = 1.0

# Each robot configuration: its MuJoCo scene and its joint PD gains file, relative to robot_models/.
_ROBOTS = {
    "anymal_b": (
        "anybotics_anymal_b/anymal_b_description/urdf/anymal_b.xml",
        "anybotics_anymal_b/anymal_b_centroidal_mpc/config/controller/joint_pd_gains.textproto",
    ),
    "drc_atlas": (
        "drc_atlas/drc_atlas_description/urdf/atlas.xml",
        "drc_atlas/drc_atlas_centroidal_mpc/config/controller/joint_pd_gains.textproto",
    ),
    "engineai_sa01": (
        "engineai_sa01/engineai_sa01_description/urdf/zq_sa01.xml",
        "engineai_sa01/engineai_sa01_centroidal_mpc/config/controller/joint_pd_gains.textproto",
    ),
    "unitree_g1": (
        "unitree_g1/g1_description/urdf/g1_29dof.xml",
        "unitree_g1/g1_centroidal_mpc/config/controller/joint_pd_gains.textproto",
    ),
    "unitree_g1_wb_mpc": (
        "unitree_g1/g1_description/urdf/g1_29dof.xml",
        "unitree_g1/g1_wb_mpc/config/controller/joint_pd_gains.textproto",
    ),
    "unitree_r1": (
        "unitree_r1/unitree_r1_description/urdf/R1.xml",
        "unitree_r1/unitree_r1_centroidal_mpc/config/controller/joint_pd_gains.textproto",
    ),
}

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _gains_by_joint(
    gains: joint_pd_gains_file_pb2.JointPdGainsFile,
) -> dict[str, tuple[float, float]]:
    """Returns the (kp, kd) of every joint the file names, each gain it leaves out at default_gains'."""
    resolved = {}
    for entry in gains.joint_gains:
        kp = entry.kp if entry.HasField("kp") else gains.default_gains.kp
        kd = entry.kd if entry.HasField("kd") else gains.default_gains.kd
        resolved[entry.joint] = (kp, kd)
    return resolved


def _load(robot: str) -> tuple[mujoco.MjModel, np.ndarray, np.ndarray]:
    """Returns `robot`'s scene and the diagonal kp and kd its PD gains file gives each degree of freedom.

    Args:
      robot: A key of _ROBOTS.

    Returns:
      The compiled scene, and kp [N m/rad] and kd [N m s/rad] per degree of freedom: a joint's default_gains when the
      file does not name it, zero on a degree of freedom without an actuator (the base).
    """
    scene, gains_file = _ROBOTS[robot]
    model = mujoco.MjModel.from_xml_path(os.path.join(_ROOT, scene))
    gains = nproto_textproto.load_textproto(
        os.path.join(_ROOT, gains_file),
        joint_pd_gains_file_pb2.JointPdGainsFile,
    )
    by_joint = _gains_by_joint(gains)
    kp = np.zeros(model.nv)
    kd = np.zeros(model.nv)
    for actuator in range(model.nu):
        joint = model.actuator_trnid[actuator, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        dof = model.jnt_dofadr[joint]
        kp[dof], kd[dof] = by_joint.get(
            name, (gains.default_gains.kp, gains.default_gains.kd)
        )
    return model, kp, kd


def _step_map(
    model: mujoco.MjModel,
    kp: np.ndarray,
    kd: np.ndarray,
    damping: np.ndarray,
) -> np.ndarray:
    """Returns the linearized map of (q, v) over one simulator step (the module docstring), at qpos0 lifted clear.

    Args:
      model: The compiled scene.
      kp: The PD stiffness per degree of freedom [N m/rad].
      kd: The PD damping per degree of freedom [N m s/rad].
      damping: The scene's joint damping per degree of freedom [N m s/rad], which MuJoCo integrates implicitly.

    Returns:
      The 2 nv x 2 nv matrix taking (q, v) to (q+, v+).
    """
    data = mujoco.MjData(model)
    data.qpos[2] += LIFT_BASE_M
    mujoco.mj_forward(model, data)
    nv = model.nv
    inertia = np.zeros((nv, nv))
    mujoco.mj_fullM(model, data, inertia)
    step = SIM_TIME_STEP_S
    implicit = np.linalg.inv(inertia + step * np.diag(damping))
    next_v_of_v = np.eye(nv) - step * implicit @ np.diag(kd + damping)
    next_v_of_q = -step * implicit @ np.diag(kp)
    return np.block(
        [
            [np.eye(nv) + step * next_v_of_q, step * next_v_of_v],
            [next_v_of_q, next_v_of_v],
        ]
    )


def _unstable_modes(model: mujoco.MjModel, step_map: np.ndarray) -> list[str]:
    """Returns one description per mode of `step_map` that grows: its eigenvalue's modulus and its largest joints."""
    eigenvalues, eigenvectors = np.linalg.eig(step_map)
    nv = model.nv
    modes = []
    for k in np.argsort(-np.abs(eigenvalues)):
        if abs(eigenvalues[k]) <= MAX_STABLE_EIGENVALUE_MODULUS:
            break
        weights = np.abs(eigenvectors[nv:, k])
        largest = np.argsort(-weights)[:2]
        joints = ", ".join(
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, model.dof_jntid[dof]
            )
            for dof in largest
        )
        modes.append(f"|lambda| = {abs(eigenvalues[k]):.3f} on {joints}")
    return modes


class JointPdStabilityTest(unittest.TestCase):
    """The explicit joint PD of every robot configuration in its scene."""

    def test_the_scenes_integrate_their_damping_implicitly(self):
        for robot in _ROBOTS:
            model, _, _ = _load(robot)
            # The step map assumes MuJoCo's Euler integrator, implicit in the joint damping only.
            self.assertEqual(
                model.opt.integrator,
                mujoco.mjtIntegrator.mjINT_EULER,
                f"{robot}: the scene's integrator",
            )

    def test_every_robot_is_stable_at_the_simulator_step(self):
        for robot in _ROBOTS:
            model, kp, kd = _load(robot)
            modes = _unstable_modes(
                model, _step_map(model, kp, kd, model.dof_damping)
            )
            self.assertEqual(
                modes,
                [],
                f"{robot}: the joint PD grows from step to step at"
                f" h = {SIM_TIME_STEP_S} s; give the joints damping in the scene or lower their kd",
            )

    def test_the_g1_ankle_rolls_are_unstable_without_their_damping(self):
        # The regression this file guards against: the G1 scene with the damping of zero it had while the simulator
        # added none.
        model, kp, kd = _load("unitree_g1")
        modes = _unstable_modes(
            model, _step_map(model, kp, kd, np.zeros(model.nv))
        )
        self.assertTrue(modes)
        self.assertIn("ankle_roll_joint", modes[0])


if __name__ == "__main__":
    unittest.main()
