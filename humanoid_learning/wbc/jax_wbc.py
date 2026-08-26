"""JAX-native QP-based Model-Based Whole-Body Controller (WBC) for General Robots."""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from humanoid_learning.wbc.robot_model_loader import RobotModelSpec

# ==============================================================================
# WBC Dimensionality & Kinematic Defaults
# ==============================================================================
DEFAULT_NUM_ACTUATORS: int = 12
DEFAULT_FLOATING_BASE_POS_DIM: int = 7  # 3 Cartesian translations + 4 quaternion params
DEFAULT_FLOATING_BASE_VEL_DIM: int = 6  # 3 linear velocities + 3 angular velocities
DEFAULT_NUM_CONTACT_POINTS: int = 2
DEFAULT_ROBOT_MASS_KG: float = 35.0
DEFAULT_GRAVITY_ACCEL: float = 9.81
DEFAULT_ROTATIONAL_INERTIA_SCALE: float = 0.05
DEFAULT_MEMBER_COUPLING_WEIGHT: float = 0.2
DEFAULT_REGULARIZATION_EPS: float = 1e-4
CONTACT_DIM_PER_POINT: int = 3  # (fx, fy, fz)

# Default Task Weights for Quadratic Program (QP)
DEFAULT_WEIGHT_BASE_ACC: float = 100.0
DEFAULT_WEIGHT_POSTURE: float = 10.0
DEFAULT_WEIGHT_CONTACT_ACC: float = 1000.0
DEFAULT_WEIGHT_FORCE_REG: float = 1e-4
DEFAULT_WEIGHT_TORQUE_REG: float = 1e-4

# Default PD Feedback Tracking Gains
DEFAULT_KP_POSTURE: float = 100.0
DEFAULT_KD_POSTURE: float = 10.0

# Default Physical Limits
DEFAULT_FRICTION_COEF: float = 0.6
DEFAULT_F_Z_MIN_N: float = 5.0
DEFAULT_F_Z_MAX_N: float = 2000.0
DEFAULT_TAU_MAX_NM: float = 100.0


class WBCControlOutput(NamedTuple):
    """Container for Whole-Body Controller outputs and telemetry."""

    torques: jax.Array  # Actuator torques [n_act] (N*m)
    qddot: jax.Array  # Generalized accelerations [nv] (rad/s^2)
    contact_forces: jax.Array  # Contact ground reaction forces [3*n_contacts] (N)
    posture_error: jax.Array  # Joint posture tracking error [n_act] (rad)
    qp_cost: jax.Array  # Optimal QP objective value


@dataclass
class JaxWBCConfig:
    """Hyperparameters and task weights for JAX Whole-Body Controller."""

    # Task weights in QP objective
    w_base_acc: float = DEFAULT_WEIGHT_BASE_ACC  # Floating base tracking weight
    w_posture: float = DEFAULT_WEIGHT_POSTURE  # Joint posture / residual tracking weight
    w_contact_acc: float = DEFAULT_WEIGHT_CONTACT_ACC  # Stance foot rigidity constraint weight
    w_force_reg: float = DEFAULT_WEIGHT_FORCE_REG  # Contact force regularization
    w_torque_reg: float = DEFAULT_WEIGHT_TORQUE_REG  # Motor torque regularization

    # PD feedback gains for posture task
    kp_posture: float = DEFAULT_KP_POSTURE  # Proportional gain (1/s^2)
    kd_posture: float = DEFAULT_KD_POSTURE  # Derivative gain (1/s)

    # Physical constraints
    friction_coef: float = DEFAULT_FRICTION_COEF  # Coulomb friction coefficient mu
    f_z_min: float = DEFAULT_F_Z_MIN_N  # Minimum normal force at contact (N)
    f_z_max: float = DEFAULT_F_Z_MAX_N  # Maximum normal force at contact (N)
    tau_max: float = DEFAULT_TAU_MAX_NM  # Motor peak torque limit (N*m)

    @classmethod
    def from_dict(cls, d: dict) -> "JaxWBCConfig":
        """Load WBC configuration from dictionary."""
        valid_keys = {
            "w_base_acc",
            "w_posture",
            "w_contact_acc",
            "w_force_reg",
            "w_torque_reg",
            "kp_posture",
            "kd_posture",
            "friction_coef",
            "f_z_min",
            "f_z_max",
            "tau_max",
        }
        filtered = {k: float(v) for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "w_base_acc": float(self.w_base_acc),
            "w_posture": float(self.w_posture),
            "w_contact_acc": float(self.w_contact_acc),
            "w_force_reg": float(self.w_force_reg),
            "w_torque_reg": float(self.w_torque_reg),
            "kp_posture": float(self.kp_posture),
            "kd_posture": float(self.kd_posture),
            "friction_coef": float(self.friction_coef),
            "f_z_min": float(self.f_z_min),
            "f_z_max": float(self.f_z_max),
            "tau_max": float(self.tau_max),
        }


class JaxWholeBodyController:
    """General, Batched JAX-differentiable QP-Based Whole-Body Controller.

    Supports any robot topology:
    - Quadrupeds (4 stance contacts)
    - Humanoids / Bipeds (2 stance contacts)
    - Drones / Aerial / Underwater vehicles (0 stance contacts)
    - Fixed-base or Mobile Manipulators
    """

    def __init__(
        self,
        nq: int | None = None,
        nv: int | None = None,
        n_act: int | None = None,
        n_contacts: int | None = None,
        total_mass: float = DEFAULT_ROBOT_MASS_KG,
        config: JaxWBCConfig | None = None,
        spec: RobotModelSpec | None = None,
    ):
        if spec is not None:
            self.spec = spec
            self.nq = int(spec.nq)
            self.nv = int(spec.nv)
            self.n_act = int(spec.n_act)
            self.n_contacts = spec.num_contact_points
            self.total_mass = float(spec.total_mass)
            self.limb_joint_indices = dict(spec.limb_joint_indices)

            # Load configuration from per-robot config if not explicitly provided
            if config is not None:
                self.config = config
            elif spec.wbc_config:
                self.config = JaxWBCConfig.from_dict(spec.wbc_config)
            else:
                tau_limit = (
                    float(max(spec.torque_limits))
                    if spec.torque_limits
                    else DEFAULT_TAU_MAX_NM
                )
                self.config = JaxWBCConfig(tau_max=tau_limit)
        else:
            self.spec = None
            self.n_act = n_act if n_act is not None else DEFAULT_NUM_ACTUATORS
            self.nv = (
                nv
                if nv is not None
                else self.n_act + DEFAULT_FLOATING_BASE_VEL_DIM
            )
            self.nq = (
                nq
                if nq is not None
                else self.n_act + DEFAULT_FLOATING_BASE_POS_DIM
            )
            self.n_contacts = (
                n_contacts
                if n_contacts is not None
                else DEFAULT_NUM_CONTACT_POINTS
            )
            self.total_mass = total_mass
            self.limb_joint_indices = {
                "left": list(range(self.n_act // 2)),
                "right": list(range(self.n_act // 2, self.n_act)),
            }
            self.config = config or JaxWBCConfig()

        # Generalized floating base coordinate counts
        self.n_fb_vel = max(0, self.nv - self.n_act)
        self.n_fb_pos = max(0, self.nq - self.n_act)

        # Actuation selection matrix S: [n_act x nv]
        self.S = jnp.zeros((self.n_act, self.nv), dtype=jnp.float32).at[
            :, self.n_fb_vel :
        ].set(jnp.eye(self.n_act))

    def solve(
        self,
        q: jax.Array,
        qd: jax.Array,
        q_des: jax.Array,
        qd_des: jax.Array | None = None,
        M: jax.Array | None = None,
        h: jax.Array | None = None,
        J_c: jax.Array | None = None,
        Jdot_qd_c: jax.Array | None = None,
    ) -> WBCControlOutput:
        """Solve Whole-Body Control optimization for joint torques and accelerations."""
        if qd_des is None:
            qd_des = jnp.zeros((self.n_act,), dtype=jnp.float32)

        # Dynamic mass matrix scaled to total robot mass
        if M is None:
            M = jnp.eye(self.nv, dtype=jnp.float32) * 1.0
            if self.n_fb_vel >= 3:
                M = M.at[:3, :3].set(jnp.eye(3) * self.total_mass)
            if self.n_fb_vel >= 6:
                M = M.at[3:6, 3:6].set(
                    jnp.eye(3) * (self.total_mass * DEFAULT_ROTATIONAL_INERTIA_SCALE)
                )

        if h is None:
            h = jnp.zeros((self.nv,), dtype=jnp.float32)
            if self.n_fb_vel >= 3:
                h = h.at[2].set(self.total_mass * DEFAULT_GRAVITY_ACCEL)

        # Extract actuated joint states dynamically (last n_act coordinates)
        joint_q = q[self.n_fb_pos :] if q.shape[0] > self.n_act else q
        joint_qd = qd[self.n_fb_vel :] if qd.shape[0] > self.n_act else qd

        # 1. Posture Task Desired Acceleration
        pos_error = q_des - joint_q
        vel_error = qd_des - joint_qd
        qddot_posture_des = (
            self.config.kp_posture * pos_error + self.config.kd_posture * vel_error
        )
        J_p = self.S  # [n_act x nv]

        # 2. Contact Task & QP Optimization Formulation
        dim_f = CONTACT_DIM_PER_POINT * self.n_contacts

        if self.n_contacts == 0:
            # Contact-Free Case (e.g. Drones / Aerial Vehicles / Floating Manipulators)
            H_qddot = (
                self.config.w_posture * (J_p.T @ J_p)
                + DEFAULT_REGULARIZATION_EPS * jnp.eye(self.nv)
            )
            g_qddot = -(self.config.w_posture * (J_p.T @ qddot_posture_des))

            qddot = jnp.linalg.solve(H_qddot, -g_qddot)
            f_c = jnp.zeros((0,), dtype=jnp.float32)

            net_wrench = M @ qddot + h
            raw_torques = self.S @ net_wrench
            clipped_torques = jnp.clip(
                raw_torques, -self.config.tau_max, self.config.tau_max
            )

            qp_cost = 0.5 * (
                jnp.sum(jnp.square(qddot))
                + self.config.w_torque_reg * jnp.sum(jnp.square(clipped_torques))
            )
        else:
            # Contact-Constrained Case (Quadrupeds, Bipeds, Hexapods)
            if J_c is None:
                J_c = jnp.zeros((dim_f, self.nv), dtype=jnp.float32)
                for c in range(self.n_contacts):
                    if self.n_fb_vel >= 3:
                        J_c = J_c.at[
                            CONTACT_DIM_PER_POINT * c : CONTACT_DIM_PER_POINT * c + 3,
                            :3,
                        ].set(jnp.eye(3))

                # Dynamically couple limb joint indices to stance contacts
                limb_keys = list(self.limb_joint_indices.keys())
                for c in range(min(self.n_contacts, len(limb_keys))):
                    group = self.limb_joint_indices[limb_keys[c]]
                    for idx in group[:3]:
                        col = self.n_fb_vel + idx
                        if col < self.nv:
                            J_c = J_c.at[
                                CONTACT_DIM_PER_POINT * c : CONTACT_DIM_PER_POINT * c + 3,
                                col,
                            ].set(jnp.ones(3) * DEFAULT_MEMBER_COUPLING_WEIGHT)

            if Jdot_qd_c is None:
                Jdot_qd_c = jnp.zeros((dim_f,), dtype=jnp.float32)

            H_qddot = (
                self.config.w_posture * (J_p.T @ J_p)
                + self.config.w_contact_acc * (J_c.T @ J_c)
                + DEFAULT_REGULARIZATION_EPS * jnp.eye(self.nv)
            )
            g_qddot = -(
                self.config.w_posture * (J_p.T @ qddot_posture_des)
                + self.config.w_contact_acc * (J_c.T @ (-Jdot_qd_c))
            )

            H_f = self.config.w_force_reg * jnp.eye(dim_f)
            total_weight = self.total_mass * DEFAULT_GRAVITY_ACCEL
            f_des = (
                jnp.zeros(dim_f)
                .at[2::CONTACT_DIM_PER_POINT]
                .set(total_weight / self.n_contacts)
            )
            g_f = -self.config.w_force_reg * f_des

            qddot = jnp.linalg.solve(H_qddot, -g_qddot)
            f_c_raw = jnp.linalg.solve(H_f, -g_f)

            # Enforce friction cone and normal limits
            f_c_clipped = []
            for i in range(self.n_contacts):
                base_idx = CONTACT_DIM_PER_POINT * i
                fx = f_c_raw[base_idx]
                fy = f_c_raw[base_idx + 1]
                fz = jnp.clip(
                    f_c_raw[base_idx + 2],
                    self.config.f_z_min,
                    self.config.f_z_max,
                )
                cone_limit = self.config.friction_coef * fz
                fx = jnp.clip(fx, -cone_limit, cone_limit)
                fy = jnp.clip(fy, -cone_limit, cone_limit)
                f_c_clipped.extend([fx, fy, fz])
            f_c = jnp.array(f_c_clipped, dtype=jnp.float32)

            net_wrench = M @ qddot + h - J_c.T @ f_c
            raw_torques = self.S @ net_wrench
            clipped_torques = jnp.clip(
                raw_torques, -self.config.tau_max, self.config.tau_max
            )

            qp_cost = 0.5 * (
                jnp.sum(jnp.square(qddot))
                + jnp.sum(jnp.square(f_c))
                + self.config.w_torque_reg * jnp.sum(jnp.square(clipped_torques))
            )

        return WBCControlOutput(
            torques=clipped_torques,
            qddot=qddot,
            contact_forces=f_c,
            posture_error=pos_error,
            qp_cost=qp_cost,
        )

    def step(
        self,
        q: jax.Array,
        qd: jax.Array,
        q_des: jax.Array,
        qd_des: jax.Array | None = None,
    ) -> WBCControlOutput:
        """One-step execution wrapper."""
        return self.solve(q=q, qd=qd, q_des=q_des, qd_des=qd_des)
