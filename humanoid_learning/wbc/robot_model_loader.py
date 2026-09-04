"""Robot Model Specification Loader for JAX Whole-Body Control and RL Training.

Supports parsing robot physical properties, kinematics, limits, and WBC tuning for ANY robot topology:
- Humanoids / Bipeds (2 stance contacts)
- Quadrupeds (4 stance contacts: LF, RF, LH, RH)
- Hexapods / Multi-legged systems (k stance contacts)
- Drones / Aerial Robots / Underwater vehicles (0 stance contacts)
- Fixed-base or Mobile Manipulators

Supports:
1. MuJoCo (`mujoco.MjModel.from_xml_path`) for MJCF XML and URDF models
2. Pinocchio (`pinocchio.buildModelFromUrdf`) for URDF models
3. Per-robot YAML configuration definitions
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

try:
    import mujoco

    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

try:
    import pinocchio as pin

    _HAS_PINOCCHIO = hasattr(pin, "buildModelFromUrdf")
except (ImportError, AttributeError):
    _HAS_PINOCCHIO = False

try:
    import yaml

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# ==============================================================================
# Specification Constants & Fallback Defaults
# ==============================================================================
DEFAULT_FALLBACK_ACTUATOR_COUNT: int = 12
DEFAULT_FALLBACK_ROBOT_MASS_KG: float = 35.0
DEFAULT_FALLBACK_STANDING_HEIGHT_M: float = 0.85
DEFAULT_JOINT_LIMIT_LOWER_RAD: float = -2.5
DEFAULT_JOINT_LIMIT_UPPER_RAD: float = 2.5
DEFAULT_MAX_TORQUE_NM: float = 100.0
GRAVITY_ACCELERATION: float = 9.81
FZ_MAX_MULTIPLIER_FACTOR: float = 2.5
MIN_FZ_MAX_LOWER_BOUND_N: float = 1500.0
MIN_VALID_ROOT_Z_M: float = 0.1

# Default Kinematic Floating Base Dimensions
FLOATING_BASE_POS_DIM: int = 7  # 3 pos + 4 quat
FLOATING_BASE_VEL_DIM: int = 6  # 3 lin + 3 ang


@dataclass
class RobotModelSpec:
    """Complete, fully general specification of ANY robot for WBC and RL training.

    Applicable to humanoids, bipeds, quadrupeds, drones (0 contacts), and manipulators.
    """

    name: str
    nq: int  # Number of generalized position coordinates (gc's)
    nv: int  # Number of generalized velocity coordinates (gv's)
    n_act: int  # Number of actuated joints/thrusters (nu)
    joint_names: List[str]  # Actuated joint names in kinematic order
    actuator_names: List[str]  # Actuator names
    q_nominal: List[float]  # Nominal standing/resting joint positions [n_act] (rad)
    joint_limits_lower: List[float]  # Lower joint limits [n_act] (rad)
    joint_limits_upper: List[float]  # Upper joint limits [n_act] (rad)
    torque_limits: List[float]  # Maximum actuator torques/forces [n_act] (N*m / N)
    total_mass: float  # Total robot mass (kg)
    body_masses: Dict[str, float]  # Mass per body link (kg)
    default_standing_height: float  # Nominal base z height (m)
    contact_body_names: List[str] = field(
        default_factory=list
    )  # Contact foot/wheel/end-effector body names (empty for drones)
    limb_joint_indices: Dict[str, List[int]] = field(
        default_factory=dict
    )  # Generic limb/branch joint groupings (e.g. LF, RF, LH, RH, left_leg, etc.)
    wbc_config: Dict[str, float] = field(
        default_factory=dict
    )  # Per-robot WBC tuning hyperparameters
    source_file: str = ""  # Source XML/URDF path
    backend: str = "mujoco"  # Parser backend used ('mujoco', 'pinocchio', or 'yaml')

    @property
    def is_floating_base(self) -> bool:
        """True if robot has a 6-DOF unactuated floating base."""
        return self.nv > self.n_act

    @property
    def num_contact_points(self) -> int:
        """Number of ground contact points (0 for aerial robots/drones)."""
        return len(self.contact_body_names)

    @property
    def num_floating_base_pos(self) -> int:
        """Number of floating base generalized position coordinates."""
        return max(0, self.nq - self.n_act)

    @property
    def num_floating_base_vel(self) -> int:
        """Number of floating base generalized velocity coordinates."""
        return max(0, self.nv - self.n_act)

    # Convenience properties for backward compatibility
    @property
    def left_leg_indices(self) -> List[int]:
        if "left_leg" in self.limb_joint_indices:
            return self.limb_joint_indices["left_leg"]
        if "FL" in self.limb_joint_indices or "LF" in self.limb_joint_indices:
            return self.limb_joint_indices.get(
                "FL", self.limb_joint_indices.get("LF", [])
            )
        return list(range(self.n_act // 2)) if self.n_act > 0 else []

    @property
    def right_leg_indices(self) -> List[int]:
        if "right_leg" in self.limb_joint_indices:
            return self.limb_joint_indices["right_leg"]
        if "FR" in self.limb_joint_indices or "RF" in self.limb_joint_indices:
            return self.limb_joint_indices.get(
                "FR", self.limb_joint_indices.get("RF", [])
            )
        return list(range(self.n_act // 2, self.n_act)) if self.n_act > 0 else []

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotModelSpec":
        """Create specification from dictionary, filtering unknown keys."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        # Backward compatibility for legacy left_leg_indices/right_leg_indices in YAML
        if "limb_joint_indices" not in filtered or not filtered["limb_joint_indices"]:
            limbs = {}
            if "left_leg_indices" in data and data["left_leg_indices"]:
                limbs["left_leg"] = data["left_leg_indices"]
            if "right_leg_indices" in data and data["right_leg_indices"]:
                limbs["right_leg"] = data["right_leg_indices"]
            filtered["limb_joint_indices"] = limbs
        return cls(**filtered)

    def get_q_nominal_jax(self) -> jax.Array:
        """Return nominal joint positions as JAX array [n_act]."""
        return jnp.array(self.q_nominal, dtype=jnp.float32)

    def get_torque_limits_jax(self) -> jax.Array:
        """Return torque limits as JAX array [n_act]."""
        return jnp.array(self.torque_limits, dtype=jnp.float32)

    def get_joint_bounds_jax(self) -> Tuple[jax.Array, jax.Array]:
        """Return lower and upper joint limits as JAX arrays [n_act]."""
        return (
            jnp.array(self.joint_limits_lower, dtype=jnp.float32),
            jnp.array(self.joint_limits_upper, dtype=jnp.float32),
        )


def _find_repo_model_path(robot_name_or_path: str) -> str:
    """Find robot model path across standard repository locations."""
    if os.path.exists(robot_name_or_path):
        return os.path.abspath(robot_name_or_path)

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(curr_dir, "..", ".."))

    # LINT.IfChange(supported_robots)
    search_candidates = [
        os.path.join(repo_root, robot_name_or_path),
        os.path.join(
            repo_root,
            "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml",
        ),
        os.path.join(
            repo_root, "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"
        ),
        os.path.join(
            repo_root,
            "robot_models/unitree_r1/unitree_r1_description/urdf/r1.xml",
        ),
        os.path.join(
            repo_root,
            "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf",
        ),
        os.path.join(
            repo_root,
            "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml",
        ),
        os.path.join(
            repo_root,
            "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
        ),
    ]

    clean_name = robot_name_or_path.lower().replace("-", "_").replace(" ", "_")
    if "g1" in clean_name:
        for p in search_candidates:
            if "g1" in p and os.path.exists(p):
                return p
    elif "r1" in clean_name:
        for p in search_candidates:
            if ("r1.xml" in p or "R1.urdf" in p) and os.path.exists(p):
                return p
    elif "atlas" in clean_name:
        for p in search_candidates:
            if "atlas.xml" in p and os.path.exists(p):
                return p
            if "atlas" in p and "gantry" not in p and os.path.exists(p):
                return p

    for p in search_candidates:
        if os.path.exists(p):
            return p
    # LINT.ThenChange(//humanoid_learning/training/generate_robot_spec.py:supported_robots)

    raise FileNotFoundError(
        f"Could not locate robot model file for '{robot_name_or_path}'."
    )


# LINT.IfChange(robot_limb_discovery)
def _discover_limb_groupings(joint_names: List[str]) -> Dict[str, List[int]]:
    """Automatically discover limb joint groups for bipeds, quadrupeds, hexapods, and arms."""
    limbs: Dict[str, List[int]] = {}

    for i, jname in enumerate(joint_names):
        low = jname.lower()

        # Quadruped leg prefixes / identifiers (FL, FR, RL, RR or LF, RF, LH, RH)
        if any(p in low for p in ["fl_", "front_left", "lf_", "left_front"]):
            limbs.setdefault("FL", []).append(i)
        elif any(p in low for p in ["fr_", "front_right", "rf_", "right_front"]):
            limbs.setdefault("FR", []).append(i)
        elif any(p in low for p in ["rl_", "rear_left", "lh_", "hind_left", "left_hind"]):
            limbs.setdefault("RL", []).append(i)
        elif any(p in low for p in ["rr_", "rear_right", "rh_", "hind_right", "right_hind"]):
            limbs.setdefault("RR", []).append(i)

        # Humanoid / Biped groupings
        elif ("left" in low or low.startswith("l_")) and any(
            k in low for k in ["hip", "knee", "ankle", "thigh", "calf", "leg"]
        ):
            limbs.setdefault("left_leg", []).append(i)
        elif ("right" in low or low.startswith("r_")) and any(
            k in low for k in ["hip", "knee", "ankle", "thigh", "calf", "leg"]
        ):
            limbs.setdefault("right_leg", []).append(i)
        elif ("left" in low or low.startswith("l_")) and any(
            k in low for k in ["shoulder", "elbow", "wrist", "arm"]
        ):
            limbs.setdefault("left_arm", []).append(i)
        elif ("right" in low or low.startswith("r_")) and any(
            k in low for k in ["shoulder", "elbow", "wrist", "arm"]
        ):
            limbs.setdefault("right_arm", []).append(i)
        elif any(k in low for k in ["waist", "torso", "spine", "neck"]):
            limbs.setdefault("torso", []).append(i)

    # Fallback generic grouping if no standard names match
    if not limbs and joint_names:
        limbs["actuators"] = list(range(len(joint_names)))

    return limbs
# LINT.ThenChange(//humanoid_learning/retargeting/joint_mapper.py:robot_limb_discovery)


def load_robot_spec_from_pinocchio(urdf_path: str) -> RobotModelSpec:
    """Extract complete RobotModelSpec from a URDF file using Pinocchio for any robot."""
    if not _HAS_PINOCCHIO:
        raise ImportError(
            "Pinocchio with URDF parser is required to parse models via Pinocchio."
        )

    # Build model with 6-DOF floating base (FreeFlyer)
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    robot_name = (
        model.name or os.path.splitext(os.path.basename(urdf_path))[0]
    )

    nq = int(model.nq)  # 3 pos + 4 quat + n_act
    nv = int(model.nv)  # 6 vel + n_act
    n_act = nv - FLOATING_BASE_VEL_DIM  # Exclude floating base

    # Actuated joint names
    joint_names = [model.names[i] for i in range(2, model.njoints)]
    actuator_names = [f"{jname}_actuator" for jname in joint_names]

    # Neutral standing configuration
    q_neutral = pin.neutral(model)
    q_nominal = [float(x) for x in q_neutral[FLOATING_BASE_POS_DIM:]]

    # Position limits
    lower_limits = [float(x) for x in model.lowerPositionLimit[FLOATING_BASE_POS_DIM:]]
    upper_limits = [float(x) for x in model.upperPositionLimit[FLOATING_BASE_POS_DIM:]]

    # Effort/torque limits
    torque_limits = []
    for x in model.effortLimit[FLOATING_BASE_VEL_DIM:]:
        val = float(x)
        torque_limits.append(val if val > 0 else DEFAULT_MAX_TORQUE_NM)

    # Masses
    total_mass = float(pin.computeTotalMass(model))
    body_masses = {
        model.names[i]: float(model.inertias[i].mass)
        for i in range(1, model.njoints)
    }

    # Discover limb groupings
    limbs = _discover_limb_groupings(joint_names)

    # Discover contact bodies (empty for drones / aerial vehicles)
    contact_bodies = []
    for bname in body_masses.keys():
        low = bname.lower()
        if any(
            k in low for k in ["foot", "ankle_roll", "toe", "sole", "heel", "wheel", "pad", "tip"]
        ):
            contact_bodies.append(bname)

    standing_height = DEFAULT_FALLBACK_STANDING_HEIGHT_M

    # Robot-specific tuned WBC hyperparameters
    max_tau = (
        float(max(torque_limits)) if torque_limits else DEFAULT_MAX_TORQUE_NM
    )
    wbc_tuning = {
        "w_base_acc": 100.0,
        "w_posture": 10.0,
        "w_contact_acc": 1000.0,
        "w_force_reg": 1e-4,
        "w_torque_reg": 1e-4,
        "kp_posture": 100.0,
        "kd_posture": 10.0,
        "friction_coef": 0.6,
        "f_z_min": 5.0,
        "f_z_max": float(
            max(
                MIN_FZ_MAX_LOWER_BOUND_N,
                total_mass * GRAVITY_ACCELERATION * FZ_MAX_MULTIPLIER_FACTOR,
            )
        ),
        "tau_max": max_tau,
    }

    return RobotModelSpec(
        name=robot_name,
        nq=nq,
        nv=nv,
        n_act=n_act,
        joint_names=joint_names,
        actuator_names=actuator_names,
        q_nominal=q_nominal,
        joint_limits_lower=lower_limits,
        joint_limits_upper=upper_limits,
        torque_limits=torque_limits,
        total_mass=total_mass,
        body_masses=body_masses,
        default_standing_height=standing_height,
        contact_body_names=contact_bodies,
        limb_joint_indices=limbs,
        wbc_config=wbc_tuning,
        source_file=urdf_path,
        backend="pinocchio",
    )


def load_robot_spec_from_mujoco(xml_path: str) -> RobotModelSpec:
    """Extract complete RobotModelSpec dynamically from an MJCF XML / URDF using MuJoCo."""
    if not _HAS_MUJOCO:
        raise ImportError(
            "MuJoCo is required to parse MJCF XML model specifications."
        )

    model = mujoco.MjModel.from_xml_path(xml_path)
    robot_name = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MODEL, 0)
        or os.path.splitext(os.path.basename(xml_path))[0]
    )

    # 1. Degrees of Freedom
    nq = int(model.nq)  # Generalized coordinates (gc's)
    nv = int(model.nv)  # Generalized velocities (gv's)
    n_act = int(model.nu)  # Actuators

    joint_names = []
    actuator_names = []
    q_nominal = []
    joint_limits_lower = []
    joint_limits_upper = []
    torque_limits = []

    for i in range(n_act):
        act_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            or f"actuator_{i}"
        )
        actuator_names.append(act_name)

        jnt_id = model.actuator_trnid[i, 0]
        jnt_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            or f"joint_{i}"
        )
        joint_names.append(jnt_name)

        # Extract joint limits
        jnt_range = model.jnt_range[jnt_id]
        if np.all(jnt_range == 0):
            lower, upper = (
                DEFAULT_JOINT_LIMIT_LOWER_RAD,
                DEFAULT_JOINT_LIMIT_UPPER_RAD,
            )
        else:
            lower, upper = float(jnt_range[0]), float(jnt_range[1])
        joint_limits_lower.append(lower)
        joint_limits_upper.append(upper)

        # Extract actuator torque limits
        force_range = model.actuator_forcerange[i]
        if np.all(force_range == 0) or force_range[1] <= 0:
            tau_lim = DEFAULT_MAX_TORQUE_NM
        else:
            tau_lim = float(force_range[1])
        torque_limits.append(tau_lim)

        # Extract nominal initial joint angle from model.qpos0
        qpos_idx = model.jnt_qposadr[jnt_id]
        q0 = float(model.qpos0[qpos_idx])
        q_nominal.append(q0)

    # 2. Extract Link Masses and Total Robot Mass
    body_masses = {}
    total_mass = 0.0
    for b in range(model.nbody):
        bname = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
            or f"body_{b}"
        )
        bmass = float(model.body_mass[b])
        body_masses[bname] = bmass
        total_mass += bmass

    # 3. Discover Limb Groupings (biped, quadruped, hexapod, arm)
    limbs = _discover_limb_groupings(joint_names)

    # 4. Dynamically Extract Nominal Standing / Operating Height from Model
    if model.nq > n_act and model.qpos0[2] > MIN_VALID_ROOT_Z_M:
        torso_height = float(model.qpos0[2])
    elif model.nbody > 1 and model.body_pos[1, 2] > MIN_VALID_ROOT_Z_M:
        torso_height = float(model.body_pos[1, 2])
    else:
        max_z = float(np.max(model.body_pos[:, 2]))
        min_z = float(np.min(model.body_pos[:, 2]))
        torso_height = max(0.5, max_z - min_z)

    # 5. Discover Contact Stance Bodies (empty for drones / aerial vehicles)
    contact_bodies = []
    for bname in body_masses.keys():
        low = bname.lower()
        if any(
            k in low for k in ["foot", "ankle_roll", "toe", "sole", "heel", "wheel", "pad", "tip"]
        ):
            contact_bodies.append(bname)

    # Robot-specific tuned WBC hyperparameters
    max_tau = (
        float(max(torque_limits)) if torque_limits else DEFAULT_MAX_TORQUE_NM
    )
    wbc_tuning = {
        "w_base_acc": 100.0,
        "w_posture": 10.0,
        "w_contact_acc": 1000.0,
        "w_force_reg": 1e-4,
        "w_torque_reg": 1e-4,
        "kp_posture": 100.0,
        "kd_posture": 10.0,
        "friction_coef": 0.6,
        "f_z_min": 5.0,
        "f_z_max": float(
            max(
                MIN_FZ_MAX_LOWER_BOUND_N,
                total_mass * GRAVITY_ACCELERATION * FZ_MAX_MULTIPLIER_FACTOR,
            )
        ),
        "tau_max": max_tau,
    }

    return RobotModelSpec(
        name=robot_name,
        nq=nq,
        nv=nv,
        n_act=n_act,
        joint_names=joint_names,
        actuator_names=actuator_names,
        q_nominal=q_nominal,
        joint_limits_lower=joint_limits_lower,
        joint_limits_upper=joint_limits_upper,
        torque_limits=torque_limits,
        total_mass=float(total_mass),
        body_masses=body_masses,
        default_standing_height=float(torso_height),
        contact_body_names=contact_bodies,
        limb_joint_indices=limbs,
        wbc_config=wbc_tuning,
        source_file=xml_path,
        backend="mujoco",
    )


def load_robot_spec(
    model_path_or_name: str, backend: str = "auto"
) -> RobotModelSpec:
    """General loader: loads RobotModelSpec via MuJoCo, Pinocchio, or YAML config."""
    # 1. Direct YAML / JSON definition
    if model_path_or_name.endswith(".yaml") or model_path_or_name.endswith(
        ".yml"
    ):
        if not os.path.exists(model_path_or_name):
            resolved = _find_repo_model_path(model_path_or_name)
        else:
            resolved = model_path_or_name
        with open(resolved, "r") as f:
            if _HAS_YAML:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return RobotModelSpec.from_dict(data)

    if model_path_or_name.endswith(".json"):
        with open(model_path_or_name, "r") as f:
            data = json.load(f)
        return RobotModelSpec.from_dict(data)

    # 2. Resolve Model File Path
    resolved_path = _find_repo_model_path(model_path_or_name)

    # Backend selection: Pinocchio
    if (
        backend == "pinocchio" or (backend == "auto" and not _HAS_MUJOCO)
    ) and _HAS_PINOCCHIO:
        if resolved_path.endswith(".urdf"):
            return load_robot_spec_from_pinocchio(resolved_path)

    # Backend selection: MuJoCo (supports both MJCF .xml and .urdf)
    if _HAS_MUJOCO and (
        resolved_path.endswith(".xml") or resolved_path.endswith(".urdf")
    ):
        return load_robot_spec_from_mujoco(resolved_path)

    # Fallback default using named constants
    n_act = DEFAULT_FALLBACK_ACTUATOR_COUNT
    return RobotModelSpec(
        name="generic_robot",
        nq=n_act + FLOATING_BASE_POS_DIM,
        nv=n_act + FLOATING_BASE_VEL_DIM,
        n_act=n_act,
        joint_names=[f"joint_{i}" for i in range(n_act)],
        actuator_names=[f"actuator_{i}" for i in range(n_act)],
        q_nominal=[0.0] * n_act,
        joint_limits_lower=[-2.0] * n_act,
        joint_limits_upper=[2.0] * n_act,
        torque_limits=[DEFAULT_MAX_TORQUE_NM] * n_act,
        total_mass=DEFAULT_FALLBACK_ROBOT_MASS_KG,
        body_masses={"base": 20.0, "leg1": 7.5, "leg2": 7.5},
        default_standing_height=DEFAULT_FALLBACK_STANDING_HEIGHT_M,
        contact_body_names=["foot_1", "foot_2"],
        limb_joint_indices={
            "left_leg": list(range(n_act // 2)),
            "right_leg": list(range(n_act // 2, n_act)),
        },
        wbc_config={
            "w_base_acc": 100.0,
            "w_posture": 10.0,
            "w_contact_acc": 1000.0,
            "w_force_reg": 1e-4,
            "w_torque_reg": 1e-4,
            "kp_posture": 100.0,
            "kd_posture": 10.0,
            "friction_coef": 0.6,
            "f_z_min": 5.0,
            "f_z_max": 2000.0,
            "tau_max": DEFAULT_MAX_TORQUE_NM,
        },
        source_file=model_path_or_name,
        backend="fallback",
    )


def save_robot_spec(spec: RobotModelSpec, output_path: str):
    """Save RobotModelSpec to YAML or JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    spec_dict = spec.to_dict()

    if output_path.endswith(".yaml") or output_path.endswith(".yml"):
        if _HAS_YAML:
            with open(output_path, "w") as f:
                yaml.dump(spec_dict, f, default_flow_style=False, sort_keys=False)
            print(f"✓ Saved robot definition YAML to '{output_path}'.")
            return

    json_path = (
        output_path
        if output_path.endswith(".json")
        else output_path + ".json"
    )
    with open(json_path, "w") as f:
        json.dump(spec_dict, f, indent=2)
    print(f"✓ Saved robot definition JSON to '{json_path}'.")
