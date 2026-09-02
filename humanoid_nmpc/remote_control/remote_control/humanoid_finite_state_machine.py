"""****************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
****************************************************************************"""

"""Humanoid Finite State Machine (HumanoidFSM) & Virtual Gantry Manager.

Dynamically loads robot model topology, actuated joint ordering, nominal postures,
and height parameters directly from Pinocchio URDF models and OCS2 YAML configurations.

Provides:
1. ControlMode: Enumeration of whole-body actuator control regimes:
   - ZERO_TORQUE: De-energized / passive motors.
   - JOINT_PD: Joint position servoing to nominal posture.
   - GRAVITY_COMP: Gravity compensation / zero-G compliance.
   - WB_MPC: Active Whole-Body Model Predictive Control.
   - SAFETY: Damped PD decay transitioning to zero torque.
2. load_robot_config: Dynamic loader extracting joints & nominal posture from URDF/Pinocchio & YAML.
3. get_available_robots: Discovers all robot model directories available in workspace.
4. VirtualGantry: Suspension harness and vertical altitude stepping (±1 cm).
5. HumanoidFSM: Supervisory state machine governing mode transitions and torque generation.
"""

from enum import Enum
import glob
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import numpy as np
import yaml


class ControlMode(str, Enum):
    """Whole-body humanoid control modes."""

    ZERO_TORQUE = "ZERO_TORQUE"
    JOINT_PD = "JOINT_PD"
    GRAVITY_COMP = "GRAVITY_COMP"
    WB_MPC = "WB_MPC"
    SAFETY = "SAFETY"


# UI Presentation metadata for dashboard badges
MODE_METADATA: Dict[ControlMode, Dict[str, str]] = {
    ControlMode.ZERO_TORQUE: {
        "title": "ZERO_TORQUE",
        "badge_color": "#f38ba8",
        "icon": "🛑",
        "description": "All motors de-energized (0 N·m). Actuators freewheeling.",
        "bg_glow": "rgba(243, 139, 168, 0.15)",
    },
    ControlMode.JOINT_PD: {
        "title": "JOINT_PD",
        "badge_color": "#89b4fa",
        "icon": "🦾",
        "description": "Active joint PD servoing holding nominal standing posture.",
        "bg_glow": "rgba(137, 180, 250, 0.15)",
    },
    ControlMode.GRAVITY_COMP: {
        "title": "GRAVITY_COMP",
        "badge_color": "#cba6f7",
        "icon": "🪂",
        "description": "Inverse-dynamics gravity cancellation. Limbs float weightlessly.",
        "bg_glow": "rgba(203, 166, 247, 0.15)",
    },
    ControlMode.WB_MPC: {
        "title": "WB_MPC",
        "badge_color": "#a6e3a1",
        "icon": "⚡",
        "description": "Active Whole-Body NMPC tracking ground contact wrenches & gait.",
        "bg_glow": "rgba(166, 227, 161, 0.15)",
    },
    ControlMode.SAFETY: {
        "title": "SAFETY (DAMPED DECAY)",
        "badge_color": "#fab387",
        "icon": "⚠️",
        "description": "Soft landing: Damped PD gains smoothly decay to ZERO_TORQUE.",
        "bg_glow": "rgba(250, 179, 135, 0.15)",
    },
}

# Standard circular cycle sequence
CYCLE_MODES: List[ControlMode] = [
    ControlMode.ZERO_TORQUE,
    ControlMode.JOINT_PD,
    ControlMode.GRAVITY_COMP,
    ControlMode.WB_MPC,
]


def _find_workspace_dir(start_dir: Optional[str] = None) -> str:
    """Finds workspace root directory containing robot_models/."""
    current = os.path.abspath(start_dir or os.getcwd())
    while current and current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "robot_models")) and os.path.exists(
            os.path.join(current, "setup_env.sh")
        ):
            return current
        current = os.path.dirname(current)
    return os.path.abspath(start_dir or os.getcwd())


def _find_robot_directory(robot_name: Optional[str], ws: str) -> Optional[str]:
    """Finds the root directory for a given robot under robot_models/."""
    robot_models_root = os.path.join(ws, "robot_models")
    if not os.path.isdir(robot_models_root):
        return None

    if not robot_name:
        with os.scandir(robot_models_root) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                if entry.is_dir() and not entry.name.startswith("."):
                    return entry.path
        return None

    target = robot_name.strip().lower()
    with os.scandir(robot_models_root) as entries:
        for entry in entries:
            if entry.is_dir() and not entry.name.startswith("."):
                name = entry.name.lower()
                if name == target or target in name.split("_") or name.endswith(target):
                    return entry.path
    return None


_ROBOT_CONFIG_CACHE: Dict[str, Dict[str, Any]] = {}


def load_robot_config(
    robot_name: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    use_pinocchio: bool = False,
    force_reload: bool = False,
) -> Dict[str, Any]:
    """Dynamically loads robot joints, limits, and nominal configuration from URDF/Pinocchio & YAML files."""
    ws = _find_workspace_dir(workspace_dir)
    robot_dir = _find_robot_directory(robot_name, ws)
    resolved_name = (
        os.path.basename(robot_dir) if robot_dir else (robot_name or "humanoid")
    )
    cache_key = f"{ws}_{robot_dir or resolved_name.lower()}"

    if not force_reload and cache_key in _ROBOT_CONFIG_CACHE:
        return _ROBOT_CONFIG_CACHE[cache_key]

    # 1. Discover URDF, Task YAML, Reference YAML, Controller YAML, and MJCF files in robot directory
    if robot_dir:
        urdf_matches = glob.glob(
            os.path.join(robot_dir, "**/urdf/*.urdf"), recursive=True
        ) or glob.glob(os.path.join(robot_dir, "**/*.urdf"), recursive=True)
        task_matches = glob.glob(
            os.path.join(robot_dir, "**/config/mpc/task.yaml"), recursive=True
        )
        ref_matches = glob.glob(
            os.path.join(robot_dir, "**/config/command/reference.yaml"),
            recursive=True,
        )
        controller_matches = glob.glob(
            os.path.join(robot_dir, "**/config/controller/*.yaml"),
            recursive=True,
        )
        mjcf_matches = glob.glob(
            os.path.join(robot_dir, "**/urdf/*.xml"), recursive=True
        ) or glob.glob(os.path.join(robot_dir, "**/*.xml"), recursive=True)
    else:
        urdf_matches, task_matches, ref_matches, controller_matches, mjcf_matches = (
            [],
            [],
            [],
            [],
            [],
        )

    urdf_path = sorted(urdf_matches)[0] if urdf_matches else None
    task_path = sorted(task_matches)[0] if task_matches else None
    ref_path = sorted(ref_matches)[0] if ref_matches else None
    controller_path = sorted(controller_matches)[0] if controller_matches else None
    mjcf_path = sorted(mjcf_matches)[0] if mjcf_matches else None

    # 2. Extract joint names and mass from Pinocchio or URDF XML
    all_joint_names: List[str] = []
    pinocchio_model = None
    total_mass = 0.0

    if use_pinocchio and urdf_path and os.path.exists(urdf_path):
        try:
            import numpy as _np

            # Only import pinocchio if numpy is not 2.x (pinocchio_pywrap requires numpy 1.x ABI)
            if not _np.__version__.startswith("2."):
                import pinocchio as pin

                pinocchio_model = pin.buildModelFromUrdf(urdf_path)
                for i in range(1, pinocchio_model.njoints):
                    jname = pinocchio_model.names[i]
                    if jname not in ("root_joint", "universe", "root"):
                        all_joint_names.append(jname)
                total_mass = sum(inertial.mass for inertial in pinocchio_model.inertias)
        except Exception:
            pinocchio_model = None

    # Fallback to URDF XML parsing
    if urdf_path and os.path.exists(urdf_path):
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            if not all_joint_names:
                for j in root.findall("joint"):
                    jtype = j.get("type", "")
                    if jtype in ("revolute", "continuous", "prismatic"):
                        all_joint_names.append(j.get("name"))
            if total_mass <= 0.0:
                for link in root.findall("link"):
                    inertial = link.find("inertial")
                    if inertial is not None:
                        mass_elem = inertial.find("mass")
                        if mass_elem is not None:
                            total_mass += float(mass_elem.get("value", 0.0))
        except Exception:
            pass

    # 3. Parse task.yaml (initialState, model_settings, gains)
    task_data: Dict[str, Any] = {}
    joint_val_map: Dict[str, float] = {}
    pelvis_height = 0.75

    if task_path and os.path.exists(task_path):
        with open(task_path, "r", encoding="utf-8") as f:
            in_init = False
            for line in f:
                line_str = line.strip()
                if line_str.startswith("initialState:"):
                    in_init = True
                    continue
                if in_init and (
                    line_str.startswith("Q:")
                    or (
                        not line.startswith(" ")
                        and not line_str.startswith("#")
                        and line_str
                    )
                ):
                    break
                if in_init and "#" in line_str:
                    parts = line_str.split("#", 1)
                    left = parts[0].strip()
                    if ":" in left:
                        k = left.split(":", 1)[0].strip().replace('"', "")
                        v_str = left.split(":", 1)[1].strip()
                        if v_str:
                            try:
                                v = float(v_str)
                                comment = (
                                    parts[1].strip().split()[0]
                                    if parts[1].strip()
                                    else ""
                                )
                                if k == "(8,0)":
                                    pelvis_height = v
                                elif (
                                    comment
                                    and not comment.startswith("h_com")
                                    and not comment.startswith("L_")
                                    and not comment.startswith("p_base")
                                    and not comment.startswith("theta_base")
                                    and not comment.startswith(";")
                                ):
                                    joint_val_map[comment] = v
                            except ValueError:
                                pass
            f.seek(0)
            try:
                task_data = yaml.safe_load(f) or {}
            except Exception:
                task_data = {}

    # 4. Parse reference.yaml (defaultBaseHeight, defaultJointState)
    ref_data: Dict[str, Any] = {}
    if ref_path and os.path.exists(ref_path):
        try:
            with open(ref_path, "r", encoding="utf-8") as f:
                ref_data = yaml.safe_load(f) or {}
                if "defaultBaseHeight" in ref_data and "(8,0)" not in joint_val_map:
                    pelvis_height = float(ref_data["defaultBaseHeight"])
        except Exception:
            pass

    # 5. Parse controller gains YAML (joint_pd_gains.yaml)
    joint_kp_map: Dict[str, float] = {}
    joint_kd_map: Dict[str, float] = {}
    default_kp_val = None
    default_kd_val = None

    if controller_path and os.path.exists(controller_path):
        try:
            with open(controller_path, "r", encoding="utf-8") as f:
                ctrl_data = yaml.safe_load(f) or {}
                def_gains = ctrl_data.get("default_gains", ctrl_data.get("default", {}))
                if def_gains:
                    default_kp_val = float(def_gains.get("kp", 150.0))
                    default_kd_val = float(def_gains.get("kd", 10.0))
                jgains = ctrl_data.get("joint_gains", {})
                for jname, gdict in jgains.items():
                    if isinstance(gdict, dict):
                        if "kp" in gdict:
                            joint_kp_map[jname] = float(gdict["kp"])
                        if "kd" in gdict:
                            joint_kd_map[jname] = float(gdict["kd"])
        except Exception:
            pass

    # 6. Extract fixed joints and build active joint list
    model_settings = task_data.get("model_settings", {})
    fixed_joints = set(model_settings.get("fixedJointNames", []))
    robot_display_name = model_settings.get("robotName", resolved_name).upper()

    if all_joint_names:
        active_joint_names = [j for j in all_joint_names if j not in fixed_joints]
    else:
        active_joint_names = list(joint_val_map.keys())

    # 7. Build ordered nominal_q vector matching active joint ordering
    nominal_q = np.zeros(len(active_joint_names), dtype=np.float64)
    for i, jname in enumerate(active_joint_names):
        if jname in joint_val_map:
            nominal_q[i] = joint_val_map[jname]

    # 8. Compute default and per-joint PD gain vectors
    if default_kp_val is not None:
        default_kp = default_kp_val
        default_kd = (
            default_kd_val
            if default_kd_val is not None
            else float(2.0 * np.sqrt(default_kp))
        )
    else:
        gains_cfg = task_data.get(
            "pd_gains", task_data.get("model_settings", {}).get("pd_gains", {})
        )
        if gains_cfg and "kp" in gains_cfg:
            default_kp = float(gains_cfg["kp"])
            default_kd = float(
                gains_cfg.get(
                    "kd", 2.0 * np.sqrt(default_kp) if default_kp > 0 else 10.0
                )
            )
        else:
            default_kp = float(
                np.clip(1.5 * total_mass if total_mass > 0 else 150.0, 100.0, 300.0)
            )
            default_kd = float(
                np.clip(
                    2.0 * np.sqrt(default_kp) if default_kp > 0 else 10.0,
                    5.0,
                    25.0,
                )
            )

    kp_vec = np.full(len(active_joint_names), default_kp, dtype=np.float64)
    kd_vec = np.full(len(active_joint_names), default_kd, dtype=np.float64)
    for i, jname in enumerate(active_joint_names):
        if jname in joint_kp_map:
            kp_vec[i] = joint_kp_map[jname]
        if jname in joint_kd_map:
            kd_vec[i] = joint_kd_map[jname]

    config: Dict[str, Any] = {
        "name": robot_display_name,
        "robot_name": resolved_name,
        "robot_dir": robot_dir,
        "urdf_path": urdf_path,
        "task_path": task_path,
        "ref_path": ref_path,
        "controller_path": controller_path,
        "mjcf_path": mjcf_path,
        "nominal_pelvis_height_bent": pelvis_height,
        "nominal_pelvis_height_straight": pelvis_height + 0.14,
        "default_kp": default_kp,
        "default_kd": default_kd,
        "kp_vector": kp_vec,
        "kd_vector": kd_vec,
        "total_mass": total_mass,
        "num_actuators": len(active_joint_names),
        "joint_names": active_joint_names,
        "all_joint_names": all_joint_names,
        "fixed_joints": list(fixed_joints),
        "nominal_q": nominal_q,
        "pinocchio_model": pinocchio_model,
    }

    _ROBOT_CONFIG_CACHE[cache_key] = config
    return config


def get_available_robots(workspace_dir: Optional[str] = None) -> List[str]:
    """Returns all discovered robot package names under robot_models/."""
    ws = _find_workspace_dir(workspace_dir)
    robot_models_root = os.path.join(ws, "robot_models")
    if not os.path.isdir(robot_models_root):
        return []
    with os.scandir(robot_models_root) as entries:
        robots = [
            entry.name
            for entry in entries
            if entry.is_dir() and not entry.name.startswith(".")
        ]
    return sorted(robots)


class VirtualGantry:
    """Virtual Gantry suspension and vertical altitude manager."""

    def __init__(
        self,
        robot_name: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        initial_height: Optional[float] = None,
        is_locked: bool = True,
        step_size: float = 0.01,
        min_height: float = 0.30,
        max_height: float = 1.40,
    ):
        self.workspace_dir = workspace_dir
        self.cfg = load_robot_config(
            robot_name=robot_name, workspace_dir=self.workspace_dir
        )
        self.robot_name = self.cfg["robot_name"]

        self.height = (
            initial_height
            if initial_height is not None
            else self.cfg["nominal_pelvis_height_bent"]
        )
        self.nominal_touch_height = self.cfg["nominal_pelvis_height_bent"]
        self.is_locked = is_locked
        self.step_size = step_size
        self.min_height = min_height
        self.max_height = max_height

    def lock(self):
        """Locks pelvis to gantry anchor."""
        self.is_locked = True

    def release(self):
        """Releases pelvis for free-floating locomotion."""
        self.is_locked = False

    def toggle_lock(self) -> bool:
        """Toggles lock state and returns new state."""
        self.is_locked = not self.is_locked
        return self.is_locked

    def step_up(self, delta: Optional[float] = None) -> float:
        """Moves gantry up by delta (default +1 cm / +0.01 m)."""
        d = delta if delta is not None else self.step_size
        self.height = min(self.max_height, self.height + d)
        return self.height

    def step_down(self, delta: Optional[float] = None) -> float:
        """Moves gantry down by delta (default -1 cm / -0.01 m)."""
        d = delta if delta is not None else self.step_size
        self.height = max(self.min_height, self.height - d)
        return self.height

    def set_height(self, height: float) -> float:
        """Sets gantry height clamped to valid range."""
        self.height = max(self.min_height, min(self.max_height, float(height)))
        return self.height

    def auto_calibrate_ground_touch(self, foot_clearance: float = 0.0) -> float:
        """Calibrates gantry height so nominal stance feet barely touch the ground."""
        self.height = self.cfg["nominal_pelvis_height_bent"] + foot_clearance
        self.is_locked = True
        return self.height

    def apply_to_mujoco(self, model, data):
        """Applies virtual gantry constraint to MuJoCo physics state."""
        if not self.is_locked or data is None:
            return

        # Pin floating base root pose
        data.qpos[0] = 0.0  # x
        data.qpos[1] = 0.0  # y
        data.qpos[2] = self.height  # z
        data.qpos[3] = 1.0  # quat w (level)
        data.qpos[4] = 0.0  # quat x
        data.qpos[5] = 0.0  # quat y
        data.qpos[6] = 0.0  # quat z

        # Zero floating base root linear & angular velocity
        data.qvel[0:6] = 0.0


class HumanoidFSM:
    """Humanoid Supervisory Finite State Machine."""

    def __init__(
        self,
        robot_name: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        initial_mode: ControlMode = ControlMode.ZERO_TORQUE,
        safety_decay_duration: float = 2.5,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
    ):
        self.workspace_dir = workspace_dir
        self.cfg = load_robot_config(
            robot_name=robot_name, workspace_dir=self.workspace_dir
        )
        self.robot_name = self.cfg["robot_name"]

        self.current_mode = initial_mode
        self.gantry = VirtualGantry(
            robot_name=self.robot_name,
            workspace_dir=self.workspace_dir,
            initial_height=self.cfg["nominal_pelvis_height_bent"],
        )

        self.nominal_q = self.cfg["nominal_q"].copy()
        self.joint_names = self.cfg["joint_names"]
        self.num_actuators = len(self.nominal_q)

        # Gain vectors from controller YAML or scalar overrides
        if kp is not None:
            self.kp_vector = np.full(self.num_actuators, float(kp), dtype=np.float64)
            self.default_kp = float(kp)
        else:
            self.kp_vector = self.cfg["kp_vector"].copy()
            self.default_kp = self.cfg["default_kp"]

        if kd is not None:
            self.kd_vector = np.full(self.num_actuators, float(kd), dtype=np.float64)
            self.default_kd = float(kd)
        else:
            self.kd_vector = self.cfg["kd_vector"].copy()
            self.default_kd = self.cfg["default_kd"]

        # Safety mode decay parameters
        self.safety_decay_duration = safety_decay_duration
        self._safety_start_time: Optional[float] = None
        self._safety_hold_q: Optional[np.ndarray] = None
        self._safety_initial_kp_vec = self.kp_vector.copy()
        self._safety_initial_kd_vec = self.kd_vector.copy()

        # Joint PD gradual snap transition parameters
        self.joint_pd_snap_duration = 2.0
        self._joint_pd_start_time: Optional[float] = None
        self._joint_pd_start_q: Optional[np.ndarray] = None

        # State transition listeners
        self._listeners: List[Callable[[ControlMode, ControlMode], None]] = []

    def set_robot(self, robot_name: str):
        """Switches the active robot model dynamically from disk configs."""
        self.cfg = load_robot_config(
            robot_name=robot_name, workspace_dir=self.workspace_dir
        )
        self.robot_name = self.cfg["robot_name"]
        self.gantry = VirtualGantry(
            robot_name=self.robot_name,
            workspace_dir=self.workspace_dir,
            initial_height=self.cfg["nominal_pelvis_height_bent"],
        )
        self.nominal_q = self.cfg["nominal_q"].copy()
        self.joint_names = self.cfg["joint_names"]
        self.num_actuators = len(self.nominal_q)
        self.kp_vector = self.cfg["kp_vector"].copy()
        self.kd_vector = self.cfg["kd_vector"].copy()
        self.default_kp = self.cfg["default_kp"]
        self.default_kd = self.cfg["default_kd"]

    def add_transition_listener(
        self, callback: Callable[[ControlMode, ControlMode], None]
    ):
        """Registers a callback invoked on state transitions (old_mode, new_mode)."""
        self._listeners.append(callback)

    def set_mode(
        self, new_mode: Union[ControlMode, str], current_q: Optional[np.ndarray] = None
    ) -> ControlMode:
        """Sets the active control mode."""
        if isinstance(new_mode, str):
            new_mode = ControlMode(new_mode)

        if new_mode == self.current_mode:
            return self.current_mode

        old_mode = self.current_mode
        self.current_mode = new_mode

        if new_mode == ControlMode.SAFETY:
            self._safety_start_time = time.time()
            self._safety_hold_q = (
                current_q.copy() if current_q is not None else self.nominal_q.copy()
            )
            self._safety_initial_kp_vec = self.kp_vector.copy()
            self._safety_initial_kd_vec = self.kd_vector.copy()
        else:
            self._safety_start_time = None
            self._safety_hold_q = None

        if new_mode == ControlMode.JOINT_PD:
            self._joint_pd_start_time = time.time()
            self._joint_pd_start_q = current_q.copy() if current_q is not None else None
        else:
            self._joint_pd_start_time = None
            self._joint_pd_start_q = None

        for cb in self._listeners:
            try:
                cb(old_mode, new_mode)
            except Exception:
                pass

        if getattr(self, "_fsm_command_pub", None) is not None:
            try:
                from std_msgs.msg import String

                msg = String()
                msg.data = new_mode.name
                self._fsm_command_pub.publish(msg)
            except Exception:
                pass

        return self.current_mode

    def cycle_next_mode(self, current_q: Optional[np.ndarray] = None) -> ControlMode:
        """Cycles to the next standard operational mode in sequence."""
        if self.current_mode in CYCLE_MODES:
            idx = CYCLE_MODES.index(self.current_mode)
            next_mode = CYCLE_MODES[(idx + 1) % len(CYCLE_MODES)]
        else:
            next_mode = ControlMode.ZERO_TORQUE
        return self.set_mode(next_mode, current_q=current_q)

    def cycle_prev_mode(self, current_q: Optional[np.ndarray] = None) -> ControlMode:
        """Cycles to the previous standard operational mode in sequence."""
        if self.current_mode in CYCLE_MODES:
            idx = CYCLE_MODES.index(self.current_mode)
            prev_mode = CYCLE_MODES[(idx - 1) % len(CYCLE_MODES)]
        else:
            next_mode = ControlMode.ZERO_TORQUE
        return self.set_mode(prev_mode, current_q=current_q)

    def trigger_safety(self, current_q: Optional[np.ndarray] = None) -> ControlMode:
        """Triggers the damped safety landing mode."""
        return self.set_mode(ControlMode.SAFETY, current_q=current_q)

    def get_safety_progress(
        self, now: Optional[float] = None
    ) -> Tuple[float, float, float]:
        """Returns (decay_fraction, current_kp, current_kd) during SAFETY mode."""
        if self.current_mode != ControlMode.SAFETY or self._safety_start_time is None:
            return 1.0, self.default_kp, self.default_kd

        current_time = now if now is not None else time.time()
        elapsed = current_time - self._safety_start_time
        fraction = max(0.0, 1.0 - (elapsed / self.safety_decay_duration))

        current_kp = self.default_kp * fraction
        current_kd = self.default_kd * fraction

        return fraction, current_kp, current_kd

    def get_joint_pd_progress(
        self, now: Optional[float] = None
    ) -> Tuple[float, bool, np.ndarray]:
        """Returns (snap_progress_fraction, is_snapping, target_q_interpolated) during JOINT_PD transition."""
        if (
            self.current_mode != ControlMode.JOINT_PD
            or self._joint_pd_start_time is None
        ):
            return 1.0, False, self.nominal_q.copy()

        current_time = now if now is not None else time.time()
        elapsed = current_time - self._joint_pd_start_time
        if elapsed >= self.joint_pd_snap_duration:
            return 1.0, False, self.nominal_q.copy()

        s = max(0.0, min(1.0, elapsed / max(1e-4, self.joint_pd_snap_duration)))
        # Minimum-jerk quintic blending polynomial: 6s^5 - 15s^4 + 10s^3
        alpha = s * s * s * (s * (s * 6.0 - 15.0) + 10.0)
        start_q = (
            self._joint_pd_start_q
            if self._joint_pd_start_q is not None
            else np.zeros_like(self.nominal_q)
        )
        target_q = (1.0 - alpha) * start_q + alpha * self.nominal_q
        return alpha, True, target_q

    def compute_torques(
        self,
        q: Optional[np.ndarray] = None,
        v: Optional[np.ndarray] = None,
        mj_model=None,
        mj_data=None,
        mpc_torques: Optional[np.ndarray] = None,
        now: Optional[float] = None,
    ) -> np.ndarray:
        """Computes commanded joint torques according to the active control mode."""
        # Extract joint states from MuJoCo data if available
        if mj_data is not None:
            if q is None:
                # 6 floating base coordinates + n actuators
                q = mj_data.qpos[7 : 7 + self.num_actuators]
            if v is None:
                v = mj_data.qvel[6 : 6 + self.num_actuators]

        if q is None:
            q = np.zeros(self.num_actuators, dtype=np.float64)
        if v is None:
            v = np.zeros(self.num_actuators, dtype=np.float64)

        # 1. ZERO_TORQUE Mode (All de-energized, passive freewheeling)
        if self.current_mode == ControlMode.ZERO_TORQUE:
            return np.zeros(self.num_actuators, dtype=np.float64)

        # 2. JOINT_PD Mode (Gradual minimum-jerk snap to nominal stance)
        elif self.current_mode == ControlMode.JOINT_PD:
            alpha, is_snapping, target_q = self.get_joint_pd_progress(now=now)
            if is_snapping:
                if self._joint_pd_start_q is None:
                    self._joint_pd_start_q = q.copy()

                start_q = self._joint_pd_start_q
                current_time = now if now is not None else time.time()
                elapsed = current_time - self._joint_pd_start_time
                s = max(
                    0.0,
                    min(1.0, elapsed / max(1e-4, self.joint_pd_snap_duration)),
                )
                alpha = s * s * s * (s * (s * 6.0 - 15.0) + 10.0)
                d_alpha = (30.0 * s**4 - 60.0 * s**3 + 30.0 * s**2) / max(
                    1e-4, self.joint_pd_snap_duration
                )

                target_q = (1.0 - alpha) * start_q + alpha * self.nominal_q
                target_qd = d_alpha * (self.nominal_q - start_q)

                # Softly ramp effective proportional gain (30% -> 100%)
                kp_eff = (0.3 + 0.7 * alpha) * self.kp_vector
                tau = kp_eff * (target_q - q) + self.kd_vector * (target_qd - v)
                return tau
            else:
                error = self.nominal_q - q
                tau = self.kp_vector * error - self.kd_vector * v
                return tau

        # 3. GRAVITY_COMP Mode
        elif self.current_mode == ControlMode.GRAVITY_COMP:
            if mj_data is not None and hasattr(mj_data, "qfrc_bias"):
                # qfrc_bias contains gravity and Coriolis forces
                tau_grav = mj_data.qfrc_bias[6 : 6 + self.num_actuators].copy()
                # Damping to stabilize posture against residual drift
                tau = tau_grav - (0.15 * self.kd_vector) * v
                return tau
            else:
                # Fallback: soft posture holding
                return (0.3 * self.kp_vector) * (
                    self.nominal_q - q
                ) - self.kd_vector * v

        # 4. WB_MPC Mode
        elif self.current_mode == ControlMode.WB_MPC:
            if mpc_torques is not None and len(mpc_torques) == self.num_actuators:
                return np.asarray(mpc_torques, dtype=np.float64)
            # Default fallback when MPC solver output is initializing
            return self.kp_vector * (self.nominal_q - q) - self.kd_vector * v

        # 5. SAFETY Mode (Damped PD Gain Decay)
        elif self.current_mode == ControlMode.SAFETY:
            fraction, _, _ = self.get_safety_progress(now=now)

            target_q = (
                self._safety_hold_q
                if self._safety_hold_q is not None
                else self.nominal_q
            )
            tau = (fraction * self._safety_initial_kp_vec) * (target_q - q) - (
                fraction * self._safety_initial_kd_vec
            ) * v

            # Once decay completes, automatically transition to ZERO_TORQUE
            if fraction <= 1e-4:
                self.set_mode(ControlMode.ZERO_TORQUE)

            return tau

        return np.zeros(self.num_actuators, dtype=np.float64)
