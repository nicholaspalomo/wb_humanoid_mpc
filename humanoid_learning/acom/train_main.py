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

"""CLI entrypoint to train and export aCOM models for humanoid robots."""

import argparse
import os
import re
import shutil

from humanoid_learning.acom.dataset_generator import AcomDatasetGenerator
from humanoid_learning.acom.train_acom import train_acom
from humanoid_learning.acom.export_acom import export_to_json, export_to_cpp_header

# LINT.IfChange(robot_paths)
# Default XML and URDF paths per robot. Keep in sync with Bazel data dependencies.
_ROBOT_CONFIGS = {
    "g1": {
        "xml": "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml",
        "urdf": "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
        "fixed_joints": [
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    },
    "atlas": {
        "xml": "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml",
        "urdf": "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
        "fixed_joints": ["l_arm_wry", "l_arm_wrx", "r_arm_wry", "r_arm_wrx"],
    },
    # The EngineAI SA01 is LEGS ONLY: twelve joints, six per leg, with no arms, no waist and no torso link - its
    # only body above the legs is base_link, which is the floating base itself. There is therefore nothing to fix
    # here, unlike the wrists the other two hold at zero, and the aCOM offset is a function of the leg joints alone.
    #
    # That also sets expectations for what the coordinate buys on this robot. The aCOM exists because limb motion
    # decouples whole-body orientation from base orientation, and the limbs doing most of that on Atlas and G1 are
    # the arms. On SA01 the legs are the only contributors, so the offset is real but smaller, and it is largest in
    # exactly the configurations where the legs are far from the nominal crouch.
    "sa01": {
        "xml": "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.xml",
        "urdf": "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf",
        "fixed_joints": [],
    },
}
# clang-format off
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/acom/AngularCenterOfMass.cpp:acom_robot_dispatch, //humanoid_learning/acom/BUILD.bazel:acom_train_data)
# clang-format on

# Number of sinusoidal layers the C++ evaluator is hard-coded to load. See the
# static_assert in AngularCenterOfMass::createFromStaticWeights, which requires
# exactly this many sine layers plus one linear readout.
# LINT.IfChange(siren_num_layers)
_CPP_SUPPORTED_NUM_LAYERS = 2
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/acom/AngularCenterOfMass.cpp:acom_layer_count)

# Frequency scaling of the SIREN sinusoidal activations.
_OMEGA_0 = 30.0

# Directory where per-robot AcomSirenWeights<Robot>.h headers are consumed by the C++ build.
_CPP_HEADER_INSTALL_DIR = (
    "humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/acom"
)

# The hidden width of the shipped headers, and therefore the default. Used only when the installed header cannot be
# read, so that the guard still has something to compare against.
_SHIPPED_HIDDEN_DIM = 64


def _header_path(robot: str) -> str:
    """Workspace-relative path of the installed weight header for `robot`."""
    return os.path.join(
        _CPP_HEADER_INSTALL_DIR, f"AcomSirenWeights{robot.capitalize()}.h"
    )


def _installed_hidden_dim(robot: str):
    """The hidden width of the weight header currently installed, or None if it cannot be determined.

    Read out of the generated `W0_rows` constant rather than tracked separately, so the guard compares against what is
    really on disk and keeps working after a deliberate architecture change.
    """
    ws_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", "")
    if not ws_dir:
        return _SHIPPED_HIDDEN_DIM
    path = os.path.join(ws_dir, _header_path(robot))
    try:
        with open(path, "r", encoding="utf-8") as header:
            for line in header:
                match = re.search(r"W0_rows\s*=\s*(\d+)", line)
                if match:
                    return int(match.group(1))
    except OSError:
        return None
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Train Angular Center of Mass (aCOM) in JAX"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="g1",
        choices=list(_ROBOT_CONFIGS.keys()),
        help="Robot model",
    )
    parser.add_argument("--xml", type=str, default=None, help="Path to MuJoCo XML file")
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to URDF file for joint order alignment with Pinocchio. "
        "If not given, inferred from --robot.",
    )
    # These three defaults are the recipe that produced the SHIPPED weight headers, and they are defaults rather than
    # values the README asks you to type because getting them wrong is silent: the C++ loader is width-agnostic, so a
    # 16-wide header compiles, loads and runs, and simply approximates the connection about twice as badly. The only
    # thing that catches it is
    # //humanoid_nmpc/humanoid_common_mpc:testAcomAngularVelocityConsistency, after the header is already installed.
    parser.add_argument(
        "--num_samples", type=int, default=20000, help="Number of dataset samples"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=_SHIPPED_HIDDEN_DIM,
        help="SIREN hidden dimension. The shipped headers are "
        f"{_SHIPPED_HIDDEN_DIM}-wide; --install_header refuses anything else unless --force_architecture is given.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=_CPP_SUPPORTED_NUM_LAYERS,
        help="Number of SIREN sinusoidal layers, excluding the linear readout. "
        f"The C++ evaluator only loads {_CPP_SUPPORTED_NUM_LAYERS}.",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument(
        "--output_dir", type=str, default="/tmp/acom_export", help="Output directory"
    )
    parser.add_argument(
        "--install_header",
        action="store_true",
        help="Install the exported C++ header directly into the workspace at "
        f"{_CPP_HEADER_INSTALL_DIR}/AcomSirenWeights<Robot>.h",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="TensorBoard log directory (defaults to <output_dir>/tb_logs)",
    )
    parser.add_argument(
        "--no_tb",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--force_architecture",
        action="store_true",
        help="Allow --install_header to overwrite a header whose architecture differs from the one being trained. "
        "Only pass this when the architecture change is the point.",
    )
    args = parser.parse_args()

    if args.install_header and args.num_layers != _CPP_SUPPORTED_NUM_LAYERS:
        parser.error(
            f"--install_header requires --num_layers {_CPP_SUPPORTED_NUM_LAYERS}: "
            "AngularCenterOfMass::createFromStaticWeights loads exactly "
            f"{_CPP_SUPPORTED_NUM_LAYERS} sinusoidal layers plus one linear "
            f"readout, so a header with {args.num_layers} would fail to compile."
        )

    # The layer count is caught above because a wrong one fails to COMPILE. The hidden width is not: every layer is
    # Eigen::Map'd at runtime from the dimensions the header declares, so a narrower network installs, builds and runs,
    # and is only ever noticed as degraded tracking. Compare against the header actually on disk rather than against a
    # constant, so this keeps working after a deliberate architecture change.
    if args.install_header and not args.force_architecture:
        installed_width = _installed_hidden_dim(args.robot)
        if installed_width is not None and installed_width != args.hidden_dim:
            parser.error(
                f"--install_header would replace the {installed_width}-wide "
                f"{_header_path(args.robot)} with a {args.hidden_dim}-wide network. The C++ loader is "
                "width-agnostic, so this would build and run while approximating the centroidal connection worse. "
                f"Pass --hidden_dim {installed_width} to match, or --force_architecture if the change is intended."
            )

    # Determine XML and URDF paths
    robot_cfg = _ROBOT_CONFIGS.get(args.robot, {})
    if args.xml is None:
        args.xml = robot_cfg.get("xml")
    if args.urdf is None:
        args.urdf = robot_cfg.get("urdf")
    fixed_joints = robot_cfg.get("fixed_joints", [])

    log_dir = None
    if not args.no_tb:
        log_dir = (
            args.log_dir
            if args.log_dir is not None
            else os.path.join(args.output_dir, "tb_logs", args.robot)
        )

    print(f"Generating aCOM dataset for {args.robot} from {args.xml}")
    if args.urdf:
        print(f"  Using URDF for joint ordering: {args.urdf}")
    if fixed_joints:
        print(f"  Holding {len(fixed_joints)} joints fixed: {fixed_joints}")
    generator = AcomDatasetGenerator(
        args.xml, urdf_path=args.urdf, fixed_joints=fixed_joints
    )
    dataset = generator.generate_dataset(num_samples=args.num_samples)
    print(
        f"  Generated {args.num_samples} configurations over "
        f"{generator.num_active_joints} active joints."
    )

    print(
        f"Training SIREN aCOM network "
        f"({args.num_layers} sinusoidal layers x {args.hidden_dim} neurons)"
    )
    model, params, history = train_acom(
        dataset=dataset,
        in_dim=generator.num_active_joints,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        omega_0=_OMEGA_0,
        num_epochs=args.epochs,
        verbose=True,
        log_dir=log_dir,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"acom_{args.robot}.json")

    # Per-robot class name and header filename (e.g., AcomSirenWeightsAtlas)
    robot_name_cap = args.robot.capitalize()
    class_name = f"AcomSirenWeights{robot_name_cap}"
    header_filename = f"{class_name}.h"
    cpp_path = os.path.join(args.output_dir, header_filename)

    export_to_json(params, json_path, omega_0=_OMEGA_0)
    export_to_cpp_header(
        params,
        cpp_path,
        class_name=class_name,
        omega_0=_OMEGA_0,
        joint_names=generator.active_joint_names,
    )

    # The RMSE of the epoch whose parameters were actually exported, which is the best one and not necessarily the
    # last. Reporting the final epoch's number here while exporting a different epoch's weights would misdescribe the
    # artefact by exactly the gap the training loop just printed.
    best_epoch = int(history["best_epoch"][0])
    exported_rmse = history["val_rmse"][best_epoch]
    print(
        f"Training complete. Exported validation RMSE: {exported_rmse:.5f} (epoch {best_epoch})"
    )
    print(f"  Exported JSON to: {json_path}")
    print(f"  Exported C++ header to: {cpp_path}")

    if args.install_header:
        ws_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY", "")
        if not ws_dir:
            raise RuntimeError(
                "--install_header requires BUILD_WORKSPACE_DIRECTORY to be set, "
                "which 'bazel run' does. Run this target with 'bazel run', or copy "
                f"{cpp_path} into {_CPP_HEADER_INSTALL_DIR} by hand."
            )
        install_path = os.path.join(ws_dir, _CPP_HEADER_INSTALL_DIR, header_filename)
        os.makedirs(os.path.dirname(install_path), exist_ok=True)
        shutil.copy2(cpp_path, install_path)
        print(f"  Installed C++ header to: {install_path}")


if __name__ == "__main__":
    main()
