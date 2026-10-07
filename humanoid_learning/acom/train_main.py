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

"""CLI entrypoint to train and export aCOM models for humanoid robots.

The functions above main() are the pipeline itself, and the training notebook
(notebooks/train_acom_siren.ipynb) drives the same functions, so the two cannot
drift into producing different headers.
"""

import argparse
import os
import re
import shutil

from humanoid_mpc_config import task_file_pb2
import numpy as np

from humanoid_learning.acom import dataset_generator
from humanoid_learning.acom import export_acom
from humanoid_learning.acom import models
from humanoid_learning.acom import train_acom
import nproto_textproto

# Model files per robot, keyed by --robot. What has to agree with the MPC - the
# robot's model_settings.robot_name and the joints it holds fixed - is read out of
# the robot's centroidal task.textproto ("task") rather than repeated here, so there
# is no second copy to drift. Keep in sync with Bazel data dependencies.
# LINT.IfChange(robot_paths)
_ROBOT_CONFIGS = {
    "g1": {
        "xml": "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml",
        "urdf": "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
        "task": "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
    },
    "atlas": {
        "xml": "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml",
        "urdf": "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
        "task": "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto",
    },
    # The EngineAI SA01 is LEGS ONLY: twelve joints, six per leg, with no arms, no waist and no torso link - its
    # only body above the legs is base_link, which is the floating base itself. Its task file therefore fixes no
    # joints, unlike the wrists the other two hold at zero, and the aCOM offset is a function of the leg joints alone.
    #
    # That also sets expectations for what the coordinate buys on this robot. The aCOM exists because limb motion
    # decouples whole-body orientation from base orientation, and the limbs doing most of that on Atlas and G1 are
    # the arms. On SA01 the legs are the only contributors, so the offset is real but smaller, and it is largest in
    # exactly the configurations where the legs are far from the nominal crouch.
    "sa01": {
        "xml": "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.xml",
        "urdf": "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf",
        "task": "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto",
    },
}
# clang-format off
# LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/acom/AngularCenterOfMass.cpp:acom_robot_dispatch, //humanoid_learning/acom/BUILD.bazel:acom_train_data, //humanoid_learning/acom/tests/BUILD.bazel:acom_test_data, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto:acom_robot_name, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto:acom_robot_name, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto:acom_robot_name)
# clang-format on

# Number of sinusoidal layers the C++ evaluator is hard-coded to load. See the
# static_assert in AngularCenterOfMass.cpp's createFromStaticWeights, which
# requires exactly this many sine layers plus one linear readout.
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

# Default dataset size and epoch count. These are a standard run, NOT a record of how every shipped header was made:
# the Atlas header came from 80 000 samples over 300 epochs, and the recipe of the G1 and SA01 headers was not
# recorded. Every header exported since records its own recipe in its banner, see export_network.
_DEFAULT_NUM_SAMPLES = 20000
_DEFAULT_EPOCHS = 150

# Seeds of the configuration sampler and of the training run (initialization, split and shuffling).
_DATASET_SEED = 42
_TRAINING_SEED = 42


def robot_names() -> list[str]:
    """The --robot keys a network can be trained for."""
    return list(_ROBOT_CONFIGS)


def load_mpc_model_settings(robot: str) -> tuple[str, list[str]]:
    """Reads what the network has to agree with out of the robot's MPC task file.

    Args:
        robot: A key of _ROBOT_CONFIGS.

    Returns:
        (model_settings.robot_name, model_settings.fixed_joint_names) of the robot's
        centroidal task.textproto. The robot name is the one the C++ runtime looks the
        network up by; the fixed joints are the ones it holds at zero and leaves
        out of the joint vector it feeds the network.

    Raises:
        KeyError: If `robot` is unknown or the task file names no robot.
        OSError: If the task file cannot be read.
        nproto_textproto.TextprotoError: If it does not parse strictly into its schema
            (humanoid_mpc_config.TaskFile); the message names the line and column.
    """
    task_path = dataset_generator.resolve_xml_path(_ROBOT_CONFIGS[robot]["task"])
    model_settings = nproto_textproto.load_textproto(
        task_path, task_file_pb2.TaskFile
    ).model_settings
    if not model_settings.robot_name:
        raise KeyError(f"{task_path} names no model_settings.robot_name")
    return str(model_settings.robot_name), [
        str(name) for name in model_settings.fixed_joint_names
    ]


def make_generator(
    robot: str, xml_path: str | None = None, urdf_path: str | None = None
) -> dataset_generator.AcomDatasetGenerator:
    """The dataset generator for `robot`, set up exactly as the MPC sees the robot.

    The joint axis is put in Pinocchio's order via the URDF, and the joints the
    task file fixes are held at zero and dropped, so the network's inputs are the
    MPC model's joints one for one.

    Args:
        robot: A key of _ROBOT_CONFIGS.
        xml_path: MuJoCo model to sample, instead of the robot's own.
        urdf_path: URDF to take the joint order from, instead of the robot's own.

    Returns:
        The generator, its fixed joints read from the robot's task file.
    """
    config = _ROBOT_CONFIGS[robot]
    _, fixed_joints = load_mpc_model_settings(robot)
    return dataset_generator.AcomDatasetGenerator(
        xml_path or config["xml"],
        urdf_path=urdf_path or config["urdf"],
        fixed_joints=fixed_joints,
    )


def header_class_name(robot: str) -> str:
    """The struct the weight header for `robot` declares, e.g. AcomSirenWeightsAtlas."""
    return f"AcomSirenWeights{robot.capitalize()}"


def _header_path(robot: str) -> str:
    """Workspace-relative path of the installed weight header for `robot`."""
    return os.path.join(_CPP_HEADER_INSTALL_DIR, f"{header_class_name(robot)}.h")


def _installed_hidden_dim(robot: str, workspace_dir: str | None = None) -> int | None:
    """The hidden width of the weight header currently installed, or None if it cannot be determined.

    Read out of the generated `W0_rows` constant rather than tracked separately, so the guard compares against what is
    really on disk and keeps working after a deliberate architecture change.

    Args:
        robot: A key of _ROBOT_CONFIGS.
        workspace_dir: The workspace the header is installed in; BUILD_WORKSPACE_DIRECTORY when not given.

    Returns:
        The header's `W0_rows`; _SHIPPED_HIDDEN_DIM when no workspace is known, None when the header cannot be read.
    """
    ws_dir = workspace_dir or os.environ.get("BUILD_WORKSPACE_DIRECTORY", "")
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


def generate_dataset(
    generator: dataset_generator.AcomDatasetGenerator, num_samples: int
) -> dict[str, np.ndarray]:
    """Samples `num_samples` configurations with the pipeline's fixed sampler seed."""
    return generator.generate_dataset(num_samples=num_samples, seed=_DATASET_SEED)


def train_network(
    dataset: dict[str, np.ndarray],
    generator: dataset_generator.AcomDatasetGenerator,
    hidden_dim: int = _SHIPPED_HIDDEN_DIM,
    epochs: int = _DEFAULT_EPOCHS,
    log_dir: str | None = None,
    verbose: bool = True,
    num_layers: int = _CPP_SUPPORTED_NUM_LAYERS,
) -> tuple[models.SirenACOM, models.SirenParams, dict[str, list[float]]]:
    """Trains a SIREN on `dataset` at _OMEGA_0.

    Args:
        dataset: generate_dataset's output.
        generator: The generator the dataset came from; it gives the input width.
        hidden_dim: Width of every sinusoidal layer.
        epochs: Number of training epochs.
        log_dir: TensorBoard log directory, or None for no logging.
        verbose: Whether to print progress.
        num_layers: Sine layers only; leave it at its default, the only count the
            C++ evaluator loads, unless the network is for analysis alone.

    Returns:
        train_acom's (model, params, history).
    """
    return train_acom.train_acom(
        dataset=dataset,
        in_dim=generator.num_active_joints,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        omega_0=_OMEGA_0,
        num_epochs=epochs,
        seed=_TRAINING_SEED,
        verbose=verbose,
        log_dir=log_dir,
    )


def export_network(
    params: models.SirenParams,
    generator: dataset_generator.AcomDatasetGenerator,
    robot: str,
    output_dir: str,
    recipe: dict[str, object],
) -> tuple[str, str]:
    """Writes the trained network as JSON and as the C++ weight header for `robot`.

    The header is named and laid out as AngularCenterOfMass.cpp includes it,
    records the joint names the network was trained on (checked against the MPC
    model at start-up), and records `recipe` - how it was trained - in its
    banner.

    Args:
        params: The trained network's parameters.
        generator: The generator the network was trained on; it names the joints.
        robot: A key of _ROBOT_CONFIGS.
        output_dir: Directory the two files are written to.
        recipe: How the network was trained, recorded in the header's banner.

    Returns:
        (json_path, cpp_path).
    """
    robot_name, fixed_joints = load_mpc_model_settings(robot)
    provenance = {
        "train_main --robot": robot,
        "model_settings.robot_name": robot_name,
        "model_settings.fixed_joint_names": fixed_joints,
        "sampling": "uniform over the full joint-limit box",
        "sine_layers": len(params) - 1,
        "omega_0": _OMEGA_0,
        "dataset_seed": _DATASET_SEED,
        "training_seed": _TRAINING_SEED,
    }
    provenance.update(recipe)

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"acom_{robot}.json")
    cpp_path = os.path.join(output_dir, f"{header_class_name(robot)}.h")
    export_acom.export_to_json(params, json_path, omega_0=_OMEGA_0)
    export_acom.export_to_cpp_header(
        params,
        cpp_path,
        class_name=header_class_name(robot),
        omega_0=_OMEGA_0,
        joint_names=generator.active_joint_names,
        provenance=provenance,
    )
    return json_path, cpp_path


def install_header(
    cpp_path: str,
    robot: str,
    hidden_dim: int,
    workspace_dir: str,
    force_architecture: bool = False,
) -> str:
    """Copies an exported header into the C++ source tree, where the next build compiles it in.

    Args:
        cpp_path: The exported header.
        robot: A key of _ROBOT_CONFIGS.
        hidden_dim: The exported network's hidden width.
        workspace_dir: The workspace root the header is installed below.
        force_architecture: Whether to replace a header of a different hidden width.

    Raises:
        ValueError: If the header would replace an installed one of a different
            hidden width and `force_architecture` is not set. The C++ loader is
            width-agnostic, so such a header would build and run while
            approximating the centroidal connection worse.

    Returns:
        The path the header was installed to.
    """
    if not force_architecture:
        installed_width = _installed_hidden_dim(robot, workspace_dir)
        if installed_width is not None and installed_width != hidden_dim:
            raise ValueError(
                f"Installing would replace the {installed_width}-wide "
                f"{_header_path(robot)} with a {hidden_dim}-wide network. The C++ loader is "
                "width-agnostic, so this would build and run while approximating the centroidal connection worse. "
                f"Pass --hidden_dim {installed_width} to match, or --force_architecture if the change is intended."
            )
    install_path = os.path.join(workspace_dir, _header_path(robot))
    os.makedirs(os.path.dirname(install_path), exist_ok=True)
    shutil.copy2(cpp_path, install_path)
    return install_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Angular Center of Mass (aCOM) in JAX"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="g1",
        choices=robot_names(),
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
    # A standard run, not a record of every shipped header: see _DEFAULT_NUM_SAMPLES. The architecture defaults below
    # ARE what every shipped header uses, and they are defaults rather than values the README asks you to type because
    # getting them wrong is silent: the C++ loader is width-agnostic, so a 16-wide header compiles, loads and runs,
    # and simply approximates the connection about twice as badly. The only thing that catches it is
    # //humanoid_nmpc/humanoid_common_mpc:testAcomAngularVelocityConsistency, after the header is already installed.
    parser.add_argument(
        "--num_samples",
        type=int,
        default=_DEFAULT_NUM_SAMPLES,
        help="Number of dataset samples",
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
    parser.add_argument(
        "--epochs", type=int, default=_DEFAULT_EPOCHS, help="Training epochs"
    )
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
            "AngularCenterOfMass.cpp's createFromStaticWeights loads exactly "
            f"{_CPP_SUPPORTED_NUM_LAYERS} sinusoidal layers plus one linear "
            f"readout, so a header with {args.num_layers} would fail to compile."
        )

    # The layer count is caught above because a wrong one fails to COMPILE. The hidden width is not: every layer is
    # Eigen::Map'd at runtime from the dimensions the header declares, so a narrower network installs, builds and runs,
    # and is only ever noticed as degraded tracking. Checked here, before training, as well as at install time, so a
    # long run is not wasted on a header that will be refused.
    if args.install_header and not args.force_architecture:
        installed_width = _installed_hidden_dim(args.robot)
        if installed_width is not None and installed_width != args.hidden_dim:
            parser.error(
                f"--install_header would replace the {installed_width}-wide "
                f"{_header_path(args.robot)} with a {args.hidden_dim}-wide network. The C++ loader is "
                "width-agnostic, so this would build and run while approximating the centroidal connection worse. "
                f"Pass --hidden_dim {installed_width} to match, or --force_architecture if the change is intended."
            )

    log_dir = None
    if not args.no_tb:
        log_dir = (
            args.log_dir
            if args.log_dir is not None
            else os.path.join(args.output_dir, "tb_logs", args.robot)
        )

    robot_name, fixed_joints = load_mpc_model_settings(args.robot)
    print(
        f"Generating aCOM dataset for {args.robot} (model_settings.robot_name: {robot_name})"
    )
    if fixed_joints:
        print(
            f"  Holding {len(fixed_joints)} joints fixed, as the task file does: {fixed_joints}"
        )
    generator = make_generator(args.robot, xml_path=args.xml, urdf_path=args.urdf)
    dataset = generate_dataset(generator, args.num_samples)
    print(
        f"  Generated {args.num_samples} configurations over "
        f"{generator.num_active_joints} active joints."
    )

    print(
        f"Training SIREN aCOM network "
        f"({args.num_layers} sinusoidal layers x {args.hidden_dim} neurons)"
    )
    _, params, history = train_network(
        dataset,
        generator,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        log_dir=log_dir,
        num_layers=args.num_layers,
    )

    # The RMSE of the epoch whose parameters were actually exported, which is the best one and not necessarily the
    # last. Reporting the final epoch's number here while exporting a different epoch's weights would misdescribe the
    # artifact by exactly the gap the training loop just printed.
    best_epoch = int(history["best_epoch"][0])
    exported_rmse = history["val_rmse"][best_epoch]
    json_path, cpp_path = export_network(
        params,
        generator,
        args.robot,
        args.output_dir,
        recipe={
            "num_samples": args.num_samples,
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "exported_epoch": best_epoch,
            "exported_val_rmse": f"{exported_rmse:.5f}",
        },
    )
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
        install_path = install_header(
            cpp_path,
            args.robot,
            args.hidden_dim,
            ws_dir,
            force_architecture=args.force_architecture,
        )
        print(f"  Installed C++ header to: {install_path}")


if __name__ == "__main__":
    main()
