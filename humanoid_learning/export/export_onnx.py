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

"""Export a trained policy checkpoint to ONNX format for C++ runtime deployment.

The graph is the whole deterministic policy actor_critic.policy_action(): the observation normalizer, the two ELU torso
layers, the actor's mean and the tanh that squashes it into (-1, 1), so that the runtime feeds it raw observations.
"""

import argparse
from collections.abc import Sequence
import os
from typing import Any

import numpy as np
import onnx
import onnx.numpy_helper

from humanoid_learning.training import actor_critic

# The ActorCritic layers the exported policy consists of: the shared torso and the actor's mean.
_POLICY_LAYERS = ("Dense_0", "Dense_1", "Dense_2")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Returns the command line of the export; `argv` None reads sys.argv."""
    parser = argparse.ArgumentParser(description="Export JAX Actor Policy to ONNX")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="The checkpoint bc_warmstart.py wrote (actor_critic.save_checkpoint())",
    )
    parser.add_argument(
        "--output_path", type=str, default="policy.onnx", help="Output ONNX file path"
    )
    return parser.parse_args(argv)


def actor_weights(params: Any) -> dict[str, np.ndarray]:
    """Returns the weights of the deterministic policy, keyed as export_policy_to_onnx reads them.

    Args:
        params: ActorCritic parameters, as `ActorCritic.init` returns them.

    Returns:
        `<layer>/kernel` and `<layer>/bias` of every layer in _POLICY_LAYERS, float32.

    Raises:
        KeyError: When `params` lacks a layer.
    """
    weights: dict[str, np.ndarray] = {}
    for layer in _POLICY_LAYERS:
        dense = params["params"][layer]
        weights[f"{layer}/kernel"] = np.asarray(dense["kernel"], dtype=np.float32)
        weights[f"{layer}/bias"] = np.asarray(dense["bias"], dtype=np.float32)
    return weights


def _checked_layers(
    weights: dict[str, np.ndarray]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns the (kernel, bias) of every layer of `weights`, whose shapes are checked to chain."""
    missing: list[str] = []
    for layer in _POLICY_LAYERS:
        missing.extend(
            f"{layer}/{leaf}"
            for leaf in ("kernel", "bias")
            if f"{layer}/{leaf}" not in weights
        )
    if missing:
        raise KeyError(f"the policy weights lack {', '.join(missing)}")
    layers = [
        (weights[f"{layer}/kernel"], weights[f"{layer}/bias"])
        for layer in _POLICY_LAYERS
    ]
    for index, (kernel, bias) in enumerate(layers):
        if kernel.ndim != 2 or bias.shape != (kernel.shape[1],):
            raise ValueError(
                f"{_POLICY_LAYERS[index]}: kernel {kernel.shape} and bias {bias.shape} are no dense layer"
            )
        if index > 0 and kernel.shape[0] != layers[index - 1][0].shape[1]:
            raise ValueError(
                f"{_POLICY_LAYERS[index]} takes {kernel.shape[0]} inputs, its predecessor gives {layers[index - 1][0].shape[1]}"
            )
    return layers


def export_policy_to_onnx(
    weights: dict[str, np.ndarray],
    normalizer: actor_critic.ObservationNormalizer,
    output_path: str,
) -> None:
    """Writes the deterministic policy tanh(MLP((observation - mean) / std)) as an ONNX graph.

    Args:
        weights: The policy's layers, as actor_weights() returns them.
        normalizer: The normalizer the weights were trained through.
        output_path: The ONNX file to write; its directory is created.

    Raises:
        KeyError: When `weights` lacks a layer.
        ValueError: When the layers' shapes do not chain or the normalizer is not of the observation's size.
    """
    layers = _checked_layers(weights)
    obs_dim = layers[0][0].shape[0]
    act_dim = layers[-1][0].shape[1]
    if normalizer.mean.shape != (obs_dim,) or normalizer.std.shape != (obs_dim,):
        raise ValueError(
            f"the normalizer ({normalizer.mean.shape}, {normalizer.std.shape}) is not of the observation's size {obs_dim}"
        )

    input_tensor = onnx.helper.make_tensor_value_info(
        "observation", onnx.TensorProto.FLOAT, [1, obs_dim]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
        "action", onnx.TensorProto.FLOAT, [1, act_dim]
    )
    initializers = [
        onnx.numpy_helper.from_array(
            np.asarray(normalizer.mean, dtype=np.float32), "observation_mean"
        ),
        onnx.numpy_helper.from_array(
            np.asarray(normalizer.std, dtype=np.float32), "observation_std"
        ),
    ]
    nodes = [
        onnx.helper.make_node("Sub", ["observation", "observation_mean"], ["centered"]),
        onnx.helper.make_node("Div", ["centered", "observation_std"], ["normalized"]),
    ]
    layer_input = "normalized"
    for index, (kernel, bias) in enumerate(layers):
        initializers.append(onnx.numpy_helper.from_array(kernel, f"W{index}"))
        initializers.append(onnx.numpy_helper.from_array(bias, f"B{index}"))
        nodes.append(
            onnx.helper.make_node(
                "Gemm",
                [layer_input, f"W{index}", f"B{index}"],
                [f"h{index}"],
                alpha=1.0,
                beta=1.0,
            )
        )
        if index + 1 < len(layers):
            nodes.append(onnx.helper.make_node("Elu", [f"h{index}"], [f"h{index}_act"]))
            layer_input = f"h{index}_act"
    nodes.append(onnx.helper.make_node("Tanh", [f"h{len(layers) - 1}"], ["action"]))

    graph = onnx.helper.make_graph(
        nodes=nodes,
        name="HumanoidPolicy",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )

    model = onnx.helper.make_model(graph, producer_name="wb_humanoid_mpc_rl")
    onnx.checker.check_model(model)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    print(f"✅ ONNX model successfully saved to: {output_path}")


def _resolved(path: str) -> str:
    """Returns `path`, a relative one taken relative to the workspace under bazel run."""
    if os.path.isabs(path):
        return path
    return os.path.join(
        os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.path.abspath(".")), path
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Exports the checkpoint of the command line."""
    args = parse_args(argv)
    checkpoint_path = _resolved(args.checkpoint_path)
    output_path = _resolved(args.output_path)
    print("=" * 60)
    print("🚀 Exporting JAX Policy to ONNX for C++ Real-Time Bridge")
    print(f"   Checkpoint:  {checkpoint_path}")
    print(f"   Output File: {output_path}")
    print("=" * 60)

    params, normalizer = actor_critic.load_checkpoint(checkpoint_path)
    export_policy_to_onnx(actor_weights(params), normalizer, output_path)


if __name__ == "__main__":
    main()
