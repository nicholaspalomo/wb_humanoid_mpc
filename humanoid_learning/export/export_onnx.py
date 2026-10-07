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

"""Export trained JAX policy weights to ONNX format for C++ runtime deployment."""

import argparse
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import onnx

from humanoid_learning.training import actor_critic

# The ActorCritic layers the exported policy consists of: the shared torso and the actor's mean.
_POLICY_LAYERS = ("Dense_0", "Dense_1", "Dense_2")


def parse_args() -> argparse.Namespace:
    """The command line of the export."""
    parser = argparse.ArgumentParser(description="Export JAX Actor Policy to ONNX")
    parser.add_argument(
        "--checkpoint_path", type=str, default="", help="Path to checkpoint"
    )
    parser.add_argument(
        "--output_path", type=str, default="policy.onnx", help="Output ONNX file path"
    )
    parser.add_argument("--obs_dim", type=int, default=25, help="Observation dimension")
    parser.add_argument("--act_dim", type=int, default=12, help="Action dimension")
    return parser.parse_args()


def actor_weights(params: Any) -> dict[str, np.ndarray]:
    """The weights of the deterministic policy, keyed as export_dense_mlp_to_onnx reads them.

    Args:
        params: ActorCritic parameters, as `ActorCritic.init` returns them.

    Returns:
        `<layer>/kernel` and `<layer>/bias` of every layer in _POLICY_LAYERS.
    """
    weights: dict[str, np.ndarray] = {}
    for layer in _POLICY_LAYERS:
        dense = params["params"][layer]
        weights[f"{layer}/kernel"] = np.array(dense["kernel"])
        weights[f"{layer}/bias"] = np.array(dense["bias"])
    return weights


def export_dense_mlp_to_onnx(
    weights_dict: dict[str, np.ndarray], obs_dim: int, act_dim: int, output_path: str
) -> None:
    """Builds a pure ONNX graph from MLP weight matrices."""
    # Inputs & outputs
    input_tensor = onnx.helper.make_tensor_value_info(
        "observation", onnx.TensorProto.FLOAT, [1, obs_dim]
    )
    output_tensor = onnx.helper.make_tensor_value_info(
        "action", onnx.TensorProto.FLOAT, [1, act_dim]
    )

    nodes = []
    initializers = []

    # Layer 1
    w1 = weights_dict.get(
        "Dense_0/kernel", np.random.randn(obs_dim, 256).astype(np.float32)
    )
    b1 = weights_dict.get("Dense_0/bias", np.zeros((256,), dtype=np.float32))
    initializers.append(
        onnx.helper.make_tensor(
            "W1", onnx.TensorProto.FLOAT, [obs_dim, 256], w1.flatten()
        )
    )
    initializers.append(
        onnx.helper.make_tensor("B1", onnx.TensorProto.FLOAT, [256], b1.flatten())
    )
    nodes.append(
        onnx.helper.make_node(
            "Gemm", ["observation", "W1", "B1"], ["h1"], alpha=1.0, beta=1.0
        )
    )
    nodes.append(onnx.helper.make_node("Elu", ["h1"], ["h1_act"]))

    # Layer 2
    w2 = weights_dict.get(
        "Dense_1/kernel", np.random.randn(256, 256).astype(np.float32)
    )
    b2 = weights_dict.get("Dense_1/bias", np.zeros((256,), dtype=np.float32))
    initializers.append(
        onnx.helper.make_tensor("W2", onnx.TensorProto.FLOAT, [256, 256], w2.flatten())
    )
    initializers.append(
        onnx.helper.make_tensor("B2", onnx.TensorProto.FLOAT, [256], b2.flatten())
    )
    nodes.append(
        onnx.helper.make_node(
            "Gemm", ["h1_act", "W2", "B2"], ["h2"], alpha=1.0, beta=1.0
        )
    )
    nodes.append(onnx.helper.make_node("Elu", ["h2"], ["h2_act"]))

    # Action Head
    w3 = weights_dict.get(
        "Dense_2/kernel", np.random.randn(256, act_dim).astype(np.float32)
    )
    b3 = weights_dict.get("Dense_2/bias", np.zeros((act_dim,), dtype=np.float32))
    initializers.append(
        onnx.helper.make_tensor(
            "W3", onnx.TensorProto.FLOAT, [256, act_dim], w3.flatten()
        )
    )
    initializers.append(
        onnx.helper.make_tensor("B3", onnx.TensorProto.FLOAT, [act_dim], b3.flatten())
    )
    nodes.append(
        onnx.helper.make_node(
            "Gemm", ["h2_act", "W3", "B3"], ["action"], alpha=1.0, beta=1.0
        )
    )

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


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("🚀 Exporting JAX Policy to ONNX for C++ Real-Time Bridge")
    print(f"   Obs Dim:     {args.obs_dim}")
    print(f"   Act Dim:     {args.act_dim}")
    print(f"   Output File: {args.output_path}")
    print("=" * 60)

    # Initialize model weights
    network = actor_critic.ActorCritic(action_dim=args.act_dim)
    rng = jax.random.PRNGKey(0)
    params = network.init(rng, jnp.zeros((1, args.obs_dim)))

    export_dense_mlp_to_onnx(
        actor_weights(params), args.obs_dim, args.act_dim, args.output_path
    )


if __name__ == "__main__":
    main()
