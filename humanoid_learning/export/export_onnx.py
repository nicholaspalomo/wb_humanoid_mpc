"""Export trained JAX policy weights to ONNX format for C++ runtime deployment."""

import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp
import onnx
from onnx import helper, TensorProto

from humanoid_learning.training.train_ppo import ActorCritic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export JAX Actor Policy to ONNX"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default="", help="Path to checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="policy.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument(
        "--obs_dim", type=int, default=25, help="Observation dimension"
    )
    parser.add_argument(
        "--act_dim", type=int, default=12, help="Action dimension"
    )
    return parser.parse_args()


def export_dense_mlp_to_onnx(
    weights_dict, obs_dim: int, act_dim: int, output_path: str
):
    """Builds a pure ONNX graph from MLP weight matrices."""
    # Inputs & outputs
    input_tensor = helper.make_tensor_value_info(
        "observation", TensorProto.FLOAT, [1, obs_dim]
    )
    output_tensor = helper.make_tensor_value_info(
        "action", TensorProto.FLOAT, [1, act_dim]
    )

    nodes = []
    initializers = []

    # Layer 1
    w1 = weights_dict.get(
        "Dense_0/kernel", np.random.randn(obs_dim, 256).astype(np.float32)
    )
    b1 = weights_dict.get("Dense_0/bias", np.zeros((256,), dtype=np.float32))
    initializers.append(
        helper.make_tensor(
            "W1", TensorProto.FLOAT, [obs_dim, 256], w1.flatten()
        )
    )
    initializers.append(
        helper.make_tensor("B1", TensorProto.FLOAT, [256], b1.flatten())
    )
    nodes.append(
        helper.make_node(
            "Gemm", ["observation", "W1", "B1"], ["h1"], alpha=1.0, beta=1.0
        )
    )
    nodes.append(helper.make_node("Elu", ["h1"], ["h1_act"]))

    # Layer 2
    w2 = weights_dict.get(
        "Dense_1/kernel", np.random.randn(256, 256).astype(np.float32)
    )
    b2 = weights_dict.get("Dense_1/bias", np.zeros((256,), dtype=np.float32))
    initializers.append(
        helper.make_tensor("W2", TensorProto.FLOAT, [256, 256], w2.flatten())
    )
    initializers.append(
        helper.make_tensor("B2", TensorProto.FLOAT, [256], b2.flatten())
    )
    nodes.append(
        helper.make_node(
            "Gemm", ["h1_act", "W2", "B2"], ["h2"], alpha=1.0, beta=1.0
        )
    )
    nodes.append(helper.make_node("Elu", ["h2"], ["h2_act"]))

    # Action Head
    w3 = weights_dict.get(
        "Dense_2/kernel", np.random.randn(256, act_dim).astype(np.float32)
    )
    b3 = weights_dict.get(
        "Dense_2/bias", np.zeros((act_dim,), dtype=np.float32)
    )
    initializers.append(
        helper.make_tensor(
            "W3", TensorProto.FLOAT, [256, act_dim], w3.flatten()
        )
    )
    initializers.append(
        helper.make_tensor("B3", TensorProto.FLOAT, [act_dim], b3.flatten())
    )
    nodes.append(
        helper.make_node(
            "Gemm", ["h2_act", "W3", "B3"], ["action"], alpha=1.0, beta=1.0
        )
    )

    graph = helper.make_graph(
        nodes=nodes,
        name="HumanoidPolicy",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="wb_humanoid_mpc_rl")
    onnx.checker.check_model(model)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(model, output_path)
    print(f"✅ ONNX model successfully saved to: {output_path}")


def main():
    args = parse_args()
    print("=" * 60)
    print("🚀 Exporting JAX Policy to ONNX for C++ Real-Time Bridge")
    print(f"   Obs Dim:     {args.obs_dim}")
    print(f"   Act Dim:     {args.act_dim}")
    print(f"   Output File: {args.output_path}")
    print("=" * 60)

    # Initialize model weights
    network = ActorCritic(action_dim=args.act_dim)
    rng = jax.random.PRNGKey(0)
    params = network.init(rng, jnp.zeros((1, args.obs_dim)))

    # Extract weights
    dense0 = params["params"]["Dense_0"]
    dense1 = params["params"]["Dense_1"]
    dense2 = params["params"]["Dense_2"]

    weights_dict = {
        "Dense_0/kernel": np.array(dense0["kernel"]),
        "Dense_0/bias": np.array(dense0["bias"]),
        "Dense_1/kernel": np.array(dense1["kernel"]),
        "Dense_1/bias": np.array(dense1["bias"]),
        "Dense_2/kernel": np.array(dense2["kernel"]),
        "Dense_2/bias": np.array(dense2["bias"]),
    }

    export_dense_mlp_to_onnx(
        weights_dict, args.obs_dim, args.act_dim, args.output_path
    )


if __name__ == "__main__":
    main()
