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

"""Export utilities for Angular Center of Mass (aCOM) models.

Exports JAX SIREN parameters to JSON and generates zero-dependency
C++ header files for real-time inference in MPC and whole-body controllers.
"""

import json
from typing import List, Optional, Tuple

import numpy as np


def export_to_json(
    params: List[Tuple[np.ndarray, np.ndarray]], filepath: str, omega_0: float = 30.0
):
    """Exports model parameters to a JSON file."""
    data = {
        "omega_0": float(omega_0),
        "num_layers": len(params),
        "layers": [],
    }
    for i, (w, b) in enumerate(params):
        data["layers"].append(
            {
                "layer_idx": i,
                "weight": np.array(w).tolist(),
                "bias": np.array(b).tolist(),
                "in_dim": int(w.shape[1]),
                "out_dim": int(w.shape[0]),
            }
        )
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def export_to_cpp_header(
    params: List[Tuple[np.ndarray, np.ndarray]],
    filepath: str,
    class_name: str = "AcomSirenWeights",
    omega_0: float = 30.0,
    joint_names: Optional[List[str]] = None,
):
    """Generates a standalone C++ header file containing SIREN weights and biases.

    Args:
        params: Weight/bias pairs, sinusoidal layers first and the linear readout
            last. `num_layers` in the generated header is `len(params)`, which is
            one more than `SirenACOM.num_layers`.
        filepath: Destination path for the generated header.
        class_name: Name of the generated struct.
        omega_0: Frequency scaling baked into the sinusoidal activations.
        joint_names: Joint names, in the order the network expects them. Recorded
            in the header so the joint ordering the network was trained on can be
            checked against the Pinocchio model at runtime.

    Raises:
        ValueError: If `joint_names` does not have one entry per network input.
    """
    input_dim = int(params[0][0].shape[1])
    if joint_names is not None and len(joint_names) != input_dim:
        raise ValueError(
            f"joint_names has {len(joint_names)} entries but the network takes "
            f"{input_dim} inputs."
        )

    lines = [
        "/******************************************************************************",
        " * Auto-generated Angular Center of Mass (aCOM) SIREN Model Parameters",
        " * Generated from JAX training pipeline. Do not edit by hand.",
        " ******************************************************************************/",
        "",
        "#pragma once",
        "",
        "#include <cstddef>",
        "",
        "namespace ocs2::humanoid::acom {",
        "",
        f"struct {class_name} {{",
        f"  static constexpr double omega_0 = {omega_0};",
        f"  static constexpr std::size_t num_layers = {len(params)};",
        f"  static constexpr std::size_t input_dim = {input_dim};",
        f"  static constexpr std::size_t output_dim = {params[-1][0].shape[0]};",
        "",
    ]

    if joint_names is not None:
        lines.append("  // Joint ordering the network was trained on. Must match the")
        lines.append("  // MPC's Pinocchio joint ordering exactly.")
        joined = ", ".join(f'"{name}"' for name in joint_names)
        lines.append(
            f"  static inline const char* const joint_names[{input_dim}] = {{{joined}}};"
        )
        lines.append("")

    for idx, (w, b) in enumerate(params):
        w_np = np.array(w, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        out_dim, in_dim = w_np.shape

        lines.append(f"  // Layer {idx}: ({out_dim} x {in_dim})")
        lines.append(f"  static constexpr std::size_t W{idx}_rows = {out_dim};")
        lines.append(f"  static constexpr std::size_t W{idx}_cols = {in_dim};")

        # Flattened row-major weight array
        w_str = ", ".join(f"{val:.10e}" for val in w_np.flatten())
        lines.append(
            f"  static inline const double W{idx}[{out_dim * in_dim}] = {{{w_str}}};"
        )

        # Bias array
        b_str = ", ".join(f"{val:.10e}" for val in b_np.flatten())
        lines.append(f"  static inline const double b{idx}[{out_dim}] = {{{b_str}}};")
        lines.append("")

    lines.extend(
        [
            "};",
            "",
            "}  // namespace ocs2::humanoid::acom",
            "",
        ]
    )

    with open(filepath, "w") as f:
        f.write("\n".join(lines))
