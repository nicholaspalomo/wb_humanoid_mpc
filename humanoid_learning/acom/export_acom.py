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
from typing import List, Tuple
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
):
    """Generates a standalone C++ header file containing SIREN weights and biases."""
    lines = [
        "/******************************************************************************",
        " * Auto-generated Angular Center of Mass (aCOM) SIREN Model Parameters",
        " * Generated from JAX training pipeline.",
        " ******************************************************************************/",
        "",
        "#pragma once",
        "",
        "#include <array>",
        "#include <vector>",
        "",
        "namespace ocs2::humanoid::acom {",
        "",
        f"struct {class_name} {{",
        f"  static constexpr double omega_0 = {omega_0};",
        f"  static constexpr size_t num_layers = {len(params)};",
        f"  static constexpr size_t input_dim = {params[0][0].shape[1]};",
        f"  static constexpr size_t output_dim = {params[-1][0].shape[0]};",
        "",
    ]

    for idx, (w, b) in enumerate(params):
        w_np = np.array(w, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        out_dim, in_dim = w_np.shape

        lines.append(f"  // Layer {idx}: ({out_dim} x {in_dim})")
        lines.append(f"  static constexpr size_t W{idx}_rows = {out_dim};")
        lines.append(f"  static constexpr size_t W{idx}_cols = {in_dim};")

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
