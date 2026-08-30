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
import jax
from humanoid_learning.acom.dataset_generator import AcomDatasetGenerator
from humanoid_learning.acom.train_acom import train_acom
from humanoid_learning.acom.export_acom import export_to_json, export_to_cpp_header


def main():
    parser = argparse.ArgumentParser(
        description="Train Angular Center of Mass (aCOM) in JAX"
    )
    parser.add_argument(
        "--robot", type=str, default="g1", choices=["g1", "atlas"], help="Robot model"
    )
    parser.add_argument("--xml", type=str, default=None, help="Path to MuJoCo XML file")
    parser.add_argument(
        "--num_samples", type=int, default=5000, help="Number of dataset samples"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="SIREN hidden dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of SIREN layers"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument(
        "--output_dir", type=str, default="/tmp/acom_export", help="Output directory"
    )
    args = parser.parse_args()

    # Determine XML path
    if args.xml is None:
        if args.robot == "g1":
            args.xml = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.xml"
        elif args.robot == "atlas":
            args.xml = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.xml"

    print(f"🤖 Generating aCOM dataset for {args.robot.upper()} from {args.xml}...")
    generator = AcomDatasetGenerator(args.xml)
    dataset = generator.generate_dataset(num_samples=args.num_samples)
    print(
        f"  Dataset generated: {args.num_samples} configurations, {generator.num_joints} joints."
    )

    print(
        f"🧠 Training JAX SIREN aCOM network ({args.num_layers} layers x {args.hidden_dim} neurons)..."
    )
    model, params, history = train_acom(
        dataset=dataset,
        in_dim=generator.num_joints,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        verbose=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"acom_{args.robot}.json")
    cpp_path = os.path.join(
        args.output_dir, f"AngularCenterOfMassWeights_{args.robot}.h"
    )

    export_to_json(params, json_path)
    export_to_cpp_header(
        params, cpp_path, class_name=f"AcomSirenWeights_{args.robot.capitalize()}"
    )

    print(f"✅ Training complete!")
    print(f"  Exported JSON to: {json_path}")
    print(f"  Exported C++ header to: {cpp_path}")


if __name__ == "__main__":
    main()
