#!/usr/bin/env python3
"""CLI utility to generate RobotModelSpec definition files from MJCF XML or URDF models."""

import argparse
import os

from humanoid_learning.wbc.robot_model_loader import (
    load_robot_spec,
    save_robot_spec,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate robot definition YAML/JSON for WBC and RL from MJCF/URDF"
    )
    # LINT.IfChange(supported_robots)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path or name of input MJCF XML / URDF model (e.g. 'g1', 'r1', 'atlas', or path to .xml/.urdf)",
    )
    # LINT.ThenChange(//humanoid_learning/wbc/robot_model_loader.py:supported_robots)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path for output YAML definition (default: humanoid_learning/configs/robots/<robot_name>.yaml)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "mujoco", "pinocchio"],
        help="Model parsing backend engine ('mujoco', 'pinocchio', or 'auto')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("🤖 Robot Specification Generator for Whole-Body Control & RL")
    print(f"   Input Model:   {args.input}")
    print(f"   Parser Engine: {args.backend}")
    print("=" * 70)

    spec = load_robot_spec(args.input, backend=args.backend)

    output_path = args.output
    if not output_path:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(curr_dir, "..", ".."))
        output_path = os.path.join(
            repo_root,
            f"humanoid_learning/configs/robots/{spec.name}.yaml",
        )

    save_robot_spec(spec, output_path)

    print("\nRobot Properties Successfully Extracted:")
    print(f" • Robot Name:           {spec.name}")
    print(f" • Total Mass:           {spec.total_mass:.2f} kg")
    print(
        f" • Degrees of Freedom:   nq={spec.nq}, nv={spec.nv}, n_act={spec.n_act}"
    )
    print(f" • Nominal Height:       {spec.default_standing_height:.2f} m")
    print(f" • Contact Bodies:       {spec.contact_body_names}")
    print(f" • Actuated Joints:      {len(spec.joint_names)} joints")
    for i, (jname, q0, low, up, tau) in enumerate(
        zip(
            spec.joint_names[:6],
            spec.q_nominal[:6],
            spec.joint_limits_lower[:6],
            spec.joint_limits_upper[:6],
            spec.torque_limits[:6],
        )
    ):
        print(
            f"   [{i:02d}] {jname:<26} | q0={q0:>+5.2f} | limits=[{low:>+5.2f}, {up:>+5.2f}] | tau_max={tau:.0f} Nm"
        )
    if len(spec.joint_names) > 6:
        print(f"   ... ({len(spec.joint_names) - 6} more joints)")

    print(f"\n✓ Generated robot specification ready at '{output_path}'.")


if __name__ == "__main__":
    main()
