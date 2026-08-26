#!/usr/bin/env python3
"""CLI utility to retarget offline trajectory demonstration datasets between robot embodiments."""

import argparse
import os

from humanoid_learning.retargeting.trajectory_retargeter import (
    DatasetRetargetingConfig,
    TrajectoryDatasetRetargeter,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retarget offline demonstration dataset from source robot to target robot"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="g1",
        help="Source robot model name or YAML path (e.g. 'g1', 'atlas')",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="r1",
        help="Target robot model name or YAML path (e.g. 'r1', 'atlas')",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to source NPZ dataset file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for output retargeted NPZ dataset file",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="anatomical",
        choices=["anatomical", "optimization"],
        help="Retargeting algorithm ('anatomical' or 'optimization')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("🦾 Cross-Robot Offline Trajectory Retargeter")
    print(f"   Source Robot: {args.source}")
    print(f"   Target Robot: {args.target}")
    print(f"   Method:       {args.method}")
    print(f"   Input Path:   {args.input}")
    print(f"   Output Path:  {args.output}")
    print("=" * 70)

    config = DatasetRetargetingConfig(method=args.method)
    retargeter = TrajectoryDatasetRetargeter(
        source_robot=args.source,
        target_robot=args.target,
        config=config,
    )

    info = retargeter.retarget_dataset(args.input, args.output)
    print(f"\n✓ Retargeting complete: {info['num_samples']} samples converted.")
    print(f"  Target Observation Dimension: {info['obs_dim']}")
    print(f"  Target Action Dimension:      {info['act_dim']}")


if __name__ == "__main__":
    main()
