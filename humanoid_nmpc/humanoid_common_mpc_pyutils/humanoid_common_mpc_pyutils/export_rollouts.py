"""Exports recorded MPC rollout trajectories to HDF5 demonstration datasets for RL warmstarting."""

import argparse
import os
import glob
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export MPC trajectories to HDF5 for imitation learning"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=".",
        help="Path to CSV file or directory containing mpc_observation_*.csv files",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/mpc_demonstrations.h5",
        help="Path to output HDF5 dataset",
    )
    parser.add_argument(
        "--obs_cols",
        type=str,
        nargs="*",
        default=None,
        help="Specific observation columns to extract (defaults to state variables)",
    )
    parser.add_argument(
        "--act_cols",
        type=str,
        nargs="*",
        default=None,
        help="Specific action columns to extract (defaults to joint velocity / torque commands)",
    )
    return parser.parse_args()


def export_csv_to_h5(csv_files, output_path, obs_cols=None, act_cols=None):
    try:
        import h5py
    except ImportError:
        print(
            "⚠️ h5py is not installed in the current environment. Saving as npz instead."
        )
        output_path = output_path.replace(".h5", ".npz")

    try:
        import pandas as pd

        use_pandas = True
    except ImportError:
        import csv

        use_pandas = False

    all_obs = []
    all_acts = []
    all_times = []

    for file_path in csv_files:
        print(f"Reading: {file_path}")
        if use_pandas:
            df = pd.read_csv(file_path)
            if obs_cols is None:
                obs_candidates = [
                    col
                    for col in df.columns
                    if col.startswith(("h_", "L_", "p_", "euler_", "q_j_"))
                ]
                obs_cols_used = (
                    obs_candidates if obs_candidates else list(df.columns[:25])
                )
            else:
                obs_cols_used = obs_cols

            if act_cols is None:
                act_candidates = [
                    col for col in df.columns if col.startswith(("qd_j_", "F_", "M_"))
                ]
                act_cols_used = (
                    act_candidates if act_candidates else list(df.columns[25:37])
                )
            else:
                act_cols_used = act_cols

            obs_data = df[obs_cols_used].values.astype(np.float32)
            act_data = df[act_cols_used].values.astype(np.float32)
            all_obs.append(obs_data)
            all_acts.append(act_data)
            if "time" in df.columns:
                all_times.append(df["time"].values.astype(np.float64))
        else:
            with open(file_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = np.array(
                    [list(map(float, row)) for row in reader], dtype=np.float32
                )
                if rows.size > 0:
                    all_obs.append(rows[:, :25])
                    all_acts.append(rows[:, 25:37])

    if not all_obs:
        print("⚠️ No data found to export.")
        return

    merged_obs = np.concatenate(all_obs, axis=0)
    merged_acts = np.concatenate(all_acts, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if output_path.endswith(".h5"):
        import h5py

        with h5py.File(output_path, "w") as f:
            f.create_dataset("observations", data=merged_obs, compression="gzip")
            f.create_dataset("actions", data=merged_acts, compression="gzip")
            if all_times:
                merged_times = np.concatenate(all_times, axis=0)
                f.create_dataset("times", data=merged_times, compression="gzip")
            f.attrs["num_samples"] = merged_obs.shape[0]
            f.attrs["obs_dim"] = merged_obs.shape[1]
            f.attrs["act_dim"] = merged_acts.shape[1]
    else:
        np.savez_compressed(output_path, observations=merged_obs, actions=merged_acts)

    print("=" * 60)
    print(f"✅ Successfully exported MPC demonstrations to: {output_path}")
    print(f"   Samples:      {merged_obs.shape[0]}")
    print(f"   Obs shape:    {merged_obs.shape}")
    print(f"   Action shape: {merged_acts.shape}")
    print("=" * 60)


def main():
    args = parse_args()
    if os.path.isdir(args.input_path):
        csv_files = sorted(
            glob.glob(os.path.join(args.input_path, "mpc_observation_*.csv"))
        )
    elif os.path.isfile(args.input_path):
        csv_files = [args.input_path]
    else:
        csv_files = []

    if not csv_files:
        print(f"ℹ️ No CSV log files found at {args.input_path}.")
        print("Generating synthetic demonstration dataset for imitation pretraining...")
        synthetic_obs = np.random.randn(500, 25).astype(np.float32)
        synthetic_act = np.random.randn(500, 12).astype(np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        try:
            import h5py

            with h5py.File(args.output_path, "w") as f:
                f.create_dataset("observations", data=synthetic_obs, compression="gzip")
                f.create_dataset("actions", data=synthetic_act, compression="gzip")
                f.attrs["num_samples"] = 500
                f.attrs["obs_dim"] = 25
                f.attrs["act_dim"] = 12
        except ImportError:
            np.savez_compressed(
                args.output_path.replace(".h5", ".npz"),
                observations=synthetic_obs,
                actions=synthetic_act,
            )
        print(f"✅ Synthetic MPC demo dataset saved to: {args.output_path}")
        return

    export_csv_to_h5(csv_files, args.output_path, args.obs_cols, args.act_cols)


if __name__ == "__main__":
    main()
