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

"""Exports recorded MPC rollout trajectories to HDF5 demonstration datasets for RL warmstarting.

The input is the CSV files of mpc_observation_logger.py (mpc_observation_*.csv): by default the observations are the
MPC state columns x0, x1, ... and the actions the MPC input columns u0, u1, .... Logs of the ROS-era logger, which named
the DRC Atlas' components (h_x, ..., q_j_*, then qd_j_*, F_*, M_*), are read too. --obs_cols and --act_cols pick other
columns. The dataset is HDF5 when h5py is installed, and an .npz file otherwise.
"""

import argparse
from collections.abc import Sequence
import csv
import glob
import os
import re

import numpy as np

# The columns of mpc_observation_logger.py.
_STATE_COLUMN = re.compile(r"^x\d+$")
_INPUT_COLUMN = re.compile(r"^u\d+$")
TIME_COLUMN = "time"
# The columns of the ROS-era logger, and the positions it fell back to.
_LEGACY_OBSERVATION_PREFIXES = ("h_", "L_", "p_", "euler_", "q_j_")
_LEGACY_ACTION_PREFIXES = ("qd_j_", "F_", "M_")
_LEGACY_OBSERVATION_SLICE = slice(0, 25)
_LEGACY_ACTION_SLICE = slice(25, 37)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parses the exporter's command line `argv` (sys.argv when None)."""
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
        help="Specific observation columns to extract (defaults to the MPC state columns x0, x1, ...)",
    )
    parser.add_argument(
        "--act_cols",
        type=str,
        nargs="*",
        default=None,
        help="Specific action columns to extract (defaults to the MPC input columns u0, u1, ...)",
    )
    return parser.parse_args(argv)


def default_columns(header: Sequence[str]) -> tuple[list[str], list[str]]:
    """The observation and action columns of a log whose header is `header`."""
    state = [column for column in header if _STATE_COLUMN.match(column)]
    inputs = [column for column in header if _INPUT_COLUMN.match(column)]
    if state and inputs:
        return state, inputs
    observations = [
        column for column in header if column.startswith(_LEGACY_OBSERVATION_PREFIXES)
    ]
    actions = [
        column for column in header if column.startswith(_LEGACY_ACTION_PREFIXES)
    ]
    return (
        observations or list(header[_LEGACY_OBSERVATION_SLICE]),
        actions or list(header[_LEGACY_ACTION_SLICE]),
    )


def read_csv(file_path: str) -> tuple[list[str], np.ndarray]:
    """The header and the rows of a log, as float64."""
    with open(file_path, "r", newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        rows = [[float(value) for value in row] for row in reader if row]
    values = np.array(rows, dtype=np.float64).reshape(len(rows), len(header))
    return header, values


def _columns(header: Sequence[str], names: Sequence[str], file_path: str) -> list[int]:
    missing = [name for name in names if name not in header]
    if missing:
        raise ValueError(f"{file_path} has no column {', '.join(missing)}")
    return [list(header).index(name) for name in names]


def export_csv_to_h5(
    csv_files: Sequence[str],
    output_path: str,
    obs_cols: Sequence[str] | None = None,
    act_cols: Sequence[str] | None = None,
) -> str | None:
    """Writes the observations and actions of `csv_files` to `output_path`; returns the path written, or None."""
    try:
        import h5py  # pylint: disable=import-outside-toplevel  # An optional dependency.
    except ImportError:
        h5py = None
        print(
            "⚠️ h5py is not installed in the current environment. Saving as npz instead."
        )
        output_path = output_path.replace(".h5", ".npz")

    all_obs = []
    all_acts = []
    all_times = []
    for file_path in csv_files:
        print(f"Reading: {file_path}")
        header, values = read_csv(file_path)
        if values.shape[0] == 0:
            continue
        default_obs, default_acts = default_columns(header)
        obs_indices = _columns(header, obs_cols or default_obs, file_path)
        act_indices = _columns(header, act_cols or default_acts, file_path)
        all_obs.append(values[:, obs_indices].astype(np.float32))
        all_acts.append(values[:, act_indices].astype(np.float32))
        if TIME_COLUMN in header:
            all_times.append(values[:, header.index(TIME_COLUMN)])

    if not all_obs:
        print("⚠️ No data found to export.")
        return None

    merged_obs = np.concatenate(all_obs, axis=0)
    merged_acts = np.concatenate(all_acts, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if h5py is not None and output_path.endswith(".h5"):
        with h5py.File(output_path, "w") as f:
            f.create_dataset("observations", data=merged_obs, compression="gzip")
            f.create_dataset("actions", data=merged_acts, compression="gzip")
            if len(all_times) == len(all_obs):
                f.create_dataset(
                    "times", data=np.concatenate(all_times, axis=0), compression="gzip"
                )
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
    return output_path


def find_csv_files(input_path: str) -> list[str]:
    """The logs at `input_path`: the file itself, or the mpc_observation_*.csv files of a directory."""
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "mpc_observation_*.csv")))
    if os.path.isfile(input_path):
        return [input_path]
    return []


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    csv_files = find_csv_files(args.input_path)

    if not csv_files:
        print(f"ℹ️ No CSV log files found at {args.input_path}.")
        print("Generating synthetic demonstration dataset for imitation pretraining...")
        synthetic_obs = np.random.randn(500, 25).astype(np.float32)
        synthetic_act = np.random.randn(500, 12).astype(np.float32)
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        try:
            import h5py  # pylint: disable=import-outside-toplevel  # An optional dependency.

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
