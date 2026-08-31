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

"""Training pipeline for Angular Center of Mass (aCOM) using JAX and Optax.

Optimizes the SIREN network parameters by minimizing the Frobenius norm error
between the analytical network Jacobian and the locked-inertia normalized CMM.
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from tensorboardX import SummaryWriter

from humanoid_learning.acom.models import SirenACOM


def create_train_step(
    model: SirenACOM, optimizer: optax.GradientTransformation, reg_weight: float = 1e-4
):
    """Creates a JIT-compiled training step function."""

    # Vectorized Jacobian and forward evaluation over batch
    v_jacobian = jax.vmap(
        lambda params, q: model.jacobian_qj(params, q), in_axes=(None, 0)
    )
    v_forward = jax.vmap(lambda params, q: model.forward(params, q), in_axes=(None, 0))

    def loss_fn(params, q_batch, target_A_bar):
        # Compute Jacobians: (B, 3, n_j)
        jacobians = v_jacobian(params, q_batch)

        # Frobenius loss on Jacobian error
        jacobian_diff = jacobians - target_A_bar
        frob_loss = jnp.mean(jnp.sum(jnp.square(jacobian_diff), axis=(-2, -1)))

        # Regularization on offset magnitude (keep Delta_theta centered near zero)
        offsets = v_forward(params, q_batch)
        reg_loss = reg_weight * jnp.mean(jnp.sum(jnp.square(offsets), axis=-1))

        total_loss = frob_loss + reg_loss
        return total_loss, {
            "frob_loss": frob_loss,
            "reg_loss": reg_loss,
            "jacobians": jacobians,
            "offsets": offsets,
        }

    @jax.jit
    def train_step(params, opt_state, q_batch, target_A_bar):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, q_batch, target_A_bar
        )
        # Compute global gradient L2 norm
        grad_norm_sq = sum(
            jnp.sum(jnp.square(w_g)) + jnp.sum(jnp.square(b_g)) for w_g, b_g in grads
        )
        grad_norm = jnp.sqrt(grad_norm_sq)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux, grads, grad_norm

    @jax.jit
    def eval_step(params, q_batch, target_A_bar):
        loss, aux = loss_fn(params, q_batch, target_A_bar)
        return loss, aux

    return train_step, eval_step


def create_jacobian_diagnostic_figure(
    target_A_bar: np.ndarray, pred_jacobian: np.ndarray
) -> plt.Figure:
    """Creates a 3-panel heatmap comparing target and predicted CMM connection Jacobians."""
    diff = np.abs(pred_jacobian - target_A_bar)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    vmax = max(
        float(np.max(np.abs(target_A_bar))), float(np.max(np.abs(pred_jacobian)))
    )
    vmin = -vmax

    im0 = axes[0].imshow(
        target_A_bar, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    axes[0].set_title(
        "Ground Truth Connection $\\bar{\\mathbf{A}}_{\\omega, j}(\\mathbf{q})$"
    )
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["Roll", "Pitch", "Yaw"])
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        pred_jacobian, aspect="auto", cmap="coolwarm", vmin=vmin, vmax=vmax
    )
    axes[1].set_title(
        "SIREN Learned Jacobian $\\mathbf{J}_{\\Delta\\theta}(\\mathbf{q}_j)$"
    )
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["Roll", "Pitch", "Yaw"])
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(diff, aspect="auto", cmap="viridis")
    axes[2].set_title(
        "Absolute Error $|\\mathbf{J}_{\\Delta\\theta} - \\bar{\\mathbf{A}}_{\\omega, j}|$"
    )
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["Roll", "Pitch", "Yaw"])
    axes[2].set_xlabel("Joint Index")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def train_acom(
    dataset: Dict[str, np.ndarray],
    in_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    omega_0: float = 30.0,
    learning_rate: float = 1e-3,
    num_epochs: int = 50,
    batch_size: int = 256,
    reg_weight: float = 1e-4,
    seed: int = 42,
    verbose: bool = True,
    log_dir: Optional[str] = None,
    log_histograms: bool = True,
    histogram_freq: int = 10,
    log_figures: bool = True,
) -> Tuple[SirenACOM, List[Tuple[jnp.ndarray, jnp.ndarray]], Dict[str, List[float]]]:
    """Trains a SIREN aCOM model with TensorBoard logging."""
    rng = jax.random.PRNGKey(seed)

    # Initialize TensorBoard SummaryWriter if log_dir provided
    writer: Optional[SummaryWriter] = None
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        if verbose:
            print(f"📊 Logging TensorBoard metrics to: {log_dir}")

    # Split dataset into train (85%) and val (15%)
    num_samples = dataset["q_joints"].shape[0]
    num_train = int(0.85 * num_samples)

    q_train = jnp.array(dataset["q_joints"][:num_train])
    A_train = jnp.array(dataset["A_bar_omega"][:num_train])

    q_val = jnp.array(dataset["q_joints"][num_train:])
    A_val = jnp.array(dataset["A_bar_omega"][num_train:])

    # Initialize model
    model = SirenACOM(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        out_dim=3,
        omega_0=omega_0,
    )
    rng, k_init = jax.random.split(rng)
    params = model.init_params(k_init)

    # Optimizer with cosine decay schedule
    steps_per_epoch = max(1, num_train // batch_size)
    total_steps = num_epochs * steps_per_epoch
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=total_steps, alpha=0.01
    )
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    train_step_fn, eval_step_fn = create_train_step(
        model, optimizer, reg_weight=reg_weight
    )

    history = {
        "train_loss": [],
        "train_frob": [],
        "train_reg": [],
        "grad_norm": [],
        "val_loss": [],
        "val_frob": [],
        "val_reg": [],
        "val_rmse": [],
    }

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Shuffle training set
        rng, k_perm = jax.random.split(rng)
        perm = jax.random.permutation(k_perm, num_train)
        q_train_shuffled = q_train[perm]
        A_train_shuffled = A_train[perm]

        train_losses = []
        train_frobs = []
        train_regs = []
        train_grad_norms = []
        last_grads = None

        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, num_train)
            q_batch = q_train_shuffled[start_idx:end_idx]
            A_batch = A_train_shuffled[start_idx:end_idx]

            params, opt_state, loss_val, aux, grads, grad_norm = train_step_fn(
                params, opt_state, q_batch, A_batch
            )
            train_losses.append(float(loss_val))
            train_frobs.append(float(aux["frob_loss"]))
            train_regs.append(float(aux["reg_loss"]))
            train_grad_norms.append(float(grad_norm))
            last_grads = grads

        # Validation evaluation
        val_loss, val_aux = eval_step_fn(params, q_val, A_val)

        avg_train_loss = float(np.mean(train_losses))
        avg_train_frob = float(np.mean(train_frobs))
        avg_train_reg = float(np.mean(train_regs))
        avg_grad_norm = float(np.mean(train_grad_norms))

        avg_val_loss = float(val_loss)
        avg_val_frob = float(val_aux["frob_loss"])
        avg_val_reg = float(val_aux["reg_loss"])

        # Calculate RMSE per Jacobian entry (in rad/s per rad/s)
        val_rmse = float(np.sqrt(avg_val_frob / (3.0 * in_dim)))

        # Per-axis decomposition on validation set
        val_jacobians = np.array(val_aux["jacobians"])
        val_targets = np.array(A_val)
        axis_diff = val_jacobians - val_targets
        roll_err = float(np.mean(np.sum(np.square(axis_diff[:, 0, :]), axis=-1)))
        pitch_err = float(np.mean(np.sum(np.square(axis_diff[:, 1, :]), axis=-1)))
        yaw_err = float(np.mean(np.sum(np.square(axis_diff[:, 2, :]), axis=-1)))

        # aCOM offset statistics
        val_offsets = np.array(val_aux["offsets"])
        offset_norms_deg = np.rad2deg(np.linalg.norm(val_offsets, axis=-1))
        mean_offset_deg = float(np.mean(offset_norms_deg))
        max_offset_deg = float(np.max(offset_norms_deg))

        # Learning rate and execution time
        current_step = epoch * steps_per_epoch
        current_lr = float(lr_schedule(current_step))
        epoch_time_ms = (time.time() - epoch_start) * 1000.0
        throughput = float(num_train / (epoch_time_ms / 1000.0))

        history["train_loss"].append(avg_train_loss)
        history["train_frob"].append(avg_train_frob)
        history["train_reg"].append(avg_train_reg)
        history["grad_norm"].append(avg_grad_norm)
        history["val_loss"].append(avg_val_loss)
        history["val_frob"].append(avg_val_frob)
        history["val_reg"].append(avg_val_reg)
        history["val_rmse"].append(val_rmse)

        # Log to TensorBoard
        if writer is not None:
            # Loss metrics
            writer.add_scalar("loss/train_total", avg_train_loss, epoch)
            writer.add_scalar("loss/train_frobenius", avg_train_frob, epoch)
            writer.add_scalar("loss/train_regularization", avg_train_reg, epoch)
            writer.add_scalar("loss/val_total", avg_val_loss, epoch)
            writer.add_scalar("loss/val_frobenius", avg_val_frob, epoch)
            writer.add_scalar("loss/val_regularization", avg_val_reg, epoch)
            writer.add_scalar("loss/val_rmse_rad_per_rad", val_rmse, epoch)

            # Gradient metrics
            writer.add_scalar("gradients/global_l2_norm", avg_grad_norm, epoch)
            if last_grads is not None:
                for layer_idx, (w_g, b_g) in enumerate(last_grads):
                    w_norm = float(np.linalg.norm(np.array(w_g)))
                    b_norm = float(np.linalg.norm(np.array(b_g)))
                    writer.add_scalar(
                        f"gradients/layer_{layer_idx}_weight_norm", w_norm, epoch
                    )
                    writer.add_scalar(
                        f"gradients/layer_{layer_idx}_bias_norm", b_norm, epoch
                    )

            # Per-axis error metrics
            writer.add_scalar("error_axes/roll_x_frob_loss", roll_err, epoch)
            writer.add_scalar("error_axes/pitch_y_frob_loss", pitch_err, epoch)
            writer.add_scalar("error_axes/yaw_z_frob_loss", yaw_err, epoch)

            # Optimization and performance
            writer.add_scalar("optim/learning_rate", current_lr, epoch)
            writer.add_scalar("stats/mean_acom_offset_deg", mean_offset_deg, epoch)
            writer.add_scalar("stats/max_acom_offset_deg", max_offset_deg, epoch)
            writer.add_scalar("perf/epoch_time_ms", epoch_time_ms, epoch)
            writer.add_scalar("perf/throughput_samples_per_sec", throughput, epoch)

            # Weight, bias, and gradient histograms
            if log_histograms and (
                epoch % histogram_freq == 0 or epoch == num_epochs - 1
            ):
                for layer_idx, (w, b) in enumerate(params):
                    writer.add_histogram(
                        f"weights/layer_{layer_idx}_weight", np.array(w), epoch
                    )
                    writer.add_histogram(
                        f"weights/layer_{layer_idx}_bias", np.array(b), epoch
                    )

                if last_grads is not None:
                    for layer_idx, (w_g, b_g) in enumerate(last_grads):
                        writer.add_histogram(
                            f"gradients/layer_{layer_idx}_weight",
                            np.array(w_g),
                            epoch,
                        )
                        writer.add_histogram(
                            f"gradients/layer_{layer_idx}_bias",
                            np.array(b_g),
                            epoch,
                        )

                writer.add_histogram(
                    "activations/val_offset_roll_deg",
                    np.rad2deg(val_offsets[:, 0]),
                    epoch,
                )
                writer.add_histogram(
                    "activations/val_offset_pitch_deg",
                    np.rad2deg(val_offsets[:, 1]),
                    epoch,
                )
                writer.add_histogram(
                    "activations/val_offset_yaw_deg",
                    np.rad2deg(val_offsets[:, 2]),
                    epoch,
                )

            # Diagnostic heatmap figures
            if log_figures and (
                epoch == 0 or epoch % 25 == 0 or epoch == num_epochs - 1
            ):
                fig = create_jacobian_diagnostic_figure(
                    target_A_bar=val_targets[0], pred_jacobian=val_jacobians[0]
                )
                writer.add_figure("diagnostics/jacobian_heatmap", fig, epoch)
                plt.close(fig)

        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train: {avg_train_loss:.5f} (Frob: {avg_train_frob:.5f}) | "
                f"Val: {avg_val_loss:.5f} (RMSE: {val_rmse:.4f}) | "
                f"GradNorm: {avg_grad_norm:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    if writer is not None:
        writer.flush()
        writer.close()

    return model, params, history
