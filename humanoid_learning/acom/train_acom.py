"""Training pipeline for Angular Center of Mass (aCOM) using JAX and Optax.

Optimizes the SIREN network parameters by minimizing the Frobenius norm error
between the analytical network Jacobian and the locked-inertia normalized CMM.
"""

from typing import Dict, List, Tuple
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax

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
        return total_loss, {"frob_loss": frob_loss, "reg_loss": reg_loss}

    @jax.jit
    def train_step(params, opt_state, q_batch, target_A_bar):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, q_batch, target_A_bar
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    @jax.jit
    def eval_step(params, q_batch, target_A_bar):
        loss, aux = loss_fn(params, q_batch, target_A_bar)
        return loss, aux

    return train_step, eval_step


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
) -> Tuple[SirenACOM, List[Tuple[jnp.ndarray, jnp.ndarray]], Dict[str, List[float]]]:
    """Trains a SIREN aCOM model on the provided dataset."""
    rng = jax.random.PRNGKey(seed)

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
    steps_per_epoch = num_train // batch_size
    total_steps = num_epochs * steps_per_epoch
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate, decay_steps=total_steps, alpha=0.01
    )
    optimizer = optax.adamw(learning_rate=lr_schedule, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    train_step_fn, eval_step_fn = create_train_step(
        model, optimizer, reg_weight=reg_weight
    )

    history = {"train_loss": [], "val_loss": [], "val_frob": []}

    start_time = time.time()
    for epoch in range(num_epochs):
        # Shuffle training set
        rng, k_perm = jax.random.split(rng)
        perm = jax.random.permutation(k_perm, num_train)
        q_train_shuffled = q_train[perm]
        A_train_shuffled = A_train[perm]

        train_losses = []
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            q_batch = q_train_shuffled[start_idx:end_idx]
            A_batch = A_train_shuffled[start_idx:end_idx]

            params, opt_state, loss_val, aux = train_step_fn(
                params, opt_state, q_batch, A_batch
            )
            train_losses.append(float(loss_val))

        # Validation
        val_loss, val_aux = eval_step_fn(params, q_val, A_val)

        avg_train_loss = float(np.mean(train_losses))
        avg_val_loss = float(val_loss)
        avg_val_frob = float(val_aux["frob_loss"])

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_frob"].append(avg_val_frob)

        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Frob Loss: {avg_val_frob:.6f} | Time: {elapsed:.1f}s"
            )

    return model, params, history
