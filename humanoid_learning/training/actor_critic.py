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


"""The actor-critic network that behavioral cloning warm-starts and the ONNX export serializes.

It is the network train_ppo.py defined until PPO training moved to Brax's own networks; bc_warmstart.py and
export_onnx.py are built around its layer names.
"""

from flax import linen
import jax
import jax.numpy as jnp

# Width of the two shared torso layers.
_HIDDEN_DIM = 256


class ActorCritic(linen.Module):
    """An actor-critic network whose actor and critic share a two-layer torso.

    The torso is two ELU layers (`Dense_0`, `Dense_1`); the actor's mean is `Dense_2` and the critic's value
    `Dense_3`, and the actor's log standard deviation `log_std` is a parameter of its own, independent of the
    observation. export_onnx.py serializes `Dense_0` to `Dense_2`, the deterministic policy.

    Attributes:
        action_dim: The number of actions.
    """

    action_dim: int

    @linen.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Evaluates the actor and the critic.

        Args:
            x: Observations, (batch, observation_dim).

        Returns:
            The actor's mean action (batch, action_dim), its log standard deviation (action_dim,), and the critic's
            value (batch,).
        """
        # Shared torso
        h = linen.Dense(_HIDDEN_DIM)(x)
        h = linen.elu(h)
        h = linen.Dense(_HIDDEN_DIM)(h)
        h = linen.elu(h)

        # Actor head (mean & log_std)
        actor_mean = linen.Dense(self.action_dim)(h)
        log_std = self.param("log_std", linen.initializers.zeros, (self.action_dim,))

        # Critic head (value)
        value = linen.Dense(1)(h)

        return actor_mean, log_std, jnp.squeeze(value, axis=-1)
