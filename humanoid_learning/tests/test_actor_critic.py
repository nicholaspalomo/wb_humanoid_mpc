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


"""The actor-critic network, the behavioral cloning warmstart that trains it, and the ONNX export of its policy."""

import os

# JAX picks its platform when it is first imported, so the tests pin it to the CPU before the imports below.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# pylint: disable=wrong-import-position  # JAX_PLATFORMS is set above, before JAX is imported.
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnx.reference

from humanoid_learning.export import export_onnx
from humanoid_learning.training import actor_critic
from humanoid_learning.training import bc_warmstart

# pylint: enable=wrong-import-position

# The dimensions bc_warmstart.py and export_onnx.py default to.
_OBS_DIM = 25
_ACT_DIM = 12


def _network_and_params() -> tuple[actor_critic.ActorCritic, dict]:
    network = actor_critic.ActorCritic(action_dim=_ACT_DIM)
    return network, network.init(jax.random.PRNGKey(0), jnp.zeros((1, _OBS_DIM)))


class ActorCriticTest(unittest.TestCase):
    def test_outputs_have_the_actor_and_critic_shapes(self) -> None:
        network, params = _network_and_params()
        mean, log_std, value = network.apply(params, jnp.zeros((4, _OBS_DIM)))
        self.assertEqual(mean.shape, (4, _ACT_DIM))
        self.assertEqual(log_std.shape, (_ACT_DIM,))
        self.assertEqual(value.shape, (4,))

    def test_the_layers_are_the_ones_the_export_reads(self) -> None:
        _, params = _network_and_params()
        self.assertEqual(
            set(params["params"]),
            {"Dense_0", "Dense_1", "Dense_2", "Dense_3", "log_std"},
        )
        self.assertEqual(params["params"]["Dense_0"]["kernel"].shape[0], _OBS_DIM)
        self.assertEqual(params["params"]["Dense_2"]["kernel"].shape[1], _ACT_DIM)


class BehavioralCloningTest(unittest.TestCase):
    def test_the_actor_learns_the_demonstrated_actions(self) -> None:
        rng = np.random.default_rng(0)
        observations = rng.standard_normal((64, _OBS_DIM)).astype(np.float32)
        actions = 0.5 * observations[:, :_ACT_DIM]
        _, losses = bc_warmstart.train_behavioral_cloning(
            observations, actions, epochs=20, learning_rate=1e-3, verbose=False
        )
        self.assertEqual(len(losses), 20)
        self.assertLess(losses[-1], losses[0])


class ExportOnnxTest(unittest.TestCase):
    def test_the_exported_policy_is_the_actors_mean(self) -> None:
        network, params = _network_and_params()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.onnx")
            export_onnx.export_dense_mlp_to_onnx(
                export_onnx.actor_weights(params), _OBS_DIM, _ACT_DIM, path
            )
            model = onnx.load(path)
        onnx.checker.check_model(model)

        observation = (
            np.random.default_rng(1).standard_normal((1, _OBS_DIM)).astype(np.float32)
        )
        (action,) = onnx.reference.ReferenceEvaluator(model).run(
            None, {"observation": observation}
        )
        mean, _, _ = network.apply(params, jnp.asarray(observation))
        np.testing.assert_allclose(action, np.asarray(mean), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
