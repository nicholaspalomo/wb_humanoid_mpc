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


"""The actor-critic network and its checkpoint, the behavioral cloning warmstart that trains it, and the ONNX export of its policy."""

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

# The dimensions of bc_warmstart.py's synthetic demonstrations, which the command-line test trains on.
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


def _random_normalizer(seed: int) -> actor_critic.ObservationNormalizer:
    rng = np.random.default_rng(seed)
    return actor_critic.ObservationNormalizer(
        mean=rng.standard_normal(_OBS_DIM).astype(np.float32),
        std=rng.uniform(0.1, 3.0, _OBS_DIM).astype(np.float32),
    )


def _random_observations(seed: int, count: int) -> np.ndarray:
    return (
        3.0 * np.random.default_rng(seed).standard_normal((count, _OBS_DIM))
    ).astype(np.float32)


class CheckpointTest(unittest.TestCase):
    def test_a_checkpoint_reads_back_to_the_same_policy(self) -> None:
        network, params = _network_and_params()
        normalizer = _random_normalizer(seed=2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "policy.npz")
            actor_critic.save_checkpoint(path, params, normalizer)
            loaded_params, loaded_normalizer = actor_critic.load_checkpoint(path)
        jax.tree_util.tree_map(
            np.testing.assert_array_equal, loaded_params, jax.device_get(params)
        )
        np.testing.assert_array_equal(loaded_normalizer.mean, normalizer.mean)
        np.testing.assert_array_equal(loaded_normalizer.std, normalizer.std)
        observations = jnp.asarray(_random_observations(seed=3, count=8))
        np.testing.assert_array_equal(
            actor_critic.policy_action(
                network, loaded_params, loaded_normalizer, observations
            ),
            actor_critic.policy_action(network, params, normalizer, observations),
        )

    def test_a_checkpoint_without_a_key_is_refused_naming_it(self) -> None:
        _, params = _network_and_params()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "policy.npz")
            actor_critic.save_checkpoint(path, params, _random_normalizer(seed=4))
            with np.load(path) as archive:
                arrays = {key: archive[key] for key in archive.files}
            for key in ("params/Dense_1/kernel", "normalizer/std"):
                incomplete = {
                    name: value for name, value in arrays.items() if name != key
                }
                incomplete_path = os.path.join(directory, "incomplete.npz")
                np.savez(incomplete_path, **incomplete)
                with self.assertRaisesRegex(KeyError, key):
                    actor_critic.load_checkpoint(incomplete_path)

    def test_the_normalizer_centers_and_scales_and_leaves_constants_unscaled(
        self,
    ) -> None:
        observations = _random_observations(seed=5, count=500)
        observations[:, 3] = 7.0
        normalizer = actor_critic.normalizer_from_observations(observations)
        normalized = np.asarray(normalizer.normalize(observations))
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-5)
        varying = np.arange(_OBS_DIM) != 3
        np.testing.assert_allclose(normalized.std(axis=0)[varying], 1.0, rtol=1e-4)
        self.assertEqual(normalizer.std[3], 1.0)
        np.testing.assert_allclose(normalized[:, 3], 0.0, atol=1e-6)


class BehavioralCloningTest(unittest.TestCase):
    def test_the_actor_learns_the_demonstrated_actions(self) -> None:
        rng = np.random.default_rng(0)
        observations = rng.standard_normal((64, _OBS_DIM)).astype(np.float32)
        actions = np.tanh(0.5 * observations[:, :_ACT_DIM])
        result = bc_warmstart.train_behavioral_cloning(
            observations,
            actions,
            epochs=20,
            learning_rate=1e-3,
            batch_size=16,
            verbose=False,
        )
        self.assertEqual(len(result.epoch_losses), 20)
        self.assertLess(result.epoch_losses[-1], 0.5 * result.epoch_losses[0])

    def test_every_minibatch_is_one_step(self) -> None:
        rng = np.random.default_rng(1)
        observations = rng.standard_normal((50, _OBS_DIM)).astype(np.float32)
        actions = rng.uniform(-1.0, 1.0, (50, _ACT_DIM)).astype(np.float32)
        for batch_size in (1, 7, 50, 64):
            result = bc_warmstart.train_behavioral_cloning(
                observations,
                actions,
                epochs=2,
                learning_rate=1e-3,
                batch_size=batch_size,
                verbose=False,
            )
            self.assertEqual(
                result.num_updates,
                2 * bc_warmstart.updates_per_epoch(50, batch_size),
                msg=f"batch_size {batch_size}",
            )
        self.assertEqual(bc_warmstart.updates_per_epoch(50, 7), 8)
        self.assertEqual(bc_warmstart.updates_per_epoch(50, 64), 1)

    def test_actions_outside_the_policy_range_are_refused(self) -> None:
        observations = np.zeros((4, _OBS_DIM), dtype=np.float32)
        actions = np.zeros((4, _ACT_DIM), dtype=np.float32)
        actions[2, 1] = 1.5
        with self.assertRaises(ValueError):
            bc_warmstart.train_behavioral_cloning(
                observations,
                actions,
                epochs=1,
                learning_rate=1e-3,
                batch_size=2,
                verbose=False,
            )

    def test_the_command_line_trains_in_its_batches_and_saves_the_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bc_policy.npz")
            bc_warmstart.main(
                ["--epochs=2", "--batch_size=300", "--seed=3", f"--output_path={path}"]
            )
            params, normalizer = actor_critic.load_checkpoint(path)
        # The same run in process: the checkpoint holds what it trained, not the initialization.
        rng = np.random.default_rng(3)
        observations = rng.standard_normal((1000, _OBS_DIM)).astype(np.float32)
        actions = rng.uniform(-1.0, 1.0, (1000, _ACT_DIM)).astype(np.float32)
        expected = bc_warmstart.train_behavioral_cloning(
            observations,
            actions,
            epochs=2,
            learning_rate=1e-3,
            batch_size=300,
            seed=3,
            verbose=False,
        )
        self.assertEqual(expected.num_updates, 8)
        jax.tree_util.tree_map(
            np.testing.assert_array_equal, params, jax.device_get(expected.params)
        )
        np.testing.assert_array_equal(normalizer.mean, expected.normalizer.mean)
        initialization = actor_critic.ActorCritic(action_dim=_ACT_DIM).init(
            jax.random.PRNGKey(3), jnp.zeros((1, _OBS_DIM))
        )
        self.assertFalse(
            np.array_equal(
                params["params"]["Dense_2"]["kernel"],
                initialization["params"]["Dense_2"]["kernel"],
            )
        )


class ExportOnnxTest(unittest.TestCase):
    def test_the_exported_graph_is_the_python_policy(self) -> None:
        network, params = _network_and_params()
        normalizer = _random_normalizer(seed=6)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "policy.npz")
            output = os.path.join(directory, "policy.onnx")
            actor_critic.save_checkpoint(checkpoint, params, normalizer)
            export_onnx.main(
                [f"--checkpoint_path={checkpoint}", f"--output_path={output}"]
            )
            model = onnx.load(output)
        onnx.checker.check_model(model)

        evaluator = onnx.reference.ReferenceEvaluator(model)
        # Observations of the normalizer's spread, where the tanh is not saturated and its slope shows any difference.
        unit = (
            np.random.default_rng(7).standard_normal((32, _OBS_DIM)).astype(np.float32)
        )
        observations = normalizer.mean + normalizer.std * unit
        expected = np.asarray(
            actor_critic.policy_action(
                network, params, normalizer, jnp.asarray(observations)
            )
        )
        self.assertTrue(np.all(np.abs(expected) < 1.0))
        for observation, action in zip(observations, expected):
            (exported,) = evaluator.run(
                None, {"observation": observation[np.newaxis, :]}
            )
            np.testing.assert_allclose(exported[0], action, rtol=1e-5, atol=1e-6)

    def test_the_export_needs_a_checkpoint_with_every_layer(self) -> None:
        with self.assertRaises(SystemExit):
            export_onnx.parse_args([])
        _, params = _network_and_params()
        weights = export_onnx.actor_weights(params)
        del weights["Dense_1/bias"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(KeyError, "Dense_1/bias"):
                export_onnx.export_policy_to_onnx(
                    weights,
                    _random_normalizer(seed=8),
                    os.path.join(directory, "policy.onnx"),
                )

    def test_a_normalizer_of_another_size_is_refused(self) -> None:
        _, params = _network_and_params()
        normalizer = actor_critic.ObservationNormalizer(
            mean=np.zeros(_OBS_DIM + 1, dtype=np.float32),
            std=np.ones(_OBS_DIM + 1, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                export_onnx.export_policy_to_onnx(
                    export_onnx.actor_weights(params),
                    normalizer,
                    os.path.join(directory, "policy.onnx"),
                )


if __name__ == "__main__":
    unittest.main()
