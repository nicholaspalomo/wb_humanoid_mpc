/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

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
******************************************************************************/

#pragma once

#include <cstddef>
#include <memory>
#include <random>

#include "ocs2_core/Types.h"
#include "ocs2_core/control/ControllerType.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PerformanceIndex.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

/**
 * Random OCS2 objects for the conversion tests, deterministic for a given generator state. The primal solutions look
 * like those of OCS2's multiple-shooting solvers: the two nodes of an event share one time, the post-event indices
 * point at the second of them, and the controller is built on the time trajectory; a ManifoldLinearController exactly
 * as the SQP builds it on a state manifold (multiple_shooting::toPrimalSolution()).
 */
namespace ocs2::humanoid::ipc::test_data {

/** The size and kind of a random primal solution. */
struct PolicyShape {
  size_t nodes = 1;
  size_t stateDim = 1;
  size_t inputDim = 1;
  // At most (nodes - 1) / 3 events fit, each with at least one node of its own on either side.
  size_t events = 0;
  ControllerType controllerType = ControllerType::FEEDFORWARD;
  /**
   * The manifold the states live on, or null for flat ones. With one, stateDim is its ambient dimension and every state
   * is a point of it (each quaternion block normalized). A MANIFOLD_LINEAR controller needs one, and two nodes at least.
   */
  std::shared_ptr<const StateManifold> stateManifold = nullptr;
};

/** The state manifold of the manifold-policy tests, [E(3), Q, E(5)]: 12 ambient and 11 tangent dimensions. */
std::shared_ptr<const StateManifold> toyStateManifold();

/** A state drawn as randomVector() draws one and projected onto `manifold`. */
vector_t randomManifoldState(std::mt19937& generator, const StateManifold& manifold);

/** A vector of `size` values drawn from N(0, 10). */
vector_t randomVector(std::mt19937& generator, size_t size);

/** A size drawn uniformly from [low, high]. */
size_t randomSize(std::mt19937& generator, size_t low, size_t high);

/** A mode number drawn from [0, 2^40], so that modes beyond one byte are the rule. */
size_t randomMode(std::mt19937& generator);

/** An observation at a random time with a state and an input of the given sizes. */
SystemObservation randomObservation(std::mt19937& generator, size_t stateDim, size_t inputDim);

/** A mode schedule with `events` sorted event times in [startTime, startTime + 1]. */
ModeSchedule randomModeSchedule(std::mt19937& generator, size_t events, scalar_t startTime);

/** Target trajectories of `nodes` sorted times; inputDim == 0 leaves the input trajectory empty. */
TargetTrajectories randomTargetTrajectories(std::mt19937& generator, size_t nodes, size_t stateDim, size_t inputDim);

/** A performance index of random values. */
PerformanceIndex randomPerformanceIndex(std::mt19937& generator);

/** Command data of an observation and target trajectories of `targetNodes` nodes. */
CommandData randomCommandData(std::mt19937& generator, size_t stateDim, size_t inputDim, size_t targetNodes);

/**
 * A primal solution of the given shape, with a controller of the given type built on its time trajectory. A
 * MANIFOLD_LINEAR controller comes from multiple_shooting::toPrimalSolution(..., shape.stateManifold): its pre-event
 * nodes and its last node carry the anchor, input and gain of the node before them, not their own state.
 */
PrimalSolution randomPrimalSolution(std::mt19937& generator, const PolicyShape& shape);

}  // namespace ocs2::humanoid::ipc::test_data
