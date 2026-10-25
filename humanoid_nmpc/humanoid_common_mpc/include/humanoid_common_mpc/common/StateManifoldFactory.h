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

#include "absl/status/statusor.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/StateManifold.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"

namespace ocs2::humanoid {

/**
 * The state manifold of a robot model's state layout, for
 * OptimalControlProblem::stateManifoldPtr, the rollout, the MRT, the MPC link
 * and the policy evaluator: the product (ProductStateManifold) of the blocks of
 * the model's getTuningLayout(), a unit quaternion for the base orientation and
 * Euclidean for every other block. Never null.
 *
 * A free factory rather than a member of the model, so that the robot models'
 * constructors stay infallible; the interfaces and the MRT controllers call it
 * in their Create(). A std::shared_ptr because
 * OptimalControlProblem::stateManifoldPtr, ManifoldLinearController,
 * MRT_BASE::setStateManifold and RolloutBase::setStateManifold take one.
 *
 * InvalidArgument, naming the layout, when it is not placed on a state of
 * model.getStateDim() entries or places its orientation on another number of
 * rows than the quaternion's 4.
 */
absl::StatusOr<std::shared_ptr<const StateManifold>> stateManifoldOf(
    const MpcRobotModelBase<scalar_t>& model);

/** The same for the tuning layout `placedLayout` (placeTuningLayout()) of a
 * state of `stateDim` entries. */
absl::StatusOr<std::shared_ptr<const StateManifold>> stateManifoldOf(
    const TuningLayout& placedLayout, size_t stateDim);

}  // namespace ocs2::humanoid
