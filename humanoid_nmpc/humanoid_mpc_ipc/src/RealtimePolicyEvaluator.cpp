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

#include "humanoid_mpc_ipc/RealtimePolicyEvaluator.h"

#include <cstddef>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/control/ControllerBase.h"
#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"
#include "ocs2_core/control/ManifoldLinearController.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/misc/LinearInterpolation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"

#include "humanoid_mpc_ipc/PolicyControllers.h"

namespace ocs2::humanoid::ipc {
namespace {

using LinearInterpolation::index_alpha_t;

// True when the nodes `segment` interpolates between are `rows` x `cols`, so that interpolateInto() combines them as
// OCS2 does (it would pick one of two nodes of different sizes) and writes into an output of that size.
template <typename Data, typename Allocator>
bool segmentHasSize(const index_alpha_t& segment, const std::vector<Data, Allocator>& nodes, Eigen::Index rows, Eigen::Index cols) {
  const size_t first = static_cast<size_t>(segment.first);
  const size_t last = nodes.size() > 1 ? first + 1 : first;
  if (last >= nodes.size()) {
    return false;
  }
  return nodes[first].rows() == rows && nodes[first].cols() == cols && nodes[last].rows() == rows && nodes[last].cols() == cols;
}

// LinearInterpolation::interpolate(segment, nodes) assigned to `result`: the same expression, so the same doubles, but
// into storage `result` already has instead of a vector or matrix returned by value.
template <typename Data, typename Allocator, typename Result>
void interpolateInto(const index_alpha_t& segment, const std::vector<Data, Allocator>& nodes, Result& result) {
  if (nodes.size() > 1) {
    const scalar_t alpha = segment.second;
    result = alpha * nodes[segment.first] + (1.0 - alpha) * nodes[segment.first + 1];
  } else {
    result = nodes[0];
  }
}

Eigen::Index sizeOf(size_t dimension) {
  return static_cast<Eigen::Index>(dimension);
}

// FeedforwardController::computeInput() into `mpcInput`; false, without writing, when the controller's nodes at `time`
// are not of `inputDim`.
bool computeFeedforwardInput(const FeedforwardController& controller, scalar_t time, Eigen::Index inputDim, vector_t& mpcInput) {
  if (controller.timeStamp_.empty() || controller.uffArray_.size() != controller.timeStamp_.size()) {
    return false;
  }
  const index_alpha_t inputSegment = LinearInterpolation::timeSegment(time, controller.timeStamp_);
  if (!segmentHasSize(inputSegment, controller.uffArray_, inputDim, /*cols=*/1)) {
    return false;
  }
  interpolateInto(inputSegment, controller.uffArray_, mpcInput);
  return true;
}

}  // namespace

RealtimePolicyEvaluator::RealtimePolicyEvaluator(const ModelDimensions& dimensions, const StateManifold* absl_nullable stateManifold)
    : dimensions_(dimensions),
      stateManifold_(stateManifold),
      gain_(stateManifold == nullptr ? matrix_t::Zero(sizeOf(dimensions.inputDim), sizeOf(dimensions.stateDim)) : matrix_t()),
      delta_(stateManifold == nullptr ? vector_t() : vector_t::Zero(sizeOf(stateManifold->getTangentDim()))),
      nodeInput_(vector_t::Zero(sizeOf(dimensions.inputDim))),
      planState_(vector_t::Zero(sizeOf(dimensions.stateDim))) {
  if (stateManifold_ != nullptr) {
    // Once, here: evaluate() hands the manifold states of dimensions.stateDim without checking them again.
    ABSL_CHECK_EQ(stateManifold_->getAmbientDim(), dimensions_.stateDim)
        << "[RealtimePolicyEvaluator] The state manifold's ambient dimension is not the model's state dimension.";
  }
}

RealtimePolicyEvaluator::Outcome RealtimePolicyEvaluator::evaluate(
    const PrimalSolution& policy, scalar_t time, const vector_t& state, vector_t& mpcState, vector_t& mpcInput, size_t& mode) {
  const Eigen::Index stateDim = sizeOf(dimensions_.stateDim);
  const Eigen::Index inputDim = sizeOf(dimensions_.inputDim);

  // Everything is checked before anything is written.
  if (policy.timeTrajectory_.empty() || policy.stateTrajectory_.size() != policy.timeTrajectory_.size() || state.size() != stateDim ||
      policy.modeSchedule_.modeSequence.size() != policy.modeSchedule_.eventTimes.size() + 1) {
    return Outcome::kMismatchedPolicy;
  }
  const index_alpha_t stateSegment = LinearInterpolation::timeSegment(time, policy.timeTrajectory_);
  if (!segmentHasSize(stateSegment, policy.stateTrajectory_, stateDim, /*cols=*/1)) {
    return Outcome::kMismatchedPolicy;
  }

  // The exact types (PolicyControllers.h): a controller of another type may compute its input otherwise. On a flat
  // state a LinearController, on a state manifold a ManifoldLinearController.
  const ControllerBase* absl_nullable controller = policy.controllerPtr_.get();
  if (controller == nullptr) {
    return Outcome::kUnsupportedController;
  }
  const FeedforwardController* absl_nullable feedforward = asFeedforwardController(*controller);
  const bool flat = stateManifold_ == nullptr;
  const LinearController* absl_nullable linear = feedforward == nullptr && flat ? asLinearController(*controller) : nullptr;
  const ManifoldLinearController* absl_nullable manifoldLinear =
      feedforward == nullptr && !flat ? asManifoldLinearController(*controller) : nullptr;
  if (feedforward == nullptr && linear == nullptr && manifoldLinear == nullptr) {
    return Outcome::kUnsupportedController;
  }

  // As FeedforwardController::computeInput(), LinearController::computeInput() and ManifoldLinearController::
  // computeInput(), on the controller's own time stamps. Each checks the controller before it writes the input.
  bool inputWritten = false;
  if (feedforward != nullptr) {
    inputWritten = computeFeedforwardInput(*feedforward, time, inputDim, mpcInput);
  } else if (linear != nullptr) {
    inputWritten = computeLinearInput(*linear, time, state, mpcInput);
  } else {
    inputWritten = computeManifoldInput(*manifoldLinear, time, state, mpcInput);
  }
  if (!inputWritten) {
    return Outcome::kMismatchedPolicy;
  }

  // As MRT_BASE::evaluatePolicy(), on the policy's time trajectory: linearly, or along the state manifold.
  if (flat) {
    interpolateInto(stateSegment, policy.stateTrajectory_, mpcState);
  } else {
    interpolatePlanState(stateSegment, policy.stateTrajectory_, mpcState);
  }
  mode = policy.modeSchedule_.modeAtTime(time);
  return Outcome::kEvaluated;
}

bool RealtimePolicyEvaluator::computeLinearInput(const LinearController& controller,
                                                 scalar_t time,
                                                 const vector_t& state,
                                                 vector_t& mpcInput) {
  if (controller.timeStamp_.empty() || controller.biasArray_.size() != controller.timeStamp_.size() ||
      controller.gainArray_.size() != controller.timeStamp_.size()) {
    return false;
  }
  const index_alpha_t inputSegment = LinearInterpolation::timeSegment(time, controller.timeStamp_);
  if (!segmentHasSize(inputSegment, controller.biasArray_, sizeOf(dimensions_.inputDim), /*cols=*/1) ||
      !segmentHasSize(inputSegment, controller.gainArray_, sizeOf(dimensions_.inputDim), sizeOf(dimensions_.stateDim))) {
    return false;
  }
  interpolateInto(inputSegment, controller.biasArray_, mpcInput);
  interpolateInto(inputSegment, controller.gainArray_, gain_);
  mpcInput.noalias() += gain_ * state;
  return true;
}

bool RealtimePolicyEvaluator::computeManifoldInput(const ManifoldLinearController& controller,
                                                   scalar_t time,
                                                   const vector_t& state,
                                                   vector_t& mpcInput) {
  const size_t nodes = controller.getTimeStamps().size();
  if (nodes == 0 || controller.getAnchorStates().size() != nodes || controller.getNominalInputs().size() != nodes ||
      controller.getFeedbackGains().size() != nodes) {
    return false;
  }
  const index_alpha_t segment = LinearInterpolation::timeSegment(time, controller.getTimeStamps());
  const Eigen::Index inputDim = sizeOf(dimensions_.inputDim);
  if (!segmentHasSize(segment, controller.getAnchorStates(), sizeOf(dimensions_.stateDim), /*cols=*/1) ||
      !segmentHasSize(segment, controller.getNominalInputs(), inputDim, /*cols=*/1) ||
      !segmentHasSize(segment, controller.getFeedbackGains(), inputDim, delta_.size())) {
    return false;
  }
  // ManifoldLinearController::computeInput()'s blend of the two node laws, alpha the weight of node k, with the same
  // expressions: the node laws into storage of their own, then (alpha, 1 - alpha) combined coefficient by coefficient.
  const size_t node = static_cast<size_t>(segment.first);
  const scalar_t alpha = segment.second;
  if (nodes == 1 || alpha >= 1.0) {
    computeNodeInput(controller, node, state, mpcInput);
  } else if (alpha <= 0.0) {
    computeNodeInput(controller, node + 1, state, mpcInput);
  } else {
    computeNodeInput(controller, node, state, mpcInput);
    computeNodeInput(controller, node + 1, state, nodeInput_);
    mpcInput = alpha * mpcInput + (1.0 - alpha) * nodeInput_;
  }
  return true;
}

void RealtimePolicyEvaluator::computeNodeInput(const ManifoldLinearController& controller,
                                               size_t node,
                                               const vector_t& state,
                                               vector_t& input) {
  stateManifold_->differenceInto(controller.getAnchorStates()[node], state, delta_);
  input = controller.getNominalInputs()[node];
  input.noalias() += controller.getFeedbackGains()[node] * delta_;
}

void RealtimePolicyEvaluator::interpolatePlanState(const index_alpha_t& segment,
                                                   const vector_array_t& stateTrajectory,
                                                   vector_t& mpcState) {
  // interpolateStateTrajectory() (StateTrajectoryInterpolation.h): the nearer node at either end of the segment, the
  // geodesic between. Its snap between nodes of different sizes cannot happen here: both have stateDim entries.
  if (stateTrajectory.size() == 1 || segment.second >= 1.0) {
    mpcState = stateTrajectory[static_cast<size_t>(segment.first)];
    return;
  }
  if (segment.second <= 0.0) {
    mpcState = stateTrajectory[static_cast<size_t>(segment.first) + 1];
    return;
  }
  stateManifold_->interpolateInto(stateTrajectory[static_cast<size_t>(segment.first)],
                                  stateTrajectory[static_cast<size_t>(segment.first) + 1], 1.0 - segment.second, planState_);
  mpcState = planState_;
}

}  // namespace ocs2::humanoid::ipc
