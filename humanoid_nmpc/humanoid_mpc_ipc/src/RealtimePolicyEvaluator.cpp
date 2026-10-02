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
#include <typeinfo>
#include <vector>

#include <Eigen/Core>

#include <ocs2_core/Types.h>
#include <ocs2_core/control/ControllerBase.h>
#include <ocs2_core/control/FeedforwardController.h>
#include <ocs2_core/control/LinearController.h>
#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>

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
    result = alpha * nodes[segment.first] + (scalar_t(1.0) - alpha) * nodes[segment.first + 1];
  } else {
    result = nodes[0];
  }
}

}  // namespace

RealtimePolicyEvaluator::RealtimePolicyEvaluator(const ModelDimensions& dimensions)
    : dimensions_(dimensions),
      gain_(matrix_t::Zero(static_cast<Eigen::Index>(dimensions.inputDim), static_cast<Eigen::Index>(dimensions.stateDim))) {}

RealtimePolicyEvaluator::Outcome RealtimePolicyEvaluator::evaluate(
    const PrimalSolution& policy, scalar_t time, const vector_t& state, vector_t& mpcState, vector_t& mpcInput, size_t& mode) {
  const Eigen::Index stateDim = static_cast<Eigen::Index>(dimensions_.stateDim);
  const Eigen::Index inputDim = static_cast<Eigen::Index>(dimensions_.inputDim);

  // Everything is checked before anything is written.
  if (policy.timeTrajectory_.empty() || policy.stateTrajectory_.size() != policy.timeTrajectory_.size() || state.size() != stateDim ||
      policy.modeSchedule_.modeSequence.size() != policy.modeSchedule_.eventTimes.size() + 1) {
    return Outcome::kMismatchedPolicy;
  }
  const index_alpha_t stateSegment = LinearInterpolation::timeSegment(time, policy.timeTrajectory_);
  if (!segmentHasSize(stateSegment, policy.stateTrajectory_, stateDim, /*cols=*/1)) {
    return Outcome::kMismatchedPolicy;
  }

  // The exact types: a class derived from either may compute its input otherwise.
  const ControllerBase* controller = policy.controllerPtr_.get();
  const FeedforwardController* feedforward = nullptr;
  const LinearController* linear = nullptr;
  if (controller != nullptr && typeid(*controller) == typeid(FeedforwardController)) {
    feedforward = static_cast<const FeedforwardController*>(controller);
  } else if (controller != nullptr && typeid(*controller) == typeid(LinearController)) {
    linear = static_cast<const LinearController*>(controller);
  } else {
    return Outcome::kUnsupportedController;
  }

  // As FeedforwardController::computeInput() and LinearController::computeInput(), on the controller's own time stamps.
  if (feedforward != nullptr) {
    if (feedforward->timeStamp_.empty() || feedforward->uffArray_.size() != feedforward->timeStamp_.size()) {
      return Outcome::kMismatchedPolicy;
    }
    const index_alpha_t inputSegment = LinearInterpolation::timeSegment(time, feedforward->timeStamp_);
    if (!segmentHasSize(inputSegment, feedforward->uffArray_, inputDim, /*cols=*/1)) {
      return Outcome::kMismatchedPolicy;
    }
    interpolateInto(inputSegment, feedforward->uffArray_, mpcInput);
  } else {
    if (linear->timeStamp_.empty() || linear->biasArray_.size() != linear->timeStamp_.size() ||
        linear->gainArray_.size() != linear->timeStamp_.size()) {
      return Outcome::kMismatchedPolicy;
    }
    const index_alpha_t inputSegment = LinearInterpolation::timeSegment(time, linear->timeStamp_);
    if (!segmentHasSize(inputSegment, linear->biasArray_, inputDim, /*cols=*/1) ||
        !segmentHasSize(inputSegment, linear->gainArray_, inputDim, stateDim)) {
      return Outcome::kMismatchedPolicy;
    }
    interpolateInto(inputSegment, linear->biasArray_, mpcInput);
    interpolateInto(inputSegment, linear->gainArray_, gain_);
    mpcInput.noalias() += gain_ * state;
  }

  // As MRT_BASE::evaluatePolicy(), on the policy's time trajectory.
  interpolateInto(stateSegment, policy.stateTrajectory_, mpcState);
  mode = policy.modeSchedule_.modeAtTime(time);
  return Outcome::kEvaluated;
}

}  // namespace ocs2::humanoid::ipc
