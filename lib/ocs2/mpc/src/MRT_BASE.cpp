/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include "ocs2_mpc/MRT_BASE.h"

#include <utility>

#include <ocs2_oc/oc_data/StateTrajectoryInterpolation.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>

#include "absl/base/nullability.h"
#include "absl/log/log.h"

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
MRT_BASE::MRT_BASE() {
  reset();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MRT_BASE::reset() {
  // The policies are taken out under the lock and destroyed after it is released (at the end of this function, in the
  // reverse order of declaration): destroying a policy frees its trajectories and controller, which must not keep the
  // control thread's try-lock in updatePolicy() failing.
  std::unique_ptr<CommandData> droppedActiveCommand;
  std::unique_ptr<CommandData> droppedBufferCommand;
  std::unique_ptr<PrimalSolution> droppedActivePrimalSolution;
  std::unique_ptr<PrimalSolution> droppedBufferPrimalSolution;
  std::unique_ptr<PerformanceIndex> droppedActivePerformanceIndices;
  std::unique_ptr<PerformanceIndex> droppedBufferPerformanceIndices;
  {
    std::lock_guard<std::mutex> lock(bufferMutex_);

    policyReceivedEver_ = false;
    newPolicyInBuffer_ = false;
    consecutiveFailedPolicyUpdateLocks_.store(0, std::memory_order_relaxed);

    droppedActiveCommand.swap(activeCommandPtr_);
    droppedBufferCommand.swap(bufferCommandPtr_);
    droppedActivePrimalSolution.swap(activePrimalSolutionPtr_);
    droppedBufferPrimalSolution.swap(bufferPrimalSolutionPtr_);
    droppedActivePerformanceIndices.swap(activePerformanceIndicesPtr_);
    droppedBufferPerformanceIndices.swap(bufferPerformanceIndicesPtr_);
    // A policy moved to the buffer before this reset must not count as current after it.
    policyEpoch_.fetch_add(1);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool MRT_BASE::discardBufferedPolicy() {
  // Taken out under the lock, destroyed after it is released (see reset()).
  std::unique_ptr<CommandData> droppedCommand;
  std::unique_ptr<PrimalSolution> droppedPrimalSolution;
  std::unique_ptr<PerformanceIndex> droppedPerformanceIndices;
  bool dropped = false;
  {
    std::lock_guard<std::mutex> lock(bufferMutex_);
    dropped = newPolicyInBuffer_;
    newPolicyInBuffer_ = false;
    droppedCommand.swap(bufferCommandPtr_);
    droppedPrimalSolution.swap(bufferPrimalSolutionPtr_);
    droppedPerformanceIndices.swap(bufferPerformanceIndicesPtr_);
    policyEpoch_.fetch_add(1);
  }
  return dropped;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
const CommandData& MRT_BASE::getCommand() const {
  if (activeCommandPtr_ != nullptr) {
    return *activeCommandPtr_;
  } else {
    throw std::runtime_error("[MRT_BASE::getCommand] updatePolicy() should be called first!");
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
const PrimalSolution& MRT_BASE::getPolicy() const {
  if (activePrimalSolutionPtr_ != nullptr) {
    return *activePrimalSolutionPtr_;
  } else {
    throw std::runtime_error("[MRT_BASE::getPolicy] updatePolicy() should be called first!");
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
const PerformanceIndex& MRT_BASE::getPerformanceIndices() const {
  if (activePerformanceIndicesPtr_ != nullptr) {
    return *activePerformanceIndicesPtr_;
  } else {
    throw std::runtime_error("[MRT_BASE::getPerformanceIndices] updatePolicy() should be called first!");
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MRT_BASE::initRollout(const RolloutBase* absl_nonnull rolloutPtr) {
  rolloutPtr_.reset(rolloutPtr->clone());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MRT_BASE::evaluatePolicy(scalar_t currentTime, const vector_t& currentState, vector_t& mpcState, vector_t& mpcInput, size_t& mode) {
  if (activePrimalSolutionPtr_ == nullptr) {
    throw std::runtime_error("[MRT_BASE::evaluatePolicy] updatePolicy() should be called first!");
  }

  if (currentTime > activePrimalSolutionPtr_->timeTrajectory_.back()) {
    LOG_EVERY_N_SEC(WARNING, 1.0) << "[MRT_BASE::evaluatePolicy] The requested currentTime is greater than the received plan: "
                                  << currentTime << " > " << activePrimalSolutionPtr_->timeTrajectory_.back();
  }

  mpcInput = activePrimalSolutionPtr_->controllerPtr_->computeInput(currentTime, currentState);
  mpcState = interpolateStateTrajectory(stateManifoldPtr_.get(), currentTime, activePrimalSolutionPtr_->timeTrajectory_,
                                        activePrimalSolutionPtr_->stateTrajectory_);

  mode = activePrimalSolutionPtr_->modeSchedule_.modeAtTime(currentTime);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void MRT_BASE::rolloutPolicy(
    scalar_t currentTime, const vector_t& currentState, const scalar_t& timeStep, vector_t& mpcState, vector_t& mpcInput, size_t& mode) {
  if (rolloutPtr_ == nullptr) {
    throw std::runtime_error("[MRT_BASE::rolloutPolicy] rollout class is not set! Use initRollout() to initialize it!");
  }

  if (activePrimalSolutionPtr_ == nullptr) {
    throw std::runtime_error("[MRT_BASE::rolloutPolicy] updatePolicy() should be called first!");
  }

  if (currentTime > activePrimalSolutionPtr_->timeTrajectory_.back()) {
    LOG_EVERY_N_SEC(WARNING, 1.0) << "[MRT_BASE::rolloutPolicy] The requested currentTime is greater than the received plan: "
                                  << currentTime << " > " << activePrimalSolutionPtr_->timeTrajectory_.back();
  }

  // perform a rollout
  scalar_array_t timeTrajectory;
  size_array_t postEventIndicesStock;
  vector_array_t stateTrajectory, inputTrajectory;
  const scalar_t finalTime = currentTime + timeStep;
  rolloutPtr_->run(currentTime, currentState, finalTime, activePrimalSolutionPtr_->controllerPtr_.get(),
                   activePrimalSolutionPtr_->modeSchedule_, timeTrajectory, postEventIndicesStock, stateTrajectory, inputTrajectory);

  mpcState = stateTrajectory.back();
  mpcInput = inputTrajectory.back();

  mode = activePrimalSolutionPtr_->modeSchedule_.modeAtTime(finalTime);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool MRT_BASE::updatePolicy() {
  std::unique_lock<std::mutex> lock(bufferMutex_, std::try_to_lock);
  if (lock.owns_lock()) {
    consecutiveFailedPolicyUpdateLocks_.store(0, std::memory_order_relaxed);
    if (newPolicyInBuffer_) {
      // update the active solution from buffer
      activeCommandPtr_.swap(bufferCommandPtr_);
      activePrimalSolutionPtr_.swap(bufferPrimalSolutionPtr_);
      activePerformanceIndicesPtr_.swap(bufferPerformanceIndicesPtr_);
      std::swap(activePolicyEpoch_, bufferPolicyEpoch_);
      newPolicyInBuffer_ = false;  // make sure we don't swap in the old policy again
      return true;
    } else {
      return false;  // No policy update: the buffer contains nothing new.
    }
  } else {
    // Counted, not logged: this is the control thread, where a log line can allocate or block. The controllers report
    // the count off this thread (getConsecutiveFailedPolicyUpdateLocks()).
    consecutiveFailedPolicyUpdateLocks_.fetch_add(1, std::memory_order_relaxed);
    failedPolicyUpdateLocks_.fetch_add(1, std::memory_order_relaxed);
    return false;  // No policy update: the lock could not be acquired.
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
bool MRT_BASE::moveToBuffer(std::unique_ptr<CommandData> commandDataPtr,
                            std::unique_ptr<PrimalSolution> primalSolutionPtr,
                            std::unique_ptr<PerformanceIndex> performanceIndicesPtr) {
  if (commandDataPtr == nullptr) {
    throw std::runtime_error("[MRT_BASE::moveToBuffer] commandDataPtr cannot be a null pointer!");
  }

  if (primalSolutionPtr == nullptr) {
    throw std::runtime_error("[MRT_BASE::moveToBuffer] primalSolutionPtr cannot be a null pointer!");
  }

  if (performanceIndicesPtr == nullptr) {
    throw std::runtime_error("[MRT_BASE::moveToBuffer] performanceIndicesPtr cannot be a null pointer!");
  }

  std::lock_guard<std::mutex> lk(bufferMutex_);
  // use swap such that the old objects are destroyed after releasing the lock.
  bufferCommandPtr_.swap(commandDataPtr);
  bufferPrimalSolutionPtr_.swap(primalSolutionPtr);
  bufferPerformanceIndicesPtr_.swap(performanceIndicesPtr);
  bufferPolicyEpoch_ = policyEpoch_.load();

  const bool replaced = newPolicyInBuffer_;
  newPolicyInBuffer_ = true;
  policyReceivedEver_ = true;
  return replaced;
}

}  // namespace ocs2
