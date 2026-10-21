/******************************************************************************
Copyright (c) 2017, Farbod Farshidian. All rights reserved.

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

#include <Eigen/Dense>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>

#include "absl/base/nullability.h"

#include <ocs2_core/Types.h>
#include <ocs2_core/control/ControllerBase.h>
#include <ocs2_core/manifold/StateManifold.h>
#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_core/reference/ModeSchedule.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_oc/oc_data/PerformanceIndex.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>
#include <ocs2_oc/rollout/RolloutBase.h>

#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/SystemObservation.h"

namespace ocs2 {

/**
 * This class implements core MRT (Model Reference Tracking) functionality.
 * The responsibility of filling the buffer variables is left to the deriving classes.
 */
class MRT_BASE {
 public:
  MRT_BASE();

  virtual ~MRT_BASE() = default;

  /**
   * Resets the class to its instantiated state.
   */
  void reset();

  /**
   * Request the MPC node to reset. This method is a blocking method.
   *
   * @param [in] initTargetTrajectories: The initial desired cost trajectories.
   */
  virtual void resetMpcNode(const TargetTrajectories& initTargetTrajectories) = 0;

  /**
   * Whether the initial MPC policy has been already received.
   */
  bool initialPolicyReceived() const { return policyReceivedEver_; }

  /**
   * @brief setCurrentObservation notifies MPC of a new state
   * @param observation: the current measurement to send to the MPC
   */
  virtual void setCurrentObservation(const SystemObservation& observation) = 0;

  /**
   * Gets a reference to the command data corresponding to the current policy.
   * @warning access to the returned reference is not threadsafe. Read access and calls to updatePolicy() must be synced by the user.
   *
   * @return a constant reference to command data.
   */
  const CommandData& getCommand() const;

  /**
   * Gets a reference to the performance indices data corresponding to the current policy.
   * @warning access to the returned reference is not threadsafe. Read access and calls to updatePolicy() must be synced by the user.
   *
   * @return a constant reference to performance indices data.
   */
  const PerformanceIndex& getPerformanceIndices() const;

  /**
   * Gets a reference to current optimized policy.
   * @warning access to the returned reference is not threadsafe. Read access and calls to updatePolicy() must be synced by the user.
   *
   * @return constant reference to the policy data.
   */
  const PrimalSolution& getPolicy() const;

  /**
   * @brief Initializes rollout class to roll out a feedback policy
   * @param rolloutPtr: The rollout object to be used
   */
  void initRollout(const RolloutBase* absl_nonnull rolloutPtr);

  /**
   * Sets the manifold the state lives on (nullptr, the default, for a flat state). evaluatePolicy() then interpolates the
   * planned state along it (a slerp on every quaternion block) instead of linearly. The policy itself evaluates its
   * feedback on the manifold already (ManifoldLinearController). Not reset by reset().
   */
  void setStateManifold(std::shared_ptr<const StateManifold> stateManifold) { stateManifoldPtr_ = std::move(stateManifold); }

  /** The manifold the state lives on, or nullptr. */
  const std::shared_ptr<const StateManifold>& getStateManifold() const { return stateManifoldPtr_; }

  /**
   * @brief Evaluates the controller
   *
   * @param [in] currentTime: the query time.
   * @param [in] currentState: the query state.
   * @param [out] mpcState: the current nominal state of MPC.
   * @param [out] mpcInput: the optimized control input.
   * @param [out] mode: the active mode.
   */
  void evaluatePolicy(scalar_t currentTime, const vector_t& currentState, vector_t& mpcState, vector_t& mpcInput, size_t& mode);

  /**
   * @brief Rolls out the control policy from the current time and state to get the next state and input using the MPC policy.
   *
   * @param [in] currentTime: start time of the rollout.
   * @param [in] currentState: state to start rollout from.
   * @param [in] timeStep: duration of the forward rollout.
   * @param [out] mpcState: the new forwarded state of MPC.
   * @param [out] mpcInput: the new control input of MPC.
   * @param [out] mode: the active mode.
   */
  void rolloutPolicy(
      scalar_t currentTime, const vector_t& currentState, const scalar_t& timeStep, vector_t& mpcState, vector_t& mpcInput, size_t& mode);

  /**
   * Checks the data buffer for an update of the MPC policy. If a new policy
   * is available on the buffer this method will load it to the in-use policy.
   *
   * @return True if the policy is updated.
   */
  bool updatePolicy();

  /**
   * The failed try-locks of the policy buffer in updatePolicy() since its last successful one: updatePolicy() never
   * waits for the buffer, so while the thread that fills it holds the lock no newer policy is taken into use. Written
   * by the thread that calls updatePolicy(), which does not log (a log line can allocate or block); any thread may read
   * it and report it, as the MRT joint controllers do once it passes kPolicyUpdateLockContentionThreshold.
   */
  size_t getConsecutiveFailedPolicyUpdateLocks() const { return consecutiveFailedPolicyUpdateLocks_.load(std::memory_order_relaxed); }

  /** The failed try-locks of the policy buffer in updatePolicy() over the lifetime of this MRT. Any thread. */
  size_t getFailedPolicyUpdateLocks() const { return failedPolicyUpdateLocks_.load(std::memory_order_relaxed); }

  /** The consecutive failed try-locks past which getConsecutiveFailedPolicyUpdateLocks() is worth reporting. */
  static constexpr size_t kPolicyUpdateLockContentionThreshold = 5;

  /**
   * Drops a policy that is waiting in the buffer and starts a new policy epoch, so that no policy solved before this
   * call can be swapped in by updatePolicy() afterwards. The policy in use is left alone: it is read by the thread that
   * calls updatePolicy() without a lock, so only that thread may replace it, and it does so with the first policy moved
   * to the buffer after this call. isActivePolicyCurrent() tells that thread whether that has happened yet.
   *
   * Call it from the thread that fills the buffer (the one running the MPC) when the MPC is reset. Thread-safe with
   * respect to updatePolicy().
   *
   * @return True when it dropped a policy that updatePolicy() never took; false when the buffer held none, because the
   *         last policy moved to it was swapped in (or none was moved since the last discard). Read in the same critical
   *         section as the drop, so a caller can tell which policy may be in use without racing the swap.
   */
  bool discardBufferedPolicy();

  /**
   * True when the policy in use was moved to the buffer after the last discardBufferedPolicy(), i.e. it was solved
   * after the last MPC reset; false before any policy was received. Call it from the thread that calls updatePolicy().
   */
  bool isActivePolicyCurrent() const { return activePrimalSolutionPtr_ != nullptr && activePolicyEpoch_ == policyEpoch_.load(); }

 protected:
  /**
   * Puts a policy in the buffer, for updatePolicy() to swap in.
   *
   * @return True when it replaced a policy that updatePolicy() never took; false when the buffer held none (the policy
   *         moved before was swapped in, discarded, or there was none). Read in the same critical section as the move.
   */
  bool moveToBuffer(std::unique_ptr<CommandData> commandDataPtr,
                    std::unique_ptr<PrimalSolution> primalSolutionPtr,
                    std::unique_ptr<PerformanceIndex> performanceIndicesPtr);

  /**
   * Holds the lock of the policy buffer that updatePolicy() try-locks, for as long as the returned lock lives: for a
   * deriving class that must keep the buffer still across several steps, and for the tests of the try-lock counters.
   * Never call it from the thread that calls updatePolicy().
   */
  std::unique_lock<std::mutex> lockPolicyBuffer() const { return std::unique_lock<std::mutex>(bufferMutex_); }

 private:
  // flags on state of the class
  std::atomic_bool policyReceivedEver_;
  bool newPolicyInBuffer_;  // whether a new policy is waiting to be swapped in

  // variables related to the MPC output
  std::unique_ptr<CommandData> activeCommandPtr_;
  std::unique_ptr<CommandData> bufferCommandPtr_;
  std::unique_ptr<PrimalSolution> activePrimalSolutionPtr_;
  std::unique_ptr<PrimalSolution> bufferPrimalSolutionPtr_;
  std::unique_ptr<PerformanceIndex> activePerformanceIndicesPtr_;
  std::unique_ptr<PerformanceIndex> bufferPerformanceIndicesPtr_;

  // Policy epochs (discardBufferedPolicy()): the epoch a policy was moved to the buffer in travels with it through the
  // swap in updatePolicy(). policyEpoch_ is only incremented, under bufferMutex_.
  std::atomic<size_t> policyEpoch_{0};
  size_t bufferPolicyEpoch_{0};  // guarded by bufferMutex_
  size_t activePolicyEpoch_{0};  // owned by the thread calling updatePolicy()

  // thread safety
  mutable std::mutex bufferMutex_;  // for policy variables with the prefix (buffer*)
  // The failed try-locks of bufferMutex_ in updatePolicy(): since its last successful one, and in all. Written only by
  // the thread that calls updatePolicy(); atomics so that any thread may read them.
  std::atomic<size_t> consecutiveFailedPolicyUpdateLocks_{0};
  std::atomic<size_t> failedPolicyUpdateLocks_{0};

  // variables needed for policy evaluation
  std::unique_ptr<RolloutBase> rolloutPtr_;
  std::shared_ptr<const StateManifold> stateManifoldPtr_;
};

}  // namespace ocs2
