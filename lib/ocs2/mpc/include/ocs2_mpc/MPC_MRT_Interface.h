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

#include <condition_variable>
#include <csignal>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>

#include "absl/status/status.h"

#include <ocs2_core/misc/Benchmark.h>
#include <ocs2_core/model_data/Multiplier.h>
#include "ocs2_mpc/MPC_BASE.h"
#include "ocs2_mpc/MRT_BASE.h"

namespace ocs2 {

/**
 * A lean ROS independent interface to OCS2. In incorporates the functionality of the MPC and the MRT (trajectory tracking) modules.
 * Please refer to ocs2_double_integrator_example for a minimal example and tests
 */
class MPC_MRT_Interface final : public MRT_BASE {
 public:
  /**
   * Constructor
   * @param [in] mpc: The underlying MPC class to be used.
   */
  explicit MPC_MRT_Interface(MPC_BASE& mpc);

  ~MPC_MRT_Interface() override = default;

  /**
   * Resets the MPC (MPC_BASE::reset(): the solver, its reference manager and its synchronized modules), drops any policy
   * still waiting in the buffer (discardBufferedPolicy()) and sets the target trajectories to start from. Call it from
   * the thread that calls advanceMpc(), between two calls.
   */
  void resetMpcNode(const TargetTrajectories& initTargetTrajectories) override;

  /**
   * As resetMpcNode(), but resets the solver alone (MPC_BASE::resetSolver()): the reference manager and the synchronized
   * modules keep the schedule and the command state they are executing. The policy waiting in the buffer is dropped
   * all the same. Same threading rule as resetMpcNode().
   */
  void resetMpcSolver(const TargetTrajectories& initTargetTrajectories);

  void setCurrentObservation(const SystemObservation& currentObservation) override;

  SystemObservation getCurrentObservation();

  /*
   * Gets the ReferenceManager which manages both ModeSchedule and TargetTrajectories.
   */
  ReferenceManagerInterface& getReferenceManager();
  const ReferenceManagerInterface& getReferenceManager() const;

  /**
   * Advance the mpc module for one iteration. The evaluation methods can be called while this method is running. They will evaluate the
   * control law that was up-to-date at the last updatePolicy() call.
   *
   * @return absl::OkStatus() on success; InternalError if the solver threw; FailedPrecondition if the MPC did not run
   *         because the observation time is past the end of the previous solution, which only a reset cures.
   */
  absl::Status advanceMpc();

  /**
   * @brief Retrieves the gain matrix from solver capable of optimizing over LinearController type.
   *
   * @note This method is not thread-safe, meaning you can only access this data safely at the end of each MPC iteration.
   *
   * @param [in] time: query time
   * @return The gain matrix.
   */
  matrix_t getLinearFeedbackGain(scalar_t time);

  /**
   * @brief Access the solver's internal value function.
   *
   * @note This method is not thread-safe, meaning you can only access this data safely at the end of each MPC iteration.
   *
   * @param [in] time: query time
   * @param [in] state: query state
   * @return The quadratic approximation of the value function at the requested time and state.
   */
  ScalarFunctionQuadraticApproximation getValueFunction(scalar_t time, const vector_t& state) const;

  /**
   * @brief Computes the Lagrange multiplier related to the state-input constraints
   *
   * @note This method is not thread-safe, meaning you can only access this data safely at the end of each MPC iteration.
   *
   * @param [in] time: query time
   * @param [in] state: query state
   * @return The Lagrange multiplier at the requested time and state
   */
  vector_t getStateInputEqualityConstraintLagrangian(scalar_t time, const vector_t& state) const;

  /**
   * @brief Returns the intermediate dual solution at the given time.
   *
   * @note This method is not thread-safe, meaning you can only access this data safely at the end of each MPC iteration.
   *
   * @param [in] time: The inquiry time
   * @return The collection of multipliers associated to state/state-input, equality/inequality Lagrangian terms.
   */
  MultiplierCollection getIntermediateDualSolution(scalar_t time) const;

 private:
  /**
   * Updates the buffer variables from the MPC object. This method is automatically called by advanceMpc()
   *
   * @param [in] mpcInitObservation: The observation used to run the MPC.
   */
  void copyToBuffer(const SystemObservation& mpcInitObservation);

  MPC_BASE& mpc_;
  benchmark::RepeatedTimer mpcTimer_;
  // Solver crashes since the last solve that did not throw; the diagnostic dump is written for the first one only.
  size_t consecutiveCrashes_ = 0;

  // MPC inputs
  SystemObservation currentObservation_;
  std::mutex observationMutex_;
  // advanceMpc()'s copy of currentObservation_. Kept from one solve to the next so that the copy taken under
  // observationMutex_ reuses its storage: the control thread, which sets the observation every cycle, then waits at most
  // for a copy of a few vectors, never for an allocation the solver thread makes while it holds the lock.
  SystemObservation solverObservation_;
};

}  // namespace ocs2
