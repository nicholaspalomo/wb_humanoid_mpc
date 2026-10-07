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

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"

#include <ocs2_core/Types.h>
#include <ocs2_core/control/FeedforwardController.h>
#include <ocs2_core/model_data/Multiplier.h>
#include <ocs2_mpc/MPC_BASE.h>
#include <ocs2_oc/oc_data/PerformanceIndex.h>
#include <ocs2_oc/oc_data/PrimalSolution.h>
#include <ocs2_oc/oc_data/ProblemMetrics.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/oc_solver/SolverBase.h>

namespace ocs2::mpc_test {

/**
 * A stand-in for a real solver, for tests of the code AROUND the solver: the MPC-MRT hand-over, the resets, the
 * controllers built on MPC_MRT_Interface. It solves nothing. Every run() goes through SolverBase::run(), so the
 * reference manager and the synchronized modules run exactly as they do around a real solver, and then the solver
 * "plans" what the test scripted: by default the initial state held over the horizon, on the reference manager's mode
 * schedule.
 *
 * The script can be changed from the test thread while another thread runs the solver.
 */
class ScriptedSolver final : public SolverBase {
 public:
  /** The state the solver plans at `time` for a solve that started at (initTime, initState). */
  using PlanFunction = std::function<vector_t(scalar_t time, scalar_t initTime, const vector_t& initState)>;

  explicit ScriptedSolver(size_t inputDim) : inputDim_(inputDim) {}
  ~ScriptedSolver() override = default;

  /** Replaces the plan; an empty function restores the default, the initial state held. */
  void setPlan(PlanFunction plan) {
    std::lock_guard<std::mutex> lock(mutex_);
    plan_ = std::move(plan);
  }

  /** The next `count` solves throw std::runtime_error, as a solver that fails its QP does. */
  void failNextSolves(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    solvesToFail_ = count;
  }

  /** Every solve throws until this is called with false. */
  void failEverySolve(bool fail) {
    std::lock_guard<std::mutex> lock(mutex_);
    failEverySolve_ = fail;
  }

  size_t numResets() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return numResets_;
  }
  size_t numSolves() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return numSolves_;
  }
  size_t numFailedSolves() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return numFailedSolves_;
  }
  /** Initial time and state of the last solve that succeeded. */
  scalar_t lastInitTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initTime_;
  }
  vector_t lastInitState() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return initState_;
  }

  void reset() override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++numResets_;
    finalTime_ = 0.0;
  }

  const OptimalControlProblem& getOptimalControlProblem() const override { return ocp_; }
  const PerformanceIndex& getPerformanceIndeces() const override { return performanceIndex_; }
  size_t getNumIterations() const override { return 1; }
  const std::vector<PerformanceIndex>& getIterationsLog() const override { return iterationsLog_; }
  scalar_t getFinalTime() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return finalTime_;
  }

  void getPrimalSolution(scalar_t finalTime, PrimalSolution* absl_nonnull primalSolutionPtr) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    constexpr size_t kNumNodes = 11;
    primalSolutionPtr->clear();
    const scalar_t endTime = std::max(finalTime, initTime_);
    for (size_t k = 0; k < kNumNodes; ++k) {
      const scalar_t time = initTime_ + (endTime - initTime_) * static_cast<scalar_t>(k) / static_cast<scalar_t>(kNumNodes - 1);
      primalSolutionPtr->timeTrajectory_.push_back(time);
      primalSolutionPtr->stateTrajectory_.push_back(plan_ ? plan_(time, initTime_, initState_) : initState_);
      primalSolutionPtr->inputTrajectory_.push_back(vector_t::Zero(inputDim_));
    }
    primalSolutionPtr->modeSchedule_ = modeSchedule_;
    primalSolutionPtr->controllerPtr_ =
        std::make_unique<FeedforwardController>(primalSolutionPtr->timeTrajectory_, primalSolutionPtr->inputTrajectory_);
  }

  const ProblemMetrics& getSolutionMetrics() const override { return problemMetrics_; }
  ScalarFunctionQuadraticApproximation getValueFunction(scalar_t /*time*/, const vector_t& state) const override {
    return ScalarFunctionQuadraticApproximation::Zero(static_cast<int>(state.size()));
  }
  ScalarFunctionQuadraticApproximation getHamiltonian(scalar_t /*time*/, const vector_t& state, const vector_t& input) override {
    return ScalarFunctionQuadraticApproximation::Zero(static_cast<int>(state.size()), static_cast<int>(input.size()));
  }
  vector_t getStateInputEqualityConstraintLagrangian(scalar_t /*time*/, const vector_t& /*state*/) const override { return vector_t(); }
  MultiplierCollection getIntermediateDualSolution(scalar_t /*time*/) const override { return MultiplierCollection(); }

 private:
  void runImpl(scalar_t initTime, const vector_t& initState, scalar_t finalTime) override {
    std::lock_guard<std::mutex> lock(mutex_);
    ++numSolves_;
    if (failEverySolve_ || solvesToFail_ > 0) {
      if (solvesToFail_ > 0) --solvesToFail_;
      ++numFailedSolves_;
      throw std::runtime_error("[ScriptedSolver] scripted failure: Failed to solve QP");
    }
    initTime_ = initTime;
    initState_ = initState;
    finalTime_ = finalTime;
    modeSchedule_ = getReferenceManager().getModeSchedule();
  }
  void runImpl(scalar_t initTime,
               const vector_t& initState,
               scalar_t finalTime,
               const ControllerBase* absl_nullable /*externalControllerPtr*/) override {
    runImpl(initTime, initState, finalTime);
  }
  void runImpl(scalar_t initTime, const vector_t& initState, scalar_t finalTime, const PrimalSolution& /*primalSolution*/) override {
    runImpl(initTime, initState, finalTime);
  }

  const size_t inputDim_;
  mutable std::mutex mutex_;
  PlanFunction plan_;
  size_t solvesToFail_ = 0;
  bool failEverySolve_ = false;
  size_t numResets_ = 0;
  size_t numSolves_ = 0;
  size_t numFailedSolves_ = 0;
  scalar_t initTime_ = 0.0;
  vector_t initState_;
  scalar_t finalTime_ = 0.0;
  ModeSchedule modeSchedule_;

  OptimalControlProblem ocp_;
  PerformanceIndex performanceIndex_;
  std::vector<PerformanceIndex> iterationsLog_;
  ProblemMetrics problemMetrics_;
};

/** An MPC around a ScriptedSolver. */
class ScriptedMpc final : public MPC_BASE {
 public:
  ScriptedMpc(mpc::Settings settings, size_t inputDim) : MPC_BASE(std::move(settings)), solver_(inputDim) {}
  ~ScriptedMpc() override = default;

  ScriptedSolver* absl_nonnull getSolverPtr() override { return &solver_; }
  const ScriptedSolver* absl_nonnull getSolverPtr() const override { return &solver_; }
  ScriptedSolver& solver() { return solver_; }

 protected:
  void calculateController(scalar_t initTime, const vector_t& initState, size_t initMode, scalar_t finalTime) override {
    solver_.run(initTime, initState, initMode, finalTime);
  }

 private:
  ScriptedSolver solver_;
};

}  // namespace ocs2::mpc_test
