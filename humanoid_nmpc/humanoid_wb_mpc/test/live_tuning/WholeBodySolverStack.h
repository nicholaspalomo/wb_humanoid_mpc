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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_mpc/MPC_MRT_Interface.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_sqp/SqpMpc.h"

#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"

namespace ocs2::humanoid::live_tuning_test {

/** Whether a WholeBodySolverStack registers the parameter updater with its solver. */
enum class Updater {
  kRegistered,
  kNone,
};

/**
 * The whole-body MPC of a task file around the SQP solver, as the MPC node runs it, with or without the parameter
 * updater registered (makeWholeBodyMpcParameterUpdater(), no file watched), on the G1 standing still and then stepping
 * once with each foot (walkingSchedule()). The robot is the MPC's own model: it moves as the last policy predicts. Not
 * thread-safe; one test drives it.
 */
class WholeBodySolverStack {
 public:
  WholeBodySolverStack(const mpc_config::TaskFile& task, Updater updater);
  ~WholeBodySolverStack() = default;
  WholeBodySolverStack(const WholeBodySolverStack&) = delete;
  WholeBodySolverStack& operator=(const WholeBodySolverStack&) = delete;

  /** Whether the MPC was built; a test failure was recorded when not. */
  bool ok() const { return interface_ != nullptr && mrt_ != nullptr; }

  WBMpcInterface& interface() { return *interface_; }
  SqpMpc& mpc() { return *mpc_; }
  /** The updater, or nullptr without one. */
  MpcParameterUpdaterModule* absl_nullable updater() { return updater_.get(); }

  /**
   * Hands `task` to the updater and applies it at once, as the preSolverRun() of the next solve would: before any
   * worker of the solver runs. Precondition: the stack has an updater.
   */
  void applyNow(const mpc_config::TaskFile& task);

  /**
   * Runs the reference manager over the next solve's horizon, as that solve would first thing: what a reload handed it
   * (the swing trajectory planner's configuration) is then in the swing trajectories the terms read.
   */
  void replanReferences();

  /** Runs `count` solves, one per kSolvePeriod of solver time from where the last left off, and returns each policy. */
  std::vector<PrimalSolution> solve(size_t count);

  // [s] of solver time between two solves.
  static constexpr scalar_t kSolvePeriod = 0.02;

 private:
  std::unique_ptr<WBMpcInterface> interface_;
  std::unique_ptr<SqpMpc> mpc_;
  // Registered with the solver; OCS2's addSynchronizedModule() takes a shared_ptr. Null without an updater.
  std::shared_ptr<MpcParameterUpdaterModule> updater_;
  std::unique_ptr<MPC_MRT_Interface> mrt_;
  SystemObservation observation_;
};

/** Returns the largest difference between the policies, relative to max(1, |a|, |b|) entry by entry; 0 when bitwise equal. */
scalar_t policyDifference(const std::vector<PrimalSolution>& expected, const std::vector<PrimalSolution>& actual);

}  // namespace ocs2::humanoid::live_tuning_test
