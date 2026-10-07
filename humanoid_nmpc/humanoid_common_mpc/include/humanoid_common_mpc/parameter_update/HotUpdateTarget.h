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
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_oc/oc_problem/OptimalControlProblem.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid {

/**
 * What one hot reload of a task file writes into: the reloaded file, the SqpSolver whose per-worker problems and settings
 * the hot-field appliers (HotFieldApplier.h) update, the reference manager and the coordinates the file's weights are
 * addressed on. MpcParameterUpdaterModule builds one per reload and hands it to every applier in turn.
 *
 * It also converts, once per reload, the blocks that several appliers read (footConstraint(), terminalCostScaling()), so
 * that a refused block is reported once rather than by every applier that needs it.
 *
 * Not thread-safe: built and used on the solver thread, between two solves, when no worker of the solver runs.
 */
class HotUpdateTarget {
 public:
  /**
   * The target of a reload of `task`, applied to `solver` and, when not null, `referenceManager`, with the weights
   * addressed on `layout` and an OCP input of `inputDim`. All four are kept and must outlive this object (it lives for
   * one reload). `source` names where the file came from, for the log.
   */
  HotUpdateTarget(const mpc_config::TaskFile* absl_nonnull task,
                  SqpSolver* absl_nonnull solver,
                  SwitchedModelReferenceManager* absl_nullable referenceManager,
                  const StateInputLayout* absl_nonnull layout,
                  size_t inputDim,
                  absl::string_view source);

  ~HotUpdateTarget() = default;
  HotUpdateTarget(const HotUpdateTarget&) = delete;
  HotUpdateTarget& operator=(const HotUpdateTarget&) = delete;

  /** The reloaded task file. */
  const mpc_config::TaskFile& task() const { return *task_; }

  /** Every worker's clone of the running problem (SqpSolver::getOcpDefinitions()), each of which a reload writes. */
  std::vector<OptimalControlProblem>& problems() { return solver_->getOcpDefinitions(); }

  /**
   * The first worker's problem, which says which terms the running problem carries: that is structural, decided when the
   * problem was assembled, and a reload never changes it. Precondition: problems() is not empty (the module checks it).
   */
  const OptimalControlProblem& runningProblem() const { return solver_->getOcpDefinitions().front(); }

  SqpSolver& solver() { return *solver_; }

  /** The reference manager, or null when the updater has none (its swing planner and ground are then not reloaded). */
  SwitchedModelReferenceManager* absl_nullable referenceManager() { return referenceManager_; }

  /** The coordinates the task file's weights are addressed on (the MPC's model settings). */
  const StateInputLayout& layout() const { return *layout_; }
  /** The contacts of the layout, in input order. */
  const std::vector<std::string>& contactNames() const { return layout_->contactNames; }
  size_t stateDim() const { return stateDim_; }
  /** The dimension of the OCP input: with basis-vector contact inputs the basis-space one, not the wrench-space one. */
  size_t inputDim() const { return inputDim_; }

  /** Where the reloaded file came from, for the log. */
  absl::string_view source() const { return source_; }

  /**
   * model_settings.foot_constraint of the reloaded file, converted on the first call; nullopt when its conversion
   * refused it, which the first call reports by the field.
   */
  const std::optional<ModelSettings::FootConstraintConfig>& footConstraint();

  /**
   * terminal_cost_scaling of the reloaded file, converted on the first call; nullopt when it was refused, which the
   * first call reports. It weighs both final_state_weights and the terminal CoM + ACoM cost.
   */
  const std::optional<scalar_t>& terminalCostScaling();

  /** Logs that `what` of this reload was refused, `status`, and the running values kept (reportNotApplied()). */
  void reportNotApplied(absl::string_view what, const absl::Status& status) const;

 private:
  const mpc_config::TaskFile* absl_nonnull task_;
  SqpSolver* absl_nonnull solver_;
  SwitchedModelReferenceManager* absl_nullable referenceManager_;
  const StateInputLayout* absl_nonnull layout_;
  size_t stateDim_ = 0;
  size_t inputDim_ = 0;
  std::string source_;
  // The blocks converted on first use: whether each was converted, and its value (nullopt: refused).
  bool footConstraintConverted_ = false;
  std::optional<ModelSettings::FootConstraintConfig> footConstraint_;
  bool terminalCostScalingConverted_ = false;
  std::optional<scalar_t> terminalCostScaling_;
};

}  // namespace ocs2::humanoid
