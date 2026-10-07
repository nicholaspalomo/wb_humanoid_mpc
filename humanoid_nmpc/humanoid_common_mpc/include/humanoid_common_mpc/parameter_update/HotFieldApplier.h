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

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_sqp/SqpSolver.h"

#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/**
 * One part of a hot reload of the task file: it converts the RELOAD_HOT fields it is responsible for (fields()) and
 * writes them into the running MPC. A formulation assembles its appliers in one list (HotFieldApplierList.h;
 * centroidalHotFieldAppliers(), wholeBodyHotFieldAppliers()) that MpcParameterUpdaterModule runs in order for every
 * reload, so that both formulations share one updater and differ only in the list.
 *
 * An applier finds the terms it writes by name in the running problem (CostTermNames.h, ContactTermNames.h) and skips the
 * ones the problem does not carry, which is the normal case; a block whose conversion refuses it is reported by its
 * field and the running values are kept. Appliers write disjoint parts of the problem, so that their order cannot change
 * the result: where two share a term, each reads it, replaces its own part and writes back the other's (the
 * stateInputQuadraticCost of a centroidal MPC with basis-vector contact inputs, whose Q is QuadraticCostWeightsApplier's
 * and whose R is BasisInputsCostApplier's).
 *
 * Every applier class defines `static absl::Span<const absl::string_view> staticFields()` over a constexpr array in its
 * .cpp, inside a LINT.IfChange block whose ThenChange names the protos of the fields it lists, and implements fields() as
 * `final`, returning it: a formulation's list of hot fields is then known without an MPC (the reload-class coverage
 * test, humanoid_nmpc/humanoid_mpc_validation/test/testHotFieldCoverage.cpp).
 *
 * Not thread-safe: called on the solver thread only, between two solves.
 */
class HotFieldApplier {
 public:
  virtual ~HotFieldApplier() = default;
  HotFieldApplier(const HotFieldApplier&) = delete;
  HotFieldApplier& operator=(const HotFieldApplier&) = delete;

  /** What it applies, for the log of a reload it stopped part-way. */
  virtual absl::string_view name() const = 0;

  /**
   * The task-file fields it applies, by their path from the file message ("state_weights",
   * "multiple_shooting.g_max"); a block's path stands for every field below it.
   */
  virtual absl::Span<const absl::string_view> fields() const = 0;

  /**
   * Converts its fields of target.task() and writes them into every worker's problem (target.problems()), the solver,
   * the reference manager or what else it holds. A refused block is reported by its field (target.reportNotApplied())
   * and the running values are kept.
   */
  virtual void apply(HotUpdateTarget& target) = 0;

  /**
   * Called first thing in every preSolverRun() of the updater, before any reload, with the solver and the reference
   * manager (null when the updater has none), which has just built this solve's references: for what has to follow the
   * references (the contact-implicit terms follow the ground they stand on). Nothing by default.
   */
  virtual void beforeEverySolve(SqpSolver& /*solver*/, const SwitchedModelReferenceManager* absl_nullable /*referenceManager*/) {}

 protected:
  HotFieldApplier() = default;
};

}  // namespace ocs2::humanoid
