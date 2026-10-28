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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/planner/ContactInvariantPlanner.h"
#include "humanoid_multi_contact/planner/MultiContactPlanner.h"
#include "humanoid_multi_contact/planner/StanceGraphPlanner.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"

namespace ocs2::humanoid::multi_contact {

/** The combined planner's registry name (kPlannerNames). */
inline constexpr absl::string_view kStanceGraphThenContactInvariantPlannerName =
    "stance_graph_then_contact_invariant";

/** The default of StanceGraphThenContactInvariantOptions::minimumStance. */
inline constexpr double kDefaultMinimumStance = 0.3;

/** The settings of StanceGraphThenContactInvariantPlanner. */
struct StanceGraphThenContactInvariantOptions {
  StanceGraphThenContactInvariantOptions();

  // Planner B's, breaking contacts at the second knot of their motions
  // (ContactBreak::kAtSecondKnot): solve 3 retimes the sequence.
  StanceGraphPlannerOptions stanceGraph;
  ContactInvariantPlannerOptions contactInvariant;
  // [s] The shortest a stance of planner B's sequence lasts once it is
  // retimed to the scenario's duration (the thresholding's minimum dwell).
  double minimumStance = kDefaultMinimumStance;
};

/**
 * Planner B as planner A's initialization (design section 5.2): the stance
 * graph planner's stance sequence and configurations, retimed to the
 * scenario's duration (retimedPlan(): planner B's timing law is quasi-static
 * and its motions take minutes), are the contact variables and the state of
 * planner A's solve 3, which alone smooths the motion and redistributes the
 * forces with the dynamics, the torque limits and the support polygons
 * (HardContactRefinement). Planner B's errors when it finds no sequence; then
 * planner A's.
 *
 * Not thread-safe.
 */
class StanceGraphThenContactInvariantPlanner final
    : public MultiContactPlanner {
 public:
  /** InvalidArgument for inconsistent options. */
  static absl::StatusOr<std::unique_ptr<StanceGraphThenContactInvariantPlanner>>
  Create(const StanceGraphThenContactInvariantOptions& options);

  StanceGraphThenContactInvariantPlanner(
      const StanceGraphThenContactInvariantPlanner&) = delete;
  StanceGraphThenContactInvariantPlanner& operator=(
      const StanceGraphThenContactInvariantPlanner&) = delete;
  ~StanceGraphThenContactInvariantPlanner() override = default;

  absl::StatusOr<config::MultiContactPlan> plan(
      const LoadedScenario& loaded) override;
  absl::string_view name() const override {
    return kStanceGraphThenContactInvariantPlannerName;
  }

 private:
  StanceGraphThenContactInvariantPlanner(
      const StanceGraphThenContactInvariantOptions& options,
      std::unique_ptr<StanceGraphPlanner> stanceGraph);

  StanceGraphThenContactInvariantOptions options_;
  std::unique_ptr<StanceGraphPlanner> stanceGraph_;
};

}  // namespace ocs2::humanoid::multi_contact
