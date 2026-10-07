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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerInterface.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "humanoid_common_mpc/contact_planning/logic/ContactLogicState.h"
#include "humanoid_common_mpc/contact_planning/model/LipBlockIndices.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "humanoid_common_mpc/contact_planning/problem/ContactPlanningProblem.h"
#include "humanoid_common_mpc/contact_planning/problem/TermCollection.h"
#include "humanoid_common_mpc/contact_planning/search/SearchStage.h"

namespace ocs2::humanoid {

/**
 * Mixed-integer contact planner on a Linear Inverted Pendulum (LIP) model.
 *
 * Over a horizon of N nodes of duration dt the planner decides, per node and foot, whether the foot is in contact (binary
 * c) and where the feet are placed (continuous), together with the LIP CoM and ZMP trajectories. The problem is an
 * OCP-structured MIQP: the continuous part is a QP that MixedIntegerOcpQp relaxes with HPIPM, the combinatorial part
 * is enforced exactly by logical propagation on the binaries.
 *
 * The problem is not written here. It is a ContactPlanningProblem assembled by ContactPlanningTermFactory from the
 * term lists of the configuration (`contact_planning.textproto`): model blocks that compose the variable layout, costs,
 * soft and hard constraints, logic rules and assignment costs, each a named term with its own parameter block. Around
 * it, the listed search stages provide the warm start and the diving heuristic before the branch-and-bound and the
 * event-shift local search and the heading re-linearization after it. This class owns the assembled problem and the
 * stages, keeps the previous plan (the warm start and the consistency terms read it), builds the per-plan context and
 * runs the search. getFormulationSummary() prints what was assembled. Not thread-safe: one thread plans with it (the
 * ContactPlannerModule's worker or its caller).
 *
 * Reduced model, per stage k (world frame, yaw-aligned constraint frame), with the default blocks:
 *   state  x_k = [c_xy, v_xy, p_L, p_R]              (CoM position / velocity, foot positions)
 *   input  u_k = [zmp_xy, dp_L, dp_R, c_L, c_R]      (ZMP, foot displacements, contact binaries)
 * and, with the heading block, x += [theta, omega, psi_L, psi_R], u += [tau_L, tau_R, dpsi_L, dpsi_R]. The constants
 * below are the indices of the first two blocks, which are always first.
 */
class LipContactPlanner final : public ContactPlannerInterface {
 public:
  // The indices of the first two blocks (model/LipBlockIndices.h), with which every layout starts.
  static constexpr int kCx = kLipCx;
  static constexpr int kCy = kLipCy;
  static constexpr int kVx = kLipVx;
  static constexpr int kVy = kLipVy;
  static constexpr int kPlx = kLipPlx;
  static constexpr int kPly = kLipPly;
  static constexpr int kPrx = kLipPrx;
  static constexpr int kPry = kLipPry;
  static constexpr int kStateDim = kLipStateDim;
  static constexpr int kZx = kLipZx;
  static constexpr int kZy = kLipZy;
  static constexpr int kDlx = kLipDlx;
  static constexpr int kDly = kLipDly;
  static constexpr int kDrx = kLipDrx;
  static constexpr int kDry = kLipDry;
  static constexpr int kCl = kLipCl;
  static constexpr int kCr = kLipCr;
  static constexpr int kInputDim = kLipInputDim;
  static constexpr int kBinariesPerNode = ContactLogicState::kBinariesPerNode;  // c_L, c_R

  using Statistics = SearchStatistics;
  // The layout and the nominal heading trajectory are the problem's (problem/Layout.h, problem/ContactPlanningContext.h).
  using Layout = ocs2::humanoid::Layout;
  using HeadingNominal = ocs2::humanoid::HeadingNominal;

  /** The variable layout of a configuration's formulation, or the InvalidArgument of a formulation that cannot be assembled. */
  static absl::StatusOr<Layout> makeLayout(const ContactPlanningConfig& config);
  const Layout& getLayout() const { return problem_.layout(); }

  /**
   * Builds a planner, or returns the rejection of ContactPlanningConfig::validateStatus() - an InvalidArgument whose
   * message names the key to change - for an invalid configuration, and that of the term factory for a formulation it
   * cannot assemble. The only way to build one: the constructor is private, and it used to be a public one that threw
   * std::invalid_argument with the same message. makeContactPlanner builds through it.
   */
  static absl::StatusOr<std::unique_ptr<LipContactPlanner>> Create(ContactPlanningConfig config);

  /**
   * Plans from the given input. The previous plan (if any) seeds the incumbent. A problem that cannot be built or a
   * solver failure is logged and returns an invalid plan; a search stage that fails is logged and skipped.
   */
  ContactPlan plan(const ContactPlannerInput& input) override;

  /**
   * Replaces the configuration and re-assembles the problem and the stages from its term lists. The warm start (the
   * previous plan and its assignment) survives unless the grid or the variable layout changed: dropping it on every hot
   * reload made the plan after each edit start from scratch and move footholds and timing abruptly. A configuration that
   * does not validate, or whose lists cannot be assembled, is refused with the InvalidArgument naming the key, and the
   * running one is kept.
   */
  absl::Status setConfig(const ContactPlanningConfig& config) override;
  const ContactPlanningConfig& getConfig() const { return config_; }

  /** Drops the warm start. */
  void reset() override;

  const MiqpResult& getLastResult() const { return lastResult_; }
  const OcpQpProblem& getLastProblem() const { return lastProblem_; }
  const Statistics& getLastStatistics() const { return statistics_; }
  const ContactPlanningProblem& getProblem() const { return problem_; }
  const TermCollection<SearchStage>& getSearchStages() const { return searchStages_; }

  /** The assembled formulation: layout, every term with its description, the search stages and the planner settings. */
  std::string getFormulationSummary() const override;
  /**
   * The same for a configuration (what a planner built from it would print), or Create()'s refusal of it. It used to
   * build the planner with the throwing constructor, so an invalid configuration threw out of a summary.
   */
  static absl::StatusOr<std::string> formulationSummary(const ContactPlanningConfig& config);

  // The following are public for testing.
  /**
   * Builds the OCP-QP around the default nominal heading trajectory (previous plan, or the commanded yaw integrated), or
   * around `nominal`; makeContext()'s errors otherwise.
   */
  absl::StatusOr<OcpQpProblem> buildProblem(const ContactPlannerInput& input) const;
  absl::StatusOr<OcpQpProblem> buildProblem(const ContactPlannerInput& input, const HeadingNominal& nominal) const;
  /**
   * The nominal trajectory the frame terms are linearized around: the previous plan shifted to `input.time` when there
   * is a usable one, the commanded yaw rate and CoM velocity integrated from the input otherwise.
   *
   * `nodeDuration > 0` builds it for a grid of that node duration instead of planner.dt, which is what the cadence
   * stretch needs: the integrated branch is a function of time, so node k of a stretched grid has to be sampled at
   * k * nodeDuration. The previous-plan branch is per-node geometry and is indexed by node on either grid.
   */
  HeadingNominal defaultNominal(const ContactPlannerInput& input, scalar_t nodeDuration = 0.0) const;
  static HeadingNominal nominalFromSolution(const Layout& layout, const OcpQpSolution& solution);
  /** Yaw inertia used by the heading model, from the input (the robot model); an InvalidArgument if it is not positive. */
  static absl::StatusOr<scalar_t> yawInertia(const ContactPlannerInput& input);
  std::vector<MiqpBinaryVariable> binaryVariables() const;
  MiqpAssignment initialAssignment(const ContactPlannerInput& input) const;
  /** Forward logical propagation of the listed contact logic rules. */
  bool propagate(const ContactPlannerInput& input, MiqpAssignment& assignment) const;
  /** The listed assignment costs of an assignment (exact for complete assignments, a lower bound for partial ones). */
  scalar_t assignmentCost(const ContactPlannerInput& input, const MiqpAssignment& assignment) const;
  static int contactBinaryIndex(int node, size_t foot) { return ContactLogicState::contactBinaryIndex(node, foot); }
  /**
   * The per-plan context the terms read (public for the equivalence tests). `dtOverride > 0` builds the context on a
   * grid of that node duration instead of planner.dt, which is how the cadence stretch re-times every phase of an
   * incumbent together (the terms are all exact functions of the node duration). The context points into `input`,
   * `nominal` and this planner, which must outlive it. With the heading block, a nominal that does not cover every node
   * or an input without a positive yaw inertia is an InvalidArgument, and a configuration without the model-derived
   * parameters a FailedPrecondition.
   */
  absl::StatusOr<ContactPlanningContext> makeContext(const ContactPlannerInput& input,
                                                     const HeadingNominal& nominal,
                                                     scalar_t dtOverride = 0.0) const;
  ContactLogicState makeLogicState(const ContactPlannerInput& input) const;

 private:
  /** Private: Create() validates the configuration and builds the problem and the search stages it is handed here. */
  LipContactPlanner(ContactPlanningConfig config, ContactPlanningProblem problem, TermCollection<SearchStage> searchStages);

  /** Node shift between the previous plan and `input.time` (nodes), or -1 when the previous plan is not usable. */
  int previousPlanShift(const ContactPlannerInput& input) const;
  ContactPlan decode(const ContactPlannerInput& input, const ContactPlanningContext& ctx, const MiqpResult& result) const;
  void rebuildSolver();

  ContactPlanningConfig config_;
  ContactPlanningProblem problem_;
  TermCollection<SearchStage> searchStages_;
  std::unique_ptr<MixedIntegerOcpQp> miqp_;
  OcpQpProblem lastProblem_;
  MiqpResult lastResult_;
  Statistics statistics_;
  std::optional<ContactPlan> previousPlan_;
  MiqpAssignment previousAssignment_;
};

}  // namespace ocs2::humanoid
