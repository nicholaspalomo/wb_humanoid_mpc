/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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
#include <vector>

#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/MixedIntegerOcpQp.h"

namespace ocs2::humanoid {

/**
 * Mixed-integer contact planner on a Linear Inverted Pendulum (LIP) model.
 *
 * Over a horizon of N nodes of duration dt the planner decides, per node and foot, whether the foot is in contact (binary
 * c) and where the feet are placed (continuous), together with the LIP CoM and ZMP trajectories. The problem is an
 * OCP-structured MIQP: the continuous part (LIP dynamics, ZMP support region, foot motion, reachability) is a QP that
 * MixedIntegerOcpQp relaxes with HPIPM, the combinatorial part (no flight phase, min/max swing and contact durations,
 * foot alternation, a price per contact switch) is enforced exactly by logical propagation on the binaries.
 *
 * Reduced model, per stage k (world frame, yaw-aligned constraint frame):
 *   state  x_k = [c_xy, v_xy, p_L, p_R]              (CoM position / velocity, foot positions)
 *   input  u_k = [zmp_xy, dp_L, dp_R, c_L, c_R]      (ZMP, foot displacements, contact binaries)
 *   c_{k+1} = LIP(c_k, v_k, zmp_k),  p_{k+1} = p_k + dp_k
 *
 * After the branch-and-bound an event-shift local search moves every lift-off / touch-down of the incumbent by one node
 * (fixed-assignment QPs) while it improves the objective, which refines the phase timing cheaply.
 */
class LipContactPlanner {
 public:
  enum StateIndex : int { CX = 0, CY, VX, VY, PLX, PLY, PRX, PRY, STATE_DIM };
  enum InputIndex : int { ZX = 0, ZY, DLX, DLY, DRX, DRY, CL, CR, INPUT_DIM };
  static constexpr int kBinariesPerNode = 2;  // c_L, c_R

  struct Statistics {
    int numBranchAndBoundRelaxations = 0;
    int numLocalSearchQps = 0;
    int totalQpIterations = 0;
    scalar_t branchAndBoundTime = 0.0;
    scalar_t localSearchTime = 0.0;
    bool localSearchImproved = false;
  };

  explicit LipContactPlanner(ContactPlanningConfig config);

  /** Plans from the given input. The previous plan (if any) seeds the incumbent. Never throws on solver failure: an invalid
   * plan is returned instead. */
  ContactPlan plan(const ContactPlannerInput& input);

  /** Replaces the configuration (validated) and drops the warm start. */
  void setConfig(const ContactPlanningConfig& config);
  const ContactPlanningConfig& getConfig() const { return config_; }

  /** Drops the warm start. */
  void reset();

  const MiqpResult& getLastResult() const { return lastResult_; }
  const OcpQpProblem& getLastProblem() const { return problem_; }
  const Statistics& getLastStatistics() const { return statistics_; }

  // The following are public for testing.
  OcpQpProblem buildProblem(const ContactPlannerInput& input) const;
  std::vector<MiqpBinaryVariable> binaryVariables() const;
  MiqpAssignment initialAssignment(const ContactPlannerInput& input) const;
  /** Forward logical propagation of the contact logic (duration limits, no flight, foot alternation). */
  bool propagate(const ContactPlannerInput& input, MiqpAssignment& assignment) const;
  /** Switch cost of the decided transitions (exact for complete assignments, a lower bound for partial ones). */
  scalar_t assignmentCost(const ContactPlannerInput& input, const MiqpAssignment& assignment) const;
  static int contactBinaryIndex(int node, size_t foot) { return kBinariesPerNode * node + static_cast<int>(foot); }

 private:
  std::optional<MiqpAssignment> warmStartAssignment(const ContactPlannerInput& input) const;
  void localSearch(const ContactPlannerInput& input, const MiqpAssignment& initial, scalar_t timeBudget);
  ContactPlan decode(const ContactPlannerInput& input, const MiqpResult& result) const;
  int initialPhaseNodes(const ContactPlannerInput& input, size_t foot) const;
  void rebuildSolver();

  ContactPlanningConfig config_;
  std::unique_ptr<MixedIntegerOcpQp> miqp_;
  OcpQpProblem problem_;
  MiqpResult lastResult_;
  Statistics statistics_;
  std::optional<ContactPlan> previousPlan_;
  MiqpAssignment previousAssignment_;
};

}  // namespace ocs2::humanoid
