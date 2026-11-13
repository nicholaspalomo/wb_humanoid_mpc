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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ContactMask.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/contact_planning/ContactPlan.h"
#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "legged_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "legged_common_mpc/contact_planning/ContactPlanningTermFactory.h"
#include "legged_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "legged_common_mpc/contact_planning/LipContactPlanner.h"
#include "legged_common_mpc/contact_planning/MixedIntegerOcpQp.h"
#include "legged_common_mpc/contact_planning/OcpQpHpipm.h"
#include "legged_common_mpc/contact_planning/cost/StepLengthCost.h"
#include "legged_common_mpc/contact_planning/cost/TerminalDcmCost.h"
#include "legged_common_mpc/contact_planning/cost/VelocityTrackingCost.h"
#include "legged_common_mpc/contact_planning/execution/ExecutionRule.h"
#include "legged_common_mpc/contact_planning/logic/ContactLogicHelpers.h"
#include "legged_common_mpc/contact_planning/logic/ContactLogicState.h"
#include "legged_common_mpc/contact_planning/logic/DoubleSupportPenaltyCost.h"
#include "legged_common_mpc/contact_planning/logic/MinimumDoubleSupportRule.h"
#include "legged_common_mpc/contact_planning/model/LipBlockIndices.h"
#include "legged_common_mpc/contact_planning/problem/AssignmentCost.h"
#include "legged_common_mpc/contact_planning/problem/ContactLogicRule.h"
#include "legged_common_mpc/contact_planning/problem/ContactPlanningContext.h"
#include "legged_common_mpc/contact_planning/problem/ContactPlanningProblem.h"
#include "legged_common_mpc/contact_planning/problem/ContactPlanningTerm.h"
#include "legged_common_mpc/contact_planning/problem/Layout.h"
#include "legged_common_mpc/contact_planning/problem/LayoutBuilder.h"
#include "legged_common_mpc/contact_planning/problem/LipConstraint.h"
#include "legged_common_mpc/contact_planning/problem/LipCost.h"
#include "legged_common_mpc/contact_planning/problem/LipModelBlock.h"
#include "legged_common_mpc/contact_planning/problem/RowBuilder.h"
#include "legged_common_mpc/contact_planning/problem/StageAccumulator.h"
#include "legged_common_mpc/contact_planning/problem/TermCollection.h"
#include "legged_common_mpc/contact_planning/search/CadenceStretchStage.h"
#include "legged_common_mpc/contact_planning/search/SearchStage.h"
#include "legged_common_mpc/gait/BipedModes.h"
#include "legged_nmpc/legged_common_mpc/test/support/StatusTestMacros.h"

namespace ocs2::legged {

namespace {

// The biped's two feet, which every case here is written for.
constexpr size_t kNumFeet = static_cast<size_t>(kBipedFootCount);

ContactPlanningConfig makeConfig() {
  ContactPlanningConfig config;
  config.planner.dt = 0.1;
  config.planner.numNodes = 12;
  config.planner.commitTime = 0.0;
  config.planner.maxBranchAndBoundNodes = 3000;
  config.planner.maxSolveTime = 10.0;
  config.shared.nomPosComRtFeetmeanEwrtWorldZ = 0.85;
  config.shared.gaitLimits.minSwingDuration = 0.3;
  config.shared.gaitLimits.maxSwingDuration = 0.5;
  config.shared.gaitLimits.minContactDuration = 0.15;
  config.shared.gaitLimits.minDoubleSupportDuration = 0.1;
  config.eventShiftLocalSearch.maxTime = 2.0;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  return config;
}

ContactPlanningConfig makeHeadingConfig() {
  ContactPlanningConfig config = makeConfig();
  config.setHeadingModel(true);
  config.yawTorqueBudgetParameters.maxTorqueFootEwrtFootZ = 20.0;
  config.yawTorqueBudgetParameters.doubleSupportMaxTorqueFeetmeanEwrtWorldZ =
      40.0;
  config.setSymmetricMaxYawFootRtBase(0.5);
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  return config;
}

ContactPlannerInput makeStandingInput() {
  ContactPlannerInput input;
  input.time = 3.0;
  input.posFeetRtWorldEwrtWorldXy[0] = vector2_t(0.0, 0.125);
  input.posFeetRtWorldEwrtWorldXy[1] = vector2_t(0.0, -0.125);
  input.contacts = {true, true};
  input.phaseElapsedTime = {5.0, 5.0};
  input.inertiaComEwrtWorldZz = 10.0;
  return input;
}

ContactPlannerInput makeWalkingInput() {
  ContactPlannerInput input = makeStandingInput();
  input.cmdLinvelBaseRtWorldEwrtWorldXy = vector2_t(0.4, 0.0);
  return input;
}

/** The term `name` of `collection`; CHECK-fails when there is none (the tests
 * ask only for terms they listed). */
template <typename T>
const T& termNamed(const TermCollection<T>& collection,
                   absl::string_view name) {
  const T* absl_nullable term = collection.find(name);
  CHECK(term != nullptr) << "no term named '" << name << "'";
  return *term;
}

/** The term `name` of `collection` as the concrete term `Derived`; CHECK-fails
 * when it is absent or of another type. */
template <typename Derived, typename T>
const Derived& termAs(const TermCollection<T>& collection,
                      absl::string_view name) {
  const Derived* absl_nullable derived =
      dynamic_cast<const Derived*>(&termNamed(collection, name));
  CHECK(derived != nullptr)
      << "the term '" << name << "' is not of the requested type";
  return *derived;
}

}  // namespace

/*============================================ layout and rows
 * ============================================*/

TEST(ContactPlanningTerms, LayoutIsComposedFromTheBlocksInOrder) {
  LEGGED_ASSERT_OK_AND_ASSIGN(const Layout lip,
                              LipContactPlanner::makeLayout(makeConfig()));
  EXPECT_EQ(lip.nx, kLipStateDim);
  EXPECT_EQ(lip.nu, kLipInputDim);
  EXPECT_FALSE(lip.hasHeading);
  EXPECT_EQ(lip.state(var::kPosComRtWorldEwrtWorldX), kLipCx);
  EXPECT_EQ(lip.state(var::posFootRtWorldEwrtWorldX(1)), kLipPrx);
  EXPECT_EQ(lip.input(var::contact(0)), kLipCl);
  EXPECT_EQ(lip.input(var::contact(1)), kLipCr);
  EXPECT_NE(
      lip.describe().find("x = [c_x c_y v_x v_y p_Lx p_Ly p_Rx p_Ry] (8)"),
      std::string::npos)
      << lip.describe();

  LEGGED_ASSERT_OK_AND_ASSIGN(
      const Layout heading, LipContactPlanner::makeLayout(makeHeadingConfig()));
  EXPECT_EQ(heading.nx, kLipStateDim + 2 + static_cast<int>(kNumFeet));
  EXPECT_EQ(heading.nu, kLipInputDim + 2 * static_cast<int>(kNumFeet));
  EXPECT_TRUE(heading.hasHeading);
  EXPECT_EQ(heading.headingModelYaw, kLipStateDim)
      << "the heading block appends, the LIP indices are unchanged";
  EXPECT_EQ(heading.state(var::kPosComRtWorldEwrtWorldX), kLipCx);
  EXPECT_EQ(heading.input(var::contact(1)), kLipCr);
  EXPECT_EQ(heading.torqueFootEwrtWorldZ(0), kLipInputDim);
  EXPECT_EQ(heading.errEulerzyxFootRtWorldYaw(1), kLipInputDim + 3);
}

TEST(ContactPlanningTerms, RowCountsPerNodeAreTheDocumentedOnes) {
  {
    LEGGED_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<LipContactPlanner> planner,
        LipContactPlanner::Create(makeConfig()));
    LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem problem,
                                planner->buildProblem(makeStandingInput()));
    EXPECT_EQ(problem.stages.front().numGeneralConstraints(), 27)
        << "1 no-flight + 12 support + 8 foot motion + 4 reach + 2 separation";
    EXPECT_EQ(problem.stages.back().numGeneralConstraints(), 6)
        << "the terminal node carries the kinematic rows only";
    EXPECT_EQ(
        static_cast<int>(problem.stages.front().softGeneralIndices.size()), 18);
    EXPECT_EQ(static_cast<int>(problem.stages.front().idxbu.size()), 6);
  }
  {
    LEGGED_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<LipContactPlanner> planner,
        LipContactPlanner::Create(makeHeadingConfig()));
    LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem problem,
                                planner->buildProblem(makeStandingInput()));
    EXPECT_EQ(problem.stages.front().numGeneralConstraints(), 41)
        << "+ 8 torque budget + 4 yaw pin + 2 hip range";
    EXPECT_EQ(problem.stages.back().numGeneralConstraints(), 8);
    EXPECT_EQ(static_cast<int>(problem.stages.front().idxbu.size()), 10);
  }
}

TEST(ContactPlanningTerms, TermsAreAssembledInListOrderAndOnTheirNodeSets) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kRegularization, term::kVelocityTracking};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {term::kNoFlight};
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  const ContactPlanningProblem& problem = planner->getProblem();
  ASSERT_EQ(problem.costs.size(), 2u);
  EXPECT_EQ(problem.costs.nameAt(0), term::kRegularization);
  EXPECT_EQ(problem.costs.nameAt(1), term::kVelocityTracking);
  EXPECT_EQ(termNamed(problem.costs, term::kVelocityTracking).nodeSet(),
            NodeSet::kAll);
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem qp,
                              planner->buildProblem(makeWalkingInput()));
  EXPECT_EQ(qp.stages.front().numGeneralConstraints(), 1);
  EXPECT_EQ(qp.stages.back().numGeneralConstraints(), 0);
  // The velocity tracking cost acts on the terminal node too: q_N carries -2 w
  // v_cmd on the velocity entries.
  EXPECT_NEAR(qp.stages.back().q(kLipVx),
              -2.0 * config.velocityTracking.weight * 0.4, 1.0e-12);
  EXPECT_NEAR(qp.stages.back().Q(kLipCx, kLipCx), config.regularization.state,
              1.0e-15)
      << "only the regularization touches the CoM";
}

// step_length: w ||dp - d_nom (1 - c)||^2 per foot and axis on the running
// nodes. The residual is affine in dp and the relaxed binary c, so the stage
// carries the cross terms of the two: R(dp, dp) = 2w, R(dp, c) = 2w d, R(c, c)
// = 2w d^2, r(dp) = -2w d, r(c) = -2w d^2 (0.5 u'Ru + r'u convention), summed
// over both axes for the c entries.
TEST(ContactPlanningTerms,
     StepLengthCostTiesTheSwingDisplacementToTheCommandedSpeed) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kStepLength};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {};
  config.stepLength.weight = 20.0;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  const StepLengthCost& cost =
      termAs<StepLengthCost>(planner->getProblem().costs, term::kStepLength);
  EXPECT_EQ(cost.nodeSet(), NodeSet::kRunning);
  // T_stride / T_swing = 2 (0.3 + 0.1) / 0.3
  EXPECT_NEAR(cost.strideToSwingRatio(), 8.0 / 3.0, 1.0e-12);
  const vector2_t nomErrPosFootRtWorldEwrtWorldXy =
      cost.computeNomErrPosFootRtWorldEwrtWorldXy(vector2_t(0.4, 0.0),
                                                  /*dt=*/0.1);
  EXPECT_NEAR(nomErrPosFootRtWorldEwrtWorldXy.x(), 0.4 * 0.1 * 8.0 / 3.0,
              1.0e-12);
  EXPECT_NEAR(nomErrPosFootRtWorldEwrtWorldXy.y(), 0.0, 1.0e-12);

  const scalar_t w = config.stepLength.weight;
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem qp,
                              planner->buildProblem(makeWalkingInput()));
  const OcpQpStage& stage = qp.stages.front();
  EXPECT_NEAR(stage.R(kLipDlx, kLipDlx), 2.0 * w, 1.0e-12);
  EXPECT_NEAR(stage.R(kLipDly, kLipDly), 2.0 * w, 1.0e-12);
  EXPECT_NEAR(stage.R(kLipDlx, kLipCl),
              2.0 * w * nomErrPosFootRtWorldEwrtWorldXy.x(), 1.0e-12);
  EXPECT_NEAR(stage.R(kLipCl, kLipDlx),
              2.0 * w * nomErrPosFootRtWorldEwrtWorldXy.x(), 1.0e-12);
  EXPECT_NEAR(stage.R(kLipDly, kLipCl), 0.0, 1.0e-12)
      << "no lateral command, no lateral nominal";
  EXPECT_NEAR(stage.R(kLipCl, kLipCl),
              2.0 * w * nomErrPosFootRtWorldEwrtWorldXy.squaredNorm(), 1.0e-12);
  EXPECT_NEAR(stage.r(kLipDlx), -2.0 * w * nomErrPosFootRtWorldEwrtWorldXy.x(),
              1.0e-12);
  EXPECT_NEAR(stage.r(kLipCl),
              -2.0 * w * nomErrPosFootRtWorldEwrtWorldXy.squaredNorm(),
              1.0e-12);
  EXPECT_NEAR(stage.R(kLipDrx, kLipCr),
              2.0 * w * nomErrPosFootRtWorldEwrtWorldXy.x(), 1.0e-12)
      << "both feet";
  EXPECT_NEAR(stage.R(kLipDlx, kLipCr), 0.0, 1.0e-12)
      << "no coupling across feet";
  EXPECT_TRUE(stage.Q.isZero()) << "the term touches no state";
  EXPECT_EQ(qp.stages.back().numInputs(), 0) << "not on the terminal node";

  // Standing still: the nominal is zero and the term is a plain regularizer of
  // the foot displacement.
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem standing,
                              planner->buildProblem(makeStandingInput()));
  EXPECT_NEAR(standing.stages.front().R(kLipDlx, kLipDlx), 2.0 * w, 1.0e-12);
  EXPECT_NEAR(standing.stages.front().R(kLipDlx, kLipCl), 0.0, 1.0e-12);
  EXPECT_NEAR(standing.stages.front().r(kLipCl), 0.0, 1.0e-12);
  EXPECT_NE(cost.describe().find("d_nom = v_cmd dt T_stride / T_swing"),
            std::string::npos);
}

// The input's world-frame command is the operator's heading-frame command
// rotated by the commanded heading at the planning instant, and the commanded
// heading turns at the commanded yaw rate over the horizon (the operator target
// rotates it knot by knot, heading_tracking ramps theta_0 + omega_cmd k dt).
// The nominal swing displacement of node k is therefore the node-0 nominal
// rotated by omega_cmd k dt; it used to stay along the initial heading at every
// node.
TEST(ContactPlanningTerms, StepLengthCostTurnsTheNominalWithTheCommandedYaw) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kStepLength};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {};
  config.stepLength.weight = 20.0;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  const StepLengthCost& cost =
      termAs<StepLengthCost>(planner->getProblem().costs, term::kStepLength);

  ContactPlannerInput input = makeStandingInput();
  input.cmdLinvelBaseRtWorldEwrtWorldXy = vector2_t(0.5, 0.0);
  input.cmdRateYawBaseRtWorld = 0.6;
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem qp,
                              planner->buildProblem(input));
  const scalar_t w = config.stepLength.weight;
  const vector2_t initNomErrPosFootRtWorldEwrtWorldXy =
      cost.computeNomErrPosFootRtWorldEwrtWorldXy(
          input.cmdLinvelBaseRtWorldEwrtWorldXy, config.planner.dt);
  for (int node = 0; node < qp.numStages(); ++node) {
    const scalar_t yaw = input.cmdRateYawBaseRtWorld * config.planner.dt * node;
    const vector2_t nomErrPosFootRtWorldEwrtWorldXy(
        std::cos(yaw) * initNomErrPosFootRtWorldEwrtWorldXy.x() -
            std::sin(yaw) * initNomErrPosFootRtWorldEwrtWorldXy.y(),
        std::sin(yaw) * initNomErrPosFootRtWorldEwrtWorldXy.x() +
            std::cos(yaw) * initNomErrPosFootRtWorldEwrtWorldXy.y());
    const OcpQpStage& stage = qp.stages[static_cast<size_t>(node)];
    EXPECT_NEAR(stage.R(kLipDlx, kLipCl),
                2.0 * w * nomErrPosFootRtWorldEwrtWorldXy.x(), 1.0e-12)
        << "node " << node;
    EXPECT_NEAR(stage.R(kLipDly, kLipCl),
                2.0 * w * nomErrPosFootRtWorldEwrtWorldXy.y(), 1.0e-12)
        << "node " << node;
    EXPECT_NEAR(stage.R(kLipDry, kLipCr),
                2.0 * w * nomErrPosFootRtWorldEwrtWorldXy.y(), 1.0e-12)
        << "node " << node;
    EXPECT_NEAR(stage.r(kLipDly),
                -2.0 * w * nomErrPosFootRtWorldEwrtWorldXy.y(), 1.0e-12)
        << "node " << node;
    EXPECT_NEAR(stage.R(kLipCl, kLipCl),
                2.0 * w * initNomErrPosFootRtWorldEwrtWorldXy.squaredNorm(),
                1.0e-12)
        << "the rotation keeps the step length";
  }
}

// terminal_dcm.trackVelocityCommand moves the target of the TERMINAL DCM from
// the last ZMP (rest) to zmp + v_cmd / omega (a CoM over the foot that keeps
// moving at the commanded velocity): only the last running node changes, and it
// evaluates to w ||xi_N - zmp_{N-1} - r||^2 with xi_N propagated exactly by the
// LIP over the last interval.
TEST(ContactPlanningTerms,
     TerminalDcmCanTrackTheCommandedVelocityInsteadOfComingToRest) {
  ContactPlanningConfig rest = makeConfig();
  rest.formulation.costs = {term::kTerminalDcm};
  rest.formulation.softConstraints = {};
  rest.formulation.hardConstraints = {};
  EXPECT_EQ(rest.validateStatus(), absl::OkStatus());
  ContactPlanningConfig walking = rest;
  walking.terminalDcm.trackVelocityCommand = true;
  LEGGED_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<LipContactPlanner> restPlanner,
      LipContactPlanner::Create(rest));
  LEGGED_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<LipContactPlanner> walkingPlanner,
      LipContactPlanner::Create(walking));
  EXPECT_FALSE(termAs<TerminalDcmCost>(restPlanner->getProblem().costs,
                                       term::kTerminalDcm)
                   .tracksVelocityCommand());
  EXPECT_TRUE(termAs<TerminalDcmCost>(walkingPlanner->getProblem().costs,
                                      term::kTerminalDcm)
                  .tracksVelocityCommand());
  EXPECT_NE(termNamed(walkingPlanner->getProblem().costs, term::kTerminalDcm)
                .describe()
                .find("v_cmd / omega"),
            std::string::npos);

  ContactPlannerInput input = makeWalkingInput();
  input.cmdLinvelBaseRtWorldEwrtWorldXy = vector2_t(0.4, -0.15);
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem qpRest,
                              restPlanner->buildProblem(input));
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem qpWalking,
                              walkingPlanner->buildProblem(input));
  // ContactPlanningProblem::assemble allocates numNodes + 1 stages: the running
  // nodes 0 ... numNodes - 1, which carry inputs and dynamics, and the terminal
  // stage numNodes, which carries neither. `last` below is therefore the index
  // of the last RUNNING node - the single node NodeSet::LAST_RUNNING selects -
  // and stages.back() is the terminal stage. The assertion pins that
  // convention, so that a change of the stage allocation cannot silently turn
  // the indices used in the rest of this test into something else.
  ASSERT_EQ(qpWalking.stages.size(),
            static_cast<size_t>(rest.planner.numNodes) + 1u);
  const size_t last = static_cast<size_t>(rest.planner.numNodes - 1);
  const scalar_t omega =
      std::sqrt(rest.shared.gravity / rest.getNomPosComRtFeetmeanEwrtWorldZ());
  const scalar_t dt = rest.planner.dt;
  const scalar_t w = rest.terminalDcm.weight;

  // The documented functional, at arbitrary states and ZMPs of the last running
  // node: the DCM xi = c + v / omega is carried to the end of the interval by
  // the LIP, xi_N = z + e^{omega dt} (xi_{N-1} - z), and compared with the
  // target.
  const absl::AnyInvocable<scalar_t(const OcpQpStage&, const vector_t&,
                                    const vector_t&) const>
      stageCost =
          [](const OcpQpStage& stage, const vector_t& x, const vector_t& u) {
            return 0.5 * x.dot(stage.Q * x) + 0.5 * u.dot(stage.R * u) +
                   u.dot(stage.S * x) + stage.q.dot(x) + stage.r.dot(u) +
                   stage.constant;
          };
  const std::vector<std::array<scalar_t, 6>> trials{
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.05, -0.02, 0.3, 0.1, 0.1, -0.04},
      {-0.2, 0.15, -0.4, 0.25, -0.1, 0.2}};
  for (const std::array<scalar_t, 6>& trial : trials) {
    vector_t x = vector_t::Zero(qpWalking.stages[last].numStates());
    vector_t u = vector_t::Zero(qpWalking.stages[last].numInputs());
    x(kLipCx) = trial[0];
    x(kLipCy) = trial[1];
    x(kLipVx) = trial[2];
    x(kLipVy) = trial[3];
    u(kLipZx) = trial[4];
    u(kLipZy) = trial[5];
    scalar_t restExpected = 0.0;
    scalar_t walkingExpected = 0.0;
    for (int axis = 0; axis < 2; ++axis) {
      const scalar_t posZmpRtWorldEwrtWorld = u(kLipZx + axis);
      const scalar_t xiNext =
          posZmpRtWorldEwrtWorld +
          std::exp(omega * dt) * (x(kLipCx + axis) + x(kLipVx + axis) / omega -
                                  posZmpRtWorldEwrtWorld);
      restExpected += w * (xiNext - posZmpRtWorldEwrtWorld) *
                      (xiNext - posZmpRtWorldEwrtWorld);
      const scalar_t walkingResidual =
          xiNext - posZmpRtWorldEwrtWorld -
          input.cmdLinvelBaseRtWorldEwrtWorldXy(axis) / omega;
      walkingExpected += w * walkingResidual * walkingResidual;
    }
    EXPECT_NEAR(stageCost(qpRest.stages[last], x, u), restExpected,
                1.0e-9 * (1.0 + restExpected))
        << "come to rest: xi_N -> zmp";
    EXPECT_NEAR(stageCost(qpWalking.stages[last], x, u), walkingExpected,
                1.0e-9 * (1.0 + walkingExpected))
        << "keep walking: xi_N -> zmp + v_cmd / omega";
  }
  // And the target is attained: at xi_N = zmp + v_cmd / omega the tracking cost
  // vanishes.
  vector_t x = vector_t::Zero(qpWalking.stages[last].numStates());
  vector_t u = vector_t::Zero(qpWalking.stages[last].numInputs());
  for (int axis = 0; axis < 2; ++axis) {
    u(kLipZx + axis) = 0.07 * (axis + 1);
    x(kLipCx + axis) = 0.02;
    // xi_{N-1} - zmp = e^{-omega dt} v_cmd / omega, so that xi_N - zmp = v_cmd
    // / omega.
    x(kLipVx + axis) =
        omega * (u(kLipZx + axis) - x(kLipCx + axis)) +
        std::exp(-omega * dt) * input.cmdLinvelBaseRtWorldEwrtWorldXy(axis);
  }
  EXPECT_NEAR(stageCost(qpWalking.stages[last], x, u), 0.0, 1.0e-9);

  // Every other stage of the horizon has to stay untouched. The bound of this
  // loop used to be `k + 1 < last`, which stopped one node too early: with the
  // twelve-node configuration of makeConfig() it inspected the nodes 0 ... 9
  // only, so node 10 - the one immediately before the last running node - was
  // never looked at, and the terminal stage was not looked at either, while the
  // failure message already claimed "only the last running node". A widening of
  // NodeSet::LAST_RUNNING that let the terminal capturability cost spill onto
  // the second-to-last running node as well (which would double the terminal
  // capturability weight and shorten the last steps of every plan) therefore
  // slipped through unnoticed. The bound is now `k < last`, and the terminal
  // stage - which has no inputs and no ZMP, so the residual of this cost is not
  // even defined there - is checked on its own afterwards.
  EXPECT_TRUE(qpRest.stages[last].q.isZero());
  EXPECT_TRUE(qpWalking.stages[last].Q.isApprox(qpRest.stages[last].Q))
      << "the target moves, the curvature does not";
  for (size_t k = 0; k < last; ++k) {
    EXPECT_TRUE(qpWalking.stages[k].q.isZero())
        << "only the last running node, k = " << k;
  }
  EXPECT_TRUE(qpWalking.stages.back().q.isZero()) << "not on the terminal node";
}

// nodeSetContains is the one place that decides, for every stage of the
// assembled QP, whether a term is invoked on it
// (ContactPlanningProblem::assemble consults it for every cost and every
// constraint). Until now it had no direct test anywhere in the repository: its
// four cases were pinned only indirectly, and only as far as the stages the
// terms of the other tests in this file happen to be written on. That is a thin
// thread to hang the semantics of the whole formulation on, so the four node
// sets are checked here against the convention they promise - N running nodes
// 0 ... N - 1 that carry inputs and dynamics, plus the terminal stage N that
// carries neither - over a complete horizon, one entry per stage, together with
// the names printed in the start-up summary of the problem.
TEST(ContactPlanningTerms, NodeSetsSelectExactlyTheStagesTheirNamesPromise) {
  constexpr int kNumNodes = 4;
  struct Expectation {
    NodeSet set;
    const char* absl_nonnull name;
    std::array<bool, 5> contains;  // one entry per stage, 0 ... kNumNodes, the
                                   // terminal stage last
  };
  const std::vector<Expectation> expectations = {
      {NodeSet::kRunning, "running nodes", {{true, true, true, true, false}}},
      {NodeSet::kTerminal,
       "terminal node",
       {{false, false, false, false, true}}},
      {NodeSet::kAll, "all nodes", {{true, true, true, true, true}}},
      {NodeSet::kLastRunning,
       "last running node",
       {{false, false, false, true, false}}},
  };
  for (const Expectation& expectation : expectations) {
    EXPECT_EQ(nodeSetName(expectation.set), expectation.name);
    for (int node = 0; node <= kNumNodes; ++node) {
      EXPECT_EQ(nodeSetContains(expectation.set, node, kNumNodes),
                expectation.contains[static_cast<size_t>(node)])
          << expectation.name << ", node " << node << " of " << kNumNodes;
    }
  }
}

// planner.logPlans prints one line per plan: search statistics, phase sequence
// with durations, step lengths.
TEST(ContactPlanningTerms, APlanDescribesItsPhasesAndSteps) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 2.0;
  plan.dt = 0.1;
  plan.objective = 1.5;
  plan.numBranchAndBoundNodes = 7;
  plan.solveTime = 0.012;
  plan.optimal = true;
  // Left: contact 0.2 s, swing 0.3 s (a 0.4 m step forward), contact 0.1 s.
  // Right: contact throughout.
  plan.contacts = {{true, true},  {true, true},  {false, true},
                   {false, true}, {false, true}, {true, true}};
  plan.desPosFeetRtWorldEwrtWorldXyTraj.assign(
      7, {vector2_t(0.0, 0.1), vector2_t(0.0, -0.1)});
  for (size_t k = 5; k < 7; ++k) {
    plan.desPosFeetRtWorldEwrtWorldXyTraj[k][0] = vector2_t(0.4, 0.1);
  }
  plan.linvelComRtWorldEwrtWorldXyTraj = {vector2_t(0.3, 0.0),
                                          vector2_t(0.5, 0.0)};
  const std::string line = plan.describe();
  EXPECT_NE(
      line.find(
          "plan t=2.000 valid J=1.500 relaxations=7 solve=12.000ms optimal"),
      std::string::npos)
      << line;
  EXPECT_NE(line.find("v0=[0.300 0.000] vN=[0.500 0.000]"), std::string::npos)
      << line;
  EXPECT_NE(line.find("| L: C0.200 S0.300(0.400,0.000) C0.100"),
            std::string::npos)
      << line;
  EXPECT_NE(line.find("| R: C0.600"), std::string::npos) << line;
  plan.valid = false;
  plan.timeLimitHit = true;
  plan.optimal = false;
  EXPECT_NE(plan.describe().find("INVALID"), std::string::npos);
  EXPECT_NE(plan.describe().find("TIME-LIMIT"), std::string::npos);
}

/*============================================ stage accumulation
 * =========================================*/
namespace {

// The functional StepLengthCost documents, w sum_i ||dp_i - d_nom (1 - c_i)||^2
// summed over the running nodes, written out from the input trajectory alone.
// It deliberately restates the formula of the term instead of reading anything
// the assembly produced, so that it is an independent statement of what the
// assembled objective has to evaluate to.
scalar_t stepLengthFunctional(scalar_t weight,
                              const vector2_t& nomErrPosFootRtWorldEwrtWorldXy,
                              absl::Span<const vector_t> inputs) {
  const std::array<std::array<int, 2>, kNumFeet> footDelta{
      {{kLipDlx, kLipDly}, {kLipDrx, kLipDry}}};
  const std::array<int, kNumFeet> contact{kLipCl, kLipCr};
  scalar_t total = 0.0;
  for (const vector_t& input : inputs) {
    for (size_t foot = 0; foot < kNumFeet; ++foot) {
      for (int axis = 0; axis < 2; ++axis) {
        const scalar_t residual =
            input(footDelta[foot][static_cast<size_t>(axis)]) -
            nomErrPosFootRtWorldEwrtWorldXy(axis) *
                (1.0 - input(contact[foot]));
        total += weight * residual * residual;
      }
    }
  }
  return total;
}

}  // namespace

// StageAccumulator::addQuadraticResidual expands w (l_x' x + l_u' u + c)^2 into
// the stage, and the expansion has three parts: the quadratic one, the linear
// one, and the constant w c^2. Only the first two used to be written, because
// the third is invisible to the QP solver - a constant moves neither the
// minimizer nor any comparison between two solutions of the SAME assembled
// problem - so the objective every consumer of the planner saw was the
// documented functional minus a per-problem constant. This pins the whole
// expansion: the stage has to evaluate to the squared residuals themselves, at
// trajectories where the residual is zero as well as at trajectories where it
// is not. The all-zero trial matters most: there the entire cost IS the dropped
// constant, and the old code reported exactly 0.
TEST(ContactPlanningTerms,
     AnAccumulatedStageEvaluatesToTheFullSquaredResidual) {
  constexpr int kNumStates = 2;
  constexpr int kNumInputs = 2;
  OcpQpProblem problem;
  problem.x0 = vector_t::Zero(kNumStates);
  problem.stages.resize(2);
  problem.stages[0] =
      OcpQpStage::Zero(kNumStates, kNumInputs, OcpQpStageKind::kIntermediate);
  problem.stages[1] =
      OcpQpStage::Zero(kNumStates, 0, OcpQpStageKind::kTerminal);

  // Two residuals on the running stage: one over both states and both inputs,
  // one over a single state. Both offsets are non-zero, which is the whole
  // point - with a zero offset the assembled stage is right either way.
  const scalar_t firstWeight = 3.0;
  const scalar_t firstOffset = 0.75;
  const scalar_t secondWeight = 0.5;
  const scalar_t secondOffset = -1.25;
  StageAccumulator accumulator(&problem.stages[0]);
  accumulator.addQuadraticResidual({{0, 1.5}, {1, -0.5}}, {{0, 2.0}, {1, 0.25}},
                                   firstOffset, firstWeight);
  accumulator.addQuadraticResidual({{1, 1.0}}, /*uCoefficients=*/{},
                                   secondOffset, secondWeight);

  const vector_t terminalState = vector_t::Zero(kNumStates);
  const std::array<std::array<scalar_t, 4>, 4> trials{
      {{0.0, 0.0, 0.0, 0.0},
       {0.3, -0.7, 0.9, -0.2},
       {-1.1, 2.0, -0.4, 1.3},
       {2.5, 0.125, -1.75, 0.0}}};
  for (const std::array<scalar_t, 4>& trial : trials) {
    vector_t state = vector_t::Zero(kNumStates);
    state << trial[0], trial[1];
    vector_t input = vector_t::Zero(kNumInputs);
    input << trial[2], trial[3];
    const scalar_t firstResidual = 1.5 * state(0) - 0.5 * state(1) +
                                   2.0 * input(0) + 0.25 * input(1) +
                                   firstOffset;
    const scalar_t secondResidual = state(1) + secondOffset;
    const scalar_t expected = firstWeight * firstResidual * firstResidual +
                              secondWeight * secondResidual * secondResidual;
    const std::vector<vector_t> stateTrajectory = {state, terminalState};
    const std::vector<vector_t> inputTrajectory = {input};
    EXPECT_NEAR(
        evaluateOcpQpObjective(problem, stateTrajectory, inputTrajectory),
        expected, 1.0e-12)
        << "x = " << state.transpose() << ", u = " << input.transpose()
        << ": the stage must evaluate to the squared residuals, the constant "
           "part of the expansion included";
  }
}

// The reason the dropped constant is not a cosmetic issue:
// CadenceStretchStage::afterSearch re-assembles the problem on a stretched node
// grid (s * dt) and scores the result against the incumbent assembled at dt,
// and the residual offsets of the shipped terms depend on dt - StepLengthCost's
// nominal displacement is v_cmd dt T_stride / T_swing. The constant therefore
// differs between the two problems, and while it was dropped, the stretched
// candidate's reported objective was depressed relative to the incumbent's by
// an amount that grows with the stretch, so the stage kept accepting stretches
// whose true cost was higher. This assembles the same single-term formulation
// on two grids and demands that both report their own documented functional, on
// an arbitrary trajectory and on the zero trajectory, where the entire cost is
// the constant and the old code called the two grids equally good at 0.
TEST(ContactPlanningTerms,
     ObjectivesOfProblemsAssembledOnDifferentNodeGridsAreComparable) {
  constexpr scalar_t kStretch = 1.25;
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kStepLength};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {};
  config.stepLength.weight = 20.0;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  ContactPlanningConfig stretchedConfig = config;
  stretchedConfig.planner.dt = kStretch * config.planner.dt;
  EXPECT_EQ(stretchedConfig.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  LEGGED_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<LipContactPlanner> stretchedPlanner,
      LipContactPlanner::Create(stretchedConfig));

  ContactPlannerInput input = makeStandingInput();
  input.cmdLinvelBaseRtWorldEwrtWorldXy = vector2_t(
      0.4,
      -0.25);  // both axes, so that neither of the two offsets per foot is zero

  const scalar_t weight = config.stepLength.weight;
  const scalar_t strideToSwingRatio =
      8.0 / 3.0;  // 2 (minSwingDuration + minDoubleSupportDuration) /
                  // minSwingDuration
  const vector2_t nomErrPosFootRtWorldEwrtWorldXy =
      input.cmdLinvelBaseRtWorldEwrtWorldXy *
      (config.planner.dt * strideToSwingRatio);
  const vector2_t stretchedNomErrPosFootRtWorldEwrtWorldXy =
      input.cmdLinvelBaseRtWorldEwrtWorldXy *
      (stretchedConfig.planner.dt * strideToSwingRatio);

  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem qp,
                              planner->buildProblem(input));
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem stretchedQp,
                              stretchedPlanner->buildProblem(input));
  // Anchor d_nom to what was actually assembled, so that a change of the
  // nominal displacement shows up as a failure here rather than silently moving
  // both sides of the comparisons below.
  ASSERT_NEAR(qp.stages.front().R(kLipDlx, kLipCl),
              2.0 * weight * nomErrPosFootRtWorldEwrtWorldXy.x(), 1.0e-12);
  ASSERT_NEAR(stretchedQp.stages.front().R(kLipDry, kLipCr),
              2.0 * weight * stretchedNomErrPosFootRtWorldEwrtWorldXy.y(),
              1.0e-12);
  ASSERT_EQ(stretchedQp.stages.size(), qp.stages.size())
      << "the stretch changes the node duration, not the node count";

  const int numStates = qp.stages.front().numStates();
  const int numInputs = qp.stages.front().numInputs();
  const int numRunningNodes = qp.numStages();
  std::vector<vector_t> stateTrajectory;
  std::vector<vector_t> inputTrajectory;
  for (int k = 0; k <= numRunningNodes; ++k) {
    vector_t stateAtNode = vector_t::Zero(numStates);
    for (int i = 0; i < numStates; ++i) {
      stateAtNode(i) =
          0.2 * static_cast<scalar_t>(i) - 0.03 * static_cast<scalar_t>(k);
    }
    stateTrajectory.push_back(stateAtNode);
    if (k == numRunningNodes) break;  // the terminal stage carries no inputs
    vector_t inputAtNode = vector_t::Zero(numInputs);
    for (int i = 0; i < numInputs; ++i) {
      inputAtNode(i) =
          0.05 * static_cast<scalar_t>(i + 1) - 0.01 * static_cast<scalar_t>(k);
    }
    inputTrajectory.push_back(inputAtNode);
  }

  const scalar_t nominalOnTrajectory =
      evaluateOcpQpObjective(qp, stateTrajectory, inputTrajectory);
  const scalar_t stretchedOnTrajectory =
      evaluateOcpQpObjective(stretchedQp, stateTrajectory, inputTrajectory);
  EXPECT_NEAR(nominalOnTrajectory,
              stepLengthFunctional(weight, nomErrPosFootRtWorldEwrtWorldXy,
                                   inputTrajectory),
              1.0e-9);
  EXPECT_NEAR(
      stretchedOnTrajectory,
      stepLengthFunctional(weight, stretchedNomErrPosFootRtWorldEwrtWorldXy,
                           inputTrajectory),
      1.0e-9);
  // The two problems are not merely shifted by a constant each: on this
  // trajectory the dropped constants reversed the ORDER of the two grids. The
  // stretched problem is the genuinely worse of the two here - 35.50
  // against 33.90 in the functional the terms document - while an objective
  // that leaves the constants out reports 23.63 against 26.30 and makes the
  // stretch look like an improvement of 2.7. That is exactly the comparison
  // CadenceStretchStage makes, so the ordering, and not only the two values, is
  // part of what this pins.
  EXPECT_GT(stretchedOnTrajectory, nominalOnTrajectory)
      << "the grid whose documented cost is the higher one must report the "
         "higher objective";

  // Feet that do not move while no foot is in contact: every residual is
  // exactly -d_nom, so the entire cost of each problem is its own dropped
  // constant and the two grids differ by the factor s^2 of the stretched
  // nominal step.
  const vector_t zeroState = vector_t::Zero(numStates);
  const vector_t zeroInput = vector_t::Zero(numInputs);
  const std::vector<vector_t> zeroStates(qp.stages.size(), zeroState);
  const std::vector<vector_t> zeroInputs(static_cast<size_t>(numRunningNodes),
                                         zeroInput);
  const scalar_t expectedNominal =
      stepLengthFunctional(weight, nomErrPosFootRtWorldEwrtWorldXy, zeroInputs);
  const scalar_t nominalObjective =
      evaluateOcpQpObjective(qp, zeroStates, zeroInputs);
  const scalar_t stretchedObjective =
      evaluateOcpQpObjective(stretchedQp, zeroStates, zeroInputs);
  ASSERT_GT(expectedNominal, 1.0) << "the trial is only meaningful while the "
                                     "dropped constant is a large number";
  EXPECT_NEAR(nominalObjective, expectedNominal, 1.0e-9);
  EXPECT_NEAR(stretchedObjective, kStretch * kStretch * expectedNominal,
              1.0e-9);
  EXPECT_NEAR(stretchedObjective - nominalObjective,
              (kStretch * kStretch - 1.0) * expectedNominal, 1.0e-9)
      << "the stretched grid is strictly worse here; reporting a tie is what "
         "let the cadence stretch overshoot";
}

/*============================================ required blocks and names
 * ==================================*/

TEST(ContactPlanningTerms, AHeadingTermWithoutTheHeadingBlockIsRejected) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs.push_back(term::kHeadingTracking);
  EXPECT_EQ(config.validateStatus().code(), absl::StatusCode::kInvalidArgument);
  // The problem checks it too, for a problem assembled by hand.
  ContactPlanningProblem problem;
  ASSERT_EQ(
      problem.model.add(term::kLipCom,
                        checkedValue(ContactPlanningTermFactory::makeModelBlock(
                            term::kLipCom))),
      absl::OkStatus());
  ASSERT_EQ(
      problem.model.add(term::kFootholdIntegrator,
                        checkedValue(ContactPlanningTermFactory::makeModelBlock(
                            term::kFootholdIntegrator))),
      absl::OkStatus());
  ASSERT_EQ(problem.costs.add(term::kHeadingTracking,
                              checkedValue(ContactPlanningTermFactory::makeCost(
                                  term::kHeadingTracking))),
            absl::OkStatus());
  const absl::Status finalized = problem.finalize(makeConfig());
  EXPECT_EQ(finalized.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(finalized.message().find(term::kHeadingDoubleIntegrator),
            std::string::npos)
      << finalized;
  EXPECT_FALSE(problem.isFinalized());
}

// A problem is used only once finalize() has succeeded: assemble() and
// configure() before it are a FailedPrecondition, and so is everything after a
// finalize() that failed.
TEST(ContactPlanningTerms,
     AProblemThatIsNotFinalizedRefusesToAssembleOrConfigure) {
  const ContactPlanningConfig config = makeConfig();
  ContactPlanningProblem problem;
  EXPECT_EQ(problem.configure(config).code(),
            absl::StatusCode::kFailedPrecondition);
  const absl::StatusOr<OcpQpProblem> assembled =
      problem.assemble(ContactPlanningContext());
  ASSERT_FALSE(assembled.ok());
  EXPECT_EQ(assembled.status().code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_NE(assembled.status().message().find("before finalize()"),
            std::string::npos)
      << assembled.status();

  // The model has to start with the LIP block and the foothold block, in that
  // order.
  ContactPlanningProblem reversed;
  ASSERT_EQ(reversed.model.add(
                term::kFootholdIntegrator,
                checkedValue(ContactPlanningTermFactory::makeModelBlock(
                    term::kFootholdIntegrator))),
            absl::OkStatus());
  ASSERT_EQ(reversed.model.add(
                term::kLipCom,
                checkedValue(
                    ContactPlanningTermFactory::makeModelBlock(term::kLipCom))),
            absl::OkStatus());
  const absl::Status finalized = reversed.finalize(config);
  EXPECT_EQ(finalized.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(finalized.message().find("must start with"), std::string::npos)
      << finalized;
  EXPECT_EQ(reversed.configure(config).code(),
            absl::StatusCode::kFailedPrecondition);
}

// The layout's lookups and the blocks' positions are invariants the problem
// establishes before any term reads them, so a violation is a programming error
// that stops the process with the name, rather than a status to propagate.
TEST(ContactPlanningTermsDeathTest, TheLayoutInvariantsCheckFailWithTheName) {
  LayoutBuilder builder;
  builder.addState(var::kPosComRtWorldEwrtWorldX);
  EXPECT_DEATH(builder.addState(var::kPosComRtWorldEwrtWorldX),
               "state 'c_x' declared twice");
  builder.addInput(var::kPosZmpRtWorldEwrtWorldX);
  EXPECT_DEATH(builder.addInput(var::kPosZmpRtWorldEwrtWorldX),
               "input 'zmp_x' declared twice");
  const Layout layout = builder.build();
  EXPECT_DEATH(static_cast<void>(layout.state("no_such_state")),
               "no state named 'no_such_state'");
  EXPECT_DEATH(static_cast<void>(layout.input("no_such_input")),
               "no input named 'no_such_input'");

  // The foothold block first: the LIP block is then not where the fixed indices
  // say it is.
  LayoutBuilder footFirst;
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LipModelBlock> foothold,
      ContactPlanningTermFactory::makeModelBlock(term::kFootholdIntegrator));
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LipModelBlock> lip,
      ContactPlanningTermFactory::makeModelBlock(term::kLipCom));
  foothold->declareVariables(footFirst);
  lip->declareVariables(footFirst);
  const Layout misordered = footFirst.build();
  EXPECT_DEATH(lip->bind(misordered), "the LIP block must be the first block");
  EXPECT_DEATH(foothold->bind(misordered),
               "the foothold block must follow the LIP block");
}

// The ordered, named collection every term list is held in: lookups by name
// agree with the insertion order before and after an erase (which has to
// re-index everything behind the erased term), and the three misuses are
// rejected with the term's name in the message.
TEST(ContactPlanningTerms, ATermCollectionKeepsItsOrderAndItsNameIndexInStep) {
  TermCollection<LipCost> costs;
  const std::vector<std::string> names{term::kRegularization,
                                       term::kVelocityTracking,
                                       term::kStepWidth, term::kTerminalDcm};
  for (const std::string& name : names) {
    ASSERT_EQ(costs.add(name, checkedValue(
                                  ContactPlanningTermFactory::makeCost(name))),
              absl::OkStatus())
        << name;
  }
  ASSERT_EQ(costs.size(), names.size());
  EXPECT_EQ(costs.names(), names);

  // Erasing from the middle shifts every later term down by one; the name index
  // has to follow.
  EXPECT_TRUE(costs.erase(term::kVelocityTracking));
  EXPECT_FALSE(costs.erase(term::kVelocityTracking))
      << "a second erase finds nothing";
  const std::vector<std::string> remaining{
      term::kRegularization, term::kStepWidth, term::kTerminalDcm};
  EXPECT_EQ(costs.names(), remaining);
  EXPECT_FALSE(costs.has(term::kVelocityTracking));
  for (size_t i = 0; i < remaining.size(); ++i) {
    ASSERT_TRUE(costs.has(remaining[i]));
    EXPECT_EQ(costs.find(remaining[i]), &costs.termAt(i))
        << remaining[i] << " must still resolve to its own term";
    EXPECT_EQ(costs.getTermNameMap().at(remaining[i]), i);
  }
  EXPECT_NE(
      dynamic_cast<const TerminalDcmCost*>(costs.find(term::kTerminalDcm)),
      nullptr);

  // A duplicate and a null term are rejected, naming the term, and a missing
  // name resolves to nothing.
  const absl::Status duplicate = costs.add(
      term::kStepWidth,
      checkedValue(ContactPlanningTermFactory::makeCost(term::kStepWidth)));
  EXPECT_EQ(duplicate.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(duplicate.message().find(term::kStepWidth), std::string::npos)
      << duplicate;
  const absl::Status null = costs.add("null_cost", /*term=*/nullptr);
  EXPECT_EQ(null.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(null.message().find("null_cost"), std::string::npos) << null;
  EXPECT_EQ(costs.find(term::kVelocityTracking), nullptr)
      << "an erased name must not resolve";
  EXPECT_EQ(costs.size(), remaining.size()) << "a rejected add changes nothing";
}

// The three builders return a rejection as a Status that names the term;
// repaired, the same lists build.
TEST(ContactPlanningTerms,
     TheBuildersReturnARejectionAsAStatusThatNamesTheTerm) {
  // A heading cost without the heading block.
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs.push_back(term::kHeadingTracking);
  absl::StatusOr<ContactPlanningProblem> problem =
      absl::InternalError("not built");
  problem = ContactPlanningTermFactory::buildProblemStatus(config);
  ASSERT_FALSE(problem.ok());
  EXPECT_EQ(problem.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(
      std::string(problem.status().message()).find(term::kHeadingTracking),
      std::string::npos)
      << problem.status().message();

  // An unknown search stage (the supported ones are listed), a repeated one,
  // and one without the block it needs.
  config = makeConfig();
  config.formulation.search = {"simulated_annealing"};
  absl::StatusOr<TermCollection<SearchStage>> stages =
      absl::InternalError("not built");
  stages = ContactPlanningTermFactory::buildSearchStagesStatus(config);
  ASSERT_FALSE(stages.ok());
  EXPECT_NE(std::string(stages.status().message()).find("simulated_annealing"),
            std::string::npos)
      << stages.status().message();
  EXPECT_NE(std::string(stages.status().message()).find(term::kDiving),
            std::string::npos)
      << stages.status().message();
  config.formulation.search = {term::kDiving, term::kDiving};
  stages = ContactPlanningTermFactory::buildSearchStagesStatus(config);
  ASSERT_FALSE(stages.ok());
  EXPECT_NE(std::string(stages.status().message()).find(term::kDiving),
            std::string::npos)
      << stages.status().message();
  config.formulation.search = {term::kHeadingRelinearization};
  stages = ContactPlanningTermFactory::buildSearchStagesStatus(config);
  ASSERT_FALSE(stages.ok());
  EXPECT_NE(std::string(stages.status().message())
                .find(term::kHeadingDoubleIntegrator),
            std::string::npos)
      << stages.status().message();

  // An execution rule only the reference manager can build (it needs the robot
  // model).
  config = makeHeadingConfig();
  config.formulation.execution = {term::kPlannedHeadingOverride};
  absl::StatusOr<TermCollection<ExecutionRule>> rules =
      absl::InternalError("not built");
  rules = ContactPlanningTermFactory::buildExecutionRulesStatus(config);
  ASSERT_FALSE(rules.ok());
  EXPECT_NE(
      std::string(rules.status().message()).find(term::kPlannedHeadingOverride),
      std::string::npos)
      << rules.status().message();

  // Positive controls: repaired, each builds.
  config.formulation.execution = {term::kPhaseResetting,
                                  term::kDcmStepAdjustment};
  rules = ContactPlanningTermFactory::buildExecutionRulesStatus(config);
  ASSERT_TRUE(rules.ok()) << rules.status().message();
  EXPECT_EQ(rules->names(),
            std::vector<std::string>(
                {term::kPhaseResetting, term::kDcmStepAdjustment}));
  config.formulation.search = {term::kDiving, term::kHeadingRelinearization};
  stages = ContactPlanningTermFactory::buildSearchStagesStatus(config);
  ASSERT_TRUE(stages.ok()) << stages.status().message();
  EXPECT_EQ(
      stages->names(),
      std::vector<std::string>({term::kDiving, term::kHeadingRelinearization}));
  problem = ContactPlanningTermFactory::buildProblemStatus(config);
  ASSERT_TRUE(problem.ok()) << problem.status().message();
  EXPECT_TRUE(problem->costs.has(term::kVelocityTracking))
      << problem->summary();
}

TEST(ContactPlanningTerms, TheFactoryNamesTheSupportedTermsInItsErrors) {
  const absl::StatusOr<std::unique_ptr<LipCost>> unknown =
      ContactPlanningTermFactory::makeCost("gravity_compensation");
  ASSERT_FALSE(unknown.ok());
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(unknown.status().message().find("gravity_compensation"),
            std::string::npos)
      << unknown.status();
  EXPECT_NE(unknown.status().message().find(term::kVelocityTracking),
            std::string::npos)
      << "the supported names are listed";
  // Every kind rejects a name of another kind the same way.
  EXPECT_EQ(ContactPlanningTermFactory::makeModelBlock(term::kVelocityTracking)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ContactPlanningTermFactory::makeSoftConstraint(term::kNoFlight)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ContactPlanningTermFactory::makeHardConstraint(term::kReachability)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ContactPlanningTermFactory::makeLogicRule(term::kContactSwitch)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ContactPlanningTermFactory::makeAssignmentCost(term::kNoFlight)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(ContactPlanningTermFactory::makeSearchStage(term::kPhaseResetting)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  const absl::StatusOr<std::unique_ptr<ExecutionRule>> override =
      ContactPlanningTermFactory::makeExecutionRule(
          term::kPlannedHeadingOverride);
  ASSERT_FALSE(override.ok())
      << "the heading override needs the reference manager's model";
  EXPECT_NE(override.status().message().find(term::kPlannedHeadingOverride),
            std::string::npos)
      << override.status();
  EXPECT_TRUE(
      ContactPlanningTermFactory::makeExecutionRule("phaseResetting").ok());
}

TEST(ContactPlanningTerms, EveryKnownTermCanBeBuiltAndDescribed) {
  const ContactPlanningConfig config = makeHeadingConfig();
  for (const std::string& name : knownTermNames(TermKind::kCost)) {
    LEGGED_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LipCost> cost,
                                ContactPlanningTermFactory::makeCost(name));
    cost->configure(config);
    EXPECT_FALSE(cost->describe().empty()) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::kSoftConstraint)) {
    LEGGED_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LipConstraint> constraint,
        ContactPlanningTermFactory::makeSoftConstraint(name));
    constraint->configure(config);
    EXPECT_EQ(constraint->softness(), Softness::kSoft) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::kHardConstraint)) {
    LEGGED_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<LipConstraint> constraint,
        ContactPlanningTermFactory::makeHardConstraint(name));
    constraint->configure(config);
    EXPECT_EQ(constraint->softness(), Softness::kHard) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::kLogicRule)) {
    EXPECT_TRUE(ContactPlanningTermFactory::makeLogicRule(name).ok()) << name;
  }
  for (const std::string& name : knownTermNames(TermKind::kAssignmentCost)) {
    EXPECT_TRUE(ContactPlanningTermFactory::makeAssignmentCost(name).ok())
        << name;
  }
  for (const std::string& name : knownTermNames(TermKind::kSearchStage)) {
    EXPECT_TRUE(ContactPlanningTermFactory::makeSearchStage(name).ok()) << name;
  }
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::string summary,
                              LipContactPlanner::formulationSummary(config));
  for (const std::string& name : config.formulation.costs) {
    EXPECT_NE(summary.find(name), std::string::npos) << summary;
  }
  EXPECT_NE(summary.find("x = [c_x c_y v_x v_y p_Lx p_Ly p_Rx p_Ry theta omega "
                         "psi_L psi_R] (12)"),
            std::string::npos)
      << summary;
}

/*============================================ hot reload
 * =================================================*/

TEST(ContactPlanningTerms, ParametersReloadByTermNameWithoutReassembly) {
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(makeConfig()));
  const ContactPlannerInput input = makeWalkingInput();
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem before,
                              planner->buildProblem(input));
  ContactPlanningConfig retuned = planner->getConfig();
  retuned.velocityTracking.weight *= 3.0;
  retuned.zmpSupportRegion.slack =
      SlackPenalty{.quadratic = 123.0, .linear = 4.0};
  ASSERT_EQ(planner->setConfig(retuned), absl::OkStatus());
  EXPECT_NEAR(termAs<VelocityTrackingCost>(planner->getProblem().costs,
                                           term::kVelocityTracking)
                  .weight(),
              retuned.velocityTracking.weight, 1.0e-12);
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem after,
                              planner->buildProblem(input));
  EXPECT_NEAR(after.stages.front().Q(kLipVx, kLipVx),
              3.0 * before.stages.front().Q(kLipVx, kLipVx) -
                  2.0 * retuned.regularization.state,
              1.0e-9);
  // The ZMP rows are the first 12 soft rows of a running node; their penalty is
  // the term's own, the rest keep the shared one.
  EXPECT_NEAR(after.stages.front().Zl(0), 123.0, 1.0e-12);
  EXPECT_NEAR(after.stages.front().zl(0), 4.0, 1.0e-12);
  EXPECT_NEAR(after.stages.front().Zl(12),
              retuned.shared.slackPenalty.quadratic, 1.0e-12);
  EXPECT_NEAR(before.stages.front().Zl(0),
              planner->getConfig().shared.slackPenalty.quadratic, 1.0e-12);
}

TEST(
    ContactPlanningTerms,
    ChangingATermListReassemblesTheProblemAndKeepsTheWarmStartWhileTheLayoutHolds) {
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(makeConfig()));
  const ContactPlannerInput input = makeWalkingInput();
  const ContactPlan first = planner->plan(input);
  ASSERT_TRUE(first.valid);
  ContactPlannerInput next = input;
  next.time += 0.1;
  next.contacts = first.contactsAtTime(next.time);
  ASSERT_GE(
      planner->defaultNominal(next).nomPosComRtWorldEwrtWorldXyTraj.size(), 1u);
  const bool warmBefore =
      checkedValue(planner->makeContext(next, planner->defaultNominal(next)))
          .previousPlanShift.has_value();
  EXPECT_TRUE(warmBefore);

  ContactPlanningConfig fewerCosts = planner->getConfig();
  fewerCosts.formulation.setCost(term::kStepWidth, /*on=*/false);
  ASSERT_EQ(planner->setConfig(fewerCosts), absl::OkStatus());
  EXPECT_FALSE(planner->getProblem().costs.has(term::kStepWidth));
  EXPECT_TRUE(
      checkedValue(planner->makeContext(next, planner->defaultNominal(next)))
          .previousPlanShift.has_value())
      << "same grid and layout: the warm start survives";

  ContactPlanningConfig heading = makeHeadingConfig();
  ASSERT_EQ(planner->setConfig(heading), absl::OkStatus());
  EXPECT_TRUE(planner->getLayout().hasHeading);
  EXPECT_FALSE(
      checkedValue(planner->makeContext(next, planner->defaultNominal(next)))
          .previousPlanShift.has_value())
      << "a new layout drops the warm start";
}

TEST(ContactPlanningTerms, SearchStagesFollowTheList) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.search = {term::kWarmStartPreviousPlan};
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  EXPECT_EQ(planner->getSearchStages().size(), 1u);
  const ContactPlan plan = planner->plan(makeWalkingInput());
  EXPECT_TRUE(plan.valid);
  EXPECT_EQ(planner->getLastStatistics().numLocalSearchQps, 0)
      << "no local search stage";
  config.formulation.search = {term::kHeadingRelinearization};
  const absl::StatusOr<std::unique_ptr<LipContactPlanner>> refused =
      LipContactPlanner::Create(config);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument)
      << "a stage that needs the heading block";
}

/*============================================ execution rules
 * ============================================*/

TEST(ContactPlanningTerms,
     ExecutionRulesAreBuiltFromTheListAndReportWhatTheyNeed) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.execution = {term::kPhaseResetting,
                                  term::kEnergyCadenceModulation,
                                  term::kDcmStepAdjustment};
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  absl::StatusOr<TermCollection<ExecutionRule>> built =
      ContactPlanningTermFactory::buildExecutionRulesStatus(config);
  ASSERT_TRUE(built.ok()) << built.status();
  const TermCollection<ExecutionRule>& rules = *built;
  ASSERT_EQ(rules.size(), 3u);
  EXPECT_FALSE(
      termNamed(rules, term::kPhaseResetting).needsPredictedTrajectory());
  EXPECT_TRUE(termNamed(rules, term::kEnergyCadenceModulation)
                  .needsPredictedTrajectory());
  EXPECT_TRUE(
      termNamed(rules, term::kDcmStepAdjustment).needsPredictedTrajectory());
  config.formulation.execution.push_back(term::kPlannedHeadingOverride);
  EXPECT_FALSE(
      ContactPlanningTermFactory::buildExecutionRulesStatus(config).ok())
      << "needs the heading block and the model";
}

TEST(ContactPlanningTerms,
     WithoutRulesTheScheduleIsLeftAloneAndTheLatchTracksTheSwing) {
  // A swing of the left foot from 1.0 to 1.4 s, the right foot down throughout.
  ModeSchedule schedule({1.0, 1.4},
                        {kBipedModeStance, kBipedModeRf, kBipedModeStance});
  BipedFeetArray<SwingTimingLatch> latches =
      makeBipedFeetArray(SwingTimingLatch{});
  const ContactPlanningConfig config = makeConfig();
  const BipedFeetArray<scalar_t> noCadence = makeBipedFeetArray(0.0);
  BipedFeetArray<bool> measured = {
      true, true};  // the left foot reports contact mid-swing
  const BipedFeetArray<ContactEventReport> reports =
      adaptScheduleToContactEvents(schedule, /*time=*/1.2, measured, noCadence,
                                   config, latches);
  EXPECT_EQ(reports[0].type, ContactEventReport::Type::kNone)
      << "no phase_resetting rule listed";
  EXPECT_TRUE(latches[0].active);
  EXPECT_NEAR(latches[0].nominalTouchDownTime, 1.4, 1.0e-12);
  EXPECT_NEAR(schedule.eventTimes[1], 1.4, 1.0e-12);
  measured = {false, true};  // the foot is still in the air at its scheduled
                             // touch-down: nothing extends it either
  const BipedFeetArray<ContactEventReport> late = adaptScheduleToContactEvents(
      schedule, /*time=*/1.45, measured, noCadence, config, latches);
  EXPECT_EQ(late[0].type, ContactEventReport::Type::kNone);
  EXPECT_FALSE(latches[0].active)
      << "the contact phase proceeds and the latch is released";
}

/** The double support penalty, on a hand-built assignment of three nodes.
 * Binaries are node-major: 2 * node + foot. */
TEST(ContactPlanningTerms,
     DoubleSupportPenaltyChargesEveryNodeWithBothFeetDown) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.assignmentCosts = {term::kDoubleSupportPenalty};
  config.doubleSupportPenalty.cost = 5.0;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  const DoubleSupportPenaltyCost& cost = termAs<DoubleSupportPenaltyCost>(
      planner->getProblem().assignmentCosts, term::kDoubleSupportPenalty);

  ContactLogicState state;
  state.numNodes = 3;
  MiqpAssignment a(3 * kNumFeet);
  const absl::AnyInvocable<void(int, int8_t, int8_t) const> set =
      [&a](int node, int8_t left, int8_t right) {
        a[static_cast<size_t>(
            ContactLogicState::contactBinaryIndex(node, /*foot=*/0))] = left;
        a[static_cast<size_t>(
            ContactLogicState::contactBinaryIndex(node, /*foot=*/1))] = right;
      };

  set(0, 0, 1);  // left swinging, right in contact
  set(1, 1, 1);  // double support
  set(2, 1, 0);  // left in contact, right swinging
  EXPECT_NEAR(cost.cost(state, a), 5.0, 1.0e-12)
      << "one double-support node out of three";

  set(2, 1, 1);
  EXPECT_NEAR(cost.cost(state, a), 10.0, 1.0e-12) << "two double-support nodes";

  set(0, 1, 1);
  set(1, 1, 1);
  set(2, 1, 1);
  EXPECT_NEAR(cost.cost(state, a), 15.0, 1.0e-12)
      << "standing costs the whole horizon";

  config.doubleSupportPenalty.cost = 0.0;
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> disabled,
                              LipContactPlanner::Create(config));
  EXPECT_NEAR(
      termAs<DoubleSupportPenaltyCost>(disabled->getProblem().assignmentCosts,
                                       term::kDoubleSupportPenalty)
          .cost(state, a),
      0.0, 1.0e-12)
      << "a zero cost is free whatever the assignment";
}

/**
 * The branch-and-bound prunes on the assignment cost of a *partial* assignment,
 * so the term has to be a lower bound there and exact only once every binary is
 * decided. A free node may still become a double support, so charging it would
 * overestimate the partial assignment and could discard the optimum.
 */
TEST(ContactPlanningTerms,
     DoubleSupportPenaltyIsALowerBoundOnAPartialAssignment) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.assignmentCosts = {term::kDoubleSupportPenalty};
  config.doubleSupportPenalty.cost = 5.0;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  const DoubleSupportPenaltyCost& cost = termAs<DoubleSupportPenaltyCost>(
      planner->getProblem().assignmentCosts, term::kDoubleSupportPenalty);

  ContactLogicState state;
  state.numNodes = 3;
  MiqpAssignment partial(3 * kNumFeet, kMiqpFree);
  EXPECT_NEAR(cost.cost(state, partial), 0.0, 1.0e-12)
      << "nothing decided, nothing charged";

  // One foot down at node 0, the other still free: not yet a double support.
  partial[static_cast<size_t>(ContactLogicState::contactBinaryIndex(0, 0))] = 1;
  EXPECT_NEAR(cost.cost(state, partial), 0.0, 1.0e-12)
      << "a half-decided node is not charged";

  // Deciding the second foot completes the double support and the cost may only
  // rise.
  partial[static_cast<size_t>(ContactLogicState::contactBinaryIndex(0, 1))] = 1;
  EXPECT_NEAR(cost.cost(state, partial), 5.0, 1.0e-12);

  // Completing the rest of the horizon can only add to it, never take away.
  MiqpAssignment complete = partial;
  for (int k = 1; k < state.numNodes; ++k) {
    complete[static_cast<size_t>(
        ContactLogicState::contactBinaryIndex(k, /*foot=*/0))] = 1;
    complete[static_cast<size_t>(
        ContactLogicState::contactBinaryIndex(k, /*foot=*/1))] = 0;
  }
  EXPECT_GE(cost.cost(state, complete), cost.cost(state, partial))
      << "the bound must not decrease when a node is decided";
}

/**
 * In a zero-double-support gait the exchange is a single event: one foot lands
 * at the very instant the other lifts. Ending a swing early holds the landing
 * foot down from the measured contact until its scheduled touch-down, so an
 * early touch-down there opens a double support exactly as long as the foot was
 * early. Since a landing within a few milliseconds of the plan is the common
 * case, that put a sliver of double support - shorter than the MPC's own time
 * step - on nearly every step, at both ends of every swing (each exchange is
 * one foot's touch-down and the other's lift-off). earlyTouchdownMinAdvance
 * executes a near-on-time touch-down as planned instead.
 */
TEST(ContactPlanningTerms, CadenceStretchIsBoundedByTheGaitLimits) {
  // The stretch re-times every phase of the incumbent together, so the phase
  // that is closest to its limit is what bounds it. A five-node swing at dt 0.1
  // is 0.5 s, already at maxSwingDuration, so nothing may be stretched; a
  // four-node swing is 0.4 s and leaves room for exactly 0.5 / 0.4.
  ContactPlanningConfig config = makeConfig();
  config.planner.numNodes = 12;
  config.cadenceStretch.samples = 4;
  config.cadenceStretch.maxStretch = 2.0;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());

  const absl::AnyInvocable<MiqpAssignment(int) const> assignmentWithSwing =
      [&config](int swingNodes) {
        MiqpAssignment assignment(
            static_cast<size_t>(ContactLogicState::kBinariesPerNode *
                                config.planner.numNodes),
            // NOLINTNEXTLINE(argument-comment): std::vector fill constructor, whose libstdc++ parameters are reserved names (__n, __value)
            1);
        for (int node = 0; node < swingNodes; ++node) {
          assignment[static_cast<size_t>(ContactLogicState::contactBinaryIndex(
              node, kContactLeftIndex))] = 0;
        }
        return assignment;
      };

  EXPECT_NEAR(CadenceStretchStage::admissibleStretch(
                  config, assignmentWithSwing(4), config.planner.numNodes,
                  /*maxStretch=*/2.0),
              0.5 / 0.4, 1.0e-9);
  EXPECT_NEAR(CadenceStretchStage::admissibleStretch(
                  config, assignmentWithSwing(5), config.planner.numNodes,
                  /*maxStretch=*/2.0),
              1.0, 1.0e-9);
  // maxStretch is the other bound, and it is the binding one for a short swing.
  EXPECT_NEAR(CadenceStretchStage::admissibleStretch(
                  config, assignmentWithSwing(3), config.planner.numNodes,
                  /*maxStretch=*/1.2),
              1.2, 1.0e-9);
  // A swing that already exceeds the limit cannot be repaired by stretching,
  // and must not push the bound below 1.
  EXPECT_NEAR(CadenceStretchStage::admissibleStretch(
                  config, assignmentWithSwing(8), config.planner.numNodes,
                  /*maxStretch=*/2.0),
              1.0, 1.0e-9);
}

TEST(ContactPlanningTerms, CadenceStretchIsDisabledWithoutSamples) {
  ContactPlanningConfig config = makeConfig();
  config.cadenceStretch.samples = 0;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SearchStage> stage,
      ContactPlanningTermFactory::makeSearchStage(term::kCadenceStretch));
  ASSERT_NE(stage, nullptr);
  stage->configure(config);
  EXPECT_NE(stage->describe().find("0 stretch"), std::string::npos);
}

TEST(ContactPlanningTerms,
     ANearlyOnTimeTouchDownDoesNotSplitAZeroDoubleSupportExchange) {
  // Left swings 1.0 -> 1.4 and lands at the instant the right lifts; right then
  // swings 1.4 -> 1.8.
  const ModeSchedule nominal({1.0, 1.4, 1.8}, {kBipedModeStance, kBipedModeRf,
                                               kBipedModeLf, kBipedModeStance});
  ContactPlanningConfig config = makeConfig();
  config.formulation.execution = {term::kPhaseResetting};
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  ASSERT_GT(config.phaseResetting.earlyTouchdownMinAdvance,
            config.phaseResetting.earlyTouchdownMinContactDuration)
      << "otherwise the debounce alone pushes every truncation inside the "
         "guard window";
  const BipedFeetArray<scalar_t> noCadence = makeBipedFeetArray(0.0);
  const BipedFeetArray<bool> bothDown = {true, true};

  // The shortest double support anywhere in the schedule, ignoring the leading
  // and trailing stance phases.
  const absl::AnyInvocable<scalar_t(const ModeSchedule&) const>
      shortestDoubleSupport = [](const ModeSchedule& schedule) {
        scalar_t shortest = std::numeric_limits<scalar_t>::infinity();
        for (size_t i = 0; i + 1 < schedule.eventTimes.size(); ++i) {
          if (schedule.modeSequence[i + 1] == kBipedModeStance) {
            shortest = std::min(
                shortest, schedule.eventTimes[i + 1] - schedule.eventTimes[i]);
          }
        }
        return shortest;
      };

  {  // Landing 30 ms early: inside the guard, the exchange is left exactly as
     // planned.
    ModeSchedule schedule = nominal;
    BipedFeetArray<SwingTimingLatch> latches =
        makeBipedFeetArray(SwingTimingLatch{});
    adaptScheduleToContactEvents(schedule, /*time=*/1.37, bothDown, noCadence,
                                 config, latches);
    const BipedFeetArray<ContactEventReport> reports =
        adaptScheduleToContactEvents(schedule, /*time=*/1.392, bothDown,
                                     noCadence, config, latches);
    EXPECT_EQ(reports[0].type, ContactEventReport::Type::kNone)
        << "a near-on-time touch-down is not an early one";
    EXPECT_EQ(schedule.eventTimes, nominal.eventTimes)
        << "the schedule was re-timed";
    EXPECT_EQ(schedule.modeSequence, nominal.modeSequence)
        << "the exchange was split by a sliver of double support";
  }
  {  // Landing 200 ms early: a real early touch-down, still reset, and the
     // double support it opens spans the guard.
    ModeSchedule schedule = nominal;
    BipedFeetArray<SwingTimingLatch> latches =
        makeBipedFeetArray(SwingTimingLatch{});
    adaptScheduleToContactEvents(schedule, /*time=*/1.20, bothDown, noCadence,
                                 config, latches);
    const BipedFeetArray<ContactEventReport> reports =
        adaptScheduleToContactEvents(schedule, /*time=*/1.222, bothDown,
                                     noCadence, config, latches);
    EXPECT_EQ(reports[0].type, ContactEventReport::Type::kEarlyTouchDown);
    EXPECT_NEAR(reports[0].touchDownTime, 1.222, 1.0e-9);
    EXPECT_GE(shortestDoubleSupport(schedule),
              config.phaseResetting.earlyTouchdownMinAdvance - 1.0e-9)
        << "the rule may not create a double support shorter than its own "
           "guard";
  }
  {  // With the guard off the sliver comes back: this is the behavior the guard
     // exists to remove.
    ContactPlanningConfig unguarded = config;
    unguarded.phaseResetting.earlyTouchdownMinAdvance = 0.0;
    EXPECT_EQ(unguarded.validateStatus(), absl::OkStatus());
    ModeSchedule schedule = nominal;
    BipedFeetArray<SwingTimingLatch> latches =
        makeBipedFeetArray(SwingTimingLatch{});
    adaptScheduleToContactEvents(schedule, /*time=*/1.37, bothDown, noCadence,
                                 unguarded, latches);
    const BipedFeetArray<ContactEventReport> reports =
        adaptScheduleToContactEvents(schedule, /*time=*/1.392, bothDown,
                                     noCadence, unguarded, latches);
    EXPECT_EQ(reports[0].type, ContactEventReport::Type::kEarlyTouchDown);
    EXPECT_NEAR(shortestDoubleSupport(schedule), 0.008, 1.0e-9)
        << "8 ms of double support, below the MPC time step";
  }
}

/*=================================== assignment costs: the lower-bound contract
 * ==========================*/
namespace {

/**
 * Every assignment of `numBinaries` binaries with three states per entry -
 * free, 0 and 1 - enumerated in base three.
 *
 * The lower-bound contract of AssignmentCost is a statement about EVERY partial
 * assignment, not about a handful of hand-picked ones, and the branch-and-bound
 * really does evaluate arbitrary partial assignments: MixedIntegerOcpQp calls
 * the assignment cost on a node whose fixings are whatever branching and
 * propagation have produced so far, which is an arbitrary mixture of decided
 * and free entries and not necessarily a prefix in time order (the propagation
 * hook fixes late variables from early ones). Enumerating all 3^6 patterns of a
 * three-node horizon is cheap and is the only way to state the contract as the
 * search relies on it.
 */
std::vector<MiqpAssignment> allPartialAssignments(int numBinaries) {
  int numCodes = 1;
  for (int i = 0; i < numBinaries; ++i) numCodes *= 3;
  std::vector<MiqpAssignment> assignments;
  assignments.reserve(static_cast<size_t>(numCodes));
  for (int code = 0; code < numCodes; ++code) {
    MiqpAssignment assignment(static_cast<size_t>(numBinaries), kMiqpFree);
    int rest = code;
    for (int i = 0; i < numBinaries; ++i) {
      assignment[static_cast<size_t>(i)] = static_cast<int8_t>(rest % 3 - 1);
      rest /= 3;
    }
    assignments.push_back(assignment);
  }
  return assignments;
}

/** Every way of deciding the free entries of `partial`; the singleton {partial}
 * when it is already complete. */
std::vector<MiqpAssignment> allCompletionsOf(const MiqpAssignment& partial) {
  std::vector<size_t> freeIndices;
  for (size_t i = 0; i < partial.size(); ++i) {
    if (partial[i] == kMiqpFree) freeIndices.push_back(i);
  }
  const int numCompletions = 1 << static_cast<int>(freeIndices.size());
  std::vector<MiqpAssignment> completions;
  completions.reserve(static_cast<size_t>(numCompletions));
  for (int mask = 0; mask < numCompletions; ++mask) {
    MiqpAssignment complete = partial;
    for (size_t j = 0; j < freeIndices.size(); ++j) {
      complete[freeIndices[j]] = static_cast<int8_t>((mask >> j) & 1);
    }
    completions.push_back(complete);
  }
  return completions;
}

/** "10 0. .." - one group per node, left foot then right, '.' for a free entry.
 * For the failure messages below. */
std::string printAssignment(const MiqpAssignment& assignment) {
  std::string out;
  for (size_t i = 0; i < assignment.size(); ++i) {
    if (i > 0 && i % kNumFeet == 0) out += " ";
    out +=
        (assignment[i] == kMiqpFree) ? "." : (assignment[i] == 1 ? "1" : "0");
  }
  return out;
}

/** Writes both feet of one node of an assignment. Binaries are node-major:
 * kNumFeet * node + foot. */
void setContacts(MiqpAssignment& assignment,
                 int node,
                 int8_t left,
                 int8_t right) {
  assignment[static_cast<size_t>(
      ContactLogicState::contactBinaryIndex(node, /*foot=*/0))] = left;
  assignment[static_cast<size_t>(
      ContactLogicState::contactBinaryIndex(node, /*foot=*/1))] = right;
}

}  // namespace

/**
 * AssignmentCost.h states the contract of every cost on the binaries that is
 * not part of the QP: "exact for a complete assignment, a lower bound for a
 * partial one", and MiqpAssignmentCostFn repeats it. It is not a stylistic
 * preference. MixedIntegerOcpQp adds the assignment cost of a node's PARTIAL
 * assignment to the bound of its QP relaxation and prunes the node when the sum
 * is within absoluteGap of the incumbent, so a term that overcharges a partial
 * assignment by even one event can cut away the subtree that contains the
 * optimum, and the search would then return a worse gait while still reporting
 * `optimal` - the one failure mode of a branch-and-bound that leaves no trace
 * anywhere.
 *
 * ContactSwitchCost had no test of that contract at all; only its sibling
 * DoubleSupportPenaltyCost did (above). The mechanism that makes it a bound is
 * subtle enough to deserve one: the term skips free entries and carries
 * `previous` across them, so what it counts is the number of alternations of
 * the SUBSEQUENCE of decided nodes, prefixed by the foot's measured contact
 * state. That is a lower bound because two decided nodes that differ force at
 * least one switch somewhere between them, while two decided nodes that agree
 * force none - but any completion may insert a lift-off and a touch-down
 * between them. Charging a free node, or resetting `previous` at one, would
 * break it in either direction.
 */
TEST(
    ContactPlanningTerms,
    ContactSwitchCostIsExactWhenCompleteAndALowerBoundOnEveryPartialAssignment) {
  constexpr int kNumNodes = 3;
  ContactPlanningConfig config = makeConfig();
  config.contactSwitch.cost = 0.25;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AssignmentCost> term,
      ContactPlanningTermFactory::makeAssignmentCost(term::kContactSwitch));
  ASSERT_NE(term, nullptr);
  term->configure(config);
  EXPECT_NE(term->describe().find("per lift-off / touch-down event"),
            std::string::npos)
      << term->describe();

  const ContactPlannerInput standing =
      makeStandingInput();  // both feet in contact at planning time
  ContactLogicState state;
  state.input = &standing;
  state.numNodes = kNumNodes;

  // The measured contact state is the node before the horizon: an assignment
  // that keeps both feet down over the whole horizon costs nothing, and the
  // same assignment costs one event when a foot is in the air at planning time.
  MiqpAssignment allDown(static_cast<size_t>(kNumNodes) * kNumFeet,
                         static_cast<int8_t>(1));
  EXPECT_NEAR(term->cost(state, allDown), 0.0, 1.0e-12)
      << "both feet were already down, nothing happens";
  ContactPlannerInput airborneLeft = standing;
  airborneLeft.contacts = {false, true};
  ContactLogicState airborneState = state;
  airborneState.input = &airborneLeft;
  EXPECT_NEAR(term->cost(airborneState, allDown), 0.25, 1.0e-12)
      << "the left foot touches down at node 0";

  // One full swing of the left foot inside the horizon: lift-off at node 0 and
  // touch-down at node 2 is two events, the right foot never moves. 2 * 0.25 =
  // 0.5.
  MiqpAssignment oneSwing(static_cast<size_t>(kNumNodes) * kNumFeet,
                          static_cast<int8_t>(1));
  setContacts(oneSwing, /*node=*/0, /*left=*/0, /*right=*/1);
  setContacts(oneSwing, /*node=*/1, /*left=*/0, /*right=*/1);
  setContacts(oneSwing, /*node=*/2, /*left=*/1, /*right=*/1);
  EXPECT_NEAR(term->cost(state, oneSwing), 0.5, 1.0e-12)
      << "one lift-off and one touch-down";

  // Both feet stepping on every node is the most expensive pattern of this
  // horizon: three alternations per foot.
  MiqpAssignment chattering(static_cast<size_t>(kNumNodes) * kNumFeet,
                            static_cast<int8_t>(0));
  setContacts(chattering, /*node=*/0, /*left=*/0, /*right=*/0);
  setContacts(chattering, /*node=*/1, /*left=*/1, /*right=*/1);
  setContacts(chattering, /*node=*/2, /*left=*/0, /*right=*/0);
  EXPECT_NEAR(term->cost(state, chattering), 6.0 * 0.25, 1.0e-12)
      << "three events per foot";

  // The subsequence argument, made concrete. Only the left foot at node 1 is
  // decided, to the value the foot already has, so nothing is charged - while
  // the completion that lifts the foot at node 0 and puts it back down at node
  // 1 costs two events. A term that charged the free nodes, or that treated a
  // free node as a swing, would report more than zero here and the search could
  // prune the completion away.
  MiqpAssignment onlyNodeOne(static_cast<size_t>(kNumNodes) * kNumFeet,
                             kMiqpFree);
  onlyNodeOne[static_cast<size_t>(
      ContactLogicState::contactBinaryIndex(1, 0))] = 1;
  EXPECT_NEAR(term->cost(state, onlyNodeOne), 0.0, 1.0e-12)
      << "the only decided node agrees with the measured state";
  MiqpAssignment completedWithASwing =
      onlyNodeOne;  // node 1 already has the left foot down
  setContacts(completedWithASwing, /*node=*/0, /*left=*/0,
              /*right=*/1);  // the left foot lifts before that decided node
  completedWithASwing[static_cast<size_t>(
      ContactLogicState::contactBinaryIndex(1, 1))] = 1;
  setContacts(completedWithASwing, /*node=*/2, /*left=*/1, /*right=*/1);
  EXPECT_NEAR(term->cost(state, completedWithASwing), 0.5, 1.0e-12)
      << "the completion pays for the lift-off and the touch-down the free "
         "nodes were hiding";

  // The contract itself, over every partial assignment of the horizon and every
  // completion of each.
  const std::vector<MiqpAssignment> partials =
      allPartialAssignments(kNumNodes * static_cast<int>(kNumFeet));
  ASSERT_EQ(partials.size(), 729u) << "3^6 patterns of free / 0 / 1";
  int numStrictBounds = 0;
  for (const MiqpAssignment& partial : partials) {
    const scalar_t bound = term->cost(state, partial);
    EXPECT_GE(bound, 0.0) << printAssignment(partial);
    const std::vector<MiqpAssignment> completions = allCompletionsOf(partial);
    for (const MiqpAssignment& complete : completions) {
      const scalar_t exact = term->cost(state, complete);
      ASSERT_LE(bound, exact + 1.0e-12)
          << "the partial assignment " << printAssignment(partial)
          << " is charged " << bound << ", more than its completion "
          << printAssignment(complete) << " at " << exact
          << "; the branch-and-bound would prune the completion away on that "
             "bound";
      if (exact > bound + 1.0e-12) ++numStrictBounds;
    }
  }
  EXPECT_GT(numStrictBounds, 0) << "a bound that never actually undercharges "
                                   "would mean the enumeration is degenerate";

  // Exactness on a complete assignment is the other half of the contract: there
  // the bound has to be reached, i.e. the partial assignment that IS complete
  // evaluates to the same number as itself. That is trivially true here, so we
  // pin that the horizon's contribution is the event
  // count and nothing else - a zero price is free whatever happens.
  ContactPlanningConfig freePrice = config;
  freePrice.contactSwitch.cost = 0.0;
  EXPECT_EQ(freePrice.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AssignmentCost> freeTerm,
      ContactPlanningTermFactory::makeAssignmentCost(term::kContactSwitch));
  freeTerm->configure(freePrice);
  EXPECT_NEAR(freeTerm->cost(state, chattering), 0.0, 1.0e-12)
      << "a zero price charges nothing for any pattern";
}

/**
 * PlanConsistencyCost is the hysteresis that stops the planner from re-deciding
 * the contact pattern every cycle, and it is under the same lower-bound
 * obligation as ContactSwitchCost (see the test above for why a violation is
 * invisible). Its arithmetic is the part that had no test: the previous plan is
 * indexed at `k + previousPlanShift` - the node of the previous plan that is
 * live at the current node, since the horizon has moved forward by that many
 * nodes - and the index is CLAMPED at the previous plan's last node, because
 * the last `shift` nodes of the current horizon reach past the end of the
 * previous one. The clamp is what makes the term compare the tail of the new
 * horizon against the previous plan's terminal pattern instead of running off
 * the end of the vector.
 */
TEST(
    ContactPlanningTerms,
    PlanConsistencyCostShiftsIntoThePreviousPlanClampsAtItsLastNodeAndBoundsEveryPartial) {
  constexpr int kNumNodes = 3;
  ContactPlanningConfig config = makeConfig();
  config.planConsistency.cost = 0.5;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<AssignmentCost> term,
      ContactPlanningTermFactory::makeAssignmentCost(term::kPlanConsistency));
  ASSERT_NE(term, nullptr);
  term->configure(config);
  EXPECT_NE(
      term->describe().find(
          "per decided node whose contact differs from the previous plan"),
      std::string::npos)
      << term->describe();

  const ContactPlannerInput standing = makeStandingInput();
  MiqpAssignment previous(static_cast<size_t>(kNumNodes) * kNumFeet,
                          static_cast<int8_t>(0));
  setContacts(previous, /*node=*/0, /*left=*/1,
              /*right=*/0);  // left down, right swinging
  setContacts(previous, /*node=*/1, /*left=*/0, /*right=*/1);  // the exchange
  setContacts(previous, /*node=*/2, /*left=*/1, /*right=*/1);  // double support

  ContactLogicState state;
  state.input = &standing;
  state.numNodes = kNumNodes;
  state.previousAssignment = &previous;

  // The assignment every case below is scored against: all four feet-nodes in
  // the air is deliberately far from the previous plan, so that each shift
  // produces a different, checkable number.
  MiqpAssignment allUp(static_cast<size_t>(kNumNodes) * kNumFeet,
                       static_cast<int8_t>(0));

  // shift 0 - the previous plan has not moved, node k is compared with node k.
  // The previous plan has four ones in it (node 0 left, node 1 right, node 2
  // both), so four of the six entries differ.
  state.previousPlanShift = 0;
  EXPECT_NEAR(term->cost(state, allUp), 4.0 * 0.5, 1.0e-12)
      << "one charge per previous contact that is now a swing";

  // shift 1 - node 0 reads the previous node 1 (one contact), node 1 reads the
  // previous node 2 (two contacts), and node 2 would read the previous node 3,
  // which does not exist and is clamped back to node 2 (two contacts).
  state.previousPlanShift = 1;
  EXPECT_NEAR(term->cost(state, allUp), 5.0 * 0.5, 1.0e-12)
      << "1 + 2 + 2 charges, the last node clamped";

  // shift 2 - every node of the horizon is at or past the end of the previous
  // plan, so all three read its last node. This is the clamp on its own: three
  // nodes times the two contacts of the previous terminal pattern.
  state.previousPlanShift = 2;
  EXPECT_NEAR(term->cost(state, allUp), 6.0 * 0.5, 1.0e-12)
      << "every node clamped to the previous plan's last node";

  // The clamp is a clamp and not a wrap: with shift 2 the previous plan's node
  // 0 and node 1 are unreachable, so rewriting them cannot change the charge,
  // while rewriting its last node changes all three of them at once.
  MiqpAssignment rewrittenHead = previous;
  setContacts(rewrittenHead, /*node=*/0, /*left=*/0, /*right=*/0);
  setContacts(rewrittenHead, /*node=*/1, /*left=*/0, /*right=*/0);
  ContactLogicState headState = state;
  headState.previousAssignment = &rewrittenHead;
  EXPECT_NEAR(term->cost(headState, allUp), 6.0 * 0.5, 1.0e-12)
      << "the nodes before the clamp are never read at shift 2";
  MiqpAssignment rewrittenTail = previous;
  setContacts(rewrittenTail, /*node=*/2, /*left=*/0, /*right=*/0);
  ContactLogicState tailState = state;
  tailState.previousAssignment = &rewrittenTail;
  EXPECT_NEAR(term->cost(tailState, allUp), 0.0, 1.0e-12)
      << "every node of the horizon reads that one node";

  // An exact, hand-counted value at the shipped shift of one cycle: node 0
  // matches the previous node 1, node 1 matches the previous node 2, and node 2
  // differs from the (clamped) previous node 2 in the right foot alone.
  MiqpAssignment nearlyConsistent(static_cast<size_t>(kNumNodes) * kNumFeet,
                                  static_cast<int8_t>(0));
  setContacts(nearlyConsistent, /*node=*/0, /*left=*/0, /*right=*/1);
  setContacts(nearlyConsistent, /*node=*/1, /*left=*/1, /*right=*/1);
  setContacts(nearlyConsistent, /*node=*/2, /*left=*/1, /*right=*/0);
  state.previousPlanShift = 1;
  EXPECT_NEAR(term->cost(state, nearlyConsistent), 0.5, 1.0e-12)
      << "exactly one node-foot pair differs";

  // No usable previous plan: the reference manager leaves previousPlanShift
  // empty on the first cycle, after a reset and whenever the stored plan no
  // longer matches the node count, and the term has to contribute nothing then
  // rather than index into a plan that is not there.
  ContactLogicState noShift = state;
  noShift.previousPlanShift = std::nullopt;
  EXPECT_NEAR(term->cost(noShift, allUp), 0.0, 1.0e-12)
      << "no usable previous plan, no hysteresis";
  ContactLogicState noPlan = state;
  noPlan.previousAssignment = nullptr;
  EXPECT_NEAR(term->cost(noPlan, allUp), 0.0, 1.0e-12)
      << "a shift without a plan must not be dereferenced";
  ContactLogicState defaulted;
  defaulted.input = &standing;
  defaulted.numNodes = kNumNodes;
  EXPECT_NEAR(term->cost(defaulted, allUp), 0.0, 1.0e-12)
      << "the defaults of ContactLogicState mean 'no previous plan'";

  // The lower-bound contract, exhaustively, at the shift where the clamp is
  // active for the last node.
  state.previousPlanShift = 1;
  const std::vector<MiqpAssignment> partials =
      allPartialAssignments(kNumNodes * static_cast<int>(kNumFeet));
  ASSERT_EQ(partials.size(), 729u);
  int numStrictBounds = 0;
  for (const MiqpAssignment& partial : partials) {
    const scalar_t bound = term->cost(state, partial);
    EXPECT_GE(bound, 0.0) << printAssignment(partial);
    const std::vector<MiqpAssignment> completions = allCompletionsOf(partial);
    for (const MiqpAssignment& complete : completions) {
      const scalar_t exact = term->cost(state, complete);
      ASSERT_LE(bound, exact + 1.0e-12)
          << "the partial assignment " << printAssignment(partial)
          << " is charged " << bound << ", more than its completion "
          << printAssignment(complete) << " at " << exact;
      if (exact > bound + 1.0e-12) ++numStrictBounds;
    }
  }
  EXPECT_GT(numStrictBounds, 0);

  // Exactness on a complete assignment: the charge is the number of disagreeing
  // node-foot pairs, counted here by an independent walk over the assignment
  // that applies the shift and the clamp itself.
  for (const MiqpAssignment& partial : partials) {
    if (std::find(partial.begin(), partial.end(), kMiqpFree) != partial.end()) {
      continue;
    }
    int expected = 0;
    for (int node = 0; node < kNumNodes; ++node) {
      const int source = std::min(node + 1, kNumNodes - 1);
      for (size_t foot = 0; foot < kNumFeet; ++foot) {
        const size_t here = static_cast<size_t>(
            ContactLogicState::contactBinaryIndex(node, foot));
        const size_t there = static_cast<size_t>(
            ContactLogicState::contactBinaryIndex(source, foot));
        if (partial[here] != previous[there]) ++expected;
      }
    }
    ASSERT_NEAR(term->cost(state, partial),
                0.5 * static_cast<scalar_t>(expected), 1.0e-12)
        << printAssignment(partial);
  }
}

/*================================ previous foothold consistency and step width
 * ===========================*/
namespace {

/**
 * A previous plan whose footholds are a distinct, easily recognizable number
 * per node, foot and axis, so that reading the wrong node shows up as a wrong
 * number rather than as a near miss of two neighboring footholds.
 */
ContactPlan makePreviousPlan(int numNodes) {
  ContactPlan plan;
  plan.valid = true;
  plan.startTime = 0.0;
  plan.dt = 0.1;
  plan.desPosFeetRtWorldEwrtWorldXyTraj.resize(static_cast<size_t>(numNodes) +
                                               1u);
  for (int node = 0; node <= numNodes; ++node) {
    const scalar_t s = static_cast<scalar_t>(node);
    plan.desPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(node)][0] =
        vector2_t(0.10 * s, 0.125 + 0.01 * s);
    plan.desPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(node)][1] =
        vector2_t(0.10 * s + 0.05, -0.125 - 0.01 * s);
  }
  return plan;
}

/** A nominal trajectory over numNodes + 1 nodes with a turning heading, for the
 * frame terms of the heading model. */
HeadingNominal makeTurningNominal(int numNodes,
                                  scalar_t nomHeadingModelInitYaw,
                                  scalar_t nomHeadingModelYawPerNode) {
  HeadingNominal nominal;
  nominal.nomHeadingModelYawTraj.resize(static_cast<size_t>(numNodes) + 1u);
  nominal.nomPosComRtWorldEwrtWorldXyTraj.resize(static_cast<size_t>(numNodes) +
                                                 1u);
  nominal.nomPosFeetRtWorldEwrtWorldXyTraj.resize(
      static_cast<size_t>(numNodes) + 1u);
  for (int node = 0; node <= numNodes; ++node) {
    const scalar_t s = static_cast<scalar_t>(node);
    nominal.nomHeadingModelYawTraj[static_cast<size_t>(node)] =
        nomHeadingModelInitYaw + nomHeadingModelYawPerNode * s;
    nominal.nomPosComRtWorldEwrtWorldXyTraj[static_cast<size_t>(node)] =
        vector2_t(0.05 * s, 0.0);
    nominal.nomPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(node)][0] =
        vector2_t(0.30 + 0.02 * s, 0.13 + 0.01 * s);
    nominal.nomPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(node)][1] =
        vector2_t(0.26 + 0.02 * s, -0.11 - 0.01 * s);
  }
  return nominal;
}

}  // namespace

/**
 * PreviousFootholdConsistencyCost is the foothold half of the hysteresis
 * between two planning cycles: it pulls every node's footholds towards the
 * place the previous plan put them, at the node of the previous plan that is
 * live now. Nothing pinned its node arithmetic. The shift is the number of
 * nodes the horizon has advanced since the previous plan started, so the
 * current node k corresponds to the previous plan's node k + shift; the last
 * `shift` nodes of the current horizon reach past the end of the previous plan
 * and are clamped to its final node, which is what makes the tail of the
 * horizon hold still rather than be released the moment it runs off the end.
 *
 * The residual is per foot and per axis, so the assembled stage is diagonal in
 * the four foothold states, and the check below reads the whole expansion - the
 * 2w on the diagonal of Q, the -2 w p_prev in q and the w p_prev^2 in the stage
 * constant - which together are the statement "this node's cost is w ||p -
 * p_prev||^2" and nothing else.
 */
TEST(
    ContactPlanningTerms,
    PreviousFootholdConsistencyCostPullsEachNodeToTheShiftedAndClampedPreviousFoothold) {
  constexpr scalar_t kWeight = 3.0;
  constexpr int kShift = 4;
  ContactPlanningConfig config = makeConfig();
  config.previousFootholdConsistency.weight = kWeight;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const Layout layout,
                              LipContactPlanner::makeLayout(config));
  ASSERT_FALSE(layout.hasHeading) << "the flat planner, where the term is a "
                                     "plain quadratic on the foothold states";
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LipCost> term,
      ContactPlanningTermFactory::makeCost(term::kPreviousFootholdConsistency));
  ASSERT_NE(term, nullptr);
  term->bind(layout);
  term->configure(config);
  EXPECT_EQ(term->nodeSet(), NodeSet::kAll);
  EXPECT_NE(term->describe().find("p_{i,prev}(k + shift)"), std::string::npos)
      << term->describe();

  const int numNodes = config.planner.numNodes;
  ASSERT_GT(numNodes, kShift)
      << "otherwise the unclamped half of the horizon below is empty";
  const ContactPlannerInput input = makeStandingInput();
  const ContactPlan previousPlan = makePreviousPlan(numNodes);
  ASSERT_EQ(previousPlan.desPosFeetRtWorldEwrtWorldXyTraj.size(),
            static_cast<size_t>(numNodes) + 1u);
  const HeadingNominal nominal =
      makeTurningNominal(numNodes, /*nomHeadingModelInitYaw=*/0.0,
                         /*nomHeadingModelYawPerNode=*/0.0);

  ContactPlanningContext ctx;
  ctx.input = &input;
  ctx.layout = &layout;
  ctx.nominal = &nominal;
  ctx.config = &config;
  ctx.previousPlan = &previousPlan;
  ctx.previousPlanShift = kShift;
  ctx.dt = config.planner.dt;
  ctx.numNodes = numNodes;
  ctx.omega = config.omega();
  ctx.bigM = config.shared.bigM;
  ctx.computeAxes();

  for (int node = 0; node <= numNodes; ++node) {
    const bool running = node < numNodes;
    OcpQpStage stage = OcpQpStage::Zero(
        layout.nx, running ? layout.nu : 0,
        running ? OcpQpStageKind::kIntermediate : OcpQpStageKind::kTerminal);
    StageAccumulator accumulator(&stage);
    term->addToStage(ctx, node, accumulator);

    const int source = std::min(node + kShift, numNodes);
    scalar_t expectedConstant = 0.0;
    for (size_t foot = 0; foot < kNumFeet; ++foot) {
      for (int axis = 0; axis < 2; ++axis) {
        const int index =
            layout.state(axis == 0 ? var::posFootRtWorldEwrtWorldX(foot)
                                   : var::posFootRtWorldEwrtWorldY(foot));
        const vector2_t& prevDesPosFootRtWorldEwrtWorldXy =
            previousPlan.desPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(
                source)][foot];
        EXPECT_NEAR(stage.Q(index, index), 2.0 * kWeight, 1.0e-12)
            << "node " << node << ", foot " << foot << ", axis " << axis;
        EXPECT_NEAR(stage.q(index),
                    -2.0 * kWeight * prevDesPosFootRtWorldEwrtWorldXy(axis),
                    1.0e-12)
            << "node " << node << " must be pulled to the previous plan's node "
            << source << ", foot " << foot << ", axis " << axis;
        expectedConstant += kWeight * prevDesPosFootRtWorldEwrtWorldXy(axis) *
                            prevDesPosFootRtWorldEwrtWorldXy(axis);
      }
    }
    EXPECT_NEAR(stage.constant, expectedConstant, 1.0e-12)
        << "node " << node << ": the full squared residual, offset included";
    EXPECT_NEAR(stage.Q(kLipCx, kLipCx), 0.0, 1.0e-12)
        << "node " << node << ": the term touches only the foothold states";
    EXPECT_NEAR(stage.Q(kLipVy, kLipVy), 0.0, 1.0e-12) << "node " << node;
    EXPECT_NEAR(stage.Q(kLipPlx, kLipPrx), 0.0, 1.0e-12)
        << "node " << node << ": the feet are not coupled";
    if (running) {
      EXPECT_TRUE(stage.R.isZero())
          << "node " << node << ": the term touches no input";
      EXPECT_TRUE(stage.r.isZero()) << "node " << node;
      EXPECT_TRUE(stage.S.isZero()) << "node " << node;
    }
  }

  // The clamp, stated on its own rather than through the loop's own arithmetic:
  // every node from numNodes - shift on is pulled to the previous plan's final
  // foothold, and the nodes before it are not.
  const std::vector<int> clampedNodes = {numNodes - kShift, numNodes - 1,
                                         numNodes};
  const scalar_t prevFinalDesPosLfootRtWorldEwrtWorldX =
      previousPlan.desPosFeetRtWorldEwrtWorldXyTraj.back()[0].x();
  for (const int node : clampedNodes) {
    OcpQpStage stage =
        OcpQpStage::Zero(layout.nx, 0, OcpQpStageKind::kTerminal);
    StageAccumulator accumulator(&stage);
    term->addToStage(ctx, node, accumulator);
    EXPECT_NEAR(stage.q(kLipPlx),
                -2.0 * kWeight * prevFinalDesPosLfootRtWorldEwrtWorldX, 1.0e-12)
        << "node " << node << " reaches past the previous plan";
  }
  {
    OcpQpStage stage =
        OcpQpStage::Zero(layout.nx, 0, OcpQpStageKind::kTerminal);
    StageAccumulator accumulator(&stage);
    term->addToStage(ctx, numNodes - kShift - 1, accumulator);
    EXPECT_GT(std::abs(stage.q(kLipPlx) +
                       2.0 * kWeight * prevFinalDesPosLfootRtWorldEwrtWorldX),
              1.0e-6)
        << "the last node before the clamp must still read a node of its own";
  }

  // No usable previous plan. previousPlanShift is empty on the first cycle,
  // after reset() and whenever the stored plan no longer matches the node
  // count, and previousPlan is then null. The term has to emit nothing at all -
  // not a pull towards the origin, which is what a missing guard would produce
  // and which would drag both feet to (0, 0).
  {
    ContactPlanningContext withoutPlan = ctx;
    withoutPlan.previousPlan = nullptr;
    withoutPlan.previousPlanShift = std::nullopt;
    OcpQpStage stage =
        OcpQpStage::Zero(layout.nx, layout.nu, OcpQpStageKind::kIntermediate);
    StageAccumulator accumulator(&stage);
    term->addToStage(withoutPlan, /*node=*/0, accumulator);
    EXPECT_TRUE(stage.Q.isZero())
        << "without a previous plan the term contributes nothing";
    EXPECT_TRUE(stage.q.isZero());
    EXPECT_NEAR(stage.constant, 0.0, 1.0e-15);
  }
  {  // A negative shift with a plan still present is the same "not usable"
     // state and must not index at a negative node.
    ContactPlanningContext staleShift = ctx;
    staleShift.previousPlanShift = std::nullopt;
    OcpQpStage stage =
        OcpQpStage::Zero(layout.nx, layout.nu, OcpQpStageKind::kIntermediate);
    StageAccumulator accumulator(&stage);
    term->addToStage(staleShift, /*node=*/0, accumulator);
    EXPECT_TRUE(stage.Q.isZero());
    EXPECT_TRUE(stage.q.isZero());
  }
  {  // A zero weight switches the term off without removing it from the
     // formulation.
    ContactPlanningConfig unweighted = config;
    unweighted.previousFootholdConsistency.weight = 0.0;
    EXPECT_EQ(unweighted.validateStatus(), absl::OkStatus());
    LEGGED_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LipCost> disabled,
                                ContactPlanningTermFactory::makeCost(
                                    term::kPreviousFootholdConsistency));
    disabled->bind(layout);
    disabled->configure(unweighted);
    OcpQpStage stage =
        OcpQpStage::Zero(layout.nx, layout.nu, OcpQpStageKind::kIntermediate);
    StageAccumulator accumulator(&stage);
    disabled->addToStage(ctx, /*node=*/0, accumulator);
    EXPECT_TRUE(stage.Q.isZero());
    EXPECT_TRUE(stage.q.isZero());
    EXPECT_NEAR(stage.constant, 0.0, 1.0e-15);
  }
}

/**
 * StepWidthCost appeared in no test anywhere in the repository, which is
 * remarkable for a term that is in the default cost list of every shipped
 * formulation and that is the only thing holding the two feet apart laterally
 * once the heading model is on. It is also the most involved of the foothold
 * costs, because it is the one that is linearized: the lateral separation
 * e_y(theta) . (p_L - p_R) is not linear in the heading, so the term keeps the
 * axis e_y at the NOMINAL heading of the node and adds the first-order heading
 * term g (theta - theta_n) with g = d e_y / d theta . d_n = -e_x(theta_n) .
 * (p_L,n - p_R,n), which couples five variables - four footholds and the
 * heading - into a single residual.
 *
 * Everything about that is easy to get wrong in a way no smoke test would
 * notice: the sign of the derivative of e_y (it is -e_x, not +e_x), which axis
 * of the frame is differentiated, whether the offset carries -g theta_n so that
 * the term is an expansion AROUND the nominal rather than a term proportional
 * to the absolute heading, and which node's nominal is read. The residual and
 * its gradient are therefore checked here against numbers written out by hand
 * at a heading of 0.5 rad, where every one of those mistakes changes the
 * answer.
 */
TEST(ContactPlanningTerms,
     StepWidthCostCouplesTheFourFootholdsAndTheHeadingAtANonZeroHeading) {
  constexpr scalar_t kWeight = 7.0;
  constexpr scalar_t kNomPosLfootRtRfootEwrtHeadingY = 0.24;
  constexpr int kNode = 3;
  ContactPlanningConfig config = makeHeadingConfig();
  config.stepWidth.weight = kWeight;
  config.stepWidth.nomPosLfootRtRfootEwrtHeadingY =
      kNomPosLfootRtRfootEwrtHeadingY;
  EXPECT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const Layout layout,
                              LipContactPlanner::makeLayout(config));
  ASSERT_TRUE(layout.hasHeading);
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LipCost> term,
      ContactPlanningTermFactory::makeCost(term::kStepWidth));
  ASSERT_NE(term, nullptr);
  term->bind(layout);
  term->configure(config);
  EXPECT_EQ(term->nodeSet(), NodeSet::kAll);
  EXPECT_NE(term->describe().find("e_y . (p_L - p_R)"), std::string::npos)
      << term->describe();

  const int numNodes = config.planner.numNodes;
  const ContactPlannerInput input = makeStandingInput();
  // heading 0.2 at node 0 and +0.1 per node, so node 3 sits at exactly 0.5 rad
  // and no two nodes share a frame.
  const HeadingNominal nominal =
      makeTurningNominal(numNodes, /*nomHeadingModelInitYaw=*/0.2,
                         /*nomHeadingModelYawPerNode=*/0.1);
  ContactPlanningContext ctx;
  ctx.input = &input;
  ctx.layout = &layout;
  ctx.nominal = &nominal;
  ctx.config = &config;
  ctx.previousPlan = nullptr;
  ctx.previousPlanShift = std::nullopt;
  ctx.inertiaComEwrtWorldZz = input.inertiaComEwrtWorldZz;
  ctx.dt = config.planner.dt;
  ctx.numNodes = numNodes;
  ctx.omega = config.omega();
  ctx.bigM = config.shared.bigM;
  ctx.computeAxes();

  // The hand computation. theta_n = 0.5, d_n = p_L,n - p_R,n = (0.36, 0.16) -
  // (0.32, -0.14) = (0.04, 0.30).
  const scalar_t theta =
      nominal.nomHeadingModelYawTraj[static_cast<size_t>(kNode)];
  ASSERT_NEAR(theta, 0.5, 1.0e-12);
  const vector2_t nomPosLfootRtRfootEwrtWorldXy =
      nominal.nomPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(kNode)][0] -
      nominal.nomPosFeetRtWorldEwrtWorldXyTraj[static_cast<size_t>(kNode)][1];
  ASSERT_NEAR(nomPosLfootRtRfootEwrtWorldXy.x(), 0.04, 1.0e-12);
  ASSERT_NEAR(nomPosLfootRtRfootEwrtWorldXy.y(), 0.30, 1.0e-12);
  const vector2_t leftDirHeadingEwrtWorldXy(-std::sin(theta), std::cos(theta));
  // g = (d e_y / d theta) . d_n = -e_x(theta_n) . d_n = -(cos 0.5 * 0.04 + sin
  // 0.5 * 0.30).
  const scalar_t g = -(std::cos(theta) * nomPosLfootRtRfootEwrtWorldXy.x() +
                       std::sin(theta) * nomPosLfootRtRfootEwrtWorldXy.y());
  ASSERT_NEAR(g, -0.178930964056875812, 1.0e-12)
      << "-(0.8775825618903728 * 0.04 + 0.4794255386042030 * 0.30)";
  // The residual is e_y . (p_L - p_R) + g (theta - theta_n) - w_nom, so the
  // constant part is -w_nom - g theta_n.
  const scalar_t offset = -kNomPosLfootRtRfootEwrtHeadingY - g * theta;
  ASSERT_NEAR(offset, -0.150534517971562094, 1.0e-12)
      << "-0.24 + 0.178930964056875812 * 0.5";

  vector_t coefficients = vector_t::Zero(layout.nx);
  coefficients(layout.state(var::posFootRtWorldEwrtWorldX(0))) =
      leftDirHeadingEwrtWorldXy.x();
  coefficients(layout.state(var::posFootRtWorldEwrtWorldY(0))) =
      leftDirHeadingEwrtWorldXy.y();
  coefficients(layout.state(var::posFootRtWorldEwrtWorldX(1))) =
      -leftDirHeadingEwrtWorldXy.x();
  coefficients(layout.state(var::posFootRtWorldEwrtWorldY(1))) =
      -leftDirHeadingEwrtWorldXy.y();
  coefficients(layout.headingModelYaw) = g;

  OcpQpStage stage =
      OcpQpStage::Zero(layout.nx, layout.nu, OcpQpStageKind::kIntermediate);
  StageAccumulator accumulator(&stage);
  term->addToStage(ctx, kNode, accumulator);

  // The expansion, entry by entry. Q = 2 w l l', q = 2 w c l, constant = w c^2.
  const matrix_t expectedQ =
      2.0 * kWeight * coefficients * coefficients.transpose();
  const vector_t expectedq = 2.0 * kWeight * offset * coefficients;
  EXPECT_TRUE(stage.Q.isApprox(expectedQ, 1.0e-12)) << "Q =\n"
                                                    << stage.Q << "\nexpected\n"
                                                    << expectedQ;
  EXPECT_TRUE(stage.q.isApprox(expectedq, 1.0e-12))
      << "q = " << stage.q.transpose() << ", expected "
      << expectedq.transpose();
  EXPECT_NEAR(stage.constant, kWeight * offset * offset, 1.0e-12);
  EXPECT_NEAR(stage.Q(kLipPly, kLipPly),
              2.0 * kWeight * leftDirHeadingEwrtWorldXy.y() *
                  leftDirHeadingEwrtWorldXy.y(),
              1.0e-12);
  EXPECT_NEAR(stage.Q(kLipPly, kLipPry),
              -2.0 * kWeight * leftDirHeadingEwrtWorldXy.y() *
                  leftDirHeadingEwrtWorldXy.y(),
              1.0e-12)
      << "the two feet enter with opposite signs";
  EXPECT_NEAR(stage.Q(kLipPlx, layout.headingModelYaw),
              2.0 * kWeight * leftDirHeadingEwrtWorldXy.x() * g, 1.0e-12)
      << "the heading is coupled to every foothold";
  EXPECT_NEAR(stage.Q(layout.headingModelYaw, layout.headingModelYaw),
              2.0 * kWeight * g * g, 1.0e-12);
  EXPECT_NEAR(stage.q(layout.headingModelYaw), 2.0 * kWeight * offset * g,
              1.0e-12);
  EXPECT_NEAR(stage.Q(kLipCx, kLipCx), 0.0, 1.0e-12)
      << "the center of mass is not part of the step width";
  EXPECT_NEAR(stage.Q(layout.headingModelYawRate, layout.headingModelYawRate),
              0.0, 1.0e-12)
      << "nor is the heading rate";
  EXPECT_TRUE(stage.R.isZero()) << "the term touches no input";
  EXPECT_TRUE(stage.r.isZero());
  EXPECT_TRUE(stage.S.isZero());

  // The residual and its gradient at a state that is not the nominal, which is
  // the statement the entries above add up to: the stage evaluates to w r^2 and
  // its gradient is 2 w r l, with r written out from the formula of the term.
  vector_t state = vector_t::Zero(layout.nx);
  state(layout.state(var::posFootRtWorldEwrtWorldX(0))) = 0.31;
  state(layout.state(var::posFootRtWorldEwrtWorldY(0))) = 0.14;
  state(layout.state(var::posFootRtWorldEwrtWorldX(1))) = 0.27;
  state(layout.state(var::posFootRtWorldEwrtWorldY(1))) = -0.12;
  state(layout.headingModelYaw) = 0.55;
  const vector2_t posLfootRtRfootEwrtWorldXy(
      state(layout.state(var::posFootRtWorldEwrtWorldX(0))) -
          state(layout.state(var::posFootRtWorldEwrtWorldX(1))),
      state(layout.state(var::posFootRtWorldEwrtWorldY(0))) -
          state(layout.state(var::posFootRtWorldEwrtWorldY(1))));
  const scalar_t residual =
      leftDirHeadingEwrtWorldXy.dot(posLfootRtRfootEwrtWorldXy) +
      g * (state(layout.headingModelYaw) - theta) -
      kNomPosLfootRtRfootEwrtHeadingY;
  ASSERT_GT(std::abs(residual), 1.0e-3)
      << "the trial is only meaningful away from the minimum of the term";
  EXPECT_NEAR(
      0.5 * state.dot(stage.Q * state) + stage.q.dot(state) + stage.constant,
      kWeight * residual * residual, 1.0e-12)
      << "the stage must evaluate to w (e_y . (p_L - p_R) + g (theta - "
         "theta_n) - w_nom)^2";
  const vector_t gradient = stage.Q * state + stage.q;
  const vector_t expectedGradient = 2.0 * kWeight * residual * coefficients;
  EXPECT_TRUE(gradient.isApprox(expectedGradient, 1.0e-9))
      << "gradient = " << gradient.transpose()
      << ", expected 2 w r l = " << expectedGradient.transpose();

  // Each node carries its own frame and its own nominal separation. Node 0 sits
  // at 0.2 rad with d_n = (0.04, 0.24), so reading the wrong node's nominal - a
  // plausible off-by-one in a term that indexes ctx.nominal by node - changes
  // both the axis and g. Checking a second node also pins that the term is not
  // accidentally constant over the horizon.
  const scalar_t thetaZero = nominal.nomHeadingModelYawTraj[0];
  ASSERT_NEAR(thetaZero, 0.2, 1.0e-12);
  const vector2_t initNomPosLfootRtRfootEwrtWorldXy =
      nominal.nomPosFeetRtWorldEwrtWorldXyTraj[0][0] -
      nominal.nomPosFeetRtWorldEwrtWorldXyTraj[0][1];
  ASSERT_NEAR(initNomPosLfootRtRfootEwrtWorldXy.y(), 0.24, 1.0e-12);
  const vector2_t initLeftDirHeadingEwrtWorldXy(-std::sin(thetaZero),
                                                std::cos(thetaZero));
  const scalar_t gZero =
      -(std::cos(thetaZero) * initNomPosLfootRtRfootEwrtWorldXy.x() +
        std::sin(thetaZero) * initNomPosLfootRtRfootEwrtWorldXy.y());
  OcpQpStage stageZero =
      OcpQpStage::Zero(layout.nx, layout.nu, OcpQpStageKind::kIntermediate);
  StageAccumulator accumulatorZero(&stageZero);
  term->addToStage(ctx, /*node=*/0, accumulatorZero);
  EXPECT_NEAR(stageZero.Q(kLipPly, kLipPly),
              2.0 * kWeight * initLeftDirHeadingEwrtWorldXy.y() *
                  initLeftDirHeadingEwrtWorldXy.y(),
              1.0e-12);
  EXPECT_NEAR(stageZero.Q(layout.headingModelYaw, layout.headingModelYaw),
              2.0 * kWeight * gZero * gZero, 1.0e-12);
  EXPECT_NEAR(stageZero.q(layout.headingModelYaw),
              2.0 * kWeight *
                  (-kNomPosLfootRtRfootEwrtHeadingY - gZero * thetaZero) *
                  gZero,
              1.0e-12);
  EXPECT_GT(
      std::abs(stageZero.Q(layout.headingModelYaw, layout.headingModelYaw) -
               stage.Q(layout.headingModelYaw, layout.headingModelYaw)),
      1.0e-6)
      << "two nodes with different nominal headings must not produce the same "
         "coupling";

  // Without the heading block the term degenerates to the plain lateral
  // separation in the frame of the measured base yaw, with no heading column
  // and no expansion offset: e_y(yaw) . (p_L - p_R) - w_nom.
  ContactPlanningConfig flatConfig = makeConfig();
  flatConfig.stepWidth.weight = kWeight;
  flatConfig.stepWidth.nomPosLfootRtRfootEwrtHeadingY =
      kNomPosLfootRtRfootEwrtHeadingY;
  EXPECT_EQ(flatConfig.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const Layout flatLayout,
                              LipContactPlanner::makeLayout(flatConfig));
  ASSERT_FALSE(flatLayout.hasHeading);
  ASSERT_EQ(flatLayout.headingModelYaw, -1);
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LipCost> flatTerm,
      ContactPlanningTermFactory::makeCost(term::kStepWidth));
  flatTerm->bind(flatLayout);
  flatTerm->configure(flatConfig);
  ContactPlannerInput yawedInput = makeStandingInput();
  yawedInput.yawHeadingRtWorld = 0.4;
  ContactPlanningContext flatCtx;
  flatCtx.input = &yawedInput;
  flatCtx.layout = &flatLayout;
  flatCtx.nominal = &nominal;
  flatCtx.config = &flatConfig;
  flatCtx.dt = flatConfig.planner.dt;
  flatCtx.numNodes = flatConfig.planner.numNodes;
  flatCtx.omega = flatConfig.omega();
  flatCtx.bigM = flatConfig.shared.bigM;
  flatCtx.computeAxes();
  OcpQpStage flatStage = OcpQpStage::Zero(flatLayout.nx, flatLayout.nu,
                                          OcpQpStageKind::kIntermediate);
  StageAccumulator flatAccumulator(&flatStage);
  flatTerm->addToStage(flatCtx, kNode, flatAccumulator);
  const vector2_t flatLeftDirHeadingEwrtWorldXy(
      -std::sin(yawedInput.yawHeadingRtWorld),
      std::cos(yawedInput.yawHeadingRtWorld));
  EXPECT_NEAR(flatStage.Q(kLipPly, kLipPly),
              2.0 * kWeight * flatLeftDirHeadingEwrtWorldXy.y() *
                  flatLeftDirHeadingEwrtWorldXy.y(),
              1.0e-12)
      << "the frame is the base yaw";
  EXPECT_NEAR(flatStage.Q(kLipPlx, kLipPly),
              2.0 * kWeight * flatLeftDirHeadingEwrtWorldXy.x() *
                  flatLeftDirHeadingEwrtWorldXy.y(),
              1.0e-12);
  EXPECT_NEAR(flatStage.q(kLipPly),
              -2.0 * kWeight * kNomPosLfootRtRfootEwrtHeadingY *
                  flatLeftDirHeadingEwrtWorldXy.y(),
              1.0e-12)
      << "the offset is -w_nom alone";
  EXPECT_NEAR(flatStage.constant,
              kWeight * kNomPosLfootRtRfootEwrtHeadingY *
                  kNomPosLfootRtRfootEwrtHeadingY,
              1.0e-12);
  EXPECT_EQ(flatLayout.nx, kLipStateDim)
      << "no heading column exists for the term to couple to";
}

/*============================================ audit findings CP-2, CP-3, CP-5,
 * CP-6 ======================================*/
namespace {

/** The rows `term` writes at `node`, as a stage. */
OcpQpStage constraintRows(const LipConstraint& term,
                          const ContactPlanningContext& ctx,
                          const Layout& layout,
                          int node) {
  RowBuilder rows(layout.nx, layout.nu);
  term.addRows(ctx, node, rows);
  OcpQpStage stage =
      OcpQpStage::Zero(layout.nx, layout.nu, OcpQpStageKind::kIntermediate);
  rows.writeTo(stage);
  return stage;
}

/** Whether lower <= C x + D u <= upper holds for every row of `stage`, to
 * `tolerance`. */
bool rowsHold(const OcpQpStage& stage,
              const vector_t& x,
              const vector_t& u,
              scalar_t tolerance) {
  const vector_t value = stage.C * x + stage.D * u;
  for (int i = 0; i < value.size(); ++i) {
    if (value(i) < stage.lg(i) - tolerance ||
        value(i) > stage.ug(i) + tolerance) {
      return false;
    }
  }
  return true;
}

ContactPlanningContext makeHeadingContext(const ContactPlannerInput& input,
                                          const Layout& layout,
                                          const HeadingNominal& nominal,
                                          const ContactPlanningConfig& config) {
  ContactPlanningContext ctx;
  ctx.input = &input;
  ctx.layout = &layout;
  ctx.nominal = &nominal;
  ctx.config = &config;
  ctx.inertiaComEwrtWorldZz = input.inertiaComEwrtWorldZz;
  ctx.dt = config.planner.dt;
  ctx.numNodes = config.planner.numNodes;
  ctx.omega = config.omega();
  ctx.bigM = config.shared.bigM;
  ctx.computeAxes();
  return ctx;
}

vector2_t rotated(scalar_t yaw, const vector2_t& vector) {
  return vector2_t(std::cos(yaw) * vector.x() - std::sin(yaw) * vector.y(),
                   std::sin(yaw) * vector.x() + std::cos(yaw) * vector.y());
}

}  // namespace

/**
 * Audit finding CP-2. Under the heading model a single-support ZMP box is the
 * stance foot's footprint, so it turns with that foot's yaw and not with the
 * heading. With the feet turned 0.6 and -0.5 rad away from the heading, every
 * corner of the rotated footprint is admitted from just inside and refused from
 * just outside - while the same corners lie outside the heading-aligned box the
 * rows used to write.
 */
TEST(ContactPlanningTerms,
     TheSingleSupportZmpBoxIsTheStanceFootsRotatedFootprint) {
  constexpr scalar_t kYawHeadingRtWorld = 0.1;
  const BipedFeetArray<scalar_t> nomEulerzyxFeetRtWorldYaw{
      kYawHeadingRtWorld + 0.6, kYawHeadingRtWorld - 0.5};
  ContactPlanningConfig config = makeHeadingConfig();
  config.zmpSupportRegion.maxNormPosZmpRtFootEwrtFootX = 0.1;
  config.zmpSupportRegion.maxNormPosZmpRtFootEwrtFootY = 0.05;
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const Layout layout,
                              LipContactPlanner::makeLayout(config));
  ASSERT_TRUE(layout.hasHeading);
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LipConstraint> term,
      ContactPlanningTermFactory::makeSoftConstraint(term::kZmpSupportRegion));
  term->bind(layout);
  term->configure(config);

  const int numNodes = config.planner.numNodes;
  HeadingNominal nominal = makeTurningNominal(
      numNodes, /*nomHeadingModelInitYaw=*/kYawHeadingRtWorld,
      /*nomHeadingModelYawPerNode=*/0.0);
  nominal.nomEulerzyxFeetRtWorldYawTraj.assign(
      static_cast<size_t>(numNodes) + 1u, nomEulerzyxFeetRtWorldYaw);
  const ContactPlannerInput input = makeStandingInput();
  const ContactPlanningContext ctx =
      makeHeadingContext(input, layout, nominal, config);

  const vector2_t maxNormPosZmpRtFootEwrtFootXy(
      config.zmpSupportRegion.maxNormPosZmpRtFootEwrtFootX,
      config.zmpSupportRegion.maxNormPosZmpRtFootEwrtFootY);
  const BipedFeetArray<vector2_t> posFeetRtWorldEwrtWorldXy{
      vector2_t(0.31, 0.12), vector2_t(0.27, -0.13)};
  for (const int node : {0, 4}) {
    const OcpQpStage stage = constraintRows(*term, ctx, layout, node);
    for (size_t stance = 0; stance < kNumFeet; ++stance) {
      vector_t x = vector_t::Zero(layout.nx);
      for (size_t foot = 0; foot < kNumFeet; ++foot) {
        x(layout.state(var::posFootRtWorldEwrtWorldX(foot))) =
            posFeetRtWorldEwrtWorldXy[foot].x();
        x(layout.state(var::posFootRtWorldEwrtWorldY(foot))) =
            posFeetRtWorldEwrtWorldXy[foot].y();
      }
      vector_t u = vector_t::Zero(layout.nu);
      u(layout.input(var::contact(stance))) = 1.0;
      int cornersOutsideTheHeadingBox = 0;
      for (const scalar_t signX : {1.0, -1.0}) {
        for (const scalar_t signY : {1.0, -1.0}) {
          const vector2_t posCornerRtFootEwrtWorldXy =
              rotated(nomEulerzyxFeetRtWorldYaw[stance],
                      vector2_t(signX * maxNormPosZmpRtFootEwrtFootXy.x(),
                                signY * maxNormPosZmpRtFootEwrtFootXy.y()));
          for (const scalar_t scale : {0.999, 1.001}) {
            const vector2_t posZmpRtWorldEwrtWorldXy =
                posFeetRtWorldEwrtWorldXy[stance] +
                scale * posCornerRtFootEwrtWorldXy;
            u(layout.input(var::kPosZmpRtWorldEwrtWorldX)) =
                posZmpRtWorldEwrtWorldXy.x();
            u(layout.input(var::kPosZmpRtWorldEwrtWorldY)) =
                posZmpRtWorldEwrtWorldXy.y();
            EXPECT_EQ(rowsHold(stage, x, u, /*tolerance=*/1.0e-12), scale < 1.0)
                << "node " << node << ", stance " << stance
                << ", posCornerRtFootEwrtWorldXy (" << signX << ", " << signY
                << "), scale " << scale;
          }
          const vector2_t posCornerRtFootEwrtHeadingXy =
              rotated(-kYawHeadingRtWorld, posCornerRtFootEwrtWorldXy);
          if (std::abs(posCornerRtFootEwrtHeadingXy.x()) >
                  maxNormPosZmpRtFootEwrtFootXy.x() ||
              std::abs(posCornerRtFootEwrtHeadingXy.y()) >
                  maxNormPosZmpRtFootEwrtFootXy.y()) {
            ++cornersOutsideTheHeadingBox;
          }
        }
      }
      EXPECT_GT(cornersOutsideTheHeadingBox, 0)
          << "the fixture must tell the footprint from the heading-aligned box";
    }
  }

  // The nominal the planner builds itself takes the stance feet at their
  // measured yaws.
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  ContactPlannerInput turned = makeStandingInput();
  turned.eulerzyxFeetRtWorldYaw = nomEulerzyxFeetRtWorldYaw;
  const HeadingNominal plannerNominal = planner->defaultNominal(turned);
  ASSERT_EQ(plannerNominal.nomEulerzyxFeetRtWorldYawTraj.size(),
            static_cast<size_t>(numNodes) + 1u);
  EXPECT_NEAR(plannerNominal.nomEulerzyxFeetRtWorldYawTraj.front()[0],
              nomEulerzyxFeetRtWorldYaw[0], 1.0e-12);
  EXPECT_NEAR(plannerNominal.nomEulerzyxFeetRtWorldYawTraj.front()[1],
              nomEulerzyxFeetRtWorldYaw[1], 1.0e-12);
}

/**
 * Audit finding CP-3. CadenceStretchStage compares objectives of the same N
 * nodes on grids of dt and s dt, i.e. over horizons of different lengths. That
 * comparison is per unit time already: a running cost summed over N equal nodes
 * is N times its time average. A trajectory with the same velocity error
 * everywhere scores the same on both grids, which a duration-weighted score
 * (the integral) would not.
 */
TEST(ContactPlanningTerms, AConstantCostRateScoresTheSameOnEveryNodeGrid) {
  constexpr scalar_t kStretch = 1.25;
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kVelocityTracking};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {};
  config.velocityTracking.weight = 3.0;
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  ContactPlanningConfig stretchedConfig = config;
  stretchedConfig.planner.dt = kStretch * config.planner.dt;
  ASSERT_EQ(stretchedConfig.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  LEGGED_ASSERT_OK_AND_ASSIGN(
      const std::unique_ptr<LipContactPlanner> stretchedPlanner,
      LipContactPlanner::Create(stretchedConfig));
  ContactPlannerInput input = makeStandingInput();
  input.cmdLinvelBaseRtWorldEwrtWorldXy = vector2_t(0.4, -0.1);
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem qp,
                              planner->buildProblem(input));
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem stretchedQp,
                              stretchedPlanner->buildProblem(input));
  ASSERT_EQ(qp.stages.size(), stretchedQp.stages.size());

  const vector2_t errLinvelComRtWorldEwrtWorldXy(0.07, -0.02);
  std::vector<vector_t> states;
  std::vector<vector_t> inputs;
  for (size_t k = 0; k < qp.stages.size(); ++k) {
    vector_t x = vector_t::Zero(qp.stages[k].numStates());
    x(kLipVx) = input.cmdLinvelBaseRtWorldEwrtWorldXy.x() +
                errLinvelComRtWorldEwrtWorldXy.x();
    x(kLipVy) = input.cmdLinvelBaseRtWorldEwrtWorldXy.y() +
                errLinvelComRtWorldEwrtWorldXy.y();
    x(kLipCx) = 0.3 * static_cast<scalar_t>(
                          k);  // where the CoM is does not enter the rate
    states.push_back(x);
    if (k + 1 < qp.stages.size()) {
      inputs.push_back(vector_t::Zero(qp.stages[k].numInputs()));
    }
  }
  const scalar_t score = evaluateOcpQpObjective(qp, states, inputs);
  const scalar_t stretchedScore =
      evaluateOcpQpObjective(stretchedQp, states, inputs);
  ASSERT_GT(score, 0.0);
  EXPECT_NEAR(stretchedScore, score, 1.0e-12)
      << "the same cost rate is the same score on both grids";
  EXPECT_GT(std::abs(stretchedConfig.planner.dt * stretchedScore -
                     config.planner.dt * score),
            1.0e-3 * score)
      << "a duration-weighted score would have charged the stretched grid for "
         "its longer horizon";
}

/**
 * Audit finding CP-5. The hip yaw joint turns the foot against the pelvis. With
 * the heading the ACoM's, the pelvis sits yawBaseRtWorld - heading away from
 * it, and the rows bound psi_i - theta_base to the joint range: at the planning
 * node (theta = heading) a foot yaw is admitted exactly when psi_i -
 * yawBaseRtWorld is inside [lower_i, upper_i], across the +-pi seam too.
 */
TEST(ContactPlanningTerms, TheHipYawRangeIsMeasuredFromThePelvis) {
  ContactPlanningConfig config = makeHeadingConfig();
  config.hipYawRange.minYawFeetRtBase = {-0.3, -0.6};
  config.hipYawRange.maxYawFeetRtBase = {0.6, 0.3};
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const Layout layout,
                              LipContactPlanner::makeLayout(config));
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LipConstraint> term,
      ContactPlanningTermFactory::makeSoftConstraint(term::kHipYawRange));
  term->bind(layout);
  term->configure(config);
  const int numNodes = config.planner.numNodes;

  struct Case {
    scalar_t yawAcomRtWorld;
    std::optional<scalar_t> yawBaseRtWorld;
  };
  const std::vector<Case> cases = {
      {.yawAcomRtWorld = 0.4, .yawBaseRtWorld = 0.55},
      {.yawAcomRtWorld = 0.4, .yawBaseRtWorld = std::nullopt},
      {.yawAcomRtWorld = 3.10,
       .yawBaseRtWorld = -3.05},  // the pelvis across the seam, 0.133 rad ahead
      {.yawAcomRtWorld = -1.0, .yawBaseRtWorld = -1.25}};
  for (const Case& c : cases) {
    ContactPlannerInput input = makeStandingInput();
    input.initYawHeadingRtWorld = c.yawAcomRtWorld;
    input.yawHeadingRtWorld = c.yawAcomRtWorld;
    input.yawBaseRtWorld = c.yawBaseRtWorld;
    const HeadingNominal nominal = makeTurningNominal(
        numNodes, /*nomHeadingModelInitYaw=*/c.yawAcomRtWorld,
        /*nomHeadingModelYawPerNode=*/0.0);
    const ContactPlanningContext ctx =
        makeHeadingContext(input, layout, nominal, config);
    const OcpQpStage stage = constraintRows(*term, ctx, layout, /*node=*/0);
    const scalar_t yawBaseRtWorld = c.yawBaseRtWorld.value_or(c.yawAcomRtWorld);
    for (size_t foot = 0; foot < kNumFeet; ++foot) {
      for (int sample = 0; sample <= 40; ++sample) {
        const scalar_t hipAngle = -1.0 + 0.05 * static_cast<scalar_t>(sample) +
                                  0.0013;  // off the bounds themselves
        vector_t x = vector_t::Zero(layout.nx);
        x(layout.headingModelYaw) = c.yawAcomRtWorld;
        for (size_t other = 0; other < kNumFeet; ++other) {
          x(layout.eulerzyxFootRtWorldYaw(other)) =
              c.yawAcomRtWorld;  // inside both ranges
        }
        // The foot yaw on the heading's branch, as the planner carries it.
        x(layout.eulerzyxFootRtWorldYaw(foot)) = moduloAngleWithReference(
            yawBaseRtWorld + hipAngle, c.yawAcomRtWorld);
        const vector_t u = vector_t::Zero(layout.nu);
        const bool insideJointRange =
            config.hipYawRange.minYawFeetRtBase[foot] <= hipAngle &&
            hipAngle <= config.hipYawRange.maxYawFeetRtBase[foot];
        EXPECT_EQ(rowsHold(stage, x, u, /*tolerance=*/1.0e-12),
                  insideJointRange)
            << "heading " << c.yawAcomRtWorld << ", foot " << foot
            << ", hip angle " << hipAngle;
      }
    }
  }
}

/**
 * Audit finding CP-6. The yaw torque a foot may carry, read off the rows for
 * every contact state and both regimes of the two budgets: exactly T_t alone,
 * (T_t + T_c) / 2 in double support and nothing swinging - or in flight, where
 * the old single row allowed (T_t - T_c) / 2 when T_c < T_t and made flight
 * infeasible when T_c > T_t. Between the corners the rows are the concave
 * envelope of the budget, the largest convex combination of the corner budgets
 * (an independent one-parameter maximization here).
 */
TEST(ContactPlanningTerms,
     TheYawTorqueBudgetIsExactAtEveryContactStateAndZeroInFlight) {
  for (const std::pair<scalar_t, scalar_t>& budgets :
       {std::pair<scalar_t, scalar_t>(20.0, 40.0),
        std::pair<scalar_t, scalar_t>(40.0, 20.0)}) {
    ContactPlanningConfig config = makeHeadingConfig();
    config.yawTorqueBudgetParameters.maxTorqueFootEwrtFootZ = budgets.first;
    config.yawTorqueBudgetParameters.doubleSupportMaxTorqueFeetmeanEwrtWorldZ =
        budgets.second;
    ASSERT_EQ(config.validateStatus(), absl::OkStatus());
    LEGGED_ASSERT_OK_AND_ASSIGN(const Layout layout,
                                LipContactPlanner::makeLayout(config));
    LEGGED_ASSERT_OK_AND_ASSIGN(std::unique_ptr<LipConstraint> term,
                                ContactPlanningTermFactory::makeHardConstraint(
                                    term::kYawTorqueBudgetName));
    term->bind(layout);
    term->configure(config);
    const ContactPlanningContext ctx;
    const OcpQpStage stage = constraintRows(*term, ctx, layout, /*node=*/0);
    const scalar_t alone = budgets.first;
    const scalar_t shared = 0.5 * (budgets.first + budgets.second);

    // The largest |tau_foot| the rows admit with the binaries at (own, other),
    // for either sign of the torque.
    const absl::AnyInvocable<scalar_t(size_t, scalar_t, scalar_t, scalar_t)
                                 const>
        budget = [&](size_t foot, scalar_t own, scalar_t other, scalar_t sign) {
          vector_t u = vector_t::Zero(layout.nu);
          u(layout.input(var::contact(foot))) = own;
          u(layout.input(var::contact(1 - foot))) = other;
          const int column = layout.input(var::torqueFootEwrtWorldZ(foot));
          scalar_t bound = std::numeric_limits<scalar_t>::infinity();
          for (int i = 0; i < stage.numGeneralConstraints(); ++i) {
            const scalar_t coefficient = sign * stage.D(i, column);
            if (coefficient <= 0.0) continue;
            bound = std::min(
                bound, (stage.ug(i) - stage.D.row(i).dot(u)) / coefficient);
          }
          return bound;
        };
    // The concave envelope: max over t of T_t (a - t) + D t, t = lambda_11 in
    // [max(0, a + b - 1), min(a, b)].
    const absl::AnyInvocable<scalar_t(scalar_t, scalar_t) const> envelope =
        [&](scalar_t own, scalar_t other) {
          const scalar_t t = shared >= alone ? std::min(own, other)
                                             : std::max(0.0, own + other - 1.0);
          return alone * (own - t) + shared * t;
        };
    for (size_t foot = 0; foot < kNumFeet; ++foot) {
      for (const scalar_t sign : {1.0, -1.0}) {
        EXPECT_NEAR(budget(foot, /*own=*/1.0, /*other=*/0.0, sign), alone,
                    1.0e-9)
            << "alone";
        EXPECT_NEAR(budget(foot, /*own=*/1.0, /*other=*/1.0, sign), shared,
                    1.0e-9)
            << "double support";
        EXPECT_NEAR(budget(foot, /*own=*/0.0, /*other=*/1.0, sign), 0.0, 1.0e-9)
            << "swinging";
        EXPECT_NEAR(budget(foot, /*own=*/0.0, /*other=*/0.0, sign), 0.0, 1.0e-9)
            << "flight: zero torque, and feasible";
        for (int sample = 0; sample < 25; ++sample) {
          const scalar_t own = 0.1 + 0.2 * static_cast<scalar_t>(sample % 5);
          const scalar_t other = 0.1 + 0.2 * static_cast<scalar_t>(sample / 5);
          EXPECT_NEAR(budget(foot, own, other, sign), envelope(own, other),
                      1.0e-9)
              << "relaxation at (" << own << ", " << other << ")";
        }
      }
    }
  }
}

/**
 * The input's world-frame command is the operator's heading-frame command
 * rotated by the commanded heading at the planning instant; the operator's
 * target turns it by omega_cmd t over the horizon, and heading_tracking ramps
 * theta_0 + omega_cmd k dt. The velocity reference of node k is therefore the
 * captured command rotated by omega_cmd k dt, with or without the heading
 * model. It used to stay along the initial heading at every node.
 */
TEST(ContactPlanningTerms,
     VelocityTrackingReferenceTurnsWithTheCommandedYawRate) {
  for (const bool withHeadingModel : {false, true}) {
    ContactPlanningConfig config =
        withHeadingModel ? makeHeadingConfig() : makeConfig();
    config.formulation.costs = {term::kVelocityTracking};
    config.formulation.softConstraints = {};
    config.formulation.hardConstraints = {};
    config.velocityTracking.weight = 3.0;
    ASSERT_EQ(config.validateStatus(), absl::OkStatus());
    LEGGED_ASSERT_OK_AND_ASSIGN(
        const std::unique_ptr<LipContactPlanner> planner,
        LipContactPlanner::Create(config));
    ContactPlannerInput input = makeStandingInput();
    input.cmdLinvelBaseRtWorldEwrtWorldXy = vector2_t(0.5, 0.1);
    input.cmdRateYawBaseRtWorld = 0.6;
    LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem qp,
                                planner->buildProblem(input));
    ASSERT_EQ(qp.stages.size(),
              static_cast<size_t>(config.planner.numNodes) + 1u);
    for (size_t node = 0; node < qp.stages.size(); ++node) {
      const OcpQpStage& stage = qp.stages[node];
      const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy =
          rotated(input.cmdRateYawBaseRtWorld * config.planner.dt *
                      static_cast<scalar_t>(node),
                  input.cmdLinvelBaseRtWorldEwrtWorldXy);
      for (int axis = 0; axis < kHorizontalPositionDim; ++axis) {
        const int column = kLipVx + axis;
        ASSERT_GT(stage.Q(column, column), 0.0);
        // w (v - v_ref)^2 contributes Q = 2w and q = -2w v_ref.
        EXPECT_NEAR(stage.q(column) / stage.Q(column, column),
                    -cmdLinvelBaseRtWorldEwrtWorldXy(axis), 1.0e-12)
            << "heading model " << withHeadingModel << ", node " << node
            << ", axis " << axis;
      }
    }
  }
}

/**
 * The terminal DCM target zmp_{N-1} + v_cmd / omega is the target of xi_N, the
 * DCM at the end of the horizon, so its velocity is the command at node N: the
 * captured command rotated by omega_cmd N dt, one node past the last running
 * node the residual is written on.
 */
TEST(ContactPlanningTerms, TerminalDcmTargetTurnsWithTheCommandedYawRate) {
  ContactPlanningConfig config = makeConfig();
  config.formulation.costs = {term::kTerminalDcm};
  config.formulation.softConstraints = {};
  config.formulation.hardConstraints = {};
  config.terminalDcm.trackVelocityCommand = true;
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  ContactPlannerInput input = makeStandingInput();
  input.cmdLinvelBaseRtWorldEwrtWorldXy = vector2_t(0.4, -0.15);
  input.cmdRateYawBaseRtWorld = -0.8;
  LEGGED_ASSERT_OK_AND_ASSIGN(const OcpQpProblem qp,
                              planner->buildProblem(input));
  const size_t last = static_cast<size_t>(config.planner.numNodes - 1);
  const OcpQpStage& stage = qp.stages[last];
  const scalar_t omega = config.omega();
  const scalar_t dt = config.planner.dt;
  const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy =
      rotated(input.cmdRateYawBaseRtWorld * dt *
                  static_cast<scalar_t>(config.planner.numNodes),
              input.cmdLinvelBaseRtWorldEwrtWorldXy);
  // At xi_N = zmp + v_cmd,N / omega the cost vanishes: xi_{N-1} - zmp =
  // e^{-omega dt} v_cmd,N / omega.
  vector_t x = vector_t::Zero(stage.numStates());
  vector_t u = vector_t::Zero(stage.numInputs());
  for (int axis = 0; axis < kHorizontalPositionDim; ++axis) {
    u(kLipZx + axis) = 0.07 * (axis + 1);
    x(kLipCx + axis) = 0.02;
    x(kLipVx + axis) =
        omega * (u(kLipZx + axis) - x(kLipCx + axis)) +
        std::exp(-omega * dt) * cmdLinvelBaseRtWorldEwrtWorldXy(axis);
  }
  const scalar_t cost = 0.5 * x.dot(stage.Q * x) + 0.5 * u.dot(stage.R * u) +
                        u.dot(stage.S * x) + stage.q.dot(x) + stage.r.dot(u) +
                        stage.constant;
  EXPECT_NEAR(cost, 0.0, 1.0e-9)
      << "the target of xi_N is the command at node N";
}

// Without the heading model the support geometry (step width, reachability,
// foot separation, ZMP boxes) and the command references (velocity tracking,
// step length, terminal DCM) of node k are in one frame: the planning-time yaw
// turned by omega_cmd k dt. The axes used to stay at the planning-time yaw at
// every node while the references turned, 1.2 rad apart at the last node here.
TEST(ContactPlanningTerms,
     WithoutTheHeadingModelTheSupportFrameTurnsWithTheCommand) {
  constexpr scalar_t kWeight = 3.0;
  constexpr scalar_t kNomPosLfootRtRfootEwrtHeadingY = 0.25;
  ContactPlanningConfig config = makeConfig();
  config.stepWidth.weight = kWeight;
  config.stepWidth.nomPosLfootRtRfootEwrtHeadingY =
      kNomPosLfootRtRfootEwrtHeadingY;
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  LEGGED_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<LipContactPlanner> planner,
                              LipContactPlanner::Create(config));
  ASSERT_FALSE(planner->getProblem().layout().hasHeading);
  ContactPlannerInput input = makeStandingInput();
  input.yawHeadingRtWorld = 0.3;
  input.cmdLinvelBaseRtWorldEwrtWorldXy = vector2_t(0.5, 0.1);
  input.cmdRateYawBaseRtWorld = 1.0;
  const HeadingNominal nominal = planner->defaultNominal(input);
  LEGGED_ASSERT_OK_AND_ASSIGN(const ContactPlanningContext ctx,
                              planner->makeContext(input, nominal));
  const scalar_t dt = config.planner.dt;
  const vector2_t initCmdLinvelBaseRtWorldEwrtHeadingXy(
      ctx.getRotmatHeadingRtWorld(/*node=*/0)[0].dot(
          input.cmdLinvelBaseRtWorldEwrtWorldXy),
      ctx.getRotmatHeadingRtWorld(/*node=*/0)[1].dot(
          input.cmdLinvelBaseRtWorldEwrtWorldXy));
  for (int node = 0; node <= ctx.numNodes; ++node) {
    const scalar_t cmdYawHeadingRtWorld =
        input.yawHeadingRtWorld +
        input.cmdRateYawBaseRtWorld * dt * static_cast<scalar_t>(node);
    const std::array<vector2_t, kHorizontalPositionDim>& axes =
        ctx.getRotmatHeadingRtWorld(node);
    EXPECT_TRUE(axes[0].isApprox(vector2_t(std::cos(cmdYawHeadingRtWorld),
                                           std::sin(cmdYawHeadingRtWorld)),
                                 1.0e-12))
        << "node " << node << ": e_x = " << axes[0].transpose();
    EXPECT_TRUE(axes[1].isApprox(vector2_t(-std::sin(cmdYawHeadingRtWorld),
                                           std::cos(cmdYawHeadingRtWorld)),
                                 1.0e-12))
        << "node " << node << ": e_y = " << axes[1].transpose();
    for (size_t foot = 0; foot < kNumFeet; ++foot) {
      EXPECT_TRUE(
          ctx.getRotmatFootRtWorld(node, foot)[1].isApprox(axes[1], 1.0e-12))
          << "node " << node << ", foot " << foot;
    }
    // The command reference of the node, read in the node's frame, is the
    // command read in the planning frame: the two turn together.
    const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy =
        ctx.getCmdLinvelBaseRtWorldEwrtWorldXy(node);
    EXPECT_NEAR(axes[0].dot(cmdLinvelBaseRtWorldEwrtWorldXy),
                initCmdLinvelBaseRtWorldEwrtHeadingXy.x(), 1.0e-12)
        << "node " << node;
    EXPECT_NEAR(axes[1].dot(cmdLinvelBaseRtWorldEwrtWorldXy),
                initCmdLinvelBaseRtWorldEwrtHeadingXy.y(), 1.0e-12)
        << "node " << node;
  }

  // The step width of the last node penalizes the separation along the lateral
  // axis of the commanded heading there, not of the planning-time yaw.
  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<LipCost> stepWidth,
      ContactPlanningTermFactory::makeCost(term::kStepWidth));
  stepWidth->bind(planner->getProblem().layout());
  stepWidth->configure(config);
  const int lastNode = ctx.numNodes - 1;
  OcpQpStage stage = OcpQpStage::Zero(planner->getProblem().layout().nx,
                                      planner->getProblem().layout().nu,
                                      OcpQpStageKind::kIntermediate);
  StageAccumulator accumulator(&stage);
  stepWidth->addToStage(ctx, lastNode, accumulator);
  const scalar_t finalCmdYawHeadingRtWorld =
      input.yawHeadingRtWorld +
      input.cmdRateYawBaseRtWorld * dt * static_cast<scalar_t>(lastNode);
  const vector2_t leftDirHeadingEwrtWorldXy(
      -std::sin(finalCmdYawHeadingRtWorld),
      std::cos(finalCmdYawHeadingRtWorld));
  EXPECT_NEAR(stage.Q(kLipPly, kLipPly),
              2.0 * kWeight * leftDirHeadingEwrtWorldXy.y() *
                  leftDirHeadingEwrtWorldXy.y(),
              1.0e-12);
  EXPECT_NEAR(stage.Q(kLipPlx, kLipPly),
              2.0 * kWeight * leftDirHeadingEwrtWorldXy.x() *
                  leftDirHeadingEwrtWorldXy.y(),
              1.0e-12);
  EXPECT_NEAR(stage.q(kLipPly),
              -2.0 * kWeight * kNomPosLfootRtRfootEwrtHeadingY *
                  leftDirHeadingEwrtWorldXy.y(),
              1.0e-12);
}

// The contact count of the model reaches the planner's terms through
// shared.numContacts. A term written for a left and a right foot refuses any
// other count by its name and the count; the terms written over the number of
// feet accept it. finalize() asks every listed term, so the default
// formulation, which lists biped-only terms, is refused as a whole.
TEST(ContactPlanningTerms, BipedOnlyTermsRefuseAContactSetThatIsNotTwoFeet) {
  constexpr size_t kFourContacts = 4;
  struct BipedOnlyTerm {
    absl::StatusOr<std::unique_ptr<ContactPlanningTerm>> term;
    std::string className;
  };
  std::vector<BipedOnlyTerm> bipedOnly;
  bipedOnly.push_back({.term = ContactPlanningTermFactory::makeModelBlock(
                           term::kFootholdIntegrator),
                       .className = "FootholdIntegrator"});
  bipedOnly.push_back({.term = ContactPlanningTermFactory::makeSoftConstraint(
                           term::kFootSeparation),
                       .className = "FootSeparationConstraint"});
  bipedOnly.push_back({.term = ContactPlanningTermFactory::makeSoftConstraint(
                           term::kZmpSupportRegion),
                       .className = "ZmpSupportRegionConstraint"});
  bipedOnly.push_back({.term = ContactPlanningTermFactory::makeHardConstraint(
                           term::kYawTorqueBudgetName),
                       .className = "YawTorqueBudgetConstraint"});
  bipedOnly.push_back({.term = ContactPlanningTermFactory::makeLogicRule(
                           term::kAlternatingFeet),
                       .className = "AlternatingFeetRule"});
  for (const BipedOnlyTerm& entry : bipedOnly) {
    ASSERT_TRUE(entry.term.ok()) << entry.term.status();
    const ContactPlanningTerm& bipedTerm = **entry.term;
    EXPECT_EQ(
        bipedTerm.validateContactSet(static_cast<size_t>(kBipedFootCount)),
        absl::OkStatus())
        << entry.className;
    const absl::Status refused = bipedTerm.validateContactSet(kFourContacts);
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(
        refused.message(),
        absl::StrCat(entry.className,
                     " is written for two contacts; the contact set has 4 "
                     "contacts"));
  }

  std::vector<absl::StatusOr<std::unique_ptr<ContactPlanningTerm>>> generic;
  generic.emplace_back(
      ContactPlanningTermFactory::makeHardConstraint(term::kNoFlight));
  generic.emplace_back(
      ContactPlanningTermFactory::makeLogicRule(term::kNoFlight));
  generic.emplace_back(
      ContactPlanningTermFactory::makeLogicRule(term::kPhaseDurations));
  generic.emplace_back(
      ContactPlanningTermFactory::makeLogicRule(term::kMinimumDoubleSupport));
  for (const absl::StatusOr<std::unique_ptr<ContactPlanningTerm>>& genericTerm :
       generic) {
    ASSERT_TRUE(genericTerm.ok()) << genericTerm.status();
    EXPECT_EQ((*genericTerm)->validateContactSet(kFourContacts),
              absl::OkStatus())
        << (*genericTerm)->describe();
  }

  ContactPlanningConfig fourContacts = makeConfig();
  fourContacts.shared.numContacts = kFourContacts;
  const absl::Status assembled =
      ContactPlanningTermFactory::buildProblemStatus(fourContacts).status();
  EXPECT_EQ(assembled.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(std::string(assembled.message())
                .find("is written for two contacts; the contact set has 4 "
                      "contacts"),
            std::string::npos)
      << assembled;
}

// The counting core of no_flight and minimum_double_support on four feet. The
// planner's variable layout and its input (ContactPlannerInput) hold two feet,
// so the rules themselves cannot be run on a four-foot problem yet; their
// count logic is the helper, which is written over the number of feet.
TEST(ContactPlanningTerms, TheMinimumContactCountPropagatesOverFourFeet) {
  const std::array<int, 4> nodeBinaries = {0, 1, 2, 3};
  const ContactMask everyFoot = ContactMask::allActive(/*numContacts=*/4);

  // no_flight: at least one foot. Three feet lifted force the fourth down.
  MiqpAssignment threeLifted = {0, 0, kMiqpFree, 0};
  bool changed = false;
  EXPECT_TRUE(propagateMinimumContactCount(nodeBinaries, everyFoot,
                                           /*minContactCount=*/1, threeLifted,
                                           changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(threeLifted, (MiqpAssignment{0, 0, 1, 0}));
  // Two feet still free: either may support, nothing is implied.
  MiqpAssignment twoFree = {0, kMiqpFree, kMiqpFree, 0};
  changed = false;
  EXPECT_TRUE(propagateMinimumContactCount(nodeBinaries, everyFoot,
                                           /*minContactCount=*/1, twoFree,
                                           changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(twoFree, (MiqpAssignment{0, kMiqpFree, kMiqpFree, 0}));
  // Every foot lifted is a flight phase.
  MiqpAssignment flight = {0, 0, 0, 0};
  changed = false;
  EXPECT_FALSE(propagateMinimumContactCount(nodeBinaries, everyFoot,
                                            /*minContactCount=*/1, flight,
                                            changed));

  // A minimum support of three during a hold that binds the first three feet
  // (the fourth was in the air before the node). With the fourth foot free,
  // every bound foot can still lift while three others may support: nothing is
  // implied. Once the fourth lifts, the three bound feet are the only possible
  // support and stay down.
  ContactMask boundFeet(/*numContacts=*/4);
  boundFeet.set(/*contactIndex=*/0, /*active=*/true);
  boundFeet.set(/*contactIndex=*/1, /*active=*/true);
  boundFeet.set(/*contactIndex=*/2, /*active=*/true);
  MiqpAssignment fourthFree = {kMiqpFree, kMiqpFree, kMiqpFree, kMiqpFree};
  changed = false;
  EXPECT_TRUE(propagateMinimumContactCount(nodeBinaries, boundFeet,
                                           /*minContactCount=*/3, fourthFree,
                                           changed));
  EXPECT_FALSE(changed);
  MiqpAssignment fourthLifted = {kMiqpFree, kMiqpFree, kMiqpFree, 0};
  changed = false;
  EXPECT_TRUE(propagateMinimumContactCount(nodeBinaries, boundFeet,
                                           /*minContactCount=*/3, fourthLifted,
                                           changed));
  EXPECT_TRUE(changed);
  EXPECT_EQ(fourthLifted, (MiqpAssignment{1, 1, 1, 0}));
  // A bound foot already lifted while the others cannot reach three is a
  // contradiction; with the fourth down they can, and nothing is fixed.
  MiqpAssignment boundLifted = {0, kMiqpFree, kMiqpFree, 0};
  changed = false;
  EXPECT_FALSE(propagateMinimumContactCount(nodeBinaries, boundFeet,
                                            /*minContactCount=*/3, boundLifted,
                                            changed));
  MiqpAssignment fourthDown = {0, kMiqpFree, kMiqpFree, 1};
  changed = false;
  EXPECT_TRUE(propagateMinimumContactCount(nodeBinaries, boundFeet,
                                           /*minContactCount=*/3, fourthDown,
                                           changed));
  EXPECT_TRUE(changed) << "the two free bound feet are the rest of the three";
  EXPECT_EQ(fourthDown, (MiqpAssignment{0, 1, 1, 1}));
}

// The counting core of phase_durations on four feet (see the test above for
// why the helper and not the rule): an overdue foot's lift is implied only
// once every other foot is decided, one of them supports, the hold does not
// keep it down, and no supporting overdue foot lifts first.
TEST(ContactPlanningTerms, AnOverdueLiftIsImpliedOverFourFeet) {
  const OtherFootAtNode supporting = {
      .decided = true, .inContact = true, .liftsFirst = false};
  const OtherFootAtNode airborne = {
      .decided = true, .inContact = false, .liftsFirst = false};
  const OtherFootAtNode undecided = {
      .decided = false, .inContact = false, .liftsFirst = false};
  const OtherFootAtNode overdueFirst = {
      .decided = true, .inContact = true, .liftsFirst = true};

  const std::array<OtherFootAtNode, 3> oneSupports = {airborne, supporting,
                                                      airborne};
  EXPECT_TRUE(overdueLiftIsImplied(oneSupports, DoubleSupportHold::kNone));
  EXPECT_FALSE(overdueLiftIsImplied(oneSupports, DoubleSupportHold::kHeld));
  const std::array<OtherFootAtNode, 3> noneSupports = {airborne, airborne,
                                                       airborne};
  EXPECT_FALSE(overdueLiftIsImplied(noneSupports, DoubleSupportHold::kNone))
      << "lifting the last support would be a flight phase";
  const std::array<OtherFootAtNode, 3> oneUndecided = {supporting, undecided,
                                                       supporting};
  EXPECT_FALSE(overdueLiftIsImplied(oneUndecided, DoubleSupportHold::kNone));
  const std::array<OtherFootAtNode, 3> anotherGoesFirst = {
      supporting, overdueFirst, airborne};
  EXPECT_FALSE(
      overdueLiftIsImplied(anotherGoesFirst, DoubleSupportHold::kNone));
  // For a biped the other feet are one foot, and the four cases of the rule
  // are the ones it always had.
  EXPECT_TRUE(overdueLiftIsImplied(std::array<OtherFootAtNode, 1>{supporting},
                                   DoubleSupportHold::kNone));
  EXPECT_FALSE(overdueLiftIsImplied(std::array<OtherFootAtNode, 1>{airborne},
                                    DoubleSupportHold::kNone));
  EXPECT_FALSE(overdueLiftIsImplied(std::array<OtherFootAtNode, 1>{undecided},
                                    DoubleSupportHold::kNone));
  EXPECT_FALSE(overdueLiftIsImplied(
      std::array<OtherFootAtNode, 1>{overdueFirst}, DoubleSupportHold::kNone));
}

// minimum_double_support as a minimum support count on the biped: the
// registry's rule (every foot) and an explicit count of two or more propagate
// identically on every partial assignment of a short horizon, and a count of
// one lets the other foot lift at a touch-down while the landing foot
// supports.
TEST(ContactPlanningTerms, TheMinimumSupportCountDefaultsToEveryFoot) {
  constexpr int kNumNodes = 3;
  ContactPlanningConfig config = makeConfig();
  config.planner.numNodes = kNumNodes;
  ASSERT_EQ(config.validateStatus(), absl::OkStatus());
  ContactPlannerInput landing = makeStandingInput();
  landing.contacts = {false, true};  // the left foot swings at planning time
  landing.phaseElapsedTime = {0.2, 1.0};
  const ContactLogicState state = ContactLogicState::make(
      landing, config, /*previousPlanShift=*/std::nullopt,
      /*previousAssignment=*/nullptr);
  ASSERT_GT(state.nDoubleSupportHold, 0);

  LEGGED_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ContactLogicRule> registryRule,
      ContactPlanningTermFactory::makeLogicRule(term::kMinimumDoubleSupport));
  registryRule->configure(config);
  MinimumDoubleSupportRule twoFeet(/*minSupportCount=*/2);
  MinimumDoubleSupportRule moreThanTheFeet(/*minSupportCount=*/5);
  twoFeet.configure(config);
  moreThanTheFeet.configure(config);
  for (const MiqpAssignment& partial :
       allPartialAssignments(state.numBinaries())) {
    MiqpAssignment expected = partial;
    bool expectedChanged = false;
    const bool expectedFeasible =
        registryRule->propagate(state, expected, expectedChanged);
    const std::array<const MinimumDoubleSupportRule* absl_nonnull, 2>
        explicitRules = {&twoFeet, &moreThanTheFeet};
    for (const MinimumDoubleSupportRule* absl_nonnull rule : explicitRules) {
      MiqpAssignment actual = partial;
      bool actualChanged = false;
      ASSERT_EQ(rule->propagate(state, actual, actualChanged), expectedFeasible)
          << printAssignment(partial);
      if (!expectedFeasible) continue;
      EXPECT_EQ(actual, expected) << printAssignment(partial);
      EXPECT_EQ(actualChanged, expectedChanged) << printAssignment(partial);
    }
  }

  // The left foot lands at node 0 with the right foot's binary still free:
  // the double support holds the right foot down, a minimum support of one does
  // not, since the landing left foot already supports.
  MiqpAssignment touchDown(static_cast<size_t>(state.numBinaries()), kMiqpFree);
  touchDown[static_cast<size_t>(ContactLogicState::contactBinaryIndex(
      /*node=*/0, kContactLeftIndex))] = 1;
  MiqpAssignment held = touchDown;
  bool changed = false;
  ASSERT_TRUE(registryRule->propagate(state, held, changed));
  EXPECT_EQ(held[static_cast<size_t>(ContactLogicState::contactBinaryIndex(
                /*node=*/0, kContactRightIndex))],
            1);
  MinimumDoubleSupportRule oneFoot(/*minSupportCount=*/1);
  oneFoot.configure(config);
  MiqpAssignment released = touchDown;
  changed = false;
  ASSERT_TRUE(oneFoot.propagate(state, released, changed));
  EXPECT_FALSE(changed);
  EXPECT_EQ(released[static_cast<size_t>(ContactLogicState::contactBinaryIndex(
                /*node=*/0, kContactRightIndex))],
            kMiqpFree);
}

}  // namespace ocs2::legged
