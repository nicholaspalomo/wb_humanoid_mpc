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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ocs2_core/initialization/Initializer.h>
#include <ocs2_core/model_data/ModelData.h>
#include <ocs2_core/model_data/Multiplier.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_oc/approximate_model/LinearQuadraticApproximator.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/oc_problem/OptimalControlProblemHelperFunction.h>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsFootCost.h"

/**
 * The whole-body MPC as it ships for the G1, built end to end: every CppAD library of its dynamics, costs and
 * constraints is taped and compiled, and the assembled problem is then evaluated once at the initial state.
 *
 * Nothing else constructs this MPC. It stopped being constructible without any test noticing: the base acceleration of
 * the floating-base dynamics was solved with a pivoting 6x6 inverse, whose comparisons between CppAD variables the code
 * generator cannot evaluate, and WBMpcInterface::Create threw "GreaterThanZero cannot be called for non-parameters"
 * while taping the dynamics (testFloatingBaseDynamics pins the solve itself).
 */
namespace ocs2::humanoid {
namespace {

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

std::unique_ptr<WBMpcInterface> createShippedG1WholeBodyMpc() {
  const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
  const std::string referenceFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml");
  const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
  EXPECT_FALSE(taskFile.empty() || referenceFile.empty() || urdfFile.empty()) << "the G1 whole-body files are not in the runfiles";
  if (taskFile.empty() || referenceFile.empty() || urdfFile.empty()) return nullptr;
  // The CppAD libraries go to a folder relative to the working directory (ModelSettings::modelFolderCppAd), and the
  // shipped file does not force a recompile, so a library left there by an earlier run would be loaded instead of
  // taped: an unsandboxed run (--spawn_strategy=local, the binary run by hand) would then pass on libraries built from
  // older code. A directory made for this run is empty, so everything is taped and compiled here.
  std::string freshDirectory = (std::filesystem::path(testing::TempDir()) / "testWBMpcConstruction_XXXXXX").string();
  if (mkdtemp(freshDirectory.data()) == nullptr) {
    ADD_FAILURE() << "cannot make a fresh working directory from " << freshDirectory;
    return nullptr;
  }
  std::filesystem::current_path(freshDirectory);
  try {
    absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(taskFile, urdfFile, referenceFile);
    EXPECT_TRUE(created.ok()) << created.status();
    return created.ok() ? *std::move(created) : nullptr;
  } catch (const std::exception& e) {
    ADD_FAILURE() << "WBMpcInterface::Create threw: " << e.what();
    return nullptr;
  }
}

/**
 * The MPC of createShippedG1WholeBodyMpc(), built once for the whole test program, since building it compiles every
 * library. Each test sets the references it needs. Never destroyed.
 */
WBMpcInterface* shippedG1WholeBodyMpc() {
  static WBMpcInterface* const interface = createShippedG1WholeBodyMpc().release();
  return interface;
}

}  // namespace

TEST(WBMpcConstructionTest, theShippedG1WholeBodyMpcIsConstructedAndItsProblemEvaluates) {
  WBMpcInterface* const interface = shippedG1WholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  // The dynamics were taped and compiled by this run, into the directory made for it.
  EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(interface->modelSettings().modelFolderCppAd) / "dynamics_flow_map"))
      << "no dynamics library under " << std::filesystem::current_path() / interface->modelSettings().modelFolderCppAd;

  const scalar_t initTime = 0.0;
  const scalar_t finalTime = initTime + interface->mpcSettings().timeHorizon_;
  const vector_t& initState = interface->getInitialState();
  ASSERT_EQ(initState.size(), static_cast<Eigen::Index>(interface->getMpcRobotModel().getStateDim()));
  ASSERT_TRUE(initState.allFinite());

  // What the MPC node does on its reset - a target at the initial state - and the solver before every solve: the
  // references of this horizon, then the problem pointed at them.
  ReferenceManagerInterface& referenceManager = *interface->getReferenceManagerPtr();
  const vector_t zeroInput = vector_t::Zero(interface->getMpcRobotModel().getInputDim());
  referenceManager.setTargetTrajectories(TargetTrajectories({initTime}, {initState}, {zeroInput}));
  referenceManager.preSolverRun(initTime, finalTime, initState, ModeNumber::STANCE);
  OptimalControlProblem problem(interface->getOptimalControlProblem());
  problem.targetTrajectoriesPtr = &referenceManager.getTargetTrajectories();

  vector_t input;
  vector_t nextState;
  // Initializer::compute() is not const, so a copy computes the input, as each solver does with its own.
  const std::unique_ptr<Initializer> initializer(interface->getInitializer().clone());
  initializer->compute(initTime, initState, initTime + 0.01, input, nextState);
  ASSERT_EQ(input.size(), static_cast<Eigen::Index>(interface->getMpcRobotModel().getInputDim()));

  // One evaluation of everything the solver asks for at an intermediate node - the dynamics, every cost and soft
  // constraint, the equality constraints - and of the inequality constraints and the final cost beside it.
  MultiplierCollection multipliers;
  initializeIntermediateMultiplierCollection(problem, initTime, multipliers);
  const ModelData intermediate = approximateIntermediateLQ(problem, initTime, initState, input, multipliers);
  EXPECT_EQ(intermediate.dynamics.f.size(), initState.size());
  EXPECT_TRUE(intermediate.dynamics.f.allFinite()) << intermediate.dynamics.f.transpose();
  EXPECT_TRUE(intermediate.dynamics.dfdx.allFinite());
  EXPECT_TRUE(intermediate.dynamics.dfdu.allFinite());
  EXPECT_TRUE(std::isfinite(intermediate.cost.f));
  EXPECT_TRUE(intermediate.cost.dfdx.allFinite());
  EXPECT_TRUE(intermediate.cost.dfdu.allFinite());
  EXPECT_TRUE(intermediate.stateInputEqConstraint.f.allFinite());
  EXPECT_TRUE(intermediate.stateInputEqConstraint.dfdu.allFinite());
  const VectorFunctionLinearApproximation inequalities =
      problem.inequalityConstraintPtr->getLinearApproximation(initTime, initState, input, *problem.preComputationPtr);
  EXPECT_TRUE(inequalities.f.allFinite());

  MultiplierCollection finalMultipliers;
  initializeFinalMultiplierCollection(problem, finalTime, finalMultipliers);
  const ModelData terminal = approximateFinalLQ(problem, finalTime, initState, finalMultipliers);
  EXPECT_TRUE(std::isfinite(terminal.cost.f));
  EXPECT_TRUE(terminal.cost.dfdx.allFinite());
}

TEST(WBMpcConstructionTest, theSwingFootCostsReadTheReferenceManagerAndTheShippedWeights) {
  WBMpcInterface* const interface = shippedG1WholeBodyMpc();
  ASSERT_NE(interface, nullptr);
  const scalar_t initTime = 0.0;
  const scalar_t finalTime = initTime + interface->mpcSettings().timeHorizon_;
  const vector_t& initState = interface->getInitialState();
  const vector_t zeroInput = vector_t::Zero(interface->getMpcRobotModel().getInputDim());
  ReferenceManagerInterface& referenceManager = *interface->getReferenceManagerPtr();
  const TargetTrajectories target({initTime}, {initState}, {zeroInput});
  referenceManager.setTargetTrajectories(target);
  referenceManager.preSolverRun(initTime, finalTime, initState, ModeNumber::STANCE);
  OptimalControlProblem problem(interface->getOptimalControlProblem());

  // The weights the shipped file gives the loader, which the cost carries as square roots among its parameters.
  const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml");
  ASSERT_FALSE(taskFile.empty());
  const VECTOR18_T<scalar_t> weights =
      EndEffectorDynamicsWeights::getWeights(taskFile, "task_space_foot_cost_weights.", /*verbose=*/false).toVector();

  vector_t movedState = initState;
  movedState.head(3).setConstant(0.3);
  const TargetTrajectories moved({initTime - 1.0, initTime + 1.0}, {movedState, 2.0 * movedState}, {zeroInput, zeroInput});
  for (const std::string& footName : interface->modelSettings().contactNames) {
    const EndEffectorDynamicsFootCost& cost =
        problem.costPtr->get<EndEffectorDynamicsFootCost>(absl::StrCat(footName, "_TaskSpaceTrackingCost"));
    const vector_t parameters = cost.getParameters(initTime, target, *problem.preComputationPtr);
    ASSERT_EQ(parameters.size(), 37) << footName;
    const vector_t sqrtWeights = parameters.segment(18, 18);
    EXPECT_TRUE(sqrtWeights.cwiseProduct(sqrtWeights).isApprox(weights, /*prec=*/1e-14)) << footName;
    // The solver's target is not read: another target, or none at all, gives the same parameters.
    EXPECT_TRUE(cost.getParameters(initTime, moved, *problem.preComputationPtr) == parameters) << footName;
    vector_t emptyTargetParameters;
    ASSERT_NO_THROW(emptyTargetParameters = cost.getParameters(initTime, TargetTrajectories(), *problem.preComputationPtr)) << footName;
    EXPECT_TRUE(emptyTargetParameters == parameters) << footName;
  }
}

}  // namespace ocs2::humanoid
