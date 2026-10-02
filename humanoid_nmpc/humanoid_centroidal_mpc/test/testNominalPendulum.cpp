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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/misc/PropertyTree.h>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningModelParameters.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

namespace ocs2::humanoid {
namespace {

/** A centroidal MPC package, located in the runfiles by its path in the repository. */
struct CentroidalRobot {
  const char* name;
  const char* mpcDirectory;  // holds config/mpc/task.yaml and config/command/reference.yaml
  const char* urdf;
};

void PrintTo(const CentroidalRobot& robot, std::ostream* os) {
  *os << robot.name;
}

// Every centroidal MPC package in robot_models, as tools/locomotion_heuristics/derive_parameters.py lists them: the
// shipped derived numbers of each are checked against this derivation. The BUILD target's `data` lists the same.
// LINT.IfChange(nominal_pendulum_robots)
constexpr CentroidalRobot kCentroidalRobots[] = {
    {"drc_atlas", "robot_models/drc_atlas/drc_atlas_centroidal_mpc", "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"},
    {"engineai_sa01", "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc",
     "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf"},
    {"unitree_g1", "robot_models/unitree_g1/g1_centroidal_mpc", "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"},
    {"unitree_r1", "robot_models/unitree_r1/unitree_r1_centroidal_mpc", "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf"},
};
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/humanoid_centroidal_mpc/BUILD.bazel:nominal_pendulum_data, //tools/locomotion_heuristics/derive_parameters.py:derive_parameters_robots)
// clang-format on

/** The absolute path of a data file of this test, or empty when the runfiles do not contain it. */
std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
    roots.emplace_back(std::filesystem::path(srcDir) / "wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

/** An optional scalar of a YAML tree, 0 when absent. */
scalar_t valueOr0(const PropertyTree& tree, const std::string& key) {
  const std::optional<scalar_t> value = tree.getOptional<scalar_t>(key);
  return value.has_value() ? *value : 0.0;
}

/** The rounding derive_parameters.py prints a derived number with (four decimals). */
constexpr scalar_t kRounding4 = 0.5e-4 + 1e-12;

class NominalPendulum : public ::testing::TestWithParam<CentroidalRobot> {
 protected:
  void SetUp() override {
    taskFile_ = runfilePath(absl::StrCat(GetParam().mpcDirectory, "/config/mpc/task.yaml"));
    referenceFile_ = runfilePath(absl::StrCat(GetParam().mpcDirectory, "/config/command/reference.yaml"));
    urdfFile_ = runfilePath(GetParam().urdf);
    ASSERT_FALSE(taskFile_.empty()) << GetParam().name << ": task.yaml is not in the runfiles; add its package to `data`";
    ASSERT_FALSE(referenceFile_.empty()) << GetParam().name << ": reference.yaml is not in the runfiles";
    ASSERT_FALSE(urdfFile_.empty()) << GetParam().name << ": the URDF is not in the runfiles";

    // The robot model exactly as CentroidalMpcInterface builds it, without the problem.
    modelSettings_ = std::make_unique<ModelSettings>(taskFile_, urdfFile_, "testNominalPendulum_", /*verbose=*/false);
    pinocchioInterface_ = std::make_unique<PinocchioInterface>(
        createCustomPinocchioInterface(taskFile_, urdfFile_, *modelSettings_, /*scaleTotalMass=*/false));
    info_ = centroidal_model::createCentroidalModelInfo(
        *pinocchioInterface_, centroidal_model::loadCentroidalType(taskFile_),
        centroidal_model::loadDefaultJointState(pinocchioInterface_->getModel().nq - 6, referenceFile_), modelSettings_->contactNames3DoF,
        modelSettings_->contactNames6DoF);
    robotModel_ = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(*modelSettings_, *pinocchioInterface_, info_);
    initialState_ = vector_t::Zero(info_.stateDim);
    loadData::loadEigenMatrix(taskFile_, "initialState", initialState_);
    loadData::readPropertyTree(taskFile_, taskTree_);
    // The planner's block: in contact_planning.yaml beside the task file, or inside the task file itself.
    const std::string planningFile = resolveContactPlanningConfigFile(taskFile_);
    if (planningFile == taskFile_) {
      planningTree_ = taskTree_;
    } else {
      loadData::readPropertyTree(planningFile, planningTree_);
    }
  }

  /**
   * The oracle, computed here from Pinocchio directly - forward kinematics, the frames looked up by name, the center of
   * mass of the whole model - rather than through computeComHeightAboveFeet().
   */
  scalar_t independentComHeight() const {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    const pinocchio::ModelTpl<scalar_t>& model = pinocchio.getModel();
    pinocchio::DataTpl<scalar_t>& data = pinocchio.getData();
    const vector_t q = robotModel_->getGeneralizedCoordinates(initialState_);
    pinocchio::forwardKinematics(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    const scalar_t comZ = pinocchio::centerOfMass(model, data, q)(2);
    scalar_t footZ = 0.0;
    for (const std::string& frame : modelSettings_->contactNames) {
      footZ += data.oMf[model.getFrameId(frame)].translation()(2) / static_cast<scalar_t>(modelSettings_->contactNames.size());
    }
    return comZ - footZ;
  }

  scalar_t sharedComHeight() const {
    PinocchioInterface pinocchio(*pinocchioInterface_);
    return computeComHeightAboveFeet(robotModel_->getGeneralizedCoordinates(initialState_), pinocchio, *robotModel_);
  }

  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo info_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> robotModel_;
  vector_t initialState_;
  PropertyTree taskTree_;
  PropertyTree planningTree_;
};

}  // namespace

/**
 * The one pendulum length: computeComHeightAboveFeet() is the model's center of mass above the mean height of its soles,
 * and every consumer of a nominal pendulum - the contact planner's model parameters, the locomotion heuristics' nominal
 * CoM height, and the DCM terminal cost through CentroidalMpcInterface::getNominalComHeight() - derives exactly that
 * number, so a shared.comHeight and a dcm_terminal_cost.comHeight of 0 cannot resolve to two different pendulums.
 */
TEST_P(NominalPendulum, EveryConsumerDerivesTheModelsCenterOfMassAboveItsFeet) {
  const scalar_t oracle = independentComHeight();
  // A humanoid standing on its feet has its center of mass between half a meter and a meter and a half above them;
  // outside that, initialState or the contact frames are wrong and nothing below means anything.
  ASSERT_GT(oracle, 0.4) << GetParam().name;
  ASSERT_LT(oracle, 1.5) << GetParam().name;
  const scalar_t shared = sharedComHeight();
  EXPECT_NEAR(shared, oracle, 1e-12) << "computeComHeightAboveFeet() is not the center of mass above the mean sole height";
  // Positive control for the oracle: the pendulum is measured from the soles, not from the base.
  const scalar_t baseHeight = robotModel_->getBasePosition(initialState_)(2);
  EXPECT_GT(std::abs(shared - baseHeight), 1e-3) << "the check above cannot tell the pendulum from the base height";
  // And not from the world origin: the robot standing on a 0.25 m box has the same pendulum, while its center of mass
  // stands 0.25 m higher in the world. The shipped initial states put the soles near z = 0, where the two coincide.
  vector_t onABox = initialState_;
  const Eigen::Index baseZ = static_cast<Eigen::Index>(6 + 2);  // [momentum (6), base position (3), ...]
  onABox(baseZ) += 0.25;
  ASSERT_NEAR(robotModel_->getBasePosition(onABox)(2), baseHeight + 0.25, 1e-12) << "the state layout assumed above is wrong";
  PinocchioInterface onABoxPinocchio(*pinocchioInterface_);
  EXPECT_NEAR(computeComHeightAboveFeet(robotModel_->getGeneralizedCoordinates(onABox), onABoxPinocchio, *robotModel_), shared, 1e-9)
      << "the pendulum is not measured from the soles";

  PinocchioInterface forPlanner(*pinocchioInterface_);
  const ContactPlanningModelParameters planner =
      deriveContactPlanningModelParameters(forPlanner, *robotModel_, initialState_, modelSettings_->contactParentJointNames,
                                           ContactPlanningGroundParameters(), /*gravity=*/9.81, /*nominalStepWidth=*/0.25);
  EXPECT_NEAR(planner.comHeight, shared, 1e-12) << "the contact planner's shared.comHeight of 0 would resolve to another pendulum";

  PinocchioInterface forHeuristics(*pinocchioInterface_);
  const absl::StatusOr<LocomotionHeuristicModelParameters> heuristics =
      deriveLocomotionHeuristicModelParameters(forHeuristics, *robotModel_, initialState_);
  ASSERT_TRUE(heuristics.ok()) << heuristics.status();
  EXPECT_NEAR(heuristics->nominalComHeight, shared, 1e-12) << "the locomotion heuristics measure another pendulum";

  // And 0 in the planner's file means that pendulum; a positive value is kept as given.
  ContactPlanningConfig derived;
  derived.shared.comHeight = 0.0;
  planner.applyTo(derived);
  EXPECT_NEAR(derived.shared.comHeight, shared, 1e-12);
  ContactPlanningConfig explicitHeight;
  explicitHeight.shared.comHeight = shared + 0.1;
  planner.applyTo(explicitHeight);
  EXPECT_NEAR(explicitHeight.shared.comHeight, shared + 0.1, 1e-12);
}

/**
 * Every pendulum number a robot's files SHIP is the model's. A LIP height of 0 (dcm_terminal_cost.comHeight, the
 * planner's shared.comHeight) is resolved from the model at start-up; a positive one is an override and must stay within
 * a few percent of the model. The two heuristic numbers derive_parameters.py writes out from the pendulum -
 * capture_point.comHeightOverride and high_speed_turning's z / g - must equal this C++ derivation within the rounding the
 * script prints them with; tools/locomotion_heuristics/test_derive_parameters.py pins the same numbers against the Python
 * derivation, so where a robot ships a derived number the two derivations agree.
 */
TEST_P(NominalPendulum, EveryShippedPendulumNumberIsTheModels) {
  const scalar_t model = sharedComHeight();
  for (const std::string& key : {std::string("dcm_terminal_cost.comHeight"), std::string("contact_planning.shared.comHeight")}) {
    const scalar_t height = valueOr0(key == "dcm_terminal_cost.comHeight" ? taskTree_ : planningTree_, key);
    EXPECT_GE(height, 0.0) << key;
    if (height > 0.0) {
      EXPECT_LT(std::abs(height - model), 0.05 * model)
          << GetParam().name << ": " << key << " = " << height << " is an override of the model's " << model << " m by more than 5 %";
    }
  }
  const scalar_t override = valueOr0(taskTree_, "locomotion_heuristics.capture_point.comHeightOverride");
  if (override > 0.0) {
    EXPECT_NEAR(override, model, kRounding4) << GetParam().name << ": capture_point.comHeightOverride is not the model's pendulum";
  }
  const scalar_t gravity = taskTree_.get<scalar_t>("locomotion_heuristics.capture_point.gravity", /*defaultValue=*/9.81);
  for (const std::string& key : {std::string("forwardPerCrossTerm"), std::string("lateralPerCrossTerm")}) {
    const scalar_t lean = valueOr0(taskTree_, absl::StrCat("locomotion_heuristics.high_speed_turning.", key));
    if (lean > 0.0) {
      EXPECT_NEAR(lean, model / gravity, kRounding4) << GetParam().name << ": high_speed_turning." << key << " is not z_com / g";
    }
  }
  // Positive control: the 0.85 m Atlas used to ship for both LIP heights is refused by the 5 % check.
  if (std::string(GetParam().name) == "drc_atlas") {
    EXPECT_GT(std::abs(0.85 - model), 0.05 * model) << "the check above would not have caught the hand-set pendulum";
  }
}

INSTANTIATE_TEST_SUITE_P(CentroidalRobots,
                         NominalPendulum,
                         ::testing::ValuesIn(kCentroidalRobots),
                         [](const ::testing::TestParamInfo<CentroidalRobot>& info) { return std::string(info.param.name); });

}  // namespace ocs2::humanoid
