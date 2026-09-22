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
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <regex>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {

namespace {

/**
 * End-to-end test of the locomotion-heuristic layer on the DRC Atlas model: CentroidalMpcInterface::Create() reads the
 * `locomotion_heuristics` block of a task file, derives the model constants, builds the layer and installs it on the
 * reference manager, and the three references the layer shapes are read back through the same accessors the cost terms
 * use.
 *
 * The property this file exists to pin is the PARITY one: with the block as every robot ships it - all three lists
 * empty - the state and contact-force references are bit for bit what they were before the layer existed. Everything
 * else here is a consequence of turning one name on.
 *
 * Building an interface compiles the CppAD libraries, so the two interfaces are built once for the whole suite rather
 * than once per case. The error paths are covered without an interface in
 * humanoid_nmpc/humanoid_common_mpc/test/testLocomotionHeuristics.cpp; only the two that have to travel through the
 * task file are repeated here.
 */
class LocomotionHeuristicIntegrationTest : public ::testing::Test {
 protected:
  static std::string configDir() { return ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc"); }
  static std::string urdfFile() { return ament_index_cpp::get_package_share_directory("drc_atlas_description") + "/urdf/atlas.urdf"; }
  static std::string referenceFile() { return configDir() + "/config/command/reference.yaml"; }
  static std::string shippedTaskFile() { return configDir() + "/config/mpc/task.yaml"; }

  static std::string readFile(const std::string& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  }

  /** The shipped task file with its three `locomotion_heuristics` lists replaced by `replacement`, in a temp file. */
  static std::string taskFileWithLists(const std::string& name, const std::string& replacement) {
    std::string content = readFile(shippedTaskFile());
    const std::regex lists("  base_pose: \\[\\]\n(.|\n)*?  wrench: \\[\\]\n");
    const std::string replaced = std::regex_replace(content, lists, replacement);
    EXPECT_NE(replaced, content) << "the shipped block was not found, so this task file would test nothing";
    // The foothold seam is only reachable with the online contact planner off, which is how every robot ships.
    content = std::regex_replace(replaced, std::regex("useContactPlanning: *(true|false)"), "useContactPlanning: false");
    const std::string directory = (std::filesystem::path(testing::TempDir()) / "locomotion_heuristics_integration").string();
    std::filesystem::create_directories(directory);
    const std::string file = (std::filesystem::path(directory) / name).string();
    std::ofstream out(file);
    out << content;
    return file;
  }

  static void SetUpTestSuite() {
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> shipped =
        CentroidalMpcInterface::Create(shippedTaskFile(), urdfFile(), referenceFile());
    ASSERT_TRUE(shipped.ok()) << shipped.status().message();
    shipped_ = shipped->release();

    // One interface with a heuristic of each shaped channel listed, to check that a name in the file reaches the
    // reference. `hip_centered_stepping` is also the one heuristic that consumes a derived model constant.
    const std::string taskFile = taskFileWithLists("heuristics_on.yaml", R"(  base_pose:
    - orientation_compensation
  foothold:
    - hip_centered_stepping
  wrench:
    - impulse_scaling
)");
    absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> enabled = CentroidalMpcInterface::Create(taskFile, urdfFile(), referenceFile());
    ASSERT_TRUE(enabled.ok()) << enabled.status().message();
    enabled_ = enabled->release();
  }

  static void TearDownTestSuite() {
    delete shipped_;
    delete enabled_;
    shipped_ = nullptr;
    enabled_ = nullptr;
  }

  /** A target trajectory holding `targetState` over the horizon, which is what the cost terms are handed. */
  static TargetTrajectories targetOf(const CentroidalMpcInterface& interface, const vector_t& targetState) {
    const vector_t zeroInput = vector_t::Zero(interface.getCentroidalModelInfo().inputDim);
    return TargetTrajectories({0.0, 1.0}, {targetState, targetState}, {zeroInput, zeroInput});
  }

  static CentroidalMpcInterface* shipped_;
  static CentroidalMpcInterface* enabled_;
};

CentroidalMpcInterface* LocomotionHeuristicIntegrationTest::shipped_ = nullptr;
CentroidalMpcInterface* LocomotionHeuristicIntegrationTest::enabled_ = nullptr;

}  // namespace

TEST_F(LocomotionHeuristicIntegrationTest, ShippedTaskFileBuildsAnEmptyLayerAndInstallsIt) {
  // Every robot in this repository ships the block with all three lists empty, and this is the test that says so: the
  // file on disk, unmodified, must produce a layer that changes nothing.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = shipped_->getLocomotionHeuristicLayerPtr();
  ASSERT_NE(layer, nullptr);
  EXPECT_TRUE(layer->empty()) << "the shipped drc_atlas task file lists a heuristic:\n" << layer->summary();
  EXPECT_FALSE(layer->wrenchNeedsWorldFrame());
  EXPECT_EQ(shipped_->getSwitchedModelReferenceManagerPtr()->getLocomotionHeuristicLayer(), layer)
      << "the interface must install the layer it built on the reference manager";
}

TEST_F(LocomotionHeuristicIntegrationTest, EmptyListsLeaveTheStateReferenceUnchanged) {
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = shipped_->getSwitchedModelReferenceManagerPtr();
  const vector_t state = shipped_->getInitialState();
  const vector_t desiredState = referenceManager->getDesiredState(targetOf(*shipped_, state), state, 0.0);

  // Roll and pitch in particular: those are the two channels the base-pose heuristics write, and they are a hard zero
  // in this reference today.
  const vector6_t basePose = shipped_->getMpcRobotModel().getBasePose(desiredState);
  const vector6_t targetBasePose = shipped_->getMpcRobotModel().getBasePose(state);
  EXPECT_NEAR(basePose(2), targetBasePose(2), 1e-12) << "height";
  EXPECT_NEAR(basePose(4), targetBasePose(4), 1e-12) << "pitch";
  EXPECT_NEAR(basePose(5), targetBasePose(5), 1e-12) << "roll";
}

TEST_F(LocomotionHeuristicIntegrationTest, EmptyListsLeaveTheContactForceReferenceUnchanged) {
  // getDesiredInput() is new plumbing that both input costs now route through, so the parity that matters is against
  // the call they used to make for themselves.
  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = shipped_->getSwitchedModelReferenceManagerPtr();
  const vector_t state = shipped_->getInitialState();
  const vector_t desiredInput = referenceManager->getDesiredInput(targetOf(*shipped_, state), state, 0.0);
  const vector_t weightCompensation =
      weightCompensatingInput(shipped_->getPinocchioInterface(), referenceManager->getContactFlags(0.0), shipped_->getMpcRobotModel());
  EXPECT_TRUE(desiredInput.isApprox(weightCompensation, 1e-12))
      << "getDesiredInput must reproduce weightCompensatingInput when no wrench heuristic is listed";
}

TEST_F(LocomotionHeuristicIntegrationTest, AListedHeuristicWithShippedCoefficientsIsStillInert) {
  // The list and the coefficients are independent switches: every fitted coefficient ships at zero, so turning a name
  // on and finding its number are two separate experiments. This is what makes that claim true end to end.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  ASSERT_FALSE(layer->empty()) << "the test's task file did not take effect";

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = enabled_->getSwitchedModelReferenceManagerPtr();
  const vector_t state = enabled_->getInitialState();
  vector_t targetState = state;
  enabled_->getMpcRobotModel().setBaseComLinearVelocity(targetState, vector3_t(1.0, 0.0, 0.0));
  const vector_t desiredState = referenceManager->getDesiredState(targetOf(*enabled_, targetState), state, 0.0);
  const vector6_t basePose = enabled_->getMpcRobotModel().getBasePose(desiredState);
  EXPECT_NEAR(basePose(4), 0.0, 1e-12) << "orientation_compensation is listed but its coefficients are zero";
}

TEST_F(LocomotionHeuristicIntegrationTest, AnEnabledBasePoseHeuristicReachesTheStateReference) {
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  // Through reconfigure(), which is also the hot-reload path the parameter updater drives.
  LocomotionHeuristicConfig config;
  config.formulation.basePose = {heuristic::kOrientationCompensation};
  config.formulation.foothold = {heuristic::kHipCenteredStepping};
  config.formulation.wrench = {heuristic::kImpulseScaling};
  config.orientationCompensation.pitchPerForwardVelocity = -0.1;
  ASSERT_TRUE(layer->reconfigure(config).ok());

  const std::shared_ptr<SwitchedModelReferenceManager> referenceManager = enabled_->getSwitchedModelReferenceManagerPtr();
  const vector_t state = enabled_->getInitialState();
  // A forward velocity command lives in the linear part of the normalised momentum channel of the target state.
  vector_t targetState = state;
  enabled_->getMpcRobotModel().setBaseComLinearVelocity(targetState, vector3_t(1.0, 0.0, 0.0));
  const vector_t desiredState = referenceManager->getDesiredState(targetOf(*enabled_, targetState), state, 0.0);

  // Index 4 is PITCH and index 5 is ROLL: the base pose is Euler ZYX with yaw FIRST.
  const vector6_t basePose = enabled_->getMpcRobotModel().getBasePose(desiredState);
  EXPECT_NEAR(basePose(4), -0.1, 1e-9) << "the pitch reference must carry the heuristic's offset";
  EXPECT_NEAR(basePose(5), 0.0, 1e-12) << "roll follows the lateral command, which is zero here";

  // Restore the shipped (all-zero) coefficients, so that the order the cases run in cannot matter.
  LocomotionHeuristicConfig shippedCoefficients;
  shippedCoefficients.formulation = config.formulation;
  ASSERT_TRUE(layer->reconfigure(shippedCoefficients).ok());
}

TEST_F(LocomotionHeuristicIntegrationTest, ModelParametersComeFromTheRobotAndNotFromTheTaskFile) {
  // r_hip and the nominal CoM height are derived from the URDF, because a number that has to agree with the model must
  // not be maintained by hand beside it. Checked through the one heuristic that consumes r_hip.
  const std::shared_ptr<LocomotionHeuristicLayer>& layer = enabled_->getLocomotionHeuristicLayerPtr();
  EXPECT_TRUE(layer->footholdMovesAnchor());

  FootholdHeuristicContext left;
  left.contactIndex = CONTACT_LEFT_INDEX;
  left.side = 1.0;
  FootholdHeuristicContext right;
  right.contactIndex = CONTACT_RIGHT_INDEX;
  right.side = -1.0;
  const vector2_t leftHip = layer->footholdOffset(left);
  const vector2_t rightHip = layer->footholdOffset(right);
  // Atlas's hips are about 0.223 m apart, so each is roughly 0.11 m to its own side; the exact number is the URDF's.
  EXPECT_GT(leftHip.y(), 0.05) << "the left hip must be to the left of the base";
  EXPECT_LT(rightHip.y(), -0.05) << "the right hip must be to the right of the base";
  EXPECT_NEAR(leftHip.y(), -rightHip.y(), 1e-6) << "a symmetric robot has symmetric hips";
}

TEST_F(LocomotionHeuristicIntegrationTest, UnknownNameInTheTaskFileIsRejectedWithAStatus) {
  const std::string taskFile = taskFileWithLists("unknown_name.yaml", R"(  base_pose:
    - orientation_compenstaion
  foothold: []
  wrench: []
)");
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(taskFile, urdfFile(), referenceFile());
  ASSERT_FALSE(interface.ok());
  EXPECT_NE(std::string(interface.status().message()).find("orientation_compensation"), std::string::npos) << interface.status().message();
}

TEST_F(LocomotionHeuristicIntegrationTest, FootholdHeuristicUnderContactPlanningIsRejectedAtStartUp) {
  // A silent no-op is the failure mode this rejection exists to prevent, so it must be loud and at start-up.
  std::string content = readFile(shippedTaskFile());
  content = std::regex_replace(content, std::regex("  base_pose: \\[\\]\n(.|\n)*?  wrench: \\[\\]\n"),
                               "  base_pose: []\n  foothold:\n    - capture_point\n  wrench: []\n");
  content = std::regex_replace(content, std::regex("useContactPlanning: *(true|false)"), "useContactPlanning: true");
  const std::string directory = (std::filesystem::path(testing::TempDir()) / "locomotion_heuristics_integration").string();
  std::filesystem::create_directories(directory);
  const std::string file = (std::filesystem::path(directory) / "foothold_with_planner.yaml").string();
  {
    std::ofstream out(file);
    out << content;
  }
  const absl::StatusOr<std::unique_ptr<CentroidalMpcInterface>> interface =
      CentroidalMpcInterface::Create(file, urdfFile(), referenceFile());
  ASSERT_FALSE(interface.ok());
  EXPECT_NE(std::string(interface.status().message()).find("useContactPlanning"), std::string::npos) << interface.status().message();
}

}  // namespace ocs2::humanoid
