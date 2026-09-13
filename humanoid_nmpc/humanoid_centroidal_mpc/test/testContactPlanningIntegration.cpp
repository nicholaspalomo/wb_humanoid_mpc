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

#include <fstream>
#include <memory>
#include <regex>
#include <string>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"

namespace ocs2::humanoid {

/**
 * End-to-end test of the contact planning path on the DRC Atlas model: the interface builds the planning reference
 * manager and module from a task file with useContactPlanning: true, the module plans synchronously from the interface's
 * initial state, and the reference manager turns the plan into a mode schedule and swing-foot references.
 */
class ContactPlanningIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
    const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");
    const std::string taskFile = configDir + "/config/mpc/task.yaml";
    referenceFile_ = configDir + "/config/command/reference.yaml";
    urdfFile_ = descriptionDir + "/urdf/atlas.urdf";

    // Temporary task file with contact planning on, planned synchronously so that the test controls the timing.
    std::ifstream in(taskFile);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    content = std::regex_replace(content, std::regex("useContactPlanning: *(true|false)"), "useContactPlanning: true");
    content = std::regex_replace(content, std::regex("runInBackgroundThread: *(true|false)"), "runInBackgroundThread: false");
    content = std::regex_replace(content, std::regex("maxSolveTime: *[0-9.]+"), "maxSolveTime: 5.0");
    content = std::regex_replace(content, std::regex("maxBranchAndBoundNodes: *[0-9]+"), "maxBranchAndBoundNodes: 2000");
    tmpTaskFile_ = testing::TempDir() + "/contact_planning_task.yaml";
    std::ofstream out(tmpTaskFile_);
    out << content;
    out.close();

    auto created = CentroidalMpcInterface::Create(tmpTaskFile_, urdfFile_, referenceFile_);
    ASSERT_TRUE(created.ok()) << created.status().message();
    interface_ = *std::move(created);
  }

  void TearDown() override { std::remove(tmpTaskFile_.c_str()); }

  std::string referenceFile_, urdfFile_, tmpTaskFile_;
  std::unique_ptr<CentroidalMpcInterface> interface_;
};

TEST_F(ContactPlanningIntegrationTest, PlansStandingAndWalkingSchedules) {
  ASSERT_TRUE(interface_->usesContactPlanning());
  auto module = interface_->getContactPlannerModulePtr();
  ASSERT_NE(module, nullptr);
  auto referenceManager = std::dynamic_pointer_cast<ContactPlanningReferenceManager>(interface_->getSwitchedModelReferenceManagerPtr());
  ASSERT_NE(referenceManager, nullptr);
  EXPECT_TRUE(referenceManager->usesContactPlanning());

  const vector_t state = interface_->getInitialState();
  const scalar_t horizon = interface_->mpcSettings().timeHorizon_;
  const size_t inputDim = interface_->getEffectiveMpcRobotModel().getInputDim();

  // 1. No plan yet: the gait schedule (double support) is used.
  scalar_t t = 0.0;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  EXPECT_FALSE(referenceManager->hasActivePlan());
  EXPECT_TRUE(referenceManager->isInStancePhase(t + 0.5));
  EXPECT_FALSE(referenceManager->getSwingFootReference(0, t + 0.5).has_value());

  // 2. Standing command: the plan is all double support.
  vector_t standingTarget = vector_t::Zero(state.size());
  standingTarget.segment(6, 6) = state.segment(6, 6);
  referenceManager->setTargetTrajectories(TargetTrajectories({t}, {standingTarget}, {vector_t::Zero(inputDim)}));
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  const auto standingStats = module->getStatistics();
  EXPECT_TRUE(standingStats.lastPlanValid);
  t += 0.02;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  ASSERT_TRUE(referenceManager->hasActivePlan());
  for (scalar_t tau = t; tau < t + horizon; tau += 0.05) {
    EXPECT_TRUE(referenceManager->isInStancePhase(tau)) << "tau=" << tau;
  }

  // 3. Walking command: the plan contains single support phases and provides swing-foot references.
  vector_t walkingTarget = standingTarget;
  walkingTarget(0) = 0.4;  // commanded CoM velocity x
  referenceManager->setTargetTrajectories(TargetTrajectories({t}, {walkingTarget}, {vector_t::Zero(inputDim)}));
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);  // swaps in the target trajectories
  module->preSolverRun(t, t + horizon, state, *referenceManager);
  const auto walkingStats = module->getStatistics();
  ASSERT_TRUE(walkingStats.lastPlanValid);
  std::cout << "walking plan: " << walkingStats.lastSolveTime * 1e3 << " ms, " << walkingStats.lastNumBranchAndBoundNodes
            << " relaxations, optimal=" << walkingStats.lastOptimal << std::endl;
  t += 0.02;
  referenceManager->preSolverRun(t, t + horizon, state, ModeNumber::STANCE);
  const ModeSchedule& schedule = referenceManager->getModeSchedule();
  std::cout << "mode schedule: " << schedule;

  bool foundSwing = false;
  for (scalar_t tau = t; tau < t + horizon; tau += 0.02) {
    const contact_flag_t contacts = referenceManager->getContactFlags(tau);
    ASSERT_TRUE(contacts[0] || contacts[1]) << "no flight phase allowed at tau=" << tau;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      const auto reference = referenceManager->getSwingFootReference(foot, tau);
      if (contacts[foot]) {
        EXPECT_FALSE(reference.has_value());
      } else {
        foundSwing = true;
        ASSERT_TRUE(reference.has_value()) << "swing foot " << foot << " at tau=" << tau << " has no reference";
        EXPECT_TRUE(reference->position.allFinite());
        EXPECT_TRUE(reference->linearVelocity.allFinite());
        EXPECT_GE(reference->position(2), -1e-6);  // never below the (flat) ground
      }
    }
  }
  EXPECT_TRUE(foundSwing) << "a walking command must produce a swing phase within the horizon";
  // The committed window right after the current time keeps the previous (double support) schedule.
  EXPECT_TRUE(referenceManager->isInStancePhase(t + 0.01));

  // 4. A swing that has started must survive a later plan: advance into the first swing, command standing (which on
  //    its own would plan no steps) and check that the swing keeps its touch-down time.
  scalar_t firstLiftOff = -1.0, firstTouchDown = -1.0;
  size_t swingFoot = 0;
  for (size_t i = 0; i < schedule.eventTimes.size() && firstLiftOff < 0.0; ++i) {
    const contact_flag_t before = modeNumber2StanceLeg(schedule.modeSequence[i]);
    const contact_flag_t after = modeNumber2StanceLeg(schedule.modeSequence[i + 1]);
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (before[foot] && !after[foot] && schedule.eventTimes[i] > t) {
        firstLiftOff = schedule.eventTimes[i];
        swingFoot = foot;
        for (size_t j = i + 1; j < schedule.eventTimes.size(); ++j) {
          if (modeNumber2StanceLeg(schedule.modeSequence[j + 1])[foot]) {
            firstTouchDown = schedule.eventTimes[j];
            break;
          }
        }
      }
    }
  }
  ASSERT_GT(firstLiftOff, 0.0);
  ASSERT_GT(firstTouchDown, firstLiftOff);
  const scalar_t midSwing = 0.5 * (firstLiftOff + firstTouchDown);
  referenceManager->setTargetTrajectories(TargetTrajectories({midSwing}, {standingTarget}, {vector_t::Zero(inputDim)}));
  referenceManager->preSolverRun(midSwing, midSwing + horizon, state, ModeNumber::STANCE);
  EXPECT_GE(referenceManager->commitBoundary(midSwing), firstTouchDown - 1e-9);
  module->preSolverRun(midSwing, midSwing + horizon, state, *referenceManager);
  ASSERT_TRUE(module->getStatistics().lastPlanValid);
  referenceManager->preSolverRun(midSwing + 0.02, midSwing + 0.02 + horizon, state, ModeNumber::STANCE);
  EXPECT_FALSE(referenceManager->isInContact(midSwing + 0.02, swingFoot));
  EXPECT_FALSE(referenceManager->isInContact(firstTouchDown - 0.01, swingFoot)) << "the in-flight swing was cut short";
  EXPECT_TRUE(referenceManager->isInContact(firstTouchDown + 0.01, swingFoot)) << "the in-flight swing was extended";
  EXPECT_TRUE(referenceManager->getSwingFootReference(swingFoot, midSwing + 0.02).has_value());
}

}  // namespace ocs2::humanoid
