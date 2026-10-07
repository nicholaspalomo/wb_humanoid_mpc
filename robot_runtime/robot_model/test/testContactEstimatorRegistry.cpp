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

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "robot_model/AlwaysInContactEstimator.h"
#include "robot_model/ContactEstimatorRegistry.h"
#include "robot_model/RobotDescription.h"

namespace robot::model {
namespace {

class ContactEstimatorRegistryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tempDir_ = std::filesystem::temp_directory_path() / "robot_model_contact_estimator_registry_test";
    std::filesystem::create_directories(tempDir_);
    urdfPath_ = tempDir_ / "robot.urdf";
    std::ofstream urdfFile(urdfPath_);
    urdfFile << R"(<?xml version="1.0"?>
        <robot name="biped">
            <link name="base_link"/>
            <link name="foot_l"/>
            <link name="foot_r"/>
            <joint name="hip_l" type="revolute">
                <parent link="base_link"/>
                <child link="foot_l"/>
                <axis xyz="0 1 0"/>
                <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
            </joint>
            <joint name="hip_r" type="revolute">
                <parent link="base_link"/>
                <child link="foot_r"/>
                <axis xyz="0 1 0"/>
                <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
            </joint>
        </robot>)";
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(tempDir_, ec);
  }

  std::filesystem::path tempDir_;
  std::filesystem::path urdfPath_;
};

TEST_F(ContactEstimatorRegistryTest, registersTheRobotModelEstimatorsAndCreatesThemByName) {
  ContactEstimatorRegistry registry;
  const std::vector<ContactEstimatorRegistry::Entry> entries = registry.available();
  ASSERT_EQ(entries.size(), 2u);
  EXPECT_EQ(entries[0].name, "robot_state");
  EXPECT_EQ(entries[1].name, "always_in_contact");
  EXPECT_FALSE(entries[0].description.empty());
  EXPECT_EQ(registry.availableNames(), "robot_state, always_in_contact");

  const absl::StatusOr<RobotDescription> description = RobotDescription::Create(urdfPath_.string());
  ASSERT_TRUE(description.ok()) << description.status();
  RobotState state(*description, /*contactSize=*/2);
  state.setContactFlag(/*index=*/0, /*contactFlag=*/false);
  EXPECT_EQ(estimateContactFlags(*registry.create("robot_state"), state), (std::vector<bool>{false, true}));
  EXPECT_EQ(estimateContactFlags(*registry.create("always_in_contact"), state), (std::vector<bool>{true, true}));
  EXPECT_EQ(registry.create("always_in_contact")->getName(), "AlwaysInContactEstimator");
}

TEST_F(ContactEstimatorRegistryTest, namesAreMatchedTrimmedAndCaseInsensitively) {
  ContactEstimatorRegistry registry;
  EXPECT_EQ(ContactEstimatorRegistry::canonicalName("  Robot_State \n"), "robot_state");
  EXPECT_EQ(ContactEstimatorRegistry::canonicalName("   "), "");
  EXPECT_TRUE(registry.has(" ROBOT_STATE "));
  EXPECT_EQ(registry.create("Always_In_Contact")->getName(), "AlwaysInContactEstimator");
}

TEST(ContactEstimatorRegistryDeathTest, creatingAnUnregisteredNameIsAProgrammingErrorListingTheAvailableOnes) {
  const ContactEstimatorRegistry registry;
  EXPECT_FALSE(registry.has("cheater_sim")) << "a name from a file is validated with has() first";
  EXPECT_DEATH(registry.create("cheater_sim"), "unknown contact estimator 'cheater_sim'; available: robot_state, always_in_contact");
}

TEST_F(ContactEstimatorRegistryTest, interfacesRegisterTheirOwnEstimators) {
  ContactEstimatorRegistry registry;
  registry.add("cheater_sim", "ground truth of a simulator", [] { return std::make_shared<AlwaysInContactEstimator>(); });
  EXPECT_TRUE(registry.has("cheater_sim"));
  EXPECT_EQ(registry.available().size(), 3u);
  EXPECT_EQ(registry.availableNames(), "robot_state, always_in_contact, cheater_sim");
}

TEST(ContactEstimatorRegistryDeathTest, aSecondRegistrationOrAnEstimatorWithoutNameOrFactoryIsAProgrammingError) {
  ContactEstimatorRegistry registry;
  registry.add("cheater_sim", "ground truth of a simulator", [] { return std::make_shared<AlwaysInContactEstimator>(); });
  EXPECT_DEATH(registry.add("Cheater_Sim", "again", [] { return std::make_shared<AlwaysInContactEstimator>(); }),
               "estimator 'cheater_sim' is already registered");
  EXPECT_DEATH(registry.add(" ", "nameless", [] { return std::make_shared<AlwaysInContactEstimator>(); }), "an estimator needs a name");
  EXPECT_DEATH(registry.add("no_factory", "factory-less", /*factory=*/nullptr), "estimator 'no_factory' has no factory");
}

}  // namespace
}  // namespace robot::model
