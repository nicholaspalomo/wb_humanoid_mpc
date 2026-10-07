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

// The conversions of the task-space costs (TaskSpaceCostFromConfig.h): every weight lands in its own entry, an absent
// weight is 0, and what the centroidal costs cannot use - a non-finite weight, an acceleration weight, the other use's
// fields - is refused by its path.

#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/costs/TaskSpaceCostFromConfig.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_mpc_config/task_space_cost_config.nproto.h"
#include "humanoid_mpc_config/task_space_weights.nproto.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

// The weights and the names are plain members: an absent one is its default.
static_assert(std::is_same_v<decltype(mpc_config::TaskSpaceWeights::pos_x), double>);
static_assert(std::is_same_v<decltype(mpc_config::TaskSpaceCostConfig::name), std::string>);
static_assert(std::is_same_v<decltype(mpc_config::TaskSpaceCostConfig::active_phases), std::string>);

/** Twelve distinct weights, in the order of EndEffectorKinematicsWeights::toVector(). */
mpc_config::TaskSpaceWeights distinctKinematicsWeights() {
  mpc_config::TaskSpaceWeights weights;
  weights.pos_x = 1.0;
  weights.pos_y = 2.0;
  weights.pos_z = 3.0;
  weights.orientation_x = 4.0;
  weights.orientation_y = 5.0;
  weights.orientation_z = 6.0;
  weights.lin_velocity_x = 7.0;
  weights.lin_velocity_y = 8.0;
  weights.lin_velocity_z = 9.0;
  weights.ang_velocity_x = 10.0;
  weights.ang_velocity_y = 11.0;
  weights.ang_velocity_z = 12.0;
  return weights;
}

TEST(EndEffectorKinematicsWeightsFromConfigTest, AnEmptyBlockIsZeroEverywhere) {
  const absl::StatusOr<EndEffectorKinematicsWeights> weights =
      endEffectorKinematicsWeightsFromConfig(mpc_config::TaskSpaceWeights{}, "task_space_foot_cost.weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  EXPECT_TRUE(weights->toVector().isZero()) << weights->toVector().transpose();
}

TEST(EndEffectorKinematicsWeightsFromConfigTest, EachWeightLandsInItsOwnEntry) {
  const absl::StatusOr<EndEffectorKinematicsWeights> weights =
      endEffectorKinematicsWeightsFromConfig(distinctKinematicsWeights(), "task_space_foot_cost.weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  for (Eigen::Index index = 0; index < 12; ++index) {
    EXPECT_EQ(weights->toVector()(index), static_cast<scalar_t>(index + 1)) << index;
  }
}

TEST(EndEffectorKinematicsWeightsFromConfigTest, ANonFiniteWeightIsRefusedByItsPath) {
  mpc_config::TaskSpaceWeights weights;
  weights.orientation_y = std::numeric_limits<double>::quiet_NaN();
  const absl::StatusOr<EndEffectorKinematicsWeights> refused =
      endEffectorKinematicsWeightsFromConfig(weights, "task_space_costs[name=torso].weights");
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "task_space_costs[name=torso].weights.orientation_y")) << refused.status();
}

TEST(EndEffectorKinematicsWeightsFromConfigTest, AnAccelerationWeightIsRefusedAsUnused) {
  mpc_config::TaskSpaceWeights weights;
  weights.ang_acceleration_z = 0.5;
  const absl::StatusOr<EndEffectorKinematicsWeights> refused =
      endEffectorKinematicsWeightsFromConfig(weights, "task_space_foot_cost.weights");
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "task_space_foot_cost.weights.ang_acceleration_z")) << refused.status();
  weights.ang_acceleration_z = -0.0;
  EXPECT_TRUE(endEffectorKinematicsWeightsFromConfig(weights, "task_space_foot_cost.weights").ok());
}

TEST(TaskSpaceFootCostFromConfigTest, CarriesTheWeightsAndTheActivePhases) {
  mpc_config::TaskSpaceCostConfig footCost;
  footCost.weights = distinctKinematicsWeights();
  footCost.active_phases = "swing_and_stance";
  const absl::StatusOr<TaskSpaceFootCostSettings> settings = taskSpaceFootCostFromConfig(footCost);
  ASSERT_TRUE(settings.ok()) << settings.status();
  EXPECT_TRUE(settings->activeInStance);
  EXPECT_EQ(settings->weights.toVector()(2), 3.0);
  const absl::StatusOr<TaskSpaceFootCostSettings> absent = taskSpaceFootCostFromConfig(mpc_config::TaskSpaceCostConfig{});
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_FALSE(absent->activeInStance) << "the default, a swing foot only";
}

TEST(TaskSpaceFootCostFromConfigTest, EachActivePhasesNameIsTheBooleanItReplaced) {
  // The parity of the retired active_in_stance: "swing" is false, "swing_and_stance" true.
  EXPECT_EQ(footCostPhasesNames(), (std::vector<std::string>{"swing", "swing_and_stance"}));
  for (const std::string& name : footCostPhasesNames()) {
    const absl::StatusOr<bool> activeInStance = footCostActiveInStanceFromName(name, "task_space_foot_cost.active_phases");
    ASSERT_TRUE(activeInStance.ok()) << activeInStance.status();
    EXPECT_EQ(*activeInStance, name == "swing_and_stance") << name;
  }
  mpc_config::TaskSpaceCostConfig footCost;
  footCost.active_phases = "stance";
  const absl::StatusOr<TaskSpaceFootCostSettings> refused = taskSpaceFootCostFromConfig(footCost);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "task_space_foot_cost.active_phases is 'stance'")) << refused.status();
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "valid names are: swing, swing_and_stance")) << refused.status();
  const absl::StatusOr<humanoid_mpc_config::TaskFile> retired =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>("task_space_foot_cost {\n  activeInStance: false\n}\n", "task.textproto");
  EXPECT_TRUE(absl::StrContains(retired.status().message(), "task.textproto:2:3: 'activeInStance' is retired")) << retired.status();
  EXPECT_TRUE(absl::StrContains(retired.status().message(), "active_phases")) << retired.status();
}

TEST(TaskSpaceFootCostFromConfigTest, RefusesTheFieldsOfTheNamedCosts) {
  mpc_config::TaskSpaceCostConfig footCost;
  footCost.link_name = "torso";
  const absl::StatusOr<TaskSpaceFootCostSettings> refused = taskSpaceFootCostFromConfig(footCost);
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "task_space_foot_cost")) << refused.status();
}

/** A named link cost. */
mpc_config::TaskSpaceCostConfig linkCost(const std::string& name, const std::string& linkName) {
  mpc_config::TaskSpaceCostConfig cost;
  cost.name = name;
  cost.link_name = linkName;
  return cost;
}

TEST(TaskSpaceLinkCostsFromConfigTest, KeepsTheFileOrder) {
  std::vector<mpc_config::TaskSpaceCostConfig> costs = {linkCost("torso", "utorso"), linkCost("pelvis", "pelvis")};
  costs[1].weights.lin_velocity_z = 2.0;
  const absl::StatusOr<std::vector<TaskSpaceLinkCostSettings>> settings = taskSpaceLinkCostsFromConfig(costs);
  ASSERT_TRUE(settings.ok()) << settings.status();
  ASSERT_EQ(settings->size(), 2U);
  EXPECT_EQ((*settings)[0].name, "torso");
  EXPECT_EQ((*settings)[0].linkName, "utorso");
  EXPECT_EQ((*settings)[1].name, "pelvis");
  EXPECT_EQ((*settings)[1].weights.contactLinearVelocityErrorWeight.z(), 2.0);
  EXPECT_TRUE(taskSpaceLinkCostsFromConfig({})->empty());
}

TEST(TaskSpaceLinkCostsFromConfigTest, RefusesWhatACostCannotBeBuiltFrom) {
  const std::vector<std::vector<mpc_config::TaskSpaceCostConfig>> refusedLists = {
      {linkCost(/*name=*/"", "utorso")},
      {linkCost("torso", /*linkName=*/"")},
      {linkCost("torso", "utorso"), linkCost("torso", "pelvis")},
  };
  for (const std::vector<mpc_config::TaskSpaceCostConfig>& costs : refusedLists) {
    EXPECT_EQ(taskSpaceLinkCostsFromConfig(costs).status().code(), absl::StatusCode::kInvalidArgument) << costs.size();
  }
  mpc_config::TaskSpaceCostConfig inStance = linkCost("torso", "utorso");
  inStance.active_phases = "swing_and_stance";
  const absl::StatusOr<std::vector<TaskSpaceLinkCostSettings>> refused = taskSpaceLinkCostsFromConfig({inStance});
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "task_space_costs[name=torso] sets active_phases \"swing_and_stance\""))
      << refused.status();
}

TEST(TaskSpaceCostConfigTest, AWeightOutsideTheWeightsBlockIsAnsweredWithTheLayout) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> refused =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>("task_space_foot_cost {\n  pos_x: 1\n}\n", "task.textproto");
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "task.textproto:2:3:")) << refused.status();
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "The weights are a block of their own")) << refused.status();
  EXPECT_TRUE(
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>("task_space_foot_cost { weights { pos_x: 1 } }\n", "task.textproto").ok());
}

}  // namespace
}  // namespace ocs2::humanoid
