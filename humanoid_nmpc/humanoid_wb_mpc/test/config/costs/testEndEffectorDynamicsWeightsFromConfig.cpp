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

// The conversions of the whole-body MPC's task-space weights (EndEffectorDynamicsWeightsFromConfig.h): every one of
// the eighteen weights lands in its own entry, an absent weight is 0, and what the whole-body foot cost cannot use is
// refused by its path.

#include <limits>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_config/task_space_cost_config.nproto.h"
#include "humanoid_mpc_config/task_space_weights.nproto.h"
#include "humanoid_wb_mpc/config/costs/EndEffectorDynamicsWeightsFromConfig.h"
#include "humanoid_wb_mpc/cost/EndEffectorDynamicsCostHelpers.h"

namespace ocs2::humanoid {
namespace {

/** Eighteen distinct weights, in the order of EndEffectorDynamicsWeights::toVector(). */
mpc_config::TaskSpaceWeights distinctWeights() {
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
  weights.lin_acceleration_x = 13.0;
  weights.lin_acceleration_y = 14.0;
  weights.lin_acceleration_z = 15.0;
  weights.ang_acceleration_x = 16.0;
  weights.ang_acceleration_y = 17.0;
  weights.ang_acceleration_z = 18.0;
  return weights;
}

TEST(EndEffectorDynamicsWeightsFromConfigTest, AnEmptyBlockIsZeroEverywhere) {
  const absl::StatusOr<EndEffectorDynamicsWeights> weights =
      endEffectorDynamicsWeightsFromConfig(mpc_config::TaskSpaceWeights{}, "task_space_foot_cost.weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  EXPECT_TRUE(weights->toVector().isZero()) << weights->toVector().transpose();
}

TEST(EndEffectorDynamicsWeightsFromConfigTest, EachWeightLandsInItsOwnEntry) {
  const absl::StatusOr<EndEffectorDynamicsWeights> weights = endEffectorDynamicsWeightsFromConfig(distinctWeights(), "weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  for (Eigen::Index index = 0; index < 18; ++index) {
    EXPECT_EQ(weights->toVector()(index), static_cast<scalar_t>(index + 1)) << index;
  }
}

TEST(EndEffectorDynamicsWeightsFromConfigTest, ANonFiniteWeightIsRefusedByItsPath) {
  mpc_config::TaskSpaceWeights weights = distinctWeights();
  weights.lin_acceleration_z = std::numeric_limits<double>::infinity();
  const absl::StatusOr<EndEffectorDynamicsWeights> refused = endEffectorDynamicsWeightsFromConfig(weights, "task_space_foot_cost.weights");
  EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "task_space_foot_cost.weights.lin_acceleration_z")) << refused.status();
}

/** distinctWeights() with the weights the whole-body foot cost multiplies by a zero error at 0. */
mpc_config::TaskSpaceWeights distinctLiveWeights() {
  mpc_config::TaskSpaceWeights weights = distinctWeights();
  weights.pos_x = 0.0;
  weights.pos_y = 0.0;
  weights.pos_z = 0.0;
  return weights;
}

TEST(WholeBodyFootCostWeightsFromConfigTest, AWeightOfAnErrorThatIsZeroByConstructionIsRefusedByItsPath) {
  struct Inert {
    const char* absl_nonnull name;
    double mpc_config::TaskSpaceWeights::*absl_nonnull field;
  };
  const std::vector<Inert> inert = {{.name = "pos_x", .field = &mpc_config::TaskSpaceWeights::pos_x},
                                    {.name = "pos_y", .field = &mpc_config::TaskSpaceWeights::pos_y},
                                    {.name = "pos_z", .field = &mpc_config::TaskSpaceWeights::pos_z}};
  mpc_config::TaskSpaceCostConfig footCost;
  footCost.weights = distinctLiveWeights();
  ASSERT_TRUE(wholeBodyFootCostWeightsFromConfig(footCost).ok()) << "every other weight is live";
  for (const Inert& weight : inert) {
    mpc_config::TaskSpaceCostConfig edited = footCost;
    edited.weights.*weight.field = 1.0;
    const absl::Status refused = wholeBodyFootCostWeightsFromConfig(edited).status();
    EXPECT_EQ(refused.code(), absl::StatusCode::kInvalidArgument) << weight.name;
    EXPECT_TRUE(absl::StrContains(refused.message(), absl::StrCat("task_space_foot_cost.weights.", weight.name, " is 1"))) << refused;
    // The weights' own conversion, which the other task-space costs use, keeps it.
    EXPECT_TRUE(endEffectorDynamicsWeightsFromConfig(edited.weights, "weights").ok()) << weight.name;
  }
}

TEST(WholeBodyFootCostWeightsFromConfigTest, RefusesWhatTheWholeBodyFootCostDoesNotHave) {
  mpc_config::TaskSpaceCostConfig footCost;
  footCost.weights = distinctLiveWeights();
  const absl::StatusOr<EndEffectorDynamicsWeights> weights = wholeBodyFootCostWeightsFromConfig(footCost);
  ASSERT_TRUE(weights.ok()) << weights.status();
  EXPECT_EQ(weights->contactAngularAccelerationErrorWeight.z(), 18.0);

  footCost.active_phases = "swing_and_stance";
  EXPECT_TRUE(absl::StrContains(wholeBodyFootCostWeightsFromConfig(footCost).status().message(), "task_space_foot_cost.active_phases"));
  footCost.active_phases = "stance";
  EXPECT_TRUE(
      absl::StrContains(wholeBodyFootCostWeightsFromConfig(footCost).status().message(), "valid names are: swing, swing_and_stance"));
  footCost.active_phases = "swing";
  EXPECT_TRUE(wholeBodyFootCostWeightsFromConfig(footCost).ok());
  footCost.name = "torso";
  EXPECT_EQ(wholeBodyFootCostWeightsFromConfig(footCost).status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid
