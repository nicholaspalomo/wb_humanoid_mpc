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

// The conversions of the centroidal MPC's own cost blocks (DcmTerminalCostFromConfig.h, IcpCostFromConfig.h): the
// schema's defaults are DcmTerminalCost::Config's, every field lands in its member, and what the cost refuses or cannot
// read is refused by its key.

#include "pinocchio/fwd.hpp"

#include <limits>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "gtest/gtest.h"

#include "humanoid_centroidal_mpc/config/costs/DcmTerminalCostFromConfig.h"
#include "humanoid_centroidal_mpc/config/costs/IcpCostFromConfig.h"
#include "humanoid_centroidal_mpc/cost/DcmTerminalCost.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_config/dcm_terminal_cost_config.nproto.h"
#include "humanoid_mpc_config/icp_cost_weights.nproto.h"

namespace ocs2::humanoid {
namespace {

TEST(DcmTerminalCostFromConfigTest, AnEmptyBlockIsTheCostsDefaultConfig) {
  const absl::StatusOr<DcmTerminalCost::Config> config = dcmTerminalCostConfigFromConfig(mpc_config::DcmTerminalCostConfig{});
  ASSERT_TRUE(config.ok()) << config.status();
  const DcmTerminalCost::Config defaults;
  EXPECT_EQ(config->comHeight, defaults.comHeight);
  EXPECT_EQ(config->gravity, defaults.gravity);
  EXPECT_EQ(config->weights, defaults.weights);
  EXPECT_EQ(config->velocityOffsetFactor, defaults.velocityOffsetFactor);
  EXPECT_EQ(config->supportBlendTime, defaults.supportBlendTime);
}

TEST(DcmTerminalCostFromConfigTest, AComHeightIsPositiveOrLeftOutForTheModels) {
  // Left out, it is the model's pendulum: the unset comHeight that DcmTerminalCost::resolveConfig() resolves.
  const absl::StatusOr<DcmTerminalCost::Config> absent = dcmTerminalCostConfigFromConfig(mpc_config::DcmTerminalCostConfig{});
  ASSERT_TRUE(absent.ok()) << absent.status();
  EXPECT_FALSE(absent->comHeight.has_value());
  // 0, which stood for the model's before the field was optional, is refused like any other height that is no height.
  for (const double height : {0.0, -0.9, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    mpc_config::DcmTerminalCostConfig block;
    block.com_height = height;
    const absl::StatusOr<DcmTerminalCost::Config> refused = dcmTerminalCostConfigFromConfig(block);
    EXPECT_EQ(refused.status().code(), absl::StatusCode::kInvalidArgument) << height;
    EXPECT_TRUE(absl::StrContains(refused.status().message(), "dcm_terminal_cost.com_height is")) << refused.status();
    EXPECT_TRUE(absl::StrContains(refused.status().message(), "leave dcm_terminal_cost.com_height out for the model's"))
        << refused.status();
  }
}

TEST(DcmTerminalCostFromConfigTest, EachFieldLandsInItsMember) {
  mpc_config::DcmTerminalCostConfig block;
  block.com_height = 0.9;
  block.gravity = 9.0;
  block.weight_x = 400.0;
  block.weight_y = 250.0;
  block.velocity_offset_factor = 0.5;
  block.support_blend_time = 0.2;
  const absl::StatusOr<DcmTerminalCost::Config> config = dcmTerminalCostConfigFromConfig(block);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->comHeight, 0.9);
  EXPECT_EQ(config->gravity, 9.0);
  EXPECT_EQ(config->weights, vector2_t(400.0, 250.0));
  EXPECT_EQ(config->velocityOffsetFactor, 0.5);
  EXPECT_EQ(config->supportBlendTime, 0.2);
}

TEST(DcmTerminalCostFromConfigTest, RefusesWhatTheCostRefuses) {
  mpc_config::DcmTerminalCostConfig block;
  block.weight_y = -1.0;
  EXPECT_TRUE(absl::StrContains(dcmTerminalCostConfigFromConfig(block).status().message(), "weight_y"));
  block = mpc_config::DcmTerminalCostConfig{};
  block.gravity = 0.0;
  EXPECT_EQ(dcmTerminalCostConfigFromConfig(block).status().code(), absl::StatusCode::kInvalidArgument);
  block = mpc_config::DcmTerminalCostConfig{};
  block.velocity_offset_factor = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(absl::StrContains(dcmTerminalCostConfigFromConfig(block).status().message(), "dcm_terminal_cost.velocity_offset_factor"));
}

TEST(IcpCostFromConfigTest, TheWeightActsAlongBothAxes) {
  EXPECT_EQ(*icpCostWeightsFromConfig(mpc_config::IcpCostWeights{}), vector2_t::Zero());
  mpc_config::IcpCostWeights weights;
  weights.icp_error_weight = 3.5;
  EXPECT_EQ(*icpCostWeightsFromConfig(weights), vector2_t(3.5, 3.5));
  weights.icp_error_weight = std::numeric_limits<double>::infinity();
  const absl::StatusOr<vector2_t> refused = icpCostWeightsFromConfig(weights);
  EXPECT_TRUE(absl::StrContains(refused.status().message(), "icp_cost_weights.icp_error_weight")) << refused.status();
}

}  // namespace
}  // namespace ocs2::humanoid
