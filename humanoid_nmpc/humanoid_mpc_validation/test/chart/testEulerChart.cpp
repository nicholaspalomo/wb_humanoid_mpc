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

#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"

#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_mpc_config/joint_value.nproto.h"
#include "humanoid_mpc_config/state_values.nproto.h"
#include "humanoid_mpc_config/xyz.nproto.h"
#include "humanoid_mpc_config/yaw_pitch_roll.nproto.h"
#include "humanoid_mpc_validation/chart/EulerChart.h"

/*
 * The Euler chart of the equivalence suites (design section 2.12): its blocks
 * are where the task file's initial_state puts them, Phi is EulerBoundary's
 * conversion and Phi^-1 its inverse with the angles wrapped, and dPhi / dx_e
 * and P_e are the derivatives of Phi, the latter in the tangent of the body
 * rotation, by central finite differences at random states.
 */

namespace ocs2::humanoid::validation {
namespace {

constexpr scalar_t kStep = 1.0e-6;
constexpr scalar_t kDerivativeTolerance = 1.0e-8;

StateInputLayout testLayout(StateInputLayout::Mpc mpc) {
  return StateInputLayout{.mpc = mpc,
                          .jointNames = {"hip", "knee", "ankle"},
                          .fixedJointNames = {},
                          .contactNames = {"left", "right"}};
}

/** An initial_state that gives every block of `layout` distinct values. */
mpc_config::StateValues distinctValues(const StateInputLayout& layout) {
  mpc_config::StateValues values;
  values.pos_base_rt_world_ewrt_world =
      mpc_config::Xyz{.x = 1.0, .y = 2.0, .z = 3.0};
  values.eulerzyx_base_rt_world =
      mpc_config::YawPitchRoll{.yaw = 0.4, .pitch = 0.5, .roll = 0.6};
  for (size_t j = 0; j < layout.jointNames.size(); ++j) {
    values.joint_positions.push_back(
        mpc_config::JointValue{.joint = layout.jointNames[j],
                               .value = 10.0 + static_cast<scalar_t>(j)});
  }
  if (layout.mpc == StateInputLayout::Mpc::kCentroidal) {
    values.mnorm_linmom_com_rt_world_ewrt_world =
        mpc_config::Xyz{.x = 7.0, .y = 8.0, .z = 9.0};
    values.mnorm_angmom_com_rt_world_ewrt_world =
        mpc_config::Xyz{.x = 4.0, .y = 5.0, .z = 6.0};
  } else {
    values.linvel_base_rt_world_ewrt_world =
        mpc_config::Xyz{.x = 21.0, .y = 22.0, .z = 23.0};
    values.angvel_base_rt_world_ewrt_base =
        mpc_config::YawPitchRoll{.yaw = 0.7, .pitch = 0.8, .roll = 0.9};
    for (size_t j = 0; j < layout.jointNames.size(); ++j) {
      values.joint_velocities.push_back(
          mpc_config::JointValue{.joint = layout.jointNames[j],
                                 .value = 30.0 + static_cast<scalar_t>(j)});
    }
  }
  return values;
}

/** The first row of the block `name` of `layout`. */
size_t blockStart(const EulerStateLayout& layout, const std::string& name) {
  size_t start = 0;
  for (const EulerStateSegment& segment : layout.segments) {
    if (segment.name == name) return start;
    start += segment.dim;
  }
  ADD_FAILURE() << "no block " << name;
  return 0;
}

vector_t randomEulerState(const EulerStateLayout& layout,
                          std::mt19937& engine) {
  std::uniform_real_distribution<scalar_t> uniform(-1.0, 1.0);
  vector_t state(static_cast<Eigen::Index>(getEulerStateDim(layout)));
  for (Eigen::Index i = 0; i < state.size(); ++i) state(i) = uniform(engine);
  return state;
}

/** The change of Phi between `minus` and `plus` in the physical tangent: body
 * rotation deltas on the quaternions. */
vector_t tangentDifference(const vector_t& minus,
                           const vector_t& plus,
                           const EulerStateLayout& layout) {
  vector_t difference(static_cast<Eigen::Index>(getEulerStateDim(layout)));
  Eigen::Index from = 0;
  Eigen::Index to = 0;
  for (const EulerStateSegment& segment : layout.segments) {
    const Eigen::Index dim = static_cast<Eigen::Index>(segment.dim);
    if (segment.kind == EulerStateSegmentKind::kEulerZyxAngles) {
      const vector4_t xiMinus = minus.segment<4>(from);
      const vector4_t xiPlus = plus.segment<4>(from);
      difference.segment<3>(to) = quaternionLog(vector4_t(
          quaternionProduct(vector4_t(quaternionConjugate(xiMinus)), xiPlus)));
      from += 4;
    } else {
      difference.segment(to, dim) =
          plus.segment(from, dim) - minus.segment(from, dim);
      from += dim;
    }
    to += dim;
  }
  return difference;
}

class EulerChartLayouts : public testing::TestWithParam<StateInputLayout::Mpc> {
};

TEST_P(EulerChartLayouts, BlocksAreWhereTheInitialStatePutsThem) {
  const StateInputLayout layout = testLayout(GetParam());
  const EulerStateLayout eulerLayout = eulerStateLayoutOf(layout);
  const absl::StatusOr<vector_t> state =
      stateValuesFromConfig(distinctValues(layout), layout, "initial_state");
  ASSERT_TRUE(state.ok()) << state.status();
  ASSERT_EQ(static_cast<size_t>(state->size()), getEulerStateDim(eulerLayout));
  EXPECT_EQ(static_cast<size_t>(state->size()), tuningDimension(layout));
  EXPECT_EQ(getPhysicalStateDim(eulerLayout),
            getEulerStateDim(eulerLayout) + 1);
  const Eigen::Index orientation = static_cast<Eigen::Index>(
      blockStart(eulerLayout, "eulerzyx_base_rt_world"));
  EXPECT_EQ(state->segment<3>(orientation), vector3_t(0.4, 0.5, 0.6));
  EXPECT_EQ(state->segment<3>(static_cast<Eigen::Index>(
                blockStart(eulerLayout, "pos_base_rt_world_ewrt_world"))),
            vector3_t(1.0, 2.0, 3.0));
  EXPECT_EQ((*state)(static_cast<Eigen::Index>(
                         blockStart(eulerLayout, "joint_positions")) +
                     2),
            12.0);
  if (GetParam() == StateInputLayout::Mpc::kCentroidal) {
    EXPECT_EQ(state->segment<3>(static_cast<Eigen::Index>(blockStart(
                  eulerLayout, "mnorm_linmom_com_rt_world_ewrt_world"))),
              vector3_t(7.0, 8.0, 9.0));
    EXPECT_EQ(state->segment<3>(static_cast<Eigen::Index>(blockStart(
                  eulerLayout, "mnorm_angmom_com_rt_world_ewrt_world"))),
              vector3_t(4.0, 5.0, 6.0));
  } else {
    EXPECT_EQ(state->segment<3>(static_cast<Eigen::Index>(
                  blockStart(eulerLayout, "linvel_base_rt_world_ewrt_world"))),
              vector3_t(21.0, 22.0, 23.0));
    EXPECT_EQ(state->segment<3>(static_cast<Eigen::Index>(
                  blockStart(eulerLayout, "angvel_base_rt_world_ewrt_base"))),
              vector3_t(0.7, 0.8, 0.9));
    EXPECT_EQ((*state)(static_cast<Eigen::Index>(
                           blockStart(eulerLayout, "joint_velocities")) +
                       1),
              31.0);
  }
}

TEST_P(EulerChartLayouts, PhiIsTheEulerBoundaryConversion) {
  const EulerStateLayout layout = eulerStateLayoutOf(testLayout(GetParam()));
  std::mt19937 engine(11U);
  const vector_t euler = randomEulerState(layout, engine);
  const absl::StatusOr<vector_t> physical =
      physicalStateFromEulerState(euler, layout);
  ASSERT_TRUE(physical.ok()) << physical.status();
  const Eigen::Index angles =
      static_cast<Eigen::Index>(blockStart(layout, "eulerzyx_base_rt_world"));
  const vector3_t theta = euler.segment<3>(angles);
  EXPECT_EQ(physical->segment<4>(angles), quaternionFromEulerZyx(theta));
  EXPECT_EQ(physical->head(angles), euler.head(angles));
  if (GetParam() == StateInputLayout::Mpc::kWholeBody) {
    const Eigen::Index rates = static_cast<Eigen::Index>(
        blockStart(layout, "angvel_base_rt_world_ewrt_base"));
    EXPECT_EQ(
        physical->segment<3>(rates + 1),
        localAngularVelocityFromEulerZyxRates(theta, euler.segment<3>(rates)));
  }
  EXPECT_EQ(physical->tail(3), euler.tail(3));
}

TEST_P(EulerChartLayouts, ChartJacobianIsTheDerivativeOfPhi) {
  const EulerStateLayout layout = eulerStateLayoutOf(testLayout(GetParam()));
  std::mt19937 engine(12U);
  for (int trial = 0; trial < 5; ++trial) {
    const vector_t euler = randomEulerState(layout, engine);
    const absl::StatusOr<matrix_t> jacobian = eulerChartJacobian(euler, layout);
    ASSERT_TRUE(jacobian.ok()) << jacobian.status();
    for (Eigen::Index j = 0; j < euler.size(); ++j) {
      vector_t plus = euler;
      vector_t minus = euler;
      plus(j) += kStep;
      minus(j) -= kStep;
      const vector_t column = (*physicalStateFromEulerState(plus, layout) -
                               *physicalStateFromEulerState(minus, layout)) /
                              (2.0 * kStep);
      EXPECT_LT((jacobian->col(j) - column).cwiseAbs().maxCoeff(),
                kDerivativeTolerance)
          << "column " << j;
    }
  }
}

TEST_P(EulerChartLayouts,
       PhysicalTangentJacobianIsTheDerivativeOfPhiInTheTangent) {
  const EulerStateLayout layout = eulerStateLayoutOf(testLayout(GetParam()));
  std::mt19937 engine(13U);
  for (int trial = 0; trial < 5; ++trial) {
    const vector_t euler = randomEulerState(layout, engine);
    const absl::StatusOr<matrix_t> tangentJacobian =
        physicalTangentJacobian(euler, layout);
    ASSERT_TRUE(tangentJacobian.ok()) << tangentJacobian.status();
    ASSERT_EQ(tangentJacobian->rows(), euler.size());
    for (Eigen::Index j = 0; j < euler.size(); ++j) {
      vector_t plus = euler;
      vector_t minus = euler;
      plus(j) += kStep;
      minus(j) -= kStep;
      const vector_t column =
          tangentDifference(*physicalStateFromEulerState(minus, layout),
                            *physicalStateFromEulerState(plus, layout),
                            layout) /
          (2.0 * kStep);
      EXPECT_LT((tangentJacobian->col(j) - column).cwiseAbs().maxCoeff(),
                kDerivativeTolerance)
          << "column " << j;
    }
    // On the angles the two are related by E+(xi): P_e = E+ dPhi, the T_B block
    // of design section 2.12.
    const Eigen::Index angles =
        static_cast<Eigen::Index>(blockStart(layout, "eulerzyx_base_rt_world"));
    const vector4_t xi = quaternionFromEulerZyx(euler.segment<3>(angles));
    const matrix_t chart = *eulerChartJacobian(euler, layout);
    const matrix3_t pulled = quaternionTangentMapPseudoInverse(xi) *
                             chart.block<4, 3>(angles, angles);
    EXPECT_LT((pulled - tangentJacobian->block<3, 3>(angles, angles))
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-14);
    EXPECT_LT((tangentJacobian->block<3, 3>(angles, angles) -
               eulerZyxRateToBodyRateMatrix(euler.segment<3>(angles)))
                  .cwiseAbs()
                  .maxCoeff(),
              1.0e-15);
  }
}

TEST_P(EulerChartLayouts,
       TheInverseChartRecoversTheEulerStateWithItsAnglesWrapped) {
  const EulerStateLayout layout = eulerStateLayoutOf(testLayout(GetParam()));
  const size_t angles = blockStart(layout, "eulerzyx_base_rt_world");
  std::mt19937 engine(31U);
  for (int trial = 0; trial < 20; ++trial) {
    vector_t euler = randomEulerState(layout, engine);
    // A yaw beyond pi comes back wrapped; the rest of the state as it was.
    euler(static_cast<Eigen::Index>(angles)) +=
        trial % 2 == 0 ? 0.0 : 2.0 * M_PI;
    const absl::StatusOr<vector_t> physical =
        physicalStateFromEulerState(euler, layout);
    ASSERT_TRUE(physical.ok()) << physical.status();
    // The inverse reads the quaternion through its direction only; the blocks
    // before the angles are Euclidean, so the quaternion starts where the
    // angles do.
    vector_t scaled = *physical;
    scaled.segment<4>(static_cast<Eigen::Index>(angles)) *= 2.5;
    const absl::StatusOr<vector_t> back =
        eulerStateFromPhysicalState(scaled, layout);
    ASSERT_TRUE(back.ok()) << back.status();
    vector_t expected = euler;
    expected(static_cast<Eigen::Index>(angles)) =
        std::remainder(euler(static_cast<Eigen::Index>(angles)), 2.0 * M_PI);
    EXPECT_LT((*back - expected).cwiseAbs().maxCoeff(), 1.0e-13) << trial;
  }
}

TEST_P(EulerChartLayouts,
       TheInverseChartRefusesAZeroQuaternionAndAStateOfAnotherSize) {
  const EulerStateLayout layout = eulerStateLayoutOf(testLayout(GetParam()));
  vector_t physical =
      vector_t::Zero(static_cast<Eigen::Index>(getPhysicalStateDim(layout)));
  EXPECT_EQ(eulerStateFromPhysicalState(physical, layout).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(
      eulerStateFromPhysicalState(vector_t::Zero(3), layout).status().code(),
      absl::StatusCode::kInvalidArgument);
}

INSTANTIATE_TEST_SUITE_P(BothMpcs,
                         EulerChartLayouts,
                         testing::Values(StateInputLayout::Mpc::kCentroidal,
                                         StateInputLayout::Mpc::kWholeBody));

TEST(EulerChart, RateCouplingIsTheDerivativeOfTheBodyRate) {
  std::mt19937 engine(14U);
  std::uniform_real_distribution<scalar_t> uniform(-1.2, 1.2);
  for (int trial = 0; trial < 10; ++trial) {
    const vector3_t angles(uniform(engine), uniform(engine), uniform(engine));
    const vector3_t rates(uniform(engine), uniform(engine), uniform(engine));
    const matrix3_t derivative =
        localAngularVelocityEulerAngleDerivative(angles, rates);
    // The chart's coupling block comes from the one Euler boundary.
    EXPECT_EQ(derivative, bodyRateEulerZyxJacobian(angles, rates));
    for (Eigen::Index j = 0; j < 3; ++j) {
      vector3_t plus = angles;
      vector3_t minus = angles;
      plus(j) += kStep;
      minus(j) -= kStep;
      const vector3_t column =
          (localAngularVelocityFromEulerZyxRates(plus, rates) -
           localAngularVelocityFromEulerZyxRates(minus, rates)) /
          (2.0 * kStep);
      EXPECT_LT((derivative.col(j) - column).cwiseAbs().maxCoeff(),
                kDerivativeTolerance);
    }
  }
}

TEST(EulerChart, TheSingularityIsWherePitchIsAQuarterTurn) {
  const EulerStateLayout layout =
      eulerStateLayoutOf(testLayout(StateInputLayout::Mpc::kCentroidal));
  vector_t euler =
      vector_t::Zero(static_cast<Eigen::Index>(getEulerStateDim(layout)));
  euler(
      static_cast<Eigen::Index>(blockStart(layout, "eulerzyx_base_rt_world")) +
      1) = 0.5 * M_PI;
  const absl::StatusOr<matrix_t> tangentJacobian =
      physicalTangentJacobian(euler, layout);
  ASSERT_TRUE(tangentJacobian.ok());
  EXPECT_NEAR(tangentJacobian->determinant(), 0.0, 1.0e-15);
}

TEST(EulerChart, RefusesAStateOfAnotherSizeAndRatesWithoutAngles) {
  const EulerStateLayout layout =
      eulerStateLayoutOf(testLayout(StateInputLayout::Mpc::kWholeBody));
  EXPECT_EQ(
      physicalStateFromEulerState(vector_t::Zero(3), layout).status().code(),
      absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(physicalTangentJacobian(vector_t::Zero(3), layout).status().code(),
            absl::StatusCode::kInvalidArgument);
  const EulerStateLayout ratesFirst{
      .name = "rates_first",
      .segments = {
          EulerStateSegment{.name = "rates",
                            .kind = EulerStateSegmentKind::kEulerZyxRates,
                            .dim = 3},
          EulerStateSegment{.name = "angles",
                            .kind = EulerStateSegmentKind::kEulerZyxAngles,
                            .dim = 3}}};
  EXPECT_EQ(physicalStateFromEulerState(vector_t::Zero(6), ratesFirst)
                .status()
                .code(),
            absl::StatusCode::kInvalidArgument);
  const EulerStateLayout shortAngles{
      .name = "short_angles",
      .segments = {
          EulerStateSegment{.name = "angles",
                            .kind = EulerStateSegmentKind::kEulerZyxAngles,
                            .dim = 2}}};
  EXPECT_EQ(eulerChartJacobian(vector_t::Zero(2), shortAngles).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid::validation
