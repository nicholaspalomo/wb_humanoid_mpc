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
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Eigenvalues"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "humanoid_state_estimator/KalmanFilter.h"

namespace ocs2::humanoid::estimation {
namespace {

using ::testing::HasSubstr;

constexpr scalar_t kNaN = std::numeric_limits<scalar_t>::quiet_NaN();

matrix_t identity(Eigen::Index dim) {
  return matrix_t::Identity(dim, dim);
}

matrix_t scalarMatrix(scalar_t value) {
  return matrix_t::Constant(1, 1, value);
}

vector_t makeVector(std::initializer_list<scalar_t> values) {
  vector_t vector(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (const scalar_t value : values) {
    vector(index++) = value;
  }
  return vector;
}

/// Entries uniform in [-1, 1] from a seeded generator, so every run sees the same numbers.
matrix_t randomMatrix(std::mt19937& generator, Eigen::Index rows, Eigen::Index cols) {
  std::uniform_real_distribution<scalar_t> distribution(-1.0, 1.0);
  matrix_t matrix(rows, cols);
  for (Eigen::Index row = 0; row < rows; ++row) {
    for (Eigen::Index col = 0; col < cols; ++col) {
      matrix(row, col) = distribution(generator);
    }
  }
  return matrix;
}

/// A random symmetric positive definite matrix.
matrix_t randomCovariance(std::mt19937& generator, Eigen::Index dim) {
  const matrix_t factor = randomMatrix(generator, dim, dim);
  return factor * factor.transpose() + 0.1 * identity(dim);
}

/// The joint mean, the channels stacked in the order reset() declared them.
vector_t jointState(const KalmanFilter& filter) {
  const std::vector<KalmanFilterState> states = filter.getStates();
  Eigen::Index dim = 0;
  for (const KalmanFilterState& state : states) {
    dim += state.state.size();
  }
  vector_t joint(dim);
  Eigen::Index offset = 0;
  for (const KalmanFilterState& state : states) {
    joint.segment(offset, state.state.size()) = state.state;
    offset += state.state.size();
  }
  return joint;
}

/// The joint covariance, assembled block by block from getCovariance().
matrix_t jointCovariance(const KalmanFilter& filter) {
  const std::vector<KalmanFilterState> states = filter.getStates();
  const Eigen::Index dim = jointState(filter).size();
  matrix_t joint(dim, dim);
  Eigen::Index row = 0;
  for (const KalmanFilterState& row_state : states) {
    Eigen::Index col = 0;
    for (const KalmanFilterState& col_state : states) {
      const absl::StatusOr<matrix_t> block = filter.getCovariance(row_state.name, col_state.name);
      EXPECT_TRUE(block.ok()) << block.status();
      joint.block(row, col, row_state.state.size(), col_state.state.size()) = *block;
      col += col_state.state.size();
    }
    row += row_state.state.size();
  }
  return joint;
}

/// |actual - expected| <= tolerance, relative to the largest entry of `expected` once that exceeds one.
void expectNear(const matrix_t& actual, const matrix_t& expected, scalar_t tolerance) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  const scalar_t scale = std::max(1.0, expected.cwiseAbs().maxCoeff());
  EXPECT_LE((actual - expected).cwiseAbs().maxCoeff(), tolerance * scale) << "actual:\n" << actual << "\nexpected:\n" << expected;
}

KalmanFilterState getStateOrDie(const KalmanFilter& filter, const std::string& name) {
  const absl::StatusOr<KalmanFilterState> state = filter.getState(name);
  EXPECT_TRUE(state.ok()) << state.status();
  return *state;
}

matrix_t getCovarianceOrDie(const KalmanFilter& filter, const std::string& row_name, const std::string& col_name) {
  const absl::StatusOr<matrix_t> covariance = filter.getCovariance(row_name, col_name);
  EXPECT_TRUE(covariance.ok()) << covariance.status();
  return *covariance;
}

std::unique_ptr<KalmanFilter> createOrDie(absl::Span<const KalmanFilterState> initial_states) {
  absl::StatusOr<std::unique_ptr<KalmanFilter>> filter = KalmanFilter::Create(initial_states);
  EXPECT_TRUE(filter.ok()) << filter.status();
  return std::move(*filter);
}

/// The textbook Kalman filter on the joint state, which the channel-organized filter must reproduce. It updates the
/// covariance in the short form (I - K H) P rather than the Joseph form; the two agree for the optimal gain.
struct DenseKalmanFilter {
  vector_t x;
  matrix_t P;

  void predict(const matrix_t& A, const matrix_t& B, const vector_t& u, const matrix_t& Q) {
    x = A * x + B * u;
    P = A * P * A.transpose() + Q;
  }

  void correct(const matrix_t& H, const vector_t& z, const matrix_t& R) {
    const matrix_t S = H * P * H.transpose() + R;
    const matrix_t K = P * H.transpose() * S.inverse();
    x = x + K * (z - H * x);
    P = (identity(P.rows()) - K * H) * P;
  }
};

TEST(KalmanFilterTest, matchesTheDenseKalmanFilterChannelByChannel) {
  std::mt19937 generator(42);
  // State channels a (2), b (3) and c (1), stacked in that order: offsets 0, 2 and 5.
  const vector_t a0 = randomMatrix(generator, /*rows=*/2, /*cols=*/1);
  const vector_t b0 = randomMatrix(generator, /*rows=*/3, /*cols=*/1);
  const vector_t c0 = randomMatrix(generator, /*rows=*/1, /*cols=*/1);
  const matrix_t P_a = randomCovariance(generator, /*dim=*/2);
  const matrix_t P_b = randomCovariance(generator, /*dim=*/3);
  const matrix_t P_c = randomCovariance(generator, /*dim=*/1);
  std::unique_ptr<KalmanFilter> filter = createOrDie({{.name = "a", .state = a0, .P_state_estimate = P_a},
                                                      {.name = "b", .state = b0, .P_state_estimate = P_b},
                                                      {.name = "c", .state = c0, .P_state_estimate = P_c}});

  DenseKalmanFilter reference;
  reference.x = (vector_t(6) << a0, b0, c0).finished();
  reference.P = matrix_t::Zero(6, 6);
  reference.P.block(0, 0, 2, 2) = P_a;
  reference.P.block(2, 2, 3, 3) = P_b;
  reference.P.block(5, 5, 1, 1) = P_c;

  // a depends on itself, on b and on input u1, with additive noise. b lists no A_bb, so it keeps the identity, depends
  // on c and on the noise-free input u2. c has no process model at all, so it is held constant.
  const matrix_t A_aa = randomMatrix(generator, /*rows=*/2, /*cols=*/2);
  const matrix_t A_ab = randomMatrix(generator, /*rows=*/2, /*cols=*/3);
  const matrix_t A_bc = randomMatrix(generator, /*rows=*/3, /*cols=*/1);
  const matrix_t B_a_u1 = randomMatrix(generator, /*rows=*/2, /*cols=*/1);
  const matrix_t B_b_u2 = randomMatrix(generator, /*rows=*/3, /*cols=*/2);
  const matrix_t Q_a = 0.01 * randomCovariance(generator, /*dim=*/2);
  const matrix_t Q_u1 = 0.01 * randomCovariance(generator, /*dim=*/1);
  const std::vector<KalmanFilterProcessModel> process_model = {
      {.state_name = "a", .A_state_transition = {{"a", A_aa}, {"b", A_ab}}, .B_control_input = {{"u1", B_a_u1}}, .Q_process_noise = Q_a},
      {.state_name = "b", .A_state_transition = {{"c", A_bc}}, .B_control_input = {{"u2", B_b_u2}}},
  };
  // Inputs u1 (1) and u2 (2), stacked at offsets 0 and 1.
  matrix_t A = identity(6);
  A.block(0, 0, 2, 2) = A_aa;
  A.block(0, 2, 2, 3) = A_ab;
  A.block(2, 5, 3, 1) = A_bc;
  matrix_t B = matrix_t::Zero(6, 3);
  B.block(0, 0, 2, 1) = B_a_u1;
  B.block(2, 1, 3, 2) = B_b_u2;
  matrix_t Q_inputs = matrix_t::Zero(3, 3);
  Q_inputs.block(0, 0, 1, 1) = Q_u1;
  matrix_t Q = B * Q_inputs * B.transpose();
  Q.block(0, 0, 2, 2) += Q_a;

  // Measurement m1 (2) observes a and c, m2 (1) observes b.
  const matrix_t H_m1_a = randomMatrix(generator, /*rows=*/2, /*cols=*/2);
  const matrix_t H_m1_c = randomMatrix(generator, /*rows=*/2, /*cols=*/1);
  const matrix_t H_m2_b = randomMatrix(generator, /*rows=*/1, /*cols=*/3);
  const matrix_t R_m1 = 0.1 * randomCovariance(generator, /*dim=*/2);
  const matrix_t R_m2 = 0.1 * randomCovariance(generator, /*dim=*/1);
  matrix_t H = matrix_t::Zero(3, 6);
  H.block(0, 0, 2, 2) = H_m1_a;
  H.block(0, 5, 2, 1) = H_m1_c;
  H.block(2, 2, 1, 3) = H_m2_b;
  matrix_t R = matrix_t::Zero(3, 3);
  R.block(0, 0, 2, 2) = R_m1;
  R.block(2, 2, 1, 1) = R_m2;

  const std::vector<std::string> names = {"a", "b", "c"};
  const std::vector<Eigen::Index> offsets = {0, 2, 5};
  const std::vector<Eigen::Index> sizes = {2, 3, 1};
  const std::function<void(const std::string&)> expectMatchesReference = [&](const std::string& step) {
    SCOPED_TRACE(step);
    for (size_t i = 0; i < names.size(); ++i) {
      const KalmanFilterState state = getStateOrDie(*filter, names[i]);
      EXPECT_EQ(state.name, names[i]);
      expectNear(state.state, reference.x.segment(offsets[i], sizes[i]), /*tolerance=*/1.0e-9);
      expectNear(state.P_state_estimate, reference.P.block(offsets[i], offsets[i], sizes[i], sizes[i]), /*tolerance=*/1.0e-9);
      for (size_t j = 0; j < names.size(); ++j) {
        expectNear(getCovarianceOrDie(*filter, names[i], names[j]), reference.P.block(offsets[i], offsets[j], sizes[i], sizes[j]),
                   /*tolerance=*/1.0e-9);
      }
    }
  };

  for (int step = 0; step < 5; ++step) {
    const vector_t u1 = randomMatrix(generator, /*rows=*/1, /*cols=*/1);
    const vector_t u2 = randomMatrix(generator, /*rows=*/2, /*cols=*/1);
    ASSERT_TRUE(filter->predict(process_model, {{.name = "u1", .input = u1, .Q_input_noise = Q_u1}, {.name = "u2", .input = u2}}).ok());
    reference.predict(A, B, (vector_t(3) << u1, u2).finished(), Q);
    expectMatchesReference(absl::StrCat("predict ", step));

    const vector_t z1 = randomMatrix(generator, /*rows=*/2, /*cols=*/1);
    const vector_t z2 = randomMatrix(generator, /*rows=*/1, /*cols=*/1);
    ASSERT_TRUE(
        filter
            ->correct(
                {{.name = "m1", .measurement = z1, .H_measurement_model = {{"a", H_m1_a}, {"c", H_m1_c}}, .R_measurement_noise = R_m1},
                 {.name = "m2", .measurement = z2, .H_measurement_model = {{"b", H_m2_b}}, .R_measurement_noise = R_m2}})
            .ok());
    reference.correct(H, (vector_t(3) << z1, z2).finished(), R);
    expectMatchesReference(absl::StrCat("correct ", step));
  }
}

TEST(KalmanFilterTest, aScalarRandomWalkFollowsTheClosedForm) {
  std::unique_ptr<KalmanFilter> filter = createOrDie({{.name = "x", .state = makeVector({1.0}), .P_state_estimate = scalarMatrix(1.0)}});

  ASSERT_TRUE(filter->predict({{.state_name = "x", .Q_process_noise = scalarMatrix(0.5)}}, /*inputs=*/{}).ok());
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "x").state(0), 1.0);
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "x").P_state_estimate(0, 0), 1.5);

  // S = 1.5 + 2 = 3.5 and K = 1.5 / 3.5 = 3/7, so x = 1 + 3/7 (3 - 1) = 13/7 and P = (1 - 3/7) 1.5 = 6/7.
  ASSERT_TRUE(filter
                  ->correct({{.name = "z",
                              .measurement = makeVector({3.0}),
                              .H_measurement_model = {{"x", scalarMatrix(1.0)}},
                              .R_measurement_noise = scalarMatrix(2.0)}})
                  .ok());
  EXPECT_NEAR(getStateOrDie(*filter, "x").state(0), 13.0 / 7.0, 1.0e-15);
  EXPECT_NEAR(getStateOrDie(*filter, "x").P_state_estimate(0, 0), 6.0 / 7.0, 1.0e-15);
}

TEST(KalmanFilterTest, aChannelWithoutAProcessModelKeepsItsMeanAndCovariance) {
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "moving", .state = makeVector({1.0}), .P_state_estimate = scalarMatrix(1.0)},
                   {.name = "still", .state = makeVector({2.0, 3.0}), .P_state_estimate = 2.0 * identity(2)}});
  // Correlate the two channels, so that the cross-covariance has something to show.
  ASSERT_TRUE(filter
                  ->correct({{.name = "sum",
                              .measurement = makeVector({4.0}),
                              .H_measurement_model = {{"moving", scalarMatrix(1.0)}, {"still", matrix_t::Ones(1, 2)}},
                              .R_measurement_noise = scalarMatrix(0.5)}})
                  .ok());
  const KalmanFilterState still = getStateOrDie(*filter, "still");
  const matrix_t P_moving_still = getCovarianceOrDie(*filter, "moving", "still");
  ASSERT_GT(P_moving_still.cwiseAbs().maxCoeff(), 0.0);

  ASSERT_TRUE(filter->predict({{.state_name = "moving", .A_state_transition = {{"moving", scalarMatrix(2.0)}}}}, /*inputs=*/{}).ok());

  // x_still[k+1] = x_still[k]: unchanged; the cross-covariance follows A_moving,moving P_moving,still.
  EXPECT_TRUE(getStateOrDie(*filter, "still").state == still.state);
  EXPECT_TRUE(getStateOrDie(*filter, "still").P_state_estimate == still.P_state_estimate);
  expectNear(getCovarianceOrDie(*filter, "moving", "still"), 2.0 * P_moving_still, /*tolerance=*/1.0e-15);
  expectNear(getCovarianceOrDie(*filter, "still", "moving"), 2.0 * P_moving_still.transpose(), /*tolerance=*/1.0e-15);
}

TEST(KalmanFilterTest, anUnlistedDiagonalBlockIsTheIdentityAndAZeroOneForgetsThePast) {
  const scalar_t dt = 0.1;
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "position", .state = makeVector({1.0}), .P_state_estimate = scalarMatrix(0.0)},
                   {.name = "velocity", .state = makeVector({2.0}), .P_state_estimate = scalarMatrix(0.0)}});

  // position lists only the velocity block, so p[k+1] = p[k] + dt v[k].
  ASSERT_TRUE(filter->predict({{.state_name = "position", .A_state_transition = {{"velocity", scalarMatrix(dt)}}}}, /*inputs=*/{}).ok());
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "position").state(0), 1.0 + dt * 2.0);
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "velocity").state(0), 2.0);

  // velocity lists its own block as zero, so v[k+1] = u[k] and its covariance is the noise of u alone.
  ASSERT_TRUE(filter
                  ->predict({{.state_name = "velocity",
                              .A_state_transition = {{"velocity", scalarMatrix(0.0)}},
                              .B_control_input = {{"commanded_velocity", scalarMatrix(1.0)}}}},
                            {{.name = "commanded_velocity", .input = makeVector({-5.0}), .Q_input_noise = scalarMatrix(0.25)}})
                  .ok());
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "velocity").state(0), -5.0);
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "velocity").P_state_estimate(0, 0), 0.25);
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "position").state(0), 1.0 + dt * 2.0);
}

TEST(KalmanFilterTest, inputNoiseIsCorrelatedAcrossTheChannelsItDrives) {
  const scalar_t dt = 0.01;
  const scalar_t sigma = 0.3;
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "position", .state = vector_t::Zero(3), .P_state_estimate = matrix_t::Zero(3, 3)},
                   {.name = "velocity", .state = vector_t::Zero(3), .P_state_estimate = matrix_t::Zero(3, 3)}});
  const vector_t acceleration = makeVector({1.0, -2.0, 0.5});

  ASSERT_TRUE(filter
                  ->predict({{.state_name = "position",
                              .A_state_transition = {{"velocity", dt * identity(3)}},
                              .B_control_input = {{"acceleration", 0.5 * dt * dt * identity(3)}}},
                             {.state_name = "velocity", .B_control_input = {{"acceleration", dt * identity(3)}}}},
                            {{.name = "acceleration", .input = acceleration, .Q_input_noise = sigma * sigma * identity(3)}})
                  .ok());

  expectNear(getStateOrDie(*filter, "position").state, 0.5 * dt * dt * acceleration, /*tolerance=*/1.0e-15);
  expectNear(getStateOrDie(*filter, "velocity").state, dt * acceleration, /*tolerance=*/1.0e-15);
  const scalar_t variance = sigma * sigma;
  expectNear(getCovarianceOrDie(*filter, "position", "position"), 0.25 * dt * dt * dt * dt * variance * identity(3), /*tolerance=*/1.0e-15);
  expectNear(getCovarianceOrDie(*filter, "position", "velocity"), 0.5 * dt * dt * dt * variance * identity(3), /*tolerance=*/1.0e-15);
  expectNear(getCovarianceOrDie(*filter, "velocity", "position"), 0.5 * dt * dt * dt * variance * identity(3), /*tolerance=*/1.0e-15);
  expectNear(getCovarianceOrDie(*filter, "velocity", "velocity"), dt * dt * variance * identity(3), /*tolerance=*/1.0e-15);
}

// The property the joint covariance exists for: velocity is never measured, and is still corrected, through the
// cross-covariance that p[k+1] = p[k] + dt v[k] builds up. A covariance stored channel by channel never corrects it.
TEST(KalmanFilterTest, positionMeasurementsCorrectTheVelocityThroughTheCrossCovariance) {
  const scalar_t dt = 0.01;
  const scalar_t true_velocity = 0.8;
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "position", .state = makeVector({0.0}), .P_state_estimate = scalarMatrix(1.0e-6)},
                   {.name = "velocity", .state = makeVector({0.0}), .P_state_estimate = scalarMatrix(1.0)}});
  const std::vector<KalmanFilterProcessModel> constant_velocity = {
      {.state_name = "position", .A_state_transition = {{"velocity", scalarMatrix(dt)}}}};

  for (int step = 1; step <= 200; ++step) {
    ASSERT_TRUE(filter->predict(constant_velocity, /*inputs=*/{}).ok());
    ASSERT_TRUE(filter
                    ->correct({{.name = "position",
                                .measurement = makeVector({true_velocity * dt * step}),
                                .H_measurement_model = {{"position", scalarMatrix(1.0)}},
                                .R_measurement_noise = scalarMatrix(1.0e-6)}})
                    .ok());
    if (step == 1) {
      EXPECT_GT(getStateOrDie(*filter, "velocity").state(0), 0.0) << "the first position measurement must already move the velocity";
    }
  }
  EXPECT_NEAR(getStateOrDie(*filter, "velocity").state(0), true_velocity, 1.0e-3);
  EXPECT_LT(getStateOrDie(*filter, "velocity").P_state_estimate(0, 0), 1.0e-3);
}

TEST(KalmanFilterTest, aJointCorrectionEqualsSequentialCorrectionsInAnyOrder) {
  std::mt19937 generator(7);
  const std::vector<KalmanFilterState> initial_states = {
      {.name = "a", .state = randomMatrix(generator, /*rows=*/2, /*cols=*/1), .P_state_estimate = randomCovariance(generator, /*dim=*/2)},
      {.name = "b", .state = randomMatrix(generator, /*rows=*/2, /*cols=*/1), .P_state_estimate = randomCovariance(generator, /*dim=*/2)}};
  // A coupled transition, so that the channels are correlated before the corrections.
  const std::vector<KalmanFilterProcessModel> process_model = {
      {.state_name = "a", .A_state_transition = {{"b", randomMatrix(generator, /*rows=*/2, /*cols=*/2)}}},
      {.state_name = "b", .A_state_transition = {{"a", randomMatrix(generator, /*rows=*/2, /*cols=*/2)}}}};
  const KalmanFilterMeasurement first = {.name = "first",
                                         .measurement = randomMatrix(generator, /*rows=*/2, /*cols=*/1),
                                         .H_measurement_model = {{"a", randomMatrix(generator, /*rows=*/2, /*cols=*/2)}},
                                         .R_measurement_noise = randomCovariance(generator, /*dim=*/2)};
  const KalmanFilterMeasurement second = {.name = "second",
                                          .measurement = randomMatrix(generator, /*rows=*/1, /*cols=*/1),
                                          .H_measurement_model = {{"a", randomMatrix(generator, /*rows=*/1, /*cols=*/2)},
                                                                  {"b", randomMatrix(generator, /*rows=*/1, /*cols=*/2)}},
                                          .R_measurement_noise = randomCovariance(generator, /*dim=*/1)};

  std::unique_ptr<KalmanFilter> joint = createOrDie(initial_states);
  std::unique_ptr<KalmanFilter> reversed = createOrDie(initial_states);
  std::unique_ptr<KalmanFilter> sequential = createOrDie(initial_states);
  for (KalmanFilter* absl_nonnull filter : {joint.get(), reversed.get(), sequential.get()}) {
    ASSERT_TRUE(filter->predict(process_model, /*inputs=*/{}).ok());
  }
  ASSERT_TRUE(joint->correct({first, second}).ok());
  ASSERT_TRUE(reversed->correct({second, first}).ok());
  ASSERT_TRUE(sequential->correct({first}).ok());
  ASSERT_TRUE(sequential->correct({second}).ok());

  expectNear(jointState(*reversed), jointState(*joint), /*tolerance=*/1.0e-12);
  expectNear(jointCovariance(*reversed), jointCovariance(*joint), /*tolerance=*/1.0e-12);
  expectNear(jointState(*sequential), jointState(*joint), /*tolerance=*/1.0e-12);
  expectNear(jointCovariance(*sequential), jointCovariance(*joint), /*tolerance=*/1.0e-12);
}

TEST(KalmanFilterTest, theCovarianceStaysSymmetricPositiveSemiDefiniteUnderNearlyExactMeasurements) {
  std::mt19937 generator(3);
  std::unique_ptr<KalmanFilter> filter = createOrDie({{.name = "a", .state = vector_t::Zero(3), .P_state_estimate = 1.0e3 * identity(3)},
                                                      {.name = "b", .state = vector_t::Zero(3), .P_state_estimate = 1.0e-3 * identity(3)}});
  for (int step = 0; step < 200; ++step) {
    ASSERT_TRUE(
        filter
            ->predict({{.state_name = "a", .A_state_transition = {{"b", 0.01 * identity(3)}}, .Q_process_noise = 1.0e-8 * identity(3)}},
                      /*inputs=*/{})
            .ok());
    ASSERT_TRUE(filter
                    ->correct({{.name = "a",
                                .measurement = randomMatrix(generator, /*rows=*/3, /*cols=*/1),
                                .H_measurement_model = {{"a", identity(3)}, {"b", randomMatrix(generator, /*rows=*/3, /*cols=*/3)}},
                                .R_measurement_noise = 1.0e-10 * identity(3)}})
                    .ok());
    const matrix_t P = jointCovariance(*filter);
    ASSERT_TRUE(P == P.transpose()) << "step " << step;
    const Eigen::SelfAdjointEigenSolver<matrix_t> eigen_solver(P, Eigen::EigenvaluesOnly);
    ASSERT_GE(eigen_solver.eigenvalues().minCoeff(), -1.0e-12 * P.cwiseAbs().maxCoeff()) << "step " << step;
  }
}

TEST(KalmanFilterTest, getStatesReturnsTheChannelsInDeclarationOrderAndRoundTripsThroughReset) {
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "z_last_alphabetically", .state = makeVector({1.0}), .P_state_estimate = scalarMatrix(1.0)},
                   {.name = "a_first_alphabetically", .state = makeVector({2.0, 3.0}), .P_state_estimate = identity(2)}});
  ASSERT_TRUE(filter
                  ->correct({{.name = "sum",
                              .measurement = makeVector({4.0}),
                              .H_measurement_model = {{"z_last_alphabetically", scalarMatrix(1.0)},
                                                      {"a_first_alphabetically", matrix_t::Ones(1, 2)}},
                              .R_measurement_noise = scalarMatrix(1.0)}})
                  .ok());

  const std::vector<KalmanFilterState> states = filter->getStates();
  ASSERT_EQ(states.size(), 2u);
  EXPECT_EQ(states[0].name, "z_last_alphabetically");
  EXPECT_EQ(states[1].name, "a_first_alphabetically");

  KalmanFilter copy;
  ASSERT_TRUE(copy.reset(states).ok());
  for (const KalmanFilterState& state : states) {
    EXPECT_TRUE(getStateOrDie(copy, state.name).state == state.state);
    EXPECT_TRUE(getStateOrDie(copy, state.name).P_state_estimate == state.P_state_estimate);
  }
  // reset() declares the channels uncorrelated.
  EXPECT_TRUE(getCovarianceOrDie(copy, "z_last_alphabetically", "a_first_alphabetically").isZero(0.0));
}

TEST(KalmanFilterTest, anEmptyCorrectionLeavesTheEstimateUnchanged) {
  std::unique_ptr<KalmanFilter> filter = createOrDie({{.name = "x", .state = makeVector({1.0, 2.0}), .P_state_estimate = identity(2)}});
  const vector_t x = jointState(*filter);
  const matrix_t P = jointCovariance(*filter);
  ASSERT_TRUE(filter->correct(/*measurements=*/{}).ok());
  EXPECT_TRUE(jointState(*filter) == x);
  EXPECT_TRUE(jointCovariance(*filter) == P);
}

// A legged robot's base, driven by a world-frame IMU acceleration and observed only through the position of a stance
// foot relative to the base, as leg kinematics measures it. The base velocity starts wrong; the foot does not move.
TEST(KalmanFilterTest, legKinematicsCorrectTheBaseVelocityOfAContactAidedEstimator) {
  const scalar_t dt = 0.002;
  const vector_t true_foot = makeVector({0.1, -0.1, 0.0});
  const vector_t acceleration = makeVector({0.2, 0.0, -0.1});
  vector_t true_base_position = makeVector({0.0, 0.0, 0.9});
  vector_t true_base_velocity = makeVector({0.5, -0.2, 0.0});

  std::unique_ptr<KalmanFilter> filter = createOrDie(
      {{.name = "base_position", .state = true_base_position, .P_state_estimate = 1.0e-4 * identity(3)},
       {.name = "base_velocity", .state = vector_t::Zero(3), .P_state_estimate = identity(3)},
       {.name = "foot_position", .state = true_foot + makeVector({0.01, 0.01, 0.0}), .P_state_estimate = 1.0e-2 * identity(3)}});
  const std::vector<KalmanFilterProcessModel> process_model = {
      {.state_name = "base_position",
       .A_state_transition = {{"base_velocity", dt * identity(3)}},
       .B_control_input = {{"imu_acceleration", 0.5 * dt * dt * identity(3)}}},
      {.state_name = "base_velocity", .B_control_input = {{"imu_acceleration", dt * identity(3)}}},
      {.state_name = "foot_position", .Q_process_noise = 1.0e-10 * identity(3)},
  };

  for (int step = 0; step < 500; ++step) {
    true_base_position += dt * true_base_velocity + 0.5 * dt * dt * acceleration;
    true_base_velocity += dt * acceleration;
    ASSERT_TRUE(
        filter->predict(process_model, {{.name = "imu_acceleration", .input = acceleration, .Q_input_noise = 1.0e-4 * identity(3)}}).ok());
    ASSERT_TRUE(filter
                    ->correct({{.name = "foot_relative_position",
                                .measurement = true_foot - true_base_position,
                                .H_measurement_model = {{"foot_position", identity(3)}, {"base_position", -identity(3)}},
                                .R_measurement_noise = 1.0e-6 * identity(3)}})
                    .ok());
  }

  expectNear(getStateOrDie(*filter, "base_velocity").state, true_base_velocity, /*tolerance=*/1.0e-2);
  const vector_t relative_estimate = getStateOrDie(*filter, "foot_position").state - getStateOrDie(*filter, "base_position").state;
  expectNear(relative_estimate, true_foot - true_base_position, /*tolerance=*/1.0e-3);
  // Only foot minus base is measured, so their absolute positions remain uncertain together: positively correlated.
  const matrix_t P_base_foot = getCovarianceOrDie(*filter, "base_position", "foot_position");
  EXPECT_GT(P_base_foot.diagonal().minCoeff(), 0.0);
}

TEST(KalmanFilterTest, createRejectsInvalidStateChannelsNamingTheProblem) {
  struct Case {
    std::string label;
    std::vector<KalmanFilterState> states;
    std::string expected_message;
  };
  const std::vector<Case> cases = {
      {"no channel", {}, "at least one state channel"},
      {"empty name", {{.name = "", .state = makeVector({1.0}), .P_state_estimate = scalarMatrix(1.0)}}, "empty name"},
      {"repeated name",
       {{.name = "x", .state = makeVector({1.0}), .P_state_estimate = scalarMatrix(1.0)},
        {.name = "x", .state = makeVector({1.0}), .P_state_estimate = scalarMatrix(1.0)}},
       "state channel 'x' is declared twice"},
      {"empty state", {{.name = "x", .state = vector_t(), .P_state_estimate = matrix_t()}}, "state channel 'x' is empty"},
      {"non-finite state",
       {{.name = "x", .state = makeVector({kNaN}), .P_state_estimate = scalarMatrix(1.0)}},
       "state of state channel 'x'"},
      {"covariance of the wrong size",
       {{.name = "x", .state = vector_t::Zero(3), .P_state_estimate = identity(2)}},
       "P_state_estimate of state channel 'x' is 2x2, expected 3x3"},
      {"asymmetric covariance",
       {{.name = "x", .state = vector_t::Zero(2), .P_state_estimate = (matrix_t(2, 2) << 1.0, 0.5, 0.0, 1.0).finished()}},
       "P_state_estimate of state channel 'x' is not symmetric"},
      {"indefinite covariance",
       {{.name = "x", .state = vector_t::Zero(2), .P_state_estimate = (matrix_t(2, 2) << 1.0, 2.0, 2.0, 1.0).finished()}},
       "P_state_estimate of state channel 'x' is not positive semi-definite"},
      {"non-finite covariance", {{.name = "x", .state = makeVector({1.0}), .P_state_estimate = scalarMatrix(kNaN)}}, "is not finite"},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    const absl::StatusOr<std::unique_ptr<KalmanFilter>> filter = KalmanFilter::Create(test_case.states);
    ASSERT_FALSE(filter.ok());
    EXPECT_EQ(filter.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(filter.status().message(), HasSubstr(test_case.expected_message));
  }
}

TEST(KalmanFilterTest, aRejectedResetKeepsThePreviousChannels) {
  std::unique_ptr<KalmanFilter> filter = createOrDie({{.name = "x", .state = makeVector({1.0}), .P_state_estimate = scalarMatrix(1.0)}});
  EXPECT_FALSE(filter
                   ->reset({{.name = "y", .state = makeVector({2.0}), .P_state_estimate = scalarMatrix(1.0)},
                            {.name = "z", .state = makeVector({kNaN}), .P_state_estimate = scalarMatrix(1.0)}})
                   .ok());
  const std::vector<KalmanFilterState> states = filter->getStates();
  ASSERT_EQ(states.size(), 1u);
  EXPECT_EQ(states[0].name, "x");
  EXPECT_DOUBLE_EQ(states[0].state(0), 1.0);
}

TEST(KalmanFilterTest, aFilterWithoutChannelsRefusesToStepUntilReset) {
  KalmanFilter filter;
  EXPECT_TRUE(filter.getStates().empty());

  const absl::Status predicted = filter.predict(/*process_model=*/{}, /*inputs=*/{});
  EXPECT_EQ(predicted.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(predicted.message(), HasSubstr("predict() before reset()"));
  const absl::Status corrected = filter.correct(/*measurements=*/{});
  EXPECT_EQ(corrected.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(corrected.message(), HasSubstr("correct() before reset()"));

  ASSERT_TRUE(filter.reset({{.name = "x", .state = makeVector({1.0}), .P_state_estimate = scalarMatrix(1.0)}}).ok());
  EXPECT_TRUE(filter.predict(/*process_model=*/{}, /*inputs=*/{}).ok());
}

TEST(KalmanFilterTest, unknownChannelLookupsAreNotFoundAndListTheDeclaredChannels) {
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "base_position", .state = vector_t::Zero(3), .P_state_estimate = identity(3)},
                   {.name = "base_velocity", .state = vector_t::Zero(3), .P_state_estimate = identity(3)}});
  const absl::StatusOr<KalmanFilterState> state = filter->getState("base_orientation");
  EXPECT_EQ(state.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(state.status().message(), HasSubstr("'base_orientation'"));
  EXPECT_THAT(state.status().message(), HasSubstr("the state channels are: base_position, base_velocity"));
  EXPECT_EQ(filter->getCovariance("base_position", "base_orientation").status().code(), absl::StatusCode::kNotFound);
  EXPECT_EQ(filter->getCovariance("base_orientation", "base_position").status().code(), absl::StatusCode::kNotFound);
}

TEST(KalmanFilterTest, getStateFillsACallerOwnedStateAndLeavesItAloneForAnUnknownName) {
  std::unique_ptr<KalmanFilter> filter = createOrDie({{.name = "a", .state = makeVector({1.0, 2.0}), .P_state_estimate = identity(2)},
                                                      {.name = "b", .state = makeVector({3.0}), .P_state_estimate = scalarMatrix(4.0)}});
  KalmanFilterState state;
  ASSERT_TRUE(filter->getState("a", state).ok());
  EXPECT_EQ(state.name, "a");
  EXPECT_TRUE(state.state == makeVector({1.0, 2.0}));
  EXPECT_TRUE(state.P_state_estimate == identity(2));

  // The same object then reads a channel of another size.
  ASSERT_TRUE(filter->getState("b", state).ok());
  EXPECT_EQ(state.name, "b");
  EXPECT_TRUE(state.state == makeVector({3.0}));
  EXPECT_TRUE(state.P_state_estimate == scalarMatrix(4.0));

  const absl::Status unknown = filter->getState("c", state);
  EXPECT_EQ(unknown.code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(unknown.message(), HasSubstr("the state channels are: a, b"));
  EXPECT_EQ(state.name, "b");
  EXPECT_TRUE(state.state == makeVector({3.0}));
}

TEST(KalmanFilterTest, reserveOnlySizesStorageBeforeOrAfterReset) {
  const std::vector<KalmanFilterState> initial_states = {
      {.name = "position", .state = makeVector({0.0, 1.0}), .P_state_estimate = identity(2)},
      {.name = "velocity", .state = makeVector({1.0, 0.0}), .P_state_estimate = 2.0 * identity(2)}};
  const std::vector<KalmanFilterProcessModel> process_model = {
      {.state_name = "position", .A_state_transition = {{"velocity", 0.1 * identity(2)}}, .Q_process_noise = 1.0e-3 * identity(2)},
      {.state_name = "velocity", .B_control_input = {{"acceleration", 0.1 * identity(2)}}}};
  const std::vector<KalmanFilterInput> inputs = {
      {.name = "acceleration", .input = makeVector({0.5, -0.5}), .Q_input_noise = 1.0e-2 * identity(2)}};
  const std::vector<KalmanFilterMeasurement> measurements = {{.name = "position",
                                                              .measurement = makeVector({0.2, 0.9}),
                                                              .H_measurement_model = {{"position", identity(2)}},
                                                              .R_measurement_noise = 1.0e-2 * identity(2)}};

  std::unique_ptr<KalmanFilter> unreserved = createOrDie(initial_states);
  KalmanFilter reserved_before_reset;
  ASSERT_TRUE(reserved_before_reset.reserve(/*max_input_dim=*/8, /*max_measurement_dim=*/16).ok());
  ASSERT_TRUE(reserved_before_reset.reset(initial_states).ok());
  std::unique_ptr<KalmanFilter> reserved_after_reset = createOrDie(initial_states);
  ASSERT_TRUE(reserved_after_reset->reserve(/*max_input_dim=*/8, /*max_measurement_dim=*/16).ok());

  for (KalmanFilter* absl_nonnull filter : {unreserved.get(), &reserved_before_reset, reserved_after_reset.get()}) {
    for (int step = 0; step < 3; ++step) {
      ASSERT_TRUE(filter->predict(process_model, inputs).ok());
      ASSERT_TRUE(filter->correct(measurements).ok());
    }
  }
  for (KalmanFilter* absl_nonnull filter : {&reserved_before_reset, reserved_after_reset.get()}) {
    expectNear(jointState(*filter), jointState(*unreserved), /*tolerance=*/1.0e-14);
    expectNear(jointCovariance(*filter), jointCovariance(*unreserved), /*tolerance=*/1.0e-14);
  }

  const absl::Status negative = unreserved->reserve(/*max_input_dim=*/-1, /*max_measurement_dim=*/0);
  EXPECT_EQ(negative.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(negative.message(), HasSubstr("non-negative dimensions"));
}

class KalmanFilterRejectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    filter_ = createOrDie({{.name = "position", .state = makeVector({1.0, 2.0}), .P_state_estimate = identity(2)},
                           {.name = "velocity", .state = makeVector({3.0, 4.0}), .P_state_estimate = 2.0 * identity(2)}});
    // Correlate the channels, so that an accidental write anywhere in the covariance shows.
    ASSERT_TRUE(
        filter_->predict({{.state_name = "position", .A_state_transition = {{"velocity", 0.1 * identity(2)}}}}, /*inputs=*/{}).ok());
    x_before_ = jointState(*filter_);
    P_before_ = jointCovariance(*filter_);
  }

  void expectRejectedWithoutChange(const absl::Status& status, const std::string& expected_message) {
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
    EXPECT_THAT(status.message(), HasSubstr(expected_message));
    EXPECT_TRUE(jointState(*filter_) == x_before_);
    EXPECT_TRUE(jointCovariance(*filter_) == P_before_);
  }

  std::unique_ptr<KalmanFilter> filter_;
  vector_t x_before_;
  matrix_t P_before_;
};

TEST_F(KalmanFilterRejectionTest, invalidPredictionsAreRejectedWithoutChangingTheEstimate) {
  struct Case {
    std::string label;
    std::vector<KalmanFilterProcessModel> process_model;
    std::vector<KalmanFilterInput> inputs;
    std::string expected_message;
  };
  const KalmanFilterInput acceleration = {.name = "acceleration", .input = makeVector({1.0, 1.0})};
  const std::vector<Case> cases = {
      {"model of an unknown channel",
       {{.state_name = "orientation"}},
       {},
       "a process model names state channel 'orientation', which reset() did not declare; the state channels are: position, velocity"},
      {"block of an unknown channel",
       {{.state_name = "position", .A_state_transition = {{"orientation", identity(2)}}}},
       {},
       "A_state_transition of state channel 'position' names state channel 'orientation'"},
      {"transition block of the wrong size",
       {{.state_name = "position", .A_state_transition = {{"velocity", identity(3)}}}},
       {},
       "A_state_transition block ('position', 'velocity') is 3x3, expected 2x2"},
      {"non-finite transition block",
       {{.state_name = "position", .A_state_transition = {{"velocity", kNaN * identity(2)}}}},
       {},
       "A_state_transition block ('position', 'velocity') is not finite"},
      {"input block of an input that was not given",
       {{.state_name = "velocity", .B_control_input = {{"jerk", identity(2)}}}},
       {acceleration},
       "B_control_input of state channel 'velocity' names input 'jerk', which predict() was not given; the inputs are: acceleration"},
      {"input block of the wrong size",
       {{.state_name = "velocity", .B_control_input = {{"acceleration", matrix_t::Identity(2, 3)}}}},
       {acceleration},
       "B_control_input block ('velocity', 'acceleration') is 2x3, expected 2x2"},
      {"two models of one channel",
       {{.state_name = "velocity"}, {.state_name = "velocity"}},
       {},
       "state channel 'velocity' is given two process models"},
      {"process noise of the wrong size",
       {{.state_name = "velocity", .Q_process_noise = identity(3)}},
       {},
       "Q_process_noise of state channel 'velocity' is 3x3, expected 2x2"},
      {"indefinite process noise",
       {{.state_name = "velocity", .Q_process_noise = -identity(2)}},
       {},
       "Q_process_noise of state channel 'velocity' is not positive semi-definite"},
      {"repeated input", {}, {acceleration, acceleration}, "input 'acceleration' is given twice"},
      {"input without a name", {}, {{.name = "", .input = makeVector({1.0})}}, "an input has an empty name"},
      {"empty input", {}, {{.name = "acceleration", .input = vector_t()}}, "input 'acceleration' is empty"},
      {"non-finite input",
       {},
       {{.name = "acceleration", .input = makeVector({kNaN, 0.0})}},
       "the value of input 'acceleration' is not finite"},
      {"asymmetric input noise",
       {},
       {{.name = "acceleration", .input = makeVector({1.0, 1.0}), .Q_input_noise = (matrix_t(2, 2) << 1.0, 0.1, 0.0, 1.0).finished()}},
       "Q_input_noise of input 'acceleration' is not symmetric"},
      // Positive variances, but a covariance larger than their geometric mean: a correlation of 2.
      {"input noise with a correlation beyond one",
       {},
       {{.name = "acceleration", .input = makeVector({1.0, 1.0}), .Q_input_noise = (matrix_t(2, 2) << 1.0, 2.0, 2.0, 1.0).finished()}},
       "Q_input_noise of input 'acceleration' is not positive semi-definite: its entry (0, 1) = 2"},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    expectRejectedWithoutChange(filter_->predict(test_case.process_model, test_case.inputs), test_case.expected_message);
  }
}

TEST_F(KalmanFilterRejectionTest, invalidCorrectionsAreRejectedWithoutChangingTheEstimate) {
  struct Case {
    std::string label;
    std::vector<KalmanFilterMeasurement> measurements;
    std::string expected_message;
  };
  const KalmanFilterMeasurement valid = {.name = "position",
                                         .measurement = makeVector({1.0, 2.0}),
                                         .H_measurement_model = {{"position", identity(2)}},
                                         .R_measurement_noise = identity(2)};
  const std::vector<Case> cases = {
      {"block of an unknown channel",
       {{.name = "gps",
         .measurement = makeVector({1.0, 2.0}),
         .H_measurement_model = {{"orientation", identity(2)}},
         .R_measurement_noise = identity(2)}},
       "H_measurement_model of measurement 'gps' names state channel 'orientation'"},
      {"observation block of the wrong size",
       {{.name = "gps",
         .measurement = makeVector({1.0, 2.0}),
         .H_measurement_model = {{"position", identity(3)}},
         .R_measurement_noise = identity(2)}},
       "H_measurement_model block ('gps', 'position') is 3x3, expected 2x2"},
      {"no observation block",
       {{.name = "gps", .measurement = makeVector({1.0, 2.0}), .R_measurement_noise = identity(2)}},
       "measurement 'gps' observes no state channel"},
      {"missing noise",
       {{.name = "gps", .measurement = makeVector({1.0, 2.0}), .H_measurement_model = {{"position", identity(2)}}}},
       "R_measurement_noise of measurement 'gps' is missing"},
      {"noise of the wrong size",
       {{.name = "gps",
         .measurement = makeVector({1.0, 2.0}),
         .H_measurement_model = {{"position", identity(2)}},
         .R_measurement_noise = identity(3)}},
       "R_measurement_noise of measurement 'gps' is 3x3, expected 2x2"},
      {"repeated measurement", {valid, valid}, "measurement 'position' is given twice"},
      {"measurement without a name",
       {{.name = "",
         .measurement = makeVector({1.0, 2.0}),
         .H_measurement_model = {{"position", identity(2)}},
         .R_measurement_noise = identity(2)}},
       "a measurement has an empty name"},
      {"empty measurement",
       {{.name = "gps", .measurement = vector_t(), .H_measurement_model = {{"position", identity(2)}}, .R_measurement_noise = identity(2)}},
       "measurement 'gps' is empty"},
      {"non-finite measurement",
       {{.name = "gps",
         .measurement = makeVector({kNaN, 2.0}),
         .H_measurement_model = {{"position", identity(2)}},
         .R_measurement_noise = identity(2)}},
       "the value of measurement 'gps' is not finite"},
      // Twice the same noise-free direction: H P H^T + R has rank one.
      {"singular innovation covariance",
       {{.name = "first",
         .measurement = makeVector({1.0}),
         .H_measurement_model = {{"velocity", matrix_t::Ones(1, 2)}},
         .R_measurement_noise = scalarMatrix(0.0)},
        {.name = "second",
         .measurement = makeVector({1.0}),
         .H_measurement_model = {{"velocity", matrix_t::Ones(1, 2)}},
         .R_measurement_noise = scalarMatrix(0.0)}},
       "the innovation covariance H P H^T + R of measurements first, second is not positive definite"},
      // The same direction scaled by 1/3: singular in exact arithmetic, and only rounding away from it in floating point,
      // so the Cholesky factorization either fails or leaves a pivot at rounding level, which the pivot test rejects.
      {"innovation covariance singular up to rounding",
       {{.name = "first",
         .measurement = makeVector({1.0}),
         .H_measurement_model = {{"velocity", matrix_t::Ones(1, 2)}},
         .R_measurement_noise = scalarMatrix(0.0)},
        {.name = "third",
         .measurement = makeVector({1.0 / 3.0}),
         .H_measurement_model = {{"velocity", matrix_t::Constant(1, 2, 1.0 / 3.0)}},
         .R_measurement_noise = scalarMatrix(0.0)}},
       "the innovation covariance H P H^T + R of measurements first, third is not positive definite"},
      {"noise with a correlation beyond one",
       {{.name = "gps",
         .measurement = makeVector({1.0, 2.0}),
         .H_measurement_model = {{"position", identity(2)}},
         .R_measurement_noise = (matrix_t(2, 2) << 1.0, -2.0, -2.0, 1.0).finished()}},
       "R_measurement_noise of measurement 'gps' is not positive semi-definite: its entry (0, 1) = -2"},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    expectRejectedWithoutChange(filter_->correct(test_case.measurements), test_case.expected_message);
  }
}

}  // namespace
}  // namespace ocs2::humanoid::estimation
