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

/// Entries uniform in [-1, 1] from a seeded generator, so every run sees the
/// same numbers.
matrix_t randomMatrix(std::mt19937& generator,
                      Eigen::Index rows,
                      Eigen::Index cols) {
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
      const absl::StatusOr<matrix_t> block =
          filter.getCovariance(row_state.name, col_state.name);
      EXPECT_TRUE(block.ok()) << block.status();
      joint.block(row, col, row_state.state.size(), col_state.state.size()) =
          *block;
      col += col_state.state.size();
    }
    row += row_state.state.size();
  }
  return joint;
}

/// |actual - expected| <= tolerance, relative to the largest entry of
/// `expected` once that exceeds one.
void expectNear(const matrix_t& actual,
                const matrix_t& expected,
                scalar_t tolerance) {
  ASSERT_EQ(actual.rows(), expected.rows());
  ASSERT_EQ(actual.cols(), expected.cols());
  const scalar_t scale = std::max(1.0, expected.cwiseAbs().maxCoeff());
  EXPECT_LE((actual - expected).cwiseAbs().maxCoeff(), tolerance * scale)
      << "actual:\n"
      << actual << "\nexpected:\n"
      << expected;
}

KalmanFilterState getStateOrDie(const KalmanFilter& filter,
                                const std::string& name) {
  const absl::StatusOr<KalmanFilterState> state = filter.getState(name);
  EXPECT_TRUE(state.ok()) << state.status();
  return *state;
}

matrix_t getCovarianceOrDie(const KalmanFilter& filter,
                            const std::string& row_name,
                            const std::string& col_name) {
  const absl::StatusOr<matrix_t> covariance =
      filter.getCovariance(row_name, col_name);
  EXPECT_TRUE(covariance.ok()) << covariance.status();
  return *covariance;
}

std::unique_ptr<KalmanFilter> createOrDie(
    absl::Span<const KalmanFilterState> initialStates) {
  absl::StatusOr<std::unique_ptr<KalmanFilter>> filter =
      KalmanFilter::Create(initialStates);
  EXPECT_TRUE(filter.ok()) << filter.status();
  return std::move(*filter);
}

/// The textbook Kalman filter on the joint state, which the channel-organized
/// filter must reproduce. It updates the covariance in the short form (I - K H)
/// P rather than the Joseph form; the two agree for the optimal gain.
struct DenseKalmanFilter {
  vector_t x;
  matrix_t P;

  void predict(const matrix_t& A,
               const matrix_t& B,
               const vector_t& u,
               const matrix_t& Q) {
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
  // State channels a (2), b (3) and c (1), stacked in that order: offsets 0, 2
  // and 5.
  const vector_t a0 = randomMatrix(generator, /*rows=*/2, /*cols=*/1);
  const vector_t b0 = randomMatrix(generator, /*rows=*/3, /*cols=*/1);
  const vector_t c0 = randomMatrix(generator, /*rows=*/1, /*cols=*/1);
  const matrix_t P_a = randomCovariance(generator, /*dim=*/2);
  const matrix_t P_b = randomCovariance(generator, /*dim=*/3);
  const matrix_t P_c = randomCovariance(generator, /*dim=*/1);
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "a", .state = a0, .stateCovariance = P_a},
                   {.name = "b", .state = b0, .stateCovariance = P_b},
                   {.name = "c", .state = c0, .stateCovariance = P_c}});

  DenseKalmanFilter reference;
  reference.x = (vector_t(6) << a0, b0, c0).finished();
  reference.P = matrix_t::Zero(6, 6);
  reference.P.block(0, 0, 2, 2) = P_a;
  reference.P.block(2, 2, 3, 3) = P_b;
  reference.P.block(5, 5, 1, 1) = P_c;

  // a depends on itself, on b and on input u1, with additive noise. b lists no
  // A_bb, so it keeps the identity, depends on c and on the noise-free input
  // u2. c has no process model at all, so it is held constant.
  const matrix_t A_aa = randomMatrix(generator, /*rows=*/2, /*cols=*/2);
  const matrix_t A_ab = randomMatrix(generator, /*rows=*/2, /*cols=*/3);
  const matrix_t A_bc = randomMatrix(generator, /*rows=*/3, /*cols=*/1);
  const matrix_t B_a_u1 = randomMatrix(generator, /*rows=*/2, /*cols=*/1);
  const matrix_t B_b_u2 = randomMatrix(generator, /*rows=*/3, /*cols=*/2);
  const matrix_t Q_a = 0.01 * randomCovariance(generator, /*dim=*/2);
  const matrix_t Q_u1 = 0.01 * randomCovariance(generator, /*dim=*/1);
  const std::vector<KalmanFilterProcessModel> processModel = {
      {.stateName = "a",
       .stateTransition = {{"a", A_aa}, {"b", A_ab}},
       .controlInputMatrix = {{"u1", B_a_u1}},
       .processNoiseCovariance = Q_a},
      {.stateName = "b",
       .stateTransition = {{"c", A_bc}},
       .controlInputMatrix = {{"u2", B_b_u2}}},
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
  const std::function<void(const std::string&)> expectMatchesReference =
      [&](const std::string& step) {
        SCOPED_TRACE(step);
        for (size_t i = 0; i < names.size(); ++i) {
          const KalmanFilterState state = getStateOrDie(*filter, names[i]);
          EXPECT_EQ(state.name, names[i]);
          expectNear(state.state, reference.x.segment(offsets[i], sizes[i]),
                     /*tolerance=*/1.0e-9);
          expectNear(
              state.stateCovariance,
              reference.P.block(offsets[i], offsets[i], sizes[i], sizes[i]),
              /*tolerance=*/1.0e-9);
          for (size_t j = 0; j < names.size(); ++j) {
            expectNear(
                getCovarianceOrDie(*filter, names[i], names[j]),
                reference.P.block(offsets[i], offsets[j], sizes[i], sizes[j]),
                /*tolerance=*/1.0e-9);
          }
        }
      };

  for (int step = 0; step < 5; ++step) {
    const vector_t u1 = randomMatrix(generator, /*rows=*/1, /*cols=*/1);
    const vector_t u2 = randomMatrix(generator, /*rows=*/2, /*cols=*/1);
    ASSERT_TRUE(filter
                    ->predict(processModel, {{.name = "u1",
                                              .input = u1,
                                              .inputNoiseCovariance = Q_u1},
                                             {.name = "u2", .input = u2}})
                    .ok());
    reference.predict(A, B, (vector_t(3) << u1, u2).finished(), Q);
    expectMatchesReference(absl::StrCat("predict ", step));

    const vector_t z1 = randomMatrix(generator, /*rows=*/2, /*cols=*/1);
    const vector_t z2 = randomMatrix(generator, /*rows=*/1, /*cols=*/1);
    ASSERT_TRUE(
        filter
            ->correct({{.name = "m1",
                        .measurement = z1,
                        .measurementModel = {{"a", H_m1_a}, {"c", H_m1_c}},
                        .measurementNoiseCovariance = R_m1},
                       {.name = "m2",
                        .measurement = z2,
                        .measurementModel = {{"b", H_m2_b}},
                        .measurementNoiseCovariance = R_m2}})
            .ok());
    reference.correct(H, (vector_t(3) << z1, z2).finished(), R);
    expectMatchesReference(absl::StrCat("correct ", step));
  }
}

TEST(KalmanFilterTest, aScalarRandomWalkFollowsTheClosedForm) {
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "x",
                    .state = makeVector({1.0}),
                    .stateCovariance = scalarMatrix(1.0)}});

  ASSERT_TRUE(filter
                  ->predict({{.stateName = "x",
                              .processNoiseCovariance = scalarMatrix(0.5)}},
                            /*inputs=*/{})
                  .ok());
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "x").state(0), 1.0);
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "x").stateCovariance(0, 0), 1.5);

  // S = 1.5 + 2 = 3.5 and K = 1.5 / 3.5 = 3/7, so x = 1 + 3/7 (3 - 1) = 13/7
  // and P = (1 - 3/7) 1.5 = 6/7.
  ASSERT_TRUE(filter
                  ->correct({{.name = "z",
                              .measurement = makeVector({3.0}),
                              .measurementModel = {{"x", scalarMatrix(1.0)}},
                              .measurementNoiseCovariance = scalarMatrix(2.0)}})
                  .ok());
  EXPECT_NEAR(getStateOrDie(*filter, "x").state(0), 13.0 / 7.0, 1.0e-15);
  EXPECT_NEAR(getStateOrDie(*filter, "x").stateCovariance(0, 0), 6.0 / 7.0,
              1.0e-15);
}

TEST(KalmanFilterTest, aChannelWithoutAProcessModelKeepsItsMeanAndCovariance) {
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "moving",
                    .state = makeVector({1.0}),
                    .stateCovariance = scalarMatrix(1.0)},
                   {.name = "still",
                    .state = makeVector({2.0, 3.0}),
                    .stateCovariance = 2.0 * identity(2)}});
  // Correlate the two channels, so that the cross-covariance has something to
  // show.
  ASSERT_TRUE(
      filter
          ->correct({{.name = "sum",
                      .measurement = makeVector({4.0}),
                      .measurementModel = {{"moving", scalarMatrix(1.0)},
                                           {"still", matrix_t::Ones(1, 2)}},
                      .measurementNoiseCovariance = scalarMatrix(0.5)}})
          .ok());
  const KalmanFilterState still = getStateOrDie(*filter, "still");
  const matrix_t P_moving_still =
      getCovarianceOrDie(*filter, "moving", "still");
  ASSERT_GT(P_moving_still.cwiseAbs().maxCoeff(), 0.0);

  ASSERT_TRUE(
      filter
          ->predict({{.stateName = "moving",
                      .stateTransition = {{"moving", scalarMatrix(2.0)}}}},
                    /*inputs=*/{})
          .ok());

  // x_still[k+1] = x_still[k]: unchanged; the cross-covariance follows
  // A_moving,moving P_moving,still.
  EXPECT_TRUE(getStateOrDie(*filter, "still").state == still.state);
  EXPECT_TRUE(getStateOrDie(*filter, "still").stateCovariance ==
              still.stateCovariance);
  expectNear(getCovarianceOrDie(*filter, "moving", "still"),
             2.0 * P_moving_still, /*tolerance=*/1.0e-15);
  expectNear(getCovarianceOrDie(*filter, "still", "moving"),
             2.0 * P_moving_still.transpose(), /*tolerance=*/1.0e-15);
}

TEST(KalmanFilterTest,
     anUnlistedDiagonalBlockIsTheIdentityAndAZeroOneForgetsThePast) {
  const scalar_t dt = 0.1;
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "position",
                    .state = makeVector({1.0}),
                    .stateCovariance = scalarMatrix(0.0)},
                   {.name = "velocity",
                    .state = makeVector({2.0}),
                    .stateCovariance = scalarMatrix(0.0)}});

  // position lists only the velocity block, so p[k+1] = p[k] + dt v[k].
  ASSERT_TRUE(
      filter
          ->predict({{.stateName = "position",
                      .stateTransition = {{"velocity", scalarMatrix(dt)}}}},
                    /*inputs=*/{})
          .ok());
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "position").state(0), 1.0 + dt * 2.0);
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "velocity").state(0), 2.0);

  // velocity lists its own block as zero, so v[k+1] = u[k] and its covariance
  // is the noise of u alone.
  ASSERT_TRUE(
      filter
          ->predict({{.stateName = "velocity",
                      .stateTransition = {{"velocity", scalarMatrix(0.0)}},
                      .controlInputMatrix = {{"commanded_velocity",
                                              scalarMatrix(1.0)}}}},
                    {{.name = "commanded_velocity",
                      .input = makeVector({-5.0}),
                      .inputNoiseCovariance = scalarMatrix(0.25)}})
          .ok());
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "velocity").state(0), -5.0);
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "velocity").stateCovariance(0, 0),
                   0.25);
  EXPECT_DOUBLE_EQ(getStateOrDie(*filter, "position").state(0), 1.0 + dt * 2.0);
}

TEST(KalmanFilterTest, inputNoiseIsCorrelatedAcrossTheChannelsItDrives) {
  const scalar_t dt = 0.01;
  const scalar_t sigma = 0.3;
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "position",
                    .state = vector_t::Zero(3),
                    .stateCovariance = matrix_t::Zero(3, 3)},
                   {.name = "velocity",
                    .state = vector_t::Zero(3),
                    .stateCovariance = matrix_t::Zero(3, 3)}});
  const vector_t controlInput = makeVector({1.0, -2.0, 0.5});

  ASSERT_TRUE(
      filter
          ->predict(
              {{.stateName = "position",
                .stateTransition = {{"velocity", dt * identity(3)}},
                .controlInputMatrix = {{"acceleration",
                                        0.5 * dt * dt * identity(3)}}},
               {.stateName = "velocity",
                .controlInputMatrix = {{"acceleration", dt * identity(3)}}}},
              {{.name = "acceleration",
                .input = controlInput,
                .inputNoiseCovariance = sigma * sigma * identity(3)}})
          .ok());

  expectNear(getStateOrDie(*filter, "position").state,
             0.5 * dt * dt * controlInput, /*tolerance=*/1.0e-15);
  expectNear(getStateOrDie(*filter, "velocity").state, dt * controlInput,
             /*tolerance=*/1.0e-15);
  const scalar_t variance = sigma * sigma;
  expectNear(getCovarianceOrDie(*filter, "position", "position"),
             0.25 * dt * dt * dt * dt * variance * identity(3),
             /*tolerance=*/1.0e-15);
  expectNear(getCovarianceOrDie(*filter, "position", "velocity"),
             0.5 * dt * dt * dt * variance * identity(3),
             /*tolerance=*/1.0e-15);
  expectNear(getCovarianceOrDie(*filter, "velocity", "position"),
             0.5 * dt * dt * dt * variance * identity(3),
             /*tolerance=*/1.0e-15);
  expectNear(getCovarianceOrDie(*filter, "velocity", "velocity"),
             dt * dt * variance * identity(3), /*tolerance=*/1.0e-15);
}

// The property the joint covariance exists for: velocity is never measured, and
// is still corrected, through the cross-covariance that p[k+1] = p[k] + dt v[k]
// builds up. A covariance stored channel by channel never corrects it.
TEST(KalmanFilterTest,
     positionMeasurementsCorrectTheVelocityThroughTheCrossCovariance) {
  const scalar_t dt = 0.01;
  const scalar_t true_rate = 0.8;
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "position",
                    .state = makeVector({0.0}),
                    .stateCovariance = scalarMatrix(1.0e-6)},
                   {.name = "velocity",
                    .state = makeVector({0.0}),
                    .stateCovariance = scalarMatrix(1.0)}});
  const std::vector<KalmanFilterProcessModel> constant_velocity_model = {
      {.stateName = "position",
       .stateTransition = {{"velocity", scalarMatrix(dt)}}}};

  for (int step = 1; step <= 200; ++step) {
    ASSERT_TRUE(filter->predict(constant_velocity_model, /*inputs=*/{}).ok());
    ASSERT_TRUE(
        filter
            ->correct({{.name = "position",
                        .measurement = makeVector({true_rate * dt * step}),
                        .measurementModel = {{"position", scalarMatrix(1.0)}},
                        .measurementNoiseCovariance = scalarMatrix(1.0e-6)}})
            .ok());
    if (step == 1) {
      EXPECT_GT(getStateOrDie(*filter, "velocity").state(0), 0.0)
          << "the first position measurement must already move the velocity";
    }
  }
  EXPECT_NEAR(getStateOrDie(*filter, "velocity").state(0), true_rate, 1.0e-3);
  EXPECT_LT(getStateOrDie(*filter, "velocity").stateCovariance(0, 0), 1.0e-3);
}

TEST(KalmanFilterTest, aJointCorrectionEqualsSequentialCorrectionsInAnyOrder) {
  std::mt19937 generator(7);
  const std::vector<KalmanFilterState> initialStates = {
      {.name = "a",
       .state = randomMatrix(generator, /*rows=*/2, /*cols=*/1),
       .stateCovariance = randomCovariance(generator, /*dim=*/2)},
      {.name = "b",
       .state = randomMatrix(generator, /*rows=*/2, /*cols=*/1),
       .stateCovariance = randomCovariance(generator, /*dim=*/2)}};
  // A coupled transition, so that the channels are correlated before the
  // corrections.
  const std::vector<KalmanFilterProcessModel> processModel = {
      {.stateName = "a",
       .stateTransition = {{"b",
                            randomMatrix(generator, /*rows=*/2, /*cols=*/2)}}},
      {.stateName = "b",
       .stateTransition = {
           {"a", randomMatrix(generator, /*rows=*/2, /*cols=*/2)}}}};
  const KalmanFilterMeasurement first = {
      .name = "first",
      .measurement = randomMatrix(generator, /*rows=*/2, /*cols=*/1),
      .measurementModel = {{"a",
                            randomMatrix(generator, /*rows=*/2, /*cols=*/2)}},
      .measurementNoiseCovariance = randomCovariance(generator, /*dim=*/2)};
  const KalmanFilterMeasurement second = {
      .name = "second",
      .measurement = randomMatrix(generator, /*rows=*/1, /*cols=*/1),
      .measurementModel = {{"a",
                            randomMatrix(generator, /*rows=*/1, /*cols=*/2)},
                           {"b",
                            randomMatrix(generator, /*rows=*/1, /*cols=*/2)}},
      .measurementNoiseCovariance = randomCovariance(generator, /*dim=*/1)};

  std::unique_ptr<KalmanFilter> joint = createOrDie(initialStates);
  std::unique_ptr<KalmanFilter> reversed = createOrDie(initialStates);
  std::unique_ptr<KalmanFilter> sequential = createOrDie(initialStates);
  for (KalmanFilter* absl_nonnull filter :
       {joint.get(), reversed.get(), sequential.get()}) {
    ASSERT_TRUE(filter->predict(processModel, /*inputs=*/{}).ok());
  }
  ASSERT_TRUE(joint->correct({first, second}).ok());
  ASSERT_TRUE(reversed->correct({second, first}).ok());
  ASSERT_TRUE(sequential->correct({first}).ok());
  ASSERT_TRUE(sequential->correct({second}).ok());

  expectNear(jointState(*reversed), jointState(*joint), /*tolerance=*/1.0e-12);
  expectNear(jointCovariance(*reversed), jointCovariance(*joint),
             /*tolerance=*/1.0e-12);
  expectNear(jointState(*sequential), jointState(*joint),
             /*tolerance=*/1.0e-12);
  expectNear(jointCovariance(*sequential), jointCovariance(*joint),
             /*tolerance=*/1.0e-12);
}

TEST(
    KalmanFilterTest,
    theCovarianceStaysSymmetricPositiveSemiDefiniteUnderNearlyExactMeasurements) {
  std::mt19937 generator(3);
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "a",
                    .state = vector_t::Zero(3),
                    .stateCovariance = 1.0e3 * identity(3)},
                   {.name = "b",
                    .state = vector_t::Zero(3),
                    .stateCovariance = 1.0e-3 * identity(3)}});
  for (int step = 0; step < 200; ++step) {
    ASSERT_TRUE(
        filter
            ->predict({{.stateName = "a",
                        .stateTransition = {{"b", 0.01 * identity(3)}},
                        .processNoiseCovariance = 1.0e-8 * identity(3)}},
                      /*inputs=*/{})
            .ok());
    ASSERT_TRUE(
        filter
            ->correct(
                {{.name = "a",
                  .measurement =
                      randomMatrix(generator, /*rows=*/3, /*cols=*/1),
                  .measurementModel = {{"a", identity(3)},
                                       {"b", randomMatrix(generator, /*rows=*/3,
                                                          /*cols=*/3)}},
                  .measurementNoiseCovariance = 1.0e-10 * identity(3)}})
            .ok());
    const matrix_t P = jointCovariance(*filter);
    ASSERT_TRUE(P == P.transpose()) << "step " << step;
    const Eigen::SelfAdjointEigenSolver<matrix_t> eigen_solver(
        P, Eigen::EigenvaluesOnly);
    ASSERT_GE(eigen_solver.eigenvalues().minCoeff(),
              -1.0e-12 * P.cwiseAbs().maxCoeff())
        << "step " << step;
  }
}

TEST(KalmanFilterTest,
     getStatesReturnsTheChannelsInDeclarationOrderAndRoundTripsThroughReset) {
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "z_last_alphabetically",
                    .state = makeVector({1.0}),
                    .stateCovariance = scalarMatrix(1.0)},
                   {.name = "a_first_alphabetically",
                    .state = makeVector({2.0, 3.0}),
                    .stateCovariance = identity(2)}});
  ASSERT_TRUE(filter
                  ->correct({{.name = "sum",
                              .measurement = makeVector({4.0}),
                              .measurementModel = {{"z_last_alphabetically",
                                                    scalarMatrix(1.0)},
                                                   {"a_first_alphabetically",
                                                    matrix_t::Ones(1, 2)}},
                              .measurementNoiseCovariance = scalarMatrix(1.0)}})
                  .ok());

  const std::vector<KalmanFilterState> states = filter->getStates();
  ASSERT_EQ(states.size(), 2u);
  EXPECT_EQ(states[0].name, "z_last_alphabetically");
  EXPECT_EQ(states[1].name, "a_first_alphabetically");

  KalmanFilter copy;
  ASSERT_TRUE(copy.reset(states).ok());
  for (const KalmanFilterState& state : states) {
    EXPECT_TRUE(getStateOrDie(copy, state.name).state == state.state);
    EXPECT_TRUE(getStateOrDie(copy, state.name).stateCovariance ==
                state.stateCovariance);
  }
  // reset() declares the channels uncorrelated.
  EXPECT_TRUE(getCovarianceOrDie(copy, "z_last_alphabetically",
                                 "a_first_alphabetically")
                  .isZero(0.0));
}

TEST(KalmanFilterTest,
     resetEstimateEqualsResetOnTheSameValuesAndRefusesAnotherLayout) {
  const std::vector<KalmanFilterState> initial = {
      {.name = "position",
       .state = makeVector({1.0, 2.0}),
       .stateCovariance = identity(2)},
      {.name = "velocity",
       .state = makeVector({0.5}),
       .stateCovariance = scalarMatrix(4.0)}};
  std::unique_ptr<KalmanFilter> filter = createOrDie(initial);
  // A correction that correlates the channels, which resetEstimate() must clear
  // as reset() does.
  ASSERT_TRUE(
      filter
          ->correct({{.name = "sum",
                      .measurement = makeVector({3.0}),
                      .measurementModel = {{"position", matrix_t::Ones(1, 2)},
                                           {"velocity", scalarMatrix(1.0)}},
                      .measurementNoiseCovariance = scalarMatrix(0.1)}})
          .ok());
  const std::vector<KalmanFilterState> restart = {
      {.name = "position",
       .state = makeVector({-1.0, 0.25}),
       .stateCovariance = 0.5 * identity(2)},
      {.name = "velocity",
       .state = makeVector({2.0}),
       .stateCovariance = scalarMatrix(0.3)}};
  ASSERT_TRUE(filter->resetEstimate(restart).ok());
  std::unique_ptr<KalmanFilter> reference = createOrDie(restart);
  for (const KalmanFilterState& state : restart) {
    EXPECT_TRUE(getStateOrDie(*filter, state.name).state ==
                getStateOrDie(*reference, state.name).state);
    EXPECT_TRUE(getStateOrDie(*filter, state.name).stateCovariance ==
                getStateOrDie(*reference, state.name).stateCovariance);
  }
  EXPECT_TRUE(getCovarianceOrDie(*filter, "position", "velocity").isZero(0.0));

  // Another order, another size, a missing channel or an invalid covariance:
  // refused, the estimate kept.
  const std::vector<KalmanFilterState> swapped = {restart[1], restart[0]};
  EXPECT_EQ(filter->resetEstimate(swapped).code(),
            absl::StatusCode::kInvalidArgument);
  std::vector<KalmanFilterState> resized = restart;
  resized[1].state = makeVector({1.0, 2.0});
  resized[1].stateCovariance = identity(2);
  EXPECT_EQ(filter->resetEstimate(resized).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(filter->resetEstimate({restart[0]}).code(),
            absl::StatusCode::kInvalidArgument);
  std::vector<KalmanFilterState> negative = restart;
  negative[1].stateCovariance = scalarMatrix(-1.0);
  EXPECT_EQ(filter->resetEstimate(negative).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(getStateOrDie(*filter, "velocity").state == restart[1].state);
  EXPECT_EQ(KalmanFilter().resetEstimate(restart).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(KalmanFilterTest, anEmptyCorrectionLeavesTheEstimateUnchanged) {
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "x",
                    .state = makeVector({1.0, 2.0}),
                    .stateCovariance = identity(2)}});
  const vector_t x = jointState(*filter);
  const matrix_t P = jointCovariance(*filter);
  ASSERT_TRUE(filter->correct(/*measurements=*/{}).ok());
  EXPECT_TRUE(jointState(*filter) == x);
  EXPECT_TRUE(jointCovariance(*filter) == P);
}

// A legged robot's base, driven by a world-frame IMU acceleration and observed
// only through the position of a stance foot relative to the base, as leg
// kinematics measures it. The base velocity starts wrong; the foot does not
// move.
TEST(KalmanFilterTest,
     legKinematicsCorrectTheBaseVelocityOfAContactAidedEstimator) {
  const scalar_t dt = 0.002;
  const vector_t true_pos_foot_rt_world_ewrt_world =
      makeVector({0.1, -0.1, 0.0});
  const vector_t linacc_base_rt_world_ewrt_world = makeVector({0.2, 0.0, -0.1});
  vector_t true_pos_base_rt_world_ewrt_world = makeVector({0.0, 0.0, 0.9});
  vector_t true_linvel_base_rt_world_ewrt_world = makeVector({0.5, -0.2, 0.0});

  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "base_position",
                    .state = true_pos_base_rt_world_ewrt_world,
                    .stateCovariance = 1.0e-4 * identity(3)},
                   {.name = "base_velocity",
                    .state = vector_t::Zero(3),
                    .stateCovariance = identity(3)},
                   {.name = "foot_position",
                    .state = true_pos_foot_rt_world_ewrt_world +
                             makeVector({0.01, 0.01, 0.0}),
                    .stateCovariance = 1.0e-2 * identity(3)}});
  const std::vector<KalmanFilterProcessModel> processModel = {
      {.stateName = "base_position",
       .stateTransition = {{"base_velocity", dt * identity(3)}},
       .controlInputMatrix = {{"imu_acceleration",
                               0.5 * dt * dt * identity(3)}}},
      {.stateName = "base_velocity",
       .controlInputMatrix = {{"imu_acceleration", dt * identity(3)}}},
      {.stateName = "foot_position",
       .processNoiseCovariance = 1.0e-10 * identity(3)},
  };

  for (int step = 0; step < 500; ++step) {
    true_pos_base_rt_world_ewrt_world +=
        dt * true_linvel_base_rt_world_ewrt_world +
        0.5 * dt * dt * linacc_base_rt_world_ewrt_world;
    true_linvel_base_rt_world_ewrt_world +=
        dt * linacc_base_rt_world_ewrt_world;
    ASSERT_TRUE(filter
                    ->predict(processModel,
                              {{.name = "imu_acceleration",
                                .input = linacc_base_rt_world_ewrt_world,
                                .inputNoiseCovariance = 1.0e-4 * identity(3)}})
                    .ok());
    ASSERT_TRUE(
        filter
            ->correct({{.name = "foot_relative_position",
                        .measurement = true_pos_foot_rt_world_ewrt_world -
                                       true_pos_base_rt_world_ewrt_world,
                        .measurementModel = {{"foot_position", identity(3)},
                                             {"base_position", -identity(3)}},
                        .measurementNoiseCovariance = 1.0e-6 * identity(3)}})
            .ok());
  }

  expectNear(getStateOrDie(*filter, "base_velocity").state,
             true_linvel_base_rt_world_ewrt_world,
             /*tolerance=*/1.0e-2);
  const vector_t est_pos_foot_rt_base_ewrt_world =
      getStateOrDie(*filter, "foot_position").state -
      getStateOrDie(*filter, "base_position").state;
  expectNear(
      est_pos_foot_rt_base_ewrt_world,
      true_pos_foot_rt_world_ewrt_world - true_pos_base_rt_world_ewrt_world,
      /*tolerance=*/1.0e-3);
  // Only foot minus base is measured, so their absolute positions remain
  // uncertain together: positively correlated.
  const matrix_t P_base_foot =
      getCovarianceOrDie(*filter, "base_position", "foot_position");
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
      {"empty name",
       {{.name = "",
         .state = makeVector({1.0}),
         .stateCovariance = scalarMatrix(1.0)}},
       "empty name"},
      {"repeated name",
       {{.name = "x",
         .state = makeVector({1.0}),
         .stateCovariance = scalarMatrix(1.0)},
        {.name = "x",
         .state = makeVector({1.0}),
         .stateCovariance = scalarMatrix(1.0)}},
       "state channel 'x' is declared twice"},
      {"empty state",
       {{.name = "x", .state = vector_t(), .stateCovariance = matrix_t()}},
       "state channel 'x' is empty"},
      {"non-finite state",
       {{.name = "x",
         .state = makeVector({kNaN}),
         .stateCovariance = scalarMatrix(1.0)}},
       "state of state channel 'x'"},
      {"covariance of the wrong size",
       {{.name = "x",
         .state = vector_t::Zero(3),
         .stateCovariance = identity(2)}},
       "stateCovariance of state channel 'x' is 2x2, expected 3x3"},
      {"asymmetric covariance",
       {{.name = "x",
         .state = vector_t::Zero(2),
         .stateCovariance = (matrix_t(2, 2) << 1.0, 0.5, 0.0, 1.0).finished()}},
       "stateCovariance of state channel 'x' is not symmetric"},
      {"indefinite covariance",
       {{.name = "x",
         .state = vector_t::Zero(2),
         .stateCovariance = (matrix_t(2, 2) << 1.0, 2.0, 2.0, 1.0).finished()}},
       "stateCovariance of state channel 'x' is not positive semi-definite"},
      {"non-finite covariance",
       {{.name = "x",
         .state = makeVector({1.0}),
         .stateCovariance = scalarMatrix(kNaN)}},
       "is not finite"},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    const absl::StatusOr<std::unique_ptr<KalmanFilter>> filter =
        KalmanFilter::Create(test_case.states);
    ASSERT_FALSE(filter.ok());
    EXPECT_EQ(filter.status().code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(filter.status().message(),
                HasSubstr(test_case.expected_message));
  }
}

TEST(KalmanFilterTest, aRejectedResetKeepsThePreviousChannels) {
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "x",
                    .state = makeVector({1.0}),
                    .stateCovariance = scalarMatrix(1.0)}});
  EXPECT_FALSE(filter
                   ->reset({{.name = "y",
                             .state = makeVector({2.0}),
                             .stateCovariance = scalarMatrix(1.0)},
                            {.name = "z",
                             .state = makeVector({kNaN}),
                             .stateCovariance = scalarMatrix(1.0)}})
                   .ok());
  const std::vector<KalmanFilterState> states = filter->getStates();
  ASSERT_EQ(states.size(), 1u);
  EXPECT_EQ(states[0].name, "x");
  EXPECT_DOUBLE_EQ(states[0].state(0), 1.0);
}

TEST(KalmanFilterTest, aFilterWithoutChannelsRefusesToStepUntilReset) {
  KalmanFilter filter;
  EXPECT_TRUE(filter.getStates().empty());

  const absl::Status predicted =
      filter.predict(/*processModel=*/{}, /*inputs=*/{});
  EXPECT_EQ(predicted.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(predicted.message(), HasSubstr("predict() before reset()"));
  const absl::Status corrected = filter.correct(/*measurements=*/{});
  EXPECT_EQ(corrected.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_THAT(corrected.message(), HasSubstr("correct() before reset()"));

  ASSERT_TRUE(filter
                  .reset({{.name = "x",
                           .state = makeVector({1.0}),
                           .stateCovariance = scalarMatrix(1.0)}})
                  .ok());
  EXPECT_TRUE(filter.predict(/*processModel=*/{}, /*inputs=*/{}).ok());
}

TEST(KalmanFilterTest,
     unknownChannelLookupsAreNotFoundAndListTheDeclaredChannels) {
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "base_position",
                    .state = vector_t::Zero(3),
                    .stateCovariance = identity(3)},
                   {.name = "base_velocity",
                    .state = vector_t::Zero(3),
                    .stateCovariance = identity(3)}});
  const absl::StatusOr<KalmanFilterState> state =
      filter->getState("base_orientation");
  EXPECT_EQ(state.status().code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(state.status().message(), HasSubstr("'base_orientation'"));
  EXPECT_THAT(
      state.status().message(),
      HasSubstr("the state channels are: base_position, base_velocity"));
  EXPECT_EQ(filter->getCovariance("base_position", "base_orientation")
                .status()
                .code(),
            absl::StatusCode::kNotFound);
  EXPECT_EQ(filter->getCovariance("base_orientation", "base_position")
                .status()
                .code(),
            absl::StatusCode::kNotFound);
}

TEST(KalmanFilterTest,
     getStateFillsACallerOwnedStateAndLeavesItAloneForAnUnknownName) {
  std::unique_ptr<KalmanFilter> filter =
      createOrDie({{.name = "a",
                    .state = makeVector({1.0, 2.0}),
                    .stateCovariance = identity(2)},
                   {.name = "b",
                    .state = makeVector({3.0}),
                    .stateCovariance = scalarMatrix(4.0)}});
  KalmanFilterState state;
  ASSERT_TRUE(filter->getState("a", state).ok());
  EXPECT_EQ(state.name, "a");
  EXPECT_TRUE(state.state == makeVector({1.0, 2.0}));
  EXPECT_TRUE(state.stateCovariance == identity(2));

  // The same object then reads a channel of another size.
  ASSERT_TRUE(filter->getState("b", state).ok());
  EXPECT_EQ(state.name, "b");
  EXPECT_TRUE(state.state == makeVector({3.0}));
  EXPECT_TRUE(state.stateCovariance == scalarMatrix(4.0));

  const absl::Status unknown = filter->getState("c", state);
  EXPECT_EQ(unknown.code(), absl::StatusCode::kNotFound);
  EXPECT_THAT(unknown.message(), HasSubstr("the state channels are: a, b"));
  EXPECT_EQ(state.name, "b");
  EXPECT_TRUE(state.state == makeVector({3.0}));
}

TEST(KalmanFilterTest, reserveOnlySizesStorageBeforeOrAfterReset) {
  const std::vector<KalmanFilterState> initialStates = {
      {.name = "position",
       .state = makeVector({0.0, 1.0}),
       .stateCovariance = identity(2)},
      {.name = "velocity",
       .state = makeVector({1.0, 0.0}),
       .stateCovariance = 2.0 * identity(2)}};
  const std::vector<KalmanFilterProcessModel> processModel = {
      {.stateName = "position",
       .stateTransition = {{"velocity", 0.1 * identity(2)}},
       .processNoiseCovariance = 1.0e-3 * identity(2)},
      {.stateName = "velocity",
       .controlInputMatrix = {{"acceleration", 0.1 * identity(2)}}}};
  const std::vector<KalmanFilterInput> inputs = {
      {.name = "acceleration",
       .input = makeVector({0.5, -0.5}),
       .inputNoiseCovariance = 1.0e-2 * identity(2)}};
  const std::vector<KalmanFilterMeasurement> measurements = {
      {.name = "position",
       .measurement = makeVector({0.2, 0.9}),
       .measurementModel = {{"position", identity(2)}},
       .measurementNoiseCovariance = 1.0e-2 * identity(2)}};

  std::unique_ptr<KalmanFilter> unreserved = createOrDie(initialStates);
  KalmanFilter reserved_before_reset;
  ASSERT_TRUE(
      reserved_before_reset.reserve(/*maxInputDim=*/8, /*maxMeasurementDim=*/16)
          .ok());
  ASSERT_TRUE(reserved_before_reset.reset(initialStates).ok());
  std::unique_ptr<KalmanFilter> reserved_after_reset =
      createOrDie(initialStates);
  ASSERT_TRUE(
      reserved_after_reset->reserve(/*maxInputDim=*/8, /*maxMeasurementDim=*/16)
          .ok());

  for (KalmanFilter* absl_nonnull filter :
       {unreserved.get(), &reserved_before_reset, reserved_after_reset.get()}) {
    for (int step = 0; step < 3; ++step) {
      ASSERT_TRUE(filter->predict(processModel, inputs).ok());
      ASSERT_TRUE(filter->correct(measurements).ok());
    }
  }
  for (KalmanFilter* absl_nonnull filter :
       {&reserved_before_reset, reserved_after_reset.get()}) {
    expectNear(jointState(*filter), jointState(*unreserved),
               /*tolerance=*/1.0e-14);
    expectNear(jointCovariance(*filter), jointCovariance(*unreserved),
               /*tolerance=*/1.0e-14);
  }

  const absl::Status negative =
      unreserved->reserve(/*maxInputDim=*/-1, /*maxMeasurementDim=*/0);
  EXPECT_EQ(negative.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(negative.message(), HasSubstr("non-negative dimensions"));
}

class KalmanFilterRejectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    filter_ = createOrDie({{.name = "position",
                            .state = makeVector({1.0, 2.0}),
                            .stateCovariance = identity(2)},
                           {.name = "velocity",
                            .state = makeVector({3.0, 4.0}),
                            .stateCovariance = 2.0 * identity(2)}});
    // Correlate the channels, so that an accidental write anywhere in the
    // covariance shows.
    ASSERT_TRUE(
        filter_
            ->predict({{.stateName = "position",
                        .stateTransition = {{"velocity", 0.1 * identity(2)}}}},
                      /*inputs=*/{})
            .ok());
    x_before_ = jointState(*filter_);
    P_before_ = jointCovariance(*filter_);
  }

  void expectRejectedWithoutChange(const absl::Status& status,
                                   const std::string& expected_message) {
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
    EXPECT_THAT(status.message(), HasSubstr(expected_message));
    EXPECT_TRUE(jointState(*filter_) == x_before_);
    EXPECT_TRUE(jointCovariance(*filter_) == P_before_);
  }

  std::unique_ptr<KalmanFilter> filter_;
  vector_t x_before_;
  matrix_t P_before_;
};

TEST_F(KalmanFilterRejectionTest,
       invalidPredictionsAreRejectedWithoutChangingTheEstimate) {
  struct Case {
    std::string label;
    std::vector<KalmanFilterProcessModel> processModel;
    std::vector<KalmanFilterInput> inputs;
    std::string expected_message;
  };
  const KalmanFilterInput control_input = {.name = "acceleration",
                                           .input = makeVector({1.0, 1.0})};
  const std::vector<Case> cases = {
      {"model of an unknown channel",
       {{.stateName = "orientation"}},
       {},
       "a process model names state channel 'orientation', which reset() did "
       "not declare; the state channels are: position, velocity"},
      {"block of an unknown channel",
       {{.stateName = "position",
         .stateTransition = {{"orientation", identity(2)}}}},
       {},
       "stateTransition of state channel 'position' names state channel "
       "'orientation'"},
      {"transition block of the wrong size",
       {{.stateName = "position",
         .stateTransition = {{"velocity", identity(3)}}}},
       {},
       "stateTransition block ('position', 'velocity') is 3x3, expected 2x2"},
      {"non-finite transition block",
       {{.stateName = "position",
         .stateTransition = {{"velocity", kNaN * identity(2)}}}},
       {},
       "stateTransition block ('position', 'velocity') is not finite"},
      {"input block of an input that was not given",
       {{.stateName = "velocity",
         .controlInputMatrix = {{"jerk", identity(2)}}}},
       {control_input},
       "controlInputMatrix of state channel 'velocity' names input 'jerk', "
       "which predict() was not given; the inputs are: acceleration"},
      {"input block of the wrong size",
       {{.stateName = "velocity",
         .controlInputMatrix = {{"acceleration", matrix_t::Identity(2, 3)}}}},
       {control_input},
       "controlInputMatrix block ('velocity', 'acceleration') is 2x3, expected "
       "2x2"},
      {"two models of one channel",
       {{.stateName = "velocity"}, {.stateName = "velocity"}},
       {},
       "state channel 'velocity' is given two process models"},
      {"process noise of the wrong size",
       {{.stateName = "velocity", .processNoiseCovariance = identity(3)}},
       {},
       "processNoiseCovariance of state channel 'velocity' is 3x3, expected "
       "2x2"},
      {"indefinite process noise",
       {{.stateName = "velocity", .processNoiseCovariance = -identity(2)}},
       {},
       "processNoiseCovariance of state channel 'velocity' is not positive "
       "semi-definite"},
      {"repeated input",
       {},
       {control_input, control_input},
       "input 'acceleration' is given twice"},
      {"input without a name",
       {},
       {{.name = "", .input = makeVector({1.0})}},
       "an input has an empty name"},
      {"empty input",
       {},
       {{.name = "acceleration", .input = vector_t()}},
       "input 'acceleration' is empty"},
      {"non-finite input",
       {},
       {{.name = "acceleration", .input = makeVector({kNaN, 0.0})}},
       "the value of input 'acceleration' is not finite"},
      {"asymmetric input noise",
       {},
       {{.name = "acceleration",
         .input = makeVector({1.0, 1.0}),
         .inputNoiseCovariance =
             (matrix_t(2, 2) << 1.0, 0.1, 0.0, 1.0).finished()}},
       "inputNoiseCovariance of input 'acceleration' is not symmetric"},
      // Positive variances, but a covariance larger than their geometric mean:
      // a correlation of 2.
      {"input noise with a correlation beyond one",
       {},
       {{.name = "acceleration",
         .input = makeVector({1.0, 1.0}),
         .inputNoiseCovariance =
             (matrix_t(2, 2) << 1.0, 2.0, 2.0, 1.0).finished()}},
       "inputNoiseCovariance of input 'acceleration' is not positive "
       "semi-definite: its entry (0, 1) = 2"},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    expectRejectedWithoutChange(
        filter_->predict(test_case.processModel, test_case.inputs),
        test_case.expected_message);
  }
}

TEST_F(KalmanFilterRejectionTest,
       invalidCorrectionsAreRejectedWithoutChangingTheEstimate) {
  struct Case {
    std::string label;
    std::vector<KalmanFilterMeasurement> measurements;
    std::string expected_message;
  };
  const KalmanFilterMeasurement valid = {
      .name = "position",
      .measurement = makeVector({1.0, 2.0}),
      .measurementModel = {{"position", identity(2)}},
      .measurementNoiseCovariance = identity(2)};
  const std::vector<Case> cases = {
      {"block of an unknown channel",
       {{.name = "gps",
         .measurement = makeVector({1.0, 2.0}),
         .measurementModel = {{"orientation", identity(2)}},
         .measurementNoiseCovariance = identity(2)}},
       "measurementModel of measurement 'gps' names state channel "
       "'orientation'"},
      {"observation block of the wrong size",
       {{.name = "gps",
         .measurement = makeVector({1.0, 2.0}),
         .measurementModel = {{"position", identity(3)}},
         .measurementNoiseCovariance = identity(2)}},
       "measurementModel block ('gps', 'position') is 3x3, expected 2x2"},
      {"no observation block",
       {{.name = "gps",
         .measurement = makeVector({1.0, 2.0}),
         .measurementNoiseCovariance = identity(2)}},
       "measurement 'gps' observes no state channel"},
      {"missing noise",
       {{.name = "gps",
         .measurement = makeVector({1.0, 2.0}),
         .measurementModel = {{"position", identity(2)}}}},
       "measurementNoiseCovariance of measurement 'gps' is missing"},
      {"noise of the wrong size",
       {{.name = "gps",
         .measurement = makeVector({1.0, 2.0}),
         .measurementModel = {{"position", identity(2)}},
         .measurementNoiseCovariance = identity(3)}},
       "measurementNoiseCovariance of measurement 'gps' is 3x3, expected 2x2"},
      {"repeated measurement",
       {valid, valid},
       "measurement 'position' is given twice"},
      {"measurement without a name",
       {{.name = "",
         .measurement = makeVector({1.0, 2.0}),
         .measurementModel = {{"position", identity(2)}},
         .measurementNoiseCovariance = identity(2)}},
       "a measurement has an empty name"},
      {"empty measurement",
       {{.name = "gps",
         .measurement = vector_t(),
         .measurementModel = {{"position", identity(2)}},
         .measurementNoiseCovariance = identity(2)}},
       "measurement 'gps' is empty"},
      {"non-finite measurement",
       {{.name = "gps",
         .measurement = makeVector({kNaN, 2.0}),
         .measurementModel = {{"position", identity(2)}},
         .measurementNoiseCovariance = identity(2)}},
       "the value of measurement 'gps' is not finite"},
      // Twice the same noise-free direction: H P H^T + R has rank one.
      {"singular innovation covariance",
       {{.name = "first",
         .measurement = makeVector({1.0}),
         .measurementModel = {{"velocity", matrix_t::Ones(1, 2)}},
         .measurementNoiseCovariance = scalarMatrix(0.0)},
        {.name = "second",
         .measurement = makeVector({1.0}),
         .measurementModel = {{"velocity", matrix_t::Ones(1, 2)}},
         .measurementNoiseCovariance = scalarMatrix(0.0)}},
       "the innovation covariance H P H^T + R of measurements first, second is "
       "not positive definite"},
      // The same direction scaled by 1/3: singular in exact arithmetic, and
      // only rounding away from it in floating point, so the Cholesky
      // factorization either fails or leaves a pivot at rounding level, which
      // the pivot test rejects.
      {"innovation covariance singular up to rounding",
       {{.name = "first",
         .measurement = makeVector({1.0}),
         .measurementModel = {{"velocity", matrix_t::Ones(1, 2)}},
         .measurementNoiseCovariance = scalarMatrix(0.0)},
        {.name = "third",
         .measurement = makeVector({1.0 / 3.0}),
         .measurementModel = {{"velocity",
                               matrix_t::Constant(1, 2, 1.0 / 3.0)}},
         .measurementNoiseCovariance = scalarMatrix(0.0)}},
       "the innovation covariance H P H^T + R of measurements first, third is "
       "not positive definite"},
      {"noise with a correlation beyond one",
       {{.name = "gps",
         .measurement = makeVector({1.0, 2.0}),
         .measurementModel = {{"position", identity(2)}},
         .measurementNoiseCovariance =
             (matrix_t(2, 2) << 1.0, -2.0, -2.0, 1.0).finished()}},
       "measurementNoiseCovariance of measurement 'gps' is not positive "
       "semi-definite: its entry (0, 1) = -2"},
  };
  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    expectRejectedWithoutChange(filter_->correct(test_case.measurements),
                                test_case.expected_message);
  }
}

}  // namespace
}  // namespace ocs2::humanoid::estimation
