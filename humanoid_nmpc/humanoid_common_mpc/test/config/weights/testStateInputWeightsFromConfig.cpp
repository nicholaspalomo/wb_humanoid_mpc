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

// The task file's weights and states by name become the MPC's matrices and vectors on the layout of its state and
// input: each block lands at the coordinates of its MPC whatever order the file names joints and contacts in, every
// element is `scaling` times its entry (also -0.0 off the diagonal), and what the MPC needs and does not have is
// refused naming the field path, every problem at once. An absent scaling is 1, and the schemas refuse their retired
// keys naming their replacement. Every shipped task file's state_weights, final_state_weights and initial_state convert
// to the n_t rows of the tuning layout - the tangent of the quaternion state, one row fewer than the state - with each
// orientation block on its own rows, and the initial state onto the quaternion state.

#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_mpc_config/acom_weights.nproto.h"
#include "humanoid_mpc_config/com_weights.nproto.h"
#include "humanoid_mpc_config/input_weights.nproto.h"
#include "humanoid_mpc_config/joint_weights.nproto.h"
#include "humanoid_mpc_config/state_values.nproto.h"
#include "humanoid_mpc_config/state_values.pb.h"
#include "humanoid_mpc_config/state_weights.nproto.h"
#include "humanoid_mpc_config/state_weights.pb.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_mpc_config/yaw_pitch_roll.nproto.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

/** Two joints and two contacts: a state of 14 (centroidal) or 16 (whole body), an input of 14. */
StateInputLayout smallLayout(StateInputLayout::Mpc mpc) {
  StateInputLayout layout;
  layout.mpc = mpc;
  layout.jointNames = {"a", "b"};
  layout.fixedJointNames = {"f"};
  layout.contactNames = {"left", "right"};
  return layout;
}

/** The task file of the textproto `text`, parsed strictly. */
absl::StatusOr<mpc_config::TaskFile> parseTaskFile(absl::string_view text) {
  absl::StatusOr<humanoid_mpc_config::TaskFile> message = nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(text, "task.textproto");
  if (!message.ok()) {
    return message.status();
  }
  mpc_config::TaskFile task;
  if (absl::Status status = mpc_config::FromProto(*message, &task); !status.ok()) {
    return status;
  }
  return task;
}

/** The task file of `text`, which the test writes valid. */
mpc_config::TaskFile taskFile(absl::string_view text) {
  absl::StatusOr<mpc_config::TaskFile> task = parseTaskFile(text);
  EXPECT_TRUE(task.ok()) << task.status();
  return task.ok() ? *std::move(task) : mpc_config::TaskFile{};
}

/** `status` is InvalidArgument and says each of `expected`. */
void expectRefused(const absl::Status& status, const std::vector<std::string>& expected) {
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument) << status;
  for (const std::string& part : expected) {
    EXPECT_TRUE(absl::StrContains(status.message(), part)) << status << "\ndoes not say: " << part;
  }
}

/** `matrix` is square and diagonal with `diagonal` on its diagonal. */
void expectDiagonal(const matrix_t& matrix, const vector_t& diagonal) {
  ASSERT_EQ(matrix.rows(), diagonal.size());
  ASSERT_EQ(matrix.cols(), diagonal.size());
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      EXPECT_EQ(matrix(row, col), row == col ? diagonal(row) : 0.0) << "(" << row << "," << col << ")";
    }
  }
}

/** 1, 2, ..., size: a value per coordinate that tells where each entry of a block landed. */
vector_t counting(Eigen::Index size) {
  return vector_t::LinSpaced(size, 1.0, static_cast<scalar_t>(size));
}

constexpr absl::string_view kCentroidalState = R"pb(
  normalized_linear_momentum { x: 1 y: 2 z: 3 }
  normalized_angular_momentum { x: 4 y: 5 z: 6 }
  base_position { x: 7 y: 8 z: 9 }
  base_orientation { yaw: 10 pitch: 11 roll: 12 }
  joint_positions { joint: "b" value: 14 }
  joint_positions { joint: "a" value: 13 }
)pb";

constexpr absl::string_view kWholeBodyState = R"pb(
  base_position { x: 1 y: 2 z: 3 }
  base_orientation { yaw: 4 pitch: 5 roll: 6 }
  joint_positions { joint: "a" value: 7 }
  joint_positions { joint: "b" value: 8 }
  base_linear_velocity { x: 9 y: 10 z: 11 }
  base_angular_velocity { yaw: 12 pitch: 13 roll: 14 }
  joint_velocities { joint: "b" value: 16 }
  joint_velocities { joint: "a" value: 15 }
)pb";

TEST(StateInputLayoutTest, TheDimensionsAreThoseOfTheMpcsStateAndInput) {
  EXPECT_EQ(tuningDimension(smallLayout(StateInputLayout::Mpc::kCentroidal)), 14U);
  EXPECT_EQ(tuningDimension(smallLayout(StateInputLayout::Mpc::kWholeBody)), 16U);
  EXPECT_EQ(inputDimension(smallLayout(StateInputLayout::Mpc::kCentroidal)), 14U);
  EXPECT_EQ(inputDimension(smallLayout(StateInputLayout::Mpc::kWholeBody)), 14U);
}

/** A block of the tuning layout as the counting states above give it: its field, kind and first and last values. */
struct ExpectedBlock {
  std::string field;
  TuningSegmentKind kind = TuningSegmentKind::kEuclidean;
  scalar_t firstValue = 0.0;
  scalar_t lastValue = 0.0;
};

/**
 * tuningLayoutOf(layout) describes the vector stateValuesFromConfig() builds from `stateText`, which counts 1, 2, ... in
 * state order: each block, in order, is the field of that name, at the offset where placeState() puts it - its first and
 * last entries hold the field's first and last values - and the field exists in both StateValues and StateWeights.
 */
void expectTuningLayoutIsPlaceStatesLayout(const StateInputLayout& layout,
                                           absl::string_view stateText,
                                           const std::vector<ExpectedBlock>& blocks) {
  const TuningLayout tuningLayout = tuningLayoutOf(layout);
  ASSERT_EQ(tuningLayout.segments.size(), blocks.size());
  EXPECT_EQ(getTuningLayoutDim(tuningLayout), tuningDimension(layout));
  const mpc_config::TaskFile task = taskFile(absl::StrCat("initial_state { ", stateText, " }"));
  const absl::StatusOr<vector_t> state = stateValuesFromConfig(task.initial_state, layout, "initial_state");
  ASSERT_TRUE(state.ok()) << state.status();
  ASSERT_EQ(static_cast<size_t>(state->size()), getTuningLayoutDim(tuningLayout));
  size_t offset = 0;
  for (size_t i = 0; i < blocks.size(); ++i) {
    const TuningSegment& segment = tuningLayout.segments[i];
    const ExpectedBlock& expected = blocks[i];
    EXPECT_EQ(segment.name, expected.field);
    EXPECT_EQ(segment.kind, expected.kind) << segment.name;
    // Not placed on a state: the robot model places it.
    EXPECT_EQ(segment.stateDim, 0U) << segment.name;
    const absl::StatusOr<size_t> start = tuningSegmentStart(tuningLayout, segment.name);
    ASSERT_TRUE(start.ok()) << start.status();
    EXPECT_EQ(*start, offset) << segment.name;
    ASSERT_GT(segment.tuningDim, 0U) << segment.name;
    EXPECT_EQ((*state)(static_cast<Eigen::Index>(*start)), expected.firstValue) << segment.name;
    EXPECT_EQ((*state)(static_cast<Eigen::Index>(*start + segment.tuningDim - 1)), expected.lastValue) << segment.name;
    EXPECT_NE(humanoid_mpc_config::StateValues::descriptor()->FindFieldByName(segment.name), nullptr) << segment.name;
    EXPECT_NE(humanoid_mpc_config::StateWeights::descriptor()->FindFieldByName(segment.name), nullptr) << segment.name;
    offset += segment.tuningDim;
  }
}

TEST(StateInputLayoutTest, TheCentroidalTuningLayoutIsWherePlaceStatePutsEachBlock) {
  expectTuningLayoutIsPlaceStatesLayout(
      smallLayout(StateInputLayout::Mpc::kCentroidal), kCentroidalState,
      {
          {.field = "normalized_linear_momentum", .firstValue = 1.0, .lastValue = 3.0},
          {.field = "normalized_angular_momentum", .firstValue = 4.0, .lastValue = 6.0},
          {.field = "base_position", .firstValue = 7.0, .lastValue = 9.0},
          {.field = "base_orientation", .kind = TuningSegmentKind::kEulerZyxOrientation, .firstValue = 10.0, .lastValue = 12.0},
          {.field = "joint_positions", .firstValue = 13.0, .lastValue = 14.0},
      });
}

TEST(StateInputLayoutTest, TheWholeBodyTuningLayoutIsWherePlaceStatePutsEachBlock) {
  expectTuningLayoutIsPlaceStatesLayout(
      smallLayout(StateInputLayout::Mpc::kWholeBody), kWholeBodyState,
      {
          {.field = "base_position", .firstValue = 1.0, .lastValue = 3.0},
          {.field = "base_orientation", .kind = TuningSegmentKind::kEulerZyxOrientation, .firstValue = 4.0, .lastValue = 6.0},
          {.field = "joint_positions", .firstValue = 7.0, .lastValue = 8.0},
          {.field = "base_linear_velocity", .firstValue = 9.0, .lastValue = 11.0},
          {.field = "base_angular_velocity", .kind = TuningSegmentKind::kZyxOrderedAngularVelocity, .firstValue = 12.0, .lastValue = 14.0},
          {.field = "joint_velocities", .firstValue = 15.0, .lastValue = 16.0},
      });
}

TEST(StateWeightsFromConfigTest, EachBlockLandsOnItsCentroidalCoordinatesScaled) {
  const mpc_config::TaskFile task = taskFile(absl::StrCat("state_weights { scaling: 2 ", kCentroidalState, " }"));
  const absl::StatusOr<matrix_t> weights =
      stateWeightsFromConfig(task.state_weights, smallLayout(StateInputLayout::Mpc::kCentroidal), "state_weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  expectDiagonal(*weights, 2.0 * counting(/*size=*/14));
}

TEST(StateWeightsFromConfigTest, EachBlockLandsOnItsWholeBodyCoordinatesScaled) {
  const mpc_config::TaskFile task = taskFile(absl::StrCat("final_state_weights { scaling: 0.5 ", kWholeBodyState, " }"));
  const absl::StatusOr<matrix_t> weights =
      stateWeightsFromConfig(task.final_state_weights, smallLayout(StateInputLayout::Mpc::kWholeBody), "final_state_weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  expectDiagonal(*weights, 0.5 * counting(/*size=*/16));
}

TEST(StateWeightsFromConfigTest, EveryElementIsTheScalingTimesItsEntry) {
  const mpc_config::TaskFile task = taskFile(absl::StrCat("state_weights { scaling: -1 ", kCentroidalState, " }"));
  const absl::StatusOr<matrix_t> weights =
      stateWeightsFromConfig(task.state_weights, smallLayout(StateInputLayout::Mpc::kCentroidal), "state_weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  // loadEigenMatrix multiplied every element, the zeros included: a negative scaling gives -0.0 off the diagonal.
  EXPECT_TRUE(std::signbit((*weights)(0, 1)));
  EXPECT_TRUE(std::signbit((*weights)(13, 0)));
  EXPECT_EQ((*weights)(13, 13), -14.0);
}

TEST(StateWeightsFromConfigTest, EmptyBlocksAndJointsWithoutAValueAreZeros) {
  const mpc_config::TaskFile task = taskFile(R"pb(
    state_weights {
      scaling: 3
      normalized_linear_momentum {}
      normalized_angular_momentum {}
      base_position { z: 1 }
      base_orientation {}
      joint_positions { joint: "a" }
      joint_positions { joint: "b" }
    }
  )pb");
  const absl::StatusOr<matrix_t> weights =
      stateWeightsFromConfig(task.state_weights, smallLayout(StateInputLayout::Mpc::kCentroidal), "state_weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  vector_t expected = vector_t::Zero(14);
  expected(8) = 3.0;
  expectDiagonal(*weights, expected);
}

TEST(StateWeightsFromConfigTest, EveryProblemIsRefusedAtOnceNamingItsFieldPath) {
  const mpc_config::TaskFile task = taskFile(R"pb(
    state_weights {
      normalized_linear_momentum {}
      base_position {}
      base_orientation {}
      base_linear_velocity {}
      joint_positions { joint: "a" }
      joint_positions { joint: "a" }
      joint_positions { joint: "f" }
      joint_positions { joint: "zz" }
      joint_velocities { joint: "a" }
    }
  )pb");
  const absl::StatusOr<matrix_t> weights =
      stateWeightsFromConfig(task.state_weights, smallLayout(StateInputLayout::Mpc::kCentroidal), "state_weights");
  expectRefused(weights.status(), {
                                      "state_weights.normalized_angular_momentum is missing",
                                      "state_weights.base_linear_velocity is not part of the centroidal MPC's state",
                                      "state_weights.joint_velocities is not part of the centroidal MPC's state",
                                      "state_weights.joint_positions: a given more than once",
                                      "state_weights.joint_positions: f is a fixed joint",
                                      "state_weights.joint_positions: 'zz' is not a joint of the MPC model",
                                      "state_weights.joint_positions: missing b",
                                  });
}

TEST(StateWeightsFromConfigTest, TheCentroidalBlocksAreNotPartOfTheWholeBodyState) {
  const mpc_config::TaskFile task = taskFile(absl::StrCat("state_weights { ", kCentroidalState, " }"));
  const absl::StatusOr<matrix_t> weights =
      stateWeightsFromConfig(task.state_weights, smallLayout(StateInputLayout::Mpc::kWholeBody), "state_weights");
  expectRefused(weights.status(), {
                                      "state_weights.normalized_linear_momentum is not part of the whole-body MPC's state",
                                      "state_weights.normalized_angular_momentum is not part of the whole-body MPC's state",
                                      "state_weights.base_linear_velocity is missing",
                                      "state_weights.joint_velocities: missing a, b",
                                  });
}

TEST(StateWeightsFromConfigTest, AFileWithoutTheWeightsIsRefused) {
  const absl::StatusOr<matrix_t> weights =
      stateWeightsFromConfig(mpc_config::StateWeights{}, smallLayout(StateInputLayout::Mpc::kCentroidal), "state_weights");
  expectRefused(weights.status(), {"state_weights.joint_positions: missing a, b"});
}

TEST(StateWeightsFromConfigTest, AValueThatIsNotFiniteIsRefused) {
  const mpc_config::TaskFile task = taskFile(R"pb(
    state_weights {
      scaling: inf
      normalized_linear_momentum { x: nan }
      normalized_angular_momentum {}
      base_position {}
      base_orientation {}
      joint_positions { joint: "a" value: -inf }
      joint_positions { joint: "b" }
    }
  )pb");
  const absl::StatusOr<matrix_t> weights =
      stateWeightsFromConfig(task.state_weights, smallLayout(StateInputLayout::Mpc::kCentroidal), "state_weights");
  expectRefused(weights.status(), {
                                      "state_weights.scaling is inf, not a finite number",
                                      "state_weights.normalized_linear_momentum.x is nan",
                                      "state_weights.joint_positions[joint=a].value is -inf",
                                  });
}

TEST(StateValuesFromConfigTest, TheInitialStateIsItsValuesOnTheLayout) {
  const mpc_config::TaskFile centroidal = taskFile(absl::StrCat("initial_state { ", kCentroidalState, " }"));
  const absl::StatusOr<vector_t> centroidalState =
      stateValuesFromConfig(centroidal.initial_state, smallLayout(StateInputLayout::Mpc::kCentroidal), "initial_state");
  ASSERT_TRUE(centroidalState.ok()) << centroidalState.status();
  EXPECT_EQ(*centroidalState, counting(/*size=*/14));

  const mpc_config::TaskFile wholeBody = taskFile(absl::StrCat("initial_state { ", kWholeBodyState, " }"));
  const absl::StatusOr<vector_t> wholeBodyState =
      stateValuesFromConfig(wholeBody.initial_state, smallLayout(StateInputLayout::Mpc::kWholeBody), "initial_state");
  ASSERT_TRUE(wholeBodyState.ok()) << wholeBodyState.status();
  EXPECT_EQ(*wholeBodyState, counting(/*size=*/16));

  expectRefused(stateValuesFromConfig(mpc_config::StateValues{}, smallLayout(StateInputLayout::Mpc::kWholeBody), "initial_state").status(),
                {"initial_state.base_position is missing", "initial_state.joint_positions: missing a, b"});
}

TEST(InputWeightsFromConfigTest, EachContactsWrenchLandsAtItsContactWhateverTheOrder) {
  const mpc_config::TaskFile task = taskFile(R"pb(
    input_weights {
      scaling: 10
      contact_wrenches {
        contact: "right"
        force { x: 7 y: 8 z: 9 }
        moment { x: 10 y: 11 z: 12 }
      }
      contact_wrenches {
        contact: "left"
        force { x: 1 y: 2 z: 3 }
        moment { x: 4 y: 5 z: 6 }
      }
      joint_velocities { joint: "a" value: 13 }
      joint_velocities { joint: "b" value: 14 }
    }
  )pb");
  const absl::StatusOr<matrix_t> weights =
      inputWeightsFromConfig(task.input_weights, smallLayout(StateInputLayout::Mpc::kCentroidal), "input_weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  expectDiagonal(*weights, 10.0 * counting(/*size=*/14));
}

TEST(InputWeightsFromConfigTest, TheWholeBodyInputWeighsJointAccelerations) {
  const mpc_config::TaskFile task = taskFile(R"pb(
    input_weights {
      contact_wrenches {
        contact: "left"
        force {}
        moment {}
      }
      contact_wrenches {
        contact: "right"
        force {}
        moment {}
      }
      joint_accelerations { joint: "b" value: 2 }
      joint_accelerations { joint: "a" value: 1 }
    }
  )pb");
  const absl::StatusOr<matrix_t> weights =
      inputWeightsFromConfig(task.input_weights, smallLayout(StateInputLayout::Mpc::kWholeBody), "input_weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  vector_t expected = vector_t::Zero(14);
  expected(12) = 1.0;
  expected(13) = 2.0;
  expectDiagonal(*weights, expected);
}

TEST(InputWeightsFromConfigTest, EveryProblemIsRefusedAtOnceNamingItsFieldPath) {
  const mpc_config::TaskFile task = taskFile(R"pb(
    input_weights {
      contact_wrenches {
        contact: "left"
        force {}
      }
      contact_wrenches {
        contact: "left"
        force {}
        moment {}
      }
      contact_wrenches {
        contact: "hand"
        force {}
        moment {}
      }
      joint_accelerations { joint: "a" }
    }
  )pb");
  const absl::StatusOr<matrix_t> weights =
      inputWeightsFromConfig(task.input_weights, smallLayout(StateInputLayout::Mpc::kCentroidal), "input_weights");
  expectRefused(weights.status(), {
                                      "input_weights.contact_wrenches[contact=left].moment is missing",
                                      "input_weights.contact_wrenches: left given more than once",
                                      "input_weights.contact_wrenches: 'hand' is not a contact of the MPC model",
                                      "input_weights.contact_wrenches: missing right",
                                      "input_weights.joint_accelerations is not part of the centroidal MPC's input",
                                      "input_weights.joint_velocities: missing a, b",
                                  });
}

TEST(ComAndAcomWeightsFromConfigTest, TheWeightsAreDiagonalInTheirOrder) {
  const mpc_config::TaskFile task = taskFile(R"pb(
    com_weights { scaling: 2 x: 1 z: 3 }
    acom_weights { yaw: 1 pitch: 2 roll: 3 }
  )pb");
  const absl::StatusOr<matrix_t> com = comWeightsFromConfig(task.com_weights, "com_weights");
  ASSERT_TRUE(com.ok()) << com.status();
  expectDiagonal(*com, (vector_t(3) << 2.0, 0.0, 6.0).finished());
  // The rows of the angular center of mass are Euler ZYX: yaw first.
  const absl::StatusOr<matrix_t> acom = acomWeightsFromConfig(task.acom_weights, "acom_weights");
  ASSERT_TRUE(acom.ok()) << acom.status();
  expectDiagonal(*acom, (vector_t(3) << 1.0, 2.0, 3.0).finished());
}

TEST(ComAndAcomWeightsFromConfigTest, ABlockThatNamesNoWeightIsRefused) {
  const mpc_config::TaskFile task = taskFile("com_weights { scaling: 85 }");
  expectRefused(comWeightsFromConfig(task.com_weights, "com_weights").status(), {"com_weights names none of x, y, z"});
  expectRefused(acomWeightsFromConfig(task.acom_weights, "acom_weights").status(), {"acom_weights names none of yaw, pitch, roll"});
}

TEST(TerminalCostScalingFromConfigTest, TheScalingIsRequired) {
  expectRefused(terminalCostScalingFromConfig(mpc_config::TaskFile{}).status(), {"terminal_cost_scaling is missing"});
  const absl::StatusOr<scalar_t> scaling = terminalCostScalingFromConfig(taskFile("terminal_cost_scaling: 4"));
  ASSERT_TRUE(scaling.ok()) << scaling.status();
  EXPECT_EQ(*scaling, 4.0);
}

TEST(TrackingStateWeightsFromConfigTest, TheBasePoseBlockIsKeptOrZeroedAndNothingElseChanges) {
  const StateInputLayout layout = smallLayout(StateInputLayout::Mpc::kCentroidal);
  const mpc_config::TaskFile task = taskFile(absl::StrCat("state_weights { scaling: 2 ", kCentroidalState, " }"));
  const absl::StatusOr<matrix_t> converted = stateWeightsFromConfig(task.state_weights, layout, "state_weights");
  ASSERT_TRUE(converted.ok()) << converted.status();
  const absl::StatusOr<matrix_t> kept = trackingStateWeightsFromConfig(task.state_weights, layout, "state_weights", BasePoseWeights::kKept);
  ASSERT_TRUE(kept.ok()) << kept.status();
  EXPECT_TRUE(*kept == *converted);

  const absl::StatusOr<matrix_t> zeroed =
      trackingStateWeightsFromConfig(task.state_weights, layout, "state_weights", BasePoseWeights::kZeroedForComAndAcomTracking);
  ASSERT_TRUE(zeroed.ok()) << zeroed.status();
  // The rows of base_position and base_orientation of the tuning layout, and only those.
  const TuningLayout tuningLayout = tuningLayoutOf(layout);
  const absl::StatusOr<size_t> first = tuningSegmentStart(tuningLayout, "base_position");
  const absl::StatusOr<size_t> orientation = tuningSegmentStart(tuningLayout, "base_orientation");
  ASSERT_TRUE(first.ok() && orientation.ok());
  const Eigen::Index begin = static_cast<Eigen::Index>(*first);
  const Eigen::Index end = static_cast<Eigen::Index>(*orientation + 3);
  for (Eigen::Index row = 0; row < zeroed->rows(); ++row) {
    for (Eigen::Index col = 0; col < zeroed->cols(); ++col) {
      const bool basePose = row >= begin && row < end && col >= begin && col < end;
      EXPECT_EQ((*zeroed)(row, col), basePose ? 0.0 : (*converted)(row, col)) << "(" << row << "," << col << ")";
    }
  }
}

TEST(TerminalStateWeightsFromConfigTest, TheScalingWeighsTheTreatedFinalStateWeights) {
  const StateInputLayout layout = smallLayout(StateInputLayout::Mpc::kCentroidal);
  const mpc_config::TaskFile task = taskFile(absl::StrCat("final_state_weights { scaling: 3 ", kCentroidalState, " }"));
  for (const BasePoseWeights basePose : {BasePoseWeights::kKept, BasePoseWeights::kZeroedForComAndAcomTracking}) {
    const absl::StatusOr<matrix_t> tracking =
        trackingStateWeightsFromConfig(task.final_state_weights, layout, "final_state_weights", basePose);
    const absl::StatusOr<matrix_t> terminal =
        terminalStateWeightsFromConfig(task.final_state_weights, /*terminalCostScaling=*/4.0, layout, "final_state_weights", basePose);
    ASSERT_TRUE(tracking.ok()) << tracking.status();
    ASSERT_TRUE(terminal.ok()) << terminal.status();
    // Element for element the multiplication the factory and the live updater each made before.
    matrix_t expected = *tracking;
    expected *= 4.0;
    EXPECT_TRUE(*terminal == expected);
  }
  // A refused conversion is the conversion's error, naming the field.
  const mpc_config::TaskFile refused = taskFile(absl::StrCat("final_state_weights { scaling: 3 ", kWholeBodyState, " }"));
  expectRefused(terminalStateWeightsFromConfig(refused.final_state_weights, /*terminalCostScaling=*/4.0, layout, "final_state_weights",
                                               BasePoseWeights::kKept)
                    .status(),
                {"final_state_weights.base_linear_velocity"});
}

TEST(LegTorqueCostFromConfigTest, TheJointsKeepTheirOrderAndAreScaled) {
  const mpc_config::TaskFile task = taskFile(R"pb(
    left_leg_torque_cost {
      scaling: 0.5
      joints { joint: "b" value: 4 }
      joints { joint: "a" value: 2 }
    }
  )pb");
  const absl::StatusOr<ExternalTorqueQuadraticCostAD::Config> config =
      legTorqueCostFromConfig(task.left_leg_torque_cost, smallLayout(StateInputLayout::Mpc::kCentroidal), "left_leg_torque_cost");
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->activeJointNames, (std::vector<std::string>{"b", "a"}));
  EXPECT_EQ(config->weights, (vector_t(2) << 2.0, 1.0).finished());
}

TEST(LegTorqueCostFromConfigTest, AnEmptyListARepeatedAFixedAndAnUnknownJointAreRefused) {
  expectRefused(
      legTorqueCostFromConfig(mpc_config::JointWeights{}, smallLayout(StateInputLayout::Mpc::kCentroidal), "right_leg_torque_cost")
          .status(),
      {"right_leg_torque_cost.joints names no joint"});
  const mpc_config::TaskFile task = taskFile(R"pb(
    right_leg_torque_cost {
      joints { joint: "a" }
      joints { joint: "a" }
      joints { joint: "f" }
      joints { joint: "zz" }
    }
  )pb");
  expectRefused(
      legTorqueCostFromConfig(task.right_leg_torque_cost, smallLayout(StateInputLayout::Mpc::kCentroidal), "right_leg_torque_cost")
          .status(),
      {"right_leg_torque_cost.joints: a given more than once", "right_leg_torque_cost.joints: f is a fixed joint",
       "right_leg_torque_cost.joints: 'zz' is not a joint of the MPC model"});
}

TEST(JointTorqueWeightsFromConfigTest, EachJointsWeightLandsAtItsJoint) {
  const mpc_config::TaskFile task = taskFile(R"pb(
    joint_torque_weights {
      scaling: 3
      joints { joint: "b" value: 2 }
      joints { joint: "a" value: 1 }
    }
  )pb");
  const absl::StatusOr<vector_t> weights =
      jointTorqueWeightsFromConfig(task.joint_torque_weights, smallLayout(StateInputLayout::Mpc::kWholeBody), "joint_torque_weights");
  ASSERT_TRUE(weights.ok()) << weights.status();
  EXPECT_EQ(*weights, (vector_t(2) << 3.0, 6.0).finished());
  expectRefused(
      jointTorqueWeightsFromConfig(mpc_config::JointWeights{}, smallLayout(StateInputLayout::Mpc::kWholeBody), "joint_torque_weights")
          .status(),
      {"joint_torque_weights.joints: missing a, b"});
}

TEST(WeightSchemaTest, AnAbsentScalingIsOne) {
  // loadEigenMatrix read `scaling` with a default of 1.
  EXPECT_EQ(mpc_config::StateWeights{}.scaling, 1.0);
  EXPECT_EQ(mpc_config::InputWeights{}.scaling, 1.0);
  EXPECT_EQ(mpc_config::JointWeights{}.scaling, 1.0);
  EXPECT_EQ(mpc_config::ComWeights{}.scaling, 1.0);
  EXPECT_EQ(mpc_config::AcomWeights{}.scaling, 1.0);
}

// A key of the old indexed matrices ("(i,i)") in each block that replaced one.
constexpr std::array<absl::string_view, 9> kRetiredKeys = {
    "state_weights { default: 1 }",
    "final_state_weights { default: 1 }",
    "input_weights { default: 1 }",
    "com_weights { default: 1 }",
    "acom_weights { default: 1 }",
    "initial_state { scaling: 2 }",
    "left_leg_torque_cost { activeJointNames: \"a\" }",
    "right_leg_torque_cost { weights {} }",
    "joint_torque_weights { default: 1 }",
};

TEST(WeightSchemaTest, TheKeysOfTheIndexedMatricesAreRefusedNamingTheirReplacement) {
  for (const absl::string_view text : kRetiredKeys) {
    const absl::StatusOr<mpc_config::TaskFile> task = parseTaskFile(text);
    EXPECT_EQ(task.status().code(), absl::StatusCode::kInvalidArgument) << text;
    EXPECT_TRUE(absl::StrContains(task.status().message(), "is retired: ")) << task.status();
  }
}

/******************************************************************************************************/
/* The shipped task files on the quaternion layout                                                    */
/******************************************************************************************************/

/** A shipped MPC's task file and the URDF its ModelSettings are made of, from the repository root. */
struct ShippedTaskFile {
  const char* absl_nonnull name;
  const char* absl_nonnull taskFile;
  const char* absl_nonnull urdfFile;
  StateInputLayout::Mpc mpc;
};

// LINT.IfChange(shipped_task_files)
constexpr std::array<ShippedTaskFile, 5> kShippedTaskFiles = {{
    {.name = "drc_atlas_centroidal_mpc",
     .taskFile = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf",
     .mpc = StateInputLayout::Mpc::kCentroidal},
    {.name = "engineai_sa01_centroidal_mpc",
     .taskFile = "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf",
     .mpc = StateInputLayout::Mpc::kCentroidal},
    {.name = "g1_centroidal_mpc",
     .taskFile = "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
     .mpc = StateInputLayout::Mpc::kCentroidal},
    {.name = "g1_wb_mpc",
     .taskFile = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf",
     .mpc = StateInputLayout::Mpc::kWholeBody},
    {.name = "unitree_r1_centroidal_mpc",
     .taskFile = "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf",
     .mpc = StateInputLayout::Mpc::kCentroidal},
}};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:config_test_data)

/** `relativePath` in the test's runfiles. */
std::string runfilesPath(absl::string_view relativePath) {
  const char* absl_nullable directory = std::getenv("TEST_SRCDIR");
  return directory == nullptr ? std::string(relativePath) : absl::StrCat(directory, "/_main/", relativePath);
}

/** (yaw, pitch, roll) of `angles`. */
vector3_t yawPitchRollOf(const mpc_config::YawPitchRoll& angles) {
  return vector3_t(angles.yaw, angles.pitch, angles.roll);
}

/** The diagonal of `matrix`, a square matrix that is diagonal (every other entry exactly zero, which the test expects). */
vector_t diagonalOf(const matrix_t& matrix) {
  EXPECT_EQ(matrix.rows(), matrix.cols());
  EXPECT_TRUE(matrix_t(matrix - matrix_t(matrix.diagonal().asDiagonal())).isZero(0.0)) << "the weights are not diagonal";
  return matrix.diagonal();
}

TEST(ShippedTaskFilesTest, TheStateRowsConvertToTheTuningLayoutWithEachOrientationBlockOnItsRows) {
  for (const ShippedTaskFile& robot : kShippedTaskFiles) {
    SCOPED_TRACE(robot.name);
    const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(runfilesPath(robot.taskFile));
    ASSERT_TRUE(task.ok()) << task.status();
    const absl::StatusOr<ModelSettings> settings =
        ModelSettings::Create(*task, runfilesPath(robot.urdfFile), "testStateInputWeightsFromConfig_", /*verbose=*/false);
    ASSERT_TRUE(settings.ok()) << settings.status();
    const StateInputLayout layout = stateInputLayout(*settings, robot.mpc);
    const TuningLayout tuning = tuningLayoutOf(layout);
    const Eigen::Index tuningDim = static_cast<Eigen::Index>(tuningDimension(layout));
    ASSERT_EQ(getTuningLayoutDim(tuning), tuningDimension(layout));
    // The quaternion state the robot models store: one entry more than the tuning layout, which is its tangent.
    const TuningLayout placed = placeTuningLayout(tuning);
    ASSERT_EQ(getTuningLayoutStateDim(placed), tuningDimension(layout) + 1);
    const bool wholeBody = robot.mpc == StateInputLayout::Mpc::kWholeBody;

    const absl::StatusOr<size_t> orientationRow = tuningSegmentStart(tuning, "base_orientation");
    ASSERT_TRUE(orientationRow.ok()) << orientationRow.status();
    const Eigen::Index orientation = static_cast<Eigen::Index>(*orientationRow);
    Eigen::Index angularVelocity = 0;
    if (wholeBody) {
      const absl::StatusOr<size_t> angularVelocityRow = tuningSegmentStart(tuning, "base_angular_velocity");
      ASSERT_TRUE(angularVelocityRow.ok()) << angularVelocityRow.status();
      angularVelocity = static_cast<Eigen::Index>(*angularVelocityRow);
    }

    const std::array<std::pair<const char* absl_nonnull, const mpc_config::StateWeights* absl_nonnull>, 2> weights = {
        {{"state_weights", &task->state_weights}, {"final_state_weights", &task->final_state_weights}}};
    for (const std::pair<const char* absl_nonnull, const mpc_config::StateWeights* absl_nonnull>& field : weights) {
      SCOPED_TRACE(field.first);
      const mpc_config::StateWeights& fileWeights = *field.second;
      const absl::StatusOr<matrix_t> Q = stateWeightsFromConfig(fileWeights, layout, field.first);
      ASSERT_TRUE(Q.ok()) << Q.status();
      ASSERT_EQ(Q->rows(), tuningDim);
      const vector_t diagonal = diagonalOf(*Q);
      const vector3_t expectedOrientation =
          fileWeights.scaling * yawPitchRollOf(fileWeights.base_orientation.value_or(mpc_config::YawPitchRoll{}));
      EXPECT_EQ(vector3_t(diagonal.segment<3>(orientation)), expectedOrientation) << "the (yaw, pitch, roll) rows";
      if (wholeBody) {
        const vector3_t expectedAngularVelocity =
            fileWeights.scaling * yawPitchRollOf(fileWeights.base_angular_velocity.value_or(mpc_config::YawPitchRoll{}));
        EXPECT_EQ(vector3_t(diagonal.segment<3>(angularVelocity)), expectedAngularVelocity) << "the body (z, y, x) rows";
      }
    }

    const absl::StatusOr<vector_t> initial = stateValuesFromConfig(task->initial_state, layout, "initial_state");
    ASSERT_TRUE(initial.ok()) << initial.status();
    ASSERT_EQ(initial->size(), tuningDim);
    const vector3_t initialOrientation = yawPitchRollOf(task->initial_state.base_orientation.value_or(mpc_config::YawPitchRoll{}));
    EXPECT_EQ(vector3_t(initial->segment<3>(orientation)), initialOrientation);
    const absl::StatusOr<vector_t> state = stateFromTuningLayout(*initial, placed);
    ASSERT_TRUE(state.ok()) << state.status();
    ASSERT_EQ(state->size(), tuningDim + 1);
    // The orientation is the quaternion of the file's angles, at the slots the tuning row of its yaw starts.
    EXPECT_EQ(vector4_t(state->segment<4>(orientation)), quaternionFromEulerZyx(initialOrientation));
    EXPECT_NEAR(state->segment<4>(orientation).norm(), 1.0, 1.0e-15);
    if (wholeBody) {
      // w_B = (x, y, z) of the rows (z, y, x), one slot further on than its tuning rows.
      EXPECT_EQ(vector3_t(state->segment<3>(angularVelocity + 1)), vector3_t(initial->segment<3>(angularVelocity).reverse()));
    }
    const absl::StatusOr<vector_t> roundTrip = tuningLayoutFromState(*state, placed);
    ASSERT_TRUE(roundTrip.ok()) << roundTrip.status();
    EXPECT_LE((*roundTrip - *initial).cwiseAbs().maxCoeff(), 1.0e-12) << "the initial state does not come back from the state";
  }
}

}  // namespace
}  // namespace ocs2::humanoid
