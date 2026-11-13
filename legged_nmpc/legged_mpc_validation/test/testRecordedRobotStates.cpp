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

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "legged_mpc_validation/closed_loop/RecordedRobotStates.h"
#include "legged_mpc_validation/io/GoldenIo.h"
#include "robot_core/Types.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

/*
 * A recording of robot states is the input of the solve benchmark, recorded
 * once by a walking run and replayed by every later benchmark: it must come
 * back from its file as the very states that were recorded, and it must refuse
 * a robot whose joints are not the ones it was recorded on.
 */

namespace ocs2::legged::validation {
namespace {

constexpr char kG1Urdf[] =
    "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kAtlasUrdf[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

// The contact set of a biped: its two feet.
constexpr size_t kNumFeet = 2;

robot::model::RobotState randomState(
    const robot::model::RobotDescription& description,
    std::mt19937& generator,
    size_t numContacts) {
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  robot::model::RobotState state(description, /*contactSize=*/numContacts);
  state.setTime(12.345 + uniform(generator));
  state.setPosBaseRtWorldEwrtWorld(Eigen::Vector3d(
      uniform(generator), uniform(generator), 0.8 + 0.1 * uniform(generator)));
  state.setQuatBaseRtWorld(Eigen::Quaterniond::UnitRandom());
  state.setLinvelBaseRtWorldEwrtBase(Eigen::Vector3d(
      uniform(generator), uniform(generator), uniform(generator)));
  state.setAngvelBaseRtWorldEwrtBase(Eigen::Vector3d(
      uniform(generator), uniform(generator), uniform(generator)));
  for (const robot::joint_index_t joint : description.getJointIndices()) {
    state.setJointPosition(joint, uniform(generator));
    state.setJointVelocity(joint, 10.0 * uniform(generator));
  }
  for (size_t contact = 0; contact < numContacts; ++contact) {
    state.setContactFlag(contact, uniform(generator) > 0.0);
  }
  return state;
}

TEST(RecordedRobotStates, ARecordingReadsBackAsTheStatesThatWereRecorded) {
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus =
      robot::model::RobotDescription::Create(kG1Urdf);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  std::mt19937 generator(7);
  RecordedRobotStates recording;
  recording.jointNames = description.getJointNames();
  std::vector<robot::model::RobotState> states;
  for (int k = 0; k < 5; ++k) {
    states.push_back(randomState(description, generator, kNumFeet));
    recording.records.push_back(recordRobotState(
        states.back(), description, Eigen::Vector4d(0.1 * k, -0.2, 0.79, 0.3)));
  }
  GoldenProvenance provenance;
  provenance.gitCommit = "c331ddd";
  const absl::StatusOr<GoldenFile> golden = toGoldenFile(recording, provenance);
  ASSERT_TRUE(golden.ok()) << golden.status();
  const absl::StatusOr<std::string> text = formatGoldenFile(*golden);
  ASSERT_TRUE(text.ok()) << text.status();
  const absl::StatusOr<GoldenFile> parsed = parseGoldenFile(*text);
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  const absl::StatusOr<RecordedRobotStates> read = fromGoldenFile(*parsed);
  ASSERT_TRUE(read.ok()) << read.status();
  ASSERT_EQ(read->jointNames, recording.jointNames);
  ASSERT_EQ(read->records.size(), states.size());

  for (size_t k = 0; k < states.size(); ++k) {
    const absl::StatusOr<robot::model::RobotState> state =
        toRobotState(read->records[k], read->jointNames, description, kNumFeet);
    ASSERT_TRUE(state.ok()) << state.status();
    const robot::model::RobotState& expected = states[k];
    EXPECT_EQ(state->getTime(), expected.getTime());
    EXPECT_EQ(state->getPosBaseRtWorldEwrtWorld(),
              expected.getPosBaseRtWorldEwrtWorld());
    EXPECT_EQ(state->getQuatBaseRtWorld().coeffs(),
              expected.getQuatBaseRtWorld().coeffs());
    EXPECT_EQ(state->getLinvelBaseRtWorldEwrtBase(),
              expected.getLinvelBaseRtWorldEwrtBase());
    EXPECT_EQ(state->getAngvelBaseRtWorldEwrtBase(),
              expected.getAngvelBaseRtWorldEwrtBase());
    for (const robot::joint_index_t joint : description.getJointIndices()) {
      EXPECT_EQ(state->getJointPosition(joint),
                expected.getJointPosition(joint));
      EXPECT_EQ(state->getJointVelocity(joint),
                expected.getJointVelocity(joint));
    }
    EXPECT_EQ(state->getContactFlags(), expected.getContactFlags());
    EXPECT_EQ(read->records[k].guiCommand, recording.records[k].guiCommand);
  }
}

TEST(RecordedRobotStates, JointsAreMatchedByNameWhateverTheirOrder) {
  // The order of a RobotDescription's joints is not the same in every process:
  // a recording made in one process is read in another, so its joints are
  // matched by name.
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus =
      robot::model::RobotDescription::Create(kG1Urdf);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  std::mt19937 generator(11);
  const robot::model::RobotState state =
      randomState(description, generator, kNumFeet);
  const RobotStateRecord record =
      recordRobotState(state, description, Eigen::Vector4d::Zero());
  const std::vector<std::string>& names = description.getJointNames();
  std::vector<size_t> order(names.size());
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = order.size() - 1 - i;  // reversed
  }
  RobotStateRecord reordered = record;
  std::vector<std::string> reorderedNames(names.size());
  for (size_t i = 0; i < order.size(); ++i) {
    reorderedNames[i] = names[order[i]];
    reordered.jointPositions(static_cast<Eigen::Index>(i)) =
        record.jointPositions(static_cast<Eigen::Index>(order[i]));
    reordered.jointVelocities(static_cast<Eigen::Index>(i)) =
        record.jointVelocities(static_cast<Eigen::Index>(order[i]));
  }
  const absl::StatusOr<robot::model::RobotState> read =
      toRobotState(reordered, reorderedNames, description, kNumFeet);
  ASSERT_TRUE(read.ok()) << read.status();
  for (const robot::joint_index_t joint : description.getJointIndices()) {
    EXPECT_EQ(read->getJointPosition(joint), state.getJointPosition(joint));
    EXPECT_EQ(read->getJointVelocity(joint), state.getJointVelocity(joint));
  }

  std::vector<std::string> duplicated = names;
  duplicated.back() = duplicated.front();
  EXPECT_EQ(
      toRobotState(record, duplicated, description, kNumFeet).status().code(),
      absl::StatusCode::kInvalidArgument);
}

TEST(RecordedRobotStates, ARecordingRefusesAnotherRobot) {
  absl::StatusOr<robot::model::RobotDescription> g1OrStatus =
      robot::model::RobotDescription::Create(kG1Urdf);
  ASSERT_TRUE(g1OrStatus.ok()) << g1OrStatus.status();
  const robot::model::RobotDescription& g1 = *g1OrStatus;
  absl::StatusOr<robot::model::RobotDescription> atlasOrStatus =
      robot::model::RobotDescription::Create(kAtlasUrdf);
  ASSERT_TRUE(atlasOrStatus.ok()) << atlasOrStatus.status();
  const robot::model::RobotDescription& atlas = *atlasOrStatus;
  std::mt19937 generator(3);
  const RobotStateRecord record = recordRobotState(
      randomState(g1, generator, kNumFeet), g1, Eigen::Vector4d::Zero());
  EXPECT_EQ(
      toRobotState(record, g1.getJointNames(), atlas, kNumFeet).status().code(),
      absl::StatusCode::kInvalidArgument);

  absl::StatusOr<GoldenFile> golden =
      toGoldenFile(RecordedRobotStates{.jointNames = g1.getJointNames(),
                                       .records = {record}},
                   GoldenProvenance());
  ASSERT_TRUE(golden.ok()) << golden.status();
  GoldenFile withoutNames = *golden;
  withoutNames.provenance.notes.clear();
  EXPECT_EQ(fromGoldenFile(withoutNames).status().code(),
            absl::StatusCode::kInvalidArgument);
  GoldenFile truncated = *golden;
  truncated.entries.pop_back();
  EXPECT_EQ(fromGoldenFile(truncated).status().code(),
            absl::StatusCode::kInvalidArgument);
}

// A recording holds the flags of the contact set it was recorded with, of any
// size, and is replayed on a robot of that set only.
TEST(RecordedRobotStates, ARecordingKeepsItsContactSetAndRefusesAnother) {
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus =
      robot::model::RobotDescription::Create(kG1Urdf);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  constexpr size_t kNumQuadrupedFeet = 4;
  std::mt19937 generator(5);
  RecordedRobotStates recording;
  recording.jointNames = description.getJointNames();
  std::vector<robot::model::RobotState> states;
  for (int k = 0; k < 4; ++k) {
    states.push_back(randomState(description, generator, kNumQuadrupedFeet));
    recording.records.push_back(
        recordRobotState(states.back(), description, Eigen::Vector4d::Zero()));
  }
  const absl::StatusOr<GoldenFile> golden =
      toGoldenFile(recording, GoldenProvenance());
  ASSERT_TRUE(golden.ok()) << golden.status();
  const golden_matrix_t* absl_nullable contactFlags =
      golden->find("contact_flags");
  ASSERT_NE(contactFlags, nullptr);
  EXPECT_EQ(contactFlags->cols(), static_cast<Eigen::Index>(kNumQuadrupedFeet));
  const absl::StatusOr<RecordedRobotStates> read = fromGoldenFile(*golden);
  ASSERT_TRUE(read.ok()) << read.status();
  for (size_t k = 0; k < states.size(); ++k) {
    const absl::StatusOr<robot::model::RobotState> state = toRobotState(
        read->records[k], read->jointNames, description, kNumQuadrupedFeet);
    ASSERT_TRUE(state.ok()) << state.status();
    EXPECT_EQ(state->getContactFlags(), states[k].getContactFlags());
  }
  EXPECT_EQ(
      toRobotState(read->records[0], read->jointNames, description, kNumFeet)
          .status()
          .code(),
      absl::StatusCode::kInvalidArgument);

  // Records of two contact sets make no recording.
  recording.records.push_back(
      recordRobotState(randomState(description, generator, kNumFeet),
                       description, Eigen::Vector4d::Zero()));
  EXPECT_EQ(toGoldenFile(recording, GoldenProvenance()).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::legged::validation
