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

#include <random>
#include <string>
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_validation/closed_loop/RecordedRobotStates.h"
#include "humanoid_mpc_validation/io/GoldenIo.h"
#include "robot_model/RobotDescription.h"
#include "robot_model/RobotState.h"

/*
 * A recording of robot states is the input of the solve benchmark, recorded once (B0) and replayed again after the
 * quaternion switch (B3): it must come back from its file as the very states that were recorded, and it must refuse a
 * robot whose joints are not the ones it was recorded on.
 */

namespace ocs2::humanoid::validation {
namespace {

constexpr char kG1Urdf[] = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kAtlasUrdf[] = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

robot::model::RobotState randomState(const robot::model::RobotDescription& description, std::mt19937& generator) {
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  robot::model::RobotState state(description, /*contactSize=*/2);
  state.setTime(12.345 + uniform(generator));
  state.setRootPositionInWorldFrame(Eigen::Vector3d(uniform(generator), uniform(generator), 0.8 + 0.1 * uniform(generator)));
  state.setRootRotationLocalToWorldFrame(Eigen::Quaterniond::UnitRandom());
  state.setRootLinearVelocityInLocalFrame(Eigen::Vector3d(uniform(generator), uniform(generator), uniform(generator)));
  state.setRootAngularVelocityInLocalFrame(Eigen::Vector3d(uniform(generator), uniform(generator), uniform(generator)));
  for (const robot::joint_index_t joint : description.getJointIndices()) {
    state.setJointPosition(joint, uniform(generator));
    state.setJointVelocity(joint, 10.0 * uniform(generator));
  }
  state.setContactFlag(/*index=*/0, uniform(generator) > 0.0);
  state.setContactFlag(/*index=*/1, uniform(generator) > 0.0);
  return state;
}

TEST(RecordedRobotStates, ARecordingReadsBackAsTheStatesThatWereRecorded) {
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(kG1Urdf);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  std::mt19937 generator(7);
  RecordedRobotStates recording;
  recording.jointNames = description.getJointNames();
  std::vector<robot::model::RobotState> states;
  for (int k = 0; k < 5; ++k) {
    states.push_back(randomState(description, generator));
    recording.records.push_back(recordRobotState(states.back(), description, Eigen::Vector4d(0.1 * k, -0.2, 0.79, 0.3)));
  }
  GoldenProvenance provenance;
  provenance.gitCommit = "c331ddd";
  const GoldenFile golden = toGoldenFile(recording, provenance);
  const absl::StatusOr<std::string> text = formatGoldenFile(golden);
  ASSERT_TRUE(text.ok()) << text.status();
  const absl::StatusOr<GoldenFile> parsed = parseGoldenFile(*text);
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  const absl::StatusOr<RecordedRobotStates> read = fromGoldenFile(*parsed);
  ASSERT_TRUE(read.ok()) << read.status();
  ASSERT_EQ(read->jointNames, recording.jointNames);
  ASSERT_EQ(read->records.size(), states.size());

  for (size_t k = 0; k < states.size(); ++k) {
    const absl::StatusOr<robot::model::RobotState> state = toRobotState(read->records[k], read->jointNames, description);
    ASSERT_TRUE(state.ok()) << state.status();
    const robot::model::RobotState& expected = states[k];
    EXPECT_EQ(state->getTime(), expected.getTime());
    EXPECT_EQ(state->getRootPositionInWorldFrame(), expected.getRootPositionInWorldFrame());
    EXPECT_EQ(state->getRootRotationLocalToWorldFrame().coeffs(), expected.getRootRotationLocalToWorldFrame().coeffs());
    EXPECT_EQ(state->getRootLinearVelocityInLocalFrame(), expected.getRootLinearVelocityInLocalFrame());
    EXPECT_EQ(state->getRootAngularVelocityInLocalFrame(), expected.getRootAngularVelocityInLocalFrame());
    for (const robot::joint_index_t joint : description.getJointIndices()) {
      EXPECT_EQ(state->getJointPosition(joint), expected.getJointPosition(joint));
      EXPECT_EQ(state->getJointVelocity(joint), expected.getJointVelocity(joint));
    }
    EXPECT_EQ(state->getContactFlags(), expected.getContactFlags());
    EXPECT_EQ(read->records[k].guiCommand, recording.records[k].guiCommand);
  }
}

TEST(RecordedRobotStates, JointsAreMatchedByNameWhateverTheirOrder) {
  // The order of a RobotDescription's joints is not the same in every process: a recording made in one process is read
  // in another, so its joints are matched by name.
  absl::StatusOr<robot::model::RobotDescription> descriptionOrStatus = robot::model::RobotDescription::Create(kG1Urdf);
  ASSERT_TRUE(descriptionOrStatus.ok()) << descriptionOrStatus.status();
  const robot::model::RobotDescription& description = *descriptionOrStatus;
  std::mt19937 generator(11);
  const robot::model::RobotState state = randomState(description, generator);
  const RobotStateRecord record = recordRobotState(state, description, Eigen::Vector4d::Zero());
  const std::vector<std::string>& names = description.getJointNames();
  std::vector<size_t> order(names.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = order.size() - 1 - i;  // reversed
  RobotStateRecord reordered = record;
  std::vector<std::string> reorderedNames(names.size());
  for (size_t i = 0; i < order.size(); ++i) {
    reorderedNames[i] = names[order[i]];
    reordered.jointPositions(static_cast<Eigen::Index>(i)) = record.jointPositions(static_cast<Eigen::Index>(order[i]));
    reordered.jointVelocities(static_cast<Eigen::Index>(i)) = record.jointVelocities(static_cast<Eigen::Index>(order[i]));
  }
  const absl::StatusOr<robot::model::RobotState> read = toRobotState(reordered, reorderedNames, description);
  ASSERT_TRUE(read.ok()) << read.status();
  for (const robot::joint_index_t joint : description.getJointIndices()) {
    EXPECT_EQ(read->getJointPosition(joint), state.getJointPosition(joint));
    EXPECT_EQ(read->getJointVelocity(joint), state.getJointVelocity(joint));
  }

  std::vector<std::string> duplicated = names;
  duplicated.back() = duplicated.front();
  EXPECT_EQ(toRobotState(record, duplicated, description).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(RecordedRobotStates, ARecordingRefusesAnotherRobot) {
  absl::StatusOr<robot::model::RobotDescription> g1OrStatus = robot::model::RobotDescription::Create(kG1Urdf);
  ASSERT_TRUE(g1OrStatus.ok()) << g1OrStatus.status();
  const robot::model::RobotDescription& g1 = *g1OrStatus;
  absl::StatusOr<robot::model::RobotDescription> atlasOrStatus = robot::model::RobotDescription::Create(kAtlasUrdf);
  ASSERT_TRUE(atlasOrStatus.ok()) << atlasOrStatus.status();
  const robot::model::RobotDescription& atlas = *atlasOrStatus;
  std::mt19937 generator(3);
  const RobotStateRecord record = recordRobotState(randomState(g1, generator), g1, Eigen::Vector4d::Zero());
  EXPECT_EQ(toRobotState(record, g1.getJointNames(), atlas).status().code(), absl::StatusCode::kInvalidArgument);

  GoldenFile withoutNames = toGoldenFile(RecordedRobotStates{.jointNames = g1.getJointNames(), .records = {record}}, GoldenProvenance());
  withoutNames.provenance.notes.clear();
  EXPECT_EQ(fromGoldenFile(withoutNames).status().code(), absl::StatusCode::kInvalidArgument);
  GoldenFile truncated = toGoldenFile(RecordedRobotStates{.jointNames = g1.getJointNames(), .records = {record}}, GoldenProvenance());
  truncated.entries.pop_back();
  EXPECT_EQ(fromGoldenFile(truncated).status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace ocs2::humanoid::validation
