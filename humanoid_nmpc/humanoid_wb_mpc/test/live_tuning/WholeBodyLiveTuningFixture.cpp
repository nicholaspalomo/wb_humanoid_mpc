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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/WholeBodyLiveTuningFixture.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/reference/ModeSchedule.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
#include "humanoid_mpc_config/joint_value.nproto.h"

namespace ocs2::humanoid::live_tuning_test {
namespace {

// LINT.IfChange(robot_files)
constexpr char kTaskFile[] = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto";
constexpr char kUrdfFile[] = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kReferenceFile[] = "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto";
constexpr char kGaitFile[] = "humanoid_nmpc/humanoid_common_mpc/config/command/gait.textproto";
// LINT.ThenChange(//humanoid_nmpc/humanoid_wb_mpc/BUILD.bazel:live_tuning_test_data)

/** The absolute path of `relativePath` in the runfiles, or an empty string. */
std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return std::filesystem::absolute(candidate).string();
  }
  return std::string();
}

WholeBodyFiles resolveFiles() {
  WholeBodyFiles files{.taskFile = runfilePath(kTaskFile),
                       .urdfFile = runfilePath(kUrdfFile),
                       .referenceFile = runfilePath(kReferenceFile),
                       .gaitFile = runfilePath(kGaitFile)};
  EXPECT_FALSE(files.taskFile.empty() || files.urdfFile.empty() || files.referenceFile.empty() || files.gaitFile.empty())
      << "the G1 whole-body files are not in the runfiles";
  return files;
}

/**
 * Makes a working directory for this run and enters it, once: the CppAD libraries go to a folder relative to it
 * (ModelSettings::modelFolderCppAd), and the shipped file does not force a recompile, so a library left by an earlier
 * unsandboxed run would be loaded instead of taped. The files are resolved before, as absolute paths.
 */
void enterFreshWorkingDirectoryOnce() {
  static const bool kEntered = []() {
    g1WholeBodyFiles();
    std::string freshDirectory = (std::filesystem::path(testing::TempDir()) / "test_wb_live_tuning_XXXXXX").string();
    if (mkdtemp(freshDirectory.data()) == nullptr) {
      ADD_FAILURE() << "cannot make a fresh working directory from " << freshDirectory;
      return false;
    }
    std::filesystem::current_path(freshDirectory);
    return true;
  }();
  EXPECT_TRUE(kEntered);
}

/** Moves `name` from `from` to `to`, where it must be listed once: the variant's term lists. */
void moveListEntry(std::vector<std::string>& from, std::vector<std::string>& to, absl::string_view name) {
  const std::vector<std::string>::iterator found = std::find(from.begin(), from.end(), name);
  EXPECT_NE(found, from.end()) << "the shipped file no longer lists " << name;
  if (found != from.end()) from.erase(found);
  to.emplace_back(name);
}

/** Replaces `name` in `list` with `replacement`. */
void replaceListEntry(std::vector<std::string>& list, absl::string_view name, absl::string_view replacement) {
  const std::vector<std::string>::iterator found = std::find(list.begin(), list.end(), name);
  EXPECT_NE(found, list.end()) << "the shipped file no longer lists " << name;
  if (found != list.end()) *found = std::string(replacement);
}

}  // namespace

const WholeBodyFiles& g1WholeBodyFiles() {
  static const absl::NoDestructor<WholeBodyFiles> kFiles(resolveFiles());
  return *kFiles;
}

mpc_config::TaskFile shippedTaskFile() {
  absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(g1WholeBodyFiles().taskFile);
  EXPECT_TRUE(task.ok()) << task.status();
  return task.ok() ? *std::move(task) : mpc_config::TaskFile{};
}

mpc_config::TaskFile variantTaskFile() {
  mpc_config::TaskFile task = shippedTaskFile();
  moveListEntry(task.hard_constraints, task.soft_constraints, "zero_velocity");
  moveListEntry(task.hard_constraints, task.soft_constraints, "normal_velocity");
  replaceListEntry(task.soft_constraints, "friction_force_cone", "contact_wrench_cone");
  task.costs.emplace_back("joint_torque_cost");
  // [rad] A toe-up swing: the swing foot's cost holds its tilt to a tilted plane, which orientation_z then weighs.
  task.swing_trajectory_config.swing_pitch_angle = 0.2;

  mpc_config::ContactsConfig::WrenchCone& cone = task.contacts.contact_wrench_cone_soft_constraint;
  cone.friction_coefficient = 0.5;
  cone.torsional_friction_coefficient = 0.05;
  cone.min_normal_force = 5.0;
  cone.gripper_force = 0.0;
  cone.num_basis_vectors = 4;
  cone.mu = 0.2;
  cone.delta = 5.0;

  // A weight for every joint of the MPC model, which the model settings read off the URDF.
  const absl::StatusOr<ModelSettings> modelSettings =
      ModelSettings::Create(task, g1WholeBodyFiles().urdfFile, "wb_mpc_", /*verbose=*/false);
  EXPECT_TRUE(modelSettings.ok()) << modelSettings.status();
  if (!modelSettings.ok()) return task;
  task.joint_torque_weights.scaling = 1.0e-3;
  task.joint_torque_weights.joints.clear();
  for (size_t i = 0; i < modelSettings->mpcModelJointNames.size(); ++i) {
    task.joint_torque_weights.joints.push_back(
        mpc_config::JointValue{.joint = modelSettings->mpcModelJointNames[i], .value = 1.0 + 0.1 * static_cast<double>(i % 5)});
  }
  return task;
}

std::unique_ptr<WBMpcInterface> createWholeBodyMpc(const mpc_config::TaskFile& task) {
  enterFreshWorkingDirectoryOnce();
  const WholeBodyFiles& files = g1WholeBodyFiles();
  const absl::StatusOr<mpc_config::ReferenceFile> reference = loadReferenceFile(files.referenceFile);
  EXPECT_TRUE(reference.ok()) << reference.status();
  if (!reference.ok()) return nullptr;
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> created = WBMpcInterface::Create(task, files.urdfFile, *reference);
  EXPECT_TRUE(created.ok()) << created.status();
  return created.ok() ? *std::move(created) : nullptr;
}

WBMpcInterface* absl_nullable shippedWholeBodyMpc() {
  static WBMpcInterface* absl_nullable const kInterface = createWholeBodyMpc(shippedTaskFile()).release();
  return kInterface;
}

WBMpcInterface* absl_nullable variantWholeBodyMpc() {
  static WBMpcInterface* absl_nullable const kInterface = createWholeBodyMpc(variantTaskFile()).release();
  return kInterface;
}

void scheduleOneSwing(WBMpcInterface& interface, size_t swingFoot) {
  contact_flag_t swingFlags = makeFeetArray(true);
  swingFlags[swingFoot] = false;
  const ModeSchedule schedule(
      /*eventTimesInput=*/{-10.0, kSwingStart, kSwingEnd, 10.0},
      /*modeSequenceInput=*/{ModeNumber::kStance, ModeNumber::kStance, stanceLeg2ModeNumber(swingFlags), ModeNumber::kStance,
                             ModeNumber::kStance});
  SwitchedModelReferenceManager& referenceManager = *interface.getSwitchedModelReferenceManagerPtr();
  referenceManager.getGaitSchedule()->updateModeSchedule(schedule);
  referenceManager.preSolverRun(kSwingStart - 0.2, kSwingEnd + 0.5, interface.getInitialState(), ModeNumber::kStance);
}

std::vector<EvaluationPoint> evaluationPoints(const WBMpcInterface& interface, size_t count) {
  std::mt19937 generator(20'261'005);
  std::uniform_real_distribution<scalar_t> unit(-1.0, 1.0);
  const WBAccelMpcRobotModel<scalar_t>& model = interface.getMpcRobotModel();
  const Eigen::Index numCoordinates = static_cast<Eigen::Index>(model.getGenCoordinatesDim());
  const Eigen::Index numWrenchInputs = static_cast<Eigen::Index>(6 * kNumContacts);
  // Inside the swing, and in stance before and after it.
  const std::vector<scalar_t> times = {0.1, 0.3, 0.45, -0.4, 0.9};
  std::vector<EvaluationPoint> points;
  for (size_t i = 0; i < count; ++i) {
    EvaluationPoint point;
    point.time = times[i % times.size()];
    point.state = interface.getInitialState();
    for (Eigen::Index j = 0; j < point.state.size(); ++j) {
      point.state(j) += (j < numCoordinates ? 0.05 : 0.5) * unit(generator);
    }
    point.input = vector_t::Zero(static_cast<Eigen::Index>(model.getInputDim()));
    for (Eigen::Index j = 0; j < point.input.size(); ++j) {
      // The contact wrenches first (a few hundred newtons), then the joint accelerations.
      point.input(j) = (j < numWrenchInputs ? 200.0 : 2.0) * unit(generator);
    }
    points.push_back(std::move(point));
  }
  return points;
}

}  // namespace ocs2::humanoid::live_tuning_test
