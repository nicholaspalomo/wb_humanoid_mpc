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

@package unitree_r1_centroidal_mpc
******************************************************************************/

#include "pinocchio/fwd.hpp"

#include <exception>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "robot_core/ResourcePaths.h"

/**
 * A playground for the Unitree R1 Pinocchio model: prints the model from the URDF and the model the MPC builds, with the
 * placement of every joint and frame of the latter.
 */

namespace ocs2::humanoid {
namespace {

// Relative to the repository root.
constexpr absl::string_view kUrdfFile = "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf";
constexpr absl::string_view kTaskFile = "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto";

/** The entries of `values`, each with `precision` decimals. */
std::string formatFixed(const Eigen::Ref<const Eigen::VectorXd>& values, int precision) {
  std::string text;
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (i > 0) text.push_back(' ');
    absl::StrAppendFormat(&text, "%.*f", precision, values(i));
  }
  return text;
}

void printModelDimensionality(const PinocchioInterface& pinocchioInterface) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  LOG(INFO) << "model name: " << model.name;
  LOG(INFO) << "n q: " << model.nq;
  LOG(INFO) << "n v: " << model.nv;
}

void printJointNames(const PinocchioInterface& pinocchioInterface) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  for (pinocchio::JointIndex jointId = 0; jointId < static_cast<pinocchio::JointIndex>(model.njoints); ++jointId) {
    LOG(INFO) << absl::StrFormat("%-28s", model.names[jointId]);
  }
}

/** Prints the placement of every joint and every frame of the model at `q`. */
void computeForwardKinematics(const PinocchioInterface& pinocchioInterface, const Eigen::VectorXd& q) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data data = pinocchioInterface.getData();
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  LOG(INFO) << "###########################################";
  LOG(INFO) << "############### Model Joints ##############";
  LOG(INFO) << "###########################################";
  for (pinocchio::JointIndex jointId = 0; jointId < static_cast<pinocchio::JointIndex>(model.njoints); ++jointId) {
    LOG(INFO) << absl::StrFormat("%-5s%d, %-28s: %s", "ID: ", jointId, model.names[jointId],
                                 formatFixed(data.oMi[jointId].translation(), /*precision=*/5));
  }
  LOG(INFO) << "###########################################";
  LOG(INFO) << "############### Model Frames ##############";
  LOG(INFO) << "###########################################";
  for (pinocchio::FrameIndex frameId = 0; frameId < static_cast<pinocchio::FrameIndex>(model.nframes); ++frameId) {
    LOG(INFO) << absl::StrFormat("%-10s%d, name: %-28s : Pos: %s", "ID: ", frameId, model.frames[frameId].name,
                                 formatFixed(data.oMf[frameId].translation(), /*precision=*/5));
  }
}

/** The default model, straight from the URDF, and the model the MPC builds from the task file. */
int printModels() {
  // From the binary's runfiles (BUILD `data`), so `bazel run` and a run from .bazel/bin read the same files.
  const absl::StatusOr<std::string> urdfFile = robot::resolveResourcePath(kUrdfFile);
  const absl::StatusOr<std::string> taskFile = robot::resolveResourcePath(kTaskFile);
  if (!urdfFile.ok() || !taskFile.ok()) {
    LOG(ERROR) << "The model files are not in the runfiles: " << urdfFile.status() << "; " << taskFile.status();
    return 1;
  }
  LOG(INFO) << "urdf filename: " << *urdfFile;

  LOG(INFO) << "\n=== Testing Default PinocchioInterface for Unitree R1 ===";
  const PinocchioInterface defaultInterface = createDefaultPinocchioInterface(*urdfFile);
  printModelDimensionality(defaultInterface);
  printJointNames(defaultInterface);

  LOG(INFO) << "\n=== Testing Custom PinocchioInterface for Unitree R1 ===";
  // The task file, read once (strictly: an unknown field is an error with its line).
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(*taskFile);
  if (!task.ok()) {
    LOG(ERROR) << task.status();
    return 1;
  }
  absl::StatusOr<ModelSettings> modelSettings = ModelSettings::Create(*task, *urdfFile, /*mpcName=*/"test_pinocchio", /*verbose=*/true);
  if (!modelSettings.ok()) {
    LOG(ERROR) << *taskFile << ": " << modelSettings.status();
    return 1;
  }
  absl::StatusOr<PinocchioInterface> loadedInterface = loadCustomPinocchioInterface(*task, *urdfFile, *modelSettings);
  if (!loadedInterface.ok()) {
    LOG(ERROR) << *taskFile << ": " << loadedInterface.status();
    return 1;
  }
  const PinocchioInterface customInterface = *std::move(loadedInterface);
  printModelDimensionality(customInterface);
  printJointNames(customInterface);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(customInterface.getModel().nq);
  q[2] = 0.68;
  computeForwardKinematics(customInterface, q);

  LOG(INFO) << "\n✅ Unitree R1 Pinocchio Interface test completed successfully.";
  return 0;
}

}  // namespace
}  // namespace ocs2::humanoid

int main() {
  // Route Abseil log records to stderr. Without InitializeLog() Abseil warns once and writes everything to
  // stderr anyway; with it the default stderr threshold is ERROR, so the INFO records have to be asked for.
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  try {
    return ocs2::humanoid::printModels();
  } catch (const std::exception& error) {  // Pinocchio reports a URDF it cannot read by throwing.
    LOG(ERROR) << error.what();
    return 1;
  }
}
