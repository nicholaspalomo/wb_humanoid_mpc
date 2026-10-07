/******************************************************************************
Copyright (c) 2022, Halodi Robotics AS. All rights reserved.

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

@package humanoid_centroidal_mpc

@author Manuel Yves Galliker
Contact:  manuel.galliker@1x.tech
******************************************************************************/

#include "pinocchio/fwd.hpp"

#include <exception>
#include <string>
#include <utility>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "robot_core/ResourcePaths.h"

/**
 * A playground for the DRC Atlas Pinocchio model: prints the dimensions, the joints and the frames of the model from the
 * URDF and of the model the MPC builds, and the placement of the contact frames.
 */

namespace ocs2::humanoid {
namespace {

// Relative to the repository root.
constexpr absl::string_view kUrdfFile = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr absl::string_view kTaskFile = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto";
constexpr int kStateDim = 34;
constexpr double kNominalBaseHeight = 0.8415;

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
    LOG(INFO) << absl::StrFormat("%-24s", model.names[jointId]);
  }
}

/** Prints the orientation and the homogeneous transform of `frameName`, local to world, at `q`. */
void printFrameRotation(const PinocchioInterface& pinocchioInterface, const Eigen::VectorXd& q, const std::string& frameName) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data data = pinocchioInterface.getData();
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  const pinocchio::FrameIndex frameId = model.getFrameId(frameName);
  const matrix3_t rotationLocalToWorld = data.oMf[frameId].rotation();
  const quaternion_t quaternionLocalToWorld = matrixToQuaternion(rotationLocalToWorld);
  const Eigen::Matrix4d transform = data.oMf[frameId].toHomogeneousMatrix();
  LOG(INFO) << "Orientation of frame: R local to world " << frameName << ": ";
  LOG(INFO) << absl::StrFormat("[%g, %g, %g, %g]", quaternionLocalToWorld.w(), quaternionLocalToWorld.x(), quaternionLocalToWorld.y(),
                               quaternionLocalToWorld.z());
  LOG(INFO) << rotationLocalToWorld;
  LOG(INFO) << "Translation from local to world frame " << frameName << ": ";
  LOG(INFO) << transform;
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
    LOG(INFO) << absl::StrFormat("%-5s%d, %s: %s", "ID: ", jointId, model.names[jointId],
                                 formatFixed(data.oMi[jointId].translation(), /*precision=*/5));
  }
  LOG(INFO) << "###########################################";
  LOG(INFO) << "############### Model Frames ##############";
  LOG(INFO) << "###########################################";
  for (pinocchio::FrameIndex frameId = 0; frameId < static_cast<pinocchio::FrameIndex>(model.nframes); ++frameId) {
    LOG(INFO) << absl::StrFormat("%-10s%d, name: %s : Pos: %s", "ID: ", frameId, model.frames[frameId].name,
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

  // The default model.
  PinocchioInterface pinocchioInterface = createDefaultPinocchioInterface(*urdfFile);
  LOG(INFO) << "Default PinocchioInterface initialized ";
  printModelDimensionality(pinocchioInterface);
  printJointNames(pinocchioInterface);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(kStateDim);
  q[2] = kNominalBaseHeight;
  computeForwardKinematics(pinocchioInterface, q);
  printFrameRotation(pinocchioInterface, q, /*frameName=*/"foot_l_contact");
  printFrameRotation(pinocchioInterface, q, /*frameName=*/"foot_r_contact");

  // The model of the MPC, from the task file read once (strictly: an unknown field is an error with its line).
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
  absl::StatusOr<PinocchioInterface> customInterface = loadCustomPinocchioInterface(*task, *urdfFile, *modelSettings);
  if (!customInterface.ok()) {
    LOG(ERROR) << *taskFile << ": " << customInterface.status();
    return 1;
  }
  pinocchioInterface = *std::move(customInterface);
  LOG(INFO) << "Custom PinocchioInterface initialized ";
  printModelDimensionality(pinocchioInterface);
  printJointNames(pinocchioInterface);
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
