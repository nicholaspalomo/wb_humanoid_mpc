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

@package engineai_sa01_centroidal_mpc
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
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "robot_core/ResourcePaths.h"

/**
 * Prints the Pinocchio model the MPC builds for the EngineAI SA01, so that the joint order of its state, the contact
 * frame placement and the nominal standing height can be checked against the URDF.
 */

namespace ocs2::humanoid {
namespace {

// Relative to the repository root.
constexpr absl::string_view kUrdfFile = "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf";
constexpr absl::string_view kTaskFile = "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto";

// SA01 is a legs-only biped. The floating base is a JointModelTranslation + JointModelSphericalZYX composite
// (createPinocchioModel.cpp getBaseJointcomposite), so it contributes SIX configuration variables - x, y, z and the
// ZYX Euler angles, not a quaternion - and the twelve leg joints one each. The MPC state is these 18 plus the six
// normalized centroidal momenta, i.e. 24.
constexpr int kConfigurationDim = 18;
constexpr int kBaseDim = 6;

// The nominal standing pose of config/command/reference.textproto. Keep the two in sync: this test prints the sole and
// contact-frame heights that default_base_height and contacts.contact_frame_translation are derived from.
// LINT.IfChange(sa01_nominal_pose)
constexpr double kNominalBaseHeight = 0.8135;
constexpr double kNominalHipPitch = -0.30;
constexpr double kNominalKnee = 0.70;
constexpr double kNominalAnklePitch = -0.40;
// LINT.ThenChange(//robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/command/reference.textproto:sa01_nominal_pose)

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
  LOG(INFO) << "total mass: " << pinocchio::computeTotalMass(model) << " kg";
}

void printJointNames(const PinocchioInterface& pinocchioInterface) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  for (pinocchio::JointIndex joint_id = 0; joint_id < static_cast<pinocchio::JointIndex>(model.njoints); ++joint_id) {
    LOG(INFO) << absl::StrFormat("%-4d%-24s", joint_id, model.names[joint_id]);
  }
}

void printFramePlacements(PinocchioInterface& pinocchioInterface, const Eigen::VectorXd& q) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data& data = pinocchioInterface.getData();

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  LOG(INFO) << "###########################################";
  LOG(INFO) << "############### Model Joints ##############";
  LOG(INFO) << "###########################################";
  for (pinocchio::JointIndex joint_id = 0; joint_id < static_cast<pinocchio::JointIndex>(model.njoints); ++joint_id) {
    LOG(INFO) << absl::StrFormat("%-5s%d, %s: %s", "ID: ", joint_id, model.names[joint_id],
                                 formatFixed(data.oMi[joint_id].translation(), /*precision=*/5));
  }
  LOG(INFO) << "###########################################";
  LOG(INFO) << "############### Model Frames ##############";
  LOG(INFO) << "###########################################";
  for (pinocchio::FrameIndex frame_id = 0; frame_id < static_cast<pinocchio::FrameIndex>(model.nframes); ++frame_id) {
    LOG(INFO) << absl::StrFormat("%-10s%d, name: %s : Pos: %s", "ID: ", frame_id, model.frames[frame_id].name,
                                 formatFixed(data.oMf[frame_id].translation(), /*precision=*/5));
  }
}

void printContactFrameHeights(PinocchioInterface& pinocchioInterface, const Eigen::VectorXd& q) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data& data = pinocchioInterface.getData();

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  const vector3_t com = pinocchio::centerOfMass(model, data, q);

  LOG(INFO) << "###########################################";
  LOG(INFO) << "########## Nominal Standing Pose ##########";
  LOG(INFO) << "###########################################";
  for (const std::string& frameName : {std::string("foot_l_contact"), std::string("foot_r_contact")}) {
    if (!model.existFrame(frameName)) {
      LOG(INFO) << frameName << ": NOT A FRAME OF THE MODEL";
      continue;
    }
    const pinocchio::FrameIndex frameId = model.getFrameId(frameName);
    LOG(INFO) << frameName << " position: " << formatFixed(data.oMf[frameId].translation(), /*precision=*/5);
  }
  LOG(INFO) << absl::StrFormat("base height : %.5f m (reference.textproto default_base_height)", q[2]);
  LOG(INFO) << "CoM         : " << com.transpose();
  LOG(INFO) << "CoM height above the contact frames: " << com[2] - data.oMf[model.getFrameId("foot_l_contact")].translation()[2] << " m"
            << " (dcm_terminal_cost.com_height)";
}

/** The default model, straight from the URDF, and the model the MPC builds from the task file. */
int printModels() {
  // From the binary's runfiles (BUILD `data`), so `bazel run` and a run from .bazel/bin read the same files.
  const absl::StatusOr<std::string> resolvedUrdfFile = robot::resolveResourcePath(kUrdfFile);
  const absl::StatusOr<std::string> resolvedTaskFile = robot::resolveResourcePath(kTaskFile);
  if (!resolvedUrdfFile.ok() || !resolvedTaskFile.ok()) {
    LOG(ERROR) << "The model files are not in the runfiles: " << resolvedUrdfFile.status() << "; " << resolvedTaskFile.status();
    return 1;
  }
  const std::string& urdfFile = *resolvedUrdfFile;
  const std::string& taskFile = *resolvedTaskFile;

  LOG(INFO) << "urdf filename: " << urdfFile;

  /// Default model, straight from the URDF: no contact frames, no fixed joints.
  PinocchioInterface pin_interface = createDefaultPinocchioInterface(urdfFile);
  LOG(INFO) << "Default PinocchioInterface initialized ";
  printModelDimensionality(pin_interface);
  printJointNames(pin_interface);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(kConfigurationDim);
  q[2] = kNominalBaseHeight;
  printFramePlacements(pin_interface, q);

  /// The model the MPC actually uses: the fixed joints of the task file removed and the contact frames added. The task
  /// file is read once, strictly: an unknown field is an error with its line.
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(taskFile);
  if (!task.ok()) {
    LOG(ERROR) << task.status();
    return 1;
  }
  absl::StatusOr<ModelSettings> modelSettings = ModelSettings::Create(*task, urdfFile, /*mpcName=*/"test_pinocchio", /*verbose=*/true);
  if (!modelSettings.ok()) {
    LOG(ERROR) << taskFile << ": " << modelSettings.status();
    return 1;
  }
  absl::StatusOr<PinocchioInterface> customInterface = loadCustomPinocchioInterface(*task, urdfFile, *modelSettings);
  if (!customInterface.ok()) {
    LOG(ERROR) << taskFile << ": " << customInterface.status();
    return 1;
  }
  pin_interface = *std::move(customInterface);
  LOG(INFO) << "Custom PinocchioInterface initialized ";
  printModelDimensionality(pin_interface);
  printJointNames(pin_interface);

  // The nominal standing crouch of reference.textproto, in the MPC model's joint order
  // (leg_l1 .. leg_l6, leg_r1 .. leg_r6).
  q = Eigen::VectorXd::Zero(pin_interface.getModel().nq);
  q[2] = kNominalBaseHeight;
  for (const int legOffset : {kBaseDim, kBaseDim + 6}) {
    q[legOffset + 2] = kNominalHipPitch;    // leg_?3_joint - hip pitch
    q[legOffset + 3] = kNominalKnee;        // leg_?4_joint - knee
    q[legOffset + 4] = kNominalAnklePitch;  // leg_?5_joint - ankle pitch
  }
  printContactFrameHeights(pin_interface, q);

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
