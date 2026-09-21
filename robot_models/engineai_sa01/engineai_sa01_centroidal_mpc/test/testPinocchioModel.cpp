/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
 *
 * @package engineai_sa01_centroidal_mpc
 *
 ******************************************************************************/

#include <pinocchio/fwd.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ocs2_core/Types.h>
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

using namespace ocs2;
using namespace ocs2::humanoid;

constexpr std::string_view kRobotModelPackagePath = "engineai_sa01_description";
constexpr std::string_view kUrdfFileName = "urdf/zq_sa01.urdf";
constexpr std::string_view kTaskConfigPath = "/../config/mpc/task.yaml";

// SA01 is a legs-only biped. The floating base is a JointModelTranslation + JointModelSphericalZYX composite
// (createPinocchioModel.cpp getBaseJointcomposite), so it contributes SIX configuration variables - x, y, z and the
// ZYX Euler angles, not a quaternion - and the twelve leg joints one each. The MPC state is these 18 plus the six
// normalised centroidal momenta, i.e. 24.
constexpr int kConfigurationDim = 18;
constexpr int kBaseDim = 6;

// The nominal standing pose of config/command/reference.yaml. Keep the two in sync: this test prints the sole and
// contact-frame heights that defaultBaseHeight and contacts.contact_frame_translation are derived from.
// LINT.IfChange(sa01_nominal_pose)
constexpr double kNominalBaseHeight = 0.8135;
constexpr double kNominalHipPitch = -0.30;
constexpr double kNominalKnee = 0.70;
constexpr double kNominalAnklePitch = -0.40;
// LINT.ThenChange(//robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/command/reference.yaml:sa01_nominal_pose)

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
    LOG(INFO) << std::setw(4) << std::left << joint_id << std::setw(24) << std::left << model.names[joint_id];
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
    LOG(INFO) << std::setw(5) << std::left << "ID: " << joint_id << ", " << model.names[joint_id] << ": " << std::fixed
              << std::setprecision(5) << data.oMi[joint_id].translation().transpose();
  }
  LOG(INFO) << "###########################################";
  LOG(INFO) << "############### Model Frames ##############";
  LOG(INFO) << "###########################################";
  for (pinocchio::FrameIndex frame_id = 0; frame_id < static_cast<pinocchio::FrameIndex>(model.nframes); ++frame_id) {
    LOG(INFO) << std::setw(10) << std::left << "ID: " << frame_id << ", name: " << model.frames[frame_id].name
              << " : Pos: " << std::setprecision(5) << data.oMf[frame_id].translation().transpose();
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
    LOG(INFO) << frameName << " position: " << std::fixed << std::setprecision(5) << data.oMf[frameId].translation().transpose();
  }
  LOG(INFO) << "base height : " << std::fixed << std::setprecision(5) << q[2] << " m (reference.yaml defaultBaseHeight)";
  LOG(INFO) << "CoM         : " << com.transpose();
  LOG(INFO) << "CoM height above the contact frames: " << com[2] - data.oMf[model.getFrameId("foot_l_contact")].translation()[2] << " m"
            << " (dcm_terminal_cost.comHeight)";
}

/**
 * @brief Prints the Pinocchio model the MPC builds for the EngineAI SA01, so that the joint order of task.yaml's Q / R
 * / initialState blocks, the contact frame placement and the nominal standing height can be checked against the URDF.
 */
int main() {
  // Route Abseil log records to stderr. Without InitializeLog() Abseil warns once and writes everything to
  // stderr anyway; with it the default stderr threshold is ERROR, so the INFO records have to be asked for.
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  const std::string path(__FILE__);
  const std::string dir = path.substr(0, path.find_last_of("/"));

  std::string urdfFile;
  try {
    urdfFile = ament_index_cpp::get_package_share_directory(std::string(kRobotModelPackagePath)) +
               std::filesystem::path::preferred_separator + std::string(kUrdfFileName);
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to get package share directory: engineai_sa01_description. Error: " + std::string(e.what()));
  }

  const std::string taskFile = dir + std::string(kTaskConfigPath);

  LOG(INFO) << "urdf filename: " << urdfFile;

  /// Default model, straight from the URDF: no contact frames, no fixed joints.
  PinocchioInterface pin_interface = createDefaultPinocchioInterface(urdfFile);
  LOG(INFO) << "Default PinocchioInterface initialized ";
  printModelDimensionality(pin_interface);
  printJointNames(pin_interface);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(kConfigurationDim);
  q[2] = kNominalBaseHeight;
  printFramePlacements(pin_interface, q);

  /// The model the MPC actually uses: the fixed joints of task.yaml removed and the contact frames added.
  ModelSettings modelSettings(taskFile, urdfFile, "test_pinocchio", "true");
  pin_interface = createCustomPinocchioInterface(taskFile, urdfFile, modelSettings);
  LOG(INFO) << "Custom PinocchioInterface initialized ";
  printModelDimensionality(pin_interface);
  printJointNames(pin_interface);

  // The nominal standing crouch of reference.yaml, in the MPC model's joint order
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
