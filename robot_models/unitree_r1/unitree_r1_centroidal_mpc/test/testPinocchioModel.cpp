/******************************************************************************
Copyright (c) 2024, Unitree R1 Centroidal MPC
 *
 * @package unitree_r1_centroidal_mpc
 ******************************************************************************/

#include <pinocchio/fwd.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/joint/joint-composite.hpp>
#include <pinocchio/multibody/model.hpp>

#include <ocs2_core/Types.h>
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"

#include <ocs2_robotic_tools/common/RotationTransforms.h>
#include <ocs2_robotic_tools/common/SkewSymmetricMatrix.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

using namespace ocs2;
using namespace ocs2::humanoid;

void printModelDimensionality(PinocchioInterface pin_interface) {
  pinocchio::Model model = pin_interface.getModel();
  std::cout << "model name: " << model.name << std::endl;
  std::cout << "n q: " << model.nq << std::endl;
  std::cout << "n v: " << model.nv << std::endl;
}

void printJointNames(PinocchioInterface pin_interface) {
  pinocchio::Model model = pin_interface.getModel();
  for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex)model.njoints; ++joint_id) {
    std::cout << std::setw(28) << std::left << model.names[joint_id] << std::endl;
  }
}

void computeForwardKinematics(PinocchioInterface pin_interface, Eigen::VectorXd q) {
  pinocchio::Model model = pin_interface.getModel();
  pinocchio::Data data = pin_interface.getData();

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  std::cout << "###########################################" << std::endl;
  std::cout << "############### Model Joints ##############" << std::endl;
  std::cout << "###########################################" << std::endl;
  for (pinocchio::JointIndex joint_id = 0; joint_id < (pinocchio::JointIndex)model.njoints; ++joint_id) {
    std::cout << std::setw(5) << std::left << "ID: " << joint_id << ", " << std::setw(28) << model.names[joint_id] << ": " << std::fixed
              << std::setprecision(5) << data.oMi[joint_id].translation().transpose() << std::endl;
  }
  std::cout << "###########################################" << std::endl;
  std::cout << "############### Model Frames ##############" << std::endl;
  std::cout << "###########################################" << std::endl;
  for (pinocchio::FrameIndex frame_id = 0; frame_id < (pinocchio::FrameIndex)model.nframes; ++frame_id) {
    std::cout << std::setw(10) << std::left << "ID: " << frame_id << ", name: " << std::setw(28) << model.frames[frame_id].name
              << " : Pos: " << std::setprecision(5) << data.oMf[frame_id].translation().transpose() << std::endl;
  }
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  const std::string path(__FILE__);
  const std::string dir = path.substr(0, path.find_last_of("/"));

  std::string urdfFile;
  try {
    urdfFile = ament_index_cpp::get_package_share_directory("unitree_r1_description") + "/urdf/R1.urdf";
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to get package share directory: unitree_r1_description. Error: " + std::string(e.what()));
  }

  const std::string taskFile = dir + "/../config/mpc/task.yaml";

  std::cout << "urdf filename: " << urdfFile << std::endl;

  std::cout << "\n=== Testing Default PinocchioInterface for Unitree R1 ===" << std::endl;
  PinocchioInterface pin_interface = createDefaultPinocchioInterface(urdfFile);
  printModelDimensionality(pin_interface);
  printJointNames(pin_interface);

  std::cout << "\n=== Testing Custom PinocchioInterface for Unitree R1 ===" << std::endl;
  ModelSettings modelSettings(taskFile, urdfFile, "test_pinocchio", "true");
  PinocchioInterface custom_pin_interface = createCustomPinocchioInterface(taskFile, urdfFile, modelSettings);
  printModelDimensionality(custom_pin_interface);
  printJointNames(custom_pin_interface);

  Eigen::VectorXd q = Eigen::VectorXd::Zero(custom_pin_interface.getModel().nq);
  q[2] = 0.68;
  computeForwardKinematics(custom_pin_interface, q);

  std::cout << "\n✅ Unitree R1 Pinocchio Interface test completed successfully." << std::endl;
  return 0;
}
