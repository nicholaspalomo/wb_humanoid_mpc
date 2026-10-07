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

#include <array>
#include <chrono>
#include <exception>
#include <random>
#include <string>
#include <utility>
#include <vector>

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
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "robot_core/ResourcePaths.h"

/**
 * A playground for the Unitree G1 Pinocchio model: prints the model from the URDF and the model the MPC builds, the
 * placement of the contact frames, an orientation error against the ground plane, and compares and times the custom
 * inverse dynamics against RNEA on the mass-scaled model.
 */

namespace ocs2::humanoid {
namespace {

// Relative to the repository root.
constexpr absl::string_view kUrdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr absl::string_view kTaskFile = "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto";

/** Random inputs of the inverse dynamics. */
struct InverseDynamicsSamples {
  std::vector<vector_t> q;
  std::vector<vector_t> qd;
  std::vector<vector_t> qddJoints;
  std::vector<std::array<VECTOR6_T<scalar_t>, 2>> footWrenches;
};

/** `size` standard-normal samples. */
vector_t randomNormal(int size, std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal;  // mean 0, standard deviation 1
  vector_t values(size);
  for (int j = 0; j < size; ++j) values[j] = normal(generator);
  return values;
}

/** `count` standard-normal samples of the generalized coordinates, velocities, joint accelerations and foot wrenches. */
InverseDynamicsSamples randomInverseDynamicsSamples(const PinocchioInterface& pinocchioInterface, int count) {
  const int nq = pinocchioInterface.getModel().nq;
  const int nv = pinocchioInterface.getModel().nv;
  const int numJoints = nv - 6;  // a floating base: 6 base degrees of freedom and the joints

  std::random_device seed;
  std::mt19937 generator(seed());
  InverseDynamicsSamples samples;
  for (int i = 0; i < count; ++i) {
    samples.q.push_back(randomNormal(nq, generator));
    samples.qd.push_back(randomNormal(nv, generator));
    samples.qddJoints.push_back(randomNormal(numJoints, generator));
    std::array<VECTOR6_T<scalar_t>, 2> footWrenches;
    for (VECTOR6_T<scalar_t>& wrench : footWrenches) wrench = randomNormal(/*size=*/6, generator);
    samples.footWrenches.push_back(footWrenches);
  }
  return samples;
}

/** The mean time of computeJointTorques() and of computeJointTorquesRNEA() over random inputs. */
void benchmarkInverseDynamics(PinocchioInterface& pinocchioInterface) {
  constexpr int kNumIterations = 10000;
  constexpr int kBatch = 10;
  const InverseDynamicsSamples samples = randomInverseDynamicsSamples(pinocchioInterface, kNumIterations);

  double customAverage = 0.0;
  double rneaAverage = 0.0;
  for (int i = 0; i < kNumIterations; i += kBatch) {
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < kBatch; ++j) {
      const vector_t result = computeJointTorques(samples.q[i + j], samples.qd[i + j], samples.qddJoints[i + j],
                                                  samples.footWrenches[i + j], pinocchioInterface);
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    customAverage += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < kBatch; ++j) {
      const vector_t result = computeJointTorquesRNEA(samples.q[i + j], samples.qd[i + j], samples.qddJoints[i + j],
                                                      samples.footWrenches[i + j], pinocchioInterface);
    }
    end = std::chrono::high_resolution_clock::now();
    rneaAverage += static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
  }
  customAverage /= kNumIterations;
  rneaAverage /= kNumIterations;

  LOG(INFO) << absl::StrFormat("Inverse Dynamics Benchmark Results (%d iterations):", kNumIterations);
  LOG(INFO) << "================================================";
  LOG(INFO) << absl::StrFormat("Custom Implementation:  %.2f μs average", customAverage);
  LOG(INFO) << absl::StrFormat("RNEA Implementation:    %.2f μs average", rneaAverage);
  LOG(INFO) << absl::StrFormat("Speed ratio (Custom/RNEA): %.2fx", customAverage / rneaAverage);
  if (customAverage < rneaAverage) {
    LOG(INFO) << absl::StrFormat("Custom implementation is %.2fx faster", rneaAverage / customAverage);
  } else {
    LOG(INFO) << absl::StrFormat("RNEA implementation is %.2fx faster", customAverage / rneaAverage);
  }
}

/** Prints computeJointTorques() next to computeJointTorquesRNEA() for a few random inputs. */
void compareInverseDynamics(PinocchioInterface& pinocchioInterface) {
  constexpr int kNumIterations = 10;
  const InverseDynamicsSamples samples = randomInverseDynamicsSamples(pinocchioInterface, kNumIterations);
  for (int i = 0; i < kNumIterations; ++i) {
    const vector_t resultCustom =
        computeJointTorques(samples.q[i], samples.qd[i], samples.qddJoints[i], samples.footWrenches[i], pinocchioInterface);
    const vector_t resultRnea =
        computeJointTorquesRNEA(samples.q[i], samples.qd[i], samples.qddJoints[i], samples.footWrenches[i], pinocchioInterface);
    LOG(INFO) << "Result custom:" << resultCustom.transpose();
    LOG(INFO) << "Result rnea  :" << resultRnea.transpose();
  }
}

/** The entries of `values`, each with `precision` decimals. */
std::string formatFixed(const Eigen::Ref<const Eigen::VectorXd>& values, int precision) {
  std::string text;
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (i > 0) text.push_back(' ');
    absl::StrAppendFormat(&text, "%.*f", precision, values(i));
  }
  return text;
}

/** Prints the distance of frame 69's orientation from the closest orientation whose z axis is the ground's normal. */
void testOrientationErrorWrtPlane(const PinocchioInterface& pinocchioInterface, const Eigen::VectorXd& q) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data data = pinocchioInterface.getData();
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  const vector3_t planeNormal(0.0, 0.0, 1.0);
  const size_t frameId = 69;
  const vector3_t zAxis(0.0, 0.0, 1.0);

  // Rotation matrix local end effector frame to world frame
  const matrix3_t rotationLocalToWorld = data.oMf[frameId].rotation();

  // Passive rotation projecting from end effector frame to the closest frame in plane.
  // Computed through the shortest arc rotation  from the end effector z axis to the plane normal (both expressed in world frame).
  const quaternion_t quaternionCorrection = getQuaternionFromUnitVectors<scalar_t>(rotationLocalToWorld * zAxis, planeNormal);
  LOG(INFO) << "quaternion_correction: " << quaternionCorrection.coeffs().transpose();

  const vector3_t error = quaternionDistance(quaternionCorrection, quaternion_t::Identity());
  LOG(INFO) << "error: " << error.transpose();
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

/** The default model, the model of the MPC, and the mass-scaled model of the MPC with its inverse dynamics. */
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

  Eigen::VectorXd q = Eigen::VectorXd::Zero(35);
  q[2] = 0.8415;
  computeForwardKinematics(pinocchioInterface, q);
  const std::string leftFootFrameName = "foot_l_contact";
  const std::string rightFootFrameName = "foot_r_contact";
  printFrameRotation(pinocchioInterface, q, leftFootFrameName);

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

  q = Eigen::VectorXd::Zero(29);
  q[2] = 0.7925;
  computeForwardKinematics(pinocchioInterface, q);
  printFrameRotation(pinocchioInterface, q, rightFootFrameName);
  printFrameRotation(pinocchioInterface, q, leftFootFrameName);
  testOrientationErrorWrtPlane(pinocchioInterface, q);

  // The model of the MPC with its total mass scaled.
  absl::StatusOr<PinocchioInterface> scaledInterface =
      loadCustomPinocchioInterface(*task, *urdfFile, *modelSettings, /*scaleTotalMass=*/true, /*totalMass=*/44.44);
  if (!scaledInterface.ok()) {
    LOG(ERROR) << *taskFile << ": " << scaledInterface.status();
    return 1;
  }
  pinocchioInterface = *std::move(scaledInterface);
  benchmarkInverseDynamics(pinocchioInterface);
  compareInverseDynamics(pinocchioInterface);
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
