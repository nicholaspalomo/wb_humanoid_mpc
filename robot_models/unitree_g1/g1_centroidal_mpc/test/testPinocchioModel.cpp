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

@package legged_centroidal_mpc

@author Manuel Yves Galliker
Contact:  manuel.galliker@1x.tech
******************************************************************************/

#include "pinocchio/fwd.hpp"

#include <chrono>
#include <cstddef>
#include <exception>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/spatial/fwd.hpp"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ContactArray.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "legged_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "legged_mpc_config/task_file.nproto.h"
#include "robot_core/ResourcePaths.h"

/**
 * A playground for the Unitree G1 Pinocchio model: prints the model from the
 * URDF and the model the MPC builds, the placement of the contact frames, an
 * orientation error against the ground plane, and compares and times the custom
 * inverse dynamics against RNEA on the mass-scaled model.
 */

namespace ocs2::legged {
namespace {

// Relative to the repository root.
constexpr absl::string_view kUrdfFile =
    "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr absl::string_view kTaskFile =
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto";

/** Random inputs of the inverse dynamics. */
struct InverseDynamicsSamples {
  std::vector<vector_t> q;
  std::vector<vector_t> qd;
  std::vector<vector_t> qddJoints;
  std::vector<ContactArray<VECTOR6_T<scalar_t>>> wrenchFeetEwrtWorld;
};

/** `size` standard-normal samples. */
vector_t randomNormal(int size, std::mt19937& generator) {
  std::normal_distribution<scalar_t> normal;  // mean 0, standard deviation 1
  vector_t values(size);
  for (int j = 0; j < size; ++j) values[j] = normal(generator);
  return values;
}

/**
 * `count` standard-normal samples of the generalized coordinates, velocities,
 * joint accelerations and foot wrenches. The base orientation of each
 * configuration is its sample normalized (pinocchio::normalize()): a unit
 * quaternion.
 */
InverseDynamicsSamples randomInverseDynamicsSamples(
    const PinocchioInterface& pinocchioInterface, int count) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  const int nq = model.nq;
  const int nv = model.nv;
  const int numJoints =
      nv - 6;  // a floating base: 6 base degrees of freedom and the joints

  std::random_device seed;
  std::mt19937 generator(seed());
  InverseDynamicsSamples samples;
  for (int i = 0; i < count; ++i) {
    vector_t q = randomNormal(nq, generator);
    pinocchio::normalize(model, q);
    samples.q.push_back(std::move(q));
    samples.qd.push_back(randomNormal(nv, generator));
    samples.qddJoints.push_back(randomNormal(numJoints, generator));
    ContactArray<VECTOR6_T<scalar_t>> wrenchFeetEwrtWorld(
        static_cast<size_t>(kBipedFootCount));
    for (VECTOR6_T<scalar_t>& wrenchFootEwrtWorld : wrenchFeetEwrtWorld) {
      wrenchFootEwrtWorld = randomNormal(/*size=*/6, generator);
    }
    samples.wrenchFeetEwrtWorld.push_back(wrenchFeetEwrtWorld);
  }
  return samples;
}

/**
 * Returns the joint torques of the floating-base inverse dynamics by
 * Pinocchio's RNEA alone: the world-frame wrenches at the origins of
 * `soleFrames` reach RNEA as external forces on their parent joints, and the
 * base acceleration is the one at which RNEA's unactuated base rows vanish
 * (RNEA is affine in the acceleration: a_b = -M_bb^-1 tau_b(a_b = 0)).
 */
vector_t rneaJointTorques(
    const vector_t& q,
    const vector_t& qd,
    const vector_t& qddJoints,
    const ContactArray<VECTOR6_T<scalar_t>>& wrenchFeetEwrtWorld,
    const ContactArray<pinocchio::FrameIndex>& soleFrames,
    PinocchioInterface& pinocchioInterface) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data& data = pinocchioInterface.getData();
  pinocchio::framesForwardKinematics(model, data, q);
  std::vector<pinocchio::Force> fext(static_cast<size_t>(model.njoints),
                                     pinocchio::Force::Zero());
  for (size_t foot = 0; foot < soleFrames.size(); ++foot) {
    const pinocchio::JointIndex joint =
        model.frames[soleFrames[foot]].parentJoint;
    const matrix3_t rotmatLinkRtWorld = data.oMi[joint].rotation();
    const vector3_t posFootRtLinkEwrtWorld =
        data.oMf[soleFrames[foot]].translation() -
        data.oMi[joint].translation();
    const vector3_t forceFootEwrtWorld = wrenchFeetEwrtWorld[foot].head<3>();
    const vector3_t torqueLinkEwrtWorld =
        wrenchFeetEwrtWorld[foot].tail<3>() +
        posFootRtLinkEwrtWorld.cross(forceFootEwrtWorld);
    fext[joint] +=
        pinocchio::Force(rotmatLinkRtWorld.transpose() * forceFootEwrtWorld,
                         rotmatLinkRtWorld.transpose() * torqueLinkEwrtWorld);
  }
  vector_t a = vector_t::Zero(model.nv);
  a.tail(qddJoints.size()) = qddJoints;
  const vector6_t baseGeneralizedForceAtZeroBaseGeneralizedAcceleration =
      pinocchio::rnea(model, data, q, qd, a, fext).head<6>();
  pinocchio::crba(model, data, q);
  const matrix6_t baseMassMatrix =
      data.M.topLeftCorner<6, 6>().selfadjointView<Eigen::Upper>();
  a.head<6>() = -baseMassMatrix.ldlt().solve(
      baseGeneralizedForceAtZeroBaseGeneralizedAcceleration);
  return pinocchio::rnea(model, data, q, qd, a, fext).tail(qddJoints.size());
}

/** The mean time of computeJointTorques() and of rneaJointTorques() over random
 * inputs. */
void benchmarkInverseDynamics(
    PinocchioInterface& pinocchioInterface,
    const ContactArray<pinocchio::FrameIndex>& soleFrames) {
  constexpr int kNumIterations = 10000;
  constexpr int kBatch = 10;
  const InverseDynamicsSamples samples =
      randomInverseDynamicsSamples(pinocchioInterface, kNumIterations);

  double customAverage = 0.0;
  double rneaAverage = 0.0;
  for (int i = 0; i < kNumIterations; i += kBatch) {
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::now();
    for (int j = 0; j < kBatch; ++j) {
      const vector_t result = computeJointTorques(
          samples.q[i + j], samples.qd[i + j], samples.qddJoints[i + j],
          samples.wrenchFeetEwrtWorld[i + j], soleFrames, pinocchioInterface);
    }
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    customAverage += static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count());

    start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < kBatch; ++j) {
      const vector_t result = rneaJointTorques(
          samples.q[i + j], samples.qd[i + j], samples.qddJoints[i + j],
          samples.wrenchFeetEwrtWorld[i + j], soleFrames, pinocchioInterface);
    }
    end = std::chrono::high_resolution_clock::now();
    rneaAverage += static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count());
  }
  customAverage /= kNumIterations;
  rneaAverage /= kNumIterations;

  LOG(INFO) << absl::StrFormat(
      "Inverse Dynamics Benchmark Results (%d iterations):", kNumIterations);
  LOG(INFO) << "================================================";
  LOG(INFO) << absl::StrFormat("Custom Implementation:  %.2f μs average",
                               customAverage);
  LOG(INFO) << absl::StrFormat("RNEA Implementation:    %.2f μs average",
                               rneaAverage);
  LOG(INFO) << absl::StrFormat("Speed ratio (Custom/RNEA): %.2fx",
                               customAverage / rneaAverage);
  if (customAverage < rneaAverage) {
    LOG(INFO) << absl::StrFormat("Custom implementation is %.2fx faster",
                                 rneaAverage / customAverage);
  } else {
    LOG(INFO) << absl::StrFormat("RNEA implementation is %.2fx faster",
                                 customAverage / rneaAverage);
  }
}

/** Prints computeJointTorques() next to rneaJointTorques() and their largest
 * difference for a few random inputs. */
void compareInverseDynamics(
    PinocchioInterface& pinocchioInterface,
    const ContactArray<pinocchio::FrameIndex>& soleFrames) {
  constexpr int kNumIterations = 10;
  const InverseDynamicsSamples samples =
      randomInverseDynamicsSamples(pinocchioInterface, kNumIterations);
  for (int i = 0; i < kNumIterations; ++i) {
    const vector_t resultCustom = computeJointTorques(
        samples.q[i], samples.qd[i], samples.qddJoints[i],
        samples.wrenchFeetEwrtWorld[i], soleFrames, pinocchioInterface);
    const vector_t resultRnea = rneaJointTorques(
        samples.q[i], samples.qd[i], samples.qddJoints[i],
        samples.wrenchFeetEwrtWorld[i], soleFrames, pinocchioInterface);
    LOG(INFO) << "Result custom:" << resultCustom.transpose();
    LOG(INFO) << "Result rnea  :" << resultRnea.transpose();
    LOG(INFO) << "Largest difference: "
              << (resultCustom - resultRnea).cwiseAbs().maxCoeff();
  }
}

/**
 * The neutral configuration of `model` (pinocchio::neutral(): the base at the
 * origin, upright, with the identity quaternion of the Composite(Translation,
 * Spherical) root, and every joint at zero) with the base raised to
 * `baseHeight`.
 */
Eigen::VectorXd neutralConfigurationAtHeight(const pinocchio::Model& model,
                                             double baseHeight) {
  Eigen::VectorXd q = pinocchio::neutral(model);
  // The root joint (joint 1, after the universe) starts its configuration with
  // the base position (x, y, z).
  q[model.joints[1].idx_q() + 2] = baseHeight;
  return q;
}

/** The entries of `values`, each with `precision` decimals. */
std::string formatFixed(const Eigen::Ref<const Eigen::VectorXd>& values,
                        int precision) {
  std::string text;
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    if (i > 0) text.push_back(' ');
    absl::StrAppendFormat(&text, "%.*f", precision, values(i));
  }
  return text;
}

/** Prints the distance of frame 69's orientation from the closest orientation
 * whose z axis is the ground's normal. */
void printPlaneTiltError(const PinocchioInterface& pinocchioInterface,
                         const Eigen::VectorXd& q) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data data = pinocchioInterface.getData();
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  const vector3_t normalSurfaceEwrtWorld(0.0, 0.0, 1.0);
  const size_t frameId = 69;
  const vector3_t zAxis(0.0, 0.0, 1.0);

  // Rotation matrix local end effector frame to world frame
  const matrix3_t rotmatLinkRtWorld = data.oMf[frameId].rotation();

  // Passive rotation projecting from end effector frame to the closest frame in
  // plane. Computed through the shortest arc rotation  from the end effector z
  // axis to the plane normal (both expressed in world frame).
  const quaternion_t shortestArc = getQuaternionFromUnitVectors<scalar_t>(
      rotmatLinkRtWorld * zAxis, normalSurfaceEwrtWorld);
  LOG(INFO) << "quaternion_correction: " << shortestArc.coeffs().transpose();

  const vector3_t error =
      quaternionDistance(shortestArc, quaternion_t::Identity());
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
  for (pinocchio::JointIndex jointId = 0;
       jointId < static_cast<pinocchio::JointIndex>(model.njoints); ++jointId) {
    LOG(INFO) << absl::StrFormat("%-24s", model.names[jointId]);
  }
}

/** Prints the orientation and the homogeneous transform of `frameName`, local
 * to world, at `q`. */
void printPoseLinkRtWorld(const PinocchioInterface& pinocchioInterface,
                          const Eigen::VectorXd& q,
                          absl::string_view frameName) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data data = pinocchioInterface.getData();
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  const pinocchio::FrameIndex frameId =
      model.getFrameId(std::string(frameName));
  const matrix3_t rotmatLinkRtWorld = data.oMf[frameId].rotation();
  const quaternion_t quatLinkRtWorld = matrixToQuaternion(rotmatLinkRtWorld);
  const Eigen::Matrix4d transform = data.oMf[frameId].toHomogeneousMatrix();
  LOG(INFO) << "Orientation of frame: R local to world " << frameName << ": ";
  LOG(INFO) << absl::StrFormat("[%g, %g, %g, %g]", quatLinkRtWorld.w(),
                               quatLinkRtWorld.x(), quatLinkRtWorld.y(),
                               quatLinkRtWorld.z());
  LOG(INFO) << rotmatLinkRtWorld;
  LOG(INFO) << "Translation from local to world frame " << frameName << ": ";
  LOG(INFO) << transform;
}

/** Prints the placement of every joint and every frame of the model at `q`. */
void computeForwardKinematics(const PinocchioInterface& pinocchioInterface,
                              const Eigen::VectorXd& q) {
  const pinocchio::Model& model = pinocchioInterface.getModel();
  pinocchio::Data data = pinocchioInterface.getData();
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  LOG(INFO) << "###########################################";
  LOG(INFO) << "############### Model Joints ##############";
  LOG(INFO) << "###########################################";
  for (pinocchio::JointIndex jointId = 0;
       jointId < static_cast<pinocchio::JointIndex>(model.njoints); ++jointId) {
    LOG(INFO) << absl::StrFormat(
        "%-5s%d, %s: %s", "ID: ", jointId, model.names[jointId],
        formatFixed(data.oMi[jointId].translation(), /*precision=*/5));
  }
  LOG(INFO) << "###########################################";
  LOG(INFO) << "############### Model Frames ##############";
  LOG(INFO) << "###########################################";
  for (pinocchio::FrameIndex frameId = 0;
       frameId < static_cast<pinocchio::FrameIndex>(model.nframes); ++frameId) {
    LOG(INFO) << absl::StrFormat(
        "%-10s%d, name: %s : Pos: %s", "ID: ", frameId,
        model.frames[frameId].name,
        formatFixed(data.oMf[frameId].translation(), /*precision=*/5));
  }
}

/** The default model, the model of the MPC, and the mass-scaled model of the
 * MPC with its inverse dynamics. */
int printModels() {
  // From the binary's runfiles (BUILD `data`), so `bazel run` and a run from
  // .bazel/bin read the same files.
  const absl::StatusOr<std::string> urdfFile =
      robot::resolveResourcePath(kUrdfFile);
  const absl::StatusOr<std::string> taskFile =
      robot::resolveResourcePath(kTaskFile);
  if (!urdfFile.ok() || !taskFile.ok()) {
    LOG(ERROR) << "The model files are not in the runfiles: "
               << urdfFile.status() << "; " << taskFile.status();
    return 1;
  }
  LOG(INFO) << "urdf filename: " << *urdfFile;

  // The default model.
  PinocchioInterface pinocchioInterface =
      createDefaultPinocchioInterface(*urdfFile);
  LOG(INFO) << "Default PinocchioInterface initialized ";
  printModelDimensionality(pinocchioInterface);
  printJointNames(pinocchioInterface);

  Eigen::VectorXd q = neutralConfigurationAtHeight(
      pinocchioInterface.getModel(), /*baseHeight=*/0.8415);
  computeForwardKinematics(pinocchioInterface, q);
  const std::string leftFootFrameName = "foot_l_contact";
  const std::string rightFootFrameName = "foot_r_contact";
  printPoseLinkRtWorld(pinocchioInterface, q, leftFootFrameName);

  // The model of the MPC, from the task file read once (strictly: an unknown
  // field is an error with its line).
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(*taskFile);
  if (!task.ok()) {
    LOG(ERROR) << task.status();
    return 1;
  }
  absl::StatusOr<ModelSettings> modelSettings = ModelSettings::Create(
      *task, *urdfFile, /*mpcName=*/"test_pinocchio", /*verbose=*/true);
  if (!modelSettings.ok()) {
    LOG(ERROR) << *taskFile << ": " << modelSettings.status();
    return 1;
  }
  absl::StatusOr<PinocchioInterface> customInterface =
      loadCustomPinocchioInterface(*task, *urdfFile, *modelSettings);
  if (!customInterface.ok()) {
    LOG(ERROR) << *taskFile << ": " << customInterface.status();
    return 1;
  }
  pinocchioInterface = *std::move(customInterface);
  LOG(INFO) << "Custom PinocchioInterface initialized ";
  printModelDimensionality(pinocchioInterface);
  printJointNames(pinocchioInterface);

  q = neutralConfigurationAtHeight(pinocchioInterface.getModel(),
                                   /*baseHeight=*/0.7925);
  computeForwardKinematics(pinocchioInterface, q);
  printPoseLinkRtWorld(pinocchioInterface, q, rightFootFrameName);
  printPoseLinkRtWorld(pinocchioInterface, q, leftFootFrameName);
  printPlaneTiltError(pinocchioInterface, q);

  // The model of the MPC with its total mass scaled.
  absl::StatusOr<PinocchioInterface> scaledInterface =
      loadCustomPinocchioInterface(*task, *urdfFile, *modelSettings,
                                   /*scaleTotalMass=*/true,
                                   /*totalMass=*/44.44);
  if (!scaledInterface.ok()) {
    LOG(ERROR) << *taskFile << ": " << scaledInterface.status();
    return 1;
  }
  pinocchioInterface = *std::move(scaledInterface);
  if (const absl::Status frames = checkContactFramesExist(
          pinocchioInterface.getModel(), modelSettings->contactNames);
      !frames.ok()) {
    LOG(ERROR) << *taskFile << ": " << frames;
    return 1;
  }
  // The G1 is a biped: its two sole frames are its contacts.
  ContactArray<pinocchio::FrameIndex> soleFrames(
      static_cast<size_t>(kBipedFootCount));
  soleFrames[kContactLeftIndex] =
      pinocchioInterface.getModel().getFrameId(modelSettings->contactNames[0]);
  soleFrames[kContactRightIndex] =
      pinocchioInterface.getModel().getFrameId(modelSettings->contactNames[1]);
  benchmarkInverseDynamics(pinocchioInterface, soleFrames);
  compareInverseDynamics(pinocchioInterface, soleFrames);
  return 0;
}

}  // namespace
}  // namespace ocs2::legged

int main() {
  // Route Abseil log records to stderr. Without InitializeLog() Abseil warns
  // once and writes everything to stderr anyway; with it the default stderr
  // threshold is ERROR, so the INFO records have to be asked for.
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  try {
    return ocs2::legged::printModels();
  } catch (const std::exception&
               error) {  // Pinocchio reports a URDF it cannot read by throwing.
    LOG(ERROR) << error.what();
    return 1;
  }
}
