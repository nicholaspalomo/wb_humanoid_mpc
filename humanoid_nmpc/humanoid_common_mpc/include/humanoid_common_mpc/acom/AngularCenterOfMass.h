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

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * Reorders an aCOM network output from the network's XYZ convention
 * [roll, pitch, yaw] to the OCS2 centroidal state's ZYX convention
 * [yaw, pitch, roll].
 */
inline vector3_t acomXyzToZyx(const vector3_t& xyz) {
  return vector3_t(xyz[2], xyz[1], xyz[0]);
}

/**
 * Reorders the rows of a (3 x n) aCOM Jacobian from the network's XYZ convention
 * to the OCS2 centroidal state's ZYX convention, by swapping the roll and yaw
 * rows and leaving the pitch row in place.
 */
inline matrix_t acomJacobianXyzToZyx(const matrix_t& J_xyz) {
  matrix_t J_zyx(J_xyz.rows(), J_xyz.cols());
  J_zyx.row(0) = J_xyz.row(2);  // Yaw.
  J_zyx.row(1) = J_xyz.row(1);  // Pitch.
  J_zyx.row(2) = J_xyz.row(0);  // Roll.
  return J_zyx;
}

/**
 * Structure holding parameters for a single SIREN layer in C++.
 */
struct SirenLayerWeights {
  matrix_t weight;  // (out_dim x in_dim)
  vector_t bias;    // (out_dim)
};

/**
 * Angular Center of Mass (aCOM) C++ Evaluator.
 *
 * Evaluates the integrable whole-body orientation coordinate theta_aCOM(q) and
 * its exact analytical Jacobian J_aCOM(q) from the generalized coordinates
 * q = [pos_base, euler_zyx_base, q_joints] of the OCS2 centroidal model.
 *
 * The underlying SIREN network is trained in Python and emits its 3-vector
 * output in XYZ order [roll, pitch, yaw]; computeJointOrientationOffset and
 * computeJointOffsetJacobian return that native ordering, while
 * computeAcomOrientation and computeAcomJacobian convert to the ZYX ordering the
 * centroidal state uses.
 */
class AngularCenterOfMass {
 public:
  AngularCenterOfMass(size_t inputDim, size_t numLayers, double omega0 = 30.0);

  /**
   * The values of model_settings.robot_name that have a network compiled into this binary, sorted. This is the
   * registry Create() resolves names against, and the list its unknown-robot error prints.
   */
  static std::vector<std::string> registeredRobotNames();

  /**
   * Creates the evaluator for `robotName` (model_settings.robot_name) and checks it against the model it will be fed.
   *
   * The network is a function of a VECTOR of joint angles, so it is only meaningful when `modelJointNames` - the
   * MPC model's joints, in the order the MPC hands them over (ModelSettings::mpcModelJointNames, or equivalently the
   * reduced Pinocchio model's joints after the floating base) - is exactly the joint list recorded in the weight
   * header at training time. A permutation, or a different set of fixed joints of the same size, keeps every
   * dimension check green while the network silently evaluates a different robot, so the names are compared one by
   * one here rather than only counted.
   *
   * @return NotFound if no network is registered for `robotName`, listing the registered names;
   *     FailedPrecondition if the joint list differs from the one the network was trained on, naming the robot and the
   *     first joint that differs.
   */
  static absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> Create(absl::string_view robotName,
                                                                     const std::vector<std::string>& modelJointNames);

  /**
   * Validates and loads layer weights and biases. Returns InvalidArgument, and leaves the evaluator unchanged, if the
   * layer count, a layer's shape or the 3-wide readout does not match the architecture this evaluator was built for.
   */
  absl::Status loadWeights(const std::vector<SirenLayerWeights>& layers);

  /**
   * Records the joint names, in input order, that the network was trained on. Returns InvalidArgument unless there is
   * exactly one name per network input.
   */
  absl::Status setJointNames(std::vector<std::string> jointNames);

  /** The joint names recorded by setJointNames(), in input order; empty if none were recorded. */
  const std::vector<std::string>& getJointNames() const { return jointNames_; }

  /**
   * Computes the joint-induced orientation offset Delta_theta(q_j) in R^3.
   */
  vector3_t computeJointOrientationOffset(const vector_t& qJoints) const;

  /**
   * Computes the exact analytical Jacobian d(Delta_theta)/d(q_j) in R^(3 x n_j).
   */
  matrix_t computeJointOffsetJacobian(const vector_t& qJoints) const;

  /**
   * Computes the whole-body aCOM orientation in the OCS2 ZYX Euler convention:
   * theta_aCOM(q) = euler_zyx_base + P * Delta_theta(q_j),
   * where P swaps the roll and yaw rows of the network's XYZ output.
   *
   * @param q: Pinocchio generalized coordinates of the OCS2 centroidal model,
   *     laid out as [pos_base(3), euler_zyx_base(3), q_joints(n_j)].
   */
  vector3_t computeAcomOrientation(const vector_t& q) const;

  /**
   * Computes the whole-body aCOM configuration Jacobian d(theta_aCOM)/dq in the
   * OCS2 ZYX Euler convention:
   * J_aCOM(q) = [0_(3x3), I_(3x3), P * J_Delta_theta_(3xn_j)].
   *
   * The base orientation block is the identity because the aCOM orientation is
   * the base Euler angle triple plus a joint-only offset. Note that this
   * differentiates with respect to the Euler angles themselves, not with respect
   * to an angular velocity.
   */
  matrix_t computeAcomJacobian(const vector_t& q) const;

  size_t getInputDim() const { return inputDim_; }

 private:
  // The two checks below guard the evaluation hot path against PROGRAMMING errors - a joint vector of the wrong size,
  // or an evaluator used before its weights were loaded - which Create() has already ruled out for every
  // configuration it accepts. They are ABSL_CHECKs, so that the evaluation methods keep returning plain values: an
  // invariant that does not hold ends the process with the message instead of evaluating out of bounds.

  /** Checks that qJoints has exactly inputDim_ entries. */
  void checkJointVectorSize(const vector_t& qJoints) const;

  /** Checks that weights have been loaded. */
  void checkWeightsLoaded() const;

  size_t inputDim_;
  size_t numLayers_;
  double omega0_;
  std::vector<SirenLayerWeights> layers_;
  std::vector<std::string> jointNames_;
};

}  // namespace ocs2::humanoid
