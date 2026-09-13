/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

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
  AngularCenterOfMass(std::size_t inputDim, std::size_t numLayers, double omega0 = 30.0);

  /**
   * Runtime dispatch: creates an AngularCenterOfMass with the correct
   * per-robot weights selected by robot name (e.g., "atlas", "g1").
   *
   * This is the only supported way to obtain a loaded evaluator.
   *
   * @throws std::runtime_error if robotName is not recognized.
   */
  static std::unique_ptr<AngularCenterOfMass> createForRobot(const std::string& robotName);

  /**
   * Loads layer weights and biases from standard Eigen matrices.
   */
  void setWeights(const std::vector<SirenLayerWeights>& layers);

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

  std::size_t getInputDim() const { return inputDim_; }
  std::size_t getNumLayers() const { return numLayers_; }
  double getOmega0() const { return omega0_; }

 private:
  /**
   * Creates an AngularCenterOfMass instance from statically compiled weights for
   * a specific robot's weight struct (e.g. AcomSirenWeightsAtlas).
   *
   * Defined in the .cpp and instantiated only by createForRobot, so that adding a
   * robot means editing one dispatch rather than risking an undefined symbol at
   * a distant call site.
   *
   * @tparam WeightsT  Auto-generated weight struct with static constexpr members.
   */
  template <typename WeightsT>
  static std::unique_ptr<AngularCenterOfMass> createFromStaticWeights();

  /** Throws if qJoints does not have exactly inputDim_ entries. */
  void checkJointVectorSize(const vector_t& qJoints) const;

  /** Throws if setWeights has not been called. */
  void checkWeightsLoaded() const;

  std::size_t inputDim_;
  std::size_t numLayers_;
  double omega0_;
  std::vector<SirenLayerWeights> layers_;
};

}  // namespace ocs2::humanoid
