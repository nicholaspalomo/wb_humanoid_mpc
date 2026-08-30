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
#include <cmath>
#include <vector>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

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
 * Evaluates the integrable whole-body orientation coordinate theta_aCOM(q)
 * and its exact analytical Jacobian J_aCOM(q) from generalized coordinates q = [pos_base, rpy_base, q_joints].
 */
class AngularCenterOfMass {
 public:
  AngularCenterOfMass(size_t inputDim, size_t hiddenDim, size_t numLayers, double omega0 = 30.0);

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
   * Computes the full whole-body aCOM orientation:
   * theta_aCOM(q) = rpy_base + Delta_theta(q_j).
   *
   * @param q: Generalized coordinates [pos_base(3), rpy_base(3), q_joints(n_j)]
   */
  vector3_t computeAcomOrientation(const vector_t& q) const;

  /**
   * Computes the full whole-body aCOM Jacobian:
   * J_aCOM(q) = [0_(3x3), I_(3x3), J_Delta_theta_(3xn_j)].
   */
  matrix_t computeAcomJacobian(const vector_t& q) const;

  size_t getInputDim() const { return inputDim_; }
  size_t getHiddenDim() const { return hiddenDim_; }
  size_t getNumLayers() const { return numLayers_; }
  double getOmega0() const { return omega0_; }

 private:
  size_t inputDim_;
  size_t hiddenDim_;
  size_t numLayers_;
  double omega0_;
  std::vector<SirenLayerWeights> layers_;
};

}  // namespace ocs2::humanoid
