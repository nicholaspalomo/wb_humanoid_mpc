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

#include "humanoid_common_mpc/acom/AngularCenterOfMass.h"
#include "humanoid_common_mpc/acom/AcomSirenWeightsAtlas.h"
#include "humanoid_common_mpc/acom/AcomSirenWeightsG1.h"

#include <stdexcept>

namespace ocs2::humanoid {

AngularCenterOfMass::AngularCenterOfMass(size_t inputDim, size_t hiddenDim, size_t numLayers, double omega0)
    : inputDim_(inputDim), hiddenDim_(hiddenDim), numLayers_(numLayers), omega0_(omega0) {}

template <typename WeightsT>
std::unique_ptr<AngularCenterOfMass> AngularCenterOfMass::createFromStaticWeights() {
  static_assert(WeightsT::num_layers == 3,
                "createFromStaticWeights() hardcodes 3 layers (2 hidden + 1 output). "
                "Update the loader if the SIREN architecture changes.");
  auto acom = std::make_unique<AngularCenterOfMass>(WeightsT::input_dim, WeightsT::W0_rows, WeightsT::num_layers - 1, WeightsT::omega_0);

  std::vector<SirenLayerWeights> layers;

  // Layer 0
  SirenLayerWeights l0;
  l0.weight = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(WeightsT::W0, WeightsT::W0_rows,
                                                                                                       WeightsT::W0_cols);
  l0.bias = Eigen::Map<const Eigen::VectorXd>(WeightsT::b0, WeightsT::W0_rows);
  layers.push_back(l0);

  // Layer 1
  SirenLayerWeights l1;
  l1.weight = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(WeightsT::W1, WeightsT::W1_rows,
                                                                                                       WeightsT::W1_cols);
  l1.bias = Eigen::Map<const Eigen::VectorXd>(WeightsT::b1, WeightsT::W1_rows);
  layers.push_back(l1);

  // Layer 2
  SirenLayerWeights l2;
  l2.weight = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(WeightsT::W2, WeightsT::W2_rows,
                                                                                                       WeightsT::W2_cols);
  l2.bias = Eigen::Map<const Eigen::VectorXd>(WeightsT::b2, WeightsT::W2_rows);
  layers.push_back(l2);

  acom->setWeights(layers);
  return acom;
}

// LINT.IfChange(acom_robot_dispatch)
std::unique_ptr<AngularCenterOfMass> AngularCenterOfMass::createForRobot(const std::string& robotName) {
  using namespace ocs2::humanoid::acom;
  if (robotName == "atlas") {
    return createFromStaticWeights<AcomSirenWeightsAtlas>();
  } else if (robotName == "g1") {
    return createFromStaticWeights<AcomSirenWeightsG1>();
  }
  throw std::runtime_error("AngularCenterOfMass::createForRobot: Unknown robot '" + robotName +
                           "'. Add a new AcomSirenWeights<Robot>.h header and update this dispatch.");
}
// LINT.ThenChange(//humanoid_learning/acom/train_main.py:robot_paths)

void AngularCenterOfMass::setWeights(const std::vector<SirenLayerWeights>& layers) {
  if (layers.size() != numLayers_ + 1) {
    throw std::runtime_error("AngularCenterOfMass::setWeights: Expected " + std::to_string(numLayers_ + 1) + " layers, got " +
                             std::to_string(layers.size()));
  }
  // Validate dimension consistency between consecutive layers
  for (size_t i = 0; i < layers.size(); ++i) {
    size_t expectedIn = (i == 0) ? inputDim_ : static_cast<size_t>(layers[i - 1].weight.rows());
    if (static_cast<size_t>(layers[i].weight.cols()) != expectedIn) {
      throw std::runtime_error("AngularCenterOfMass::setWeights: Layer " + std::to_string(i) + " weight has " +
                               std::to_string(layers[i].weight.cols()) + " columns, expected " + std::to_string(expectedIn));
    }
    if (layers[i].bias.size() != layers[i].weight.rows()) {
      throw std::runtime_error("AngularCenterOfMass::setWeights: Layer " + std::to_string(i) + " bias has " +
                               std::to_string(layers[i].bias.size()) + " elements, expected " + std::to_string(layers[i].weight.rows()));
    }
  }
  layers_ = layers;
}

vector3_t AngularCenterOfMass::computeJointOrientationOffset(const vector_t& qJoints) const {
  if (layers_.empty()) {
    return vector3_t::Zero();
  }

  vector_t x = qJoints;

  // Hidden sinusoidal layers: x = sin(omega_0 * (W * x + b))
  for (size_t i = 0; i < numLayers_; ++i) {
    vector_t affine = omega0_ * (layers_[i].weight * x + layers_[i].bias);
    x = affine.array().sin().matrix();
  }

  // Output linear layer: y = W_out * x + b_out
  const auto& outLayer = layers_[numLayers_];
  vector3_t out = outLayer.weight * x + outLayer.bias;
  return out;
}

matrix_t AngularCenterOfMass::computeJointOffsetJacobian(const vector_t& qJoints) const {
  if (layers_.empty()) {
    return matrix_t::Zero(3, inputDim_);
  }

  // Forward activation tracking + chain rule Jacobians
  vector_t x = qJoints;
  matrix_t J = matrix_t::Identity(inputDim_, inputDim_);

  for (size_t i = 0; i < numLayers_; ++i) {
    vector_t affine = omega0_ * (layers_[i].weight * x + layers_[i].bias);
    vector_t cos_affine = affine.array().cos().matrix();

    // d(h_i)/d(h_{i-1}) = omega_0 * diag(cos(affine)) * W_i
    matrix_t d_layer = (omega0_ * cos_affine.asDiagonal()) * layers_[i].weight;

    J = d_layer * J;
    x = affine.array().sin().matrix();
  }

  // Output layer derivative: J = W_out * J_{L-1}
  const auto& outLayer = layers_[numLayers_];
  matrix_t J_out = outLayer.weight * J;
  return J_out;
}

vector3_t AngularCenterOfMass::computeAcomOrientation(const vector_t& q) const {
  // q = [pos_base (3), rpy_base (3), q_joints (n_j)]
  vector3_t rpyBase = q.segment<3>(3);
  vector_t qJoints = q.tail(q.size() - 6);
  return rpyBase + computeJointOrientationOffset(qJoints);
}

matrix_t AngularCenterOfMass::computeAcomJacobian(const vector_t& q) const {
  size_t nJoints = q.size() - 6;
  vector_t qJoints = q.tail(nJoints);
  matrix_t J_delta = computeJointOffsetJacobian(qJoints);

  matrix_t J_acom = matrix_t::Zero(3, 6 + nJoints);
  // Base linear velocity columns are zero
  // Base angular velocity columns are Identity (3x3)
  J_acom.block<3, 3>(0, 3) = matrix_t::Identity(3, 3);
  // Joint velocity columns are J_delta
  J_acom.block(0, 6, 3, nJoints) = J_delta;

  return J_acom;
}

}  // namespace ocs2::humanoid
