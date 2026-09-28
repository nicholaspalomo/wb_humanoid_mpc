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

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/acom/AcomSirenWeightsAtlas.h"
#include "humanoid_common_mpc/acom/AcomSirenWeightsG1.h"
#include "humanoid_common_mpc/acom/AcomSirenWeightsSa01.h"
#include "humanoid_common_mpc/common/StatusMacros.h"

namespace ocs2::humanoid {

namespace {

/// Number of generalized coordinates the floating base occupies in the Pinocchio
/// model: 3 for position and 3 for the ZYX Euler angles.
constexpr Eigen::Index kGeneralizedBaseDim = 6;

/// Index of the first base orientation coordinate within q.
constexpr Eigen::Index kBaseOrientationOffset = 3;

using RowMajorMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/** One layer of a generated weight header, copied out of its row-major C arrays. */
SirenLayerWeights mapLayer(const double* weight, std::size_t rows, std::size_t cols, const double* bias) {
  SirenLayerWeights layer;
  layer.weight = Eigen::Map<const RowMajorMatrix>(weight, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
  layer.bias = Eigen::Map<const Eigen::VectorXd>(bias, static_cast<Eigen::Index>(rows));
  return layer;
}

/**
 * Builds an evaluator from a generated weight header (e.g. AcomSirenWeightsAtlas), including the joint names the
 * network was trained on.
 *
 * @tparam WeightsT  Auto-generated weight struct with static constexpr members.
 */
template <typename WeightsT>
absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> createFromStaticWeights() {
  // LINT.IfChange(acom_layer_count)
  static_assert(WeightsT::num_layers == 3,
                "createFromStaticWeights() hardcodes 3 layers (2 sinusoidal + 1 linear readout). "
                "Retrain with --num_layers 2, or extend this loader if the SIREN architecture changes.");
  // LINT.ThenChange(//humanoid_learning/acom/train_main.py:siren_num_layers)
  static_assert(WeightsT::output_dim == 3,
                "The aCOM orientation offset is a 3-vector. A header with a different "
                "output_dim would overflow the fixed-size vector3_t return value.");
  auto acom = std::make_unique<AngularCenterOfMass>(WeightsT::input_dim, WeightsT::num_layers - 1, WeightsT::omega_0);

  const std::vector<SirenLayerWeights> layers = {
      mapLayer(WeightsT::W0, WeightsT::W0_rows, WeightsT::W0_cols, WeightsT::b0),
      mapLayer(WeightsT::W1, WeightsT::W1_rows, WeightsT::W1_cols, WeightsT::b1),
      mapLayer(WeightsT::W2, WeightsT::W2_rows, WeightsT::W2_cols, WeightsT::b2),
  };
  RETURN_IF_ERROR(acom->loadWeights(layers));
  RETURN_IF_ERROR(acom->setJointNames(std::vector<std::string>(std::begin(WeightsT::joint_names), std::end(WeightsT::joint_names))));
  return acom;
}

using AcomFactory = absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> (*)();

/** One compiled-in network, and the names it is known by on either side of the training pipeline. */
struct RegisteredAcomNetwork {
  /// model_settings.robotName of the robots that run this network.
  absl::string_view robotName;
  /// The --robot key of humanoid_learning/acom/train_main.py that trains it.
  absl::string_view trainMainRobot;
  /// The generated header the weights live in.
  absl::string_view headerName;
  AcomFactory create;
};

// The registry. Adding a robot means training its header with train_main.py and adding one line here; the acceptance
// test then refuses to pass until the robot also has a case of its own, see testAcomAngularVelocityConsistency.cpp.
// LINT.IfChange(acom_robot_dispatch)
constexpr RegisteredAcomNetwork kRegisteredAcomNetworks[] = {
    {"atlas", "atlas", "AcomSirenWeightsAtlas.h", &createFromStaticWeights<acom::AcomSirenWeightsAtlas>},
    {"engineai_sa01", "sa01", "AcomSirenWeightsSa01.h", &createFromStaticWeights<acom::AcomSirenWeightsSa01>},
    {"g1", "g1", "AcomSirenWeightsG1.h", &createFromStaticWeights<acom::AcomSirenWeightsG1>},
};
// clang-format off
// LINT.ThenChange(//humanoid_learning/acom/train_main.py:robot_paths, //humanoid_nmpc/humanoid_common_mpc/test/testAcomAngularVelocityConsistency.cpp:acom_acceptance_robots, //robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:acom_robot_name, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:acom_robot_name, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:acom_robot_name)
// clang-format on

const RegisteredAcomNetwork* findRegisteredNetwork(absl::string_view robotName) {
  for (const RegisteredAcomNetwork& network : kRegisteredAcomNetworks) {
    if (network.robotName == robotName) {
      return &network;
    }
  }
  return nullptr;
}

absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> createRegisteredNetwork(absl::string_view robotName) {
  const RegisteredAcomNetwork* network = findRegisteredNetwork(robotName);
  if (network == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("No Angular Center of Mass (ACoM) network is registered for model_settings.robotName '", robotName,
                     "'; the registered robots are: ", absl::StrJoin(AngularCenterOfMass::registeredRobotNames(), ", "),
                     ". Leave com_and_acom_tracking_cost out of task.yaml's costs list and heading_double_integrator out of "
                     "contact_planning.dynamics in contact_planning.yaml, or train a network with `bazel run "
                     "//humanoid_learning/acom:train_main` and register it in AngularCenterOfMass.cpp."));
  }
  return network->create();
}

/** The retraining instruction every joint-order error ends with. */
std::string retrainAdvice(const RegisteredAcomNetwork& network) {
  return absl::StrCat(
      "The MPC model's joints are the URDF's minus model_settings.fixedJointNames, so either restore fixedJointNames to the set ",
      network.headerName, " was trained without (its joint_names[] lists the joints it expects, in order), or retrain against the ",
      "current task.yaml with `bazel run //humanoid_learning/acom:train_main -- --robot ", network.trainMainRobot, " --install_header`.");
}

}  // namespace

AngularCenterOfMass::AngularCenterOfMass(std::size_t inputDim, std::size_t numLayers, double omega0)
    : inputDim_(inputDim), numLayers_(numLayers), omega0_(omega0) {}

std::vector<std::string> AngularCenterOfMass::registeredRobotNames() {
  std::vector<std::string> names;
  for (const RegisteredAcomNetwork& network : kRegisteredAcomNetworks) {
    names.emplace_back(network.robotName);
  }
  std::sort(names.begin(), names.end());
  return names;
}

absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> AngularCenterOfMass::Create(absl::string_view robotName,
                                                                                 const std::vector<std::string>& modelJointNames) {
  ASSIGN_OR_RETURN(std::unique_ptr<AngularCenterOfMass> acom, createRegisteredNetwork(robotName));
  const RegisteredAcomNetwork& network = *findRegisteredNetwork(robotName);

  const std::vector<std::string>& trainedJointNames = acom->getJointNames();
  if (trainedJointNames.size() != modelJointNames.size()) {
    return absl::FailedPreconditionError(
        absl::StrCat("The Angular Center of Mass network for model_settings.robotName '", robotName, "' (", network.headerName, ") takes ",
                     trainedJointNames.size(), " joints but the MPC model has ", modelJointNames.size(), ". ", retrainAdvice(network)));
  }
  for (std::size_t joint = 0; joint < modelJointNames.size(); ++joint) {
    if (trainedJointNames[joint] != modelJointNames[joint]) {
      return absl::FailedPreconditionError(absl::StrCat("The Angular Center of Mass network for model_settings.robotName '", robotName,
                                                        "' was trained on a different joint vector than the MPC model: joint ", joint,
                                                        " is '", trainedJointNames[joint], "' in ", network.headerName, " but '",
                                                        modelJointNames[joint], "' in the MPC model. ", retrainAdvice(network)));
    }
  }
  return acom;
}

absl::Status AngularCenterOfMass::loadWeights(const std::vector<SirenLayerWeights>& layers) {
  if (layers.size() != numLayers_ + 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("AngularCenterOfMass::loadWeights: expected ", numLayers_ + 1, " layers, got ", layers.size(), "."));
  }
  // Validate dimension consistency between consecutive layers.
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const std::size_t expectedIn = (i == 0) ? inputDim_ : static_cast<std::size_t>(layers[i - 1].weight.rows());
    if (static_cast<std::size_t>(layers[i].weight.cols()) != expectedIn) {
      return absl::InvalidArgumentError(absl::StrCat("AngularCenterOfMass::loadWeights: layer ", i, " weight has ", layers[i].weight.cols(),
                                                     " columns, expected ", expectedIn, "."));
    }
    if (layers[i].bias.size() != layers[i].weight.rows()) {
      return absl::InvalidArgumentError(absl::StrCat("AngularCenterOfMass::loadWeights: layer ", i, " bias has ", layers[i].bias.size(),
                                                     " elements, expected ", layers[i].weight.rows(), "."));
    }
  }
  // The orientation offset is returned as a fixed-size vector3_t, so a readout of
  // any other width would overflow it.
  if (layers.back().weight.rows() != 3) {
    return absl::InvalidArgumentError(
        absl::StrCat("AngularCenterOfMass::loadWeights: readout layer has ", layers.back().weight.rows(), " rows, expected 3."));
  }
  layers_ = layers;
  return absl::OkStatus();
}

void AngularCenterOfMass::setWeights(const std::vector<SirenLayerWeights>& layers) {
  const absl::Status status = loadWeights(layers);
  if (!status.ok()) {
    throw std::runtime_error(std::string(status.message()));
  }
}

absl::Status AngularCenterOfMass::setJointNames(std::vector<std::string> jointNames) {
  if (jointNames.size() != inputDim_) {
    return absl::InvalidArgumentError(absl::StrCat("AngularCenterOfMass::setJointNames: the network takes ", inputDim_,
                                                   " joint positions but ", jointNames.size(), " joint names were given."));
  }
  jointNames_ = std::move(jointNames);
  return absl::OkStatus();
}

void AngularCenterOfMass::checkWeightsLoaded() const {
  if (layers_.empty()) {
    throw std::runtime_error("AngularCenterOfMass: no weights loaded. Construct via Create(), or call loadWeights() before evaluating.");
  }
}

void AngularCenterOfMass::checkJointVectorSize(const vector_t& qJoints) const {
  if (static_cast<std::size_t>(qJoints.size()) != inputDim_) {
    throw std::runtime_error(absl::StrCat("AngularCenterOfMass: expected ", inputDim_, " joint positions, got ", qJoints.size(),
                                          ". The SIREN weights were trained for a different robot model than the one in use."));
  }
}

vector3_t AngularCenterOfMass::computeJointOrientationOffset(const vector_t& qJoints) const {
  checkWeightsLoaded();
  checkJointVectorSize(qJoints);

  vector_t x = qJoints;

  // Hidden sinusoidal layers: x = sin(omega_0 * (W * x + b))
  for (std::size_t i = 0; i < numLayers_; ++i) {
    vector_t affine = omega0_ * (layers_[i].weight * x + layers_[i].bias);
    x = affine.array().sin().matrix();
  }

  // Output linear layer: y = W_out * x + b_out
  const SirenLayerWeights& outLayer = layers_[numLayers_];
  vector3_t out = outLayer.weight * x + outLayer.bias;
  return out;
}

matrix_t AngularCenterOfMass::computeJointOffsetJacobian(const vector_t& qJoints) const {
  checkWeightsLoaded();
  checkJointVectorSize(qJoints);

  // Forward activation tracking + chain rule Jacobians
  vector_t x = qJoints;
  matrix_t J = matrix_t::Identity(inputDim_, inputDim_);

  for (std::size_t i = 0; i < numLayers_; ++i) {
    vector_t affine = omega0_ * (layers_[i].weight * x + layers_[i].bias);
    vector_t cos_affine = affine.array().cos().matrix();

    // d(h_i)/d(h_{i-1}) = omega_0 * diag(cos(affine)) * W_i
    matrix_t d_layer = (omega0_ * cos_affine.asDiagonal()) * layers_[i].weight;

    J = d_layer * J;
    x = affine.array().sin().matrix();
  }

  // Output layer derivative: J = W_out * J_{L-1}
  const SirenLayerWeights& outLayer = layers_[numLayers_];
  matrix_t J_out = outLayer.weight * J;
  return J_out;
}

vector3_t AngularCenterOfMass::computeAcomOrientation(const vector_t& q) const {
  // q = [pos_base(3), euler_zyx_base(3), q_joints(n_j)]
  const vector3_t eulerZyxBase = q.segment<3>(kBaseOrientationOffset);
  const vector_t qJoints = q.tail(q.size() - kGeneralizedBaseDim);
  return eulerZyxBase + acomXyzToZyx(computeJointOrientationOffset(qJoints));
}

matrix_t AngularCenterOfMass::computeAcomJacobian(const vector_t& q) const {
  const Eigen::Index nJoints = q.size() - kGeneralizedBaseDim;
  const matrix_t J_delta_zyx = acomJacobianXyzToZyx(computeJointOffsetJacobian(q.tail(nJoints)));

  matrix_t J_acom = matrix_t::Zero(3, kGeneralizedBaseDim + nJoints);
  // The base position columns are zero: translating the base does not rotate it.
  // The base orientation columns are the identity, since the aCOM orientation is
  // the base Euler triple plus a joint-only offset.
  J_acom.block<3, 3>(0, kBaseOrientationOffset).setIdentity();
  J_acom.block(0, kGeneralizedBaseDim, 3, nJoints) = J_delta_zyx;
  return J_acom;
}

}  // namespace ocs2::humanoid
