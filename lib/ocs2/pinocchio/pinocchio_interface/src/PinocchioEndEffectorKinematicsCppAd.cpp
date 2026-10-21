/******************************************************************************
Copyright (c) 2021, Farbod Farshidian. All rights reserved.

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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"

#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_robotic_tools/common/RotationTransforms.h>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

namespace ocs2 {

namespace {

/*
 * The taped functions of the end-effector frames `frameIds` of `pinocchioInterfaceCppAd` (whose data they overwrite) and
 * of the state and input of `mapping`, one block of rows per frame in the order of `frameIds`.
 */

ad_vector_t getPositionCppAd(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                             const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                             const std::vector<size_t>& frameIds,
                             const ad_vector_t& state) {
  const auto& model = pinocchioInterfaceCppAd.getModel();
  auto& data = pinocchioInterfaceCppAd.getData();
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t positions(kPositionDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    positions.segment<kPositionDim>(kPositionDim * i) = data.oMf[frameId].translation();
  }
  return positions;
}

ad_vector_t getVelocityCppAd(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                             const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                             const std::vector<size_t>& frameIds,
                             const ad_vector_t& state,
                             const ad_vector_t& input) {
  const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  const auto& model = pinocchioInterfaceCppAd.getModel();
  auto& data = pinocchioInterfaceCppAd.getData();
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);
  const ad_vector_t v = mapping.getPinocchioJointVelocity(state, input);

  pinocchio::forwardKinematics(model, data, q, v);

  ad_vector_t velocities(kPositionDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    velocities.segment<kPositionDim>(kPositionDim * i) = pinocchio::getFrameVelocity(model, data, frameId, rf).linear();
  }
  return velocities;
}

ad_vector_t getOrientationCppAd(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                                const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                                const std::vector<size_t>& frameIds,
                                const ad_vector_t& state) {
  const auto& model = pinocchioInterfaceCppAd.getModel();
  auto& data = pinocchioInterfaceCppAd.getData();
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t orientations(kQuaternionCoefficients * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    orientations.segment<kQuaternionCoefficients>(kQuaternionCoefficients * i) =
        matrixToQuaternion(data.oMf[frameIds[i]].rotation()).coeffs();
  }
  return orientations;
}

ad_vector_t getOrientationErrorWrtPlaneCppAd(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                                             const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                                             const std::vector<size_t>& frameIds,
                                             const ad_vector_t& state,
                                             const ad_vector_t& params) {
  const auto& model = pinocchioInterfaceCppAd.getModel();
  auto& data = pinocchioInterfaceCppAd.getData();
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t errors(kRotationTangentDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    ad_vector_t planeNormal = params.segment<kPositionDim>(kPositionDim * i);
    errors.segment<kRotationTangentDim>(kRotationTangentDim * i) =
        rotationMatrixDistanceToPlane<ad_scalar_t>(data.oMf[frameId].rotation(), planeNormal);
  }
  return errors;
}

ad_vector_t getAngularVelocityCppAd(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                                    const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                                    const std::vector<size_t>& frameIds,
                                    const ad_vector_t& state,
                                    const ad_vector_t& input) {
  const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  const auto& model = pinocchioInterfaceCppAd.getModel();
  auto& data = pinocchioInterfaceCppAd.getData();
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);
  const ad_vector_t v = mapping.getPinocchioJointVelocity(state, input);

  pinocchio::forwardKinematics(model, data, q, v);

  ad_vector_t angularVelocities(kRotationTangentDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    angularVelocities.segment<kRotationTangentDim>(kRotationTangentDim * i) =
        pinocchio::getFrameVelocity(model, data, frameId, rf).angular();
  }
  return angularVelocities;
}

ad_vector_t getTwistCppAd(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                          const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                          const std::vector<size_t>& frameIds,
                          const ad_vector_t& state,
                          const ad_vector_t& input) {
  const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  const auto& model = pinocchioInterfaceCppAd.getModel();
  auto& data = pinocchioInterfaceCppAd.getData();
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);
  const ad_vector_t v = mapping.getPinocchioJointVelocity(state, input);

  pinocchio::forwardKinematics(model, data, q, v);

  ad_vector_t twists(kTwistDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    auto motion = pinocchio::getFrameVelocity(model, data, frameId, rf);
    ad_vector_t currTwist(kTwistDim);
    currTwist.head(kPositionDim) = motion.linear();
    currTwist.tail(kRotationTangentDim) = motion.angular();
    twists.segment<kTwistDim>(kTwistDim * i) = currTwist;
  }
  return twists;
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PinocchioEndEffectorKinematicsCppAd::PinocchioEndEffectorKinematicsCppAd(const PinocchioInterface& pinocchioInterface,
                                                                         const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                                                                         std::vector<std::string> endEffectorIds,
                                                                         size_t stateDim,
                                                                         size_t inputDim,
                                                                         update_pinocchio_interface_callback updateCallback,
                                                                         const std::string& modelName,
                                                                         const std::string& modelFolder,
                                                                         bool recompileLibraries,
                                                                         bool verbose)
    : endEffectorIds_(std::move(endEffectorIds)) {
  // The functions capture by value what they read (the class comment). Every copy of a CppAdInterface holds a copy of its
  // function, so the Pinocchio interface on AD scalars, which the functions overwrite, and the mapping set to it are held
  // by std::shared_ptr: shared by the functions of this object and of every copy, and alive as long as any of them is.
  std::vector<size_t> frameIds;
  frameIds.reserve(endEffectorIds_.size());
  for (const std::string& bodyName : endEffectorIds_) {
    frameIds.push_back(pinocchioInterface.getModel().getFrameId(bodyName));
  }
  const std::shared_ptr<PinocchioInterfaceCppAd> pinocchioInterfaceCppAd =
      std::make_shared<PinocchioInterfaceCppAd>(pinocchioInterface.toCppAd());
  const std::shared_ptr<PinocchioStateInputMapping<ad_scalar_t>> mappingPtr(mapping.clone());
  mappingPtr->setPinocchioInterface(*pinocchioInterfaceCppAd);

  // position function
  const CppAdInterface::ad_function_t positionFunc = [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback](const ad_vector_t& x,
                                                                                                                     ad_vector_t& y) {
    updateCallback(x, *pinocchioInterfaceCppAd);
    y = getPositionCppAd(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, x);
  };
  positionCppAdInterfacePtr_.reset(new CppAdInterface(positionFunc, stateDim, modelName + "_position", modelFolder));

  // velocity function
  const CppAdInterface::ad_function_t velocityFunc = [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback, stateDim, inputDim](
                                                         const ad_vector_t& x, ad_vector_t& y) {
    const ad_vector_t state = x.head(stateDim);
    const ad_vector_t input = x.tail(inputDim);
    updateCallback(state, *pinocchioInterfaceCppAd);
    y = getVelocityCppAd(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, state, input);
  };
  velocityCppAdInterfacePtr_.reset(new CppAdInterface(velocityFunc, stateDim + inputDim, modelName + "_velocity", modelFolder));

  // orientation function
  const CppAdInterface::ad_function_t orientationFunc = [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback](
                                                            const ad_vector_t& x, ad_vector_t& y) {
    updateCallback(x, *pinocchioInterfaceCppAd);
    y = getOrientationCppAd(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, x);
  };
  orientationCppAdInterfacePtr_.reset(new CppAdInterface(orientationFunc, stateDim, modelName + "_orientation", modelFolder));

  // orientation w.r.t plane function
  const CppAdInterface::ad_parameterized_function_t orientationWrtPlaneFunc =
      [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback](const ad_vector_t& x, const ad_vector_t& params, ad_vector_t& y) {
        updateCallback(x, *pinocchioInterfaceCppAd);
        y = getOrientationErrorWrtPlaneCppAd(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, x, params);
      };
  orientationErrorWrtPlaneCppAdInterfacePtr_.reset(new CppAdInterface(orientationWrtPlaneFunc, stateDim, kPositionDim * frameIds.size(),
                                                                      modelName + "_orientation_wrt_plane", modelFolder));

  // velocity function
  const CppAdInterface::ad_function_t angularVelocityFunc = [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback, stateDim,
                                                             inputDim](const ad_vector_t& x, ad_vector_t& y) {
    const ad_vector_t state = x.head(stateDim);
    const ad_vector_t input = x.tail(inputDim);
    updateCallback(state, *pinocchioInterfaceCppAd);
    y = getAngularVelocityCppAd(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, state, input);
  };
  angularVelocityCppAdInterfacePtr_.reset(
      new CppAdInterface(angularVelocityFunc, stateDim + inputDim, modelName + "_angular_velocity", modelFolder));

  // twist function
  const CppAdInterface::ad_function_t twistFunc = [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback, stateDim, inputDim](
                                                      const ad_vector_t& x, ad_vector_t& y) {
    const ad_vector_t state = x.head(stateDim);
    const ad_vector_t input = x.tail(inputDim);
    updateCallback(state, *pinocchioInterfaceCppAd);
    y = getTwistCppAd(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, state, input);
  };
  twistCppAdInterfacePtr_.reset(new CppAdInterface(twistFunc, stateDim + inputDim, modelName + "_twist", modelFolder));

  if (recompileLibraries) {
    positionCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
    velocityCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
    orientationCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::Zero, verbose);
    orientationErrorWrtPlaneCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
    angularVelocityCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
    twistCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
  } else {
    positionCppAdInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose);
    velocityCppAdInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose);
    orientationCppAdInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::Zero, verbose);
    orientationErrorWrtPlaneCppAdInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose);
    angularVelocityCppAdInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose);
    twistCppAdInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PinocchioEndEffectorKinematicsCppAd::PinocchioEndEffectorKinematicsCppAd(const PinocchioEndEffectorKinematicsCppAd& rhs)
    : EndEffectorKinematics<scalar_t>(rhs),
      positionCppAdInterfacePtr_(new CppAdInterface(*rhs.positionCppAdInterfacePtr_)),
      velocityCppAdInterfacePtr_(new CppAdInterface(*rhs.velocityCppAdInterfacePtr_)),
      orientationCppAdInterfacePtr_(new CppAdInterface(*rhs.orientationCppAdInterfacePtr_)),
      orientationErrorWrtPlaneCppAdInterfacePtr_(new CppAdInterface(*rhs.orientationErrorWrtPlaneCppAdInterfacePtr_)),
      angularVelocityCppAdInterfacePtr_(new CppAdInterface(*rhs.angularVelocityCppAdInterfacePtr_)),
      twistCppAdInterfacePtr_(new CppAdInterface(*rhs.twistCppAdInterfacePtr_)),
      endEffectorIds_(rhs.endEffectorIds_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PinocchioEndEffectorKinematicsCppAd* absl_nonnull PinocchioEndEffectorKinematicsCppAd::clone() const {
  return new PinocchioEndEffectorKinematicsCppAd(*this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
const std::vector<std::string>& PinocchioEndEffectorKinematicsCppAd::getIds() const {
  return endEffectorIds_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getPosition(const vector_t& state) const -> std::vector<vector3_t> {
  const vector_t positionValues = positionCppAdInterfacePtr_->getFunctionValue(state);

  std::vector<vector3_t> positions;
  positions.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    positions.emplace_back(positionValues.segment<kPositionDim>(kPositionDim * i));
  }
  return positions;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation> PinocchioEndEffectorKinematicsCppAd::getPositionLinearApproximation(
    const vector_t& state) const {
  const vector_t positionValues = positionCppAdInterfacePtr_->getFunctionValue(state);
  const matrix_t positionJacobian = positionCppAdInterfacePtr_->getJacobian(state);

  std::vector<VectorFunctionLinearApproximation> positions;
  positions.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    VectorFunctionLinearApproximation pos;
    pos.f = positionValues.segment<kPositionDim>(kPositionDim * i);
    pos.dfdx = positionJacobian.block(kPositionDim * i, 0, kPositionDim, state.rows());
    positions.emplace_back(std::move(pos));
  }
  return positions;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getVelocity(const vector_t& state, const vector_t& input) const -> std::vector<vector3_t> {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t velocityValues = velocityCppAdInterfacePtr_->getFunctionValue(stateInput);

  std::vector<vector3_t> velocities;
  velocities.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    velocities.emplace_back(velocityValues.segment<kPositionDim>(kPositionDim * i));
  }
  return velocities;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation> PinocchioEndEffectorKinematicsCppAd::getVelocityLinearApproximation(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t velocityValues = velocityCppAdInterfacePtr_->getFunctionValue(stateInput);
  const matrix_t velocityJacobian = velocityCppAdInterfacePtr_->getJacobian(stateInput);

  std::vector<VectorFunctionLinearApproximation> velocities;
  velocities.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    VectorFunctionLinearApproximation vel;
    vel.f = velocityValues.segment<kPositionDim>(kPositionDim * i);
    vel.dfdx = velocityJacobian.block(kPositionDim * i, 0, kPositionDim, state.rows());
    vel.dfdu = velocityJacobian.block(kPositionDim * i, state.rows(), kPositionDim, input.rows());
    velocities.emplace_back(std::move(vel));
  }
  return velocities;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getOrientation(const vector_t& state) const -> std::vector<quaternion_t> {
  const vector_t orientationValues = orientationCppAdInterfacePtr_->getFunctionValue(state);

  std::vector<quaternion_t> orientations;
  orientations.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    orientations.emplace_back(quaternion_t(orientationValues.segment<kQuaternionCoefficients>(kQuaternionCoefficients * i)));
  }
  return orientations;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getOrientationErrorWrtPlane(const vector_t& state,
                                                                      const std::vector<vector3_t>& planeNormals) const
    -> std::vector<vector3_t> {
  vector_t params(kPositionDim * endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    params.segment<kPositionDim>(kPositionDim * i) = planeNormals[i];
  }

  const vector_t errorValues = orientationErrorWrtPlaneCppAdInterfacePtr_->getFunctionValue(state, params);

  std::vector<vector3_t> errors;
  errors.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    errors.emplace_back(errorValues.segment<kRotationTangentDim>(kRotationTangentDim * i));
  }
  return errors;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<VectorFunctionLinearApproximation> PinocchioEndEffectorKinematicsCppAd::getOrientationErrorWrtPlaneLinearApproximation(
    const vector_t& state, const std::vector<vector3_t>& planeNormals) const {
  vector_t params(kPositionDim * endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    params.segment<kPositionDim>(kPositionDim * i) = planeNormals[i];
  }
  const vector_t errorValues = orientationErrorWrtPlaneCppAdInterfacePtr_->getFunctionValue(state, params);
  const matrix_t errorJacobian = orientationErrorWrtPlaneCppAdInterfacePtr_->getJacobian(state, params);

  std::vector<VectorFunctionLinearApproximation> errors;
  errors.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    VectorFunctionLinearApproximation err;
    err.f = errorValues.segment<kRotationTangentDim>(kRotationTangentDim * i);
    err.dfdx = errorJacobian.block(kRotationTangentDim * i, 0, kRotationTangentDim, state.rows());
    errors.emplace_back(std::move(err));
  }
  return errors;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getAngularVelocity(const vector_t& state, const vector_t& input) const -> std::vector<vector3_t> {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t velocityValues = angularVelocityCppAdInterfacePtr_->getFunctionValue(stateInput);

  std::vector<vector3_t> velocities;
  velocities.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    velocities.emplace_back(velocityValues.segment<kRotationTangentDim>(kRotationTangentDim * i));
  }
  return velocities;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation> PinocchioEndEffectorKinematicsCppAd::getAngularVelocityLinearApproximation(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t velocityValues = angularVelocityCppAdInterfacePtr_->getFunctionValue(stateInput);
  const matrix_t velocityJacobian = angularVelocityCppAdInterfacePtr_->getJacobian(stateInput);

  std::vector<VectorFunctionLinearApproximation> velocities;
  velocities.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    VectorFunctionLinearApproximation vel;
    vel.f = velocityValues.segment<kRotationTangentDim>(kRotationTangentDim * i);
    vel.dfdx = velocityJacobian.block(kRotationTangentDim * i, 0, kRotationTangentDim, state.rows());
    vel.dfdu = velocityJacobian.block(kRotationTangentDim * i, state.rows(), kRotationTangentDim, input.rows());
    velocities.emplace_back(std::move(vel));
  }
  return velocities;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getTwist(const vector_t& state, const vector_t& input) const -> std::vector<vector6_t> {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t velocityValues = twistCppAdInterfacePtr_->getFunctionValue(stateInput);

  std::vector<vector6_t> velocities;
  velocities.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    velocities.emplace_back(velocityValues.segment<kTwistDim>(kTwistDim * i));
  }
  return velocities;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation> PinocchioEndEffectorKinematicsCppAd::getTwistLinearApproximation(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t velocityValues = twistCppAdInterfacePtr_->getFunctionValue(stateInput);
  const matrix_t velocityJacobian = twistCppAdInterfacePtr_->getJacobian(stateInput);

  std::vector<VectorFunctionLinearApproximation> velocities;
  velocities.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    VectorFunctionLinearApproximation vel;
    vel.f = velocityValues.segment<kTwistDim>(kTwistDim * i);
    vel.dfdx = velocityJacobian.block(kTwistDim * i, 0, kTwistDim, state.rows());
    vel.dfdu = velocityJacobian.block(kTwistDim * i, state.rows(), kTwistDim, input.rows());
    velocities.emplace_back(std::move(vel));
  }
  return velocities;
}

}  // namespace ocs2
