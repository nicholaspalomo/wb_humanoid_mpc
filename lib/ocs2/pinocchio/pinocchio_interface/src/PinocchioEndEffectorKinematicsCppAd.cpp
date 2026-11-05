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

ad_vector_t computePosEesRtWorldEwrtWorld(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                             const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                             const std::vector<size_t>& frameIds,
                             const ad_vector_t& state) {
  const auto& model = pinocchioInterfaceCppAd.getModel();
  auto& data = pinocchioInterfaceCppAd.getData();
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t posEesRtWorldEwrtWorld(kPositionDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    posEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i) = data.oMf[frameId].translation();
  }
  return posEesRtWorldEwrtWorld;
}

ad_vector_t computeLinvelEesRtWorldEwrtWorld(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
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

  ad_vector_t linvelEesRtWorldEwrtWorld(kPositionDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    linvelEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i) = pinocchio::getFrameVelocity(model, data, frameId, rf).linear();
  }
  return linvelEesRtWorldEwrtWorld;
}

ad_vector_t computeQuatEesRtWorldXyzw(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                                const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                                const std::vector<size_t>& frameIds,
                                const ad_vector_t& state) {
  const auto& model = pinocchioInterfaceCppAd.getModel();
  auto& data = pinocchioInterfaceCppAd.getData();
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t quatEesRtWorldXyzw(kQuaternionCoefficients * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    quatEesRtWorldXyzw.segment<kQuaternionCoefficients>(kQuaternionCoefficients * i) =
        matrixToQuaternion(data.oMf[frameIds[i]].rotation()).coeffs();
  }
  return quatEesRtWorldXyzw;
}

ad_vector_t computeTiltEesRtSurfaceEwrtWorld(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                                             const PinocchioStateInputMapping<ad_scalar_t>& mapping,
                                             const std::vector<size_t>& frameIds,
                                             const ad_vector_t& state,
                                             const ad_vector_t& stackedNormalSurfacesEwrtWorld) {
  const auto& model = pinocchioInterfaceCppAd.getModel();
  auto& data = pinocchioInterfaceCppAd.getData();
  const ad_vector_t q = mapping.getPinocchioJointPosition(state);

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t tiltEesRtSurfaceEwrtWorld(kRotationTangentDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    ad_vector_t normalSurfaceEwrtWorld = stackedNormalSurfacesEwrtWorld.segment<kPositionDim>(kPositionDim * i);
    tiltEesRtSurfaceEwrtWorld.segment<kRotationTangentDim>(kRotationTangentDim * i) =
        rotationMatrixDistanceToPlane<ad_scalar_t>(data.oMf[frameId].rotation(), normalSurfaceEwrtWorld);
  }
  return tiltEesRtSurfaceEwrtWorld;
}

ad_vector_t computeAngvelEesRtWorldEwrtWorld(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
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

  ad_vector_t angvelEesRtWorldEwrtWorld(kRotationTangentDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    angvelEesRtWorldEwrtWorld.segment<kRotationTangentDim>(kRotationTangentDim * i) =
        pinocchio::getFrameVelocity(model, data, frameId, rf).angular();
  }
  return angvelEesRtWorldEwrtWorld;
}

ad_vector_t computeTwistEesRtWorldEwrtWorld(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
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

  ad_vector_t twistEesRtWorldEwrtWorld(kTwistDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    auto pinocchioTwistEeRtWorldEwrtWorld = pinocchio::getFrameVelocity(model, data, frameId, rf);
    ad_vector_t twistEeRtWorldEwrtWorld(kTwistDim);
    twistEeRtWorldEwrtWorld.head(kPositionDim) = pinocchioTwistEeRtWorldEwrtWorld.linear();
    twistEeRtWorldEwrtWorld.tail(kRotationTangentDim) = pinocchioTwistEeRtWorldEwrtWorld.angular();
    twistEesRtWorldEwrtWorld.segment<kTwistDim>(kTwistDim * i) = twistEeRtWorldEwrtWorld;
  }
  return twistEesRtWorldEwrtWorld;
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
  const CppAdInterface::ad_function_t tapePosEesRtWorldEwrtWorld = [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback](
                                                                           const ad_vector_t& x, ad_vector_t& y) {
    updateCallback(x, *pinocchioInterfaceCppAd);
    y = computePosEesRtWorldEwrtWorld(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, x);
  };
  positionCppAdInterfacePtr_.reset(new CppAdInterface(tapePosEesRtWorldEwrtWorld, stateDim, modelName + "_position", modelFolder));

  // linear velocity function
  const CppAdInterface::ad_function_t tapeLinvelEesRtWorldEwrtWorld = [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback,
                                                                            stateDim, inputDim](const ad_vector_t& x, ad_vector_t& y) {
    const ad_vector_t state = x.head(stateDim);
    const ad_vector_t input = x.tail(inputDim);
    updateCallback(state, *pinocchioInterfaceCppAd);
    y = computeLinvelEesRtWorldEwrtWorld(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, state, input);
  };
  velocityCppAdInterfacePtr_.reset(new CppAdInterface(tapeLinvelEesRtWorldEwrtWorld, stateDim + inputDim, modelName + "_velocity", modelFolder));

  // orientation function
  const CppAdInterface::ad_function_t tapeQuatEesRtWorldXyzw = [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback](
                                                            const ad_vector_t& x, ad_vector_t& y) {
    updateCallback(x, *pinocchioInterfaceCppAd);
    y = computeQuatEesRtWorldXyzw(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, x);
  };
  orientationCppAdInterfacePtr_.reset(new CppAdInterface(tapeQuatEesRtWorldXyzw, stateDim, modelName + "_orientation", modelFolder));

  // tilt relative to the surface function (the library keeps its name _orientation_wrt_plane)
  const CppAdInterface::ad_parameterized_function_t tapeTiltEesRtSurfaceEwrtWorld =
      [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback](const ad_vector_t& x, const ad_vector_t& params, ad_vector_t& y) {
        updateCallback(x, *pinocchioInterfaceCppAd);
        y = computeTiltEesRtSurfaceEwrtWorld(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, x, params);
      };
  tiltEesCppAdInterfacePtr_.reset(new CppAdInterface(tapeTiltEesRtSurfaceEwrtWorld, stateDim, kPositionDim * frameIds.size(),
                                                                      modelName + "_orientation_wrt_plane", modelFolder));

  // angular velocity function
  const CppAdInterface::ad_function_t tapeAngvelEesRtWorldEwrtWorld = [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback, stateDim,
                                                             inputDim](const ad_vector_t& x, ad_vector_t& y) {
    const ad_vector_t state = x.head(stateDim);
    const ad_vector_t input = x.tail(inputDim);
    updateCallback(state, *pinocchioInterfaceCppAd);
    y = computeAngvelEesRtWorldEwrtWorld(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, state, input);
  };
  angularVelocityCppAdInterfacePtr_.reset(
      new CppAdInterface(tapeAngvelEesRtWorldEwrtWorld, stateDim + inputDim, modelName + "_angular_velocity", modelFolder));

  // twist function
  const CppAdInterface::ad_function_t tapeTwistEesRtWorldEwrtWorld = [pinocchioInterfaceCppAd, mappingPtr, frameIds, updateCallback, stateDim, inputDim](
                                                      const ad_vector_t& x, ad_vector_t& y) {
    const ad_vector_t state = x.head(stateDim);
    const ad_vector_t input = x.tail(inputDim);
    updateCallback(state, *pinocchioInterfaceCppAd);
    y = computeTwistEesRtWorldEwrtWorld(*pinocchioInterfaceCppAd, *mappingPtr, frameIds, state, input);
  };
  twistCppAdInterfacePtr_.reset(new CppAdInterface(tapeTwistEesRtWorldEwrtWorld, stateDim + inputDim, modelName + "_twist", modelFolder));

  if (recompileLibraries) {
    positionCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
    velocityCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
    orientationCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::Zero, verbose);
    tiltEesCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
    angularVelocityCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
    twistCppAdInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First, verbose);
  } else {
    positionCppAdInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose);
    velocityCppAdInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose);
    orientationCppAdInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::Zero, verbose);
    tiltEesCppAdInterfacePtr_->loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, verbose);
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
      tiltEesCppAdInterfacePtr_(new CppAdInterface(*rhs.tiltEesCppAdInterfacePtr_)),
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
auto PinocchioEndEffectorKinematicsCppAd::getPosEesRtWorldEwrtWorld(const vector_t& state) const -> std::vector<vector3_t> {
  const vector_t stackedPosEesRtWorldEwrtWorld = positionCppAdInterfacePtr_->getFunctionValue(state);

  std::vector<vector3_t> posEesRtWorldEwrtWorld;
  posEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    posEesRtWorldEwrtWorld.emplace_back(stackedPosEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i));
  }
  return posEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation> PinocchioEndEffectorKinematicsCppAd::getLinearApproximationPosEesRtWorldEwrtWorld(
    const vector_t& state) const {
  const vector_t stackedPosEesRtWorldEwrtWorld = positionCppAdInterfacePtr_->getFunctionValue(state);
  const matrix_t stackedJacobianPosEesRtWorldEwrtWorld = positionCppAdInterfacePtr_->getJacobian(state);

  std::vector<VectorFunctionLinearApproximation> approxPosEesRtWorldEwrtWorld;
  approxPosEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    VectorFunctionLinearApproximation approxPosEeRtWorldEwrtWorld;
    approxPosEeRtWorldEwrtWorld.f = stackedPosEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i);
    approxPosEeRtWorldEwrtWorld.dfdx = stackedJacobianPosEesRtWorldEwrtWorld.block(kPositionDim * i, 0, kPositionDim, state.rows());
    approxPosEesRtWorldEwrtWorld.emplace_back(std::move(approxPosEeRtWorldEwrtWorld));
  }
  return approxPosEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getLinvelEesRtWorldEwrtWorld(const vector_t& state, const vector_t& input) const -> std::vector<vector3_t> {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t stackedLinvelEesRtWorldEwrtWorld = velocityCppAdInterfacePtr_->getFunctionValue(stateInput);

  std::vector<vector3_t> linvelEesRtWorldEwrtWorld;
  linvelEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    linvelEesRtWorldEwrtWorld.emplace_back(stackedLinvelEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i));
  }
  return linvelEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation> PinocchioEndEffectorKinematicsCppAd::getLinearApproximationLinvelEesRtWorldEwrtWorld(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t stackedLinvelEesRtWorldEwrtWorld = velocityCppAdInterfacePtr_->getFunctionValue(stateInput);
  const matrix_t stackedJacobianLinvelEesRtWorldEwrtWorld = velocityCppAdInterfacePtr_->getJacobian(stateInput);

  std::vector<VectorFunctionLinearApproximation> approxLinvelEesRtWorldEwrtWorld;
  approxLinvelEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    VectorFunctionLinearApproximation approxLinvelEeRtWorldEwrtWorld;
    approxLinvelEeRtWorldEwrtWorld.f = stackedLinvelEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i);
    approxLinvelEeRtWorldEwrtWorld.dfdx = stackedJacobianLinvelEesRtWorldEwrtWorld.block(kPositionDim * i, 0, kPositionDim, state.rows());
    approxLinvelEeRtWorldEwrtWorld.dfdu = stackedJacobianLinvelEesRtWorldEwrtWorld.block(kPositionDim * i, state.rows(), kPositionDim, input.rows());
    approxLinvelEesRtWorldEwrtWorld.emplace_back(std::move(approxLinvelEeRtWorldEwrtWorld));
  }
  return approxLinvelEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getQuatEesRtWorld(const vector_t& state) const -> std::vector<quaternion_t> {
  const vector_t stackedQuatEesRtWorldXyzw = orientationCppAdInterfacePtr_->getFunctionValue(state);

  std::vector<quaternion_t> quatEesRtWorld;
  quatEesRtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    quatEesRtWorld.emplace_back(quaternion_t(stackedQuatEesRtWorldXyzw.segment<kQuaternionCoefficients>(kQuaternionCoefficients * i)));
  }
  return quatEesRtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getTiltEesRtSurfaceEwrtWorld(const vector_t& state,
                                                                      const std::vector<vector3_t>& normalSurfacesEwrtWorld) const
    -> std::vector<vector3_t> {
  vector_t stackedNormalSurfacesEwrtWorld(kPositionDim * endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    stackedNormalSurfacesEwrtWorld.segment<kPositionDim>(kPositionDim * i) = normalSurfacesEwrtWorld[i];
  }

  const vector_t stackedTiltEesRtSurfaceEwrtWorld = tiltEesCppAdInterfacePtr_->getFunctionValue(state, stackedNormalSurfacesEwrtWorld);

  std::vector<vector3_t> tiltEesRtSurfaceEwrtWorld;
  tiltEesRtSurfaceEwrtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    tiltEesRtSurfaceEwrtWorld.emplace_back(stackedTiltEesRtSurfaceEwrtWorld.segment<kRotationTangentDim>(kRotationTangentDim * i));
  }
  return tiltEesRtSurfaceEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<VectorFunctionLinearApproximation> PinocchioEndEffectorKinematicsCppAd::getLinearApproximationTiltEesRtSurfaceEwrtWorld(
    const vector_t& state, const std::vector<vector3_t>& normalSurfacesEwrtWorld) const {
  vector_t stackedNormalSurfacesEwrtWorld(kPositionDim * endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    stackedNormalSurfacesEwrtWorld.segment<kPositionDim>(kPositionDim * i) = normalSurfacesEwrtWorld[i];
  }
  const vector_t stackedTiltEesRtSurfaceEwrtWorld = tiltEesCppAdInterfacePtr_->getFunctionValue(state, stackedNormalSurfacesEwrtWorld);
  const matrix_t stackedJacobianTiltEesRtSurfaceEwrtWorld = tiltEesCppAdInterfacePtr_->getJacobian(state, stackedNormalSurfacesEwrtWorld);

  std::vector<VectorFunctionLinearApproximation> approxTiltEesRtSurfaceEwrtWorld;
  approxTiltEesRtSurfaceEwrtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    VectorFunctionLinearApproximation approxTiltEeRtSurfaceEwrtWorld;
    approxTiltEeRtSurfaceEwrtWorld.f = stackedTiltEesRtSurfaceEwrtWorld.segment<kRotationTangentDim>(kRotationTangentDim * i);
    approxTiltEeRtSurfaceEwrtWorld.dfdx = stackedJacobianTiltEesRtSurfaceEwrtWorld.block(kRotationTangentDim * i, 0, kRotationTangentDim, state.rows());
    approxTiltEesRtSurfaceEwrtWorld.emplace_back(std::move(approxTiltEeRtSurfaceEwrtWorld));
  }
  return approxTiltEesRtSurfaceEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getAngvelEesRtWorldEwrtWorld(const vector_t& state, const vector_t& input) const -> std::vector<vector3_t> {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t stackedAngvelEesRtWorldEwrtWorld = angularVelocityCppAdInterfacePtr_->getFunctionValue(stateInput);

  std::vector<vector3_t> angvelEesRtWorldEwrtWorld;
  angvelEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    angvelEesRtWorldEwrtWorld.emplace_back(stackedAngvelEesRtWorldEwrtWorld.segment<kRotationTangentDim>(kRotationTangentDim * i));
  }
  return angvelEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation> PinocchioEndEffectorKinematicsCppAd::getLinearApproximationAngvelEesRtWorldEwrtWorld(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t stackedAngvelEesRtWorldEwrtWorld = angularVelocityCppAdInterfacePtr_->getFunctionValue(stateInput);
  const matrix_t stackedJacobianAngvelEesRtWorldEwrtWorld = angularVelocityCppAdInterfacePtr_->getJacobian(stateInput);

  std::vector<VectorFunctionLinearApproximation> approxAngvelEesRtWorldEwrtWorld;
  approxAngvelEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    VectorFunctionLinearApproximation approxAngvelEeRtWorldEwrtWorld;
    approxAngvelEeRtWorldEwrtWorld.f = stackedAngvelEesRtWorldEwrtWorld.segment<kRotationTangentDim>(kRotationTangentDim * i);
    approxAngvelEeRtWorldEwrtWorld.dfdx = stackedJacobianAngvelEesRtWorldEwrtWorld.block(kRotationTangentDim * i, 0, kRotationTangentDim, state.rows());
    approxAngvelEeRtWorldEwrtWorld.dfdu = stackedJacobianAngvelEesRtWorldEwrtWorld.block(kRotationTangentDim * i, state.rows(), kRotationTangentDim, input.rows());
    approxAngvelEesRtWorldEwrtWorld.emplace_back(std::move(approxAngvelEeRtWorldEwrtWorld));
  }
  return approxAngvelEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
auto PinocchioEndEffectorKinematicsCppAd::getTwistEesRtWorldEwrtWorld(const vector_t& state, const vector_t& input) const -> std::vector<vector6_t> {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t stackedTwistEesRtWorldEwrtWorld = twistCppAdInterfacePtr_->getFunctionValue(stateInput);

  std::vector<vector6_t> twistEesRtWorldEwrtWorld;
  twistEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    twistEesRtWorldEwrtWorld.emplace_back(stackedTwistEesRtWorldEwrtWorld.segment<kTwistDim>(kTwistDim * i));
  }
  return twistEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation> PinocchioEndEffectorKinematicsCppAd::getLinearApproximationTwistEesRtWorldEwrtWorld(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t stackedTwistEesRtWorldEwrtWorld = twistCppAdInterfacePtr_->getFunctionValue(stateInput);
  const matrix_t stackedJacobianTwistEesRtWorldEwrtWorld = twistCppAdInterfacePtr_->getJacobian(stateInput);

  std::vector<VectorFunctionLinearApproximation> approxTwistEesRtWorldEwrtWorld;
  approxTwistEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (int i = 0; i < static_cast<int>(endEffectorIds_.size()); i++) {
    VectorFunctionLinearApproximation approxTwistEeRtWorldEwrtWorld;
    approxTwistEeRtWorldEwrtWorld.f = stackedTwistEesRtWorldEwrtWorld.segment<kTwistDim>(kTwistDim * i);
    approxTwistEeRtWorldEwrtWorld.dfdx = stackedJacobianTwistEesRtWorldEwrtWorld.block(kTwistDim * i, 0, kTwistDim, state.rows());
    approxTwistEeRtWorldEwrtWorld.dfdu = stackedJacobianTwistEesRtWorldEwrtWorld.block(kTwistDim * i, state.rows(), kTwistDim, input.rows());
    approxTwistEesRtWorldEwrtWorld.emplace_back(std::move(approxTwistEeRtWorldEwrtWorld));
  }
  return approxTwistEesRtWorldEwrtWorld;
}

}  // namespace ocs2
