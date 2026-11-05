/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_wb_mpc/end_effector/PinocchioEndEffectorDynamicsCppAd.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"
#include "humanoid_wb_mpc/end_effector/OrientationErrorWrtPlaneRates.h"

namespace ocs2::humanoid {

namespace {

using RobotModelCppAd = WBAccelMpcRobotModel<ad_scalar_t>;

/** A plane's unit normal, a vector of space: the parameters of each frame of
 * the pose-wrt-plane libraries. */
constexpr int kPlaneNormalDim = kPositionDim;

/** [state; input], the variables of the libraries of functions of both. */
vector_t stateAndInput(const vector_t& state, const vector_t& input) {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  return stateInput;
}

/** The plane normals one after the other, the parameters of the pose-wrt-plane
 * libraries. */
vector_t planeNormalParameters(
    const std::vector<vector3_t>& normalSurfacesEwrtWorld) {
  vector_t params(kPlaneNormalDim * normalSurfacesEwrtWorld.size());
  for (size_t i = 0; i < normalSurfacesEwrtWorld.size(); ++i) {
    params.segment<kPlaneNormalDim>(kPlaneNormalDim * i) =
        normalSurfacesEwrtWorld[i];
  }
  return params;
}

/** The kPoseTangentDim-row blocks of `values`, a pose's change against the
 * plane, one per end effector of `numEndEffectors`. */
std::vector<vector6_t> xeeBlocks(const vector_t& values,
                                 size_t numEndEffectors) {
  std::vector<vector6_t> blocks;
  blocks.reserve(numEndEffectors);
  for (size_t i = 0; i < numEndEffectors; ++i) {
    blocks.emplace_back(values.segment<kPoseTangentDim>(kPoseTangentDim * i));
  }
  return blocks;
}

/**
 * The linear approximations of the kPoseTangentDim-row blocks of the function
 * of `library` of [state; input] and `params`, one per end effector of
 * `numEndEffectors`: values and Jacobians of the same library, so that they
 * agree.
 */
std::vector<VectorFunctionLinearApproximation> xeeLinearApproximations(
    const CppAdInterface& library,
    const vector_t& state,
    const vector_t& input,
    const vector_t& params,
    size_t numEndEffectors) {
  const vector_t stateInput = stateAndInput(state, input);
  const vector_t values = library.getFunctionValue(stateInput, params);
  const matrix_t jacobian = library.getJacobian(stateInput, params);

  std::vector<VectorFunctionLinearApproximation> approximations;
  approximations.reserve(numEndEffectors);
  for (size_t i = 0; i < numEndEffectors; ++i) {
    VectorFunctionLinearApproximation approximation;
    approximation.f = values.segment<kPoseTangentDim>(kPoseTangentDim * i);
    approximation.dfdx =
        jacobian.block(kPoseTangentDim * i, 0, kPoseTangentDim, state.rows());
    approximation.dfdu = jacobian.block(kPoseTangentDim * i, state.rows(),
                                        kPoseTangentDim, input.rows());
    approximations.emplace_back(std::move(approximation));
  }
  return approximations;
}

/*
 * The taped functions, of the end-effector frames `frameIds` of
 * `pinocchioInterface` (whose data they overwrite) and of the state and input
 * of `robotModel`, 3 or 6 rows per frame in the order of `frameIds`.
 */

/** The world positions of the frames: the function of `<modelName>_position`.
 */
ad_vector_t computePosEesRtWorldEwrtWorld(
    PinocchioInterfaceCppAd& pinocchioInterface,
    const RobotModelCppAd& robotModel,
    const std::vector<size_t>& frameIds,
    const ad_vector_t& state) {
  const PinocchioInterfaceCppAd::Model& model = pinocchioInterface.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterface.getData();
  const ad_vector_t q = robotModel.getGeneralizedCoordinates(state);

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t posEesRtWorldEwrtWorld(kPositionDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    posEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i) =
        data.oMf[frameId].translation();
  }
  return posEesRtWorldEwrtWorld;
}

/** The world linear velocities of the frames' origins: the function of
 * `<modelName>_velocity`. */
ad_vector_t computeLinvelEesRtWorldEwrtWorld(
    PinocchioInterfaceCppAd& pinocchioInterface,
    RobotModelCppAd& robotModel,
    const std::vector<size_t>& frameIds,
    const ad_vector_t& state,
    const ad_vector_t& input) {
  const pinocchio::ReferenceFrame rf =
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  const PinocchioInterfaceCppAd::Model& model = pinocchioInterface.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterface.getData();
  const ad_vector_t q = robotModel.getGeneralizedCoordinates(state);
  const ad_vector_t v = robotModel.getGeneralizedVelocities(state, input);

  pinocchio::forwardKinematics(model, data, q, v);

  ad_vector_t linvelEesRtWorldEwrtWorld(kPositionDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    linvelEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i) =
        pinocchio::getFrameVelocity(model, data, frameId, rf).linear();
  }
  return linvelEesRtWorldEwrtWorld;
}

/**
 * The tilts of the frames relative to the planes of the unit
 * normals `params` (3 entries per frame, rotationMatrixDistanceToPlane()): the
 * function of `<modelName>_orientation_wrt_plane`.
 */
ad_vector_t computeTiltEesRtSurfaceEwrtWorld(
    PinocchioInterfaceCppAd& pinocchioInterface,
    const RobotModelCppAd& robotModel,
    const std::vector<size_t>& frameIds,
    const ad_vector_t& state,
    const ad_vector_t& params) {
  const PinocchioInterfaceCppAd::Model& model = pinocchioInterface.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterface.getData();
  const ad_vector_t q = robotModel.getGeneralizedCoordinates(state);

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t tiltEesRtSurfaceEwrtWorld(kRotationTangentDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    ad_vector_t normalSurfaceEwrtWorld =
        params.segment<kPlaneNormalDim>(kPlaneNormalDim * i);
    tiltEesRtSurfaceEwrtWorld.segment<kRotationTangentDim>(kRotationTangentDim *
                                                           i) =
        rotationMatrixDistanceToPlane<ad_scalar_t>(data.oMf[frameId].rotation(),
                                                   normalSurfaceEwrtWorld);
  }
  return tiltEesRtSurfaceEwrtWorld;
}

/**
 * The world classical linear accelerations of the frames' origins, at the
 * whole-body dynamics' generalized acceleration: the function of
 * `<modelName>_linear_acceleration`.
 */
ad_vector_t computeLinaccEesRtWorldEwrtWorld(
    PinocchioInterfaceCppAd& pinocchioInterface,
    RobotModelCppAd& robotModel,
    const std::vector<size_t>& frameIds,
    const ad_vector_t& state,
    const ad_vector_t& input) {
  const pinocchio::ReferenceFrame rf =
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  const PinocchioInterfaceCppAd::Model& model = pinocchioInterface.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterface.getData();
  const ad_vector_t q = robotModel.getGeneralizedCoordinates(state);
  const ad_vector_t v = robotModel.getGeneralizedVelocities(state, input);
  const ad_vector_t a = computeGeneralizedAccelerations<ad_scalar_t>(
      state, input, pinocchioInterface, robotModel);

  pinocchio::forwardKinematics(model, data, q, v, a);

  ad_vector_t linaccEesRtWorldEwrtWorld(kPositionDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    linaccEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i) =
        pinocchio::getFrameClassicalAcceleration(model, data, frameId, rf)
            .linear();
  }
  return linaccEesRtWorldEwrtWorld;
}

/**
 * [linear velocity; planeTiltRate()] of every frame (6 rows
 * each), `params` its unit plane normal (3 entries each): the function of
 * `<modelName>_pose_wrt_plane_rate`.
 */
ad_vector_t computeXeeRates(PinocchioInterfaceCppAd& pinocchioInterface,
                            RobotModelCppAd& robotModel,
                            const std::vector<size_t>& frameIds,
                            const ad_vector_t& state,
                            const ad_vector_t& input,
                            const ad_vector_t& params) {
  const pinocchio::ReferenceFrame rf =
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  const PinocchioInterfaceCppAd::Model& model = pinocchioInterface.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterface.getData();
  const ad_vector_t q = robotModel.getGeneralizedCoordinates(state);
  const ad_vector_t v = robotModel.getGeneralizedVelocities(state, input);

  pinocchio::forwardKinematics(model, data, q, v);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t xeeRates(kPoseTangentDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    // LOCAL_WORLD_ALIGNED: the velocity of the frame's origin and the angular
    // velocity, both in the world frame.
    const pinocchio::MotionTpl<ad_scalar_t> twistEeRtWorldEwrtWorld =
        pinocchio::getFrameVelocity(model, data, frameId, rf);
    const ad_matrix3_t rotmatEeRtWorld = data.oMf[frameId].rotation();
    const ad_vector3_t angvelEeRtWorldEwrtWorld =
        twistEeRtWorldEwrtWorld.angular();
    const ad_vector3_t normalSurfaceEwrtWorld =
        params.segment<kPlaneNormalDim>(kPlaneNormalDim * i);
    xeeRates.segment<kPositionDim>(kPoseTangentDim * i) =
        twistEeRtWorldEwrtWorld.linear();
    xeeRates.segment<kRotationTangentDim>(kPoseTangentDim * i + kPositionDim) =
        planeTiltRate<ad_scalar_t>(rotmatEeRtWorld, angvelEeRtWorldEwrtWorld,
                                   normalSurfaceEwrtWorld);
  }
  return xeeRates;
}

/**
 * [classical linear acceleration; planeTiltSecondRate()] of
 * every frame (6 rows each), `params` its unit plane normal (3 entries each),
 * at the whole-body dynamics' generalized acceleration: the function of
 * `<modelName>_pose_wrt_plane_acceleration`.
 */
ad_vector_t computeXeeSecondRates(PinocchioInterfaceCppAd& pinocchioInterface,
                                  RobotModelCppAd& robotModel,
                                  const std::vector<size_t>& frameIds,
                                  const ad_vector_t& state,
                                  const ad_vector_t& input,
                                  const ad_vector_t& params) {
  const pinocchio::ReferenceFrame rf =
      pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
  const PinocchioInterfaceCppAd::Model& model = pinocchioInterface.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterface.getData();
  const ad_vector_t q = robotModel.getGeneralizedCoordinates(state);
  const ad_vector_t v = robotModel.getGeneralizedVelocities(state, input);
  const ad_vector_t a = computeGeneralizedAccelerations<ad_scalar_t>(
      state, input, pinocchioInterface, robotModel);

  pinocchio::forwardKinematics(model, data, q, v, a);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t xeeSecondRates(kPoseTangentDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    // LOCAL_WORLD_ALIGNED: the classical acceleration's linear part is the
    // second derivative of the frame's origin, and its angular part the time
    // derivative of the world-frame angular velocity (getFrameAcceleration's R
    // a_angular).
    const pinocchio::MotionTpl<ad_scalar_t> twistEeRtWorldEwrtWorld =
        pinocchio::getFrameVelocity(model, data, frameId, rf);
    const pinocchio::MotionTpl<ad_scalar_t> classicalaccEeRtWorldEwrtWorld =
        pinocchio::getFrameClassicalAcceleration(model, data, frameId, rf);
    const ad_matrix3_t rotmatEeRtWorld = data.oMf[frameId].rotation();
    const ad_vector3_t angvelEeRtWorldEwrtWorld =
        twistEeRtWorldEwrtWorld.angular();
    const ad_vector3_t angaccEeRtWorldEwrtWorld =
        classicalaccEeRtWorldEwrtWorld.angular();
    const ad_vector3_t normalSurfaceEwrtWorld =
        params.segment<kPlaneNormalDim>(kPlaneNormalDim * i);
    xeeSecondRates.segment<kPositionDim>(kPoseTangentDim * i) =
        classicalaccEeRtWorldEwrtWorld.linear();
    xeeSecondRates.segment<kRotationTangentDim>(kPoseTangentDim * i +
                                                kPositionDim) =
        planeTiltSecondRate<ad_scalar_t>(
            rotmatEeRtWorld, angvelEeRtWorldEwrtWorld, angaccEeRtWorldEwrtWorld,
            normalSurfaceEwrtWorld);
  }
  return xeeSecondRates;
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

PinocchioEndEffectorDynamicsCppAd::PinocchioEndEffectorDynamicsCppAd(
    const PinocchioInterface& pinocchioInterface,
    WBAccelMpcRobotModel<ad_scalar_t>& mpcRobotModel,
    std::vector<std::string> endEffectorIds,
    const std::string& modelName,
    const std::string& modelFolder,
    bool recompileLibraries,
    bool verbose)
    : endEffectorIds_(std::move(endEffectorIds)) {
  // What the functions read, each capturing its own copy (the class comment).
  // Every copy of a CppAdInterface holds a copy of its function, so the
  // Pinocchio interface the functions overwrite is held by a std::shared_ptr:
  // shared by the functions of this object and of every copy, and alive as long
  // as any of them is. The robot model is the caller's (the constructor's
  // contract).
  const std::shared_ptr<PinocchioInterfaceCppAd> pinocchioInterfaceCppAd =
      std::make_shared<PinocchioInterfaceCppAd>(pinocchioInterface.toCppAd());
  RobotModelCppAd* absl_nonnull const robotModel = &mpcRobotModel;
  std::vector<size_t> frameIds;
  frameIds.reserve(endEffectorIds_.size());
  for (const std::string& bodyName : endEffectorIds_) {
    frameIds.push_back(pinocchioInterface.getModel().getFrameId(bodyName));
  }
  const size_t stateDim = mpcRobotModel.getStateDim();
  const size_t inputDim = mpcRobotModel.getInputDim();
  const size_t numPlaneParameters = kPlaneNormalDim * frameIds.size();

  const CppAdInterface::ad_function_t tapePosEesRtWorldEwrtWorld =
      [pinocchioInterfaceCppAd, robotModel, frameIds](const ad_vector_t& x,
                                                      ad_vector_t& y) {
        y = computePosEesRtWorldEwrtWorld(*pinocchioInterfaceCppAd, *robotModel,
                                          frameIds, x);
      };
  positionCppAdInterfacePtr_ =
      std::make_unique<CppAdInterface>(tapePosEesRtWorldEwrtWorld, stateDim,
                                       modelName + "_position", modelFolder);

  const CppAdInterface::ad_function_t tapeLinvelEesRtWorldEwrtWorld =
      [pinocchioInterfaceCppAd, robotModel, frameIds, stateDim, inputDim](
          const ad_vector_t& x, ad_vector_t& y) {
        const ad_vector_t state = x.head(stateDim);
        const ad_vector_t input = x.tail(inputDim);
        y = computeLinvelEesRtWorldEwrtWorld(
            *pinocchioInterfaceCppAd, *robotModel, frameIds, state, input);
      };
  velocityCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      tapeLinvelEesRtWorldEwrtWorld, stateDim + inputDim,
      modelName + "_velocity", modelFolder);

  const CppAdInterface::ad_parameterized_function_t
      tapeTiltEesRtSurfaceEwrtWorld =
          [pinocchioInterfaceCppAd, robotModel, frameIds](
              const ad_vector_t& x, const ad_vector_t& params, ad_vector_t& y) {
            y = computeTiltEesRtSurfaceEwrtWorld(
                *pinocchioInterfaceCppAd, *robotModel, frameIds, x, params);
          };
  tiltEesCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      tapeTiltEesRtSurfaceEwrtWorld, stateDim, numPlaneParameters,
      modelName + "_orientation_wrt_plane", modelFolder);

  const CppAdInterface::ad_function_t tapeLinaccEesRtWorldEwrtWorld =
      [pinocchioInterfaceCppAd, robotModel, frameIds, stateDim, inputDim](
          const ad_vector_t& x, ad_vector_t& y) {
        const ad_vector_t state = x.head(stateDim);
        const ad_vector_t input = x.tail(inputDim);
        y = computeLinaccEesRtWorldEwrtWorld(
            *pinocchioInterfaceCppAd, *robotModel, frameIds, state, input);
      };
  linearAccelerationCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      tapeLinaccEesRtWorldEwrtWorld, stateDim + inputDim,
      modelName + "_linear_acceleration", modelFolder);

  // The pose rates with respect to a plane, parameterized by the plane normals.
  const CppAdInterface::ad_parameterized_function_t tapeXeeRates =
      [pinocchioInterfaceCppAd, robotModel, frameIds, stateDim, inputDim](
          const ad_vector_t& x, const ad_vector_t& params, ad_vector_t& y) {
        const ad_vector_t state = x.head(stateDim);
        const ad_vector_t input = x.tail(inputDim);
        y = computeXeeRates(*pinocchioInterfaceCppAd, *robotModel, frameIds,
                            state, input, params);
      };
  xeeRateCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      tapeXeeRates, stateDim + inputDim, numPlaneParameters,
      modelName + "_pose_wrt_plane_rate", modelFolder);

  const CppAdInterface::ad_parameterized_function_t tapeXeeSecondRates =
      [pinocchioInterfaceCppAd, robotModel, frameIds, stateDim, inputDim](
          const ad_vector_t& x, const ad_vector_t& params, ad_vector_t& y) {
        const ad_vector_t state = x.head(stateDim);
        const ad_vector_t input = x.tail(inputDim);
        y = computeXeeSecondRates(*pinocchioInterfaceCppAd, *robotModel,
                                  frameIds, state, input, params);
      };
  xeeSecondRateCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      tapeXeeSecondRates, stateDim + inputDim, numPlaneParameters,
      modelName + "_pose_wrt_plane_acceleration", modelFolder);

  if (recompileLibraries) {
    positionCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
    velocityCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
    tiltEesCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
    linearAccelerationCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
    xeeRateCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
    xeeSecondRateCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
  } else {
    positionCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
    velocityCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
    tiltEesCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
    linearAccelerationCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
    xeeRateCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
    xeeSecondRateCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PinocchioEndEffectorDynamicsCppAd::PinocchioEndEffectorDynamicsCppAd(
    const PinocchioEndEffectorDynamicsCppAd& rhs)
    : EndEffectorDynamics<scalar_t>(rhs),
      positionCppAdInterfacePtr_(
          std::make_unique<CppAdInterface>(*rhs.positionCppAdInterfacePtr_)),
      velocityCppAdInterfacePtr_(
          std::make_unique<CppAdInterface>(*rhs.velocityCppAdInterfacePtr_)),
      linearAccelerationCppAdInterfacePtr_(std::make_unique<CppAdInterface>(
          *rhs.linearAccelerationCppAdInterfacePtr_)),
      tiltEesCppAdInterfacePtr_(
          std::make_unique<CppAdInterface>(*rhs.tiltEesCppAdInterfacePtr_)),
      xeeRateCppAdInterfacePtr_(
          std::make_unique<CppAdInterface>(*rhs.xeeRateCppAdInterfacePtr_)),
      xeeSecondRateCppAdInterfacePtr_(std::make_unique<CppAdInterface>(
          *rhs.xeeSecondRateCppAdInterfacePtr_)),
      endEffectorIds_(rhs.endEffectorIds_) {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
PinocchioEndEffectorDynamicsCppAd* absl_nonnull
PinocchioEndEffectorDynamicsCppAd::clone() const {
  return new PinocchioEndEffectorDynamicsCppAd(*this);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
const std::vector<std::string>& PinocchioEndEffectorDynamicsCppAd::getIds()
    const {
  return endEffectorIds_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<PinocchioEndEffectorDynamicsCppAd::vector3_t>
PinocchioEndEffectorDynamicsCppAd::getPosEesRtWorldEwrtWorld(
    const vector_t& state) const {
  const vector_t stackedPosEesRtWorldEwrtWorld =
      positionCppAdInterfacePtr_->getFunctionValue(state);

  std::vector<vector3_t> posEesRtWorldEwrtWorld;
  posEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    posEesRtWorldEwrtWorld.emplace_back(
        stackedPosEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i));
  }
  return posEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::getLinearApproximationPosEesRtWorldEwrtWorld(
    const vector_t& state) const {
  const vector_t stackedPosEesRtWorldEwrtWorld =
      positionCppAdInterfacePtr_->getFunctionValue(state);
  const matrix_t stackedJacobianPosEesRtWorldEwrtWorld =
      positionCppAdInterfacePtr_->getJacobian(state);

  std::vector<VectorFunctionLinearApproximation> approxPosEesRtWorldEwrtWorld;
  approxPosEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    VectorFunctionLinearApproximation approxPosEeRtWorldEwrtWorld;
    approxPosEeRtWorldEwrtWorld.f =
        stackedPosEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim * i);
    approxPosEeRtWorldEwrtWorld.dfdx =
        stackedJacobianPosEesRtWorldEwrtWorld.block(kPositionDim * i, 0,
                                                    kPositionDim, state.rows());
    approxPosEesRtWorldEwrtWorld.emplace_back(
        std::move(approxPosEeRtWorldEwrtWorld));
  }
  return approxPosEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<PinocchioEndEffectorDynamicsCppAd::vector3_t>
PinocchioEndEffectorDynamicsCppAd::getLinvelEesRtWorldEwrtWorld(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t stackedLinvelEesRtWorldEwrtWorld =
      velocityCppAdInterfacePtr_->getFunctionValue(stateInput);

  std::vector<vector3_t> linvelEesRtWorldEwrtWorld;
  linvelEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    linvelEesRtWorldEwrtWorld.emplace_back(
        stackedLinvelEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim *
                                                               i));
  }
  return linvelEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::
    getLinearApproximationLinvelEesRtWorldEwrtWorld(
        const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t stackedLinvelEesRtWorldEwrtWorld =
      velocityCppAdInterfacePtr_->getFunctionValue(stateInput);
  const matrix_t stackedJacobianLinvelEesRtWorldEwrtWorld =
      velocityCppAdInterfacePtr_->getJacobian(stateInput);

  std::vector<VectorFunctionLinearApproximation>
      approxLinvelEesRtWorldEwrtWorld;
  approxLinvelEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    VectorFunctionLinearApproximation approxLinvelEeRtWorldEwrtWorld;
    approxLinvelEeRtWorldEwrtWorld.f =
        stackedLinvelEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim *
                                                               i);
    approxLinvelEeRtWorldEwrtWorld.dfdx =
        stackedJacobianLinvelEesRtWorldEwrtWorld.block(
            kPositionDim * i, 0, kPositionDim, state.rows());
    approxLinvelEeRtWorldEwrtWorld.dfdu =
        stackedJacobianLinvelEesRtWorldEwrtWorld.block(
            kPositionDim * i, state.rows(), kPositionDim, input.rows());
    approxLinvelEesRtWorldEwrtWorld.emplace_back(
        std::move(approxLinvelEeRtWorldEwrtWorld));
  }
  return approxLinvelEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<PinocchioEndEffectorDynamicsCppAd::vector3_t>
PinocchioEndEffectorDynamicsCppAd::getTiltEesRtSurfaceEwrtWorld(
    const vector_t& state,
    const std::vector<vector3_t>& normalSurfacesEwrtWorld) const {
  vector_t params(kPlaneNormalDim * endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    params.segment<kPlaneNormalDim>(kPlaneNormalDim * i) =
        normalSurfacesEwrtWorld[i];
  }

  const vector_t stackedTiltEesRtSurfaceEwrtWorld =
      tiltEesCppAdInterfacePtr_->getFunctionValue(state, params);

  std::vector<vector3_t> tiltEesRtSurfaceEwrtWorld;
  tiltEesRtSurfaceEwrtWorld.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    tiltEesRtSurfaceEwrtWorld.emplace_back(
        stackedTiltEesRtSurfaceEwrtWorld.segment<kRotationTangentDim>(
            kRotationTangentDim * i));
  }
  return tiltEesRtSurfaceEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::
    getLinearApproximationTiltEesRtSurfaceEwrtWorld(
        const vector_t& state,
        const std::vector<vector3_t>& normalSurfacesEwrtWorld) const {
  vector_t params(kPlaneNormalDim * endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    params.segment<kPlaneNormalDim>(kPlaneNormalDim * i) =
        normalSurfacesEwrtWorld[i];
  }
  const vector_t stackedTiltEesRtSurfaceEwrtWorld =
      tiltEesCppAdInterfacePtr_->getFunctionValue(state, params);
  const matrix_t stackedJacobianTiltEesRtSurfaceEwrtWorld =
      tiltEesCppAdInterfacePtr_->getJacobian(state, params);

  std::vector<VectorFunctionLinearApproximation>
      approxTiltEesRtSurfaceEwrtWorld;
  approxTiltEesRtSurfaceEwrtWorld.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    VectorFunctionLinearApproximation approxTiltEeRtSurfaceEwrtWorld;
    approxTiltEeRtSurfaceEwrtWorld.f =
        stackedTiltEesRtSurfaceEwrtWorld.segment<kRotationTangentDim>(
            kRotationTangentDim * i);
    approxTiltEeRtSurfaceEwrtWorld.dfdx =
        stackedJacobianTiltEesRtSurfaceEwrtWorld.block(
            kRotationTangentDim * i, 0, kRotationTangentDim, state.rows());
    approxTiltEesRtSurfaceEwrtWorld.emplace_back(
        std::move(approxTiltEeRtSurfaceEwrtWorld));
  }
  return approxTiltEesRtSurfaceEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<PinocchioEndEffectorDynamicsCppAd::vector3_t>
PinocchioEndEffectorDynamicsCppAd::getLinaccEesRtWorldEwrtWorld(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t stackedLinaccEesRtWorldEwrtWorld =
      linearAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput);

  std::vector<vector3_t> linaccEesRtWorldEwrtWorld;
  linaccEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    linaccEesRtWorldEwrtWorld.emplace_back(
        stackedLinaccEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim *
                                                               i));
  }
  return linaccEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::
    getLinearApproximationLinaccEesRtWorldEwrtWorld(
        const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t stackedLinaccEesRtWorldEwrtWorld =
      linearAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput);
  const matrix_t stackedJacobianLinaccEesRtWorldEwrtWorld =
      linearAccelerationCppAdInterfacePtr_->getJacobian(stateInput);

  std::vector<VectorFunctionLinearApproximation>
      approxLinaccEesRtWorldEwrtWorld;
  approxLinaccEesRtWorldEwrtWorld.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    VectorFunctionLinearApproximation approxLinaccEeRtWorldEwrtWorld;
    approxLinaccEeRtWorldEwrtWorld.f =
        stackedLinaccEesRtWorldEwrtWorld.segment<kPositionDim>(kPositionDim *
                                                               i);
    approxLinaccEeRtWorldEwrtWorld.dfdx =
        stackedJacobianLinaccEesRtWorldEwrtWorld.block(
            kPositionDim * i, 0, kPositionDim, state.rows());
    approxLinaccEeRtWorldEwrtWorld.dfdu =
        stackedJacobianLinaccEesRtWorldEwrtWorld.block(
            kPositionDim * i, state.rows(), kPositionDim, input.rows());
    approxLinaccEesRtWorldEwrtWorld.emplace_back(
        std::move(approxLinaccEeRtWorldEwrtWorld));
  }
  return approxLinaccEesRtWorldEwrtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<PinocchioEndEffectorDynamicsCppAd::vector6_t>
PinocchioEndEffectorDynamicsCppAd::getXeeRate(
    const vector_t& state,
    const vector_t& input,
    const std::vector<vector3_t>& normalSurfacesEwrtWorld) const {
  return xeeBlocks(xeeRateCppAdInterfacePtr_->getFunctionValue(
                       stateAndInput(state, input),
                       planeNormalParameters(normalSurfacesEwrtWorld)),
                   endEffectorIds_.size());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<PinocchioEndEffectorDynamicsCppAd::vector6_t>
PinocchioEndEffectorDynamicsCppAd::getXeeSecondRate(
    const vector_t& state,
    const vector_t& input,
    const std::vector<vector3_t>& normalSurfacesEwrtWorld) const {
  return xeeBlocks(xeeSecondRateCppAdInterfacePtr_->getFunctionValue(
                       stateAndInput(state, input),
                       planeNormalParameters(normalSurfacesEwrtWorld)),
                   endEffectorIds_.size());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::getLinearApproximationXeeRate(
    const vector_t& state,
    const vector_t& input,
    const std::vector<vector3_t>& normalSurfacesEwrtWorld) const {
  return xeeLinearApproximations(*xeeRateCppAdInterfacePtr_, state, input,
                                 planeNormalParameters(normalSurfacesEwrtWorld),
                                 endEffectorIds_.size());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::getLinearApproximationXeeSecondRate(
    const vector_t& state,
    const vector_t& input,
    const std::vector<vector3_t>& normalSurfacesEwrtWorld) const {
  return xeeLinearApproximations(*xeeSecondRateCppAdInterfacePtr_, state, input,
                                 planeNormalParameters(normalSurfacesEwrtWorld),
                                 endEffectorIds_.size());
}

}  // namespace ocs2::humanoid
