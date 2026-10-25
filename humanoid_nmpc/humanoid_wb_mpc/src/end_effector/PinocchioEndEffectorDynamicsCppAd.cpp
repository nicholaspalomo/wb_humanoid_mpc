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
vector_t planeNormalParameters(const std::vector<vector3_t>& planeNormals) {
  vector_t params(kPlaneNormalDim * planeNormals.size());
  for (size_t i = 0; i < planeNormals.size(); ++i) {
    params.segment<kPlaneNormalDim>(kPlaneNormalDim * i) = planeNormals[i];
  }
  return params;
}

/** The kPoseTangentDim-row blocks of `values`, a pose's change against the
 * plane, one per end effector of `numEndEffectors`. */
std::vector<vector6_t> poseWrtPlaneBlocks(const vector_t& values,
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
std::vector<VectorFunctionLinearApproximation> poseWrtPlaneLinearApproximations(
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
ad_vector_t positionsCppAd(PinocchioInterfaceCppAd& pinocchioInterface,
                           const RobotModelCppAd& robotModel,
                           const std::vector<size_t>& frameIds,
                           const ad_vector_t& state) {
  const PinocchioInterfaceCppAd::Model& model = pinocchioInterface.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterface.getData();
  const ad_vector_t q = robotModel.getGeneralizedCoordinates(state);

  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  ad_vector_t positions(kPositionDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    positions.segment<kPositionDim>(kPositionDim * i) =
        data.oMf[frameId].translation();
  }
  return positions;
}

/** The world linear velocities of the frames' origins: the function of
 * `<modelName>_velocity`. */
ad_vector_t velocitiesCppAd(PinocchioInterfaceCppAd& pinocchioInterface,
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

  ad_vector_t velocities(kPositionDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    velocities.segment<kPositionDim>(kPositionDim * i) =
        pinocchio::getFrameVelocity(model, data, frameId, rf).linear();
  }
  return velocities;
}

/**
 * The orientation errors of the frames with respect to the planes of the unit
 * normals `params` (3 entries per frame, rotationMatrixDistanceToPlane()): the
 * function of `<modelName>_orientation_wrt_plane`.
 */
ad_vector_t orientationErrorsWrtPlaneCppAd(
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

  ad_vector_t errors(kRotationTangentDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    ad_vector_t planeNormal =
        params.segment<kPlaneNormalDim>(kPlaneNormalDim * i);
    errors.segment<kRotationTangentDim>(kRotationTangentDim * i) =
        rotationMatrixDistanceToPlane<ad_scalar_t>(data.oMf[frameId].rotation(),
                                                   planeNormal);
  }
  return errors;
}

/**
 * The world classical linear accelerations of the frames' origins, at the
 * whole-body dynamics' generalized acceleration: the function of
 * `<modelName>_linear_acceleration`.
 */
ad_vector_t linearAccelerationsCppAd(
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

  ad_vector_t accelerations(kPositionDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    accelerations.segment<kPositionDim>(kPositionDim * i) =
        pinocchio::getFrameClassicalAcceleration(model, data, frameId, rf)
            .linear();
  }
  return accelerations;
}

/**
 * [linear velocity; orientationErrorWrtPlaneRate()] of every frame (6 rows
 * each), `params` its unit plane normal (3 entries each): the function of
 * `<modelName>_pose_wrt_plane_rate`.
 */
ad_vector_t poseWrtPlaneRatesCppAd(PinocchioInterfaceCppAd& pinocchioInterface,
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

  ad_vector_t rates(kPoseTangentDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    // LOCAL_WORLD_ALIGNED: the velocity of the frame's origin and the angular
    // velocity, both in the world frame.
    const pinocchio::MotionTpl<ad_scalar_t> velocity =
        pinocchio::getFrameVelocity(model, data, frameId, rf);
    const ad_matrix3_t rotation = data.oMf[frameId].rotation();
    const ad_vector3_t angularVelocity = velocity.angular();
    const ad_vector3_t planeNormal =
        params.segment<kPlaneNormalDim>(kPlaneNormalDim * i);
    rates.segment<kPositionDim>(kPoseTangentDim * i) = velocity.linear();
    rates.segment<kRotationTangentDim>(kPoseTangentDim * i + kPositionDim) =
        orientationErrorWrtPlaneRate<ad_scalar_t>(rotation, angularVelocity,
                                                  planeNormal);
  }
  return rates;
}

/**
 * [classical linear acceleration; orientationErrorWrtPlaneAcceleration()] of
 * every frame (6 rows each), `params` its unit plane normal (3 entries each),
 * at the whole-body dynamics' generalized acceleration: the function of
 * `<modelName>_pose_wrt_plane_acceleration`.
 */
ad_vector_t poseWrtPlaneAccelerationsCppAd(
    PinocchioInterfaceCppAd& pinocchioInterface,
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

  ad_vector_t accelerations(kPoseTangentDim * frameIds.size());
  for (size_t i = 0; i < frameIds.size(); ++i) {
    const size_t frameId = frameIds[i];
    // LOCAL_WORLD_ALIGNED: the classical acceleration's linear part is the
    // second derivative of the frame's origin, and its angular part the time
    // derivative of the world-frame angular velocity (getFrameAcceleration's R
    // a_angular).
    const pinocchio::MotionTpl<ad_scalar_t> velocity =
        pinocchio::getFrameVelocity(model, data, frameId, rf);
    const pinocchio::MotionTpl<ad_scalar_t> acceleration =
        pinocchio::getFrameClassicalAcceleration(model, data, frameId, rf);
    const ad_matrix3_t rotation = data.oMf[frameId].rotation();
    const ad_vector3_t angularVelocity = velocity.angular();
    const ad_vector3_t angularAcceleration = acceleration.angular();
    const ad_vector3_t planeNormal =
        params.segment<kPlaneNormalDim>(kPlaneNormalDim * i);
    accelerations.segment<kPositionDim>(kPoseTangentDim * i) =
        acceleration.linear();
    accelerations.segment<kRotationTangentDim>(kPoseTangentDim * i +
                                               kPositionDim) =
        orientationErrorWrtPlaneAcceleration<ad_scalar_t>(
            rotation, angularVelocity, angularAcceleration, planeNormal);
  }
  return accelerations;
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

  const CppAdInterface::ad_function_t positionFunc =
      [pinocchioInterfaceCppAd, robotModel, frameIds](const ad_vector_t& x,
                                                      ad_vector_t& y) {
        y = positionsCppAd(*pinocchioInterfaceCppAd, *robotModel, frameIds, x);
      };
  positionCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      positionFunc, stateDim, modelName + "_position", modelFolder);

  const CppAdInterface::ad_function_t velocityFunc =
      [pinocchioInterfaceCppAd, robotModel, frameIds, stateDim, inputDim](
          const ad_vector_t& x, ad_vector_t& y) {
        const ad_vector_t state = x.head(stateDim);
        const ad_vector_t input = x.tail(inputDim);
        y = velocitiesCppAd(*pinocchioInterfaceCppAd, *robotModel, frameIds,
                            state, input);
      };
  velocityCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      velocityFunc, stateDim + inputDim, modelName + "_velocity", modelFolder);

  const CppAdInterface::ad_parameterized_function_t orientationWrtPlaneFunc =
      [pinocchioInterfaceCppAd, robotModel, frameIds](
          const ad_vector_t& x, const ad_vector_t& params, ad_vector_t& y) {
        y = orientationErrorsWrtPlaneCppAd(*pinocchioInterfaceCppAd,
                                           *robotModel, frameIds, x, params);
      };
  orientationErrorWrtPlaneCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      orientationWrtPlaneFunc, stateDim, numPlaneParameters,
      modelName + "_orientation_wrt_plane", modelFolder);

  const CppAdInterface::ad_function_t linearAccelerationFunc =
      [pinocchioInterfaceCppAd, robotModel, frameIds, stateDim, inputDim](
          const ad_vector_t& x, ad_vector_t& y) {
        const ad_vector_t state = x.head(stateDim);
        const ad_vector_t input = x.tail(inputDim);
        y = linearAccelerationsCppAd(*pinocchioInterfaceCppAd, *robotModel,
                                     frameIds, state, input);
      };
  linearAccelerationCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      linearAccelerationFunc, stateDim + inputDim,
      modelName + "_linear_acceleration", modelFolder);

  // The pose rates with respect to a plane, parameterized by the plane normals.
  const CppAdInterface::ad_parameterized_function_t poseWrtPlaneRateFunc =
      [pinocchioInterfaceCppAd, robotModel, frameIds, stateDim, inputDim](
          const ad_vector_t& x, const ad_vector_t& params, ad_vector_t& y) {
        const ad_vector_t state = x.head(stateDim);
        const ad_vector_t input = x.tail(inputDim);
        y = poseWrtPlaneRatesCppAd(*pinocchioInterfaceCppAd, *robotModel,
                                   frameIds, state, input, params);
      };
  poseWrtPlaneRateCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      poseWrtPlaneRateFunc, stateDim + inputDim, numPlaneParameters,
      modelName + "_pose_wrt_plane_rate", modelFolder);

  const CppAdInterface::ad_parameterized_function_t
      poseWrtPlaneAccelerationFunc =
          [pinocchioInterfaceCppAd, robotModel, frameIds, stateDim, inputDim](
              const ad_vector_t& x, const ad_vector_t& params, ad_vector_t& y) {
            const ad_vector_t state = x.head(stateDim);
            const ad_vector_t input = x.tail(inputDim);
            y = poseWrtPlaneAccelerationsCppAd(*pinocchioInterfaceCppAd,
                                               *robotModel, frameIds, state,
                                               input, params);
          };
  poseWrtPlaneAccelerationCppAdInterfacePtr_ = std::make_unique<CppAdInterface>(
      poseWrtPlaneAccelerationFunc, stateDim + inputDim, numPlaneParameters,
      modelName + "_pose_wrt_plane_acceleration", modelFolder);

  if (recompileLibraries) {
    positionCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
    velocityCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
    orientationErrorWrtPlaneCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
    linearAccelerationCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
    poseWrtPlaneRateCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
    poseWrtPlaneAccelerationCppAdInterfacePtr_->createModels(
        CppAdInterface::ApproximationOrder::First, verbose);
  } else {
    positionCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
    velocityCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
    orientationErrorWrtPlaneCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
    linearAccelerationCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
    poseWrtPlaneRateCppAdInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, verbose);
    poseWrtPlaneAccelerationCppAdInterfacePtr_->loadModelsIfAvailable(
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
      orientationErrorWrtPlaneCppAdInterfacePtr_(
          std::make_unique<CppAdInterface>(
              *rhs.orientationErrorWrtPlaneCppAdInterfacePtr_)),
      poseWrtPlaneRateCppAdInterfacePtr_(std::make_unique<CppAdInterface>(
          *rhs.poseWrtPlaneRateCppAdInterfacePtr_)),
      poseWrtPlaneAccelerationCppAdInterfacePtr_(
          std::make_unique<CppAdInterface>(
              *rhs.poseWrtPlaneAccelerationCppAdInterfacePtr_)),
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
PinocchioEndEffectorDynamicsCppAd::getPosition(const vector_t& state) const {
  const vector_t positionValues =
      positionCppAdInterfacePtr_->getFunctionValue(state);

  std::vector<vector3_t> positions;
  positions.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    positions.emplace_back(
        positionValues.segment<kPositionDim>(kPositionDim * i));
  }
  return positions;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::getPositionLinearApproximation(
    const vector_t& state) const {
  const vector_t positionValues =
      positionCppAdInterfacePtr_->getFunctionValue(state);
  const matrix_t positionJacobian =
      positionCppAdInterfacePtr_->getJacobian(state);

  std::vector<VectorFunctionLinearApproximation> positions;
  positions.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    VectorFunctionLinearApproximation pos;
    pos.f = positionValues.segment<kPositionDim>(kPositionDim * i);
    pos.dfdx =
        positionJacobian.block(kPositionDim * i, 0, kPositionDim, state.rows());
    positions.emplace_back(std::move(pos));
  }
  return positions;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<PinocchioEndEffectorDynamicsCppAd::vector3_t>
PinocchioEndEffectorDynamicsCppAd::getVelocity(const vector_t& state,
                                               const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t velocityValues =
      velocityCppAdInterfacePtr_->getFunctionValue(stateInput);

  std::vector<vector3_t> velocities;
  velocities.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    velocities.emplace_back(
        velocityValues.segment<kPositionDim>(kPositionDim * i));
  }
  return velocities;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::getVelocityLinearApproximation(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t velocityValues =
      velocityCppAdInterfacePtr_->getFunctionValue(stateInput);
  const matrix_t velocityJacobian =
      velocityCppAdInterfacePtr_->getJacobian(stateInput);

  std::vector<VectorFunctionLinearApproximation> velocities;
  velocities.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    VectorFunctionLinearApproximation vel;
    vel.f = velocityValues.segment<kPositionDim>(kPositionDim * i);
    vel.dfdx =
        velocityJacobian.block(kPositionDim * i, 0, kPositionDim, state.rows());
    vel.dfdu = velocityJacobian.block(kPositionDim * i, state.rows(),
                                      kPositionDim, input.rows());
    velocities.emplace_back(std::move(vel));
  }
  return velocities;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<PinocchioEndEffectorDynamicsCppAd::vector3_t>
PinocchioEndEffectorDynamicsCppAd::getOrientationErrorWrtPlane(
    const vector_t& state, const std::vector<vector3_t>& planeNormals) const {
  vector_t params(kPlaneNormalDim * endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    params.segment<kPlaneNormalDim>(kPlaneNormalDim * i) = planeNormals[i];
  }

  const vector_t errorValues =
      orientationErrorWrtPlaneCppAdInterfacePtr_->getFunctionValue(state,
                                                                   params);

  std::vector<vector3_t> errors;
  errors.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    errors.emplace_back(
        errorValues.segment<kRotationTangentDim>(kRotationTangentDim * i));
  }
  return errors;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::
    getOrientationErrorWrtPlaneLinearApproximation(
        const vector_t& state,
        const std::vector<vector3_t>& planeNormals) const {
  vector_t params(kPlaneNormalDim * endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    params.segment<kPlaneNormalDim>(kPlaneNormalDim * i) = planeNormals[i];
  }
  const vector_t errorValues =
      orientationErrorWrtPlaneCppAdInterfacePtr_->getFunctionValue(state,
                                                                   params);
  const matrix_t errorJacobian =
      orientationErrorWrtPlaneCppAdInterfacePtr_->getJacobian(state, params);

  std::vector<VectorFunctionLinearApproximation> errors;
  errors.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    VectorFunctionLinearApproximation err;
    err.f = errorValues.segment<kRotationTangentDim>(kRotationTangentDim * i);
    err.dfdx = errorJacobian.block(kRotationTangentDim * i, 0,
                                   kRotationTangentDim, state.rows());
    errors.emplace_back(std::move(err));
  }
  return errors;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<PinocchioEndEffectorDynamicsCppAd::vector3_t>
PinocchioEndEffectorDynamicsCppAd::getLinearAcceleration(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t accelerationValues =
      linearAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput);

  std::vector<vector3_t> accelerations;
  accelerations.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    accelerations.emplace_back(
        accelerationValues.segment<kPositionDim>(kPositionDim * i));
  }
  return accelerations;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::getLinearAccelerationLinearApproximation(
    const vector_t& state, const vector_t& input) const {
  vector_t stateInput(state.rows() + input.rows());
  stateInput << state, input;
  const vector_t accelerationValues =
      linearAccelerationCppAdInterfacePtr_->getFunctionValue(stateInput);
  const matrix_t accelerationJacobian =
      linearAccelerationCppAdInterfacePtr_->getJacobian(stateInput);

  std::vector<VectorFunctionLinearApproximation> accelerations;
  accelerations.reserve(endEffectorIds_.size());
  for (size_t i = 0; i < endEffectorIds_.size(); ++i) {
    VectorFunctionLinearApproximation acc;
    acc.f = accelerationValues.segment<kPositionDim>(kPositionDim * i);
    acc.dfdx = accelerationJacobian.block(kPositionDim * i, 0, kPositionDim,
                                          state.rows());
    acc.dfdu = accelerationJacobian.block(kPositionDim * i, state.rows(),
                                          kPositionDim, input.rows());
    accelerations.emplace_back(std::move(acc));
  }
  return accelerations;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<PinocchioEndEffectorDynamicsCppAd::vector6_t>
PinocchioEndEffectorDynamicsCppAd::getPoseWrtPlaneRate(
    const vector_t& state,
    const vector_t& input,
    const std::vector<vector3_t>& planeNormals) const {
  return poseWrtPlaneBlocks(
      poseWrtPlaneRateCppAdInterfacePtr_->getFunctionValue(
          stateAndInput(state, input), planeNormalParameters(planeNormals)),
      endEffectorIds_.size());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<PinocchioEndEffectorDynamicsCppAd::vector6_t>
PinocchioEndEffectorDynamicsCppAd::getPoseWrtPlaneAcceleration(
    const vector_t& state,
    const vector_t& input,
    const std::vector<vector3_t>& planeNormals) const {
  return poseWrtPlaneBlocks(
      poseWrtPlaneAccelerationCppAdInterfacePtr_->getFunctionValue(
          stateAndInput(state, input), planeNormalParameters(planeNormals)),
      endEffectorIds_.size());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::getPoseWrtPlaneRateLinearApproximation(
    const vector_t& state,
    const vector_t& input,
    const std::vector<vector3_t>& planeNormals) const {
  return poseWrtPlaneLinearApproximations(
      *poseWrtPlaneRateCppAdInterfacePtr_, state, input,
      planeNormalParameters(planeNormals), endEffectorIds_.size());
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::vector<VectorFunctionLinearApproximation>
PinocchioEndEffectorDynamicsCppAd::
    getPoseWrtPlaneAccelerationLinearApproximation(
        const vector_t& state,
        const vector_t& input,
        const std::vector<vector3_t>& planeNormals) const {
  return poseWrtPlaneLinearApproximations(
      *poseWrtPlaneAccelerationCppAdInterfacePtr_, state, input,
      planeNormalParameters(planeNormals), endEffectorIds_.size());
}

}  // namespace ocs2::humanoid
