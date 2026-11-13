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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "legged_centroidal_mpc/cost/DcmTerminalCost.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PreComputation.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_core/cost/StateCost.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/ContactMask.h"
#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/common/StatusMacros.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/gait/BipedModes.h"
#include "legged_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "legged_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::legged {

namespace {
// supportWeights(2), omega, refLinvelComRtWorldEwrtWorldXy(2), offsetFactor,
// sqrtWeights(2), plannedDcmWeight, plannedDcm(2), refYawHeadingRtWorld.
constexpr size_t kNumParameters = 12;
constexpr size_t kResidualDim = 2;

// The layout of the parameters (getParameters()); each of the reference CoM
// velocity (from kSlotRefLinvelComRtWorldEwrtWorldXy), the sqrt weights and the
// planned DCM is a horizontal (x, y) pair.
constexpr int kLeftSupportWeightIndex = 0;
constexpr int kRightSupportWeightIndex = 1;
constexpr int kOmegaIndex = kRightSupportWeightIndex + 1;
constexpr int kSlotRefLinvelComRtWorldEwrtWorldXy = kOmegaIndex + 1;
constexpr int kVelocityOffsetFactorIndex =
    kSlotRefLinvelComRtWorldEwrtWorldXy + kHorizontalPositionDim;
constexpr int kSqrtWeightsStartIndex = kVelocityOffsetFactorIndex + 1;
constexpr int kPlannedDcmWeightIndex =
    kSqrtWeightsStartIndex + kHorizontalPositionDim;
constexpr int kPlannedDcmStartIndex = kPlannedDcmWeightIndex + 1;
// [rad] the yaw of the heading frame whose axes the weights act in.
constexpr int kSlotRefYawHeadingRtWorld =
    kPlannedDcmStartIndex + kHorizontalPositionDim;
static_assert(static_cast<size_t>(kSlotRefYawHeadingRtWorld + 1) ==
              kNumParameters);

// Below this total support weight no foot carries the support (a flight phase,
// or both feet at a transition).
constexpr scalar_t kMinTotalSupportWeight = 1.0e-6;

absl::Status keyMustBe(absl::string_view key,
                       scalar_t value,
                       absl::string_view requirement) {
  return absl::InvalidArgumentError(
      absl::StrCat("[DcmTerminalCost] ", DcmTerminalCost::kConfigPrefix, key,
                   " (", value, ") must be ", requirement, "."));
}

/**
 * The weighted DCM error (DcmTerminalCost::getParameters() lays out
 * `parameters`): the residual the library tapes. It computes in
 * `pinocchioInterfaceCppAd`, whose data it overwrites.
 */
ad_vector_t dcmResidual(PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                        const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                        const ad_vector_t& state,
                        const ad_vector_t& parameters) {
  const ad_scalar_t& weightLeft = parameters(kLeftSupportWeightIndex);
  const ad_scalar_t& weightRight = parameters(kRightSupportWeightIndex);
  const ad_scalar_t& omega = parameters(kOmegaIndex);
  const ad_vector2_t refLinvelComRtWorldEwrtWorldXy =
      parameters.segment<kHorizontalPositionDim>(
          kSlotRefLinvelComRtWorldEwrtWorldXy);
  const ad_scalar_t& offsetFactor = parameters(kVelocityOffsetFactorIndex);
  const ad_vector2_t sqrtWeights =
      parameters.segment<kHorizontalPositionDim>(kSqrtWeightsStartIndex);

  const PinocchioInterfaceCppAd::Model& model =
      pinocchioInterfaceCppAd.getModel();
  PinocchioInterfaceCppAd::Data& data = pinocchioInterfaceCppAd.getData();
  const ad_vector_t q = mpcRobotModelAD.getGeneralizedCoordinates(state);
  pinocchio::centerOfMass(model, data, q, /*computeSubtreeComs=*/false);
  pinocchio::updateFramePlacements(model, data);
  const ad_vector2_t posComRtWorldEwrtWorldXy =
      data.com[0].head<kHorizontalPositionDim>();
  // The CoM velocity: the centroidal model's getLinvelComRtWorldEwrtWorld().
  const ad_vector2_t linvelComRtWorldEwrtWorldXy =
      mpcRobotModelAD.getLinvelComRtWorldEwrtWorld(state)
          .head<kHorizontalPositionDim>();
  const std::vector<ad_vector3_t> posContactsRtWorldEwrtWorld =
      getPosContactsRtWorldEwrtWorld<ad_scalar_t>(pinocchioInterfaceCppAd,
                                                  mpcRobotModelAD);
  // The mean of the feet weighted by the support blend (`supportcenter`), not
  // the plain feetmean.
  const ad_vector2_t posSupportcenterRtWorldEwrtWorldXy =
      (weightLeft *
           posContactsRtWorldEwrtWorld[0].head<kHorizontalPositionDim>() +
       weightRight *
           posContactsRtWorldEwrtWorld[1].head<kHorizontalPositionDim>()) /
      (weightLeft + weightRight);

  // Where the horizon should end. Without a reduced-order plan that is the
  // center of the terminal support plus the commanded drift: "come to rest over
  // the feet". With one it is the plan's own DCM, which lies beyond the stance
  // foot towards the next foothold - referencing the support center instead
  // pulls the center of mass back over the foot, and the planner then reads a
  // state with no lateral velocity and narrows its next step until the robot
  // falls (legged_nmpc/docs/hlip_contact_planner/README.md). Blended rather
  // than branched, so the expression stays differentiable and one compiled
  // model serves both.
  const ad_scalar_t& plannedWeight = parameters(kPlannedDcmWeightIndex);
  const ad_vector2_t desPosDcmRtWorldEwrtWorldXy =
      parameters.segment<kHorizontalPositionDim>(kPlannedDcmStartIndex);

  const ad_vector2_t posDcmRtWorldEwrtWorldXy =
      posComRtWorldEwrtWorldXy + linvelComRtWorldEwrtWorldXy / omega;
  const ad_vector2_t supportRefPosDcmRtWorldEwrtWorldXy =
      posSupportcenterRtWorldEwrtWorldXy +
      offsetFactor * refLinvelComRtWorldEwrtWorldXy / omega;
  const ad_vector2_t refPosDcmRtWorldEwrtWorldXy =
      plannedWeight * desPosDcmRtWorldEwrtWorldXy +
      (ad_scalar_t(1.0) - plannedWeight) * supportRefPosDcmRtWorldEwrtWorldXy;
  // The weights are on the longitudinal and the lateral error: the error in
  // the axes of the heading frame, rotated by the heading's yaw about world z.
  const ad_vector2_t errPosDcmRtWorldEwrtWorldXy =
      posDcmRtWorldEwrtWorldXy - refPosDcmRtWorldEwrtWorldXy;
  const ad_scalar_t& refYawHeadingRtWorld =
      parameters(kSlotRefYawHeadingRtWorld);
  const ad_scalar_t cosYaw = CppAD::cos(refYawHeadingRtWorld);
  const ad_scalar_t sinYaw = CppAD::sin(refYawHeadingRtWorld);
  ad_vector2_t errPosDcmRtWorldEwrtHeadingXy;
  errPosDcmRtWorldEwrtHeadingXy << cosYaw * errPosDcmRtWorldEwrtWorldXy(0) +
                                       sinYaw * errPosDcmRtWorldEwrtWorldXy(1),
      -sinYaw * errPosDcmRtWorldEwrtWorldXy(0) +
          cosYaw * errPosDcmRtWorldEwrtWorldXy(1);
  ad_vector_t r(kResidualDim);
  r = errPosDcmRtWorldEwrtHeadingXy.cwiseProduct(sqrtWeights);
  return r;
}

}  // namespace

std::optional<scalar_t> DcmTerminalCost::Config::omega() const {
  if (!nomPosComRtFeetmeanEwrtWorldZ.has_value()) return std::nullopt;
  return std::sqrt(gravityMagnitude / *nomPosComRtFeetmeanEwrtWorldZ);
}

absl::Status DcmTerminalCost::Config::validate() const {
  if (nomPosComRtFeetmeanEwrtWorldZ.has_value() &&
      !(std::isfinite(*nomPosComRtFeetmeanEwrtWorldZ) &&
        *nomPosComRtFeetmeanEwrtWorldZ > 0.0)) {
    return keyMustBe("nom_pos_com_rt_feetmean_ewrt_world_z",
                     *nomPosComRtFeetmeanEwrtWorldZ,
                     "positive; left out, it is the model's center of mass "
                     "above its feet at initial_state");
  }
  if (!std::isfinite(weights(0)) || weights(0) < 0.0) {
    return keyMustBe("weight_x", weights(0), "non-negative");
  }
  if (!std::isfinite(weights(1)) || weights(1) < 0.0) {
    return keyMustBe("weight_y", weights(1), "non-negative");
  }
  if (!std::isfinite(velocityCommandOffsetFactor)) {
    return keyMustBe("velocity_command_offset_factor",
                     velocityCommandOffsetFactor, "finite");
  }
  if (!std::isfinite(supportBlendTime) || supportBlendTime < 0.0) {
    return keyMustBe("support_blend_time", supportBlendTime, "non-negative");
  }
  return absl::OkStatus();
}

absl::StatusOr<DcmTerminalCost::Config> DcmTerminalCost::resolveConfig(
    Config config,
    scalar_t modelNomPosComRtFeetmeanEwrtWorldZ,
    scalar_t modelGravityMagnitude) {
  config.gravityMagnitude = modelGravityMagnitude;
  RETURN_IF_ERROR(config.validate());
  if (!config.nomPosComRtFeetmeanEwrtWorldZ.has_value()) {
    if (!std::isfinite(modelNomPosComRtFeetmeanEwrtWorldZ) ||
        modelNomPosComRtFeetmeanEwrtWorldZ <= 0.0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[DcmTerminalCost] ", kConfigPrefix,
          "nom_pos_com_rt_feetmean_ewrt_world_z is left out, i.e. the model's "
          "center of mass above its "
          "feet at initial_state, but that is ",
          modelNomPosComRtFeetmeanEwrtWorldZ, " m. Give ", kConfigPrefix,
          "nom_pos_com_rt_feetmean_ewrt_world_z a positive value, or correct "
          "initial_state."));
    }
    config.nomPosComRtFeetmeanEwrtWorldZ = modelNomPosComRtFeetmeanEwrtWorldZ;
  }
  return config;
}

absl::StatusOr<std::unique_ptr<DcmTerminalCost>> DcmTerminalCost::Create(
    const SwitchedModelReferenceManager& referenceManager,
    const Config& config,
    scalar_t modelNomPosComRtFeetmeanEwrtWorldZ,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
    absl::string_view costName,
    const ModelSettings& modelSettings) {
  // The support weights blend between the left and the right foot of a
  // two-foot support (computeSupportWeights()).
  if (!modelSettings.isBipedContactSet()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[DcmTerminalCost] dcm_terminal_cost is written for two 6-DoF feet: "
        "its support blends a left and a right foot. The contact set has ",
        modelSettings.contactNames.size(), " contacts (",
        absl::StrJoin(modelSettings.contactNames, ", "), ")."));
  }
  ASSIGN_OR_RETURN(const scalar_t modelGravityMagnitude,
                   modelGravity(pinocchioInterface.getModel()));
  ASSIGN_OR_RETURN(Config resolved,
                   resolveConfig(config, modelNomPosComRtFeetmeanEwrtWorldZ,
                                 modelGravityMagnitude));
  return absl::WrapUnique(new DcmTerminalCost(
      referenceManager, std::move(resolved), modelNomPosComRtFeetmeanEwrtWorldZ,
      modelGravityMagnitude, pinocchioInterface, mpcRobotModelAD, costName,
      modelSettings));
}

DcmTerminalCost::DcmTerminalCost(
    const SwitchedModelReferenceManager& referenceManager,
    Config config,
    scalar_t modelNomPosComRtFeetmeanEwrtWorldZ,
    scalar_t modelGravityMagnitude,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
    absl::string_view costName,
    const ModelSettings& modelSettings)
    : referenceManagerPtr_(&referenceManager),
      config_(std::move(config)),
      modelNomPosComRtFeetmeanEwrtWorldZ_(modelNomPosComRtFeetmeanEwrtWorldZ),
      modelGravityMagnitude_(modelGravityMagnitude),
      mpcRobotModelAdPtr_(mpcRobotModelAD.clone()) {
  // The documented guard behind Create(), which has already resolved and
  // validated the configuration.
  CHECK(config_.validate().ok() &&
        config_.nomPosComRtFeetmeanEwrtWorldZ.has_value())
      << "[DcmTerminalCost] build it with DcmTerminalCost::Create()";
  omega_ = config_.omega().value_or(0.0);  // resolved, as checked above
  // The function holds what it reads and never this cost: every copy of the
  // library holds a copy of the function and may tape it after the cost it was
  // made of is gone (CppAdInterface::getTapeOperationCount()). What it reads is
  // shared, read-only, by the functions of this cost and of its copies - a
  // std::shared_ptr, because OCS2's CppAdInterface keeps and copies the
  // function as a std::function, which must be copyable - and each evaluation
  // computes in a copy of the Pinocchio interface.
  const std::shared_ptr<const PinocchioInterfaceCppAd> pinocchioInterfaceCppAd =
      std::make_shared<const PinocchioInterfaceCppAd>(
          pinocchioInterface.toCppAd());
  const std::shared_ptr<const MpcRobotModelBase<ad_scalar_t>> robotModelAD(
      mpcRobotModelAD.clone());
  const CppAdInterface::ad_parameterized_function_t residualAd =
      [pinocchioInterfaceCppAd, robotModelAD](
          const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
        PinocchioInterfaceCppAd pinocchioInterfaceCopy =
            *pinocchioInterfaceCppAd;
        y = dcmResidual(pinocchioInterfaceCopy, *robotModelAD, x, p);
      };
  // The generated library is keyed to the parameter count. A cached model from
  // an earlier signature has the right file name but the wrong parameter
  // dimension, and with recompileLibrariesCppAd false it would be loaded and
  // then fed the new parameter vector; the suffix makes a stale library simply
  // not exist, so it is regenerated instead.
  const std::string modelName = absl::StrCat(costName, "_p", kNumParameters);
  adInterfacePtr_ = std::make_unique<CppAdInterface>(
      residualAd, mpcRobotModelAD.getStateDim(), kNumParameters, modelName,
      modelSettings.modelFolderCppAd);
  if (modelSettings.recompileLibrariesCppAd) {
    adInterfacePtr_->createModels(CppAdInterface::ApproximationOrder::First,
                                  modelSettings.verboseCppAd);
  } else {
    adInterfacePtr_->loadModelsIfAvailable(
        CppAdInterface::ApproximationOrder::First, modelSettings.verboseCppAd);
  }
  const scalar_t nomPosComRtFeetmeanEwrtWorldZ =
      config_.nomPosComRtFeetmeanEwrtWorldZ.value_or(
          modelNomPosComRtFeetmeanEwrtWorldZ_);  // resolved, as checked above
  LOG(INFO) << "Initialized DcmTerminalCost with weights "
            << config_.weights.transpose() << ", pendulum height "
            << nomPosComRtFeetmeanEwrtWorldZ << " m"
            << (nomPosComRtFeetmeanEwrtWorldZ ==
                        modelNomPosComRtFeetmeanEwrtWorldZ_
                    ? " (the model's)"
                    : " (explicit)")
            << ", omega " << omega_ << " rad/s, velocityCommandOffsetFactor "
            << config_.velocityCommandOffsetFactor;
}

DcmTerminalCost::DcmTerminalCost(const DcmTerminalCost& other)
    : StateCost(other),
      referenceManagerPtr_(other.referenceManagerPtr_),
      config_(other.config_),
      modelNomPosComRtFeetmeanEwrtWorldZ_(
          other.modelNomPosComRtFeetmeanEwrtWorldZ_),
      modelGravityMagnitude_(other.modelGravityMagnitude_),
      omega_(other.omega_),
      isActive_(other.isActive_),
      mpcRobotModelAdPtr_(other.mpcRobotModelAdPtr_->clone()),
      adInterfacePtr_(
          std::make_unique<CppAdInterface>(*other.adInterfacePtr_)) {}

absl::Status DcmTerminalCost::setConfig(const Config& config) {
  ASSIGN_OR_RETURN(Config resolved,
                   resolveConfig(config, modelNomPosComRtFeetmeanEwrtWorldZ_,
                                 modelGravityMagnitude_));
  config_ = std::move(resolved);
  omega_ =
      config_.omega().value_or(0.0);  // resolveConfig() resolves the height
  return absl::OkStatus();
}

vector2_t DcmTerminalCost::computeSupportWeights(scalar_t time) const {
  // Written for a left and a right foot: the cost is refused for any other
  // contact set (checkCentroidalContactSet()).
  const ContactMask contacts = referenceManagerPtr_->getContactFlags(time);
  vector2_t weights(contacts[kContactLeftIndex] ? 1.0 : 0.0,
                    contacts[kContactRightIndex] ? 1.0 : 0.0);
  const ModeSchedule& schedule = referenceManagerPtr_->getModeSchedule();
  const std::vector<scalar_t>& eventTimes = schedule.eventTimes;
  const std::vector<size_t>& modeSequence = schedule.modeSequence;
  if (config_.supportBlendTime > 0.0 && !modeSequence.empty()) {
    for (size_t foot = 0; foot < static_cast<size_t>(kBipedFootCount); ++foot) {
      if (!contacts[foot]) continue;
      // Contact phase [start, end] of this foot around `time`.
      size_t index = static_cast<size_t>(
          std::upper_bound(eventTimes.begin(), eventTimes.end(), time) -
          eventTimes.begin());
      if (index >= modeSequence.size()) index = modeSequence.size() - 1;
      size_t first = index;
      while (first > 0 &&
             bipedFootContactsOfMode(modeSequence[first - 1])[foot]) {
        --first;
      }
      size_t last = index;
      while (last + 1 < modeSequence.size() &&
             bipedFootContactsOfMode(modeSequence[last + 1])[foot]) {
        ++last;
      }
      const scalar_t start = (first == 0)
                                 ? -std::numeric_limits<scalar_t>::infinity()
                                 : eventTimes[first - 1];
      const scalar_t end = (last + 1 >= modeSequence.size())
                               ? std::numeric_limits<scalar_t>::infinity()
                               : eventTimes[last];
      const scalar_t ramp = std::min((time - start) / config_.supportBlendTime,
                                     (end - time) / config_.supportBlendTime);
      weights(foot) = std::clamp(ramp, 0.0, 1.0);
    }
  }
  if (weights.sum() < kMinTotalSupportWeight) {
    weights.setOnes();  // flight, or both feet at a transition: fall back to
                        // the center of both feet
  }
  return weights;
}

vector_t DcmTerminalCost::getParameters(
    scalar_t time, const TargetTrajectories& targetTrajectories) const {
  const vector2_t supportWeights = computeSupportWeights(time);
  // Off the target that was handed in, so that the parameters stay a function
  // of this term's arguments. Under online contact planning that channel
  // carries the planned CoM velocity rather than the raw command
  // (planned_com_override), which is what the terminal DCM should be consistent
  // with: the reference then leads towards the planned foothold instead of
  // towards a straight line the gait is not following.
  // The heading the weights' longitudinal and lateral axes follow is the
  // target's at `time`, a parameter rather than a function of the state, so
  // that the Gauss-Newton Hessian keeps the error's Jacobian alone; without a
  // target it is world x.
  vector2_t refLinvelComRtWorldEwrtWorldXy = vector2_t::Zero();
  scalar_t refYawHeadingRtWorld = 0.0;
  if (!targetTrajectories.empty()) {
    const vector_t desiredState =
        mpcRobotModelAdPtr_->getDesiredState(targetTrajectories, time);
    if (desiredState.size() >= kHorizontalPositionDim) {
      refLinvelComRtWorldEwrtWorldXy =
          desiredState.head<kHorizontalPositionDim>();
    }
    if (static_cast<size_t>(desiredState.size()) ==
        mpcRobotModelAdPtr_->getStateDim()) {
      refYawHeadingRtWorld =
          mpcRobotModelAdPtr_->getYawBaseRtWorld(desiredState);
    }
  }
  // Under a plan the reference is the plan's DCM, which is only the plan's DCM
  // on the plan's own pendulum; the robot's DCM is then taken on that pendulum
  // too, so the residual compares two points of one pendulum rather than mixing
  // the plan's center of mass with this cost's omega. Both are the model's
  // height as shipped, but either key can be overridden on its own.
  const std::optional<SwitchedModelReferenceManager::PlannedDcm> planned =
      referenceManagerPtr_->getPlannedDcm(time);
  vector_t parameters(kNumParameters);
  parameters(kLeftSupportWeightIndex) = supportWeights(0);
  parameters(kRightSupportWeightIndex) = supportWeights(1);
  parameters(kOmegaIndex) = planned.has_value() ? planned->omega : omega_;
  parameters.segment<kHorizontalPositionDim>(
      kSlotRefLinvelComRtWorldEwrtWorldXy) = refLinvelComRtWorldEwrtWorldXy;
  parameters(kVelocityOffsetFactorIndex) = config_.velocityCommandOffsetFactor;
  parameters.segment<kHorizontalPositionDim>(kSqrtWeightsStartIndex) =
      config_.weights.cwiseSqrt();
  parameters(kPlannedDcmWeightIndex) = planned.has_value() ? 1.0 : 0.0;
  parameters.segment<kHorizontalPositionDim>(kPlannedDcmStartIndex) =
      planned.has_value() ? planned->desPosDcmRtWorldEwrtWorldXy
                          : vector2_t::Zero();
  parameters(kSlotRefYawHeadingRtWorld) = refYawHeadingRtWorld;
  return parameters;
}

vector2_t DcmTerminalCost::computeErrPosDcmRtWorldEwrtWorldXy(
    const vector_t& state, const vector_t& parameters) const {
  vector_t unweighted = parameters;
  unweighted.segment<kHorizontalPositionDim>(kSqrtWeightsStartIndex).setOnes();
  unweighted(kSlotRefYawHeadingRtWorld) = 0.0;  // world axes
  return adInterfacePtr_->getFunctionValue(state, unweighted)
      .head<kHorizontalPositionDim>();
}

matrix_t DcmTerminalCost::computeJacobianErrPosDcmRtWorldEwrtWorldXy(
    const vector_t& state, const vector_t& parameters) const {
  vector_t unweighted = parameters;
  unweighted.segment<kHorizontalPositionDim>(kSqrtWeightsStartIndex).setOnes();
  unweighted(kSlotRefYawHeadingRtWorld) = 0.0;  // world axes
  return adInterfacePtr_->getJacobian(state, unweighted)
      .topRows<kHorizontalPositionDim>();
}

scalar_t DcmTerminalCost::getValue(
    scalar_t time,
    const vector_t& state,
    const TargetTrajectories& targetTrajectories,
    const PreComputation& /*preComputation*/) const {
  const vector_t parameters = getParameters(time, targetTrajectories);
  const vector_t r = adInterfacePtr_->getFunctionValue(state, parameters);
  return 0.5 * r.squaredNorm();
}

ScalarFunctionQuadraticApproximation DcmTerminalCost::getQuadraticApproximation(
    scalar_t time,
    const vector_t& state,
    const TargetTrajectories& targetTrajectories,
    const PreComputation& /*preComputation*/) const {
  const vector_t parameters = getParameters(time, targetTrajectories);
  return adInterfacePtr_->getGaussNewtonApproximation(state, parameters);
}

}  // namespace ocs2::legged
