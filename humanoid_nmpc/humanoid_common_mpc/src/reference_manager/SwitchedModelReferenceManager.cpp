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

#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/misc/Numerics.h"
#include "ocs2_oc/synchronized_module/ReferenceManager.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/multibody/fwd.hpp"

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact/ContactPatchFrame.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_common_mpc/gait/GaitSchedule.h"
#include "humanoid_common_mpc/gait/LegSwingPhase.h"
#include "humanoid_common_mpc/locomotion_heuristics/BasePoseHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/FootholdHeuristic.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicModelParameters.h"
#include "humanoid_common_mpc/locomotion_heuristics/WrenchHeuristic.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc/reference_manager/ArmSwing.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/arm_swing_config.nproto.h"

namespace ocs2::humanoid {
namespace {

/** The gait phase at which right stance (RF) begins: [0, 0.5) is left stance,
 * [0.5, 1) right stance (getPhaseVariable()). */
constexpr scalar_t kRightStancePhaseStart = 0.5;
/** [s] How far before a phase's first event getPhaseVariable() reads the mode
 * of the phase before it. */
constexpr scalar_t kPhaseVariableModeLookBack = 0.01;

/** [s] How far before a touch-down the nominal step looks for the swing that
 * ends at it. */
constexpr scalar_t kNominalStepSwingLookBack = 1.0e-9;

/** The procedural arm swing: sin(kArmSwingRadiansPerCycle (phase -
 * phaseOffset)) v_x, times the gains (ArmSwingSettings). */
constexpr scalar_t kArmSwingRadiansPerCycle = 2 * std::numbers::pi;

/** Whether a state of the layout `layout` stores the base's body angular
 * velocity (a kZyxOrderedAngularVelocity block). */
bool storesAngvelBaseRtWorldEwrtBase(const TuningLayout& layout) {
  for (const TuningSegment& segment : layout.segments) {
    if (segment.kind == TuningSegmentKind::kZyxOrderedAngularVelocity) {
      return true;
    }
  }
  return false;
}

/** The tuning block of the centroidal state that holds the normalized angular
 * momentum L / m (tuningLayoutOf()). */
constexpr char kMnormAngmomSegmentName[] =
    "mnorm_angmom_com_rt_world_ewrt_world";

/** The first state row of the normalized angular momentum L / m, for a state
 * that stores it (the centroidal one). */
std::optional<size_t> mnormAngmomStartIndex(const TuningLayout& layout) {
  for (const TuningSegment& segment : layout.segments) {
    if (segment.name == kMnormAngmomSegmentName) return segment.stateStart;
  }
  return std::nullopt;
}

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
SwitchedModelReferenceManager::SwitchedModelReferenceManager(
    std::shared_ptr<GaitSchedule> gaitSchedulePtr,
    std::unique_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
    const PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel)
    : ReferenceManager(TargetTrajectories(), ModeSchedule()),
      // The reference manager gets a copy of the pinocchio model to use for
      // initializing the ground height
      pinocchioInterface_(pinocchioInterface),
      mpcRobotModelPtr_(&mpcRobotModel),
      numContacts_(mpcRobotModel.modelSettings.contactNames.size()),
      liftoffPosContactsRtWorldEwrtWorldXy_(numContacts_, vector2_t::Zero()),
      gaitSchedulePtr_(std::move(gaitSchedulePtr)),
      swingTrajectoryPtr_(std::move(swingTrajectoryPtr)) {
  ABSL_CHECK(swingTrajectoryPtr_ != nullptr)
      << "[SwitchedModelReferenceManager] needs a swing trajectory planner";
  // A constant of the robot, so it is read once here rather than being latched
  // with the measurements. The yaw inertia beside it is not a constant - it
  // depends on the posture - and is latched per solve in
  // captureMeasuredState().
  totalMass_ = pinocchio::computeTotalMass(pinocchioInterface_.getModel());
  // The ground starts where the task file puts it; from here on it is this
  // manager's (setPosSurfaceRtWorldEwrtWorldZ).
  posSurfaceRtWorldEwrtWorldZ_.store(
      mpcRobotModel.modelSettings.posSurfaceRtWorldEwrtWorldZ);
  appliedPosSurfaceRtWorldEwrtWorldZ_.store(
      mpcRobotModel.modelSettings.posSurfaceRtWorldEwrtWorldZ);
  refPosSurfaceRtWorldEwrtWorldZ_ =
      mpcRobotModel.modelSettings.posSurfaceRtWorldEwrtWorldZ;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void SwitchedModelReferenceManager::reset() {
  ReferenceManager::reset();
  // The ground is configuration and stays where it was last set. A target
  // installed after the reset is taken from the measured state, i.e. it is
  // already a world pose on that ground, so there is no change of ground left
  // to apply.
  const scalar_t posSurfaceRtWorldEwrtWorldZ =
      posSurfaceRtWorldEwrtWorldZ_.load();
  appliedPosSurfaceRtWorldEwrtWorldZ_.store(posSurfaceRtWorldEwrtWorldZ);
  refPosSurfaceRtWorldEwrtWorldZ_ = posSurfaceRtWorldEwrtWorldZ;
  resetRuntimeState();
}

void SwitchedModelReferenceManager::resetRuntimeState() {
  gaitSchedulePtr_->reset();
  modeSchedule_ = ModeSchedule();

  hasMeasuredState_ = false;
  lastSolveTime_ = 0.0;
  measPosBaseRtWorldEwrtWorldXy_.setZero();
  measYawBaseRtWorld_ = 0.0;
  liftoffPosContactsRtWorldEwrtWorldXy_.assign(numContacts_, vector2_t::Zero());
  measProxyLinvelComRtWorldEwrtWorldXy_.setZero();
  measPosComRtFeetmeanEwrtWorldZ_ = 0.0;
  inertiaComEwrtWorldZz_ = 0.0;
  inertiaComEwrtHeading_.setZero();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ContactMask SwitchedModelReferenceManager::getContactFlags(
    scalar_t time) const {
  return ContactMask::fromModeNumber(this->getModeSchedule().modeAtTime(time),
                                     numContacts_);
}

ContactPatchFrame SwitchedModelReferenceManager::getContactPatchFrame(
    size_t /*contactIndex*/, scalar_t /*time*/) const {
  return ContactPatchFrame{.rotmatWorldRtSurface = matrix3_t::Identity(),
                           .posSurfaceRtWorldEwrtSurfaceZ =
                               appliedPosSurfaceRtWorldEwrtWorldZ_.load()};
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwitchedModelReferenceManager::getPhaseVariable(scalar_t time) const {
  const std::vector<scalar_t>::const_iterator it = std::upper_bound(
      modeSchedule_.eventTimes.begin(), modeSchedule_.eventTimes.end(), time);
  // `time` outside the scheduled events, or a schedule with fewer than two of
  // them, leaves nothing to interpolate between: upper_bound returns end() past
  // the last event and begin() before the first, and dereferencing either was
  // undefined. This was reachable before only through the procedural arm swing,
  // which no legs-only robot runs; the base-pose heuristics are a second caller
  // and every robot can reach them, so the degenerate cases are answered rather
  // than stepped into. 0 is the start of a cycle, which is what a phase that
  // cannot be computed should be.
  if (it == modeSchedule_.eventTimes.end() ||
      it == modeSchedule_.eventTimes.begin()) {
    return 0.0;
  }
  const scalar_t nextEventTime = *it;
  const scalar_t prevEventTime = *(it - 1);
  if (!(nextEventTime > prevEventTime)) return 0.0;

  if (modeSchedule_.modeAtTime(time) == kBipedModeLf) {
    return 0.5 * (time - prevEventTime) / (nextEventTime - prevEventTime);
  } else if (modeSchedule_.modeAtTime(time) == kBipedModeRf) {
    return kRightStancePhaseStart +
           0.5 * (time - prevEventTime) / (nextEventTime - prevEventTime);
  } else {
    if (modeSchedule_.modeAtTime(prevEventTime - kPhaseVariableModeLookBack) ==
        kBipedModeLf) {
      return kRightStancePhaseStart;
    } else {
      return 0;
    }
  }
}

scalar_t SwitchedModelReferenceManager::getPhaseVariableRate(
    scalar_t time) const {
  // The derivative of getPhaseVariable(), case for case: 0.5 / (next - prev)
  // inside a single-support mode, whose half of the cycle the phase sweeps
  // linearly, and 0 wherever that function returns a constant - double support,
  // a degenerate schedule and a time outside the events.
  const std::vector<scalar_t>::const_iterator it = std::upper_bound(
      modeSchedule_.eventTimes.begin(), modeSchedule_.eventTimes.end(), time);
  if (it == modeSchedule_.eventTimes.end() ||
      it == modeSchedule_.eventTimes.begin()) {
    return 0.0;
  }
  const scalar_t nextEventTime = *it;
  const scalar_t prevEventTime = *(it - 1);
  if (!(nextEventTime > prevEventTime)) return 0.0;
  const size_t mode = modeSchedule_.modeAtTime(time);
  if (mode == kBipedModeLf || mode == kBipedModeRf) {
    return 0.5 / (nextEventTime - prevEventTime);
  }
  return 0.0;
}

scalar_t SwitchedModelReferenceManager::getSwingPhaseVariable(
    size_t legIndex, scalar_t time) const {
  const std::vector<ModelSettings::Leg>& legs =
      mpcRobotModelPtr_->modelSettings.legs;
  if (legIndex >= legs.size()) return 0.0;
  return legSwingPhaseVariable(this->getModeSchedule(), numContacts_,
                               legs[legIndex].contactIndices, time);
}

scalar_t SwitchedModelReferenceManager::getSwingPhaseVariableRate(
    size_t legIndex, scalar_t time) const {
  const std::vector<ModelSettings::Leg>& legs =
      mpcRobotModelPtr_->modelSettings.legs;
  if (legIndex >= legs.size()) return 0.0;
  return legSwingPhaseVariableRate(this->getModeSchedule(), numContacts_,
                                   legs[legIndex].contactIndices, time);
}

void SwitchedModelReferenceManager::configureNominalStance(
    const vector_t& nominalState) {
  PinocchioInterface pinocchioInterface = pinocchioInterface_;
  nomPosContactsRtBaseEwrtHeadingXy_ = computeNomPosContactsRtBaseEwrtHeadingXy(
      pinocchioInterface, *mpcRobotModelPtr_, nominalState);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector3_t SwitchedModelReferenceManager::getSwingRefNormalSurfaceEwrtWorld(
    size_t contactIndex, scalar_t time) const {
  // A flat foot's normal is the world z axis.
  if (isInContact(time, contactIndex)) return vector3_t::UnitZ();

  const scalar_t swingRefTiltangleFootRtSurface =
      swingTrajectoryPtr_->getSwingRefTiltangleFootRtSurface(contactIndex,
                                                             time);
  if (numerics::almost_eq(swingRefTiltangleFootRtSurface, /*y=*/0.0)) {
    return vector3_t::UnitZ();
  }

  // The yaw of the swing plane: the planner's ZYX foot yaw
  // (desEulerzyxFootRtWorldYaw, SwingFootReference::desYawFootRtWorld)
  // when the contact planner's heading model provides one, otherwise the
  // reference base heading (refYawBaseRtWorld), so that the tilt stays about
  // the foot's lateral axis and does not leak into roll.
  scalar_t swingRefYawFootRtWorld = 0.0;
  const std::optional<SwingFootReference> swingReference =
      getSwingFootReference(contactIndex, time);
  if (swingReference.has_value() &&
      swingReference->desYawFootRtWorld.has_value()) {
    swingRefYawFootRtWorld = *swingReference->desYawFootRtWorld;
  } else if (const TargetTrajectories& targetTrajectories =
                 getTargetTrajectories();
             !targetTrajectories.empty()) {
    const vector_t desiredState =
        mpcRobotModelPtr_->getDesiredState(targetTrajectories, time);
    if (static_cast<size_t>(desiredState.size()) ==
        mpcRobotModelPtr_->getStateDim()) {
      swingRefYawFootRtWorld =
          mpcRobotModelPtr_->getYawBaseRtWorld(desiredState);
    }
  }

  // Toe-up by swingRefTiltangleFootRtSurface is a rotation of
  // -swingRefTiltangleFootRtSurface about the foot's lateral (+y)
  // axis, which tilts the sole normal backwards along the heading. The result
  // is a unit vector, as rotationMatrixDistanceToPlane expects.
  const scalar_t sp = std::sin(swingRefTiltangleFootRtSurface);
  return vector3_t(-sp * std::cos(swingRefYawFootRtWorld),
                   -sp * std::sin(swingRefYawFootRtWorld),
                   std::cos(swingRefTiltangleFootRtSurface));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::optional<vector2_t>
SwitchedModelReferenceManager::getSwingRefLinvelFootRtWorldEwrtWorldXy(
    size_t contactIndex, scalar_t time) const {
  if (isInContact(time, contactIndex)) return std::nullopt;

  return getCmdLinvelBaseRtWorldEwrtWorldXy(time);
}

absl::Status SwitchedModelReferenceManager::checkArmSwingContactSet(
    const ArmSwingSettings& settings) const {
  const ModelSettings& modelSettings = mpcRobotModelPtr_->modelSettings;
  if (!armSwingReferenceActive_ ||
      settings.generator != ArmSwingGenerator::kProcedural ||
      modelSettings.isBipedContactSet()) {
    return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "[ArmSwing] arm_swing.generator '",
      armSwingGeneratorName(settings.generator),
      "' is written for two 6-DoF feet: it swings the arms against the "
      "biped's gait phase. The contact set has ",
      modelSettings.contactNames.size(), " contacts (",
      absl::StrJoin(modelSettings.contactNames, ", "),
      "). Set arm_swing.generator 'none', or leave out "
      "model_settings.arm_joint_names."));
}

absl::Status SwitchedModelReferenceManager::setArmSwingSettings(
    const ArmSwingSettings& settings) {
  RETURN_IF_ERROR(checkArmSwingContactSet(settings));
  armSwingSettings_ = settings;
  return absl::OkStatus();
}

absl::Status SwitchedModelReferenceManager::configureArmSwing(
    const mpc_config::ArmSwingConfig& config, const vector_t& nominalState) {
  ASSIGN_OR_RETURN(const ArmSwingSettings settings,
                   armSwingSettingsFromConfig(config));
  RETURN_IF_ERROR(checkArmSwingContactSet(settings));
  if (mpcRobotModelPtr_->modelSettings.hasArmSwingJoints) {
    ASSIGN_OR_RETURN(
        armSwingDirections_,
        deriveArmSwingDirections(pinocchioInterface_, *mpcRobotModelPtr_,
                                 nominalState));
  }
  armSwingSettings_ = settings;
  return absl::OkStatus();
}

void SwitchedModelReferenceManager::setLocomotionHeuristicLayer(
    std::shared_ptr<LocomotionHeuristicLayer> layer) {
  if (layer != nullptr) heuristicLayerPtr_ = std::move(layer);
}

bool SwitchedModelReferenceManager::needsMeasuredState() const {
  // Every listed heuristic wants something from here, not only the foothold
  // ones: the base-pose and wrench families read the commanded yaw rate, which
  // is recovered by inverting the target's momentum channel with the composite
  // yaw inertia this function's caller latches.
  const ModelSettings::NominalFootholdConfig& nominalFoothold =
      mpcRobotModelPtr_->modelSettings.nominalFootholdConfig;
  return nominalFoothold.nomPosLfootRtRfootEwrtHeadingY > 0.0 ||
         nominalFoothold.anchor ==
             ModelSettings::NominalFootholdAnchor::kNominalStance ||
         !heuristicLayerPtr_->empty();
}

SwitchedModelReferenceManager::PlanarPoseChange
SwitchedModelReferenceManager::integratePlanarCommand(
    const PlanarCommandAt& start,
    const PlanarCommandAt& end,
    scalar_t duration) {
  const scalar_t meanCmdRateYawBaseRtWorld =
      0.5 * (start.cmdRateYawBaseRtWorld + end.cmdRateYawBaseRtWorld);
  const scalar_t errYawBaseRtWorld = meanCmdRateYawBaseRtWorld * duration;
  // R_z(-theta) v_1: the end velocity in the frame that has turned with the
  // command, where a constant body twist keeps it.
  const scalar_t cosine = std::cos(errYawBaseRtWorld);
  const scalar_t sine = std::sin(errYawBaseRtWorld);
  const vector2_t& finalCmdLinvelBaseRtWorldEwrtWorldXy =
      end.cmdLinvelBaseRtWorldEwrtWorldXy;
  const vector2_t turnedFinalCmdLinvelBaseRtWorldEwrtWorldXy(
      cosine * finalCmdLinvelBaseRtWorldEwrtWorldXy(0) +
          sine * finalCmdLinvelBaseRtWorldEwrtWorldXy(1),
      -sine * finalCmdLinvelBaseRtWorldEwrtWorldXy(0) +
          cosine * finalCmdLinvelBaseRtWorldEwrtWorldXy(1));
  const vector2_t meanCmdLinvelBaseRtWorldEwrtWorldXy =
      0.5 * (start.cmdLinvelBaseRtWorldEwrtWorldXy +
             turnedFinalCmdLinvelBaseRtWorldEwrtWorldXy);
  return PlanarPoseChange{
      .errPosBaseRtWorldEwrtWorldXy =
          TargetTrajectoriesCalculatorBase::computeErrPosBaseRtWorldEwrtWorldXy(
              meanCmdLinvelBaseRtWorldEwrtWorldXy, meanCmdRateYawBaseRtWorld,
              duration),
      .errYawBaseRtWorld = errYawBaseRtWorld};
}

SwitchedModelReferenceManager::PlanarBasePose
SwitchedModelReferenceManager::predictedBaseAt(scalar_t time) const {
  // The measurement carried forward by the command over the time until `time`
  // (integratePlanarCommand()). Only the foothold heuristics read the position
  // (footholdContext()), and no shipped robot lists one; the nominal step reads
  // the yaw.
  const scalar_t elapsed = std::max(time - lastSolveTime_, 0.0);
  const PlanarPoseChange change = integratePlanarCommand(
      PlanarCommandAt{
          .cmdLinvelBaseRtWorldEwrtWorldXy =
              getCmdLinvelBaseRtWorldEwrtWorldXy(lastSolveTime_),
          .cmdRateYawBaseRtWorld = getCmdRateYawBaseRtWorld(lastSolveTime_)},
      PlanarCommandAt{.cmdLinvelBaseRtWorldEwrtWorldXy =
                          getCmdLinvelBaseRtWorldEwrtWorldXy(time),
                      .cmdRateYawBaseRtWorld = getCmdRateYawBaseRtWorld(time)},
      elapsed);
  return PlanarBasePose{
      .posBaseRtWorldEwrtWorldXy = vector2_t(
          measPosBaseRtWorldEwrtWorldXy_ + change.errPosBaseRtWorldEwrtWorldXy),
      .yawBaseRtWorld = measYawBaseRtWorld_ + change.errYawBaseRtWorld};
}

FootholdHeuristicContext SwitchedModelReferenceManager::footholdContext(
    size_t contactIndex, scalar_t time) const {
  const ModelSettings& modelSettings = mpcRobotModelPtr_->modelSettings;
  FootholdHeuristicContext context;
  context.contactIndex = contactIndex;
  // The leg of the landing contact. A set without legs is the biped's (the
  // only set without legs that reaches here, see
  // getNomPosFootRtWorldEwrtWorldXy()), whose contact index is its leg index
  // and whose left foot is contact 0.
  const std::optional<size_t> leg = modelSettings.legOfContact(contactIndex);
  context.legIndex = leg.value_or(contactIndex);
  context.side = leg.has_value() ? modelSettings.legs[*leg].lateralSign
                 : (contactIndex == kContactLeftIndex) ? 1.0
                                                       : -1.0;
  if (contactIndex < nomPosContactsRtBaseEwrtHeadingXy_.size()) {
    context.nomPosContactRtBaseEwrtHeadingXy =
        nomPosContactsRtBaseEwrtHeadingXy_[contactIndex];
  }
  const PlanarBasePose predicted = predictedBaseAt(time);
  context.posBaseRtWorldEwrtWorldXy = predicted.posBaseRtWorldEwrtWorldXy;
  context.yawBaseRtWorld = predicted.yawBaseRtWorld;
  context.measProxyLinvelComRtWorldEwrtWorldXy =
      measProxyLinvelComRtWorldEwrtWorldXy_;
  context.prevCmdLinvelBaseRtWorldEwrtWorldXy =
      getCmdLinvelBaseRtWorldEwrtWorldXy(lastSolveTime_);
  context.cmdLinvelBaseRtWorldEwrtWorldXy =
      getCmdLinvelBaseRtWorldEwrtWorldXy(time);
  context.cmdRateYawBaseRtWorld = getCmdRateYawBaseRtWorld(time);
  context.measPosComRtFeetmeanEwrtWorldZ = measPosComRtFeetmeanEwrtWorldZ_;
  // The biped's feet keep the schedule queries written for them; a leg of any
  // other set is in stance while any of its contacts is.
  context.stanceDuration =
      modelSettings.isBipedContactSet() || !leg.has_value()
          ? upcomingStanceDuration(modeSchedule_, contactIndex, time)
          : legUpcomingStanceDuration(modeSchedule_, numContacts_,
                                      modelSettings.legs[*leg].contactIndices,
                                      time);
  return context;
}

std::optional<vector2_t>
SwitchedModelReferenceManager::getNomPosFootRtWorldEwrtWorldXy(
    size_t contactIndex, scalar_t time) const {
  // The step_width anchor places a left and a right foot (kContactLeftIndex,
  // kContactRightIndex) a step width apart; another contact set has no
  // foothold on it rather than reading its first two contacts as feet. The
  // nominal_stance anchor places every contact of a legged set.
  // footholdContext() is reached only from below this guard.
  const ModelSettings& modelSettings = mpcRobotModelPtr_->modelSettings;
  if (modelSettings.nominalFootholdConfig.anchor ==
      ModelSettings::NominalFootholdAnchor::kNominalStance) {
    if (!modelSettings.isLeggedContactSet()) return std::nullopt;
    return footholdOnNominalStance(contactIndex, time);
  }
  if (!modelSettings.isBipedContactSet()) {
    return std::nullopt;
  }
  const scalar_t nomPosLfootRtRfootEwrtHeadingY =
      mpcRobotModelPtr_->modelSettings.nominalFootholdConfig
          .nomPosLfootRtRfootEwrtHeadingY;
  // Nothing has been measured yet, so nothing can be placed: this is the first
  // solve, before captureMeasuredState().
  if (!hasMeasuredState_) return std::nullopt;
  const bool haveHeuristics = !heuristicLayerPtr_->footholdEmpty();
  // No opinion from either source: a robot with
  // `nom_pos_lfoot_rt_rfoot_ewrt_heading_y: 0` and no listed heuristic returns
  // nullopt, and the foot cost's xy weights stay switched off exactly as they
  // were.
  if (nomPosLfootRtRfootEwrtHeadingY <= 0.0 && !haveHeuristics) {
    return std::nullopt;
  }

  if (!haveHeuristics) {
    return nomStepPosFootRtWorldEwrtWorldXy(contactIndex, time);
  }
  // BLEDT'S H_r: the hip at the moment of landing, plus the listed heuristics'
  // offsets. See the header.
  const FootholdHeuristicContext context = footholdContext(contactIndex, time);
  const vector2_t offsetPosFootRtWorldEwrtWorldXy =
      heuristicLayerPtr_->computeOffsetPosFootRtWorldEwrtWorldXy(context);
  if (heuristicLayerPtr_->footholdMovesAnchor()) {
    // hip_centered_stepping is listed: its offset is the hip relative to the
    // predicted base, so the base is the anchor.
    return vector2_t(context.posBaseRtWorldEwrtWorldXy +
                     offsetPosFootRtWorldEwrtWorldXy);
  }
  // Otherwise the hip is half a step width to this foot's side of the predicted
  // base. Create() refuses a zero step width here. (It used to be the STANCE
  // FOOT plus a whole step width across the heading, the predicted base along
  // it: the stance foot is where the body WAS when that foot landed, so under a
  // lateral command the feet fell behind the base by v_y T_step - k T_st v_y at
  // every step, k T_st v_y being translational_stepping's lead, and the stance
  // drifted away from the body without bound. See
  // computePosHipRtWorldEwrtWorldXy().)
  return vector2_t(
      computePosHipRtWorldEwrtWorldXy(context.posBaseRtWorldEwrtWorldXy,
                                      context.yawBaseRtWorld, context.side,
                                      nomPosLfootRtRfootEwrtHeadingY) +
      offsetPosFootRtWorldEwrtWorldXy);
}

std::optional<vector2_t>
SwitchedModelReferenceManager::nomStepPosFootRtWorldEwrtWorldXy(
    size_t contactIndex, scalar_t time) const {
  const size_t stanceIndex = (contactIndex == kContactLeftIndex)
                                 ? kContactRightIndex
                                 : kContactLeftIndex;
  const scalar_t side = (contactIndex == kContactLeftIndex) ? 1.0 : -1.0;
  const scalar_t nomPosLfootRtRfootEwrtHeadingY =
      mpcRobotModelPtr_->modelSettings.nominalFootholdConfig
          .nomPosLfootRtRfootEwrtHeadingY;
  // A stance foot that lands at the same instant as this one is not a landmark
  // for this step, and takes the latched position; that also bounds the
  // recursion, since each nested call asks about a strictly earlier touch-down.
  const std::optional<scalar_t> stanceLanded =
      previousTouchDownTime(modeSchedule_, stanceIndex, time);
  const std::optional<vector2_t> anchorPosStanceRtWorldEwrtWorldXy =
      anchorPosFootRtWorldEwrtWorldXy(
          stanceIndex, (stanceLanded.has_value() && *stanceLanded < time)
                           ? stanceLanded
                           : std::nullopt);
  if (!anchorPosStanceRtWorldEwrtWorldXy.has_value()) return std::nullopt;
  const std::optional<SwingWindow> swing = swingPhaseAtTime(
      modeSchedule_, contactIndex, time - kNominalStepSwingLookBack);
  const scalar_t stepStart =
      stanceLanded.has_value()
          ? *stanceLanded
          : (swing.has_value() ? swing->liftOff : lastSolveTime_);
  const scalar_t predictedYawBaseRtWorld = predictedBaseAt(time).yawBaseRtWorld;
  const vector2_t advanceErrPosFootRtWorldEwrtWorldXy =
      getCmdLinvelBaseRtWorldEwrtWorldXy(time) *
      std::max(time - stepStart, 0.0);
  const vector2_t lateralErrPosFootRtWorldEwrtWorldXy(
      -std::sin(predictedYawBaseRtWorld) * side *
          nomPosLfootRtRfootEwrtHeadingY,
      std::cos(predictedYawBaseRtWorld) * side *
          nomPosLfootRtRfootEwrtHeadingY);
  return vector2_t(*anchorPosStanceRtWorldEwrtWorldXy +
                   advanceErrPosFootRtWorldEwrtWorldXy +
                   lateralErrPosFootRtWorldEwrtWorldXy);
}

std::optional<vector2_t>
SwitchedModelReferenceManager::anchorPosFootRtWorldEwrtWorldXy(
    size_t contactIndex, std::optional<scalar_t> touchDownTime) const {
  if (!touchDownTime.has_value() || *touchDownTime <= lastSolveTime_) {
    return liftoffPosContactsRtWorldEwrtWorldXy_[contactIndex];
  }
  return getNomPosFootRtWorldEwrtWorldXy(contactIndex, *touchDownTime);
}

vector2_t SwitchedModelReferenceManager::computePosHipRtWorldEwrtWorldXy(
    const vector2_t& predictedPosBaseRtWorldEwrtWorldXy,
    scalar_t predictedYawBaseRtWorld,
    scalar_t side,
    scalar_t nomPosLfootRtRfootEwrtHeadingY) {
  const vector2_t leftDirHeadingEwrtWorldXy(-std::sin(predictedYawBaseRtWorld),
                                            std::cos(predictedYawBaseRtWorld));
  return vector2_t(predictedPosBaseRtWorldEwrtWorldXy +
                   leftDirHeadingEwrtWorldXy *
                       (0.5 * side * nomPosLfootRtRfootEwrtHeadingY));
}

std::optional<vector2_t> SwitchedModelReferenceManager::footholdOnNominalStance(
    size_t contactIndex, scalar_t time) const {
  // Nothing measured yet (the first solve), or no nominal posture to stand in.
  if (!hasMeasuredState_ ||
      contactIndex >= nomPosContactsRtBaseEwrtHeadingXy_.size()) {
    return std::nullopt;
  }
  const FootholdHeuristicContext context = footholdContext(contactIndex, time);
  vector2_t nomPosContactRtWorldEwrtWorldXy =
      computeNomPosContactRtWorldEwrtWorldXy(context);
  if (heuristicLayerPtr_->footholdEmpty()) {
    return nomPosContactRtWorldEwrtWorldXy;
  }
  const vector2_t offsetPosFootRtWorldEwrtWorldXy =
      heuristicLayerPtr_->computeOffsetPosFootRtWorldEwrtWorldXy(context);
  if (heuristicLayerPtr_->footholdMovesAnchor()) {
    // hip_centered_stepping is listed: its offset is the hip relative to the
    // predicted base, so the base is the anchor.
    return vector2_t(context.posBaseRtWorldEwrtWorldXy +
                     offsetPosFootRtWorldEwrtWorldXy);
  }
  return vector2_t(nomPosContactRtWorldEwrtWorldXy +
                   offsetPosFootRtWorldEwrtWorldXy);
}

vector2_t SwitchedModelReferenceManager::computeNomPosContactRtWorldEwrtWorldXy(
    const FootholdHeuristicContext& context) {
  const scalar_t cosYaw = std::cos(context.yawBaseRtWorld);
  const scalar_t sinYaw = std::sin(context.yawBaseRtWorld);
  const vector2_t& nomPosContactRtBaseEwrtHeadingXy =
      context.nomPosContactRtBaseEwrtHeadingXy;
  return vector2_t(context.posBaseRtWorldEwrtWorldXy.x() +
                       cosYaw * nomPosContactRtBaseEwrtHeadingXy.x() -
                       sinYaw * nomPosContactRtBaseEwrtHeadingXy.y(),
                   context.posBaseRtWorldEwrtWorldXy.y() +
                       sinYaw * nomPosContactRtBaseEwrtHeadingXy.x() +
                       cosYaw * nomPosContactRtBaseEwrtHeadingXy.y());
}

std::optional<SwingFootReference>
SwitchedModelReferenceManager::getSwingFootReference(size_t contactIndex,
                                                     scalar_t time) const {
  if (isInContact(time, contactIndex)) return std::nullopt;
  // The biped's feet keep the schedule queries written for them; a contact of
  // any other set is its own one-contact leg (LegSwingPhase.h).
  const bool bipedContactSet =
      mpcRobotModelPtr_->modelSettings.isBipedContactSet();
  const std::array<size_t, 1> contact = {contactIndex};
  std::optional<SwingWindow> phase;
  if (bipedContactSet) {
    phase = swingPhaseAtTime(modeSchedule_, contactIndex, time);
  } else if (const std::optional<LegSwingWindow> window = legSwingWindowAtTime(
                 modeSchedule_, numContacts_, contact, time);
             window.has_value()) {
    phase =
        SwingWindow{.liftOff = window->liftOff, .touchDown = window->touchDown};
  }
  if (!phase.has_value()) return std::nullopt;
  const scalar_t liftOffTime = phase->liftOff;
  const scalar_t touchDownTime = phase->touchDown;
  const scalar_t duration = touchDownTime - liftOffTime;
  if (duration <= kMinSwingReferenceDuration) return std::nullopt;

  // The swing starts where the foot actually lifted off, not where the nominal
  // offset would have put it. Blending between two nominal points describes a
  // path the foot is not on, and the step the cost then demands at lift-off is
  // whatever error had accumulated by then. That holds for a swing whose
  // previous touch-down the last solve saw; for a later swing in the horizon,
  // whose foot has not yet landed from the swing before, the latched position
  // is where it lifted off from one step earlier, and the swing starts from
  // that earlier step's nominal target instead
  // (anchorPosFootRtWorldEwrtWorldXy()).
  const std::optional<vector2_t> desPosFootRtWorldEwrtWorldXy =
      getNomPosFootRtWorldEwrtWorldXy(contactIndex, touchDownTime);
  if (!desPosFootRtWorldEwrtWorldXy.has_value()) return std::nullopt;
  const std::optional<vector2_t> liftoffPosFootRtWorldEwrtWorldXy =
      anchorPosFootRtWorldEwrtWorldXy(
          contactIndex,
          bipedContactSet
              ? previousTouchDownTime(modeSchedule_, contactIndex, liftOffTime)
              : legPreviousTouchDownTime(modeSchedule_, numContacts_, contact,
                                         liftOffTime));
  if (!liftoffPosFootRtWorldEwrtWorldXy.has_value()) return std::nullopt;

  // The same cubic blend the planned-foothold path uses: p'(0) = 1, p'(1) = 0,
  // so the foot leaves the ground moving with the step and settles onto the
  // target instead of arriving at speed.
  const scalar_t tau = std::clamp((time - liftOffTime) / duration, 0.0, 1.0);
  const scalar_t tau2 = tau * tau;
  const scalar_t blend = -tau2 * tau + tau2 + tau;
  const scalar_t blendRate =
      (-kSwingBlendCubicDegree * tau2 + 2.0 * tau + 1.0) / duration;
  const vector2_t errPosFootRtWorldEwrtWorldXy =
      *desPosFootRtWorldEwrtWorldXy - *liftoffPosFootRtWorldEwrtWorldXy;

  SwingFootReference reference;
  reference.desPosFootRtWorldEwrtWorld.head<kHorizontalPositionDim>() =
      *liftoffPosFootRtWorldEwrtWorldXy + blend * errPosFootRtWorldEwrtWorldXy;
  reference.desPosFootRtWorldEwrtWorld(kZAxisIndex) =
      swingTrajectoryPtr_->getDesPosFootRtWorldEwrtWorldZ(contactIndex, time);
  reference.desLinvelFootRtWorldEwrtWorld.head<kHorizontalPositionDim>() =
      blendRate * errPosFootRtWorldEwrtWorldXy;
  reference.desLinvelFootRtWorldEwrtWorld(kZAxisIndex) =
      swingTrajectoryPtr_->getDesLinvelFootRtWorldEwrtWorldZ(contactIndex,
                                                             time);
  return reference;
}

void SwitchedModelReferenceManager::captureMeasuredState(
    scalar_t initTime, const vector_t& initState) {
  // The guard is now "does anything downstream want this", rather than the
  // nominal step width alone: the foothold heuristics are the second consumer,
  // and they are available on robots whose step width is zero.
  if (!needsMeasuredState()) return;
  lastSolveTime_ = initTime;
  measPosBaseRtWorldEwrtWorldXy_ =
      mpcRobotModelPtr_->getPosBaseRtWorldEwrtWorld(initState)
          .head<kHorizontalPositionDim>();
  measYawBaseRtWorld_ = mpcRobotModelPtr_->getYawBaseRtWorld(initState);

  // Every foot that is in contact records where it is. The last value a foot
  // recorded before it began to swing is therefore where it lifted off from,
  // with no event to detect and nothing to reset.
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(initState);
  const std::vector<vector3_t> posContactsRtWorldEwrtWorld =
      computePosContactsRtWorldEwrtWorld<scalar_t>(q, pinocchioInterface_,
                                                   *mpcRobotModelPtr_);
  const ContactMask contacts = getContactFlags(initTime);
  for (size_t contact = 0; contact < contacts.size() &&
                           contact < posContactsRtWorldEwrtWorld.size();
       ++contact) {
    if (contacts[contact]) {
      liftoffPosContactsRtWorldEwrtWorldXy_[contact] =
          posContactsRtWorldEwrtWorld[contact].head<kHorizontalPositionDim>();
    }
  }

  // The composite yaw inertia inverts the target's momentum channel into the
  // commanded yaw rate, which predictedBaseAt() needs for the heading at
  // touch-down. The nominal step reads that heading too, so it is latched
  // whenever anything here is wanted: latching it only for a listed heuristic
  // would lay the nominal step across the MEASURED heading on a robot with none
  // listed, and listing an unrelated base-pose heuristic would then move the
  // feet.
  const pinocchio::Model& model = pinocchioInterface_.getModel();
  pinocchio::Data& data = pinocchioInterface_.getData();
  pinocchio::ccrba(
      model, data, q,
      vector_t::Zero(
          model.nv));  // composite inertia about the CoM, in the world frame
  const matrix3_t inertiaComEwrtWorld = data.Ig.inertia().matrix();
  latchInertiaComEwrtHeading(inertiaComEwrtWorld, measYawBaseRtWorld_);
  inertiaComEwrtWorldZz_ = inertiaComEwrtWorld(kZAxisIndex, kZAxisIndex);

  // The rest is for the heuristics only, and costs a center-of-mass pass, so it
  // is skipped entirely when none is listed.
  if (!heuristicLayerPtr_->empty()) {
    // The center-of-mass velocity the heuristics read: on the centroidal state
    // the linear part of the NORMALIZED momentum h = [p/m, L/m] (no division
    // by the mass is needed); a whole-body state stores only the base velocity,
    // which stands in for it (getProxyLinvelComRtWorldEwrtWorld()).
    measProxyLinvelComRtWorldEwrtWorldXy_ =
        getProxyLinvelComRtWorldEwrtWorld(*mpcRobotModelPtr_, initState)
            .head<kHorizontalPositionDim>();
    // The capture point's pendulum length, measured the way every LIP consumer
    // measures its nominal one: above the MEAN FOOT HEIGHT rather than above
    // the world origin, so that it is the same number on a robot standing on a
    // box as on one standing on the floor.
    measPosComRtFeetmeanEwrtWorldZ_ = computeProxyPosComRtFeetmeanEwrtWorldZ(
        q, pinocchioInterface_, *mpcRobotModelPtr_);
  }
  hasMeasuredState_ = true;
}

std::optional<vector3_t> SwitchedModelReferenceManager::targetPlanarCommandAt(
    scalar_t time) const {
  const TargetTrajectories& targetTrajectories = getTargetTrajectories();
  if (targetTrajectories.empty()) return std::nullopt;
  const vector_t desiredState =
      mpcRobotModelPtr_->getDesiredState(targetTrajectories, time);
  if (static_cast<size_t>(desiredState.size()) !=
      mpcRobotModelPtr_->getStateDim()) {
    return std::nullopt;
  }
  // Read the way the model's target calculator wrote the knot (design defect
  // D3: this read the centroidal slots on every model, the base position and a
  // roll rate of the whole-body knot). A centroidal knot holds the turn as the
  // momentum h_z = I_zz psidot / m of the body at the measured configuration,
  // so it comes back out with the inertia latched for this solve; until one is
  // latched it reads 0, which is accurate and safe: every
  // heuristic that reads the yaw rate is linear in it and so contributes
  // nothing rather than something wrong.
  return mpcRobotModelPtr_->getRefPlanarMotion(desiredState,
                                               inertiaComEwrtWorldZz_);
}

scalar_t SwitchedModelReferenceManager::getCmdRateYawBaseRtWorld(
    scalar_t time) const {
  const std::optional<vector3_t> command = targetPlanarCommandAt(time);
  return command.has_value()
             ? (*command)(
                   MpcRobotModelBase<scalar_t>::kRefPlanarMotionYawRateIndex)
             : 0.0;
}

vector2_t SwitchedModelReferenceManager::getCmdLinvelBaseRtWorldEwrtWorldXy(
    scalar_t time) const {
  const std::optional<vector3_t> command = targetPlanarCommandAt(time);
  return command.has_value()
             ? vector2_t(command->head<kHorizontalPositionDim>())
             : vector2_t::Zero();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t SwitchedModelReferenceManager::applyPosSurfaceRtWorldEwrtWorldZ(
    TargetTrajectories& targetTrajectories,
    const vector_t& /*initState*/,
    size_t /*initMode*/) {
  // The ground, and the only definition of it in the controller: the task
  // file's `posSurfaceRtWorldEwrtWorldZ` at start-up, moved by
  // setPosSurfaceRtWorldEwrtWorldZ() on a hot reload. This used to estimate
  // the ground's height and then overwrite the estimate with a hard-coded 0 on
  // the next line, which left a reader believing the swing
  // trajectories tracked a measured ground height when they tracked a constant
  // - and it later read the launch-time ModelSettings value, which a hot reload
  // of `posSurfaceRtWorldEwrtWorldZ` moved for the contact-implicit terms but
  // not for the swing trajectories, the landing targets or the base height.
  // Read once here, so every reference of this run is built on the same ground,
  // and recorded as the applied one for the contact-implicit terms to follow.
  const scalar_t posSurfaceRtWorldEwrtWorldZ =
      posSurfaceRtWorldEwrtWorldZ_.load();
  appliedPosSurfaceRtWorldEwrtWorldZ_.store(posSurfaceRtWorldEwrtWorldZ);

  // The target in use stands on the ground of the last run
  // (refPosSurfaceRtWorldEwrtWorldZ_): move it by the change, once. Only the
  // change: this target is kept from run to run until a new one is published,
  // and each run would otherwise lift it again. A target published after this
  // run was built on the ground applied here (the target calculators read
  // getAppliedPosSurfaceRtWorldEwrtWorldZ()), so the next run finds no change
  // to apply to it.
  // The applied ground minus the previous run's.
  const scalar_t errPosSurfaceRtWorldEwrtWorldZ =
      posSurfaceRtWorldEwrtWorldZ - refPosSurfaceRtWorldEwrtWorldZ_;
  if (errPosSurfaceRtWorldEwrtWorldZ != 0.0) {
    for (vector_t& targetState : targetTrajectories.stateTrajectory) {
      mpcRobotModelPtr_->adaptPosBaseRtWorldEwrtWorldZ(
          targetState, errPosSurfaceRtWorldEwrtWorldZ);
    }
  }
  refPosSurfaceRtWorldEwrtWorldZ_ = posSurfaceRtWorldEwrtWorldZ;
  return posSurfaceRtWorldEwrtWorldZ;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t SwitchedModelReferenceManager::getDesiredState(
    const TargetTrajectories& targetTrajectories,
    const vector_t& /*state*/,
    scalar_t time) const {
  // Interpolated in the model's layout, as every reference read of this manager
  // is.
  vector_t xNominal =
      mpcRobotModelPtr_->getDesiredState(targetTrajectories, time);

  // Off on a legs-only robot, and off while com_and_acom_tracking_cost is
  // listed: the arm orientations then emerge from the ACoM cost and would
  // conflict with the generator. The interface decides; see
  // setArmSwingReferenceActive().
  if (armSwingReferenceActive_ &&
      armSwingSettings_.generator == ArmSwingGenerator::kProcedural) {
    const ModelSettings& modelSettings = mpcRobotModelPtr_->modelSettings;
    scalar_t phaseVariable = this->getPhaseVariable(time);
    vector_t desiredJointAngles = mpcRobotModelPtr_->getJointAngles(xNominal);

    // The robot's forward speed: its center of mass's, or on a whole-body
    // state the base's, which stands in for it.
    vector3_t refLinvelComRtWorldEwrtWorld =
        getProxyLinvelComRtWorldEwrtWorld(*mpcRobotModelPtr_, xNominal);
    // The forward command in the REFERENCE's heading, not the heading of the
    // node's state: a reference that depends on the state is a cost whose
    // Gauss-Newton gradient (J' Q (x - x_ref), with x_ref held constant) is
    // wrong by -J' Q dx_ref/dx, and the arm swing's heading is the one channel
    // of this reference that used to read the state. The reference's heading is
    // the commanded path's, which is also the frame shapePoseBaseRtWorld()
    // reads "forward" in.
    const scalar_t refYawBaseRtWorld =
        mpcRobotModelPtr_->getYawBaseRtWorld(xNominal);

    const scalar_t refLinvelComRtWorldEwrtHeadingX =
        (std::cos(refYawBaseRtWorld) * refLinvelComRtWorldEwrtWorld[0] +
         std::sin(refYawBaseRtWorld) * refLinvelComRtWorldEwrtWorld[1]);

    // How far the left arm swings forward; the right one swings back by as
    // much. Each joint turns the way that swings its arm forward (a shoulder)
    // or flexes it (an elbow), as derived from the model (ArmSwingDirections),
    // so that the same numbers mean the same motion on a shoulder pitch joint
    // and on the shoulder yaw joint of an arm that hangs to the side.
    const scalar_t leftSwing =
        std::sin(kArmSwingRadiansPerCycle *
                 (phaseVariable - armSwingSettings_.phaseOffset)) *
        refLinvelComRtWorldEwrtHeadingX;
    const scalar_t rightSwing = -leftSwing;
    desiredJointAngles[modelSettings.j_l_shoulder_y_index] +=
        armSwingDirections_.leftShoulder * armSwingSettings_.shoulderGain *
        leftSwing;
    desiredJointAngles[modelSettings.j_r_shoulder_y_index] +=
        armSwingDirections_.rightShoulder * armSwingSettings_.shoulderGain *
        rightSwing;
    desiredJointAngles[modelSettings.j_l_elbow_y_index] +=
        armSwingDirections_.leftElbow * armSwingSettings_.elbowGain * leftSwing;
    desiredJointAngles[modelSettings.j_r_elbow_y_index] +=
        armSwingDirections_.rightElbow * armSwingSettings_.elbowGain *
        rightSwing;

    mpcRobotModelPtr_->setJointAngles(xNominal, desiredJointAngles);
  }

  // The base-pose seam of the locomotion heuristics; see
  // shapePoseBaseRtWorld(), which the terminal and torso costs call too.
  return shapePoseBaseRtWorld(time, xNominal);
}

vector_t SwitchedModelReferenceManager::shapePoseBaseRtWorld(
    scalar_t time, const vector_t& xNominal) const {
  if (heuristicLayerPtr_->basePoseHeuristicsEmpty()) return xNominal;
  // Here, per node, at the point of use - not in the target-trajectory
  // calculator, where the roll and pitch zeros are written: that trajectory's
  // knots are placed by the command (three on a straight walk) and could not
  // carry a limit cycle at gait frequency, it is published one solve ahead of
  // the solver that reads it, and it is also what the operator's delta-pose
  // commands are measured against. The zeros there stay; this writes on top of
  // them.
  BasePoseHeuristicContext context;
  // The OPERATOR's command, through the same accessor the foothold and wrench
  // seams use. Not the target's momentum channel: under online contact planning
  // the planned_com_override rule rewrites that with the plan's own CoM
  // velocity, which sways with the gait, and Bledt's H_Theta and H_z are
  // functions of the commanded velocity.
  const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy =
      getCmdLinvelBaseRtWorldEwrtWorldXy(time);
  // Into the reference's own heading frame, so that "forward" means the robot's
  // forward rather than the world's x. The reference's heading is the right one
  // and not the measured one: the heuristics describe how the robot should
  // carry itself along the path it was told to take.
  const scalar_t refYawBaseRtWorld =
      mpcRobotModelPtr_->getYawBaseRtWorld(xNominal);
  const scalar_t cosYaw = std::cos(refYawBaseRtWorld);
  const scalar_t sinYaw = std::sin(refYawBaseRtWorld);
  context.cmdLinvelBaseRtWorldEwrtHeadingXy =
      vector2_t(cosYaw * cmdLinvelBaseRtWorldEwrtWorldXy(0) +
                    sinYaw * cmdLinvelBaseRtWorldEwrtWorldXy(1),
                -sinYaw * cmdLinvelBaseRtWorldEwrtWorldXy(0) +
                    cosYaw * cmdLinvelBaseRtWorldEwrtWorldXy(1));
  context.cmdRateYawBaseRtWorld = getCmdRateYawBaseRtWorld(time);
  context.gaitPhase = getPhaseVariable(time);
  context.gaitPhaseRate = getPhaseVariableRate(time);

  const BasePoseOffset offset =
      heuristicLayerPtr_->basePoseHeuristicOffset(context);
  vector_t shaped = xNominal;
  // LINT.IfChange(base_pose_heuristic_seam)
  // The heuristics' Euler boundary (quaternion_base_orientation/README.md,
  // section 4.2): getEulerzyxBaseRtWorld() returns (yaw, pitch, roll),
  // Euler ZYX with YAW FIRST, so roll, the angle about x, is index 2
  // (kEulerAnglesZyxXIndex) and pitch is index 1 (kEulerAnglesZyxYIndex) - the
  // reverse of the (roll, pitch, yaw) order every textbook writes - and the
  // height is index 2 of the base position. This is the one place in the
  // subsystem that knows it.
  const vector3_t refEulerzyxBaseRtWorld =
      mpcRobotModelPtr_->getEulerzyxBaseRtWorld(shaped);
  vector3_t eulerzyxBaseRtWorld = refEulerzyxBaseRtWorld;
  eulerzyxBaseRtWorld(kEulerAnglesZyxXIndex) +=
      offset.offsetEulerzyxBaseRtWorldRoll;
  eulerzyxBaseRtWorld(kEulerAnglesZyxYIndex) +=
      offset.offsetEulerzyxBaseRtWorldPitch;
  mpcRobotModelPtr_->setEulerzyxBaseRtWorld(shaped, eulerzyxBaseRtWorld);
  vector3_t offsetRateEulerzyxBaseRtWorld = vector3_t::Zero();
  offsetRateEulerzyxBaseRtWorld(kEulerAnglesZyxXIndex) =
      offset.offsetRateEulerzyxBaseRtWorldRoll;
  offsetRateEulerzyxBaseRtWorld(kEulerAnglesZyxYIndex) =
      offset.offsetRateEulerzyxBaseRtWorldPitch;
  // The rate channel follows the attitude: the shaped attitude is Theta_n(t) +
  // delta(t), so its rate is the derivative of THAT. A state that stores the
  // body angular velocity (the whole body's) gets the body rate of Theta_n +
  // delta with the Euler rates dTheta_n/dt + d(delta)/dt
  // (computeShapedAngvelBaseRtWorldEwrtBase()), which keeps the reference's
  // turn in the world and adds the rate of a periodic tilt. A state that stores
  // the centroidal momentum gets the angular momentum of the tilt rate, I_G
  // w_tilt / m with the composite inertia latched at the solve, on top of the
  // turn the target calculator already wrote there. The node's body is the
  // measured one turned about the world z axis to the node's heading, so its
  // inertia is the latched one turned there too, R_z I_H R_z^T, as in the
  // target calculator. Left as they were, the
  // reference would describe an attitude that oscillates while its rate says it
  // holds still.
  if (storesAngvelBaseRtWorldEwrtBase(mpcRobotModelPtr_->getTuningLayout())) {
    mpcRobotModelPtr_->setAngvelBaseRtWorldEwrtBase(
        shaped, computeShapedAngvelBaseRtWorldEwrtBase(
                    refEulerzyxBaseRtWorld,
                    mpcRobotModelPtr_->getAngvelBaseRtWorldEwrtBase(xNominal),
                    eulerzyxBaseRtWorld, offsetRateEulerzyxBaseRtWorld));
  } else if (const std::optional<size_t> angmomStart =
                 mnormAngmomStartIndex(mpcRobotModelPtr_->getTuningLayout());
             angmomStart.has_value() && totalMass_ > 0.0 &&
             !offsetRateEulerzyxBaseRtWorld.isZero(0.0)) {
    const matrix3_t refInertiaComEwrtWorld =
        rotateInertiaAboutWorldZ(inertiaComEwrtHeading_, refYawBaseRtWorld);
    shaped.segment<kRotationTangentDim>(*angmomStart) +=
        refInertiaComEwrtWorld *
        computeOffsetAngvelBaseRtWorldEwrtWorld(eulerzyxBaseRtWorld,
                                                offsetRateEulerzyxBaseRtWorld) /
        totalMass_;
  }
  vector3_t posBaseRtWorldEwrtWorld =
      mpcRobotModelPtr_->getPosBaseRtWorldEwrtWorld(shaped);
  posBaseRtWorldEwrtWorld(kZAxisIndex) += offset.offsetPosBaseRtWorldEwrtWorldZ;
  mpcRobotModelPtr_->setPosBaseRtWorldEwrtWorld(shaped,
                                                posBaseRtWorldEwrtWorld);
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/locomotion_heuristics/BasePoseHeuristic.h:base_pose_offset_convention)
  // clang-format on
  return shaped;
}

void SwitchedModelReferenceManager::latchInertiaComEwrtHeading(
    const matrix3_t& inertiaComEwrtWorld, scalar_t yawBaseRtWorld) {
  inertiaComEwrtHeading_ =
      rotateInertiaAboutWorldZ(inertiaComEwrtWorld, -yawBaseRtWorld);
}

matrix3_t SwitchedModelReferenceManager::rotateInertiaAboutWorldZ(
    const matrix3_t& inertia, scalar_t yaw) {
  const matrix3_t rotmat =
      Eigen::AngleAxis<scalar_t>(yaw, vector3_t::UnitZ()).toRotationMatrix();
  return rotmat * inertia * rotmat.transpose();
}

vector3_t SwitchedModelReferenceManager::computeShapedAngvelBaseRtWorldEwrtBase(
    const vector3_t& refEulerzyxBaseRtWorld,
    const vector3_t& refAngvelBaseRtWorldEwrtBase,
    const vector3_t& shapedEulerzyxBaseRtWorld,
    const vector3_t& offsetRateEulerzyxBaseRtWorld) {
  const vector3_t refRateEulerzyxBaseRtWorld = eulerZyxRatesFromBodyRates(
      refEulerzyxBaseRtWorld, refAngvelBaseRtWorldEwrtBase);
  return bodyRatesFromEulerZyxRates(
      shapedEulerzyxBaseRtWorld,
      vector3_t(refRateEulerzyxBaseRtWorld + offsetRateEulerzyxBaseRtWorld));
}

vector3_t
SwitchedModelReferenceManager::computeOffsetAngvelBaseRtWorldEwrtWorld(
    const vector3_t& shapedEulerzyxBaseRtWorld,
    const vector3_t& offsetRateEulerzyxBaseRtWorld) {
  return quaternionRotationMatrix<scalar_t>(
             quaternionFromEulerZyx(shapedEulerzyxBaseRtWorld)) *
         bodyRatesFromEulerZyxRates(shapedEulerzyxBaseRtWorld,
                                    offsetRateEulerzyxBaseRtWorld);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t SwitchedModelReferenceManager::getDesiredInput(
    const TargetTrajectories& targetTrajectories,
    const vector_t& /*state*/,
    scalar_t time) const {
  const ContactMask contactMask = getContactFlags(time);
  // Any contact set other than the biped's shares the weight among its active
  // contacts; the wrench heuristics, which are written for two feet, are
  // refused for it (checkCentroidalContactSet()).
  if (!mpcRobotModelPtr_->modelSettings.isBipedContactSet() ||
      heuristicLayerPtr_->wrenchHeuristicsEmpty()) {
    // Byte for byte the call the two input costs used to make for themselves.
    // Input-only on purpose: the nominal contact force is purely vertical and a
    // stance foot is flat, so it is the same in the world and local contact
    // frames, and the state-aware overload would add a forward-kinematics pass
    // per node and iteration for nothing.
    return weightCompensatingInput(pinocchioInterface_, contactMask,
                                   *mpcRobotModelPtr_);
  }

  // The wrench heuristics are written for a left and a right foot
  // (WrenchHeuristicContext), and the set is the biped's here.
  const BipedFeetArray<bool> contactFlags = {contactMask[kContactLeftIndex],
                                             contactMask[kContactRightIndex]};
  WrenchHeuristicContext context;
  context.contactFlags = contactFlags;
  context.numStanceFeet = numberOfLegsInContacts(contactFlags);
  context.cmdLinvelBaseRtWorldEwrtWorldXy =
      getCmdLinvelBaseRtWorldEwrtWorldXy(time);
  context.cmdRateYawBaseRtWorld = getCmdRateYawBaseRtWorld(time);
  // The same gravity weightCompensatingInput() uses, so that the offsets and
  // the reference they correct are measured in the same weight. It is a
  // constant in both places because the centroidal model carries no gravity
  // constant of its own.
  context.totalWeight = totalMass_ * kGravityMagnitude;
  // Each foot's duty factor over its OWN stride, lift-off to lift-off, from the
  // schedule this node is solved on. On a periodic gait that is the gait's duty
  // factor at every node, which is what makes the scaled reference average to
  // exactly the weight; see stanceDutyFactor(). The same whichever reference
  // manager built the schedule, so nothing has to be latched per solve.
  const ModeSchedule& schedule = getModeSchedule();
  for (size_t foot = 0; foot < contactFlags.size(); ++foot) {
    context.stanceDutyFactor[foot] = stanceDutyFactor(schedule, foot, time);
  }

  // Which path writes the reference is decided by what the listed heuristics
  // PRODUCE, not by the input parameterization: a vertical force is the same in
  // the world and in a flat foot's own frame, so it takes the cheap path even
  // under basis-vector inputs, while a horizontal one has to be rotated into
  // the local contact frame. See
  // WrenchHeuristic::producesForceFootEwrtWorldXy().
  if (!heuristicLayerPtr_->wrenchHeuristicsNeedWorldFrame()) {
    vector_t input = weightCompensatingInput(pinocchioInterface_, contactMask,
                                             *mpcRobotModelPtr_);
    if (context.numStanceFeet == 0) return input;
    for (size_t foot = 0; foot < contactFlags.size(); ++foot) {
      if (!contactFlags[foot]) continue;
      const vector3_t offsetForceFootEwrtWorld =
          heuristicLayerPtr_->computeOffsetForceFootEwrtWorld(context, foot);
      if (offsetForceFootEwrtWorld.isZero()) continue;
      mpcRobotModelPtr_->setForceContactEwrtParam(
          input,
          mpcRobotModelPtr_->getForceContactEwrtParam(input, foot) +
              offsetForceFootEwrtWorld,
          foot);
    }
    return input;
  }

  // The world-frame path. Every setForceContactEwrtWorld() /
  // getForceContactEwrtWorld() call on the basis-vector model copies the
  // whole pinocchio::Data and runs a forward-kinematics pass, so the total
  // force is assembled here and written ONCE per stance foot rather than being
  // built by weightCompensatingInput() and then read back and re-written -
  // which would cost three passes per foot instead of one. Starting from a zero
  // input is exact, because weightCompensatingInput() leaves everything except
  // the stance contact forces at zero.
  vector_t input = vector_t::Zero(mpcRobotModelPtr_->getInputDim());
  if (context.numStanceFeet == 0) return input;
  // The contact frames the world force is written into are those of the
  // REFERENCE configuration at this node - the target's interpolated knot, a
  // level base at the commanded heading on the nominal posture - and not of the
  // node's state. A reference that moves with the state is not a constant of
  // the Gauss-Newton step that treats it as one: the gradient of |u -
  // u_ref(x)|^2 would be missing the -2 (du_ref/dx)' R (u - u_ref) term. On the
  // reference the stance feet are flat and turned with the commanded heading;
  // where a planted foot's yaw differs from it, the world force the input
  // describes is the reference's turned by that difference, which the momentum
  // and friction terms of the problem, not this regularization, are responsible
  // for.
  const vector_t referenceState =
      mpcRobotModelPtr_->getDesiredState(targetTrajectories, time);
  const vector3_t nomForceFootEwrtWorld(
      0.0, 0.0,
      context.totalWeight / static_cast<scalar_t>(context.numStanceFeet));
  for (size_t foot = 0; foot < contactFlags.size(); ++foot) {
    if (!contactFlags[foot]) continue;
    mpcRobotModelPtr_->setForceContactEwrtWorld(
        referenceState, input,
        nomForceFootEwrtWorld +
            heuristicLayerPtr_->computeOffsetForceFootEwrtWorld(context, foot),
        foot);
  }
  return input;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void SwitchedModelReferenceManager::modifyReferences(
    scalar_t initTime,
    scalar_t finalTime,
    const vector_t& initState,
    size_t initMode,
    TargetTrajectories& targetTrajectories,
    ModeSchedule& modeSchedule) {
  const scalar_t timeHorizon = finalTime - initTime;
  modeSchedule = gaitSchedulePtr_->getModeSchedule(initTime - timeHorizon,
                                                   finalTime + timeHorizon);

  scalar_t posSurfaceRtWorldEwrtWorldZ =
      applyPosSurfaceRtWorldEwrtWorldZ(targetTrajectories, initState, initMode);

  swingTrajectoryPtr_->update(modeSchedule, posSurfaceRtWorldEwrtWorldZ);

  modeSchedule_ = modeSchedule;
  captureMeasuredState(initTime, initState);
}

}  // namespace ocs2::humanoid
