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
#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/manifold/UnitQuaternionMath.h"
#include "ocs2_core/misc/Numerics.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

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
constexpr scalar_t kArmSwingRadiansPerCycle = 2 * M_PI;

/** Whether a state of the layout `layout` stores the base's body angular
 * velocity (a kZyxOrderedAngularVelocity block). */
bool storesBaseAngularVelocity(const TuningLayout& layout) {
  for (const TuningSegment& segment : layout.segments) {
    if (segment.kind == TuningSegmentKind::kZyxOrderedAngularVelocity)
      return true;
  }
  return false;
}

/** The tuning block of the centroidal state that holds the normalized angular
 * momentum L / m (tuningLayoutOf()). */
constexpr char kNormalizedAngularMomentumSegment[] =
    "normalized_angular_momentum";

/** The first state row of the normalized angular momentum L / m, for a state
 * that stores it (the centroidal one). */
std::optional<size_t> normalizedAngularMomentumRow(const TuningLayout& layout) {
  for (const TuningSegment& segment : layout.segments) {
    if (segment.name == kNormalizedAngularMomentumSegment)
      return segment.stateStart;
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
  // manager's (setTerrainHeight).
  terrainHeight_.store(mpcRobotModel.modelSettings.terrainHeight);
  appliedTerrainHeight_.store(mpcRobotModel.modelSettings.terrainHeight);
  targetTerrainHeight_ = mpcRobotModel.modelSettings.terrainHeight;
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
  const scalar_t terrainHeight = terrainHeight_.load();
  appliedTerrainHeight_.store(terrainHeight);
  targetTerrainHeight_ = terrainHeight;
  resetRuntimeState();
}

void SwitchedModelReferenceManager::resetRuntimeState() {
  gaitSchedulePtr_->reset();
  modeSchedule_ = ModeSchedule();

  hasMeasuredState_ = false;
  lastSolveTime_ = 0.0;
  measuredBasePosition_.setZero();
  measuredBaseYaw_ = 0.0;
  liftOffPositions_ = makeFeetArray(vector2_t(vector2_t::Zero()));
  measuredComVelocity_.setZero();
  measuredComHeight_ = 0.0;
  yawInertia_ = 0.0;
  compositeInertia_.setZero();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
contact_flag_t SwitchedModelReferenceManager::getContactFlags(
    scalar_t time) const {
  return modeNumber2StanceLeg(this->getModeSchedule().modeAtTime(time));
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
      it == modeSchedule_.eventTimes.begin())
    return 0.0;
  const scalar_t nextEventTime = *it;
  const scalar_t prevEventTime = *(it - 1);
  if (!(nextEventTime > prevEventTime)) return 0.0;

  if (modeSchedule_.modeAtTime(time) == ModeNumber::kLf) {
    return 0.5 * (time - prevEventTime) / (nextEventTime - prevEventTime);
  } else if (modeSchedule_.modeAtTime(time) == ModeNumber::kRf) {
    return kRightStancePhaseStart +
           0.5 * (time - prevEventTime) / (nextEventTime - prevEventTime);
  } else {
    if (modeSchedule_.modeAtTime(prevEventTime - kPhaseVariableModeLookBack) ==
        ModeNumber::kLf) {
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
      it == modeSchedule_.eventTimes.begin())
    return 0.0;
  const scalar_t nextEventTime = *it;
  const scalar_t prevEventTime = *(it - 1);
  if (!(nextEventTime > prevEventTime)) return 0.0;
  const size_t mode = modeSchedule_.modeAtTime(time);
  if (mode == ModeNumber::kLf || mode == ModeNumber::kRf)
    return 0.5 / (nextEventTime - prevEventTime);
  return 0.0;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector3_t SwitchedModelReferenceManager::getSwingFootPlaneNormal(
    size_t contactIndex, scalar_t time) const {
  // A flat foot's normal is the world z axis.
  if (isInContact(time, contactIndex)) return vector3_t::UnitZ();

  const scalar_t pitch =
      swingTrajectoryPtr_->getSwingPitchAngle(contactIndex, time);
  if (numerics::almost_eq(pitch, /*y=*/0.0)) return vector3_t::UnitZ();

  // Heading of the foot: the planned foot yaw when the contact planner's
  // heading model provides one, otherwise the commanded base heading, so that
  // the tilt stays about the foot's lateral axis and does not leak into roll.
  scalar_t yaw = 0.0;
  const std::optional<SwingFootReference> swingReference =
      getSwingFootReference(contactIndex, time);
  if (swingReference.has_value() && swingReference->yaw.has_value()) {
    yaw = *swingReference->yaw;
  } else if (const TargetTrajectories& targetTrajectories =
                 getTargetTrajectories();
             !targetTrajectories.empty()) {
    const vector_t desiredState =
        mpcRobotModelPtr_->getDesiredState(targetTrajectories, time);
    if (static_cast<size_t>(desiredState.size()) ==
        mpcRobotModelPtr_->getStateDim()) {
      yaw = mpcRobotModelPtr_->getBaseHeading(desiredState);
    }
  }

  // Toe-up by `pitch` is a rotation of -pitch about the foot's lateral (+y)
  // axis, which tilts the sole normal backwards along the heading. The result
  // is a unit vector, as rotationMatrixDistanceToPlane expects.
  const scalar_t sp = std::sin(pitch);
  return vector3_t(-sp * std::cos(yaw), -sp * std::sin(yaw), std::cos(pitch));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::optional<vector2_t>
SwitchedModelReferenceManager::getSwingFootVelocityReference(
    size_t contactIndex, scalar_t time) const {
  if (isInContact(time, contactIndex)) return std::nullopt;

  return getCommandedVelocity(time);
}

absl::Status SwitchedModelReferenceManager::configureArmSwing(
    const mpc_config::ArmSwingConfig& config, const vector_t& nominalState) {
  ASSIGN_OR_RETURN(const ArmSwingSettings settings,
                   armSwingSettingsFromConfig(config));
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
  return mpcRobotModelPtr_->modelSettings.nominalFootholdConfig.stepWidth >
             0.0 ||
         !heuristicLayerPtr_->empty();
}

SwitchedModelReferenceManager::PlanarPoseChange
SwitchedModelReferenceManager::commandedPoseChange(const PlanarCommandAt& start,
                                                   const PlanarCommandAt& end,
                                                   scalar_t duration) {
  const scalar_t meanYawRate = 0.5 * (start.yawRate + end.yawRate);
  const scalar_t yawChange = meanYawRate * duration;
  // R_z(-theta) v_1: the end velocity in the frame that has turned with the
  // command, where a constant body twist keeps it.
  const scalar_t cosine = std::cos(yawChange);
  const scalar_t sine = std::sin(yawChange);
  const vector2_t& endVelocity = end.velocityWorld;
  const vector2_t endInTurningFrame(
      cosine * endVelocity(0) + sine * endVelocity(1),
      -sine * endVelocity(0) + cosine * endVelocity(1));
  const vector2_t meanVelocity =
      0.5 * (start.velocityWorld + endInTurningFrame);
  return PlanarPoseChange{
      .displacement = TargetTrajectoriesCalculatorBase::planarArcDisplacement(
          meanVelocity, meanYawRate, duration),
      .yawChange = yawChange};
}

SwitchedModelReferenceManager::PlanarBasePose
SwitchedModelReferenceManager::predictedBaseAt(scalar_t time) const {
  // The measurement carried forward by the command over the time until `time`
  // (commandedPoseChange()). Only the foothold heuristics read the position
  // (footholdContext()), and no shipped robot lists one; the nominal step reads
  // the yaw.
  const scalar_t elapsed = std::max(time - lastSolveTime_, 0.0);
  const PlanarPoseChange change = commandedPoseChange(
      PlanarCommandAt{.velocityWorld = getCommandedVelocity(lastSolveTime_),
                      .yawRate = getCommandedYawRate(lastSolveTime_)},
      PlanarCommandAt{.velocityWorld = getCommandedVelocity(time),
                      .yawRate = getCommandedYawRate(time)},
      elapsed);
  return PlanarBasePose{
      .position = vector2_t(measuredBasePosition_ + change.displacement),
      .yaw = measuredBaseYaw_ + change.yawChange};
}

FootholdHeuristicContext SwitchedModelReferenceManager::footholdContext(
    size_t contactIndex, scalar_t time) const {
  FootholdHeuristicContext context;
  context.contactIndex = contactIndex;
  context.side = (contactIndex == kContactLeftIndex) ? 1.0 : -1.0;
  const PlanarBasePose predicted = predictedBaseAt(time);
  context.basePosition = predicted.position;
  context.baseYaw = predicted.yaw;
  context.measuredVelocity = measuredComVelocity_;
  context.commandedVelocityAtMeasurement = getCommandedVelocity(lastSolveTime_);
  context.commandedVelocity = getCommandedVelocity(time);
  context.commandedYawRate = getCommandedYawRate(time);
  context.comHeight = measuredComHeight_;
  context.stanceDuration =
      upcomingStanceDuration(modeSchedule_, contactIndex, time);
  return context;
}

std::optional<vector2_t> SwitchedModelReferenceManager::nominalFoothold(
    size_t contactIndex, scalar_t time) const {
  const scalar_t stepWidth =
      mpcRobotModelPtr_->modelSettings.nominalFootholdConfig.stepWidth;
  // Nothing has been measured yet, so nothing can be placed: this is the first
  // solve, before captureMeasuredState().
  if (!hasMeasuredState_) return std::nullopt;
  const bool haveHeuristics = !heuristicLayerPtr_->footholdEmpty();
  // No opinion from either source: a robot with `stepWidth: 0` and no listed
  // heuristic returns nullopt, and the foot cost's xy weights stay switched off
  // exactly as they were.
  if (stepWidth <= 0.0 && !haveHeuristics) return std::nullopt;

  const size_t stanceIndex = (contactIndex == kContactLeftIndex)
                                 ? kContactRightIndex
                                 : kContactLeftIndex;
  const scalar_t side = (contactIndex == kContactLeftIndex) ? 1.0 : -1.0;
  const vector2_t& stanceFoot = liftOffPositions_[stanceIndex];

  if (!haveHeuristics) {
    // THE NOMINAL STEP. A whole step width to this foot's side of the stance
    // foot, carried forward at the commanded velocity over one step: from the
    // stance foot's own touch-down to this one. Both ends are fixed instants,
    // so the target holds still through the swing. (It used to be carried
    // forward over the time from the LAST SOLVE to the touch-down, which
    // shrinks to nothing as the swing ends: the target slid backwards through
    // every swing and converged on the stance foot's own position, a
    // zero-length step at any speed.)
    //
    // Measured from the stance foot, not from the base: in single support the
    // base sits roughly over the stance foot, so half a step width from the
    // base is half a step width from the stance foot - half the separation
    // intended - and the base moves further over the stance foot with every
    // step, so the feet converge.
    //
    // One step of travel from the stance foot is where Raibert's rule puts the
    // foot in steady state, with no coefficient: the stance foot itself landed
    // half a stance of travel ahead of the CoM one step earlier.
    const std::optional<scalar_t> stanceLanded =
        previousTouchDownTime(modeSchedule_, stanceIndex, time);
    const std::optional<SwingWindow> swing = swingPhaseAtTime(
        modeSchedule_, contactIndex, time - kNominalStepSwingLookBack);
    const scalar_t stepStart =
        stanceLanded.has_value()
            ? *stanceLanded
            : (swing.has_value() ? swing->liftOff : lastSolveTime_);
    const scalar_t yaw = predictedBaseAt(time).yaw;
    const vector2_t advance =
        getCommandedVelocity(time) * std::max(time - stepStart, 0.0);
    const vector2_t lateral(-std::sin(yaw) * side * stepWidth,
                            std::cos(yaw) * side * stepWidth);
    return vector2_t(stanceFoot + advance + lateral);
  }

  // BLEDT'S H_r: the hip at the moment of landing, plus the listed heuristics'
  // offsets. See the header.
  const FootholdHeuristicContext context = footholdContext(contactIndex, time);
  const vector2_t offsets = heuristicLayerPtr_->footholdOffset(context);
  if (heuristicLayerPtr_->footholdMovesAnchor()) {
    // hip_centered_stepping is listed: its offset is the hip relative to the
    // predicted base, so the base is the anchor.
    return vector2_t(context.basePosition + offsets);
  }
  // Otherwise the hip is half a step width to this foot's side of the predicted
  // base. Create() refuses a zero step width here. (It used to be the STANCE
  // FOOT plus a whole step width across the heading, the predicted base along
  // it: the stance foot is where the body WAS when that foot landed, so under a
  // lateral command the feet fell behind the base by v_y T_step - k T_st v_y at
  // every step, k T_st v_y being translational_stepping's lead, and the stance
  // drifted away from the body without bound. See footholdAnchor().)
  return vector2_t(
      footholdAnchor(context.basePosition, context.baseYaw, side, stepWidth) +
      offsets);
}

vector2_t SwitchedModelReferenceManager::footholdAnchor(
    const vector2_t& predictedBase,
    scalar_t predictedYaw,
    scalar_t side,
    scalar_t stepWidth) {
  const vector2_t leftward(-std::sin(predictedYaw), std::cos(predictedYaw));
  return vector2_t(predictedBase + leftward * (0.5 * side * stepWidth));
}

std::optional<SwingFootReference>
SwitchedModelReferenceManager::getSwingFootReference(size_t contactIndex,
                                                     scalar_t time) const {
  if (isInContact(time, contactIndex)) return std::nullopt;
  const std::optional<SwingWindow> phase =
      swingPhaseAtTime(modeSchedule_, contactIndex, time);
  if (!phase.has_value()) return std::nullopt;
  const scalar_t liftOffTime = phase->liftOff;
  const scalar_t touchDownTime = phase->touchDown;
  const scalar_t duration = touchDownTime - liftOffTime;
  if (duration <= kMinSwingReferenceDuration) return std::nullopt;

  // The swing starts where the foot actually lifted off, not where the nominal
  // offset would have put it. Blending between two nominal points describes a
  // path the foot is not on, and the step the cost then demands at lift-off is
  // whatever error had accumulated by then.
  const std::optional<vector2_t> target =
      nominalFoothold(contactIndex, touchDownTime);
  if (!target.has_value()) return std::nullopt;
  const vector2_t start = liftOffPositions_[contactIndex];

  // The same cubic blend the planned-foothold path uses: p'(0) = 1, p'(1) = 0,
  // so the foot leaves the ground moving with the step and settles onto the
  // target instead of arriving at speed.
  const scalar_t tau = std::clamp((time - liftOffTime) / duration, 0.0, 1.0);
  const scalar_t tau2 = tau * tau;
  const scalar_t blend = -tau2 * tau + tau2 + tau;
  const scalar_t blendRate =
      (-kSwingBlendCubicDegree * tau2 + 2.0 * tau + 1.0) / duration;
  const vector2_t delta = *target - start;

  SwingFootReference reference;
  reference.position.head<kHorizontalPositionDim>() = start + blend * delta;
  reference.position(kZAxisIndex) =
      swingTrajectoryPtr_->getZpositionConstraint(contactIndex, time);
  reference.linearVelocity.head<kHorizontalPositionDim>() = blendRate * delta;
  reference.linearVelocity(kZAxisIndex) =
      swingTrajectoryPtr_->getZvelocityConstraint(contactIndex, time);
  return reference;
}

void SwitchedModelReferenceManager::captureMeasuredState(
    scalar_t initTime, const vector_t& initState) {
  // The guard is now "does anything downstream want this", rather than the
  // nominal step width alone: the foothold heuristics are the second consumer,
  // and they are available on robots whose step width is zero.
  if (!needsMeasuredState()) return;
  lastSolveTime_ = initTime;
  measuredBasePosition_ = mpcRobotModelPtr_->getBasePosition(initState)
                              .head<kHorizontalPositionDim>();
  measuredBaseYaw_ = mpcRobotModelPtr_->getBaseHeading(initState);

  // Every foot that is in contact records where it is. The last value a foot
  // recorded before it began to swing is therefore where it lifted off from,
  // with no event to detect and nothing to reset.
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(initState);
  const std::vector<vector3_t> feet = computeContactPositions<scalar_t>(
      q, pinocchioInterface_, *mpcRobotModelPtr_);
  const contact_flag_t contacts = getContactFlags(initTime);
  for (size_t foot = 0; foot < kNumContacts && foot < feet.size(); ++foot) {
    if (contacts[foot])
      liftOffPositions_[foot] = feet[foot].head<kHorizontalPositionDim>();
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
  compositeInertia_ = data.Ig.inertia().matrix();
  yawInertia_ = compositeInertia_(kZAxisIndex, kZAxisIndex);

  // The rest is for the heuristics only, and costs a center-of-mass pass, so it
  // is skipped entirely when none is listed.
  if (!heuristicLayerPtr_->empty()) {
    // The linear part of the state's momentum channel IS the center-of-mass
    // velocity: the state carries the NORMALIZED momentum h = [p/m, L/m], so no
    // division by the mass is needed here.
    measuredComVelocity_ =
        mpcRobotModelPtr_->getBaseComLinearVelocity(initState)
            .head<kHorizontalPositionDim>();
    // The capture point's pendulum length, measured the way every LIP consumer
    // measures its nominal one: above the MEAN FOOT HEIGHT rather than above
    // the world origin, so that it is the same number on a robot standing on a
    // box as on one standing on the floor.
    measuredComHeight_ =
        computeComHeightAboveFeet(q, pinocchioInterface_, *mpcRobotModelPtr_);
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
      mpcRobotModelPtr_->getStateDim())
    return std::nullopt;
  // Read the way the model's target calculator wrote the knot (design defect
  // D3: this read the centroidal slots on every model, the base position and a
  // roll rate of the whole-body knot). A centroidal knot holds the turn as the
  // momentum h_z = I_zz psidot / m of the body at the measured configuration,
  // so it comes back out with the inertia latched for this solve; until one is
  // latched it reads 0, the honest answer and also the safe one: every
  // heuristic that reads the yaw rate is linear in it and so contributes
  // nothing rather than something wrong.
  return mpcRobotModelPtr_->getTargetPlanarCommand(desiredState, yawInertia_);
}

scalar_t SwitchedModelReferenceManager::getCommandedYawRate(
    scalar_t time) const {
  const std::optional<vector3_t> command = targetPlanarCommandAt(time);
  return command.has_value()
             ? (*command)(
                   MpcRobotModelBase<scalar_t>::kPlanarCommandYawRateIndex)
             : 0.0;
}

vector2_t SwitchedModelReferenceManager::getCommandedVelocity(
    scalar_t time) const {
  const std::optional<vector3_t> command = targetPlanarCommandAt(time);
  return command.has_value()
             ? vector2_t(command->head<kHorizontalPositionDim>())
             : vector2_t::Zero();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t SwitchedModelReferenceManager::adaptToCurrentGroundHeight(
    TargetTrajectories& targetTrajectories,
    const vector_t& /*initState*/,
    size_t /*initMode*/) {
  // The ground, and the only definition of it in the controller: the task
  // file's `terrainHeight` at start-up, moved by setTerrainHeight() on a hot
  // reload. This used to call computeGroundHeightEstimate() and then overwrite
  // the result with a hard-coded 0 on the next line, which left a reader
  // believing the swing trajectories tracked a measured ground height when they
  // tracked a constant - and it later read the launch-time ModelSettings value,
  // which a hot reload of `terrainHeight` moved for the contact-implicit terms
  // but not for the swing trajectories, the landing targets or the base height.
  // Read once here, so every reference of this run is built on the same ground,
  // and recorded as the applied one for the contact-implicit terms to follow.
  const scalar_t terrainHeight = terrainHeight_.load();
  appliedTerrainHeight_.store(terrainHeight);

  // The target in use stands on the ground of the last run
  // (targetTerrainHeight_): move it by the change, once. Only the change: this
  // target is kept from run to run until a new one is published, and each run
  // would otherwise lift it again. A target published after this run was built
  // on the ground applied here (the target calculators read
  // getAppliedTerrainHeight()), so the next run finds no change to apply to it.
  const scalar_t heightChange = terrainHeight - targetTerrainHeight_;
  if (heightChange != 0.0) {
    for (vector_t& targetState : targetTrajectories.stateTrajectory) {
      mpcRobotModelPtr_->adaptBasePoseHeight(targetState, heightChange);
    }
  }
  targetTerrainHeight_ = terrainHeight;
  return terrainHeight;
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

    vector3_t linVelCommand =
        mpcRobotModelPtr_->getBaseComLinearVelocity(xNominal);
    // The forward command in the REFERENCE's heading, not the heading of the
    // node's state: a reference that depends on the state is a cost whose
    // Gauss-Newton gradient (J' Q (x - x_ref), with x_ref held constant) is
    // wrong by -J' Q dx_ref/dx, and the arm swing's heading is the one channel
    // of this reference that used to read the state. The reference's heading is
    // the commanded path's, which is also the frame shapeBasePose() reads
    // "forward" in.
    const scalar_t referenceHeading =
        mpcRobotModelPtr_->getBaseHeading(xNominal);

    const scalar_t localVelXCommand =
        (std::cos(referenceHeading) * linVelCommand[0] +
         std::sin(referenceHeading) * linVelCommand[1]);

    // How far the left arm swings forward; the right one swings back by as
    // much. Each joint turns the way that swings its arm forward (a shoulder)
    // or flexes it (an elbow), as derived from the model (ArmSwingDirections),
    // so that the same numbers mean the same motion on a shoulder pitch joint
    // and on the shoulder yaw joint of an arm that hangs to the side.
    const scalar_t leftSwing =
        std::sin(kArmSwingRadiansPerCycle *
                 (phaseVariable - armSwingSettings_.phaseOffset)) *
        localVelXCommand;
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

  // The base-pose seam of the locomotion heuristics; see shapeBasePose(), which
  // the terminal and torso costs call too.
  return shapeBasePose(time, xNominal);
}

vector_t SwitchedModelReferenceManager::shapeBasePose(
    scalar_t time, const vector_t& xNominal) const {
  if (heuristicLayerPtr_->basePoseEmpty()) return xNominal;
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
  const vector2_t commandedVelocityInWorld = getCommandedVelocity(time);
  // Into the reference's own heading frame, so that "forward" means the robot's
  // forward rather than the world's x. The reference's heading is the right one
  // and not the measured one: the heuristics describe how the robot should
  // carry itself along the path it was told to take.
  const scalar_t referenceYaw = mpcRobotModelPtr_->getBaseHeading(xNominal);
  const scalar_t cosYaw = std::cos(referenceYaw);
  const scalar_t sinYaw = std::sin(referenceYaw);
  context.commandedVelocityInBaseFrame =
      vector2_t(cosYaw * commandedVelocityInWorld(0) +
                    sinYaw * commandedVelocityInWorld(1),
                -sinYaw * commandedVelocityInWorld(0) +
                    cosYaw * commandedVelocityInWorld(1));
  context.commandedYawRate = getCommandedYawRate(time);
  context.gaitPhase = getPhaseVariable(time);
  context.gaitPhaseRate = getPhaseVariableRate(time);

  const BasePoseOffset offset = heuristicLayerPtr_->basePoseOffset(context);
  vector_t shaped = xNominal;
  // LINT.IfChange(base_pose_heuristic_seam)
  // The heuristics' Euler boundary (quaternion_base_orientation/README.md,
  // section 4.2): getBaseOrientationEulerZyx() returns (yaw, pitch, roll),
  // Euler ZYX with YAW FIRST, so roll, the angle about x, is index 2
  // (kEulerAnglesZyxXIndex) and pitch is index 1 (kEulerAnglesZyxYIndex) - the
  // reverse of the (roll, pitch, yaw) order every textbook writes - and the
  // height is index 2 of the base position. This is the one place in the
  // subsystem that knows it.
  const vector3_t nominalEulerZyx =
      mpcRobotModelPtr_->getBaseOrientationEulerZyx(shaped);
  vector3_t eulerZyx = nominalEulerZyx;
  eulerZyx(kEulerAnglesZyxXIndex) += offset.roll;
  eulerZyx(kEulerAnglesZyxYIndex) += offset.pitch;
  mpcRobotModelPtr_->setBaseOrientationEulerZyx(shaped, eulerZyx);
  vector3_t offsetEulerZyxRates = vector3_t::Zero();
  offsetEulerZyxRates(kEulerAnglesZyxXIndex) = offset.rollRate;
  offsetEulerZyxRates(kEulerAnglesZyxYIndex) = offset.pitchRate;
  // The rate channel follows the attitude: the shaped attitude is Theta_n(t) +
  // delta(t), so its rate is the derivative of THAT. A state that stores the
  // body angular velocity (the whole body's) gets the body rate of Theta_n +
  // delta with the Euler rates dTheta_n/dt + d(delta)/dt
  // (shapedBodyAngularVelocity()), which keeps the reference's turn in the
  // world and adds the rate of a periodic tilt. A state that stores the
  // centroidal momentum gets the angular momentum of the tilt rate, I_G w_tilt
  // / m with the composite inertia latched at the solve, on top of the turn the
  // target calculator already wrote there. Left as they were, the reference
  // would describe an attitude that oscillates while its rate says it holds
  // still.
  if (storesBaseAngularVelocity(mpcRobotModelPtr_->getTuningLayout())) {
    mpcRobotModelPtr_->setBaseAngularVelocityLocal(
        shaped, shapedBodyAngularVelocity(
                    nominalEulerZyx,
                    mpcRobotModelPtr_->getBaseAngularVelocityLocal(xNominal),
                    eulerZyx, offsetEulerZyxRates));
  } else if (const std::optional<size_t> momentumRow =
                 normalizedAngularMomentumRow(
                     mpcRobotModelPtr_->getTuningLayout());
             momentumRow.has_value() && totalMass_ > 0.0 &&
             !offsetEulerZyxRates.isZero(0.0)) {
    shaped.segment<kRotationTangentDim>(*momentumRow) +=
        compositeInertia_ *
        tiltAngularVelocityInWorld(eulerZyx, offsetEulerZyxRates) / totalMass_;
  }
  vector3_t basePosition = mpcRobotModelPtr_->getBasePosition(shaped);
  basePosition(kZAxisIndex) += offset.height;
  mpcRobotModelPtr_->setBasePosition(shaped, basePosition);
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/locomotion_heuristics/BasePoseHeuristic.h:base_pose_offset_convention)
  // clang-format on
  return shaped;
}

vector3_t SwitchedModelReferenceManager::shapedBodyAngularVelocity(
    const vector3_t& nominalEulerZyx,
    const vector3_t& nominalBodyAngularVelocity,
    const vector3_t& shapedEulerZyx,
    const vector3_t& offsetEulerZyxRates) {
  const vector3_t nominalRates = eulerZyxRatesFromLocalAngularVelocity(
      nominalEulerZyx, nominalBodyAngularVelocity);
  return localAngularVelocityFromEulerZyxRates(
      shapedEulerZyx, vector3_t(nominalRates + offsetEulerZyxRates));
}

vector3_t SwitchedModelReferenceManager::tiltAngularVelocityInWorld(
    const vector3_t& shapedEulerZyx, const vector3_t& offsetEulerZyxRates) {
  return quaternionRotationMatrix<scalar_t>(
             quaternionFromEulerZyx(shapedEulerZyx)) *
         localAngularVelocityFromEulerZyxRates(shapedEulerZyx,
                                               offsetEulerZyxRates);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t SwitchedModelReferenceManager::getDesiredInput(
    const TargetTrajectories& targetTrajectories,
    const vector_t& /*state*/,
    scalar_t time) const {
  const contact_flag_t contactFlags = getContactFlags(time);
  if (heuristicLayerPtr_->wrenchEmpty()) {
    // Byte for byte the call the two input costs used to make for themselves.
    // Input-only on purpose: the nominal contact force is purely vertical and a
    // stance foot is flat, so it is the same in the world and local contact
    // frames, and the state-aware overload would add a forward-kinematics pass
    // per node and iteration for nothing.
    return weightCompensatingInput(pinocchioInterface_, contactFlags,
                                   *mpcRobotModelPtr_);
  }

  WrenchHeuristicContext context;
  context.contactFlags = contactFlags;
  context.numStanceFeet = numberOfLegsInContacts(contactFlags);
  context.commandedVelocity = getCommandedVelocity(time);
  context.commandedYawRate = getCommandedYawRate(time);
  // The same gravity weightCompensatingInput() uses, so that the offsets and
  // the reference they correct are measured in the same weight. It is a
  // constant in both places because the centroidal model carries no gravity
  // constant of its own.
  context.totalWeight = totalMass_ * kGravitationalAcceleration;
  // Each foot's duty factor over its OWN stride, lift-off to lift-off, from the
  // schedule this node is solved on. On a periodic gait that is the gait's duty
  // factor at every node, which is what makes the scaled reference average to
  // exactly the weight; see stanceDutyFactor(). The same whichever reference
  // manager built the schedule, so nothing has to be latched per solve.
  const ModeSchedule& schedule = getModeSchedule();
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    context.stanceDutyFactor[foot] = stanceDutyFactor(schedule, foot, time);
  }

  // Which path writes the reference is decided by what the listed heuristics
  // PRODUCE, not by the input parameterization: a vertical force is the same in
  // the world and in a flat foot's own frame, so it takes the cheap path even
  // under basis-vector inputs, while a horizontal one has to be rotated into
  // the local contact frame. See WrenchHeuristic::producesHorizontalForce().
  if (!heuristicLayerPtr_->wrenchNeedsWorldFrame()) {
    vector_t input = weightCompensatingInput(pinocchioInterface_, contactFlags,
                                             *mpcRobotModelPtr_);
    if (context.numStanceFeet == 0) return input;
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      if (!contactFlags[foot]) continue;
      const vector3_t offset = heuristicLayerPtr_->wrenchOffset(context, foot);
      if (offset.isZero()) continue;
      mpcRobotModelPtr_->setContactForce(
          input, mpcRobotModelPtr_->getContactForce(input, foot) + offset,
          foot);
    }
    return input;
  }

  // The world-frame path. Every setContactForceInWorldFrame() /
  // getContactForceInWorldFrame() call on the basis-vector model copies the
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
  const vector3_t weightCompensation(
      0.0, 0.0,
      context.totalWeight / static_cast<scalar_t>(context.numStanceFeet));
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    if (!contactFlags[foot]) continue;
    mpcRobotModelPtr_->setContactForceInWorldFrame(
        referenceState, input,
        weightCompensation + heuristicLayerPtr_->wrenchOffset(context, foot),
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

  scalar_t terrainHeight =
      adaptToCurrentGroundHeight(targetTrajectories, initState, initMode);

  swingTrajectoryPtr_->update(modeSchedule, terrainHeight);

  modeSchedule_ = modeSchedule;
  captureMeasuredState(initTime, initState);
}

}  // namespace ocs2::humanoid
