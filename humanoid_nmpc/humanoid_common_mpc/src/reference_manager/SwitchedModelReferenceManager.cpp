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

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"

#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>
#include <ocs2_core/misc/Numerics.h>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

#include <algorithm>
#include <cmath>

namespace ocs2::humanoid {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
SwitchedModelReferenceManager::SwitchedModelReferenceManager(std::shared_ptr<GaitSchedule> gaitSchedulePtr,
                                                             std::shared_ptr<SwingTrajectoryPlanner> swingTrajectoryPtr,
                                                             const PinocchioInterface& pinocchioInterface,
                                                             const MpcRobotModelBase<scalar_t>& mpcRobotModel)
    : ReferenceManager(TargetTrajectories(), ModeSchedule()),
      gaitSchedulePtr_(std::move(gaitSchedulePtr)),
      swingTrajectoryPtr_(std::move(swingTrajectoryPtr)),
      // The reference manager gets a copy of the pinocchio model to use for initializing the ground height
      pinocchioInterface_(pinocchioInterface),
      mpcRobotModelPtr_(&mpcRobotModel) {
  // A constant of the robot, so it is read once here rather than being latched with the measurements. The yaw inertia
  // beside it is not a constant - it depends on the posture - and is latched per solve in captureMeasuredState().
  totalMass_ = pinocchio::computeTotalMass(pinocchioInterface_.getModel());
  // The ground starts where the task file puts it; from here on it is this manager's (setTerrainHeight).
  terrainHeight_.store(mpcRobotModel.modelSettings.terrainHeight);
  appliedTerrainHeight_.store(mpcRobotModel.modelSettings.terrainHeight);
  targetTerrainHeight_ = mpcRobotModel.modelSettings.terrainHeight;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void SwitchedModelReferenceManager::reset() {
  ReferenceManager::reset();
  // The ground is configuration and stays where it was last set. A target installed after the reset is taken from the
  // measured state, i.e. it is already a world pose on that ground, so there is no change of ground left to apply.
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
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
contact_flag_t SwitchedModelReferenceManager::getContactFlags(scalar_t time) const {
  return modeNumber2StanceLeg(this->getModeSchedule().modeAtTime(time));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t SwitchedModelReferenceManager::getPhaseVariable(scalar_t time) const {
  const std::vector<scalar_t>::const_iterator it = std::upper_bound(modeSchedule_.eventTimes.begin(), modeSchedule_.eventTimes.end(), time);
  // `time` outside the scheduled events, or a schedule with fewer than two of them, leaves nothing to interpolate
  // between: upper_bound returns end() past the last event and begin() before the first, and dereferencing either was
  // undefined. This was reachable before only through the procedural arm swing, which no legs-only robot runs; the
  // base-pose heuristics are a second caller and every robot can reach them, so the degenerate cases are answered
  // rather than stepped into. 0 is the start of a cycle, which is what a phase that cannot be computed should be.
  if (it == modeSchedule_.eventTimes.end() || it == modeSchedule_.eventTimes.begin()) return 0.0;
  const scalar_t nextEventTime = *it;
  const scalar_t prevEventTime = *(it - 1);
  if (!(nextEventTime > prevEventTime)) return 0.0;

  if (modeSchedule_.modeAtTime(time) == LF) {
    return (0.5 * (time - prevEventTime) / (nextEventTime - prevEventTime));
  } else if (modeSchedule_.modeAtTime(time) == RF) {
    return (0.5 + 0.5 * (time - prevEventTime) / (nextEventTime - prevEventTime));
  } else {
    if (modeSchedule_.modeAtTime(prevEventTime - 0.01) == LF) {
      return 0.5;
    } else {
      return 0;
    }
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
vector3_t SwitchedModelReferenceManager::getSwingFootPlaneNormal(size_t contactIndex, scalar_t time) const {
  const vector3_t flat(0.0, 0.0, 1.0);
  if (isInContact(time, contactIndex)) return flat;

  const scalar_t pitch = swingTrajectoryPtr_->getSwingPitchAngle(contactIndex, time);
  if (numerics::almost_eq(pitch, /*y=*/0.0)) return flat;

  // Heading of the foot: the planned foot yaw when the contact planner's heading model provides one, otherwise the
  // commanded base yaw, so that the tilt stays about the foot's lateral axis and does not leak into roll.
  scalar_t yaw = 0.0;
  const std::optional<SwingFootReference> swingReference = getSwingFootReference(contactIndex, time);
  if (swingReference.has_value() && swingReference->yaw.has_value()) {
    yaw = *swingReference->yaw;
  } else {
    const vector_t desiredState = getTargetTrajectories().getDesiredState(time);
    if (desiredState.size() == mpcRobotModelPtr_->getStateDim()) {
      yaw = mpcRobotModelPtr_->getBaseOrientationEulerZYX(desiredState)(0);
    }
  }

  // Toe-up by `pitch` is a rotation of -pitch about the foot's lateral (+y) axis, which tilts the sole normal backwards
  // along the heading. The result is a unit vector, as rotationMatrixDistanceToPlane expects.
  const scalar_t sp = std::sin(pitch);
  return vector3_t(-sp * std::cos(yaw), -sp * std::sin(yaw), std::cos(pitch));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::optional<vector2_t> SwitchedModelReferenceManager::getSwingFootVelocityReference(size_t contactIndex, scalar_t time) const {
  if (isInContact(time, contactIndex)) return std::nullopt;

  return getCommandedVelocity(time);
}

void SwitchedModelReferenceManager::setLocomotionHeuristicLayer(std::shared_ptr<LocomotionHeuristicLayer> layer) {
  if (layer != nullptr) heuristicLayerPtr_ = std::move(layer);
}

bool SwitchedModelReferenceManager::needsMeasuredState() const {
  // Every listed heuristic wants something from here, not only the foothold ones: the base-pose and wrench families
  // read the commanded yaw rate, which is recovered by inverting the target's momentum channel with the composite
  // yaw inertia this function's caller latches.
  return mpcRobotModelPtr_->modelSettings.nominalFootholdConfig.stepWidth > 0.0 || !heuristicLayerPtr_->empty();
}

std::pair<vector2_t, scalar_t> SwitchedModelReferenceManager::predictedBaseAt(scalar_t time) const {
  // The measurement carried forward by the command over the time until `time`, by the trapezoid of the command at the
  // two ends: exact for a constant command, and first-order right for one that is ramping.
  const scalar_t elapsed = std::max(time - lastSolveTime_, 0.0);
  const vector2_t meanVelocity = 0.5 * (getCommandedVelocity(lastSolveTime_) + getCommandedVelocity(time));
  const scalar_t meanYawRate = 0.5 * (getCommandedYawRate(lastSolveTime_) + getCommandedYawRate(time));
  return {vector2_t(measuredBasePosition_ + meanVelocity * elapsed), measuredBaseYaw_ + meanYawRate * elapsed};
}

FootholdHeuristicContext SwitchedModelReferenceManager::footholdContext(size_t contactIndex, scalar_t time) const {
  FootholdHeuristicContext context;
  context.contactIndex = contactIndex;
  context.side = (contactIndex == CONTACT_LEFT_INDEX) ? 1.0 : -1.0;
  const std::pair<vector2_t, scalar_t> predicted = predictedBaseAt(time);
  context.basePosition = predicted.first;
  context.baseYaw = predicted.second;
  context.measuredVelocity = measuredComVelocity_;
  context.commandedVelocity = getCommandedVelocity(time);
  context.commandedYawRate = getCommandedYawRate(time);
  context.comHeight = measuredComHeight_;
  context.stanceDuration = upcomingStanceDuration(modeSchedule_, contactIndex, time);
  return context;
}

std::optional<vector2_t> SwitchedModelReferenceManager::nominalFoothold(size_t contactIndex, scalar_t time) const {
  const scalar_t stepWidth = mpcRobotModelPtr_->modelSettings.nominalFootholdConfig.stepWidth;
  // Nothing has been measured yet, so nothing can be placed: this is the first solve, before captureMeasuredState().
  if (!hasMeasuredState_) return std::nullopt;
  const bool haveHeuristics = !heuristicLayerPtr_->footholdEmpty();
  // No opinion from either source: a robot with `stepWidth: 0` and no listed heuristic returns nullopt, and the foot
  // cost's xy weights stay switched off exactly as they were.
  if (stepWidth <= 0.0 && !haveHeuristics) return std::nullopt;

  const size_t stanceIndex = (contactIndex == CONTACT_LEFT_INDEX) ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX;
  const scalar_t side = (contactIndex == CONTACT_LEFT_INDEX) ? 1.0 : -1.0;
  const vector2_t& stanceFoot = liftOffPositions_[stanceIndex];

  if (!haveHeuristics) {
    // THE NOMINAL STEP. A whole step width to this foot's side of the stance foot, carried forward at the commanded
    // velocity over one step: from the stance foot's own touch-down to this one. Both ends are fixed instants, so the
    // target holds still through the swing. (It used to be carried forward over the time from the LAST SOLVE to the
    // touch-down, which shrinks to nothing as the swing ends: the target slid backwards through every swing and
    // converged on the stance foot's own position, a zero-length step at any speed.)
    //
    // Measured from the stance foot, not from the base: in single support the base sits roughly over the stance foot,
    // so half a step width from the base is half a step width from the stance foot - half the separation intended -
    // and the base moves further over the stance foot with every step, so the feet converge.
    //
    // One step of travel from the stance foot is where Raibert's rule puts the foot in steady state, with no
    // coefficient: the stance foot itself landed half a stance of travel ahead of the CoM one step earlier.
    const std::optional<scalar_t> stanceLanded = previousTouchDownTime(modeSchedule_, stanceIndex, time);
    const std::optional<std::pair<scalar_t, scalar_t>> swing = swingPhaseAtTime(modeSchedule_, contactIndex, time - 1e-9);
    const scalar_t stepStart = stanceLanded.has_value() ? *stanceLanded : (swing.has_value() ? swing->first : lastSolveTime_);
    const scalar_t yaw = predictedBaseAt(time).second;
    const vector2_t advance = getCommandedVelocity(time) * std::max(time - stepStart, 0.0);
    const vector2_t lateral(-std::sin(yaw) * side * stepWidth, std::cos(yaw) * side * stepWidth);
    return vector2_t(stanceFoot + advance + lateral);
  }

  // BLEDT'S H_r: the hip at the moment of landing, plus the listed heuristics' offsets. See the header.
  const FootholdHeuristicContext context = footholdContext(contactIndex, time);
  const vector2_t offsets = heuristicLayerPtr_->footholdOffset(context);
  if (heuristicLayerPtr_->footholdMovesAnchor()) {
    // hip_centered_stepping is listed: its offset is the hip relative to the predicted base, so the base is the anchor.
    return vector2_t(context.basePosition + offsets);
  }
  // Along the predicted heading, the predicted base; across it, the stance foot plus a whole step width, which keeps
  // the feet apart for the reason given for the nominal step above. Create() refuses a zero step width here.
  const vector2_t forward(std::cos(context.baseYaw), std::sin(context.baseYaw));
  const vector2_t leftward(-std::sin(context.baseYaw), std::cos(context.baseYaw));
  const vector2_t anchor = forward * forward.dot(context.basePosition) + leftward * (leftward.dot(stanceFoot) + side * stepWidth);
  return vector2_t(anchor + offsets);
}

std::optional<SwingFootReference> SwitchedModelReferenceManager::getSwingFootReference(size_t contactIndex, scalar_t time) const {
  if (isInContact(time, contactIndex)) return std::nullopt;
  const std::optional<std::pair<scalar_t, scalar_t>> phase = swingPhaseAtTime(modeSchedule_, contactIndex, time);
  if (!phase.has_value()) return std::nullopt;
  const scalar_t liftOffTime = phase->first;
  const scalar_t touchDownTime = phase->second;
  const scalar_t duration = touchDownTime - liftOffTime;
  if (duration <= 1e-6) return std::nullopt;

  // The swing starts where the foot actually lifted off, not where the nominal offset would have put it. Blending
  // between two nominal points describes a path the foot is not on, and the step the cost then demands at lift-off is
  // whatever error had accumulated by then.
  const std::optional<vector2_t> target = nominalFoothold(contactIndex, touchDownTime);
  if (!target.has_value()) return std::nullopt;
  const vector2_t start = liftOffPositions_[contactIndex];

  // The same cubic blend the planned-foothold path uses: p'(0) = 1, p'(1) = 0, so the foot leaves the ground moving
  // with the step and settles onto the target instead of arriving at speed.
  const scalar_t tau = std::clamp((time - liftOffTime) / duration, 0.0, 1.0);
  const scalar_t tau2 = tau * tau;
  const scalar_t blend = -tau2 * tau + tau2 + tau;
  const scalar_t blendRate = (-3.0 * tau2 + 2.0 * tau + 1.0) / duration;
  const vector2_t delta = *target - start;

  SwingFootReference reference;
  reference.position.head<2>() = start + blend * delta;
  reference.position(2) = swingTrajectoryPtr_->getZpositionConstraint(contactIndex, time);
  reference.linearVelocity.head<2>() = blendRate * delta;
  reference.linearVelocity(2) = swingTrajectoryPtr_->getZvelocityConstraint(contactIndex, time);
  return reference;
}

void SwitchedModelReferenceManager::captureMeasuredState(scalar_t initTime, const vector_t& initState) {
  // The guard is now "does anything downstream want this", rather than the nominal step width alone: the foothold
  // heuristics are the second consumer, and they are available on robots whose step width is zero.
  if (!needsMeasuredState()) return;
  lastSolveTime_ = initTime;
  measuredBasePosition_ = mpcRobotModelPtr_->getBasePosition(initState).head<2>();
  measuredBaseYaw_ = mpcRobotModelPtr_->getBaseOrientationEulerZYX(initState)(0);

  // Every foot that is in contact records where it is. The last value a foot recorded before it began to swing is
  // therefore where it lifted off from, with no event to detect and nothing to reset.
  const vector_t q = mpcRobotModelPtr_->getGeneralizedCoordinates(initState);
  const std::vector<vector3_t> feet = computeContactPositions<scalar_t>(q, pinocchioInterface_, *mpcRobotModelPtr_);
  const contact_flag_t contacts = getContactFlags(initTime);
  for (size_t foot = 0; foot < N_CONTACTS && foot < feet.size(); ++foot) {
    if (contacts[foot]) liftOffPositions_[foot] = feet[foot].head<2>();
  }

  // The composite yaw inertia inverts the target's momentum channel into the commanded yaw rate, which predictedBaseAt()
  // needs for the heading at touch-down. The nominal step reads that heading too, so it is latched whenever anything
  // here is wanted: latching it only for a listed heuristic would lay the nominal step across the MEASURED heading on a
  // robot with none listed, and listing an unrelated base-pose heuristic would then move the feet.
  const pinocchio::Model& model = pinocchioInterface_.getModel();
  pinocchio::Data& data = pinocchioInterface_.getData();
  pinocchio::ccrba(model, data, q, vector_t::Zero(model.nv));  // composite inertia about the CoM, in the world frame
  yawInertia_ = data.Ig.inertia().matrix()(2, 2);

  // The rest is for the heuristics only, and costs a center-of-mass pass, so it is skipped entirely when none is listed.
  if (!heuristicLayerPtr_->empty()) {
    // The linear part of the state's momentum channel IS the center-of-mass velocity: the state carries the
    // NORMALIZED momentum h = [p/m, L/m], so no division by the mass is needed here.
    measuredComVelocity_ = mpcRobotModelPtr_->getBaseComLinearVelocity(initState).head<2>();
    // The capture point's pendulum length, measured the way every LIP consumer measures its nominal one: above the
    // MEAN FOOT HEIGHT rather than above the world origin, so that it is the same number on a robot standing on a box
    // as on one standing on the floor.
    measuredComHeight_ = computeComHeightAboveFeet(q, pinocchioInterface_, *mpcRobotModelPtr_);
  }
  hasMeasuredState_ = true;
}

scalar_t SwitchedModelReferenceManager::getCommandedYawRate(scalar_t time) const {
  // Nothing has measured the composite inertia yet, so the momentum channel cannot be inverted. Zero is the honest
  // answer and it is also the safe one: every heuristic that reads this is linear in it and therefore contributes
  // nothing at all rather than something wrong.
  if (yawInertia_ <= 0.0 || totalMass_ <= 0.0) return 0.0;
  const TargetTrajectories& targetTrajectories = getTargetTrajectories();
  if (targetTrajectories.empty()) return 0.0;
  const vector_t desiredState = targetTrajectories.getDesiredState(time);
  if (desiredState.size() < 6) return 0.0;
  // h_z = I_zz * psidot / m, written by CentroidalMpcTargetTrajectoriesCalculator; inverted here.
  return totalMass_ * mpcRobotModelPtr_->getBaseComVelocity(desiredState)(5) / yawInertia_;
}

vector2_t SwitchedModelReferenceManager::getCommandedVelocity(scalar_t time) const {
  const TargetTrajectories& targetTrajectories = getTargetTrajectories();
  if (targetTrajectories.empty()) return vector2_t::Zero();
  // Relying on the convention that the first two elements of the target state are the CoM XY velocity command.
  const vector_t desiredState = targetTrajectories.getDesiredState(time);
  if (desiredState.size() < 2) return vector2_t::Zero();
  return desiredState.head<2>();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t SwitchedModelReferenceManager::adaptToCurrentGroundHeight(TargetTrajectories& targetTrajectories,
                                                                   const vector_t& /*initState*/,
                                                                   size_t /*initMode*/) {
  // The ground, and the only definition of it in the controller: the task file's `terrainHeight` at start-up, moved by
  // setTerrainHeight() on a hot reload. This used to call computeGroundHeightEstimate() and then overwrite the result
  // with a hard-coded 0 on the next line, which left a reader believing the swing trajectories tracked a measured ground
  // height when they tracked a constant - and it later read the launch-time ModelSettings value, which a hot reload of
  // `terrainHeight` moved for the contact-implicit terms but not for the swing trajectories, the landing targets or the
  // base height. Read once here, so every reference of this run is built on the same ground, and recorded as the
  // applied one for the contact-implicit terms to follow.
  const scalar_t terrainHeight = terrainHeight_.load();
  appliedTerrainHeight_.store(terrainHeight);

  // The target in use stands on the ground of the last run (targetTerrainHeight_): move it by the change, once. Only
  // the change: this target is kept from run to run until a new one is published, and each run would otherwise lift it
  // again. A target published after this run was built on the ground applied here (the target calculators read
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

vector_t SwitchedModelReferenceManager::getDesiredState(const TargetTrajectories& targetTrajectories,
                                                        const vector_t& state,
                                                        scalar_t time) const {
  vector_t xNominal = targetTrajectories.getDesiredState(time);

  // Off on a legs-only robot, and off while com_and_acom_tracking_cost is listed: the arm orientations then emerge from
  // the ACoM cost and would conflict with the generator. The interface decides; see setArmSwingReferenceActive().
  if (armSwingReferenceActive_) {
    const ModelSettings& modelSettings = mpcRobotModelPtr_->modelSettings;
    scalar_t phaseVariable = this->getPhaseVariable(time);
    vector_t desiredJointAngles = mpcRobotModelPtr_->getJointAngles(xNominal);

    vector3_t linVelCommand = mpcRobotModelPtr_->getBaseComLinearVelocity(xNominal);
    scalar_t currentEulerZ = mpcRobotModelPtr_->getBasePose(state)[3];

    const scalar_t localVelXCommand = (std::cos(currentEulerZ) * linVelCommand[0] + std::sin(currentEulerZ) * linVelCommand[1]);

    scalar_t gaitCycleFactor = std::sin(2 * M_PI * (phaseVariable - 0.15)) * localVelXCommand;
    desiredJointAngles[modelSettings.j_l_shoulder_y_index] += -0.15 * gaitCycleFactor;
    desiredJointAngles[modelSettings.j_r_shoulder_y_index] += 0.15 * gaitCycleFactor;
    desiredJointAngles[modelSettings.j_l_elbow_y_index] += -0.15 * gaitCycleFactor;
    desiredJointAngles[modelSettings.j_r_elbow_y_index] += 0.15 * gaitCycleFactor;

    mpcRobotModelPtr_->setJointAngles(xNominal, desiredJointAngles);
  }

  // The base-pose seam of the locomotion heuristics; see shapeBasePose(), which the terminal and torso costs call too.
  return shapeBasePose(time, xNominal);
}

vector_t SwitchedModelReferenceManager::shapeBasePose(scalar_t time, const vector_t& xNominal) const {
  if (heuristicLayerPtr_->basePoseEmpty()) return xNominal;
  // Here, per node, at the point of use - not in the target-trajectory calculator, where the roll and pitch zeros are
  // written: that trajectory has three knots over the whole horizon and could not carry a limit cycle at gait
  // frequency, it is published one solve ahead of the solver that reads it, and it is also what the operator's
  // delta-pose commands are measured against. The zeros there stay; this writes on top of them.
  BasePoseHeuristicContext context;
  // The OPERATOR's command, through the same accessor the foothold and wrench seams use. Not the target's momentum
  // channel: under online contact planning the planned_com_override rule rewrites that with the plan's own CoM
  // velocity, which sways with the gait, and Bledt's H_Theta and H_z are functions of the commanded velocity.
  const vector2_t commandedVelocityInWorld = getCommandedVelocity(time);
  // Into the reference's own yaw frame, so that "forward" means the robot's forward rather than the world's x. The
  // reference's yaw is the right one and not the measured one: the heuristics describe how the robot should carry
  // itself along the path it was told to take.
  const scalar_t referenceYaw = mpcRobotModelPtr_->getBasePose(xNominal)(3);
  const scalar_t cosYaw = std::cos(referenceYaw);
  const scalar_t sinYaw = std::sin(referenceYaw);
  context.commandedVelocityInBaseFrame = vector2_t(cosYaw * commandedVelocityInWorld(0) + sinYaw * commandedVelocityInWorld(1),
                                                   -sinYaw * commandedVelocityInWorld(0) + cosYaw * commandedVelocityInWorld(1));
  context.commandedYawRate = getCommandedYawRate(time);
  context.gaitPhase = getPhaseVariable(time);

  const BasePoseOffset offset = heuristicLayerPtr_->basePoseOffset(context);
  vector_t shaped = xNominal;
  vector6_t basePose = mpcRobotModelPtr_->getBasePose(shaped);
  // LINT.IfChange(base_pose_heuristic_seam)
  // getBasePose() returns [x, y, z, yaw, pitch, roll]: Euler ZYX with YAW FIRST, so roll is index 5 and pitch is
  // index 4 - the reverse of the (roll, pitch, yaw) order every textbook writes. This is the one place in the
  // subsystem that knows it.
  basePose(5) += offset.roll;
  basePose(4) += offset.pitch;
  basePose(2) += offset.height;
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/locomotion_heuristics/BasePoseHeuristic.h:base_pose_offset_convention)
  // clang-format on
  mpcRobotModelPtr_->setBasePose(shaped, basePose);
  return shaped;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t SwitchedModelReferenceManager::getDesiredInput(const TargetTrajectories& /*targetTrajectories*/,
                                                        const vector_t& state,
                                                        scalar_t time) const {
  const contact_flag_t contactFlags = getContactFlags(time);
  if (heuristicLayerPtr_->wrenchEmpty()) {
    // Byte for byte the call the two input costs used to make for themselves. Input-only on purpose: the nominal
    // contact force is purely vertical and a stance foot is flat, so it is the same in the world and local contact
    // frames, and the state-aware overload would add a forward-kinematics pass per node and iteration for nothing.
    return weightCompensatingInput(pinocchioInterface_, contactFlags, *mpcRobotModelPtr_);
  }

  WrenchHeuristicContext context;
  context.contactFlags = contactFlags;
  context.numStanceFeet = numberOfLegsInContacts(contactFlags);
  context.commandedVelocity = getCommandedVelocity(time);
  context.commandedYawRate = getCommandedYawRate(time);
  // The same 9.81 weightCompensatingInput() uses, so that the offsets and the reference they correct are measured in
  // the same weight. It is a literal in both places because the centroidal model carries no gravity constant of its own.
  context.totalWeight = totalMass_ * 9.81;
  // Each foot's duty factor over its OWN stride, lift-off to lift-off, from the schedule this node is solved on. On a
  // periodic gait that is the gait's duty factor at every node, which is what makes the scaled reference average to
  // exactly the weight; see stanceDutyFactor(). The same whichever reference manager built the schedule, so nothing
  // has to be latched per solve.
  const ModeSchedule& schedule = getModeSchedule();
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    context.stanceDutyFactor[foot] = stanceDutyFactor(schedule, foot, time);
  }

  // Which path writes the reference is decided by what the listed heuristics PRODUCE, not by the input
  // parameterization: a vertical force is the same in the world and in a flat foot's own frame, so it takes the cheap
  // path even under basis-vector inputs, while a horizontal one has to be rotated into the local contact frame. See
  // WrenchHeuristic::producesHorizontalForce().
  if (!heuristicLayerPtr_->wrenchNeedsWorldFrame()) {
    vector_t input = weightCompensatingInput(pinocchioInterface_, contactFlags, *mpcRobotModelPtr_);
    if (context.numStanceFeet == 0) return input;
    for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
      if (!contactFlags[foot]) continue;
      const vector3_t offset = heuristicLayerPtr_->wrenchOffset(context, foot);
      if (offset.isZero()) continue;
      mpcRobotModelPtr_->setContactForce(input, mpcRobotModelPtr_->getContactForce(input, foot) + offset, foot);
    }
    return input;
  }

  // The world-frame path. Every setContactForceInWorldFrame() / getContactForceInWorldFrame() call on the
  // basis-vector model copies the whole pinocchio::Data and runs a forward-kinematics pass, so the total force is
  // assembled here and written ONCE per stance foot rather than being built by weightCompensatingInput() and then
  // read back and re-written - which would cost three passes per foot instead of one. Starting from a zero input is
  // exact, because weightCompensatingInput() leaves everything except the stance contact forces at zero.
  vector_t input = vector_t::Zero(mpcRobotModelPtr_->getInputDim());
  if (context.numStanceFeet == 0) return input;
  const vector3_t weightCompensation(0.0, 0.0, context.totalWeight / static_cast<scalar_t>(context.numStanceFeet));
  for (size_t foot = 0; foot < N_CONTACTS; ++foot) {
    if (!contactFlags[foot]) continue;
    mpcRobotModelPtr_->setContactForceInWorldFrame(state, input, weightCompensation + heuristicLayerPtr_->wrenchOffset(context, foot),
                                                   foot);
  }
  return input;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void SwitchedModelReferenceManager::modifyReferences(scalar_t initTime,
                                                     scalar_t finalTime,
                                                     const vector_t& initState,
                                                     size_t initMode,
                                                     TargetTrajectories& targetTrajectories,
                                                     ModeSchedule& modeSchedule) {
  const scalar_t timeHorizon = finalTime - initTime;
  modeSchedule = gaitSchedulePtr_->getModeSchedule(initTime - timeHorizon, finalTime + timeHorizon);

  scalar_t terrainHeight = adaptToCurrentGroundHeight(targetTrajectories, initState, initMode);

  swingTrajectoryPtr_->update(modeSchedule, terrainHeight);

  modeSchedule_ = modeSchedule;
  captureMeasuredState(initTime, initState);
}

}  // namespace ocs2::humanoid
