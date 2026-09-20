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

#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

#include "humanoid_common_mpc/contact_planning/ContactScheduleAdaptation.h"

#include <humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h>
#include <ocs2_core/misc/Numerics.h>

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
      mpcRobotModelPtr_(&mpcRobotModel) {}

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
  const auto it = std::upper_bound(modeSchedule_.eventTimes.begin(), modeSchedule_.eventTimes.end(), time);
  scalar_t nextEventTime = *it;
  scalar_t prevEventTime = *(it - 1);

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
  if (numerics::almost_eq(pitch, 0.0)) return flat;

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

std::optional<vector2_t> SwitchedModelReferenceManager::nominalFoothold(size_t contactIndex, scalar_t time) const {
  const scalar_t stepWidth = mpcRobotModelPtr_->modelSettings.nominalFootholdConfig.stepWidth;
  if (stepWidth <= 0.0 || !hasMeasuredState_) return std::nullopt;

  // Measured from the OTHER foot, which is the one on the ground: a whole step width to this foot's side of it, carried
  // forward at the operator's commanded velocity.
  //
  // Not measured from the base. In single support the base sits roughly over the stance foot, so half a step width
  // from the base is half a step width from the stance foot - half the separation intended - and the base moves
  // further over the stance foot with every step, so the feet converge. The stance foot is the one landmark in this
  // problem that does not move while the other foot swings, which is why the reduced-order planner places steps
  // relative to it too.
  const size_t stanceIndex = (contactIndex == CONTACT_LEFT_INDEX) ? CONTACT_RIGHT_INDEX : CONTACT_LEFT_INDEX;
  const vector2_t advance = getCommandedVelocity(time) * (time - lastSolveTime_);
  const scalar_t side = (contactIndex == CONTACT_LEFT_INDEX) ? 1.0 : -1.0;
  const scalar_t lateral = side * stepWidth;
  const vector2_t offsetInWorld(-std::sin(measuredBaseYaw_) * lateral, std::cos(measuredBaseYaw_) * lateral);
  return vector2_t(liftOffPositions_[stanceIndex] + advance + offsetInWorld);
}

std::optional<SwingFootReference> SwitchedModelReferenceManager::getSwingFootReference(size_t contactIndex, scalar_t time) const {
  if (isInContact(time, contactIndex)) return std::nullopt;
  const std::optional<std::pair<scalar_t, scalar_t>> phase = swingPhaseAtTime(modeSchedule_, contactIndex, time);
  if (!phase.has_value()) return std::nullopt;
  const auto [liftOffTime, touchDownTime] = *phase;
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
  if (mpcRobotModelPtr_->modelSettings.nominalFootholdConfig.stepWidth <= 0.0) return;
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
  hasMeasuredState_ = true;
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
  // The configured ground, and the only definition of it in the controller. This used to call
  // computeGroundHeightEstimate() and then overwrite the result with a hard-coded 0 on the next line, which left a
  // reader believing the swing trajectories tracked a measured ground height when they tracked a constant - and left
  // the contact-implicit terms free to be configured against a different constant entirely.
  const scalar_t terrainHeight = mpcRobotModelPtr_->modelSettings.terrainHeight;

  // adapt target Trajectories to current terrain height
  // Since they are published in the past the current observations ground height might have drifted.

  // Adapt the ground height difference for every state in the target Trajectories.
  // The height difference between last update and the current update is applied here
  // to prevent applying the same difference twice in case the trajectories have not been updated.
  for (size_t i = 0; i < targetTrajectories.stateTrajectory.size(); i++) {
    vector_t& targetState = targetTrajectories.stateTrajectory[i];
    scalar_t heightDifference = terrainHeight - previousGroundHeightEstimate_;
    mpcRobotModelPtr_->adaptBasePoseHeight(targetState, heightDifference);
  }
  previousGroundHeightEstimate_ = terrainHeight;
  return terrainHeight;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector_t SwitchedModelReferenceManager::getDesiredState(const TargetTrajectories& targetTrajectories,
                                                        const vector_t& state,
                                                        scalar_t time) const {
  vector_t xNominal = targetTrajectories.getDesiredState(time);

  if (armSwingReferenceActive_) {
    // Skip procedural arm swing when ACoM tracking is active; arm orientations
    // emerge naturally from the ACoM cost and would conflict with the generator.
    const ModelSettings& modelSettings = mpcRobotModelPtr_->modelSettings;
    if (!modelSettings.useComAndAcomTracking) {
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
  }
  return xNominal;
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
  const auto timeHorizon = finalTime - initTime;
  modeSchedule = gaitSchedulePtr_->getModeSchedule(initTime - timeHorizon, finalTime + timeHorizon);

  scalar_t terrainHeight = adaptToCurrentGroundHeight(targetTrajectories, initState, initMode);

  swingTrajectoryPtr_->update(modeSchedule, terrainHeight);

  modeSchedule_ = modeSchedule;
  captureMeasuredState(initTime, initState);
}

}  // namespace ocs2::humanoid
