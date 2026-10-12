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

#include "humanoid_common_mpc/command/TargetTrajectoriesCalculatorBase.h"

#include <algorithm>  // For std::clamp
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"

#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {
namespace {

/** sin(x) / x, 1 at x = 0; its two-term series below |x| = 1e-4, where the next term is below 1e-18. */
scalar_t sinc(scalar_t x) {
  return std::abs(x) < 1.0e-4 ? 1.0 - x * x / 6.0 : std::sin(x) / x;
}

/** `vector` (x, y) turned by `angle` about z: R_z(angle) vector; exactly `vector` at angle 0. */
vector2_t rotatedAboutZ(scalar_t angle, const vector2_t& vector) {
  const scalar_t cosine = std::cos(angle);
  const scalar_t sine = std::sin(angle);
  return vector2_t(cosine * vector(0) - sine * vector(1), sine * vector(0) + cosine * vector(1));
}

/** A segment of a velocity command's reference: a constant planar twist held for a time. Passive data. */
struct TwistSegment {
  /** [s] From the first knot to the segment's end. */
  scalar_t endOffset = 0.0;
  /** [s] */
  scalar_t duration = 0.0;
  /** [m/s] The world-frame velocity at the segment's start. */
  vector2_t startVelocityWorld = vector2_t::Zero();
  /** [rad/s] */
  scalar_t yawRate = 0.0;
};

/**
 * [rad/s] `yawRate` limited to the rate that turns a segment of `duration` [s] by
 * TargetTrajectoriesCalculatorBase::kMaxSegmentYawChange: unchanged at any rate a robot turns at, and not a number for
 * one that is not.
 */
scalar_t clampedSegmentYawRate(scalar_t yawRate, scalar_t duration) {
  const scalar_t limit = TargetTrajectoriesCalculatorBase::kMaxSegmentYawChange / duration;
  return std::clamp(yawRate, -limit, limit);
}

}  // namespace

TargetTrajectoriesCalculatorBase::TargetTrajectoriesCalculatorBase(const ReferenceSettings& referenceSettings,
                                                                   const vector_t& defaultJointState,
                                                                   const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                   scalar_t mpcHorizon)
    : mpcRobotModelPtr_(mpcRobotModel.clone()), targetJointState_(defaultJointState), mpcHorizon_(mpcHorizon) {
  ABSL_CHECK_EQ(static_cast<size_t>(targetJointState_.size()), mpcRobotModel.getJointDim())
      << "TargetTrajectoriesCalculatorBase: the default joint state has one entry per MPC joint (defaultJointStateFromConfig())";
  applyCommandLimits(referenceSettings);
}

void TargetTrajectoriesCalculatorBase::applyCommandLimits(const ReferenceSettings& referenceSettings) {
  // LINT.IfChange(apply_command_limits)
  defaultBaseHeight_.store(referenceSettings.defaultBaseHeight);
  targetRotationVelocity_.store(referenceSettings.targetRotationVelocity);
  targetDisplacementVelocity_.store(referenceSettings.targetDisplacementVelocity);
  maxDisplacementVelocityX_.store(referenceSettings.maxDisplacementVelocityX);
  maxDisplacementVelocityY_.store(referenceSettings.maxDisplacementVelocityY);
  maxDeltaPelvisHeight_.store(referenceSettings.maxDeltaPelvisHeight);
  maxRotationVelocity_.store(referenceSettings.maxRotationVelocity);
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/reference/ReferenceFromConfig.cpp:hot_reference_file_fields)
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TargetTrajectoriesCalculatorBase::reset() {
  filteredVelocityCommand_.setZero();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TargetTrajectoriesCalculatorBase::setTerrainHeightSource(std::function<scalar_t()> terrainHeightSource) {
  std::lock_guard<std::mutex> lock(terrainHeightSourceMutex_);
  terrainHeightSource_ = std::move(terrainHeightSource);
}

scalar_t TargetTrajectoriesCalculatorBase::getTerrainHeight() const {
  std::lock_guard<std::mutex> lock(terrainHeightSourceMutex_);
  return terrainHeightSource_ ? terrainHeightSource_() : mpcRobotModelPtr_->modelSettings.terrainHeight;
}

scalar_t TargetTrajectoriesCalculatorBase::commandedBaseHeight(scalar_t commandedPelvisHeight) const {
  const scalar_t heightAboveGround = commandedPelvisHeight > kMinCommandedPelvisHeight ? commandedPelvisHeight : defaultBaseHeight_.load();
  return getTerrainHeight() + heightAboveGround;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::getDeltaBaseTarget(const vector4_t& commadLinePoseTarget,
                                                               const vector6_t& currentPoseTarget) const {
  vector_t target(6);

  // X facing forward to the robot, Y to the left side in the baseFrame
  const scalar_t baseFrameDeltaX = commadLinePoseTarget(0);
  const scalar_t baseFrameDeltaY = commadLinePoseTarget(1);
  const scalar_t currentEulerZ = currentPoseTarget(3);

  const scalar_t globalFrameDeltaX = std::cos(currentEulerZ) * baseFrameDeltaX - std::sin(currentEulerZ) * baseFrameDeltaY;
  const scalar_t globalFrameDeltaY = std::sin(currentEulerZ) * baseFrameDeltaX + std::cos(currentEulerZ) * baseFrameDeltaY;

  // base p_x, p_y are relative to current state
  target(0) = currentPoseTarget(0) + globalFrameDeltaX;
  target(1) = currentPoseTarget(1) + globalFrameDeltaY;
  // base z relative to the default height above the ground
  const scalar_t maxDeltaPelvisHeight = maxDeltaPelvisHeight_;
  const scalar_t deltaPelvisHeight = std::clamp(commadLinePoseTarget(2), -maxDeltaPelvisHeight, maxDeltaPelvisHeight);
  target(2) = getTerrainHeight() + defaultBaseHeight_ + deltaPelvisHeight;
  // theta_z relative to current
  target(3) = currentPoseTarget(3) + commadLinePoseTarget(3) * M_PI / 180.0;
  target(4) = 0.0;
  target(5) = 0.0;

  return target;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::getCurrentBasePoseTarget(const vector_t& state) const {
  // Level: the target trajectories start from the current position and heading, with the torso's roll and pitch zero.
  vector6_t currentPoseTarget;
  currentPoseTarget << mpcRobotModelPtr_->getBasePosition(state), mpcRobotModelPtr_->getBaseHeading(state), 0.0, 0.0;
  return currentPoseTarget;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector4_t TargetTrajectoriesCalculatorBase::filterAndTransformVelCommandToLocal(const vector4_t& commandedVelLocal,
                                                                                scalar_t currentEulerZ,
                                                                                scalar_t filterAlpha) {
  filteredVelocityCommand_ = filteredVelocityCommand_ * filterAlpha + commandedVelLocal * (1 - filterAlpha);

  vector4_t globalTargetVel = filteredVelocityCommand_;

  globalTargetVel(0) = std::cos(currentEulerZ) * filteredVelocityCommand_[0] - std::sin(currentEulerZ) * filteredVelocityCommand_[1];
  globalTargetVel(1) = std::sin(currentEulerZ) * filteredVelocityCommand_[0] + std::cos(currentEulerZ) * filteredVelocityCommand_[1];

  return globalTargetVel;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::integrateTargetBasePose(
    const vector6_t& currentPose, const vector2_t& startVelocityWorld, scalar_t yawRate, scalar_t baseHeight, scalar_t deltaT) const {
  vector6_t targetPose = currentPose;
  const vector2_t displacement = planarArcDisplacement(startVelocityWorld, yawRate, deltaT);
  targetPose[0] += displacement(0);
  targetPose[1] += displacement(1);
  // A world height already, ground included (commandedBaseHeight()). The pelvis-height fallback is not applied to it a
  // second time, as it used to be: on ground below z = 0.1 m a correct world height would read as "no command".
  targetPose[2] = baseHeight;
  targetPose[3] += yawRate * deltaT;
  targetPose[4] = 0.0;
  targetPose[5] = 0.0;
  return targetPose;
}

vector2_t TargetTrajectoriesCalculatorBase::planarArcDisplacement(const vector2_t& startVelocityWorld,
                                                                  scalar_t yawRate,
                                                                  scalar_t duration) {
  // int_0^T R_z(omega s) ds = T sinc(theta / 2) R_z(theta / 2), theta = omega T: [[sin(theta), -(1 - cos(theta))], [1 -
  // cos(theta), sin(theta)]] / omega, with sin(theta) = 2 sin(theta / 2) cos(theta / 2) and 1 - cos(theta) = 2 sin(theta /
  // 2)^2. At omega = 0 the rotation is the identity and the sinc 1, so the displacement is T v_0 bit for bit.
  const scalar_t halfTurn = 0.5 * yawRate * duration;
  return (duration * sinc(halfTurn)) * rotatedAboutZ(halfTurn, startVelocityWorld);
}

std::vector<TargetTrajectoriesCalculatorBase::PoseCommandKnot> TargetTrajectoriesCalculatorBase::subdivideYawKnots(
    scalar_t startTime, const vector6_t& startPose, scalar_t targetTime, const vector6_t& targetPose) {
  vector6_t endPose = targetPose;
  const scalar_t yawChange = targetPose(3) - startPose(3);
  size_t parts = 1;
  if (std::isfinite(yawChange)) {
    endPose(3) = startPose(3) + std::clamp(yawChange, -kMaxSegmentYawChange, kMaxSegmentYawChange);
    const scalar_t clampedYawChange = std::abs(endPose(3) - startPose(3));
    // At most kMaxKnotSubdivisions: the clamped turn is at most kMaxKnotSubdivisions kMaxKnotYawSpacing.
    const scalar_t yawParts = std::ceil(clampedYawChange / kMaxKnotYawSpacing);
    if (yawParts > 1.0) parts = std::min(static_cast<size_t>(yawParts), kMaxKnotSubdivisions);
  }
  std::vector<PoseCommandKnot> knots;
  knots.reserve(parts + 1);
  knots.push_back(PoseCommandKnot{.time = startTime, .basePose = startPose});
  for (size_t part = 1; part < parts; ++part) {
    const scalar_t fraction = static_cast<scalar_t>(part) / static_cast<scalar_t>(parts);
    knots.push_back(
        PoseCommandKnot{.time = startTime + fraction * (targetTime - startTime), .basePose = startPose + fraction * (endPose - startPose)});
  }
  knots.push_back(PoseCommandKnot{.time = targetTime, .basePose = endPose});
  return knots;
}

size_t TargetTrajectoriesCalculatorBase::knotSubdivisionCount(const KnotSegmentExtent& extent) {
  const scalar_t yawChange = std::abs(extent.yawChange);
  const scalar_t arcLength = std::abs(extent.arcLength);
  const scalar_t targetSpeed = std::abs(extent.targetSpeed);
  if (!std::isfinite(yawChange) || !std::isfinite(arcLength) || !std::isfinite(targetSpeed)) return 1;
  // Knots dpsi = yawChange / n apart on the arc of radius rho = arcLength / yawChange: the chord between them is off the
  // arc by at most rho (1 - cos(dpsi / 2)) <= rho dpsi^2 / 8 = arcLength yawChange / (8 n^2), and the interpolation of
  // their velocity targets R_z(psi_k) v_c off R_z(psi(t)) v_c by targetSpeed (1 - cos(dpsi / 2)) <= targetSpeed
  // yawChange^2 / (8 n^2), each at the middle of the interval.
  const scalar_t parts =
      std::ceil(std::max({yawChange / kMaxKnotYawSpacing, std::sqrt(arcLength * yawChange / (8.0 * kMaxKnotPositionDeviation)),
                          yawChange * std::sqrt(targetSpeed / (8.0 * kMaxKnotVelocityDeviation))}));
  if (parts <= 1.0) return 1;
  return parts >= static_cast<scalar_t>(kMaxKnotSubdivisions) ? kMaxKnotSubdivisions : static_cast<size_t>(parts);
}

std::vector<TargetTrajectoriesCalculatorBase::VelocityCommandKnot> TargetTrajectoriesCalculatorBase::velocityCommandKnots(
    const vector4_t& commandedVelocities, scalar_t initTime, const vector_t& initState, const BasePlanarMotion& measuredMotion) {
  // The level pose the reference starts from, at the commanded pelvis height (or default_base_height) above the ground
  // the reference manager stands the robot on.
  vector6_t startPose = getCurrentBasePoseTarget(initState);
  const scalar_t startHeading = startPose(3);
  const vector4_t commandGlobal = filterAndTransformVelCommandToLocal(commandedVelocities, startHeading, /*filterAlpha=*/0.8);
  const vector2_t commandWorld(commandGlobal(0), commandGlobal(1));
  const scalar_t commandYawRate = commandGlobal(3);
  const scalar_t baseHeight = commandedBaseHeight(commandGlobal(2));
  startPose(2) = baseHeight;
  // The measured motion, in the world frame at the start heading like the command.
  const vector2_t measuredWorld = rotatedAboutZ(startHeading, measuredMotion.velocityInHeadingFrame);

  // The first segment holds the mean of the measured and the commanded twist, the second the command, from the
  // intermediate knot's heading on.
  const scalar_t intermediateOffset = 0.7 * mpcHorizon_;
  TwistSegment first;
  first.endOffset = intermediateOffset;
  first.duration = intermediateOffset;
  first.startVelocityWorld = vector2_t((measuredWorld(0) + commandWorld(0)) / 2, (measuredWorld(1) + commandWorld(1)) / 2);
  // Each segment turns by at most kMaxSegmentYawChange, so that its knots are at most kMaxKnotYawSpacing apart for any
  // measurement (the class comment).
  first.yawRate = clampedSegmentYawRate((measuredMotion.headingRate + commandYawRate) / 2, first.duration);
  TwistSegment second;
  second.endOffset = mpcHorizon_;
  second.duration = mpcHorizon_ - intermediateOffset;
  second.yawRate = clampedSegmentYawRate(commandYawRate, second.duration);

  std::vector<VelocityCommandKnot> knots;
  knots.push_back(VelocityCommandKnot{.time = initTime,
                                      .basePose = startPose,
                                      .commandedVelocityWorld = commandWorld,
                                      .commandedYawRate = commandYawRate,
                                      .headingChange = 0.0,
                                      .secondSegmentFraction = 0.0});
  const std::array<bool, 2> isSecond = {false, true};
  for (const bool inSecond : isSecond) {
    TwistSegment& segment = inSecond ? second : first;
    const VelocityCommandKnot segmentStart = knots.back();
    // The command turns with the knots: it is given in the heading frame (design README section 2.9).
    if (inSecond) segment.startVelocityWorld = rotatedAboutZ(segmentStart.headingChange, commandWorld);
    const size_t parts = knotSubdivisionCount(KnotSegmentExtent{.yawChange = segment.yawRate * segment.duration,
                                                                .arcLength = segment.startVelocityWorld.norm() * segment.duration,
                                                                .targetSpeed = commandWorld.norm()});
    for (size_t part = 1; part <= parts; ++part) {
      // Every knot of the segment from its start, on the same arc; the last one at the segment's end, exactly.
      const bool last = part == parts;
      const scalar_t offset = last ? segment.duration : segment.duration * static_cast<scalar_t>(part) / static_cast<scalar_t>(parts);
      VelocityCommandKnot knot;
      knot.time = last ? initTime + segment.endOffset : segmentStart.time + offset;
      knot.basePose = integrateTargetBasePose(segmentStart.basePose, segment.startVelocityWorld, segment.yawRate, baseHeight, offset);
      knot.headingChange = knot.basePose(3) - startHeading;
      knot.commandedVelocityWorld = rotatedAboutZ(knot.headingChange, commandWorld);
      knot.commandedYawRate = commandYawRate;
      knot.secondSegmentFraction = !inSecond ? 0.0 : (last ? 1.0 : static_cast<scalar_t>(part) / static_cast<scalar_t>(parts));
      knots.push_back(knot);
    }
  }
  return knots;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t TargetTrajectoriesCalculatorBase::estimateTimeToTarget(const vector_t& desiredBaseDisplacement) const {
  const scalar_t& dx = desiredBaseDisplacement(0);
  const scalar_t& dy = desiredBaseDisplacement(1);
  const scalar_t& dyaw = desiredBaseDisplacement(3);
  const scalar_t rotationTime = std::abs(dyaw) / targetRotationVelocity_;
  const scalar_t displacement = std::sqrt(dx * dx + dy * dy);
  const scalar_t displacementTime = displacement / targetDisplacementVelocity_;
  return std::max(rotationTime, displacementTime);
}

}  // namespace ocs2::humanoid
