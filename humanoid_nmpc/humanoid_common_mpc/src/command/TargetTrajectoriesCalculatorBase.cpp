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

#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/common/Sinc.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {
namespace {

/** [deg] Half a turn, pi rad: a pose command's yaw is given in degrees. */
constexpr scalar_t kHalfTurnDegrees = 180.0;

/** The 8 of the sagitta bound rho (1 - cos(dpsi / 2)) <= rho dpsi^2 / 8 of a
 * chord dpsi apart on an arc of radius rho. */
constexpr scalar_t kSagittaBoundDivisor = 8.0;

/** The twists the first segment of a velocity command holds the mean of: the
 * measured and the commanded one. */
constexpr int kAveragedMotionCount = 2;

/** The constant-twist segments of a velocity command's reference: to the
 * intermediate knot, and on to the horizon. */
constexpr int kTwistSegmentCount = 2;

/** `vector` (x, y) turned by `angle` about z: R_z(angle) vector; exactly
 * `vector` at angle 0. */
vector2_t rotatedAboutZ(scalar_t angle, const vector2_t& vector) {
  const scalar_t cosine = std::cos(angle);
  const scalar_t sine = std::sin(angle);
  return vector2_t(cosine * vector(0) - sine * vector(1),
                   sine * vector(0) + cosine * vector(1));
}

/** A segment of a velocity command's reference: a constant planar twist held
 * for a time. Passive data. */
struct TwistSegment {
  /** [s] From the first knot to the segment's end. */
  scalar_t endOffset = 0.0;
  /** [s] */
  scalar_t duration = 0.0;
  /** [m/s] The world-frame velocity at the segment's start. */
  vector2_t initLinvelBaseRtWorldEwrtWorldXy = vector2_t::Zero();
  /** [rad/s] */
  scalar_t rateYawBaseRtWorld = 0.0;
};

/**
 * [rad/s] `rateYawBaseRtWorld` limited to the rate that turns a segment of
 * `duration` [s] by TargetTrajectoriesCalculatorBase::kMaxSegmentYawChange:
 * unchanged at any rate a robot turns at, and not a number for one that is not.
 */
scalar_t clampedSegmentYawRate(scalar_t rateYawBaseRtWorld, scalar_t duration) {
  const scalar_t limit =
      TargetTrajectoriesCalculatorBase::kMaxSegmentYawChange / duration;
  return std::clamp(rateYawBaseRtWorld, -limit, limit);
}

}  // namespace

TargetTrajectoriesCalculatorBase::TargetTrajectoriesCalculatorBase(
    const ReferenceSettings& referenceSettings,
    const vector_t& defaultJointState,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    scalar_t mpcHorizon)
    : mpcRobotModelPtr_(mpcRobotModel.clone()),
      targetJointState_(defaultJointState),
      mpcHorizon_(mpcHorizon) {
  ABSL_CHECK_EQ(static_cast<size_t>(targetJointState_.size()),
                mpcRobotModel.getJointDim())
      << "TargetTrajectoriesCalculatorBase: the default joint state has one "
         "entry per MPC joint (defaultJointStateFromConfig())";
  applyCommandLimits(referenceSettings);
}

void TargetTrajectoriesCalculatorBase::applyCommandLimits(
    const ReferenceSettings& referenceSettings) {
  // LINT.IfChange(apply_command_limits)
  nomPosBaseRtWorldEwrtWorldZ_.store(
      referenceSettings.nomPosBaseRtWorldEwrtWorldZ);
  refRateYawBaseRtWorld_.store(referenceSettings.refRateYawBaseRtWorld);
  refSpeedBaseRtWorld_.store(referenceSettings.refSpeedBaseRtWorld);
  cmdMaxLinvelBaseRtWorldEwrtHeadingX_.store(
      referenceSettings.cmdMaxLinvelBaseRtWorldEwrtHeadingX);
  cmdMaxLinvelBaseRtWorldEwrtHeadingY_.store(
      referenceSettings.cmdMaxLinvelBaseRtWorldEwrtHeadingY);
  cmdMaxErrPosBaseRtWorldEwrtWorldZ_.store(
      referenceSettings.cmdMaxErrPosBaseRtWorldEwrtWorldZ);
  cmdMaxAngvelBaseRtWorldEwrtWorldZ_.store(
      referenceSettings.cmdMaxAngvelBaseRtWorldEwrtWorldZ);
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/src/config/reference/ReferenceFromConfig.cpp:hot_reference_file_fields)
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TargetTrajectoriesCalculatorBase::reset() {}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void TargetTrajectoriesCalculatorBase::setTerrainHeightSource(
    std::function<scalar_t()> terrainHeightSource) {
  std::lock_guard<std::mutex> lock(terrainHeightSourceMutex_);
  terrainHeightSource_ = std::move(terrainHeightSource);
}

scalar_t TargetTrajectoriesCalculatorBase::getTerrainHeight() const {
  std::lock_guard<std::mutex> lock(terrainHeightSourceMutex_);
  return terrainHeightSource_ ? terrainHeightSource_()
                              : mpcRobotModelPtr_->modelSettings.terrainHeight;
}

scalar_t TargetTrajectoriesCalculatorBase::commandedBaseHeight(
    scalar_t commandedPelvisHeight) const {
  const scalar_t heightAboveGround =
      commandedPelvisHeight > kMinCommandedPelvisHeight
          ? commandedPelvisHeight
          : nomPosBaseRtWorldEwrtWorldZ_.load();
  return getTerrainHeight() + heightAboveGround;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::getDeltaBaseTarget(
    const vector4_t& poseCommand,
    const vector6_t& levelPoseeulerzyxBaseRtWorld) const {
  vector_t target(kPoseTargetDim);

  // X facing forward to the robot, Y to the left side in the baseFrame
  const scalar_t baseFrameDeltaX = poseCommand(0);
  const scalar_t baseFrameDeltaY = poseCommand(1);
  const scalar_t currentEulerZ =
      levelPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex);

  const scalar_t globalFrameDeltaX = std::cos(currentEulerZ) * baseFrameDeltaX -
                                     std::sin(currentEulerZ) * baseFrameDeltaY;
  const scalar_t globalFrameDeltaY = std::sin(currentEulerZ) * baseFrameDeltaX +
                                     std::cos(currentEulerZ) * baseFrameDeltaY;

  // base p_x, p_y are relative to current state
  target(0) = levelPoseeulerzyxBaseRtWorld(0) + globalFrameDeltaX;
  target(1) = levelPoseeulerzyxBaseRtWorld(1) + globalFrameDeltaY;
  // base z relative to the default height above the ground
  const scalar_t cmdMaxErrPosBaseRtWorldEwrtWorldZ =
      cmdMaxErrPosBaseRtWorldEwrtWorldZ_;
  const scalar_t deltaPelvisHeight = std::clamp(
      poseCommand(kPoseCommandHeightIndex), -cmdMaxErrPosBaseRtWorldEwrtWorldZ,
      cmdMaxErrPosBaseRtWorldEwrtWorldZ);
  target(kPoseTargetZIndex) =
      getTerrainHeight() + nomPosBaseRtWorldEwrtWorldZ_ + deltaPelvisHeight;
  // theta_z relative to current
  target(kPoseTargetYawIndex) =
      levelPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex) +
      poseCommand(kPoseCommandYawIndex) * M_PI / kHalfTurnDegrees;
  target(kPoseTargetPitchIndex) = 0.0;
  target(kPoseTargetRollIndex) = 0.0;

  return target;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::getLevelPoseeulerzyxBaseRtWorld(
    const vector_t& state) const {
  // Level: the target trajectories start from the current position and heading,
  // with the torso's roll and pitch zero.
  vector6_t levelPoseeulerzyxBaseRtWorld;
  levelPoseeulerzyxBaseRtWorld
      << mpcRobotModelPtr_->getPosBaseRtWorldEwrtWorld(state),
      mpcRobotModelPtr_->getYawBaseRtWorld(state), 0.0, 0.0;
  return levelPoseeulerzyxBaseRtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector4_t TargetTrajectoriesCalculatorBase::rotateVelocityCommandToWorld(
    const vector4_t& commandHeadingFrame, scalar_t heading) {
  const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy = rotatedAboutZ(
      heading,
      vector2_t(
          commandHeadingFrame(WalkingVelocityCommand::kLinearVelocityXIndex),
          commandHeadingFrame(WalkingVelocityCommand::kLinearVelocityYIndex)));
  vector4_t velocityCommandWorld = commandHeadingFrame;
  velocityCommandWorld(WalkingVelocityCommand::kLinearVelocityXIndex) =
      cmdLinvelBaseRtWorldEwrtWorldXy(0);
  velocityCommandWorld(WalkingVelocityCommand::kLinearVelocityYIndex) =
      cmdLinvelBaseRtWorldEwrtWorldXy(1);
  return velocityCommandWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::integrateRefPoseeulerzyxBaseRtWorld(
    const vector6_t& initPoseeulerzyxBaseRtWorld,
    const vector2_t& initLinvelBaseRtWorldEwrtWorldXy,
    scalar_t rateYawBaseRtWorld,
    scalar_t refPosBaseRtWorldEwrtWorldZ,
    scalar_t deltaT) const {
  vector6_t finalPoseeulerzyxBaseRtWorld = initPoseeulerzyxBaseRtWorld;
  const vector2_t displacement = planarArcDisplacement(
      initLinvelBaseRtWorldEwrtWorldXy, rateYawBaseRtWorld, deltaT);
  finalPoseeulerzyxBaseRtWorld[0] += displacement(0);
  finalPoseeulerzyxBaseRtWorld[1] += displacement(1);
  // A world height already, ground included (commandedBaseHeight()). The
  // pelvis-height fallback is not applied to it a second time, as it used to
  // be: on ground below z = 0.1 m a correct world height would read as "no
  // command".
  finalPoseeulerzyxBaseRtWorld[kPoseTargetZIndex] = refPosBaseRtWorldEwrtWorldZ;
  finalPoseeulerzyxBaseRtWorld[kPoseTargetYawIndex] +=
      rateYawBaseRtWorld * deltaT;
  finalPoseeulerzyxBaseRtWorld[kPoseTargetPitchIndex] = 0.0;
  finalPoseeulerzyxBaseRtWorld[kPoseTargetRollIndex] = 0.0;
  return finalPoseeulerzyxBaseRtWorld;
}

vector2_t TargetTrajectoriesCalculatorBase::planarArcDisplacement(
    const vector2_t& initLinvelBaseRtWorldEwrtWorldXy,
    scalar_t rateYawBaseRtWorld,
    scalar_t duration) {
  // int_0^T R_z(omega s) ds = T sinc(theta / 2) R_z(theta / 2), theta = omega
  // T: [[sin(theta), -(1 - cos(theta))], [1 - cos(theta), sin(theta)]] / omega,
  // with sin(theta) = 2 sin(theta / 2) cos(theta / 2) and 1 - cos(theta) = 2
  // sin(theta / 2)^2. At omega = 0 the rotation is the identity and the sinc 1,
  // so the displacement is T v_0 bit for bit.
  const scalar_t halfTurn = 0.5 * rateYawBaseRtWorld * duration;
  return (duration * sinc(halfTurn)) *
         rotatedAboutZ(halfTurn, initLinvelBaseRtWorldEwrtWorldXy);
}

std::vector<TargetTrajectoriesCalculatorBase::PoseCommandKnot>
TargetTrajectoriesCalculatorBase::subdivideYawKnots(
    scalar_t startTime,
    const vector6_t& initPoseeulerzyxBaseRtWorld,
    scalar_t targetTime,
    const vector6_t& finalPoseeulerzyxBaseRtWorld) {
  vector6_t clampedFinalPoseeulerzyxBaseRtWorld = finalPoseeulerzyxBaseRtWorld;
  const scalar_t yawChange = finalPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex) -
                             initPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex);
  size_t parts = 1;
  if (std::isfinite(yawChange)) {
    clampedFinalPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex) =
        initPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex) +
        std::clamp(yawChange, -kMaxSegmentYawChange, kMaxSegmentYawChange);
    const scalar_t clampedYawChange =
        std::abs(clampedFinalPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex) -
                 initPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex));
    // At most kMaxKnotSubdivisions: the clamped turn is at most
    // kMaxKnotSubdivisions kMaxKnotYawSpacing.
    const scalar_t yawParts = std::ceil(clampedYawChange / kMaxKnotYawSpacing);
    if (yawParts > 1.0)
      parts = std::min(static_cast<size_t>(yawParts), kMaxKnotSubdivisions);
  }
  std::vector<PoseCommandKnot> knots;
  knots.reserve(parts + 1);
  knots.push_back(
      PoseCommandKnot{.time = startTime,
                      .poseeulerzyxBaseRtWorld = initPoseeulerzyxBaseRtWorld});
  for (size_t part = 1; part < parts; ++part) {
    const scalar_t fraction =
        static_cast<scalar_t>(part) / static_cast<scalar_t>(parts);
    knots.push_back(
        PoseCommandKnot{.time = startTime + fraction * (targetTime - startTime),
                        .poseeulerzyxBaseRtWorld =
                            initPoseeulerzyxBaseRtWorld +
                            fraction * (clampedFinalPoseeulerzyxBaseRtWorld -
                                        initPoseeulerzyxBaseRtWorld)});
  }
  knots.push_back(PoseCommandKnot{
      .time = targetTime,
      .poseeulerzyxBaseRtWorld = clampedFinalPoseeulerzyxBaseRtWorld});
  return knots;
}

size_t TargetTrajectoriesCalculatorBase::knotSubdivisionCount(
    const KnotSegmentExtent& extent) {
  const scalar_t yawChange = std::abs(extent.yawChange);
  const scalar_t arcLength = std::abs(extent.arcLength);
  const scalar_t targetSpeed = std::abs(extent.targetSpeed);
  if (!std::isfinite(yawChange) || !std::isfinite(arcLength) ||
      !std::isfinite(targetSpeed))
    return 1;
  // Knots dpsi = yawChange / n apart on the arc of radius rho = arcLength /
  // yawChange: the chord between them is off the arc by at most rho (1 -
  // cos(dpsi / 2)) <= rho dpsi^2 / 8 = arcLength yawChange / (8 n^2), and the
  // interpolation of their velocity targets R_z(psi_k) v_c off R_z(psi(t)) v_c
  // by targetSpeed (1 - cos(dpsi / 2)) <= targetSpeed yawChange^2 / (8 n^2),
  // each at the middle of the interval.
  const scalar_t parts = std::ceil(std::max(
      {yawChange / kMaxKnotYawSpacing,
       std::sqrt(arcLength * yawChange /
                 (kSagittaBoundDivisor * kKnotMaxErrPosBaseRtWorldEwrtWorld)),
       yawChange *
           std::sqrt(targetSpeed / (kSagittaBoundDivisor *
                                    kKnotMaxErrLinvelBaseRtWorldEwrtWorld))}));
  if (parts <= 1.0) return 1;
  return parts >= static_cast<scalar_t>(kMaxKnotSubdivisions)
             ? kMaxKnotSubdivisions
             : static_cast<size_t>(parts);
}

std::vector<TargetTrajectoriesCalculatorBase::VelocityCommandKnot>
TargetTrajectoriesCalculatorBase::velocityCommandKnots(
    const vector4_t& velocityCommand,
    scalar_t initTime,
    const vector_t& initState,
    const BasePlanarMotion& measuredMotion) {
  // The level pose the reference starts from, at the commanded pelvis height
  // (or nom_pos_base_rt_world_ewrt_world_z) above the ground the reference
  // manager stands the robot on.
  vector6_t initPoseeulerzyxBaseRtWorld =
      getLevelPoseeulerzyxBaseRtWorld(initState);
  const scalar_t initYawBaseRtWorld =
      initPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex);
  const vector4_t commandGlobal =
      rotateVelocityCommandToWorld(velocityCommand, initYawBaseRtWorld);
  const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy(
      commandGlobal(WalkingVelocityCommand::kLinearVelocityXIndex),
      commandGlobal(WalkingVelocityCommand::kLinearVelocityYIndex));
  const scalar_t cmdRateYawBaseRtWorld =
      commandGlobal(WalkingVelocityCommand::kAngularVelocityZIndex);
  const scalar_t refPosBaseRtWorldEwrtWorldZ = commandedBaseHeight(
      commandGlobal(WalkingVelocityCommand::kDesiredPelvisHeightIndex));
  initPoseeulerzyxBaseRtWorld(kPoseTargetZIndex) = refPosBaseRtWorldEwrtWorldZ;
  // The measured motion, in the world frame at the start heading like the
  // command.
  const vector2_t measLinvelBaseRtWorldEwrtWorldXy = rotatedAboutZ(
      initYawBaseRtWorld, measuredMotion.linvelBaseRtWorldEwrtHeadingXy);

  // The first segment holds the mean of the measured and the commanded twist,
  // the second the command, from the intermediate knot's heading on.
  const scalar_t intermediateOffset =
      kIntermediateKnotHorizonFraction * mpcHorizon_;
  TwistSegment first;
  first.endOffset = intermediateOffset;
  first.duration = intermediateOffset;
  first.initLinvelBaseRtWorldEwrtWorldXy =
      vector2_t((measLinvelBaseRtWorldEwrtWorldXy(0) +
                 cmdLinvelBaseRtWorldEwrtWorldXy(0)) /
                    kAveragedMotionCount,
                (measLinvelBaseRtWorldEwrtWorldXy(1) +
                 cmdLinvelBaseRtWorldEwrtWorldXy(1)) /
                    kAveragedMotionCount);
  // Each segment turns by at most kMaxSegmentYawChange, so that its knots are
  // at most kMaxKnotYawSpacing apart for any measurement (the class comment).
  first.rateYawBaseRtWorld = clampedSegmentYawRate(
      (measuredMotion.rateYawBaseRtWorld + cmdRateYawBaseRtWorld) /
          kAveragedMotionCount,
      first.duration);
  TwistSegment second;
  second.endOffset = mpcHorizon_;
  second.duration = mpcHorizon_ - intermediateOffset;
  second.rateYawBaseRtWorld =
      clampedSegmentYawRate(cmdRateYawBaseRtWorld, second.duration);

  std::vector<VelocityCommandKnot> knots;
  knots.push_back(VelocityCommandKnot{
      .time = initTime,
      .poseeulerzyxBaseRtWorld = initPoseeulerzyxBaseRtWorld,
      .cmdLinvelBaseRtWorldEwrtWorldXy = cmdLinvelBaseRtWorldEwrtWorldXy,
      .cmdRateYawBaseRtWorld = cmdRateYawBaseRtWorld,
      .headingChange = 0.0,
      .secondSegmentFraction = 0.0});
  const std::array<bool, kTwistSegmentCount> isSecond = {false, true};
  for (const bool inSecond : isSecond) {
    TwistSegment& segment = inSecond ? second : first;
    const VelocityCommandKnot segmentStart = knots.back();
    // The command turns with the knots: it is given in the heading frame
    // (design README section 2.9).
    if (inSecond)
      segment.initLinvelBaseRtWorldEwrtWorldXy = rotatedAboutZ(
          segmentStart.headingChange, cmdLinvelBaseRtWorldEwrtWorldXy);
    const size_t parts = knotSubdivisionCount(KnotSegmentExtent{
        .yawChange = segment.rateYawBaseRtWorld * segment.duration,
        .arcLength =
            segment.initLinvelBaseRtWorldEwrtWorldXy.norm() * segment.duration,
        .targetSpeed = cmdLinvelBaseRtWorldEwrtWorldXy.norm()});
    for (size_t part = 1; part <= parts; ++part) {
      // Every knot of the segment from its start, on the same arc; the last one
      // at the segment's end, exactly.
      const bool last = part == parts;
      const scalar_t offset = last ? segment.duration
                                   : segment.duration *
                                         static_cast<scalar_t>(part) /
                                         static_cast<scalar_t>(parts);
      VelocityCommandKnot knot;
      knot.time =
          last ? initTime + segment.endOffset : segmentStart.time + offset;
      knot.poseeulerzyxBaseRtWorld = integrateRefPoseeulerzyxBaseRtWorld(
          segmentStart.poseeulerzyxBaseRtWorld,
          segment.initLinvelBaseRtWorldEwrtWorldXy, segment.rateYawBaseRtWorld,
          refPosBaseRtWorldEwrtWorldZ, offset);
      knot.headingChange = knot.poseeulerzyxBaseRtWorld(kPoseTargetYawIndex) -
                           initYawBaseRtWorld;
      knot.cmdLinvelBaseRtWorldEwrtWorldXy =
          rotatedAboutZ(knot.headingChange, cmdLinvelBaseRtWorldEwrtWorldXy);
      knot.cmdRateYawBaseRtWorld = cmdRateYawBaseRtWorld;
      knot.secondSegmentFraction =
          !inSecond ? 0.0
                    : (last ? 1.0
                            : static_cast<scalar_t>(part) /
                                  static_cast<scalar_t>(parts));
      knots.push_back(knot);
    }
  }
  return knots;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

scalar_t TargetTrajectoriesCalculatorBase::estimateTimeToTarget(
    const vector_t& desiredBaseDisplacement) const {
  const scalar_t& dx = desiredBaseDisplacement(0);
  const scalar_t& dy = desiredBaseDisplacement(1);
  const scalar_t& dyaw = desiredBaseDisplacement(kPoseTargetYawIndex);
  const scalar_t turnTime = std::abs(dyaw) / refRateYawBaseRtWorld_;
  const scalar_t displacement = std::sqrt(dx * dx + dy * dy);
  const scalar_t displacementTime = displacement / refSpeedBaseRtWorld_;
  return std::max(turnTime, displacementTime);
}

}  // namespace ocs2::humanoid
