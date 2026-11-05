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
#include <numbers>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/command/WalkingVelocityCommand.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Sinc.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"

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
 * `duration` [s] by
 * TargetTrajectoriesCalculatorBase::kSegmentMaxErrYawBaseRtWorld: unchanged at
 * any rate a robot turns at, and not a number for one that is not.
 */
scalar_t computeClampedSegmentRateYawBaseRtWorld(scalar_t rateYawBaseRtWorld,
                                                 scalar_t duration) {
  const scalar_t limit =
      TargetTrajectoriesCalculatorBase::kSegmentMaxErrYawBaseRtWorld / duration;
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
  nomPosBaseRtSurfaceEwrtWorldZ_.store(
      referenceSettings.nomPosBaseRtSurfaceEwrtWorldZ);
  refRateYawBaseRtWorld_.store(referenceSettings.refRateYawBaseRtWorld);
  refSpeedBaseRtWorld_.store(referenceSettings.refSpeedBaseRtWorld);
  cmdMaxLinvelBaseRtWorldEwrtHeadingX_.store(
      referenceSettings.cmdMaxLinvelBaseRtWorldEwrtHeadingX);
  cmdMaxLinvelBaseRtWorldEwrtHeadingY_.store(
      referenceSettings.cmdMaxLinvelBaseRtWorldEwrtHeadingY);
  cmdMaxErrPosBaseRtSurfaceEwrtWorldZ_.store(
      referenceSettings.cmdMaxErrPosBaseRtSurfaceEwrtWorldZ);
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

void TargetTrajectoriesCalculatorBase::setSourcePosSurfaceRtWorldEwrtWorldZ(
    std::function<scalar_t()> sourcePosSurfaceRtWorldEwrtWorldZ) {
  std::lock_guard<std::mutex> lock(surfaceSourceMutex_);
  sourcePosSurfaceRtWorldEwrtWorldZ_ =
      std::move(sourcePosSurfaceRtWorldEwrtWorldZ);
}

scalar_t TargetTrajectoriesCalculatorBase::getPosSurfaceRtWorldEwrtWorldZ()
    const {
  std::lock_guard<std::mutex> lock(surfaceSourceMutex_);
  return sourcePosSurfaceRtWorldEwrtWorldZ_
             ? sourcePosSurfaceRtWorldEwrtWorldZ_()
             : mpcRobotModelPtr_->modelSettings.posSurfaceRtWorldEwrtWorldZ;
}

scalar_t TargetTrajectoriesCalculatorBase::computeRefPosBaseRtWorldEwrtWorldZ(
    scalar_t cmdPosBaseRtSurfaceEwrtWorldZ) const {
  const scalar_t refPosBaseRtSurfaceEwrtWorldZ =
      cmdPosBaseRtSurfaceEwrtWorldZ > kCmdMinPosBaseRtSurfaceEwrtWorldZ
          ? cmdPosBaseRtSurfaceEwrtWorldZ
          : nomPosBaseRtSurfaceEwrtWorldZ_.load();
  return getPosSurfaceRtWorldEwrtWorldZ() + refPosBaseRtSurfaceEwrtWorldZ;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::getFinalPoseeulerzyxBaseRtWorld(
    const vector4_t& poseCommand,
    const vector6_t& poseeulerzyxHeadingRtWorld) const {
  vector_t finalPoseeulerzyxBaseRtWorld(kPoseTargetDim);

  // X facing forward to the robot, Y to the left side, in the heading frame
  const scalar_t cmdErrPosBaseRtWorldEwrtHeadingX = poseCommand(0);
  const scalar_t cmdErrPosBaseRtWorldEwrtHeadingY = poseCommand(1);
  const scalar_t yawBaseRtWorld =
      poseeulerzyxHeadingRtWorld(kPoseTargetYawIndex);

  const scalar_t cmdErrPosBaseRtWorldEwrtWorldX =
      std::cos(yawBaseRtWorld) * cmdErrPosBaseRtWorldEwrtHeadingX -
      std::sin(yawBaseRtWorld) * cmdErrPosBaseRtWorldEwrtHeadingY;
  const scalar_t cmdErrPosBaseRtWorldEwrtWorldY =
      std::sin(yawBaseRtWorld) * cmdErrPosBaseRtWorldEwrtHeadingX +
      std::cos(yawBaseRtWorld) * cmdErrPosBaseRtWorldEwrtHeadingY;

  // base p_x, p_y are relative to current state
  finalPoseeulerzyxBaseRtWorld(0) =
      poseeulerzyxHeadingRtWorld(0) + cmdErrPosBaseRtWorldEwrtWorldX;
  finalPoseeulerzyxBaseRtWorld(1) =
      poseeulerzyxHeadingRtWorld(1) + cmdErrPosBaseRtWorldEwrtWorldY;
  // base z relative to the default height above the ground
  const scalar_t cmdMaxErrPosBaseRtSurfaceEwrtWorldZ =
      cmdMaxErrPosBaseRtSurfaceEwrtWorldZ_;
  const scalar_t cmdErrPosBaseRtSurfaceEwrtWorldZ =
      std::clamp(poseCommand(kPoseCommandHeightIndex),
                 -cmdMaxErrPosBaseRtSurfaceEwrtWorldZ,
                 cmdMaxErrPosBaseRtSurfaceEwrtWorldZ);
  finalPoseeulerzyxBaseRtWorld(kPoseTargetZIndex) =
      getPosSurfaceRtWorldEwrtWorldZ() + nomPosBaseRtSurfaceEwrtWorldZ_ +
      cmdErrPosBaseRtSurfaceEwrtWorldZ;
  // theta_z relative to current
  finalPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex) =
      poseeulerzyxHeadingRtWorld(kPoseTargetYawIndex) +
      poseCommand(kPoseCommandYawIndex) * std::numbers::pi / kHalfTurnDegrees;
  finalPoseeulerzyxBaseRtWorld(kPoseTargetPitchIndex) = 0.0;
  finalPoseeulerzyxBaseRtWorld(kPoseTargetRollIndex) = 0.0;

  return finalPoseeulerzyxBaseRtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector6_t TargetTrajectoriesCalculatorBase::getPoseeulerzyxHeadingRtWorld(
    const vector_t& state) const {
  // Level: the target trajectories start from the current position and heading,
  // with the torso's roll and pitch zero.
  vector6_t poseeulerzyxHeadingRtWorld;
  poseeulerzyxHeadingRtWorld
      << mpcRobotModelPtr_->getPosBaseRtWorldEwrtWorld(state),
      mpcRobotModelPtr_->getYawBaseRtWorld(state), 0.0, 0.0;
  return poseeulerzyxHeadingRtWorld;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

vector4_t TargetTrajectoriesCalculatorBase::rotateVelocityCommandToWorld(
    const vector4_t& commandHeadingFrame, scalar_t yawBaseRtWorld) {
  const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy = rotatedAboutZ(
      yawBaseRtWorld,
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
  const vector2_t errPosBaseRtWorldEwrtWorldXy =
      computeErrPosBaseRtWorldEwrtWorldXy(initLinvelBaseRtWorldEwrtWorldXy,
                                          rateYawBaseRtWorld, deltaT);
  finalPoseeulerzyxBaseRtWorld[0] += errPosBaseRtWorldEwrtWorldXy(0);
  finalPoseeulerzyxBaseRtWorld[1] += errPosBaseRtWorldEwrtWorldXy(1);
  // A world height already, ground included
  // (computeRefPosBaseRtWorldEwrtWorldZ()). The pelvis-height fallback is not
  // applied to it a second time, as it used to be: on ground below z = 0.1 m a
  // correct world height would read as "no command".
  finalPoseeulerzyxBaseRtWorld[kPoseTargetZIndex] = refPosBaseRtWorldEwrtWorldZ;
  finalPoseeulerzyxBaseRtWorld[kPoseTargetYawIndex] +=
      rateYawBaseRtWorld * deltaT;
  finalPoseeulerzyxBaseRtWorld[kPoseTargetPitchIndex] = 0.0;
  finalPoseeulerzyxBaseRtWorld[kPoseTargetRollIndex] = 0.0;
  return finalPoseeulerzyxBaseRtWorld;
}

vector2_t TargetTrajectoriesCalculatorBase::computeErrPosBaseRtWorldEwrtWorldXy(
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
  const scalar_t errYawBaseRtWorld =
      finalPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex) -
      initPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex);
  size_t parts = 1;
  if (std::isfinite(errYawBaseRtWorld)) {
    clampedFinalPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex) =
        initPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex) +
        std::clamp(errYawBaseRtWorld, -kSegmentMaxErrYawBaseRtWorld,
                   kSegmentMaxErrYawBaseRtWorld);
    const scalar_t clampedErrYawBaseRtWorld =
        std::abs(clampedFinalPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex) -
                 initPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex));
    // At most kMaxKnotSubdivisions: the clamped turn is at most
    // kMaxKnotSubdivisions kKnotMaxErrYawBaseRtWorld.
    const scalar_t yawParts =
        std::ceil(clampedErrYawBaseRtWorld / kKnotMaxErrYawBaseRtWorld);
    if (yawParts > 1.0) {
      parts = std::min(static_cast<size_t>(yawParts), kMaxKnotSubdivisions);
    }
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
  const scalar_t errYawBaseRtWorld = std::abs(extent.errYawBaseRtWorld);
  const scalar_t arcLength = std::abs(extent.arcLength);
  const scalar_t cmdSpeedBaseRtWorld = std::abs(extent.cmdSpeedBaseRtWorld);
  if (!std::isfinite(errYawBaseRtWorld) || !std::isfinite(arcLength) ||
      !std::isfinite(cmdSpeedBaseRtWorld)) {
    return 1;
  }
  // Knots dpsi = errYawBaseRtWorld / n apart on the arc of radius rho =
  // arcLength / errYawBaseRtWorld: the chord between them is off the arc by at
  // most rho (1 - cos(dpsi / 2)) <= rho dpsi^2 / 8 = arcLength
  // errYawBaseRtWorld / (8 n^2), and the interpolation of their velocity
  // targets R_z(psi_k) v_c off R_z(psi(t)) v_c by cmdSpeedBaseRtWorld (1 -
  // cos(dpsi / 2)) <= cmdSpeedBaseRtWorld errYawBaseRtWorld^2 / (8 n^2), each
  // at the middle of the interval.
  const scalar_t parts = std::ceil(std::max(
      {errYawBaseRtWorld / kKnotMaxErrYawBaseRtWorld,
       std::sqrt(arcLength * errYawBaseRtWorld /
                 (kSagittaBoundDivisor * kKnotMaxErrPosBaseRtWorldEwrtWorld)),
       errYawBaseRtWorld *
           std::sqrt(cmdSpeedBaseRtWorld /
                     (kSagittaBoundDivisor *
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
  // (or nom_pos_base_rt_surface_ewrt_world_z) above the ground the reference
  // manager stands the robot on.
  vector6_t initPoseeulerzyxBaseRtWorld =
      getPoseeulerzyxHeadingRtWorld(initState);
  const scalar_t initYawBaseRtWorld =
      initPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex);
  const vector4_t commandGlobal =
      rotateVelocityCommandToWorld(velocityCommand, initYawBaseRtWorld);
  const vector2_t cmdLinvelBaseRtWorldEwrtWorldXy(
      commandGlobal(WalkingVelocityCommand::kLinearVelocityXIndex),
      commandGlobal(WalkingVelocityCommand::kLinearVelocityYIndex));
  const scalar_t cmdRateYawBaseRtWorld =
      commandGlobal(WalkingVelocityCommand::kAngularVelocityZIndex);
  const scalar_t refPosBaseRtWorldEwrtWorldZ =
      computeRefPosBaseRtWorldEwrtWorldZ(
          commandGlobal(WalkingVelocityCommand::kBaseHeightIndex));
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
  // Each segment turns by at most kSegmentMaxErrYawBaseRtWorld, so that its
  // knots are at most kKnotMaxErrYawBaseRtWorld apart for any measurement (the
  // class comment).
  first.rateYawBaseRtWorld = computeClampedSegmentRateYawBaseRtWorld(
      (measuredMotion.rateYawBaseRtWorld + cmdRateYawBaseRtWorld) /
          kAveragedMotionCount,
      first.duration);
  TwistSegment second;
  second.endOffset = mpcHorizon_;
  second.duration = mpcHorizon_ - intermediateOffset;
  second.rateYawBaseRtWorld = computeClampedSegmentRateYawBaseRtWorld(
      cmdRateYawBaseRtWorld, second.duration);

  std::vector<VelocityCommandKnot> knots;
  knots.push_back(VelocityCommandKnot{
      .time = initTime,
      .poseeulerzyxBaseRtWorld = initPoseeulerzyxBaseRtWorld,
      .cmdLinvelBaseRtWorldEwrtWorldXy = cmdLinvelBaseRtWorldEwrtWorldXy,
      .cmdRateYawBaseRtWorld = cmdRateYawBaseRtWorld,
      .errYawBaseRtWorld = 0.0,
      .secondSegmentFraction = 0.0});
  const std::array<bool, kTwistSegmentCount> isSecond = {false, true};
  for (const bool inSecond : isSecond) {
    TwistSegment& segment = inSecond ? second : first;
    const VelocityCommandKnot segmentStart = knots.back();
    // The command turns with the knots: it is given in the heading frame
    // (design README section 2.9).
    if (inSecond) {
      segment.initLinvelBaseRtWorldEwrtWorldXy = rotatedAboutZ(
          segmentStart.errYawBaseRtWorld, cmdLinvelBaseRtWorldEwrtWorldXy);
    }
    const size_t parts = knotSubdivisionCount(KnotSegmentExtent{
        .errYawBaseRtWorld = segment.rateYawBaseRtWorld * segment.duration,
        .arcLength =
            segment.initLinvelBaseRtWorldEwrtWorldXy.norm() * segment.duration,
        .cmdSpeedBaseRtWorld = cmdLinvelBaseRtWorldEwrtWorldXy.norm()});
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
      knot.errYawBaseRtWorld =
          knot.poseeulerzyxBaseRtWorld(kPoseTargetYawIndex) -
          initYawBaseRtWorld;
      knot.cmdLinvelBaseRtWorldEwrtWorldXy = rotatedAboutZ(
          knot.errYawBaseRtWorld, cmdLinvelBaseRtWorldEwrtWorldXy);
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
    const vector_t& errPoseeulerzyxBaseRtWorld) const {
  const scalar_t& errPosBaseRtWorldEwrtWorldX = errPoseeulerzyxBaseRtWorld(0);
  const scalar_t& errPosBaseRtWorldEwrtWorldY = errPoseeulerzyxBaseRtWorld(1);
  const scalar_t& errPoseeulerzyxBaseRtWorldYaw =
      errPoseeulerzyxBaseRtWorld(kPoseTargetYawIndex);
  const scalar_t turnTime =
      std::abs(errPoseeulerzyxBaseRtWorldYaw) / refRateYawBaseRtWorld_;
  const scalar_t normErrPosBaseRtWorldEwrtWorldXy =
      std::sqrt(errPosBaseRtWorldEwrtWorldX * errPosBaseRtWorldEwrtWorldX +
                errPosBaseRtWorldEwrtWorldY * errPosBaseRtWorldEwrtWorldY);
  const scalar_t displacementTime =
      normErrPosBaseRtWorldEwrtWorldXy / refSpeedBaseRtWorld_;
  return std::max(turnTime, displacementTime);
}

}  // namespace ocs2::humanoid
