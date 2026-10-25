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

#pragma once

#include <atomic>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/SystemObservation.h"

#include "humanoid_common_mpc/command/BasePlanarMotion.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"

namespace ocs2::humanoid {

/**
 * Turns operator commands - a walking velocity or a target base pose - into the
 * TargetTrajectories the MPC tracks, within the command limits and reference
 * defaults of the reference file.
 *
 * The formulations derive from it and build the state and input trajectories.
 * The command limits are atomics, so that applyCommandLimits() may run on the
 * solver thread while targets are built on the command thread; the members
 * document which functions are thread-safe.
 *
 * The reference of a velocity command (velocityCommandKnots(), both
 * formulations). The command (v_x, v_y, omega) is a planar twist in the robot's
 * heading frame, used as given: any smoothing of it is
 * ProceduralMpcMotionManager's, on the solver time (its velocity command filter
 * and ramp), so that the reference does not depend on the solve rate. The base
 * knots follow a body whose planar twist in its own heading frame is, over the
 * first segment [t0, t0 + tau], tau = 0.7 T, the mean of the measured and the
 * commanded twist, ((v_m + v_c) / 2, (omega_m + omega_c) / 2), and over the
 * second, up to t0 + T, the command. The measured twist is the robot's motion
 * averaged over its last stride, [t0 - W, t0] with W the cycle of the gait the
 * schedule runs, so it lags the instantaneous motion by about W / 2
 * (ProceduralMpcMotionManager, StrideMotionHistory; design defects D1, D1b).
 *
 * The first segment states one assumption (design README section 2.9): the
 * robot's sway-free motion turns into the command over tau, its body twist
 * ramping linearly from (v_m, omega_m) to (v_c, omega_c). The mean twist held
 * over tau ends at the ramp's heading, psi0 + tau (omega_m + omega_c) / 2,
 * exactly, and on a straight line at its position, the integral p0 + tau
 * R_z(psi0) (v_m + v_c) / 2. On a turn the position differs from the ramp's by
 *   p_ramp - p_held = (tau^2 / 12) R_z(psi0) J (omega_m v_c - omega_c v_m) +
 * O(tau^3 omega^2 |v|),   J = R_z(pi / 2), (expand R_z(psi(s)) to first order
 * in the turn), which vanishes only where omega_m v_c = omega_c v_m: not on a
 * steady turn whose speed changes (4.4 mm at tau = 0.84 s and 0.5 rad/s from
 * 0.15 to 0.3 m/s; testVelocityCommandArc).
 *
 * The stride mean keeps the gait's sway out of the measured term: on the
 * Unitree G1's recorded straight walk the instantaneous base twist, even the
 * exact one, puts up to 0.15 m of lateral sway and 0.24 rad of pelvis yaw into
 * the intermediate knot, the stride mean 0.014 m and 0.013 rad. The knots and
 * the command are laid out from the heading the solve starts from, psi0 =
 * getBaseHeading(x0), which still sways with the gait: by up to 0.093 rad from
 * its stride mean on that walk (0.145 rad on the Unitree R1's), which turns the
 * intermediate knot sideways by tau |v| sin(0.093), 0.039 m at 0.5 m/s.
 *
 * A constant planar twist (v, omega) moves the base on a circular arc, p(t_s +
 * t) = p(t_s) + t sinc(omega t / 2) R_z(psi_s
 * + omega t / 2) v and psi(t_s + t) = psi_s + omega t
 * (planarArcDisplacement()), and the knots lie on it exactly (they lay on a
 * straight chord at the starting heading, 0.065 m off the arc at the end of
 * a 1.2 s horizon at 0.3 m/s and 0.3 rad/s). The velocity target of every knot
 * is the command in the knot's own heading frame: R_z(psi_k) (v_x, v_y) in the
 * world and the commanded yaw rate (it kept the first knot's heading, 0.36 rad
 * off at the end of that horizon). Each segment is cut into
 * knotSubdivisionCount() equal parts in time, the fewest for which
 *   - consecutive knots are at most kMaxKnotYawSpacing = pi / 2 apart in yaw:
 * the shortest-path slerp of a quaternion layout then turns the way the knots
 * do, which it does not above pi;
 *   - the linear interpolation the solver applies between knots dpsi apart
 * stays within kMaxKnotPositionDeviation of the arc and within
 * kMaxKnotVelocityDeviation of the arc's velocity target: the chord is at most
 * rho (1 - cos(dpsi / 2)) off the arc of radius rho = |v| / |omega| and the
 * interpolated target |v_c| (1 - cos(dpsi / 2)) off R_z(psi(t)) v_c, both at
 * the middle of the interval (with only the three knots, 0.21 m and 0.21 m/s at
 * the Unitree G1's command limits, 2.4 m/s at 1 rad/s over T = 1.2 s; 11 knots
 * bring that under 5 mm and 5 mm/s). A segment that only turns or only
 * translates is cut for the yaw spacing alone, and where |omega| tau <= pi / 2
 * (on the first segment |omega_m + omega_c| <= pi / tau, 3.74 rad/s at the
 * longest shipped horizon, 1.2 s) not at all. On the Euler layout a cut moves
 * no interpolated yaw: it is linear in time within a segment. A segment turns
 * by at most kMaxSegmentYawChange = kMaxKnotSubdivisions pi / 2: a faster yaw
 * rate is clamped to it, which takes a measured spin of |omega_m + omega_c| > 2
 * kMaxSegmentYawChange / tau (239 rad/s at T = 1.2 s), so that the yaw spacing
 * holds for every input. The deviation bounds hold while the cap of
 * kMaxKnotSubdivisions parts does not bind them, for an arc length times turn
 * and a |v_c| times turn squared below 8 kMaxKnotSubdivisions^2 5e-3 = 164 (m
 * rad, m/s rad^2), at most 1.9 at the G1's command limits.
 */
class TargetTrajectoriesCalculatorBase {
 public:
  /**
   * The calculator of the command limits and reference defaults
   * `referenceSettings` (referenceSettingsFromConfig() of the robot's reference
   * file, so already checked) and the nominal joint posture `defaultJointState`
   * (defaultJointStateFromConfig(), one entry per joint of `mpcRobotModel`).
   */
  TargetTrajectoriesCalculatorBase(
      const ReferenceSettings& referenceSettings,
      const vector_t& defaultJointState,
      const MpcRobotModelBase<scalar_t>& mpcRobotModel,
      scalar_t mpcHorizon);

  TargetTrajectoriesCalculatorBase(
      const TargetTrajectoriesCalculatorBase& rhs) = delete;
  TargetTrajectoriesCalculatorBase& operator=(
      const TargetTrajectoriesCalculatorBase&) = delete;

  virtual ~TargetTrajectoriesCalculatorBase() = default;

  /**
   * Forgets the filter state the command path carries from one target to the
   * next, so that the next target is built exactly as the first one of a
   * freshly constructed calculator. This base carries none; a derived
   * calculator with filters of its own overrides this and calls it. The limits
   * and the ground are configuration and are kept. Not thread-safe with respect
   * to building a target: call it from the thread that builds them (the solver
   * thread, through ProceduralMpcMotionManager::setResetHook()).
   */
  virtual void reset();

  /**
   * Applies the command limits and reference defaults of `referenceSettings`
   * (the scalars this class is constructed with, not the nominal joint posture,
   * which the joint targets path owns): on a hot reload of the reference file,
   * which replaces every one of them.
   *
   * Called by the parameter updater when the reference file changes on disk,
   * i.e. from the solver thread, while the command path reads these values from
   * the subscription thread. They are therefore atomics: each is an independent
   * scalar, so a reload that lands between two reads can only mix an old limit
   * with a new one, which is what a slider drag looks like anyway.
   */
  void applyCommandLimits(const ReferenceSettings& referenceSettings);

  /**
   * Where the ground under the commanded base height comes from.
   * `defaultBaseHeight` and a commanded pelvis height are heights ABOVE THE
   * GROUND (WalkingVelocityCommand::desired_pelvis_height says so), while the
   * target is a world pose, so every base height a command produces is written
   * at that height above getTerrainHeight() - see commandedBaseHeight(). A base
   * height taken from the measured state is already a world height and is left
   * alone.
   *
   * Without a source the ground is the task file's top-level `terrain_height`
   * (ModelSettings::terrainHeight of the model this calculator was built with),
   * which is all a command node running outside the MPC can know. A calculator
   * that runs inside the MPC - the one ProceduralMpcMotionManager calls every
   * solve - must be handed
   * SwitchedModelReferenceManager::getAppliedTerrainHeight: the ground the
   * reference manager applied in this very solve, so that a target built after
   * a hot reload of `terrain_height` stands on the new ground while the manager
   * moves the target already in use by the change, exactly once
   * (SwitchedModelReferenceManager::adaptToCurrentGroundHeight).
   * getTerrainHeight() would not do: a reload landing between the manager's run
   * and the target being built would then be applied twice. Thread-safe.
   */
  void setTerrainHeightSource(std::function<scalar_t()> terrainHeightSource);

  /** [m] the ground the commanded base heights stand on: the source's, or the
   * task file's without one. Thread-safe. */
  scalar_t getTerrainHeight() const;

  /**
   * [m] The world height of the base a command asks for: the commanded pelvis
   * height above the ground, or `defaultBaseHeight` above it when the command
   * carries none (a value at or below kMinCommandedPelvisHeight).
   */
  scalar_t commandedBaseHeight(scalar_t commandedPelvisHeight) const;

  /** [m] a commanded pelvis height at or below this is no command, and
   * `defaultBaseHeight` is used instead. */
  static constexpr scalar_t kMinCommandedPelvisHeight = 0.1;

  /**
   * The entries of a base pose target [x, y, z, yaw, pitch, roll]
   * (PoseCommandKnot::basePose, getCurrentBasePoseTarget()) past x and y, and
   * its size.
   */
  static constexpr int kPoseTargetZIndex = 2;
  static constexpr int kPoseTargetYawIndex = 3;
  static constexpr int kPoseTargetPitchIndex = 4;
  static constexpr int kPoseTargetRollIndex = 5;
  static constexpr int kPoseTargetDim = 6;

  /** The entries of a pose command [delta x, delta y, delta z, delta yaw] past
   * delta x and y (getDeltaBaseTarget()). */
  static constexpr int kPoseCommandHeightIndex = 2;
  static constexpr int kPoseCommandYawIndex = 3;

  /** The intermediate knot of a velocity command at t0 + tau, tau =
   * kIntermediateKnotHorizonFraction T (the class comment). */
  static constexpr scalar_t kIntermediateKnotHorizonFraction = 0.7;

  void setTargetDisplacementVelocity(scalar_t targetDisplacementVelocity) {
    targetDisplacementVelocity_ = targetDisplacementVelocity;
  }
  /** [rad/s] the yaw rate a pose command turns at; a reload of the reference
   * file resets it to its target_rotation_velocity. */
  void setTargetRotationVelocity(scalar_t targetRotationVelocity) {
    targetRotationVelocity_ = targetRotationVelocity;
  }

  /**
   * Converts command line to TargetTrajectories.
   * @param [in] commadLineTarget : [deltaX, deltaY, deltaZ, deltaYaw] defined
   * in pelvis frame
   * @param [in] observation : the current observation
   */
  virtual TargetTrajectories commandedPositionToTargetTrajectories(
      const vector4_t& commandedVelocities,
      scalar_t initTime,
      const vector_t& initState) = 0;

  /**
   * Converts desired velocities to TargetTrajectories (the class comment:
   * velocityCommandKnots()).
   * @param [in] commandedVelocities : [v_x, v_y, pelvis height, v_yaw]: the
   * twist in the pelvis (heading) frame
   * @param [in] initTime, initState : the start of the solve
   * @param [in] measuredMotion : the robot's planar motion the reference starts
   * from, the stride mean of
   *             ProceduralMpcMotionManager::getMeasuredPlanarMotion(); zero for
   * a robot at rest
   */
  virtual TargetTrajectories commandedVelocityToTargetTrajectories(
      const vector4_t& commandedVelocities,
      scalar_t initTime,
      const vector_t& initState,
      const BasePlanarMotion& measuredMotion) = 0;

  /**
   * [m] The planar displacement of a base that moves for `duration` [s] at a
   * constant planar twist, starting with the world-frame velocity
   * `startVelocityWorld` [m/s] and turning at `yawRate` [rad/s]: the arc
   *
   *   int_0^T R_z(omega s) v_0 ds = T sinc(theta / 2) R_z(theta / 2) v_0, theta
   * = omega T,
   *
   * (the chord of the arc: its length T |v_0| sinc(theta / 2), its direction
   * halfway through the turn). Exactly T v_0 for omega = 0.
   */
  static vector2_t planarArcDisplacement(const vector2_t& startVelocityWorld,
                                         scalar_t yawRate,
                                         scalar_t duration);

  /** What the subdivision of one constant-twist segment of a velocity command's
   * knots depends on. Passive data. */
  struct KnotSegmentExtent {
    /** [rad] The segment's turn, omega tau. */
    scalar_t yawChange = 0.0;
    /** [m] The length of the arc its base travels, |v| tau. */
    scalar_t arcLength = 0.0;
    /** [m/s] The magnitude |v_c| of the velocity target its knots carry, which
     * turns with them. */
    scalar_t targetSpeed = 0.0;
  };

  /**
   * The number n of equal parts in time a constant-twist segment of `extent` is
   * cut into (the class comment; design decision D10): the fewest for which the
   * knots are at most kMaxKnotYawSpacing apart in yaw, |yawChange| / n <= pi /
   * 2, and the bounds on the linear interpolation between them hold, arcLength
   * |yawChange| / (8 n^2) <= kMaxKnotPositionDeviation and targetSpeed
   * yawChange^2 / (8 n^2) <= kMaxKnotVelocityDeviation (from 1 - cos(x) <= x^2
   * / 2); at most kMaxKnotSubdivisions; 1 for an extent with an entry that is
   * not finite.
   */
  static size_t knotSubdivisionCount(const KnotSegmentExtent& extent);

  /** [rad] The largest yaw between consecutive knots of a velocity command. */
  static constexpr scalar_t kMaxKnotYawSpacing = 0.5 * M_PI;
  /** [m] The largest distance between the linear interpolation of a velocity
   * command's knots and the arc they lie on. */
  static constexpr scalar_t kMaxKnotPositionDeviation = 0.005;
  /** [m/s] The same between the interpolated velocity targets and the target in
   * the arc's heading frame, R_z(psi(t)) v_c. */
  static constexpr scalar_t kMaxKnotVelocityDeviation = 0.005;
  /** The cap of knotSubdivisionCount(). */
  static constexpr size_t kMaxKnotSubdivisions = 64;
  /**
   * [rad] The largest turn of a segment, kMaxKnotSubdivisions
   * kMaxKnotYawSpacing (100.5 rad): velocityCommandKnots() clamps the yaw rate
   * of a segment that would turn further, and subdivideYawKnots() the turn of a
   * pose command, so that kMaxKnotYawSpacing holds for every input.
   */
  static constexpr scalar_t kMaxSegmentYawChange =
      static_cast<scalar_t>(kMaxKnotSubdivisions) * kMaxKnotYawSpacing;

  /** A base knot of a pose command's reference (subdivideYawKnots()). Passive
   * data. */
  struct PoseCommandKnot {
    /** [s] */
    scalar_t time = 0.0;
    /** [x, y, z, yaw, pitch, roll], the yaw unbounded. */
    vector6_t basePose = vector6_t::Zero();
  };

  /**
   * The base knots of a pose command that moves the base from `startPose` at
   * `startTime` to `targetPose` at `targetTime` (both [x, y, z, yaw, pitch,
   * roll], the yaw unbounded, as getDeltaBaseTarget() adds the commanded turn
   * to it): the n + 1 knots of n = max(1, ceil(|dyaw| / kMaxKnotYawSpacing))
   * equal parts in time, interpolated linearly between the two poses in every
   * coordinate, the yaw included (design section 2.9). Consecutive knots are
   * then at most pi / 2 apart in yaw, so the shortest-path slerp between them
   * (MpcRobotModelBase::interpolateStateTrajectory()) turns the commanded way
   * for a turn of any size, not the short way round; and between level knots a
   * slerp about z is linear in the yaw, so the reference is the linear
   * interpolation of the two poses. A turn beyond kMaxSegmentYawChange is
   * clamped to it, which keeps the spacing for every input; n is 1 for a turn
   * that is not finite.
   */
  static std::vector<PoseCommandKnot> subdivideYawKnots(
      scalar_t startTime,
      const vector6_t& startPose,
      scalar_t targetTime,
      const vector6_t& targetPose);

 protected:
  /** A base knot of a velocity command's reference (velocityCommandKnots()).
   * Passive data. */
  struct VelocityCommandKnot {
    /** [s] */
    scalar_t time = 0.0;
    /** [x, y, z, yaw, pitch, roll]: on the arc, level, at the commanded base
     * height above the ground. */
    vector6_t basePose = vector6_t::Zero();
    /** [m/s] The commanded planar velocity in the world frame: the command's
     * (v_x, v_y) turned by the knot's yaw. */
    vector2_t commandedVelocityWorld = vector2_t::Zero();
    /** [rad/s] The commanded yaw rate. */
    scalar_t commandedYawRate = 0.0;
    /** [rad] The knot's yaw less the first knot's: by how much its heading
     * frame, and its target with it, has turned. */
    scalar_t headingChange = 0.0;
    /** Where the knot lies in the second segment, (t - t0 - 0.7 T) / (0.3 T): 0
     * up to the intermediate knot, 1 at the end. */
    scalar_t secondSegmentFraction = 0.0;
  };

  /**
   * The base knots of the velocity command `commandedVelocities` ([v_x, v_y,
   * pelvis height, yaw rate], pelvis frame) at `initTime` from `initState`, the
   * robot moving at `measuredMotion` (the class comment), in increasing time
   * from initTime to initTime + T: the knots of the first segment, then those
   * of the second.
   */
  std::vector<VelocityCommandKnot> velocityCommandKnots(
      const vector4_t& commandedVelocities,
      scalar_t initTime,
      const vector_t& initState,
      const BasePlanarMotion& measuredMotion);

  scalar_t estimateTimeToTarget(const vector_t& desiredBaseDisplacement) const;

  /**
   * The level base pose [x, y, z, yaw, pitch, roll] a target starts from at
   * `state`: the base position, the heading (MpcRobotModelBase::getBaseHeading)
   * as the yaw, and zero roll and pitch.
   */
  virtual vector6_t getCurrentBasePoseTarget(const vector_t& state) const;

  vector6_t getDeltaBaseTarget(const vector4_t& commadLinePoseTarget,
                               const vector6_t& currentPoseTarget) const;

  /**
   * Returns the velocity command `commandHeadingFrame` ([v_x, v_y, pelvis
   * height, yaw rate]) with its linear part rotated from the heading frame of
   * `heading` into the world frame, R_z(heading) (v_x, v_y); the pelvis height
   * and the yaw rate, which a rotation about z leaves unchanged, are copied.
   */
  static vector4_t rotateVelocityCommandToWorld(
      const vector4_t& commandHeadingFrame, scalar_t heading);

  /**
   * `currentPose` moved on for `deltaT` along the arc of a constant planar
   * twist that starts with the world velocity `startVelocityWorld` and turns at
   * `yawRate` (planarArcDisplacement()), at the world base height `baseHeight`
   * (commandedBaseHeight()), level: pitch and roll zero.
   */
  vector6_t integrateTargetBasePose(const vector6_t& currentPose,
                                    const vector2_t& startVelocityWorld,
                                    scalar_t yawRate,
                                    scalar_t baseHeight,
                                    scalar_t deltaT) const;

  /** The calculator's own clone of the model it was built with: owned, so
   * destroyed with the calculator. */
  const std::unique_ptr<const MpcRobotModelBase<scalar_t>> mpcRobotModelPtr_;

  // For pose control mode
  std::atomic<scalar_t> targetDisplacementVelocity_;
  std::atomic<scalar_t> targetRotationVelocity_;

  // For velocity control mode. The values before the constructor's
  // applyCommandLimits() replaces them.
  std::atomic<scalar_t> maxDisplacementVelocityX_{
      kInitialMaxDisplacementVelocityX};
  std::atomic<scalar_t> maxDisplacementVelocityY_{
      kInitialMaxDisplacementVelocityY};
  std::atomic<scalar_t> maxDeltaPelvisHeight_{kInitialMaxDeltaPelvisHeight};
  std::atomic<scalar_t> maxRotationVelocity_{kInitialMaxRotationVelocity};

  std::atomic<scalar_t> defaultBaseHeight_;
  // The ground under the commanded base height (setTerrainHeightSource).
  // Guarded, because the source is replaced from the thread that wires the
  // calculator while a command thread may be building a target.
  mutable std::mutex terrainHeightSourceMutex_;
  std::function<scalar_t()> terrainHeightSource_;
  vector_t targetJointState_;
  scalar_t mpcHorizon_;
};

}  // namespace ocs2::humanoid
