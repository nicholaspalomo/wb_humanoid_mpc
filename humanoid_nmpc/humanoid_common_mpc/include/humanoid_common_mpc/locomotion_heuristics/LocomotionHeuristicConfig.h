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

#pragma once

#include <cmath>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"

namespace ocs2::humanoid {

/**
 * The key of the block these parameters are read from, directly under the root of the robot's task.yaml.
 *
 * It lives here rather than being spelled out at each call site because the loader, the parameter updater and the
 * tests all have to agree on it.
 */
inline constexpr const char* kLocomotionHeuristicsBlockKey = "locomotion_heuristics";

/**
 * H_Theta(pdot) = a1 pdot + a0, Table C.2, applied to roll and pitch independently.
 *
 * The velocities are the COMMANDED CoM velocity in the base's own yaw frame, so `a1` is a lean per unit of commanded
 * speed and `a0` is a standing trim that survives at zero speed. Bledt fitted the quadruped's own values by running
 * the optimizer offline and regressing the orientation it chose against the commanded velocity (section 4.2,
 * equations 4.11 and 4.12): roll -0.339 rad per m/s of lateral speed and pitch +0.0725 rad per m/s of forward speed.
 * Those numbers are NOT defaults here - a 9 kg quadruped's lean is not a humanoid's - and every coefficient ships at
 * zero, which makes a listed name with an untouched block exactly a no-op.
 *
 * Sign convention is the base pose's own: Euler ZYX about a z-up world, in which a POSITIVE pitch rotates the body's
 * +x axis towards -z, i.e. it puts the NOSE DOWN. Leaning into a forward command is therefore a POSITIVE
 * `pitchPerForwardVelocity`, which is also the sign of Bledt's own fitted value (+0.0725 rad per m/s, equation 4.12).
 */
struct OrientationCompensationParameters {
  scalar_t rollPerLateralVelocity = 0.0;   // a1 [rad s/m]
  scalar_t rollOffset = 0.0;               // a0 [rad]
  scalar_t pitchPerForwardVelocity = 0.0;  // a1 [rad s/m]
  scalar_t pitchOffset = 0.0;              // a0 [rad]
  /** [rad] the summed roll and pitch offsets are each clamped to +/- this, so a command spike cannot tip the reference over. */
  scalar_t maximumTilt = 0.35;
};

/**
 * H_Theta(Phi) = b1 sin(c1 Phi + d1), Table C.2: the orientation limit cycle a legged robot naturally falls into.
 *
 * This is the heuristic that most needs the per-node seam. Bledt found it by regressing the optimizer's chosen pitch
 * against the gait phase and recovering a sinusoid at exactly the step frequency (section 4.2, equation 4.13, figure
 * 4-6), which over a one-second horizon is two or three whole cycles - something the three-knot target trajectory
 * could not carry even in principle.
 *
 * `phaseRate` is in radians per unit of gait phase, and the phase runs 0 to 1 over one LEFT-RIGHT CYCLE, so 2*pi is
 * one period per cycle and 4*pi is one period per step. The defaults are those two rates with zero amplitude: the
 * shapes are pre-wired so that only the amplitude has to be found.
 *
 * Phase [0, 0.5) is LEFT STANCE (right foot in the air) and [0.5, 1) its mirror - the mode names LF and RF name the
 * foot in CONTACT. That is what `phaseOffset` is measured against.
 */
struct PeriodicOrientationParameters {
  scalar_t rollAmplitude = 0.0;          // b1 [rad]
  scalar_t rollPhaseRate = 2.0 * M_PI;   // c1 [rad per unit phase]; 2 pi = one period per gait cycle
  scalar_t rollPhaseOffset = 0.0;        // d1 [rad]
  scalar_t pitchAmplitude = 0.0;         // b1 [rad]
  scalar_t pitchPhaseRate = 4.0 * M_PI;  // c1 [rad per unit phase]; 4 pi = one period per step
  scalar_t pitchPhaseOffset = 0.0;       // d1 [rad]
};

/**
 * H_z(v) = a2 v^2 + a1 v + a0, Table C.2: the base height a robot wants at speed.
 *
 * Applied as an OFFSET to the reference height and never as an absolute one. An absolute height would overwrite what
 * SwitchedModelReferenceManager::adaptToCurrentGroundHeight() has just written into the same channel, which is how
 * this controller tracks the terrain, and the heuristic would silently become a terrain-height override.
 *
 * `v` is the magnitude of the commanded horizontal velocity, so the offset is symmetric in the direction of travel:
 * crouching to walk forwards and rising to walk backwards is not a thing any legged system does.
 */
struct HeightCompensationParameters {
  scalar_t heightPerSpeedSquared = 0.0;  // a2 [m s^2/m^2]
  scalar_t heightPerSpeed = 0.0;         // a1 [m s/m]; expect negative, i.e. lower with speed
  scalar_t heightOffset = 0.0;           // a0 [m]
  /** [m] the result is clamped to +/- this. A velocity spike must not be able to drop the base into its own knees. */
  scalar_t maximumHeightOffset = 0.05;
};

/**
 * H_r(Theta) = PTP(R(Theta) r_hip), Table C.1: the foot placed under its own hip, projected onto the ground.
 *
 * `r_hip` is not a key. It is the position of this leg's hip in the base frame, taken from the robot model once at
 * start-up, because a number that has to agree with the URDF should not be maintained by hand - the same reasoning
 * that keeps the robot's mass out of the contact-implicit block. The two scales below stretch it, so that the nominal
 * stance can be widened or narrowed without touching the model.
 *
 * THIS HEURISTIC MOVES THE ANCHOR. The rest of the family nudges the landing target that
 * SwitchedModelReferenceManager::nominalFoothold() already computes from the STANCE foot; this one re-anchors it on
 * the measured base. The header of that class argues against exactly that, and the argument is sound: in single
 * support the base sits roughly over the stance foot, so an offset taken from it gives less separation than intended.
 * It is here because Bledt never uses it alone - it is the base term the velocity-dependent ones are summed on top of
 * (figure 4-8), and it is the only member of the family that distinguishes left from right at zero velocity. The
 * layer warns when it is listed alone.
 */
struct HipCenteredSteppingParameters {
  scalar_t lateralScale = 1.0;       // multiplies the model's hip lateral offset; 0 removes the lateral part
  scalar_t longitudinalScale = 1.0;  // multiplies the model's hip fore-aft offset
};

/**
 * H_r(pdot, Phi) = gain sqrt(p_z / g) (pdot - pdot_d), Table C.1: the capture point of the VELOCITY ERROR.
 *
 * The one heuristic in the family that reacts to a disturbance rather than to a command: it is zero whenever the
 * robot is going as fast as it was asked to, and steps into the error when it is not. `gain` of 1 is the textbook
 * capture point of a linear inverted pendulum of height `p_z`.
 */
struct CapturePointParameters {
  scalar_t gain = 1.0;
  /** [m] 0: use the centre-of-mass height measured at each solve. Positive: use this fixed height instead. */
  scalar_t comHeightOverride = 0.0;
  scalar_t gravity = 9.81;  // [m/s^2]
  /** [m] the offset is clamped to this magnitude, so a bad velocity estimate cannot throw the landing target out of reach. */
  scalar_t maximumOffset = 0.25;
};

/**
 * H_r(pdot) = a1 pdot + a0, Table C.2: step in the direction you are travelling.
 *
 * "So as to maximize the utility of the contact feet over the stance period" (section 4.3, figure 4-9): a foot placed
 * under the hip at lift-off is behind the robot by touch-down, and the stance that follows is spent catching up.
 * Forward and lateral are in the base's yaw frame and are rotated into the world by the measured base yaw.
 *
 * The lateral offset `a0` is signed per foot, so one number widens both sides of the stance rather than shifting the
 * whole robot to the left.
 */
struct TranslationalSteppingParameters {
  scalar_t forwardPerForwardVelocity = 0.0;  // a1 [m s/m]
  scalar_t forwardOffset = 0.0;              // a0 [m]
  scalar_t lateralPerLateralVelocity = 0.0;  // a1 [m s/m]
  scalar_t lateralOffset = 0.0;              // a0 [m], applied to this foot's own side
};

/**
 * H_r(psidot) = a1 psidot + a0, Table C.2: step along the arc while turning on the spot.
 *
 * The rotational analogue of translational stepping and, like it, about keeping the foot useful over the stance: the
 * feet lead the hips into the turn instead of trailing them to the end of their workspace (figure 4-10). `psidot` is
 * the COMMANDED yaw rate, taken from the command rather than differentiated from the measured heading, because the
 * whole point is to step where the robot is being asked to go.
 *
 * The forward term is signed per foot, because a foot to the LEFT of a body turning counter-clockwise travels
 * BACKWARDS (its velocity is omega x r) while the right foot travels forwards: the two fore-aft displacements must be
 * equal and opposite. The implementation carries that sign, so a POSITIVE coefficient places each foot further along
 * the direction its own hip is travelling - it leads the hips into the turn, which is what the heuristic is for.
 */
struct InPlaceTurningParameters {
  scalar_t forwardPerYawRate = 0.0;  // a1 [m s/rad], positive leads the hips into the turn
  scalar_t forwardOffset = 0.0;      // a0 [m]
  scalar_t lateralPerYawRate = 0.0;  // a1 [m s/rad]
  scalar_t lateralOffset = 0.0;      // a0 [m], applied to this foot's own side
};

/**
 * H_r(pdot x omega) = a1 (pdot x omega) + a0, Table C.2: throw the feet out along the turn radius at speed.
 *
 * The heuristic the extraction framework found and the designer had not thought of, and the one that made fast
 * turning possible at all (section 4.3, figure 4-11): without it the robot "tended to fall outwards along the turning
 * radius as if it were an object slipping off a spinning plate". Bledt then recognised it after the fact as the foot
 * placement that lines up with the resultant of gravity and the centripetal acceleration (equation 4.31).
 *
 * With omega = psidot e_z and a horizontal velocity, the cross product reduces to the two planar scalars
 * (v_y psidot, -v_x psidot), which are what the two coefficients below multiply. That is why this is not the same
 * term as `in_place_turning`: it is zero at zero speed however fast the robot is spinning.
 */
struct HighSpeedTurningParameters {
  scalar_t forwardPerCrossTerm = 0.0;  // a1 [m s^2/(m rad)] on  v_y psidot
  scalar_t forwardOffset = 0.0;        // a0 [m]
  scalar_t lateralPerCrossTerm = 0.0;  // a1 [m s^2/(m rad)] on -v_x psidot
  scalar_t lateralOffset = 0.0;        // a0 [m], applied to this foot's own side
};

/**
 * H_f(Phi) = m g / (F beta), Table C.1: the vertical stance force scaled by the reciprocal of the duty factor.
 *
 * A foot that is on the ground for a fraction beta of the gait cycle has to push harder than static weight
 * compensation while it is down, or the vertical impulse over the cycle does not add up to the robot's weight. The
 * reference this layer shapes divides the weight over the feet that are in contact AT THIS INSTANT, which is the
 * beta = 1 case; the offset below is the difference between the two, so an empty list is exactly today's behaviour.
 *
 * Both clamps are load-bearing rather than defensive: beta goes to zero at the onset of a flight phase, and 1/beta is
 * unbounded there.
 */
struct ImpulseScalingParameters {
  scalar_t scale = 1.0;              // scales the analytic 1/beta correction; 0 disables it
  scalar_t minimumDutyFactor = 0.2;  // beta is clamped up to this before the reciprocal is taken
  scalar_t maximumForceRatio = 2.0;  // the scaled reference is clamped to this many times weight compensation
};

/**
 * H_f(Theta x pdot) = m omega x pdot, Table C.1: the horizontal force that holds the robot on a circular path.
 *
 * The only heuristic of the ten whose force is HORIZONTAL, which is what makes it the only one that has to be written
 * through MpcRobotModelBase::setContactForceInWorldFrame(): under the basis-vector input parameterization the input
 * lives in the local contact frame, and a horizontal world-frame force has to be rotated into it by the foot's
 * orientation. That rotation is a forward-kinematics pass per node, which is why WrenchHeuristic asks each heuristic
 * whether it needs one and only this one says yes.
 *
 * A CAVEAT WORTH READING BEFORE TURNING IT ON. Bledt's control model is a single rigid body whose force reference is
 * the only thing telling the QP how hard to push. The dynamics here are full centroidal and the state cost already
 * tracks the commanded linear and angular momentum, so m omega x v is the rate of change of a momentum the problem is
 * already regulating - an output the optimizer must produce rather than information the reference is missing. On this
 * formulation it is therefore a lower-fidelity second statement of physics the problem already has, and it can fight
 * the first where the two disagree, as they do in double support and wherever the friction cone binds. It is
 * implemented because it is one of the ten and because the seam for it is clean; `scale` is the knob for deciding in
 * simulation whether it earns its place.
 */
struct CentripetalAccelerationParameters {
  scalar_t scale = 1.0;  // scales m omega x v; 0 disables it
  /** [N] the horizontal force per foot is clamped to this magnitude; 0 uses `maximumForceRatioOfWeight` instead. */
  scalar_t maximumForce = 0.0;
  /** Used when `maximumForce` is 0: the clamp as a fraction of the robot's weight. A foot cannot pull sideways harder
   *  than friction allows, and the friction coefficient of the wrench cone is around 0.5 on these robots. */
  scalar_t maximumForceRatioOfWeight = 0.3;
};

/**
 * Everything the `locomotion_heuristics` block of a robot's task.yaml carries: the three name lists and one parameter
 * block per heuristic.
 *
 * One aggregate holding every term's struct, rather than a map of type-erased blocks, is the shape
 * ContactPlanningConfig already uses in this repository, and it is the shape the tuning GUI needs: the GUI walks the
 * YAML tree and turns every numeric leaf into a slider, so the parameters have to be plain nested scalars with the
 * heuristic's own registry name as their key.
 */
struct LocomotionHeuristicConfig {
  LocomotionHeuristicFormulation formulation;

  OrientationCompensationParameters orientationCompensation;
  PeriodicOrientationParameters periodicOrientation;
  HeightCompensationParameters heightCompensation;
  HipCenteredSteppingParameters hipCenteredStepping;
  CapturePointParameters capturePoint;
  TranslationalSteppingParameters translationalStepping;
  InPlaceTurningParameters inPlaceTurning;
  HighSpeedTurningParameters highSpeedTurning;
  ImpulseScalingParameters impulseScaling;
  CentripetalAccelerationParameters centripetalAcceleration;

  /** Rejects a configuration that cannot do what it says; the formulation's own checks plus the numeric ranges. */
  absl::Status validate() const;
};

/**
 * Reads the `locomotion_heuristics` block of `taskFile`.
 *
 * A file without the block yields the default configuration, whose three lists are empty and which is therefore an
 * exact no-op - so every robot that has not been given the block keeps behaving as it did. Every scalar key is
 * likewise optional and leaves its struct member's initializer in place when absent.
 *
 * Returns an error only for a configuration that is present and wrong: an unknown name, a name in the wrong list, a
 * duplicate, or a parameter outside its admissible range.
 */
absl::StatusOr<LocomotionHeuristicConfig> loadLocomotionHeuristicConfig(absl::string_view taskFile, bool verbose = false);

}  // namespace ocs2::humanoid
