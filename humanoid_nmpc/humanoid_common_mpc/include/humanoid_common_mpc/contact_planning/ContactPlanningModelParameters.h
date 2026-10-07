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

#include <string>
#include <vector>

#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"

namespace ocs2::humanoid {

/** Ground and sole parameters the whole-body wrench cone is built from; the planner's limits are derived from the same. */
struct ContactPlanningGroundParameters {
  scalar_t frictionCoefficient = 0.7;
  scalar_t torsionalFrictionCoefficient = 0.05;
  scalar_t footprintHalfLengthX = 0.0;  // [m] half extent of the sole along x; 0: unknown
  scalar_t footprintHalfWidthY = 0.0;   // [m] half extent of the sole along y; 0: unknown
};

/**
 * Planner parameters that are properties of the robot and of the ground rather than tuning, derived once from the model
 * and applied to every planner configuration (the initial one and every hot reload from the task file), so that they
 * are never task-file keys and the planner never assumes more than the whole-body constraints allow.
 */
struct ContactPlanningModelParameters {
  scalar_t totalMass = 0.0;      // [kg]
  scalar_t comHeight = 0.0;      // [m] center of mass above the mean foot height at the nominal state
  scalar_t zmpHalfWidthX = 0.0;  // [m] sole footprint half extents (0: unknown)
  scalar_t zmpHalfWidthY = 0.0;
  scalar_t torsionalFrictionTorque = 0.0;                          // [N m] torsional friction coefficient * weight
  scalar_t doubleSupportYawCouple = 0.0;                           // [N m] friction coefficient * half the weight * nominal step width
  feet_array_t<scalar_t> footYawOffsetLower = makeFeetArray(0.0);  // [rad] hip yaw joint limits per foot
  feet_array_t<scalar_t> footYawOffsetUpper = makeFeetArray(0.0);
  std::vector<std::string> hipYawJoints;  // per foot, empty where none was found (fallback bounds then)

  /**
   * Writes the derived values into the term blocks of `config`: yaw_torque_budget and hip_yaw_range always; shared.comHeight
   * only where the file left it unset, and the zmp_support_region box only where the file left it at 0 ("from the model").
   */
  void applyTo(ContactPlanningConfig& config) const;
  std::string summary() const;
};

/** The foot yaw range the hip yaw joint of one leg allows, and the joint it was read from. */
struct HipYawRange {
  scalar_t lower = -ContactPlanningConfig::kDefaultFootYawOffset;  // [rad]
  scalar_t upper = ContactPlanningConfig::kDefaultFootYawOffset;   // [rad]
  std::string joint;                                               // empty when the leg has no hip yaw joint (the symmetric fallback above)
};

/**
 * The hip yaw range of the leg whose contact frame hangs off `contactParentJointName`: walks the kinematic tree up
 * from that joint to the first revolute joint whose axis is vertical IN THE WORLD FRAME at the neutral configuration,
 * and reads its position limits as a range of foot yaw.
 *
 * The world frame matters for the sign as well as for the test. The foot yaw is +q when the axis points up and -q
 * when it points down, so an axis along -z - which pinocchio's URDF parser turns into an unaligned revolute joint -
 * or a +z axis under a parent frame that is flipped upside down maps the joint limits [lower, upper] to the foot yaw
 * range [-upper, -lower]. Copying them unmirrored planned yaws the leg cannot reach on one side and forbade the ones
 * it can on the other. A joint about a non-vertical axis is not a hip yaw, whatever its local axis is.
 *
 * A joint that stores an angle (nq == 1) contributes its limits clamped to [-pi, pi], and only when they contain 0;
 * an unbounded (continuous) joint stores (cos q, sin q) and turns all the way round, so its range is [-pi, pi].
 * Returns the symmetric fallback when no such joint exists or its limits do not contain 0.
 */
HipYawRange deriveHipYawRange(const PinocchioInterface::Model& model, const std::string& contactParentJointName);

/**
 * Derives the planner's model parameters from the robot model and the ground parameters of the wrench cone:
 *  - comHeight: the center of mass height above the mean foot height at `nominalState`;
 *  - zmpHalfWidthX / zmpHalfWidthY: the sole's footprint half extents (the center of pressure region);
 *  - torsionalFrictionTorque: torsional friction coefficient * total weight (one stance foot carrying the robot);
 *  - doubleSupportYawCouple: friction coefficient * half the weight * `nominalStepWidth` (the couple of two stance feet);
 *  - foot yaw bounds: the position limits of the hip yaw joint of every leg, as a range of foot yaw (deriveHipYawRange);
 *    a symmetric fallback of ContactPlanningConfig::kDefaultFootYawOffset where no such joint exists.
 * The heading model's yaw inertia is taken from the model at every plan and is not part of this.
 */
ContactPlanningModelParameters deriveContactPlanningModelParameters(PinocchioInterface& pinocchioInterface,
                                                                    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
                                                                    const vector_t& nominalState,
                                                                    const std::vector<std::string>& contactParentJointNames,
                                                                    const ContactPlanningGroundParameters& ground,
                                                                    scalar_t gravity,
                                                                    scalar_t nominalStepWidth);

}  // namespace ocs2::humanoid
