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

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/PhysicalConstants.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "legged_common_mpc/common/BipedFeet.h"
#include "legged_common_mpc/common/MpcRobotModelBase.h"
#include "legged_common_mpc/constraint/ContactWrenchConeConstraint.h"
#include "legged_common_mpc/contact_planning/ContactPlanningConfig.h"

namespace ocs2::legged {

/**
 * Ground and sole parameters the whole-body wrench cone is built from; the
 * planner's limits are derived from the same, and the friction coefficients
 * default to the wrench cone's.
 */
struct ContactPlanningGroundParameters {
  scalar_t frictionCoefficient =
      ContactWrenchConeConstraint::Config::kDefaultFrictionCoefficient;
  scalar_t torsionalFrictionCoefficient =
      ContactWrenchConeConstraint::Config::kDefaultTorsionalFrictionCoefficient;
  scalar_t maxPosCornersRtFootprintEwrtContactX =
      0.0;  // [m] half extent of the sole along x; 0: unknown
  scalar_t maxPosCornersRtFootprintEwrtContactY =
      0.0;  // [m] half extent of the sole along y; 0: unknown
};

/**
 * Planner parameters that are properties of the robot and of the ground rather
 * than tuning, derived once from the model and applied to every planner
 * configuration (the initial one and every hot reload from the task file), so
 * that they are never task-file keys and the planner never assumes more than
 * the whole-body constraints allow.
 */
struct ContactPlanningModelParameters {
  scalar_t totalMass = 0.0;  // [kg]
  // [m/s^2] the magnitude of the model's gravity, the one its dynamics use:
  // shared.gravity of every configuration.
  scalar_t gravity = kGravityMagnitude;
  // The model's contact count, ModelSettings::numContacts(): shared.numContacts
  // of every configuration.
  size_t numContacts = static_cast<size_t>(kBipedFootCount);
  scalar_t nomPosComRtFeetmeanEwrtWorldZ =
      0.0;  // [m] center of mass above the mean foot height at
            // the nominal state
  scalar_t maxPosCornersRtFootprintEwrtContactX =
      0.0;  // [m] sole footprint half extents (0: unknown)
  scalar_t maxPosCornersRtFootprintEwrtContactY = 0.0;
  scalar_t maxTorqueFootEwrtFootZ =
      0.0;  // [N m] torsional friction coefficient * weight
  // [N] friction coefficient * half the weight: the yaw couple of two stance
  // feet per meter of their separation. applyTo() multiplies it by the
  // configuration's own stepWidth.nomPosLfootRtRfootEwrtHeadingY, so that a
  // reloaded step width moves the couple as a fresh start with it would.
  scalar_t perStepWidthDoubleSupportMaxTorqueFeetmeanEwrtWorldZ = 0.0;
  BipedFeetArray<scalar_t> minYawFeetRtBase =
      makeBipedFeetArray(0.0);  // [rad] hip yaw joint limits per foot
  BipedFeetArray<scalar_t> maxYawFeetRtBase = makeBipedFeetArray(0.0);
  std::vector<std::string> hipYawJoints;  // per foot, empty where none was
                                          // found (fallback bounds then)

  /**
   * Writes the derived values into the term blocks of `config`: shared.gravity,
   * shared.numContacts, yaw_torque_budget and hip_yaw_range always;
   * shared.nomPosComRtFeetmeanEwrtWorldZ only where the file left it unset, and
   * the zmp_support_region box only where the file left it at 0 ("from the
   * model").
   */
  void applyTo(ContactPlanningConfig& config) const;
  std::string summary() const;
};

/** The foot yaw range the hip yaw joint of one leg allows, and the joint it was
 * read from. */
struct HipYawRange {
  scalar_t minYawFootRtBase =
      -ContactPlanningConfig::kNomMaxYawFootRtBase;  // [rad]
  scalar_t maxYawFootRtBase =
      ContactPlanningConfig::kNomMaxYawFootRtBase;  // [rad]
  std::string joint;  // empty when the leg has no hip yaw joint (the symmetric
                      // fallback above)
};

/**
 * The hip yaw range of the leg whose contact frame hangs off
 * `contactParentJointName`: walks the kinematic tree up from that joint to the
 * first revolute joint whose axis is vertical IN THE WORLD FRAME at the neutral
 * configuration, and reads its position limits as a range of foot yaw.
 *
 * The world frame matters for the sign as well as for the test. The foot yaw is
 * +q when the axis points up and -q when it points down, so an axis along -z -
 * which pinocchio's URDF parser turns into an unaligned revolute joint - or a
 * +z axis under a parent frame that is flipped upside down maps the joint
 * limits [lower, upper] to the foot yaw range [-upper, -lower]. Copying them
 * unmirrored planned yaws the leg cannot reach on one side and forbade the ones
 * it can on the other. A joint about a non-vertical axis is not a hip yaw,
 * whatever its local axis is.
 *
 * A joint that stores an angle (nq == 1) contributes its limits clamped to
 * [-pi, pi], and only when they contain 0; an unbounded (continuous) joint
 * stores (cos q, sin q) and turns all the way round, so its range is [-pi, pi].
 * Returns the symmetric fallback when no such joint exists or its limits do not
 * contain 0.
 */
HipYawRange deriveHipYawRange(const PinocchioInterface::Model& model,
                              absl::string_view contactParentJointName);

/**
 * Derives the planner's model parameters from the robot model and the ground
 * parameters of the wrench cone:
 *  - numContacts: `numContacts`, the model's contact count
 * (ModelSettings::numContacts()), which the planners refuse unless it is two;
 *  - gravity: the magnitude of the model's gravity (pinocchio::Model::gravity),
 * which the weights below are made with;
 *  - nomPosComRtFeetmeanEwrtWorldZ: the center of mass height above the mean
 * foot height at `nominalState`;
 *  - maxPosCornersRtFootprintEwrtContactX /
 * maxPosCornersRtFootprintEwrtContactY: the sole's footprint half extents (the
 * center of pressure region);
 *  - maxTorqueFootEwrtFootZ: torsional friction coefficient * total weight
 * (one stance foot carrying the robot);
 *  - perStepWidthDoubleSupportMaxTorqueFeetmeanEwrtWorldZ: friction coefficient
 * * half the weight (the couple of two stance feet per meter of their
 * separation; applyTo() makes the couple with the configuration's nominal step
 * width);
 *  - foot yaw bounds: the position limits of the hip yaw joint of every leg, as
 * a range of foot yaw (deriveHipYawRange); a symmetric fallback of
 * ContactPlanningConfig::kNomMaxYawFootRtBase where no such joint exists.
 * The heading model's yaw inertia is taken from the model at every plan and is
 * not part of this.
 */
ContactPlanningModelParameters deriveContactPlanningModelParameters(
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState,
    absl::Span<const std::string> contactParentJointNames,
    size_t numContacts,
    const ContactPlanningGroundParameters& ground);

}  // namespace ocs2::legged
