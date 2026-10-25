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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_mpc_config/arm_swing_config.nproto.h"

namespace ocs2::humanoid {

/** The generator of the arm-swing reference, selected by name
 * (armSwingGeneratorFromName()). */
enum class ArmSwingGenerator {
  /** The arms hold the target's posture. */
  kNone,
  /** The arms swing against the legs at the gait's cadence
     (SwitchedModelReferenceManager::getDesiredState()). */
  kProcedural,
};

/** The names of the generators, in the registry's order: "procedural", "none".
 */
std::vector<std::string> armSwingGeneratorNames();

/** The generator named `name`; InvalidArgument listing the valid names
 * otherwise. */
absl::StatusOr<ArmSwingGenerator> armSwingGeneratorFromName(
    absl::string_view name);

/**
 * The tuning of the procedural arm swing (the task file's arm_swing block).
 * Passive data; the defaults are the swing the reference manager has always
 * generated.
 *
 * At gait phase phi and commanded forward speed v (in the reference's heading),
 * the left arm swings forward by s = sin(2 pi (phi - phaseOffset)) v and the
 * right arm by -s: the left shoulder turns by shoulderGain s in the direction
 * that swings it forward, the left elbow by elbowGain s in the direction that
 * flexes it (ArmSwingDirections), and the right ones by the same with -s.
 */
struct ArmSwingSettings {
  /** The default gains [rad s/m] and phase offset [gait cycles] of the
   * arm_swing block. */
  static constexpr scalar_t kDefaultGain = 0.15;
  static constexpr scalar_t kDefaultPhaseOffset = 0.15;

  ArmSwingGenerator generator = ArmSwingGenerator::kProcedural;
  scalar_t shoulderGain = kDefaultGain;        // [rad s/m], not negative
  scalar_t elbowGain = kDefaultGain;           // [rad s/m], not negative
  scalar_t phaseOffset = kDefaultPhaseOffset;  // [gait cycles], finite
};

/** The arm_swing block as ArmSwingSettings: a known generator, finite gains
 * that are not negative, a finite phase. */
absl::StatusOr<ArmSwingSettings> armSwingSettingsFromConfig(
    const mpc_config::ArmSwingConfig& config);

/**
 * Which way each driven arm joint turns: +1 when a positive joint rate swings
 * the arm forward (a shoulder) or flexes the elbow (an elbow), -1 when a
 * negative one does. Passive data. The defaults are the convention of a
 * shoulder pitch and an elbow whose positive rotation is about the robot's +y
 * axis, which swings a hanging arm backward and extends an elbow; they are the
 * signs the reference manager used for every robot before they were derived.
 */
struct ArmSwingDirections {
  scalar_t leftShoulder = -1.0;
  scalar_t rightShoulder = -1.0;
  scalar_t leftElbow = -1.0;
  scalar_t rightElbow = -1.0;
};

/**
 * Derives the ArmSwingDirections of the joints model_settings.arm_joint_names
 * names from the robot model at `nominalState`, the nominal standing posture: a
 * shoulder's sign is that of the forward (base x) velocity of the center of
 * mass of everything it carries, per unit of its joint rate; an elbow's is that
 * of the rate at which its joint closes the angle between the upper arm (elbow
 * to shoulder joint) and the center of mass of everything the elbow carries.
 * FailedPrecondition when the model has no arm-swing joints, and
 * InvalidArgument when a named joint hardly swings its arm forward (under a
 * tenth of its lever) or its axis hardly lies across the arm's plane, because
 * that joint is not one an arm swing can be built on. `pinocchioInterface` is
 * used for scratch forward kinematics.
 */
absl::StatusOr<ArmSwingDirections> deriveArmSwingDirections(
    PinocchioInterface& pinocchioInterface,
    const MpcRobotModelBase<scalar_t>& mpcRobotModel,
    const vector_t& nominalState);

}  // namespace ocs2::humanoid
