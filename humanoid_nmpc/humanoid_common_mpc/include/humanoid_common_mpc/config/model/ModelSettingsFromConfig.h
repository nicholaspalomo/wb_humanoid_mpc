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

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_mpc_config/contact_implicit_config.nproto.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/nominal_foothold_config.nproto.h"

/**
 * The model settings of the typed task file (humanoid_nmpc/humanoid_mpc_config): its model_settings block and the
 * formulation blocks that ModelSettings holds, converted into ModelSettings' own structs. The schema's defaults are the
 * structs' defaults, so a block the file leaves out converts to a default-constructed struct.
 *
 * ModelSettings::Create(const mpc_config::TaskFile&, ...), which builds the whole settings from these and the URDF, is
 * defined with them (ModelSettingsFromConfig.cpp).
 */
namespace ocs2::humanoid {

// The names of model_settings.foot_constraint.stance_constraint: what the stance foot's zero_velocity constraint holds.
// LINT.IfChange(stance_constraint_names)
/** 3 rows, the foot's position: a stance foot may tilt and pivot. */
inline constexpr absl::string_view kPositionStanceConstraint = "position";
/** 6 rows whose orientation part sees the foot's tilt only: a stance foot may pivot about the contact normal. */
inline constexpr absl::string_view kPositionAndTiltStanceConstraint = "position_and_tilt";
/** 6 rows with the yaw rate about the contact normal: a stance foot cannot pivot on the spot. */
inline constexpr absl::string_view kPositionAndOrientationStanceConstraint = "position_and_orientation";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/model_settings_config.proto:stance_constraint)

/** Every name stance_constraint accepts, from the fewest rows to the most. */
std::vector<std::string> stanceConstraintNames();

/**
 * The stance_constraint name of the rows that `config` selects: footConstraintFromConfig() undone for its two settings
 * constrainOrientation and constrainYawRateAboutContactNormal, of which constrainOrientation false is "position" whatever
 * the other one is (three rows have no yaw-rate row).
 */
absl::string_view stanceConstraintName(const ModelSettings::FootConstraintConfig& config);

/**
 * model_settings.foot_constraint: the gains of the foot constraints, the weights of their soft forms and the rows of the
 * stance constraint, whose name sets the two settings of ModelSettings::FootConstraintConfig that the zero_velocity
 * constraint reads (constrainOrientation, constrainYawRateAboutContactNormal). The two booleans of the file that the name
 * replaced are retired fields, which the parser refuses with the names.
 *
 * @return InvalidArgument naming model_settings.foot_constraint.stance_constraint and listing the names, for a name that
 *         is none of stanceConstraintNames(); InvalidArgument naming the field, for a soft_constraint_weight or
 *         normal_velocity_soft_constraint_weight that is not a finite positive number (the start-up and a hot reload
 *         refuse the same weights).
 */
absl::StatusOr<ModelSettings::FootConstraintConfig> footConstraintFromConfig(
    const mpc_config::ModelSettingsConfig::FootConstraintConfig& config);

/**
 * contact_implicit: the weights and references of the contact-implicit terms, as the file has them. Their range is
 * checked where the terms are built, by validateContactImplicitConfig().
 */
ModelSettings::ContactImplicitConfig contactImplicitFromConfig(const mpc_config::ContactImplicitConfig& config);

/** nominal_foothold: the step width of the nominal foothold reference. */
ModelSettings::NominalFootholdConfig nominalFootholdFromConfig(const mpc_config::NominalFootholdConfig& config);

}  // namespace ocs2::humanoid
