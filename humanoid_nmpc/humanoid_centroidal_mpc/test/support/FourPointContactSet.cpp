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

#include "support/FourPointContactSet.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "ocs2_core/Types.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_mpc_config/input_weights.nproto.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"

namespace ocs2::humanoid::four_point_set {
namespace {

// The sole corners in the ankle link (max: the toe, min: the heel), from the
// Atlas sole of the shipped task file: the sole center 0.04 m forward of and
// 0.082 m below the ankle, 0.12 m from the center to the toe and to the heel.
constexpr scalar_t kMaxPosCornerRtAnkleEwrtAnkleX = 0.16;
constexpr scalar_t kMinPosCornerRtAnkleEwrtAnkleX = -0.08;
constexpr scalar_t kPosCornerRtAnkleEwrtAnkleZ = -0.082;

// The input weight of a point contact's force, per axis.
constexpr scalar_t kForceWeightX = 1.0e-5;
constexpr scalar_t kForceWeightY = 2.0e-5;
constexpr scalar_t kForceWeightZ = 5.0e-5;

/** A 3-DoF contact on the sole of the foot whose ankle is `ankle`. */
mpc_config::ModelSettingsConfig::PointContact soleCorner(
    absl::string_view name,
    absl::string_view ankle,
    scalar_t posContactRtAnkleEwrtAnkleX) {
  mpc_config::ModelSettingsConfig::PointContact contact;
  contact.name = std::string(name);
  contact.parent_joint_name = std::string(ankle);
  contact.pos_contact_rt_link_ewrt_link.x = posContactRtAnkleEwrtAnkleX;
  contact.pos_contact_rt_link_ewrt_link.z = kPosCornerRtAnkleEwrtAnkleZ;
  return contact;
}

/** The four sole corners, in contact order. */
std::vector<mpc_config::ModelSettingsConfig::PointContact> soleCorners() {
  return {
      soleCorner(kFrontLeft, "l_leg_akx", kMaxPosCornerRtAnkleEwrtAnkleX),
      soleCorner(kFrontRight, "r_leg_akx", kMaxPosCornerRtAnkleEwrtAnkleX),
      soleCorner(kHindLeft, "l_leg_akx", kMinPosCornerRtAnkleEwrtAnkleX),
      soleCorner(kHindRight, "r_leg_akx", kMinPosCornerRtAnkleEwrtAnkleX),
  };
}

/** The input weight of a 3-DoF contact: its force alone. */
mpc_config::InputWeights::ContactWrench pointWeight(absl::string_view name) {
  mpc_config::InputWeights::ContactWrench weight;
  weight.contact = std::string(name);
  mpc_config::Xyz forceWeights;
  forceWeights.x = kForceWeightX;
  forceWeights.y = kForceWeightY;
  forceWeights.z = kForceWeightZ;
  // NOLINTNEXTLINE(frame-naming): a per-component weight field
  weight.force = forceWeights;
  return weight;
}

/** A leg of the four-point set: one contact. */
mpc_config::ModelSettingsConfig::Leg pointLeg(absl::string_view name,
                                              absl::string_view contact) {
  mpc_config::ModelSettingsConfig::Leg leg;
  leg.name = std::string(name);
  leg.contacts = {std::string(contact)};
  return leg;
}

}  // namespace

CentroidalMpcConfig fourPointConfig(const CentroidalMpcConfig& atlas) {
  CentroidalMpcConfig config = atlas;
  mpc_config::TaskFile& task = config.task;
  task.model_settings.contacts_3dof = soleCorners();
  task.model_settings.contact_names_6dof = {};
  task.model_settings.contact_parent_joint_names = {};
  // NOLINTBEGIN(frame-naming): a per-component weight field
  task.input_weights.contact_wrenches = {
      pointWeight(kFrontLeft), pointWeight(kFrontRight), pointWeight(kHindLeft),
      pointWeight(kHindRight)};
  // NOLINTEND(frame-naming)
  task.contact_input_parameterization = "wrench";
  task.contact_schedule_source = "gait_schedule";
  // The Atlas's step width is the biped's step_width foothold anchor, which
  // this set does not run.
  task.nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y = 0.0;
  // The procedural arm swing follows the biped's gait phase, and is refused on
  // this set.
  task.arm_swing.generator = "none";
  task.costs = {"state_quadratic_cost", "input_quadratic_cost"};
  task.hard_constraints = {"zero_wrench", "zero_velocity"};
  task.soft_constraints = {"joint_limits", "contact_wrench_cone",
                           "friction_force_cone"};
  return config;
}

CentroidalMpcConfig leggedFourPointConfig(const CentroidalMpcConfig& atlas) {
  CentroidalMpcConfig config = fourPointConfig(atlas);
  config.task.model_settings.legs = {
      pointLeg("LF", kFrontLeft), pointLeg("RF", kFrontRight),
      pointLeg("LH", kHindLeft), pointLeg("RH", kHindRight)};
  return config;
}

}  // namespace ocs2::humanoid::four_point_set
