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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_common_mpc/config/model/ModelSettingsFromConfig.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/absl_check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/multibody/model.hpp"  // IWYU pragma: keep

#include "humanoid_common_mpc/common/BipedFeet.h"
#include "humanoid_common_mpc/common/ContactArray.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StateLayout.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigValueChecks.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "humanoid_mpc_config/contact_implicit_config.nproto.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/model_settings_config.nproto.pb.h"
#include "humanoid_mpc_config/model_settings_config.pb.h"
#include "humanoid_mpc_config/nominal_foothold_config.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

// The names of nominal_foothold.anchor (ModelSettings::NominalFootholdAnchor).
// LINT.IfChange(nominal_foothold_anchor_names)
constexpr char kStepWidthAnchorName[] = "step_width";
constexpr char kNominalStanceAnchorName[] = "nominal_stance";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/nominal_foothold_config.proto:nominal_foothold_anchor)

// ContactImplicitConfig holds nothing but its seven scalars, so a field added
// to it without a line in contactImplicitFromConfig() - which would leave it at
// its default whatever the file says - changes its size and fails to compile
// here.
constexpr int kContactImplicitScalarCount = 7;
static_assert(sizeof(ModelSettings::ContactImplicitConfig) ==
                  kContactImplicitScalarCount * sizeof(scalar_t),
              "every field of ModelSettings::ContactImplicitConfig needs its "
              "line in contactImplicitFromConfig()");

/** A stance_constraint name, and the two settings of the zero_velocity
 * constraint it selects. */
struct StanceConstraintEntry {
  // NOLINTNEXTLINE(totw-view-member): every entry is a string literal of the constexpr table below, alive for the whole program.
  absl::string_view name;
  bool constrainAttitude = true;
  bool constrainYawRateAboutContactNormal = false;
};

// The single place a stance constraint is added; stanceConstraintName() is its
// inverse.
constexpr std::array<StanceConstraintEntry, 3> kStanceConstraints = {{
    {.name = kPositionStanceConstraint,
     .constrainAttitude = false,
     .constrainYawRateAboutContactNormal = false},
    {.name = kPositionAndTiltStanceConstraint,
     .constrainAttitude = true,
     .constrainYawRateAboutContactNormal = false},
    {.name = kPositionAndOrientationStanceConstraint,
     .constrainAttitude = true,
     .constrainYawRateAboutContactNormal = true},
}};

// The error gains of model_settings.foot_constraint.
constexpr size_t kFootConstraintGains = 8;

/**
 * An error gain of model_settings.foot_constraint, by its field name (a string
 * literal). Neither field has a default, since a nonnull pointer has none to
 * give: every initializer names both.
 */
struct NamedGain {
  const char* absl_nonnull field;
  double value;
};

/** The index of each name of `names` in it. */
absl::flat_hash_map<std::string, size_t> indexMap(
    absl::Span<const std::string> names) {
  absl::flat_hash_map<std::string, size_t> indices;
  for (size_t i = 0; i < names.size(); ++i) {
    indices[names[i]] = i;
  }
  return indices;
}

/** The joints of the URDF the MPC keeps: every joint that
 * model_settings.fixed_joint_names does not name, in order. */
absl::StatusOr<std::vector<std::string>> activeJointNames(
    absl::Span<const std::string> fullJointNames,
    absl::Span<const std::string> fixedJointNames,
    bool verbose) {
  // Every fixed joint once, and each a joint of the URDF: a misspelled name
  // would otherwise leave the joint active without a word, and a name given
  // twice would count twice in the size check below.
  const absl::flat_hash_set<std::string> urdfJoints(fullJointNames.begin(),
                                                    fullJointNames.end());
  absl::flat_hash_set<std::string> seen;
  std::vector<std::string> unknown;
  std::vector<std::string> twice;
  for (const std::string& joint : fixedJointNames) {
    if (!urdfJoints.contains(joint)) {
      unknown.push_back(joint);
    } else if (!seen.insert(joint).second) {
      twice.push_back(joint);
    }
  }
  if (!unknown.empty() || !twice.empty()) {
    std::vector<std::string> problems;
    if (!unknown.empty()) {
      problems.push_back(absl::StrCat("names ", absl::StrJoin(unknown, ", "),
                                      ", which the URDF does not have"));
    }
    if (!twice.empty()) {
      problems.push_back(absl::StrCat("names ", absl::StrJoin(twice, ", "),
                                      " more than once"));
    }
    return absl::InvalidArgumentError(
        absl::StrCat("[ModelSettings] model_settings.fixed_joint_names ",
                     absl::StrJoin(problems, " and "),
                     "; it lists joints of the URDF, each once."));
  }
  // Compared before subtracting: more fixed joints than the URDF has would wrap
  // the unsigned difference around.
  if (fullJointNames.size() <= fixedJointNames.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ModelSettings] the URDF has ", fullJointNames.size(),
                     " joints and ", fixedJointNames.size(),
                     " are fixed by model_settings.fixed_joint_names; the MPC "
                     "needs at least one active joint."));
  }
  std::vector<std::string> active;
  active.reserve(fullJointNames.size() - fixedJointNames.size());
  for (const std::string& joint : fullJointNames) {
    if (std::find(fixedJointNames.begin(), fixedJointNames.end(), joint) ==
        fixedJointNames.end()) {
      active.push_back(joint);
    }
  }
  if (verbose) {
    LOG(INFO) << "[ModelSettings] the " << active.size()
              << " active MPC joints: " << absl::StrJoin(active, ", ");
  }
  return active;
}

/** Where each MPC joint is among the URDF's joints. */
std::vector<size_t> mpcToFullJointIndices(
    absl::Span<const std::string> fullJointNames,
    absl::Span<const std::string> mpcJointNames) {
  const absl::flat_hash_map<std::string, size_t> fullIndices =
      indexMap(fullJointNames);
  std::vector<size_t> indices;
  indices.reserve(mpcJointNames.size());
  for (const std::string& joint : mpcJointNames) {
    // The MPC joints are the URDF's joints less the fixed ones
    // (activeJointNames()), so every one of them is found.
    const absl::flat_hash_map<std::string, size_t>::const_iterator found =
        fullIndices.find(joint);
    ABSL_CHECK(found != fullIndices.end())
        << "MPC joint " << joint << " is not a joint of the URDF";
    indices.push_back(found->second);
  }
  return indices;
}

/** The index of the arm joint `name` (model_settings.arm_joint_names.`key`)
 * among the MPC joints, or InvalidArgument. */
absl::StatusOr<size_t> armJointIndex(
    const absl::flat_hash_map<std::string, size_t>& jointIndexMap,
    absl::string_view key,
    absl::string_view name) {
  const absl::flat_hash_map<std::string, size_t>::const_iterator found =
      jointIndexMap.find(name);
  if (found == jointIndexMap.end()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ModelSettings] model_settings.arm_joint_names.", key, " is '", name,
        "', which is not an active joint of the MPC model: a misspelled joint, "
        "or one that "
        "model_settings.fixed_joint_names fixes. Name all four arm joints, or "
        "none."));
  }
  return found->second;
}

/**
 * InvalidArgument naming model_settings.foot_constraint.`field` unless
 * `weight`, the weight of a soft foot term, is a finite positive number: a
 * weight of 0 switches the term off, a negative one rewards what it holds, and
 * a NaN poisons every solve.
 */
absl::Status checkSoftTermWeight(absl::string_view field, double weight) {
  // Negated so that NaN is refused too.
  if (!(weight > 0.0) || !std::isfinite(weight)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ModelSettings] model_settings.foot_constraint.", field,
                     " is ", weight,
                     ", but the weight of a soft foot term must be a finite "
                     "positive number."));
  }
  return absl::OkStatus();
}

// LINT.IfChange(contacts_3dof)
/**
 * The 3-DoF contacts of model_settings.contacts_3dof, each named, attached to a
 * named parent and at a finite translation, or InvalidArgument naming the
 * entry. `contactNames6DoF` are the 6-DoF contacts, which no 3-DoF contact may
 * share a name with. A name may be a frame of the URDF model at exactly the
 * configured placement, which the MPC then uses instead of adding one; that is
 * checked when the MPC builds its model (loadCustomPinocchioInterface()), not
 * here.
 */
absl::StatusOr<std::vector<ModelSettings::PointContact>>
pointContactsFromConfig(
    absl::Span<const mpc_config::ModelSettingsConfig::PointContact> config,
    absl::Span<const std::string> contactNames6DoF) {
  absl::flat_hash_set<std::string> names(contactNames6DoF.begin(),
                                         contactNames6DoF.end());
  std::vector<ModelSettings::PointContact> contacts;
  contacts.reserve(config.size());
  for (size_t i = 0; i < config.size(); ++i) {
    const mpc_config::ModelSettingsConfig::PointContact& contact = config[i];
    const std::string entry =
        absl::StrCat("model_settings.contacts_3dof[", i, "]");
    if (contact.name.empty() || contact.parent_joint_name.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[ModelSettings] ", entry,
          " needs a name and a parent_joint_name: the frame of a 3-DoF "
          "contact hangs from a joint of the robot model."));
    }
    if (!names.insert(contact.name).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[ModelSettings] ", entry, ".name is '", contact.name,
          "', which another contact has: every contact of the MPC is a frame "
          "of its own."));
    }
    RETURN_IF_ERROR(
        requireFinite(absl::StrCat(entry, ".pos_contact_rt_link_ewrt_link.x"),
                      contact.pos_contact_rt_link_ewrt_link.x));
    RETURN_IF_ERROR(
        requireFinite(absl::StrCat(entry, ".pos_contact_rt_link_ewrt_link.y"),
                      contact.pos_contact_rt_link_ewrt_link.y));
    RETURN_IF_ERROR(
        requireFinite(absl::StrCat(entry, ".pos_contact_rt_link_ewrt_link.z"),
                      contact.pos_contact_rt_link_ewrt_link.z));
    contacts.push_back(ModelSettings::PointContact{
        .name = contact.name,
        .parentJointName = contact.parent_joint_name,
        .posContactRtLinkEwrtLink =
            vector3_t(contact.pos_contact_rt_link_ewrt_link.x,
                      contact.pos_contact_rt_link_ewrt_link.y,
                      contact.pos_contact_rt_link_ewrt_link.z)});
  }
  return contacts;
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/model_settings_config.proto:contacts_3dof)

/** The joints of the URDF's floating-base model `model`, in its order: every
 * joint after the universe and the floating base. */
std::vector<std::string> urdfJointNames(const pinocchio::Model& model,
                                        bool verbose) {
  std::vector<std::string> names;
  for (pinocchio::JointIndex joint = kFirstJointAfterFloatingBase;
       joint < static_cast<pinocchio::JointIndex>(model.njoints); ++joint) {
    names.emplace_back(model.names[joint]);
  }
  if (verbose) {
    LOG(INFO) << "[ModelSettings] the URDF's joints: "
              << absl::StrJoin(names, ", ");
  }
  return names;
}

// The mode names every contact set has, which a leg name would shadow.
// LINT.IfChange(reserved_leg_names)
constexpr std::array<absl::string_view, 2> kReservedLegNames = {"FLY",
                                                                "STANCE"};
// The foot aliases of a set with the biped's stance, which parseModeNumber()
// reads before the leg names, so a leg of that name would be shadowed.
constexpr std::array<absl::string_view, 2> kBipedFootAliasNames = {"LF", "RF"};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/gait/MotionPhaseDefinition.h:mode_names)

// LINT.IfChange(legs)
// [m] A hip closer than this to the base's x-z plane has no side.
constexpr scalar_t kHipSideTolerance = 1.0e-3;
// [m] A hip closer than this to the mean hip along the base's x axis is
// neither a front nor a hind leg.
constexpr scalar_t kHipFrontHindTolerance = 1.0e-2;
// The names of the two legs of a biped whose task file lists none.
constexpr char kDefaultLeftLegName[] = "left";
constexpr char kDefaultRightLegName[] = "right";

/** A hip: the frame's name and its position relative to the base. */
struct Hip {
  std::string frameName;
  vector3_t posHipRtBaseEwrtBase = vector3_t::Zero();
};

/**
 * The URDF's floating-base model at its neutral configuration: the base at the
 * world origin with its axes, so that every placement is relative to the base
 * and in its axes. Keeps a reference to the model, which must outlive it. Its
 * const methods are safe to call concurrently.
 */
class NeutralModel {
 public:
  explicit NeutralModel(const pinocchio::Model& model)
      : model_(model), data_(model) {
    pinocchio::framesForwardKinematics(model_, data_,
                                       pinocchio::neutral(model_));
  }
  NeutralModel(const NeutralModel&) = delete;
  NeutralModel& operator=(const NeutralModel&) = delete;
  NeutralModel(NeutralModel&&) = delete;
  NeutralModel& operator=(NeutralModel&&) = delete;
  ~NeutralModel() = default;

  /** The hip found by walking up from the joint `parentJointName`
   * (hipJointOf()), or nullopt when it is not a joint of the model. */
  std::optional<Hip> derivedHip(absl::string_view parentJointName) const {
    // Pinocchio looks names up by const std::string&.
    const std::string parentJoint(parentJointName);
    if (parentJoint.empty() || !model_.existJointName(parentJoint)) {
      return std::nullopt;
    }
    const pinocchio::JointIndex hip =
        hipJointOf(model_, model_.getJointId(parentJoint));
    return Hip{.frameName = model_.names[hip],
               .posHipRtBaseEwrtBase = data_.oMi[hip].translation()};
  }

  /** The frame `frameName` as a hip, or nullopt when the model has no such
   * frame. */
  std::optional<Hip> explicitHip(absl::string_view frameName) const {
    std::string frame(frameName);
    if (!model_.existFrame(frame)) return std::nullopt;
    const pinocchio::FrameIndex frameIndex = model_.getFrameId(frame);
    return Hip{.frameName = std::move(frame),
               .posHipRtBaseEwrtBase = data_.oMf[frameIndex].translation()};
  }

 private:
  const pinocchio::Model& model_;
  pinocchio::Data data_;
};

/** InvalidArgument naming `entry` unless `name` is a valid leg name that
 * `taken` does not hold. */
absl::Status checkLegName(absl::string_view entry,
                          absl::string_view name,
                          const absl::flat_hash_set<std::string>& taken) {
  const bool alphanumeric =
      !name.empty() && std::all_of(name.begin(), name.end(), [](char c) {
        return absl::ascii_isalnum(static_cast<unsigned char>(c));
      });
  const bool allDigits = std::all_of(name.begin(), name.end(), [](char c) {
    return absl::ascii_isdigit(static_cast<unsigned char>(c));
  });
  if (!alphanumeric || allDigits) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ModelSettings] ", entry, ".name is '", name,
        "'; a leg name is letters and digits, not all digits (mode names join "
        "leg names with '_')."));
  }
  if (std::find(kReservedLegNames.begin(), kReservedLegNames.end(), name) !=
      kReservedLegNames.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ModelSettings] ", entry, ".name is '", name,
                     "', which is a mode name of every contact set (",
                     absl::StrJoin(kReservedLegNames, ", "), ")."));
  }
  if (taken.contains(name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ModelSettings] ", entry, ".name is '", name,
                     "', which another leg has: leg names are unique."));
  }
  return absl::OkStatus();
}

/** Returns InvalidArgument naming the leg and the alias when a leg of
 * `settings`, whose legs are set, has the name of a foot alias on a set with
 * the biped's stance (ModelSettings::hasBipedStance()); OK otherwise. */
absl::Status checkLegNamesAgainstFootAliases(const ModelSettings& settings) {
  if (!settings.hasBipedStance()) return absl::OkStatus();
  for (size_t i = 0; i < settings.legs.size(); ++i) {
    const std::string& name = settings.legs[i].name;
    if (std::find(kBipedFootAliasNames.begin(), kBipedFootAliasNames.end(),
                  name) != kBipedFootAliasNames.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[ModelSettings] model_settings.legs[", i, "].name is '", name,
          "', which is the foot alias '", name,
          "' of a contact set that stands on the biped's two feet: a gait "
          "file's '",
          name,
          "' would name that foot, not the leg. Name the legs other than ",
          absl::StrJoin(kBipedFootAliasNames, ", "), "."));
    }
  }
  return absl::OkStatus();
}

/** The side of a leg whose hip is `hip`: +1 left, -1 right, or
 * InvalidArgument naming `leg` and saying `remedy` for a hip on the base's x-z
 * plane. */
absl::StatusOr<scalar_t> lateralSignOf(absl::string_view leg,
                                       const Hip& hip,
                                       absl::string_view remedy) {
  const scalar_t y = hip.posHipRtBaseEwrtBase.y();
  if (y > kHipSideTolerance) return 1.0;
  if (y < -kHipSideTolerance) return -1.0;
  return absl::InvalidArgumentError(absl::StrCat(
      "[ModelSettings] ", leg, " has its hip ", hip.frameName, " at y = ", y,
      " m relative to the base, within ", kHipSideTolerance,
      " m of the base's x-z plane: a leg's side is the side of its hip; ",
      remedy, "."));
}

/** Sets each leg's front/hind position from its hip x against the mean x of
 * the legs whose hip is known. */
void classifyFrontAndHindLegs(std::vector<ModelSettings::Leg>& legs) {
  scalar_t sumX = 0.0;
  size_t numKnown = 0;
  for (const ModelSettings::Leg& leg : legs) {
    if (leg.hipFrameName.empty()) continue;
    sumX += leg.posHipRtBaseEwrtBase.x();
    ++numKnown;
  }
  if (numKnown == 0) return;
  const scalar_t meanX = sumX / static_cast<scalar_t>(numKnown);
  for (ModelSettings::Leg& leg : legs) {
    if (leg.hipFrameName.empty()) continue;
    const scalar_t dx = leg.posHipRtBaseEwrtBase.x() - meanX;
    leg.longitudinal =
        dx > kHipFrontHindTolerance
            ? ModelSettings::LegLongitudinalPosition::kFront
            : (dx < -kHipFrontHindTolerance
                   ? ModelSettings::LegLongitudinalPosition::kHind
                   : ModelSettings::LegLongitudinalPosition::kCenter);
  }
}

/**
 * The contact indices of the leg `entry`, model_settings.legs[legIndex] (named
 * `field`), in the contact order of `settings`. Records each contact's leg in
 * `legOfContact`; InvalidArgument for a leg without contacts, a name that is
 * not a contact of the set, or a contact another leg already has.
 */
absl::StatusOr<std::vector<size_t>> legContactIndicesFromConfig(
    absl::string_view field,
    size_t legIndex,
    const mpc_config::ModelSettingsConfig::Leg& entry,
    const ModelSettings& settings,
    absl::flat_hash_map<size_t, size_t>& legOfContact) {
  if (entry.contacts.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ModelSettings] ", field, " ('", entry.name,
                     "') names no contact; a leg ends in at least one."));
  }
  std::vector<size_t> contactIndices;
  contactIndices.reserve(entry.contacts.size());
  for (const std::string& contact : entry.contacts) {
    const std::vector<std::string>::const_iterator found = std::find(
        settings.contactNames.begin(), settings.contactNames.end(), contact);
    if (found == settings.contactNames.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[ModelSettings] ", field, " ('", entry.name, "') names contact '",
          contact, "', which is not a contact of the set (",
          absl::StrJoin(settings.contactNames, ", "), ")."));
    }
    const size_t index =
        static_cast<size_t>(found - settings.contactNames.begin());
    if (const std::pair<absl::flat_hash_map<size_t, size_t>::iterator, bool>
            inserted = legOfContact.try_emplace(index, legIndex);
        !inserted.second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[ModelSettings] ", field, " ('", entry.name, "') names contact '",
          contact, "', which model_settings.legs[", inserted.first->second,
          "] names too: a contact is in at most one leg."));
    }
    contactIndices.push_back(index);
  }
  return contactIndices;
}

/**
 * The hip of the leg `entry` (named `field`): its hip_frame_name, or the hip
 * derived from `firstParentJoint`, the parent joint of its first contact.
 * InvalidArgument when the named frame or the parent joint is not in `model`.
 */
absl::StatusOr<Hip> hipOfLegEntry(
    absl::string_view field,
    const mpc_config::ModelSettingsConfig::Leg& entry,
    absl::string_view firstParentJoint,
    const NeutralModel& model) {
  if (!entry.hip_frame_name.empty()) {
    std::optional<Hip> hip = model.explicitHip(entry.hip_frame_name);
    if (!hip.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "[ModelSettings] ", field, ".hip_frame_name is '",
          entry.hip_frame_name, "', which is not a frame of the URDF."));
    }
    return *std::move(hip);
  }
  std::optional<Hip> hip = model.derivedHip(firstParentJoint);
  if (!hip.has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ModelSettings] ", field, " ('", entry.name,
        "') has no hip_frame_name, and its first contact '",
        entry.contacts.front(), "' hangs from '", firstParentJoint,
        "', which is not a joint of the URDF to derive the hip from; name "
        "the hip_frame_name."));
  }
  return *std::move(hip);
}

/**
 * The legs of model_settings.legs, on the contacts and parent joints of
 * `settings` and the hips of `model`, or InvalidArgument naming
 * model_settings.legs[i] and what is wrong with it.
 */
absl::StatusOr<std::vector<ModelSettings::Leg>> legsFromConfig(
    absl::Span<const mpc_config::ModelSettingsConfig::Leg> config,
    const ModelSettings& settings,
    const NeutralModel& model) {
  const std::vector<std::string> parents =
      settings.contactParentJointNamesInContactOrder();
  absl::flat_hash_set<std::string> names;
  absl::flat_hash_map<size_t, size_t> legOfContact;
  std::vector<ModelSettings::Leg> legs;
  legs.reserve(config.size());
  for (size_t i = 0; i < config.size(); ++i) {
    const mpc_config::ModelSettingsConfig::Leg& entry = config[i];
    const std::string field = absl::StrCat("model_settings.legs[", i, "]");
    RETURN_IF_ERROR(checkLegName(field, entry.name, names));
    names.insert(entry.name);
    ModelSettings::Leg leg{.name = entry.name};
    ASSIGN_OR_RETURN(
        leg.contactIndices,
        legContactIndicesFromConfig(field, i, entry, settings, legOfContact));
    ASSIGN_OR_RETURN(const Hip hip,
                     hipOfLegEntry(field, entry,
                                   parents[leg.contactIndices.front()], model));
    ASSIGN_OR_RETURN(
        leg.lateralSign,
        lateralSignOf(absl::StrCat(field, " ('", entry.name, "')"), hip,
                      /*remedy=*/"name another hip_frame_name"));
    leg.hipFrameName = hip.frameName;
    leg.posHipRtBaseEwrtBase = hip.posHipRtBaseEwrtBase;
    legs.push_back(std::move(leg));
  }
  classifyFrontAndHindLegs(legs);
  return legs;
}

/**
 * The two legs of a biped contact set whose task file lists none: "left" on
 * contact 0 and "right" on contact 1, with the hips derived from their parent
 * joints. A foot without a parent joint of the URDF keeps the side of its
 * index and no hip; InvalidArgument when a derived hip is on the other side
 * or on the base's x-z plane.
 */
absl::StatusOr<std::vector<ModelSettings::Leg>> defaultBipedLegs(
    const ModelSettings& settings, const NeutralModel& model) {
  const std::vector<std::string> parents =
      settings.contactParentJointNamesInContactOrder();
  struct DefaultLeg {
    const char* absl_nonnull name;
    size_t contact;
    scalar_t lateralSign;
  };
  const std::array<DefaultLeg, kBipedFootCount> defaults = {{
      {.name = kDefaultLeftLegName,
       .contact = kContactLeftIndex,
       .lateralSign = 1.0},
      {.name = kDefaultRightLegName,
       .contact = kContactRightIndex,
       .lateralSign = -1.0},
  }};
  std::vector<ModelSettings::Leg> legs;
  legs.reserve(defaults.size());
  for (const DefaultLeg& entry : defaults) {
    ModelSettings::Leg leg{.name = entry.name,
                           .contactIndices = {entry.contact},
                           .lateralSign = entry.lateralSign};
    if (const std::optional<Hip> hip = model.derivedHip(parents[entry.contact]);
        hip.has_value()) {
      const std::string description =
          absl::StrCat("the default biped leg '", entry.name, "' on contact '",
                       settings.contactNames[entry.contact], "'");
      ASSIGN_OR_RETURN(const scalar_t lateralSign,
                       lateralSignOf(description, *hip,
                                     /*remedy=*/
                                     "name the legs in "
                                     "model_settings.legs"));
      if (lateralSign != entry.lateralSign) {
        return absl::InvalidArgumentError(absl::StrCat(
            "[ModelSettings] ", description, " has its hip ", hip->frameName,
            " at y = ", hip->posHipRtBaseEwrtBase.y(),
            " m relative to the base, on the other side: contact_names_6dof "
            "lists the left foot first, or model_settings.legs names the "
            "legs."));
      }
      leg.hipFrameName = hip->frameName;
      leg.posHipRtBaseEwrtBase = hip->posHipRtBaseEwrtBase;
    }
    legs.push_back(std::move(leg));
  }
  classifyFrontAndHindLegs(legs);
  return legs;
}

/** The legs of `settings`: model_settings.legs, the default biped legs, or
 * none. */
absl::StatusOr<std::vector<ModelSettings::Leg>> legsOf(
    absl::Span<const mpc_config::ModelSettingsConfig::Leg> config,
    const ModelSettings& settings,
    const pinocchio::Model& urdfModel) {
  const NeutralModel model(urdfModel);
  if (!config.empty()) return legsFromConfig(config, settings, model);
  if (settings.isBipedContactSet()) return defaultBipedLegs(settings, model);
  return std::vector<ModelSettings::Leg>();
}
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/model_settings_config.proto:legs)

}  // namespace

std::vector<std::string> stanceConstraintNames() {
  std::vector<std::string> names;
  names.reserve(kStanceConstraints.size());
  for (const StanceConstraintEntry& entry : kStanceConstraints) {
    names.emplace_back(entry.name);
  }
  return names;
}

absl::string_view stanceConstraintName(
    const ModelSettings::FootConstraintConfig& config) {
  // Three rows have no yaw-rate row, so the yaw-rate setting counts only beside
  // the orientation rows.
  const bool yawRate =
      config.constrainAttitude && config.constrainYawRateAboutContactNormal;
  for (const StanceConstraintEntry& entry : kStanceConstraints) {
    if (entry.constrainAttitude == config.constrainAttitude &&
        entry.constrainYawRateAboutContactNormal == yawRate) {
      return entry.name;
    }
  }
  // Unreachable while kStanceConstraints names all three rows; the registry
  // test walks them.
  return kPositionAndTiltStanceConstraint;
}

absl::StatusOr<ModelSettings::FootConstraintConfig> footConstraintFromConfig(
    const mpc_config::ModelSettingsConfig::FootConstraintConfig& config) {
  RETURN_IF_ERROR(checkSoftTermWeight("soft_constraint_weight",
                                      config.soft_constraint_weight));
  RETURN_IF_ERROR(
      checkSoftTermWeight("normal_velocity_soft_constraint_weight",
                          config.normal_velocity_soft_constraint_weight));
  // The error gains of the foot constraints, e.g. v_z = zdot_ref +
  // position_error_gain_z * (z_ref - z): a negative gain makes the error grow
  // instead of decay, and NaN poisons every solve.
  const std::array<NamedGain, kFootConstraintGains> gains = {{
      {.field = "position_error_gain_z", .value = config.position_error_gain_z},
      {.field = "orientation_error_gain",
       .value = config.orientation_error_gain},
      {.field = "linear_velocity_error_gain_z",
       .value = config.linear_velocity_error_gain_z},
      {.field = "linear_velocity_error_gain_xy",
       .value = config.linear_velocity_error_gain_xy},
      {.field = "angular_velocity_error_gain",
       .value = config.angular_velocity_error_gain},
      {.field = "linear_acceleration_error_gain_z",
       .value = config.linear_acceleration_error_gain_z},
      {.field = "linear_acceleration_error_gain_xy",
       .value = config.linear_acceleration_error_gain_xy},
      {.field = "angular_acceleration_error_gain",
       .value = config.angular_acceleration_error_gain},
  }};
  for (const NamedGain& gain : gains) {
    RETURN_IF_ERROR(requireNonNegative(
        absl::StrCat("model_settings.foot_constraint.", gain.field),
        gain.value));
  }
  ModelSettings::FootConstraintConfig converted = {
      .positionErrorGain_z = config.position_error_gain_z,
      .orientationErrorGain = config.orientation_error_gain,
      .linearVelocityErrorGain_z = config.linear_velocity_error_gain_z,
      .linearVelocityErrorGain_xy = config.linear_velocity_error_gain_xy,
      .angularVelocityErrorGain = config.angular_velocity_error_gain,
      .linearAccelerationErrorGain_z = config.linear_acceleration_error_gain_z,
      .linearAccelerationErrorGain_xy =
          config.linear_acceleration_error_gain_xy,
      .angularAccelerationErrorGain = config.angular_acceleration_error_gain,
      .softConstraintWeight = config.soft_constraint_weight,
      .normalVelocitySoftConstraintWeight =
          config.normal_velocity_soft_constraint_weight,
  };
  for (const StanceConstraintEntry& entry : kStanceConstraints) {
    if (entry.name == config.stance_constraint) {
      converted.constrainAttitude = entry.constrainAttitude;
      converted.constrainYawRateAboutContactNormal =
          entry.constrainYawRateAboutContactNormal;
      return converted;
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "[ModelSettings] model_settings.foot_constraint.stance_constraint is '",
      config.stance_constraint,
      "', which is not a stance constraint; valid names are: ",
      absl::StrJoin(stanceConstraintNames(), ", "), "."));
}

ModelSettings::ContactImplicitConfig contactImplicitFromConfig(
    const mpc_config::ContactImplicitConfig& config) {
  return {
      .complementarityWeight = config.complementarity_weight,
      .slipWeight = config.slip_weight,
      .scalePosContactRtSurfaceEwrtWorldZ =
          config.scale_pos_contact_rt_surface_ewrt_world_z,
      .scaleLinvelContactRtWorldEwrtWorldXy =
          config.scale_linvel_contact_rt_world_ewrt_world_xy,
      .scaleAngvelContactRtWorldEwrtWorldZ =
          config.scale_angvel_contact_rt_world_ewrt_world_z,
      .penetrationWeight = config.penetration_weight,
      .gapSmoothing = config.gap_smoothing,
  };
}

std::vector<std::string> nominalFootholdAnchorNames() {
  return {kStepWidthAnchorName, kNominalStanceAnchorName};
}

absl::StatusOr<ModelSettings::NominalFootholdAnchor>
nominalFootholdAnchorFromName(absl::string_view name) {
  if (name == kStepWidthAnchorName) {
    return ModelSettings::NominalFootholdAnchor::kStepWidth;
  }
  if (name == kNominalStanceAnchorName) {
    return ModelSettings::NominalFootholdAnchor::kNominalStance;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("nominal_foothold.anchor is '", name, "'; the anchors are ",
                   absl::StrJoin(nominalFootholdAnchorNames(), ", "), "."));
}

std::string nominalFootholdAnchorName(
    ModelSettings::NominalFootholdAnchor anchor) {
  switch (anchor) {
    case ModelSettings::NominalFootholdAnchor::kStepWidth:
      return kStepWidthAnchorName;
    case ModelSettings::NominalFootholdAnchor::kNominalStance:
      return kNominalStanceAnchorName;
  }
  return "unknown";
}

absl::StatusOr<ModelSettings::NominalFootholdConfig> nominalFootholdFromConfig(
    const mpc_config::NominalFootholdConfig& config) {
  RETURN_IF_ERROR(requireNonNegative(
      "nominal_foothold.nom_pos_lfoot_rt_rfoot_ewrt_heading_y",
      config.nom_pos_lfoot_rt_rfoot_ewrt_heading_y));
  ASSIGN_OR_RETURN(const ModelSettings::NominalFootholdAnchor anchor,
                   nominalFootholdAnchorFromName(config.anchor));
  return ModelSettings::NominalFootholdConfig{
      .nomPosLfootRtRfootEwrtHeadingY =
          config.nom_pos_lfoot_rt_rfoot_ewrt_heading_y,
      .anchor = anchor};
}

absl::StatusOr<ModelSettings> ModelSettings::Create(
    const mpc_config::TaskFile& taskFile,
    absl::string_view urdfFile,
    absl::string_view mpcName,
    bool verbose) {
  const mpc_config::ModelSettingsConfig& config = taskFile.model_settings;
  if (verbose) {
    humanoid_mpc_config::ModelSettingsConfig message;
    mpc_config::ToProto(config, &message);
    LOG(INFO) << "[ModelSettings] Robot Model Settings, the model_settings of "
                 "the task file:\n"
              << nproto::WriteTextproto(message);
  }

  ModelSettings settings;
  settings.robotName = config.robot_name;
  settings.verboseCppAd = config.verbose_cpp_ad;
  settings.recompileLibrariesCppAd = config.recompile_libraries_cpp_ad;
  // LINT.IfChange(cppad_model_folder)
  // One folder per state layout (StateLayout.h): a library taped for another
  // layout is never loaded for this one.
  settings.modelFolderCppAd =
      absl::StrCat("cppad_code_gen/cppad_", mpcName, settings.robotName, "/",
                   kStateLayoutTag);
  // LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/include/humanoid_common_mpc/common/StateLayout.h:cppad_layout_tag)
  RETURN_IF_ERROR(
      requireNonNegative("model_settings.phase_transition_stance_time",
                         config.phase_transition_stance_time));
  settings.phaseTransitionStanceTime = config.phase_transition_stance_time;
  settings.fixedJointNames = config.fixed_joint_names;
  settings.contactNames6DoF = config.contact_names_6dof;
  settings.contactParentJointNames = config.contact_parent_joint_names;
  ASSIGN_OR_RETURN(
      settings.pointContacts,
      pointContactsFromConfig(config.contacts_3dof, settings.contactNames6DoF));
  for (const ModelSettings::PointContact& contact : settings.pointContacts) {
    settings.contactNames3DoF.push_back(contact.name);
  }

  // The URDF's floating-base model, which the joint names and the hips of the
  // legs are read from.
  std::optional<PinocchioInterface> urdfModel;
  // NOLINTNEXTLINE(exceptions): Pinocchio's URDF parser reports a file it cannot read by throwing; converted to a Status here, once.
  try {
    urdfModel.emplace(createDefaultPinocchioInterface(urdfFile));
    // NOLINTNEXTLINE(exceptions): the boundary of the try above.
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(
        absl::StrCat("[ModelSettings] cannot read the joints of the URDF ",
                     urdfFile, ": ", error.what()));
  }
  settings.fullJointNames = urdfJointNames(urdfModel->getModel(), verbose);
  ASSIGN_OR_RETURN(settings.mpcModelJointNames,
                   activeJointNames(settings.fullJointNames,
                                    settings.fixedJointNames, verbose));
  settings.mpcModelToFullJointsIndices = mpcToFullJointIndices(
      settings.fullJointNames, settings.mpcModelJointNames);
  settings.jointIndexMap = indexMap(settings.mpcModelJointNames);
  settings.contactNames = settings.contactNames3DoF;
  settings.contactNames.insert(settings.contactNames.end(),
                               settings.contactNames6DoF.begin(),
                               settings.contactNames6DoF.end());
  // A mode number holds one bit per contact (common/ContactMask.h).
  if (settings.contactNames.size() > kMaxNumContacts) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ModelSettings] the contact set has ", settings.contactNames.size(),
        " contacts (model_settings.contacts_3dof and contact_names_6dof); at "
        "most ",
        kMaxNumContacts, " are supported"));
  }
  ASSIGN_OR_RETURN(settings.legs,
                   legsOf(config.legs, settings, urdfModel->getModel()));
  RETURN_IF_ERROR(checkLegNamesAgainstFootAliases(settings));
  settings.mpc_joint_dim = settings.mpcModelJointNames.size();
  settings.full_joint_dim = settings.fullJointNames.size();

  // The arm joints of the procedural arm swing: all four, or none (a legs-only
  // robot such as the EngineAI SA01, whose swing is then disabled). Any other
  // combination is refused, because a robot that names three of the four,
  // misspells one or fixes one out would otherwise walk with a half-built arm
  // swing.
  const mpc_config::ModelSettingsConfig::ArmJointNames& arms =
      config.arm_joint_names;
  settings.hasArmSwingJoints =
      !(arms.left_shoulder_y.empty() && arms.right_shoulder_y.empty() &&
        arms.left_elbow_y.empty() && arms.right_elbow_y.empty());
  if (settings.hasArmSwingJoints) {
    ASSIGN_OR_RETURN(settings.j_l_shoulder_y_index,
                     armJointIndex(settings.jointIndexMap, "left_shoulder_y",
                                   arms.left_shoulder_y));
    ASSIGN_OR_RETURN(settings.j_r_shoulder_y_index,
                     armJointIndex(settings.jointIndexMap, "right_shoulder_y",
                                   arms.right_shoulder_y));
    ASSIGN_OR_RETURN(settings.j_l_elbow_y_index,
                     armJointIndex(settings.jointIndexMap, "left_elbow_y",
                                   arms.left_elbow_y));
    ASSIGN_OR_RETURN(settings.j_r_elbow_y_index,
                     armJointIndex(settings.jointIndexMap, "right_elbow_y",
                                   arms.right_elbow_y));
  } else if (verbose) {
    LOG(INFO) << "[ModelSettings] model_settings.arm_joint_names is not set: "
                 "the procedural arm swing reference is disabled.";
  }

  ASSIGN_OR_RETURN(settings.footConstraintConfig,
                   footConstraintFromConfig(config.foot_constraint));
  RETURN_IF_ERROR(requireFinite("pos_surface_rt_world_ewrt_world_z",
                                taskFile.pos_surface_rt_world_ewrt_world_z));
  settings.posSurfaceRtWorldEwrtWorldZ =
      taskFile.pos_surface_rt_world_ewrt_world_z;
  settings.contactImplicitConfig =
      contactImplicitFromConfig(taskFile.contact_implicit);
  ASSIGN_OR_RETURN(settings.nominalFootholdConfig,
                   nominalFootholdFromConfig(taskFile.nominal_foothold));
  // The nominal_stance anchor places each leg's contacts; a set that is not
  // described by its legs has none to place, and the anchor would be silently
  // inert on it.
  if (settings.nominalFootholdConfig.anchor ==
          ModelSettings::NominalFootholdAnchor::kNominalStance &&
      !settings.isLeggedContactSet()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[ModelSettings] nominal_foothold.anchor '", kNominalStanceAnchorName,
        "' places every contact of a leg, and the contact set has ",
        settings.contactNames.size(), " contacts (",
        absl::StrJoin(settings.contactNames, ", "),
        ") that model_settings.legs does not all assign to a leg. List the "
        "legs, or use the anchor '",
        kStepWidthAnchorName, "'."));
  }
  return settings;
}

}  // namespace ocs2::humanoid
