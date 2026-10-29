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

#include "humanoid_multi_contact/scenarios/Scenario.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_multi_contact/scenarios/scenario.nproto.pb.h"
#include "humanoid_multi_contact/scenarios/scenario.pb.h"
#include "humanoid_multi_contact/stability/StaticEquilibrium.h"
#include "nproto/Textproto.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::multi_contact {
namespace {

// How far from 1 the norm of a pose's quaternion may be.
constexpr double kQuaternionNormTolerance = 1.0e-6;

absl::Status scenarioError(const config::Scenario& scenario,
                           absl::string_view what) {
  return absl::InvalidArgumentError(
      absl::StrCat("scenario '", scenario.name, "': ", what));
}

template <size_t kCount>
bool isOneOf(absl::string_view name,
             const std::array<absl::string_view, kCount>& names) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

/** Every name of `names` is a candidate and, when `allowed` is given, one of
 * them. */
absl::Status checkContacts(
    const config::Scenario& scenario,
    const std::vector<std::string>& names,
    absl::string_view field,
    const ContactSet& contactSet,
    const absl::flat_hash_set<std::string>* absl_nullable allowed) {
  for (const std::string& name : names) {
    if (!contactSet.indexOf(name).ok()) {
      return scenarioError(
          scenario, absl::StrCat(field, ": '", name,
                                 "' is not a candidate of the contact set "
                                 "of '",
                                 contactSet.robot(), "'"));
    }
    if (allowed != nullptr && !allowed->contains(name)) {
      return scenarioError(
          scenario, absl::StrCat(field, ": '", name, "' is not allowed"));
    }
  }
  return absl::OkStatus();
}

absl::Status checkPose(const config::Scenario& scenario,
                       const config::Scenario::Pose& pose,
                       absl::string_view field,
                       const ContactSet& contactSet,
                       const absl::flat_hash_set<std::string>& allowed) {
  if (absl::StatusOr<RobotConfiguration> configuration = configurationOf(pose);
      !configuration.ok()) {
    return scenarioError(
        scenario, absl::StrCat(field, ": ", configuration.status().message()));
  }
  return checkContacts(scenario, pose.active_contacts,
                       absl::StrCat(field, ".active_contacts"), contactSet,
                       &allowed);
}

absl::Status checkSimulationStart(const config::Scenario& scenario) {
  const config::Scenario::SimulationStart& start = scenario.simulation_start;
  for (const std::string& arming :
       {start.tilt_catch_arming, start.gantry_arming}) {
    if (!isOneOf(arming, kArmingNames)) {
      return scenarioError(
          scenario,
          absl::StrCat("simulation_start: unknown arming '", arming,
                       "' (known: ", absl::StrJoin(kArmingNames, ", "), ")"));
    }
    if (arming == "when_standing" && !(start.standing_base_height > 0.0)) {
      return scenarioError(
          scenario,
          "simulation_start: \"when_standing\" needs a positive "
          "standing_base_height");
    }
  }
  return absl::OkStatus();
}

absl::Status checkMargins(const config::Scenario& scenario) {
  if (scenario.static_margin.has_value() && !(*scenario.static_margin >= 0.0)) {
    return scenarioError(scenario, "static_margin must not be negative");
  }
  if (!(scenario.torque_margin > 0.0 && scenario.torque_margin <= 1.0)) {
    return scenarioError(scenario, "torque_margin must lie in (0, 1]");
  }
  ASSIGN_OR_RETURN(const JointLimits limits,
                   jointLimitsOf(scenario.joint_limits));
  if (relaxesPositionLimits(limits)
          ? !(scenario.joint_limit_relaxation > 0.0 &&
              scenario.joint_limit_relaxation <= kMaxJointLimitRelaxation)
          : scenario.joint_limit_relaxation != 0.0) {
    return scenarioError(scenario,
                         absl::StrCat("joint_limit_relaxation must lie in (0, ",
                                      kMaxJointLimitRelaxation,
                                      "] under \"relaxed_position\" and be 0 "
                                      "otherwise"));
  }
  if (const absl::StatusOr<LinkAbove> linkAbove =
          linkAboveOf(scenario.point_pad_link_above);
      !linkAbove.ok()) {
    return scenarioError(scenario, linkAbove.status().message());
  }
  if (scenario.sphere_clearance.has_value() &&
      !(*scenario.sphere_clearance > 0.0)) {
    return scenarioError(scenario, "sphere_clearance must be positive");
  }
  if (const absl::StatusOr<SupportPolygon> polygon =
          supportPolygonOf(scenario.support_polygon);
      !polygon.ok()) {
    return scenarioError(scenario, polygon.status().message());
  }
  if (!(scenario.duration > 0.0) || !std::isfinite(scenario.duration)) {
    return scenarioError(scenario, "duration must be positive");
  }
  return absl::OkStatus();
}

/** Checks the goal stance `goal`, named `field` in messages (the goal or a
 * waypoint). */
absl::Status checkGoalStance(const config::Scenario& scenario,
                             const config::Scenario::Goal& goal,
                             absl::string_view field,
                             const ContactSet& contactSet,
                             const TerrainModel& terrain,
                             const absl::flat_hash_set<std::string>& allowed) {
  if (goal.contacts.empty()) {
    return scenarioError(scenario, absl::StrCat(field, " names no contact"));
  }
  std::vector<std::string> names;
  for (const config::Scenario::GoalContact& contact : goal.contacts) {
    names.push_back(contact.contact);
    if (!terrain.primitiveIndex(contact.primitive).ok()) {
      return scenarioError(scenario,
                           absl::StrCat(field, ": contact '", contact.contact,
                                        "' rests on the unknown primitive '",
                                        contact.primitive, "'"));
    }
  }
  RETURN_IF_ERROR(checkContacts(
      scenario, names, absl::StrCat(field, ".contacts"), contactSet, &allowed));
  if (goal.base_region.has_value()) {
    const config::Scenario::BaseRegion& region = *goal.base_region;
    if (region.center.size() != kPositionDim ||
        region.half_extents.size() != kPositionDim ||
        !(region.half_extents.minCoeff() > 0.0)) {
      return scenarioError(
          scenario, absl::StrCat(field,
                                 ".base_region needs a center and positive "
                                 "half extents of three values"));
    }
  }
  if (goal.pose.has_value()) {
    RETURN_IF_ERROR(checkPose(scenario, *goal.pose,
                              absl::StrCat(field, ".pose"), contactSet,
                              allowed));
  }
  return absl::OkStatus();
}

absl::Status checkGoal(const config::Scenario& scenario,
                       const ContactSet& contactSet,
                       const TerrainModel& terrain,
                       const absl::flat_hash_set<std::string>& allowed) {
  RETURN_IF_ERROR(checkGoalStance(scenario, scenario.goal, "goal", contactSet,
                                  terrain, allowed));
  for (size_t i = 0; i < scenario.waypoints.size(); ++i) {
    RETURN_IF_ERROR(checkGoalStance(scenario, scenario.waypoints[i],
                                    absl::StrCat("waypoints[", i, "]"),
                                    contactSet, terrain, allowed));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<config::Scenario> loadScenarioFile(absl::string_view path) {
  return nproto::LoadTextprotoFile<config::Scenario,
                                   humanoid_multi_contact::Scenario>(path);
}

absl::Status validateScenario(const config::Scenario& scenario,
                              const ContactSet& contactSet,
                              const TerrainModel& terrain) {
  if (scenario.name.empty() || scenario.robot.empty()) {
    return scenarioError(scenario, "needs a name and a robot");
  }
  if (!isOneOf(scenario.planner, kPlannerNames)) {
    return scenarioError(
        scenario,
        absl::StrCat("unknown planner '", scenario.planner,
                     "' (known: ", absl::StrJoin(kPlannerNames, ", "), ")"));
  }
  if (absl::StatusOr<JointLimits> limits = jointLimitsOf(scenario.joint_limits);
      !limits.ok()) {
    return scenarioError(scenario, limits.status().message());
  }
  RETURN_IF_ERROR(checkMargins(scenario));
  RETURN_IF_ERROR(checkSimulationStart(scenario));
  RETURN_IF_ERROR(checkContacts(scenario, scenario.allowed_contacts,
                                "allowed_contacts", contactSet,
                                /*allowed=*/nullptr));
  const absl::flat_hash_set<std::string> allowed(
      scenario.allowed_contacts.begin(), scenario.allowed_contacts.end());
  RETURN_IF_ERROR(checkPose(scenario, scenario.initial_pose, "initial_pose",
                            contactSet, allowed));
  return checkGoal(scenario, contactSet, terrain, allowed);
}

absl::StatusOr<RobotConfiguration> configurationOf(
    const config::Scenario::Pose& pose) {
  if (pose.base_position.size() != kPositionDim ||
      pose.base_orientation_xyzw.size() != kQuaternionCoefficients ||
      !(std::abs(pose.base_orientation_xyzw.norm() - 1.0) <=
        kQuaternionNormTolerance)) {
    return absl::InvalidArgumentError(
        absl::StrCat("pose '", pose.name,
                     "' needs a base_position of three values and a unit "
                     "base_orientation_xyzw"));
  }
  RobotConfiguration configuration;
  configuration.basePosition = Eigen::Vector3d(pose.base_position);
  configuration.baseOrientation =
      Eigen::Quaterniond(Eigen::Vector4d(pose.base_orientation_xyzw));
  configuration.jointPositions.resize(pose.joint_positions.size());
  for (size_t i = 0; i < pose.joint_positions.size(); ++i) {
    configuration.jointNames.push_back(pose.joint_positions[i].joint);
    configuration.jointPositions(static_cast<Eigen::Index>(i)) =
        pose.joint_positions[i].position;
  }
  return configuration;
}

absl::StatusOr<ContactMask> contactsOf(const config::Scenario::Pose& pose,
                                       const ContactSet& contactSet) {
  return contactSet.maskOf(pose.active_contacts);
}

absl::StatusOr<CertifierOptions> certifierOptionsOf(
    const config::Scenario& scenario) {
  CertifierOptions options;
  options.staticMargin = scenario.static_margin;
  ASSIGN_OR_RETURN(options.jointLimits, jointLimitsOf(scenario.joint_limits));
  options.torqueMargin = scenario.torque_margin;
  ASSIGN_OR_RETURN(options.equilibrium.supportPolygon,
                   supportPolygonOf(scenario.support_polygon));
  return options;
}

absl::StatusOr<LoadedScenario> loadScenario(absl::string_view path) {
  ASSIGN_OR_RETURN(config::Scenario scenario, loadScenarioFile(path));
  ASSIGN_OR_RETURN(const std::string contactSetPath,
                   robot::resolveResourcePath(scenario.contact_set_file));
  ASSIGN_OR_RETURN(config::ContactSet contactSetConfig,
                   loadContactSetFile(contactSetPath));
  ASSIGN_OR_RETURN(ContactSet contactSet, ContactSet::Create(contactSetConfig));
  ASSIGN_OR_RETURN(TerrainModel terrain,
                   TerrainModel::Create(scenario.terrain_scene));
  ASSIGN_OR_RETURN(std::string urdfPath,
                   robot::resolveResourcePath(scenario.urdf_file));
  std::optional<config::CollisionModel> collisionModelConfig;
  if (!scenario.collision_model_file.empty()) {
    ASSIGN_OR_RETURN(const std::string collisionModelPath,
                     robot::resolveResourcePath(scenario.collision_model_file));
    ASSIGN_OR_RETURN(collisionModelConfig,
                     loadCollisionModelFile(collisionModelPath));
  }
  RETURN_IF_ERROR(validateScenario(scenario, contactSet, terrain));
  return LoadedScenario{
      .scenario = std::move(scenario),
      .contactSetConfig = std::move(contactSetConfig),
      .contactSet = std::move(contactSet),
      .terrain = std::move(terrain),
      .urdfPath = std::move(urdfPath),
      .collisionModelConfig = std::move(collisionModelConfig)};
}

absl::StatusOr<std::unique_ptr<StabilityCertifier>> makeCertifier(
    const LoadedScenario& loaded) {
  ASSIGN_OR_RETURN(
      std::unique_ptr<RobotStanceModel> model,
      RobotStanceModel::Create(loaded.urdfPath, loaded.contactSet));
  std::optional<CollisionModel> collisionModel;
  if (loaded.collisionModelConfig.has_value()) {
    ASSIGN_OR_RETURN(collisionModel, CollisionModel::Create(
                                         *loaded.collisionModelConfig, *model));
    ASSIGN_OR_RETURN(const LinkAbove linkAbove,
                     linkAboveOf(loaded.scenario.point_pad_link_above));
    collisionModel->setLinkAbove(linkAbove);
  }
  ASSIGN_OR_RETURN(const CertifierOptions options,
                   certifierOptionsOf(loaded.scenario));
  return StabilityCertifier::Create(std::move(model), loaded.terrain, options,
                                    std::move(collisionModel));
}

}  // namespace ocs2::humanoid::multi_contact
