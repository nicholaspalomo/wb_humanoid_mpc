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

// Pinocchio forward declarations must be included first.
#include "pinocchio/fwd.hpp"

#include "humanoid_common_mpc_app/visualization/SceneBuilder.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody.hpp"
#include "pinocchio/multibody/fwd.hpp"
#include "pinocchio/multibody/joint.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial.hpp"
#include "urdf_world/types.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/costs/CollisionConstraintFromConfig.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/contact/ContactWrenchMapper.h"
#include "humanoid_common_mpc/contact/RollingContact.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/PinocchioFrameConversions.h"
#include "humanoid_common_mpc/pinocchio_model/pinocchioUtils.h"
#include "humanoid_common_mpc_app/visualization/PolicySnapshot.h"
#include "humanoid_common_mpc_app/visualization/SceneContract.h"
#include "humanoid_common_mpc_app/visualization/VisualizationConfig.h"
#include "humanoid_common_mpc_app/visualization/VisualizationModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_msgs/arrows.pb.h"
#include "humanoid_mpc_msgs/color.pb.h"
#include "humanoid_mpc_msgs/line_strip.pb.h"
#include "humanoid_mpc_msgs/pose.pb.h"
#include "humanoid_mpc_msgs/quaternion.pb.h"
#include "humanoid_mpc_msgs/robot_model_instance.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.nproto.h"
#include "humanoid_mpc_msgs/spheres.pb.h"
#include "humanoid_mpc_msgs/vector3.pb.h"
#include "humanoid_mpc_msgs/visualization_scene.pb.h"
#include "robot_model/UrdfContinuousJoints.h"

namespace ocs2::humanoid::visualization {

namespace {

using Model = PinocchioInterface::Model;
using Data = PinocchioInterface::Data;

/** The frames of the task file's collision constraint that a collision sphere
 * may be drawn at. */
constexpr size_t kCollisionSphereFrames = 10;
/** The line strips of the plan: the end effectors', the base's and the CoM's
 * paths. */
constexpr size_t kPlanLineStrips = 3;

/**
 * Returns the point a contact touches the ground at, from the origin
 * `posLinkRtWorldEwrtWorld` of its frame: the frame's origin for a fixed
 * contact (`wheelRadius` 0), the wheel's lowest point below its center for a
 * rolling one (contact/RollingContact.h).
 */
vector3_t contactPointOfFrame(const vector3_t& posLinkRtWorldEwrtWorld,
                              scalar_t wheelRadius) {
  if (wheelRadius > 0.0) {
    return posContactRtWorldEwrtWorld(posLinkRtWorldEwrtWorld, wheelRadius,
                                      normalTerrainEwrtWorld<scalar_t>());
  }
  return posLinkRtWorldEwrtWorld;
}

void setVector3(const vector3_t& value,
                humanoid_mpc_msgs::Vector3* absl_nonnull message) {
  message->set_x(value.x());
  message->set_y(value.y());
  message->set_z(value.z());
}

void setColor(const scene::Rgba& color,
              humanoid_mpc_msgs::Color* absl_nonnull message) {
  message->set_r(color.r);
  message->set_g(color.g);
  message->set_b(color.b);
  message->set_a(color.a);
}

/** Appends to `arrows` an arrow of the force `forceLocalEwrtWorld` ending at
 * the point it acts at, `posLocalRtWorldEwrtWorld` (a contact frame, a center
 * of pressure or a polygon corner, as the caller passes):
 * kNewtonsPerArrowMeter newtons per meter. */
void addContactArrow(const vector3_t& posLocalRtWorldEwrtWorld,
                     const vector3_t& forceLocalEwrtWorld,
                     humanoid_mpc_msgs::Arrows* absl_nonnull arrows) {
  const vector3_t posArrowheadRtArrowEwrtWorld =
      forceLocalEwrtWorld / scene::kNewtonsPerArrowMeter;
  setVector3(posLocalRtWorldEwrtWorld - posArrowheadRtArrowEwrtWorld,
             arrows->add_pos_arrows_rt_world_ewrt_world());
  setVector3(posArrowheadRtArrowEwrtWorld,
             arrows->add_pos_arrowheads_rt_arrow_ewrt_world());
}

/** The BODY frames of `model` (one per URDF link), without the universe. */
void collectLinkFrames(const Model& model,
                       std::vector<pinocchio::FrameIndex>* absl_nonnull frames,
                       std::vector<std::string>* absl_nonnull names) {
  frames->clear();
  names->clear();
  for (size_t frame = 0; frame < model.frames.size(); ++frame) {
    if (model.frames[frame].type == pinocchio::BODY &&
        model.frames[frame].name != "universe") {
      frames->push_back(static_cast<pinocchio::FrameIndex>(frame));
      names->push_back(model.frames[frame].name);
    }
  }
}

/** The world pose of every link of `frames` (frame placements must be up to
 * date) as the instance `name`. */
void writePoseLinksRtWorld(
    const Data& data,
    absl::Span<const pinocchio::FrameIndex> frames,
    absl::Span<const std::string> names,
    absl::string_view name,
    humanoid_mpc_msgs::RobotModelInstance* absl_nonnull instance) {
  instance->Clear();
  instance->set_name(std::string(name));
  for (size_t link = 0; link < frames.size(); ++link) {
    instance->add_link_names(names[link]);
    const pinocchio::SE3Tpl<scalar_t>& poseLinkRtWorld = data.oMf[frames[link]];
    humanoid_mpc_msgs::Pose* absl_nonnull msgPoseLinkRtWorld =
        instance->add_pose_links_rt_world();
    const quaternion_t quatLinkRtWorld(poseLinkRtWorld.rotation());
    // NOLINTBEGIN(frame-naming): the components of a Pose message, whose frames its field names
    setVector3(poseLinkRtWorld.translation(),
               msgPoseLinkRtWorld->mutable_position());
    humanoid_mpc_msgs::Quaternion* absl_nonnull msgQuatLinkRtWorld =
        msgPoseLinkRtWorld->mutable_orientation();
    // NOLINTEND(frame-naming)
    msgQuatLinkRtWorld->set_w(quatLinkRtWorld.w());
    msgQuatLinkRtWorld->set_x(quatLinkRtWorld.x());
    msgQuatLinkRtWorld->set_y(quatLinkRtWorld.y());
    msgQuatLinkRtWorld->set_z(quatLinkRtWorld.z());
  }
}

absl::StatusOr<pinocchio::FrameIndex> findFrame(const Model& model,
                                                absl::string_view frame,
                                                absl::string_view what) {
  // Pinocchio looks frames up by const std::string&.
  const std::string frameName(frame);
  if (!model.existFrame(frameName)) {
    return absl::InvalidArgumentError(
        absl::StrCat(what, " names the frame '", frame,
                     "', which the MPC's robot model does not have."));
  }
  return model.getFrameId(frameName);
}

}  // namespace

SceneBuilder::SceneBuilder(const VisualizationModel& model,
                           const VisualizationConfig& /*config*/)
    : pinocchioInterface_(*model.pinocchioInterface),
      robotModel_(model.mpcRobotModel->clone()),
      stateManifold_(model.stateManifold) {}

absl::StatusOr<std::unique_ptr<SceneBuilder>> SceneBuilder::Create(
    const VisualizationModel& model, const VisualizationConfig& config) {
  RETURN_IF_ERROR(checkVisualizationModel(model));
  std::unique_ptr<SceneBuilder> builder =
      absl::WrapUnique(new SceneBuilder(model, config));
  RETURN_IF_ERROR(builder->initialize(model, config));
  return builder;
}

absl::Status SceneBuilder::initialize(const VisualizationModel& model,
                                      const VisualizationConfig& config) {
  const Model& mpcModel = pinocchioInterface_.getModel();
  collectLinkFrames(mpcModel, &mpcLinkFrames_, &mpcLinkNames_);

  // A continuous joint gets one coordinate, its angle, as in the MPC's model.
  absl::StatusOr<urdf::ModelInterfaceSharedPtr> urdfTree =
      robot::model::parseUrdfFileForPinocchio(model.urdfFile);
  if (!urdfTree.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the URDF '", model.urdfFile,
        "' does not give a Pinocchio model: ", urdfTree.status().message()));
  }
  try {  // NOLINT(exceptions): Pinocchio's URDF parser throws; its exception becomes a Status here.
    pinocchio::urdf::buildModel(
        *urdfTree, pinocchio::JointModelFreeFlyerTpl<scalar_t>(), fullModel_);
    // NOLINTNEXTLINE(exceptions): the boundary of the try above.
  } catch (const std::exception& e) {
    return absl::InvalidArgumentError(
        absl::StrCat("the URDF '", model.urdfFile,
                     "' does not give a Pinocchio model: ", e.what()));
  }
  fullData_ = std::make_unique<Data>(fullModel_);
  fullConfiguration_ = pinocchio::neutral(fullModel_);
  collectLinkFrames(fullModel_, &fullLinkFrames_, &fullLinkNames_);
  for (pinocchio::JointIndex joint = kFirstJointAfterFloatingBase;
       joint < static_cast<pinocchio::JointIndex>(fullModel_.njoints);
       ++joint) {
    // One coordinate: a revolute (a continuous one included) or prismatic
    // joint.
    if (fullModel_.joints[joint].nq() == 1) {
      fullJoints_.push_back(
          FullModelJoint{.name = fullModel_.names[joint],
                         .idxQ = fullModel_.joints[joint].idx_q()});
    }
  }

  if (modelSettings().contactNames.empty()) {
    return absl::InvalidArgumentError(
        "the model settings name no contact; the visualization needs one.");
  }
  // The contact polygons and the collision spheres of the typed task file.
  const absl::StatusOr<mpc_config::TaskFile> task =
      loadTaskFile(model.taskFile);
  if (!task.ok()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the contact polygons do not load: ", task.status().message()));
  }
  const size_t numContacts = modelSettings().contactNames.size();
  contactFrames_.assign(numContacts, 0);
  contactWheelRadii_.assign(numContacts, 0.0);
  cornerFrames_.assign(numContacts, {});
  cornerMappers_.assign(numContacts, std::nullopt);
  for (size_t contact = 0; contact < numContacts; ++contact) {
    ASSIGN_OR_RETURN(contactFrames_[contact],
                     findFrame(mpcModel, modelSettings().contactNames[contact],
                               "contactNames"));
    // A 3-DoF contact has no rectangle: its force is drawn at its frame, or a
    // rolling contact's at its wheel's lowest point.
    if (modelSettings().isRollingContact(contact)) {
      contactWheelRadii_[contact] = modelSettings().wheelRadius(contact);
    }
    if (modelSettings().isPointContact(contact)) continue;
    const absl::StatusOr<ContactRectangle> rectangle =
        contactRectangleFromConfig(task->contacts, modelSettings(),
                                   static_cast<int>(contact));
    if (!rectangle.ok()) {
      return absl::InvalidArgumentError(
          absl::StrCat(model.taskFile, ": the contact polygon of ",
                       modelSettings().contactNames[contact],
                       " does not load: ", rectangle.status().message()));
    }
    if (rectangle->getNumberOfContactPoints() != kFootCorners) {
      return absl::InvalidArgumentError(absl::StrCat(
          "the contact polygon of ", modelSettings().contactNames[contact],
          " has ", rectangle->getNumberOfContactPoints(),
          " corners; the corner forces need ", kFootCorners, "."));
    }
    for (size_t corner = 0; corner < kFootCorners; ++corner) {
      ASSIGN_OR_RETURN(cornerFrames_[contact][corner],
                       findFrame(mpcModel,
                                 rectangle->getPolygonPointFrameName(
                                     static_cast<int>(corner)),
                                 "the contact polygon"));
    }
    cornerMappers_[contact].emplace(*rectangle);
  }

  for (const std::string& frame : config.planFrames) {
    ASSIGN_OR_RETURN(const pinocchio::FrameIndex index,
                     findFrame(mpcModel, frame, kRerunPlanFramesField));
    planFrames_.push_back(index);
    // A plan frame that is a rolling contact's is drawn at its contact point.
    scalar_t wheelRadius = 0.0;
    for (size_t contact = 0; contact < numContacts; ++contact) {
      if (contactFrames_[contact] == index) {
        wheelRadius = contactWheelRadii_[contact];
      }
    }
    planFrameWheelRadii_.push_back(wheelRadius);
  }

  const absl::StatusOr<FootCollisionConstraint::Config> loaded =
      footCollisionConstraintConfigFromConfig(task->collision_constraint);
  if (!loaded.ok()) {
    LOG(WARNING) << "[SceneBuilder] " << model.taskFile
                 << ": the collision spheres do not load, none is drawn: "
                 << loaded.status().message();
    return absl::OkStatus();
  }
  const FootCollisionConstraint::Config& collision = *loaded;
  const std::array<std::pair<const std::string* absl_nonnull, scalar_t>,
                   kCollisionSphereFrames>
      candidates = {{
          {&collision.leftAnkleFrame, collision.footCollisionSphereRadius},
          {&collision.rightAnkleFrame, collision.footCollisionSphereRadius},
          {&collision.leftFootCenterFrame, collision.footCollisionSphereRadius},
          {&collision.rightFootCenterFrame,
           collision.footCollisionSphereRadius},
          {&collision.leftFootFrame1, collision.footCollisionSphereRadius},
          {&collision.rightFootFrame1, collision.footCollisionSphereRadius},
          {&collision.leftFootFrame2, collision.footCollisionSphereRadius},
          {&collision.rightFootFrame2, collision.footCollisionSphereRadius},
          {&collision.leftKneeFrame, collision.kneeCollisionSphereRadius},
          {&collision.rightKneeFrame, collision.kneeCollisionSphereRadius},
      }};
  // As HumanoidVisualizer drew them: a frame the task file leaves unnamed, or
  // that the model lacks, has no sphere.
  for (const std::pair<const std::string* absl_nonnull, scalar_t>& candidate :
       candidates) {
    if (!candidate.first->empty() && mpcModel.existFrame(*candidate.first) &&
        std::isfinite(candidate.second) && candidate.second > 0.0) {
      collisionSpheres_.push_back(
          CollisionSphere{.frame = mpcModel.getFrameId(*candidate.first),
                          .radius = candidate.second});
    }
  }
  return absl::OkStatus();
}

void SceneBuilder::writeMeasuredFromSample(
    const msgs::RobotStateSample& sample,
    humanoid_mpc_msgs::RobotModelInstance* absl_nonnull instance) {
  if (!hasSampleJointIndices_ || sample.joint_names != sampleJointNames_) {
    absl::flat_hash_map<std::string, int> sampleIndices;
    for (size_t index = 0; index < sample.joint_names.size(); ++index) {
      sampleIndices.emplace(sample.joint_names[index], static_cast<int>(index));
    }
    for (FullModelJoint& joint : fullJoints_) {
      const absl::flat_hash_map<std::string, int>::const_iterator found =
          sampleIndices.find(joint.name);
      joint.sampleIndex = found != sampleIndices.end() ? found->second : -1;
    }
    sampleJointNames_ = sample.joint_names;
    hasSampleJointIndices_ = true;
  }

  // The free flyer's configuration is the position and the quaternion (x, y, z,
  // w).
  quaternion_t quatBaseRtWorld(
      vector4_t(sample.quat_base_rt_world.x, sample.quat_base_rt_world.y,
                sample.quat_base_rt_world.z, sample.quat_base_rt_world.w));
  if (quatBaseRtWorld.norm() > 0.0) {
    quatBaseRtWorld.normalize();
  } else {
    quatBaseRtWorld = quaternion_t::Identity();
  }
  fullConfiguration_.head<kPositionDim>() =
      vector3_t(sample.pos_base_rt_world_ewrt_world.x,
                sample.pos_base_rt_world_ewrt_world.y,
                sample.pos_base_rt_world_ewrt_world.z);
  fullConfiguration_.segment<kQuaternionCoefficients>(kPositionDim) =
      quatBaseRtWorld.coeffs();
  for (const FullModelJoint& joint : fullJoints_) {
    if (joint.sampleIndex < 0 ||
        joint.sampleIndex >= sample.joint_positions.size()) {
      continue;
    }
    fullConfiguration_[joint.idxQ] = sample.joint_positions[joint.sampleIndex];
  }
  pinocchio::forwardKinematics(fullModel_, *fullData_, fullConfiguration_);
  pinocchio::updateFramePlacements(fullModel_, *fullData_);
  writePoseLinksRtWorld(*fullData_, fullLinkFrames_, fullLinkNames_,
                        scene::kMeasuredInstance, instance);
}

void SceneBuilder::writeObservationMarkers(
    const SystemObservation& observation,
    const PolicySnapshot* absl_nullable policy,
    humanoid_mpc_msgs::VisualizationScene* absl_nonnull scene) {
  humanoid_mpc_msgs::Arrows* absl_nonnull contactArrows = scene->add_arrows();
  contactArrows->set_path(std::string(scene::kContactForceMarkers));
  humanoid_mpc_msgs::Spheres* absl_nonnull centerOfPressure =
      scene->add_spheres();
  centerOfPressure->set_path(std::string(scene::kCenterOfPressure));
  humanoid_mpc_msgs::Arrows* absl_nonnull cornerArrows = scene->add_arrows();
  cornerArrows->set_path(std::string(scene::kCornerForceMarkers));
  humanoid_mpc_msgs::Spheres* absl_nonnull collisionSpheres =
      scene->add_spheres();
  collisionSpheres->set_path(std::string(scene::kCollisionSpheres));

  // The frame placements are those of the observation's state (build() updated
  // them).
  const Data& data = pinocchioInterface_.getData();
  for (const CollisionSphere& sphere : collisionSpheres_) {
    setVector3(data.oMf[sphere.frame].translation(),
               collisionSpheres->add_pos_spheres_rt_world_ewrt_world());
    collisionSpheres->add_radii(static_cast<float>(sphere.radius));
  }

  if (policy == nullptr) {
    return;
  }
  vector_t planState;
  vector_t planInput;
  samplePlan(*policy, stateManifold_, observation.time, &planState, &planInput);
  // The sums of f_z c and f_z over the 6-DoF stance contacts, c the center of
  // pressure and f_z the normal force in the world frame.
  vector3_t weightedSumPosCopRtWorldEwrtWorld = vector3_t::Zero();
  scalar_t sumForceContactsEwrtWorldZ = 0.0;
  const size_t numContacts = contactFrames_.size();
  for (size_t contact = 0; contact < numContacts; ++contact) {
    // The contact's bit of the mode number; a number with a bit past the
    // contact set is no mode, and no contact is active in it.
    if (!contactIsActiveInMode(observation.mode, contact, numContacts)) {
      continue;
    }
    const pinocchio::FrameIndex frame = contactFrames_[contact];
    const vector6_t wrenchContactEwrtWorld =
        robotModel_->getWrenchContactEwrtWorld(observation.state, planInput,
                                               contact);
    const std::optional<ContactWrenchMapper<kFootCorners>>& cornerMapper =
        cornerMappers_[contact];
    if (!cornerMapper.has_value()) {
      addContactArrow(contactPointOfFrame(data.oMf[frame].translation(),
                                          contactWheelRadii_[contact]),
                      wrenchContactEwrtWorld.head<kForceDim>(), contactArrows);
      continue;
    }
    const vector6_t wrenchContactEwrtContact =
        rotateVectorWorldToLocal<scalar_t>(wrenchContactEwrtWorld, data, frame);
    vector3_t posCopRtWorldEwrtWorld = data.oMf[frame].translation();
    if (wrenchContactEwrtContact[kWrenchForceZIndex] >
        kCopMinForceContactEwrtContactZ) {
      const vector3_t posCopRtContactEwrtContact(
          -wrenchContactEwrtContact[kWrenchTorqueYIndex] /
              wrenchContactEwrtContact[kWrenchForceZIndex],
          wrenchContactEwrtContact[kWrenchTorqueXIndex] /
              wrenchContactEwrtContact[kWrenchForceZIndex],
          0.0);
      posCopRtWorldEwrtWorld = data.oMf[frame].act(posCopRtContactEwrtContact);
      weightedSumPosCopRtWorldEwrtWorld +=
          wrenchContactEwrtWorld[kWrenchForceZIndex] * posCopRtWorldEwrtWorld;
      sumForceContactsEwrtWorldZ += wrenchContactEwrtWorld[kWrenchForceZIndex];
    }
    addContactArrow(posCopRtWorldEwrtWorld,
                    wrenchContactEwrtWorld.head<kForceDim>(), contactArrows);

    // The force at each corner, indexed by corner.
    const std::array<vector3_t, kFootCorners> forceCornersEwrtContact =
        cornerMapper->computeForceCornersEwrtContact(wrenchContactEwrtContact);
    for (size_t corner = 0; corner < kFootCorners; ++corner) {
      addContactArrow(data.oMf[cornerFrames_[contact][corner]].translation(),
                      rotateVectorLocalToWorld<scalar_t>(
                          forceCornersEwrtContact[corner], data, frame),
                      cornerArrows);
    }
  }
  if (sumForceContactsEwrtWorldZ > kCopMinSumForceContactsEwrtWorldZ) {
    setVector3(weightedSumPosCopRtWorldEwrtWorld / sumForceContactsEwrtWorldZ,
               centerOfPressure->add_pos_spheres_rt_world_ewrt_world());
  }
}

void SceneBuilder::clearPlan() {
  hasPlan_ = false;
  hasTerminalState_ = false;
  hasTerminalTarget_ = false;
  planEndEffectors_.Clear();
  planBase_.Clear();
  planCom_.Clear();
  planFootholds_.Clear();
}

void SceneBuilder::computePlan(const PolicySnapshot& policy) {
  clearPlan();
  const Model& model = pinocchioInterface_.getModel();
  Data& data = pinocchioInterface_.getData();

  for (size_t frame = 0; frame < planFrames_.size(); ++frame) {
    planEndEffectors_.add_strips();
  }
  humanoid_mpc_msgs::LineStrip* absl_nonnull baseStrip = planBase_.add_strips();
  humanoid_mpc_msgs::LineStrip* absl_nonnull comStrip = planCom_.add_strips();
  for (const vector_t& state : policy.state) {
    const vector_t q = robotModel_->getGeneralizedCoordinates(state);
    // centerOfMass() runs the forward kinematics the frame placements below are
    // updated from. The CoM is drawn where it is, at its height: its height is
    // the pendulum height of the plan, which a projection to the ground would
    // hide.
    setVector3(
        pinocchio::centerOfMass(model, data, q, /*computeSubtreeComs=*/false),
        comStrip->add_pos_strippoints_rt_world_ewrt_world());
    setVector3(robotModel_->getPosBaseRtWorldEwrtWorld(state),
               baseStrip->add_pos_strippoints_rt_world_ewrt_world());
    for (size_t frame = 0; frame < planFrames_.size(); ++frame) {
      setVector3(contactPointOfFrame(pinocchio::updateFramePlacement(
                                         model, data, planFrames_[frame])
                                         .translation(),
                                     planFrameWheelRadii_[frame]),
                 planEndEffectors_.mutable_strips(static_cast<int>(frame))
                     ->add_pos_strippoints_rt_world_ewrt_world());
    }
  }

  // A foot lands at an event inside the horizon where it goes from swing to
  // stance.
  const scalar_t startTime = policy.time.front();
  const scalar_t finalTime = policy.time.back();
  const ModeSchedule& schedule = policy.modeSchedule;
  vector_t eventState;
  vector_t eventInput;
  for (size_t event = 0; event < schedule.eventTimes.size() &&
                         event + 1 < schedule.modeSequence.size();
       ++event) {
    const scalar_t eventTime = schedule.eventTimes[event];
    if (!(startTime < eventTime && eventTime < finalTime)) {
      continue;
    }
    const size_t numContacts = contactFrames_.size();
    bool kinematicsDone = false;
    for (size_t contact = 0; contact < numContacts; ++contact) {
      // A contact lands at an event where it goes from swing to stance.
      if (contactIsActiveInMode(schedule.modeSequence[event], contact,
                                numContacts) ||
          !contactIsActiveInMode(schedule.modeSequence[event + 1], contact,
                                 numContacts)) {
        continue;
      }
      if (!kinematicsDone) {
        samplePlan(policy, stateManifold_, eventTime, &eventState, &eventInput);
        pinocchio::forwardKinematics(
            model, data, robotModel_->getGeneralizedCoordinates(eventState));
        kinematicsDone = true;
      }
      setVector3(contactPointOfFrame(pinocchio::updateFramePlacement(
                                         model, data, contactFrames_[contact])
                                         .translation(),
                                     contactWheelRadii_[contact]),
                 planFootholds_.add_pos_spheres_rt_world_ewrt_world());
      setColor(scene::kContactColors[contact % scene::kContactColors.size()],
               planFootholds_.add_colors());
    }
  }

  pinocchio::forwardKinematics(
      model, data, robotModel_->getGeneralizedCoordinates(policy.state.back()));
  pinocchio::updateFramePlacements(model, data);
  writePoseLinksRtWorld(data, mpcLinkFrames_, mpcLinkNames_,
                        scene::kTerminalStateInstance, &terminalState_);
  hasTerminalState_ = true;

  vector_t targetState;
  vector_t targetInput;
  if (sampleTarget(policy.target, stateManifold_, finalTime,
                   robotModel_->getStateDim(), robotModel_->getInputDim(),
                   &targetState, &targetInput) != TargetSample::kNone) {
    pinocchio::forwardKinematics(
        model, data, robotModel_->getGeneralizedCoordinates(targetState));
    pinocchio::updateFramePlacements(model, data);
    writePoseLinksRtWorld(data, mpcLinkFrames_, mpcLinkNames_,
                          scene::kTerminalTargetInstance, &terminalTarget_);
    hasTerminalTarget_ = true;
  }
  hasPlan_ = true;
}

absl::Status SceneBuilder::build(
    const SceneInputs& inputs,
    humanoid_mpc_msgs::VisualizationScene* absl_nonnull scene) {
  scene->Clear();
  const bool hasObservation =
      inputs.observation != nullptr &&
      static_cast<size_t>(inputs.observation->state.size()) ==
          robotModel_->getStateDim();
  const bool hasPolicy =
      inputs.policy != nullptr &&
      isConsistentPlan(*inputs.policy, robotModel_->getStateDim(),
                       robotModel_->getInputDim());
  if (!hasObservation && inputs.robotState == nullptr) {
    return absl::FailedPreconditionError(
        "there is neither an observation nor a robot/state sample to draw.");
  }
  scene->set_time(inputs.robotState != nullptr ? inputs.robotState->time
                                               : inputs.observation->time);

  if (inputs.robotState != nullptr) {
    writeMeasuredFromSample(*inputs.robotState, scene->add_robots());
  }
  if (hasObservation) {
    const Model& model = pinocchioInterface_.getModel();
    Data& data = pinocchioInterface_.getData();
    pinocchio::forwardKinematics(
        model, data,
        robotModel_->getGeneralizedCoordinates(inputs.observation->state));
    pinocchio::updateFramePlacements(model, data);
    if (inputs.robotState == nullptr) {
      writePoseLinksRtWorld(data, mpcLinkFrames_, mpcLinkNames_,
                            scene::kMeasuredInstance, scene->add_robots());
    }
    writeObservationMarkers(*inputs.observation,
                            hasPolicy ? inputs.policy : nullptr, scene);
  } else {
    for (const absl::string_view path :
         {scene::kContactForceMarkers, scene::kCornerForceMarkers}) {
      scene->add_arrows()->set_path(std::string(path));
    }
    for (const absl::string_view path :
         {scene::kCenterOfPressure, scene::kCollisionSpheres}) {
      scene->add_spheres()->set_path(std::string(path));
    }
  }

  // The plan is computed after the observation's markers: it moves the frame
  // placements to its own nodes.
  if (!hasPolicy) {
    clearPlan();
  } else if (!hasPlan_ || inputs.policyVersion != planVersion_) {
    computePlan(*inputs.policy);
    planVersion_ = inputs.policyVersion;
  }
  if (hasTerminalState_) {
    *scene->add_robots() = terminalState_;
  }
  if (hasTerminalTarget_) {
    *scene->add_robots() = terminalTarget_;
  }
  const std::array<std::pair<absl::string_view,
                             const humanoid_mpc_msgs::LineStrips* absl_nonnull>,
                   kPlanLineStrips>
      lineStrips = {{
          {scene::kPlanEndEffectors, &planEndEffectors_},
          {scene::kPlanBase, &planBase_},
          {scene::kPlanCom, &planCom_},
      }};
  for (const std::pair<absl::string_view,
                       const humanoid_mpc_msgs::LineStrips* absl_nonnull>&
           strips : lineStrips) {
    humanoid_mpc_msgs::LineStrips* absl_nonnull message =
        scene->add_line_strips();
    *message = *strips.second;
    message->set_path(std::string(strips.first));
  }
  humanoid_mpc_msgs::Spheres* absl_nonnull footholds = scene->add_spheres();
  *footholds = planFootholds_;
  footholds->set_path(std::string(scene::kPlanFootholds));
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::visualization
