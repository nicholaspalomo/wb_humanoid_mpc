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
#include <exception>
#include <memory>
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
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/joint/joint-free-flyer.hpp"
#include "pinocchio/parsers/urdf.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/costs/CollisionConstraintFromConfig.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/constraint/FootCollisionConstraint.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/PinocchioFrameConversions.h"
#include "humanoid_common_mpc_app/visualization/SceneContract.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_msgs/arrows.pb.h"
#include "humanoid_mpc_msgs/color.pb.h"
#include "humanoid_mpc_msgs/pose.pb.h"
#include "humanoid_mpc_msgs/vector3.pb.h"

namespace ocs2::humanoid::visualization {

namespace {

using Model = PinocchioInterface::Model;
using Data = PinocchioInterface::Data;

void setVector3(const vector3_t& value, humanoid_mpc_msgs::Vector3* absl_nonnull message) {
  message->set_x(value.x());
  message->set_y(value.y());
  message->set_z(value.z());
}

void setColor(const scene::Rgba& color, humanoid_mpc_msgs::Color* absl_nonnull message) {
  message->set_r(color.r);
  message->set_g(color.g);
  message->set_b(color.b);
  message->set_a(color.a);
}

/** An arrow of `force` ending at `tip`: kForceScale newtons per meter. */
void addForceArrow(const vector3_t& tip, const vector3_t& force, humanoid_mpc_msgs::Arrows* absl_nonnull arrows) {
  const vector3_t vector = force / scene::kForceScale;
  setVector3(tip - vector, arrows->add_origins());
  setVector3(vector, arrows->add_vectors());
}

/** The BODY frames of `model` (one per URDF link), without the universe. */
void collectLinkFrames(const Model& model,
                       std::vector<pinocchio::FrameIndex>* absl_nonnull frames,
                       std::vector<std::string>* absl_nonnull names) {
  frames->clear();
  names->clear();
  for (size_t frame = 0; frame < model.frames.size(); ++frame) {
    if (model.frames[frame].type == pinocchio::BODY && model.frames[frame].name != "universe") {
      frames->push_back(static_cast<pinocchio::FrameIndex>(frame));
      names->push_back(model.frames[frame].name);
    }
  }
}

/** The world pose of every link of `frames` (frame placements must be up to date) as the instance `name`. */
void writeLinkPoses(const Data& data,
                    const std::vector<pinocchio::FrameIndex>& frames,
                    const std::vector<std::string>& names,
                    absl::string_view name,
                    humanoid_mpc_msgs::RobotModelInstance* absl_nonnull instance) {
  instance->Clear();
  instance->set_name(std::string(name));
  for (size_t link = 0; link < frames.size(); ++link) {
    instance->add_link_names(names[link]);
    const pinocchio::SE3Tpl<scalar_t>& placement = data.oMf[frames[link]];
    humanoid_mpc_msgs::Pose* absl_nonnull pose = instance->add_link_poses();
    setVector3(placement.translation(), pose->mutable_position());
    const quaternion_t orientation(placement.rotation());
    humanoid_mpc_msgs::Quaternion* absl_nonnull quaternion = pose->mutable_orientation();
    quaternion->set_w(orientation.w());
    quaternion->set_x(orientation.x());
    quaternion->set_y(orientation.y());
    quaternion->set_z(orientation.z());
  }
}

absl::StatusOr<pinocchio::FrameIndex> findFrame(const Model& model, const std::string& frame, absl::string_view what) {
  if (!model.existFrame(frame)) {
    return absl::InvalidArgumentError(absl::StrCat(what, " names the frame '", frame, "', which the MPC's robot model does not have."));
  }
  return model.getFrameId(frame);
}

}  // namespace

SceneBuilder::SceneBuilder(const VisualizationModel& model, const VisualizationConfig& /*config*/)
    : pinocchioInterface_(*model.pinocchioInterface),
      robotModel_(model.mpcRobotModel->clone()),
      modelSettings_(robotModel_->modelSettings),
      groundHeight_(robotModel_->modelSettings.terrainHeight) {}

absl::StatusOr<std::unique_ptr<SceneBuilder>> SceneBuilder::Create(const VisualizationModel& model, const VisualizationConfig& config) {
  RETURN_IF_ERROR(checkVisualizationModel(model));
  std::unique_ptr<SceneBuilder> builder = absl::WrapUnique(new SceneBuilder(model, config));
  RETURN_IF_ERROR(builder->initialize(model, config));
  return builder;
}

absl::Status SceneBuilder::initialize(const VisualizationModel& model, const VisualizationConfig& config) {
  const Model& mpcModel = pinocchioInterface_.getModel();
  collectLinkFrames(mpcModel, &mpcLinkFrames_, &mpcLinkNames_);

  try {  // NOLINT(exceptions): Pinocchio's URDF parser throws; its exception becomes a Status here.
    pinocchio::urdf::buildModel(model.urdfFile, pinocchio::JointModelFreeFlyerTpl<scalar_t>(), fullModel_);
  } catch (const std::exception& e) {  // NOLINT(exceptions): the boundary of the try above.
    return absl::InvalidArgumentError(absl::StrCat("the URDF '", model.urdfFile, "' does not give a Pinocchio model: ", e.what()));
  }
  fullData_ = std::make_unique<Data>(fullModel_);
  fullConfiguration_ = pinocchio::neutral(fullModel_);
  collectLinkFrames(fullModel_, &fullLinkFrames_, &fullLinkNames_);
  // Joint 0 is the universe and joint 1 the free-flyer root.
  for (pinocchio::JointIndex joint = 2; joint < static_cast<pinocchio::JointIndex>(fullModel_.njoints); ++joint) {
    const int nq = fullModel_.joints[joint].nq();
    if (nq == 1 || nq == 2) {
      fullJoints_.push_back(FullModelJoint{.name = fullModel_.names[joint], .idxQ = fullModel_.joints[joint].idx_q(), .nq = nq});
    }
  }

  if (modelSettings_.contactNames.size() < kNumContacts) {
    return absl::InvalidArgumentError(absl::StrCat("the model settings name ", modelSettings_.contactNames.size(),
                                                   " contacts; the visualization needs ", kNumContacts, "."));
  }
  // The contact polygons and the collision spheres of the typed task file.
  const absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(model.taskFile);
  if (!task.ok()) {
    return absl::InvalidArgumentError(absl::StrCat("the contact polygons do not load: ", task.status().message()));
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    ASSIGN_OR_RETURN(contactFrames_[contact], findFrame(mpcModel, modelSettings_.contactNames[contact], "contactNames"));
    const absl::StatusOr<ContactRectangle> rectangle =
        contactRectangleFromConfig(task->contacts, modelSettings_, static_cast<int>(contact));
    if (!rectangle.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(model.taskFile, ": the contact polygon of ", modelSettings_.contactNames[contact],
                                                     " does not load: ", rectangle.status().message()));
    }
    if (rectangle->getNumberOfContactPoints() != 4) {
      return absl::InvalidArgumentError(absl::StrCat("the contact polygon of ", modelSettings_.contactNames[contact], " has ",
                                                     rectangle->getNumberOfContactPoints(), " corners; the corner forces need 4."));
    }
    for (size_t corner = 0; corner < 4; ++corner) {
      ASSIGN_OR_RETURN(cornerFrames_[contact][corner],
                       findFrame(mpcModel, rectangle->getPolygonPointFrameName(static_cast<int>(corner)), "the contact polygon"));
    }
    cornerForceMappers_.emplace_back(*rectangle);
  }

  for (const std::string& frame : config.planFrames) {
    ASSIGN_OR_RETURN(const pinocchio::FrameIndex index, findFrame(mpcModel, frame, kRerunPlanFramesField));
    planFrames_.push_back(index);
  }

  const absl::StatusOr<FootCollisionConstraint::Config> loaded = footCollisionConstraintConfigFromConfig(task->collision_constraint);
  if (!loaded.ok()) {
    LOG(WARNING) << "[SceneBuilder] " << model.taskFile
                 << ": the collision spheres do not load, none is drawn: " << loaded.status().message();
    return absl::OkStatus();
  }
  const FootCollisionConstraint::Config& collision = *loaded;
  const std::array<std::pair<const std::string* absl_nonnull, scalar_t>, 10> candidates = {{
      {&collision.leftAnkleFrame, collision.footCollisionSphereRadius},
      {&collision.rightAnkleFrame, collision.footCollisionSphereRadius},
      {&collision.leftFootCenterFrame, collision.footCollisionSphereRadius},
      {&collision.rightFootCenterFrame, collision.footCollisionSphereRadius},
      {&collision.leftFootFrame1, collision.footCollisionSphereRadius},
      {&collision.rightFootFrame1, collision.footCollisionSphereRadius},
      {&collision.leftFootFrame2, collision.footCollisionSphereRadius},
      {&collision.rightFootFrame2, collision.footCollisionSphereRadius},
      {&collision.leftKneeFrame, collision.kneeCollisionSphereRadius},
      {&collision.rightKneeFrame, collision.kneeCollisionSphereRadius},
  }};
  // As HumanoidVisualizer drew them: a frame the task file leaves unnamed, or that the model lacks, has no sphere.
  for (const std::pair<const std::string* absl_nonnull, scalar_t>& candidate : candidates) {
    if (!candidate.first->empty() && mpcModel.existFrame(*candidate.first) && std::isfinite(candidate.second) && candidate.second > 0.0) {
      collisionSpheres_.push_back(CollisionSphere{.frame = mpcModel.getFrameId(*candidate.first), .radius = candidate.second});
    }
  }
  return absl::OkStatus();
}

void SceneBuilder::writeMeasuredFromSample(const msgs::RobotStateSample& sample,
                                           humanoid_mpc_msgs::RobotModelInstance* absl_nonnull instance) {
  if (!hasSampleJointIndices_ || sample.joint_names != sampleJointNames_) {
    absl::flat_hash_map<std::string, int> sampleIndices;
    for (size_t index = 0; index < sample.joint_names.size(); ++index) {
      sampleIndices.emplace(sample.joint_names[index], static_cast<int>(index));
    }
    for (FullModelJoint& joint : fullJoints_) {
      const absl::flat_hash_map<std::string, int>::const_iterator found = sampleIndices.find(joint.name);
      joint.sampleIndex = found != sampleIndices.end() ? found->second : -1;
    }
    sampleJointNames_ = sample.joint_names;
    hasSampleJointIndices_ = true;
  }

  // The free flyer's configuration is the position and the quaternion (x, y, z, w).
  quaternion_t orientation(sample.base_orientation_world.w, sample.base_orientation_world.x, sample.base_orientation_world.y,
                           sample.base_orientation_world.z);
  if (orientation.norm() > 0.0) {
    orientation.normalize();
  } else {
    orientation = quaternion_t::Identity();
  }
  fullConfiguration_.head<3>() = vector3_t(sample.base_position_world.x, sample.base_position_world.y, sample.base_position_world.z);
  fullConfiguration_.segment<4>(3) = orientation.coeffs();
  for (const FullModelJoint& joint : fullJoints_) {
    if (joint.sampleIndex < 0 || joint.sampleIndex >= sample.joint_positions.size()) {
      continue;
    }
    const scalar_t position = sample.joint_positions[joint.sampleIndex];
    if (joint.nq == 1) {
      fullConfiguration_[joint.idxQ] = position;
    } else {
      // An unbounded revolute joint: (cos, sin) of its angle.
      fullConfiguration_[joint.idxQ] = std::cos(position);
      fullConfiguration_[joint.idxQ + 1] = std::sin(position);
    }
  }
  pinocchio::forwardKinematics(fullModel_, *fullData_, fullConfiguration_);
  pinocchio::updateFramePlacements(fullModel_, *fullData_);
  writeLinkPoses(*fullData_, fullLinkFrames_, fullLinkNames_, scene::kMeasuredInstance, instance);
}

void SceneBuilder::updateGroundHeight(size_t mode) {
  // HumanoidVisualizer's estimate (getGroundHeightEstimate()), without its function-static state: the stance feet's
  // mean height, held while no foot is in contact.
  const contact_flag_t stance = modeNumber2StanceLeg(mode);
  const Data& data = pinocchioInterface_.getData();
  scalar_t heightSum = 0.0;
  size_t stanceFeet = 0;
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    if (stance[contact]) {
      heightSum += data.oMf[contactFrames_[contact]].translation().z();
      ++stanceFeet;
    }
  }
  if (stanceFeet > 0) {
    groundHeight_ = heightSum / static_cast<scalar_t>(stanceFeet);
  }
}

void SceneBuilder::writeObservationMarkers(const SystemObservation& observation,
                                           const PolicySnapshot* absl_nullable policy,
                                           humanoid_mpc_msgs::VisualizationScene* absl_nonnull scene) {
  humanoid_mpc_msgs::Arrows* absl_nonnull contactForces = scene->add_arrows();
  contactForces->set_path(std::string(scene::kContactForces));
  humanoid_mpc_msgs::Spheres* absl_nonnull centerOfPressure = scene->add_spheres();
  centerOfPressure->set_path(std::string(scene::kCenterOfPressure));
  humanoid_mpc_msgs::Arrows* absl_nonnull cornerForces = scene->add_arrows();
  cornerForces->set_path(std::string(scene::kCornerForces));
  humanoid_mpc_msgs::Spheres* absl_nonnull collisionSpheres = scene->add_spheres();
  collisionSpheres->set_path(std::string(scene::kCollisionSpheres));

  // The frame placements are those of the observation's state (build() updated them).
  const Data& data = pinocchioInterface_.getData();
  for (const CollisionSphere& sphere : collisionSpheres_) {
    setVector3(data.oMf[sphere.frame].translation(), collisionSpheres->add_centers());
    collisionSpheres->add_radii(static_cast<float>(sphere.radius));
  }

  if (policy == nullptr) {
    return;
  }
  vector_t planState;
  vector_t planInput;
  samplePlan(*policy, observation.time, &planState, &planInput);
  const contact_flag_t stance = modeNumber2StanceLeg(observation.mode);
  vector3_t weightedCop = vector3_t::Zero();
  scalar_t normalForceSum = 0.0;
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    if (!stance[contact]) {
      continue;
    }
    const pinocchio::FrameIndex frame = contactFrames_[contact];
    const vector6_t worldWrench = robotModel_->getContactWrenchInWorldFrame(observation.state, planInput, contact);
    const vector6_t localWrench = rotateVectorWorldToLocal<scalar_t>(worldWrench, data, frame);
    vector3_t cop = data.oMf[frame].translation();
    if (localWrench[kWrenchForceZIndex] > kMinNormalForceForCop) {
      const vector3_t localCop(-localWrench[kWrenchTorqueYIndex] / localWrench[kWrenchForceZIndex],
                               localWrench[kWrenchTorqueXIndex] / localWrench[kWrenchForceZIndex], 0.0);
      cop = data.oMf[frame].act(localCop);
      weightedCop += worldWrench[kWrenchForceZIndex] * cop;
      normalForceSum += worldWrench[kWrenchForceZIndex];
    }
    addForceArrow(cop, worldWrench.head<3>(), contactForces);

    const std::array<vector3_t, 4> cornerForcesLocal = cornerForceMappers_[contact].computeVisualizationForceArray(localWrench);
    for (size_t corner = 0; corner < 4; ++corner) {
      addForceArrow(data.oMf[cornerFrames_[contact][corner]].translation(),
                    rotateVectorLocalToWorld<scalar_t>(cornerForcesLocal[corner], data, frame), cornerForces);
    }
  }
  if (normalForceSum > kMinNormalForceForCop) {
    setVector3(weightedCop / normalForceSum, centerOfPressure->add_centers());
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
    // centerOfMass() runs the forward kinematics the frame placements below are updated from.
    vector3_t com = pinocchio::centerOfMass(model, data, q, /*computeSubtreeComs=*/false);
    com.z() = groundHeight_;
    setVector3(com, comStrip->add_points());
    setVector3(robotModel_->getBasePosition(state), baseStrip->add_points());
    for (size_t frame = 0; frame < planFrames_.size(); ++frame) {
      setVector3(pinocchio::updateFramePlacement(model, data, planFrames_[frame]).translation(),
                 planEndEffectors_.mutable_strips(static_cast<int>(frame))->add_points());
    }
  }

  // A foot lands at an event inside the horizon where it goes from swing to stance.
  const scalar_t startTime = policy.time.front();
  const scalar_t finalTime = policy.time.back();
  const ModeSchedule& schedule = policy.modeSchedule;
  vector_t eventState;
  vector_t eventInput;
  for (size_t event = 0; event < schedule.eventTimes.size() && event + 1 < schedule.modeSequence.size(); ++event) {
    const scalar_t eventTime = schedule.eventTimes[event];
    if (!(startTime < eventTime && eventTime < finalTime)) {
      continue;
    }
    const contact_flag_t before = modeNumber2StanceLeg(schedule.modeSequence[event]);
    const contact_flag_t after = modeNumber2StanceLeg(schedule.modeSequence[event + 1]);
    bool kinematicsDone = false;
    for (size_t contact = 0; contact < kNumContacts; ++contact) {
      if (before[contact] || !after[contact]) {
        continue;
      }
      if (!kinematicsDone) {
        samplePlan(policy, eventTime, &eventState, &eventInput);
        pinocchio::forwardKinematics(model, data, robotModel_->getGeneralizedCoordinates(eventState));
        kinematicsDone = true;
      }
      setVector3(pinocchio::updateFramePlacement(model, data, contactFrames_[contact]).translation(), planFootholds_.add_centers());
      setColor(scene::kContactColors[contact % scene::kContactColors.size()], planFootholds_.add_colors());
    }
  }

  pinocchio::forwardKinematics(model, data, robotModel_->getGeneralizedCoordinates(policy.state.back()));
  pinocchio::updateFramePlacements(model, data);
  writeLinkPoses(data, mpcLinkFrames_, mpcLinkNames_, scene::kTerminalStateInstance, &terminalState_);
  hasTerminalState_ = true;

  vector_t targetState;
  vector_t targetInput;
  if (sampleTarget(policy.target, finalTime, robotModel_->getStateDim(), robotModel_->getInputDim(), &targetState, &targetInput) !=
      TargetSample::kNone) {
    pinocchio::forwardKinematics(model, data, robotModel_->getGeneralizedCoordinates(targetState));
    pinocchio::updateFramePlacements(model, data);
    writeLinkPoses(data, mpcLinkFrames_, mpcLinkNames_, scene::kTerminalTargetInstance, &terminalTarget_);
    hasTerminalTarget_ = true;
  }
  hasPlan_ = true;
}

absl::Status SceneBuilder::build(const SceneInputs& inputs, humanoid_mpc_msgs::VisualizationScene* absl_nonnull scene) {
  scene->Clear();
  const bool hasObservation =
      inputs.observation != nullptr && static_cast<size_t>(inputs.observation->state.size()) == robotModel_->getStateDim();
  const bool hasPolicy =
      inputs.policy != nullptr && isConsistentPlan(*inputs.policy, robotModel_->getStateDim(), robotModel_->getInputDim());
  if (!hasObservation && inputs.robotState == nullptr) {
    return absl::FailedPreconditionError("there is neither an observation nor a robot/state sample to draw.");
  }
  scene->set_time(inputs.robotState != nullptr ? inputs.robotState->time : inputs.observation->time);

  if (inputs.robotState != nullptr) {
    writeMeasuredFromSample(*inputs.robotState, scene->add_robots());
  }
  if (hasObservation) {
    const Model& model = pinocchioInterface_.getModel();
    Data& data = pinocchioInterface_.getData();
    pinocchio::forwardKinematics(model, data, robotModel_->getGeneralizedCoordinates(inputs.observation->state));
    pinocchio::updateFramePlacements(model, data);
    updateGroundHeight(inputs.observation->mode);
    if (inputs.robotState == nullptr) {
      writeLinkPoses(data, mpcLinkFrames_, mpcLinkNames_, scene::kMeasuredInstance, scene->add_robots());
    }
    writeObservationMarkers(*inputs.observation, hasPolicy ? inputs.policy : nullptr, scene);
  } else {
    for (const absl::string_view path : {scene::kContactForces, scene::kCornerForces}) {
      scene->add_arrows()->set_path(std::string(path));
    }
    for (const absl::string_view path : {scene::kCenterOfPressure, scene::kCollisionSpheres}) {
      scene->add_spheres()->set_path(std::string(path));
    }
  }

  // The plan is computed after the observation's markers: it moves the frame placements to its own nodes.
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
  const std::array<std::pair<absl::string_view, const humanoid_mpc_msgs::LineStrips* absl_nonnull>, 3> lineStrips = {{
      {scene::kPlanEndEffectors, &planEndEffectors_},
      {scene::kPlanBase, &planBase_},
      {scene::kPlanCom, &planCom_},
  }};
  for (const std::pair<absl::string_view, const humanoid_mpc_msgs::LineStrips* absl_nonnull>& strips : lineStrips) {
    humanoid_mpc_msgs::LineStrips* absl_nonnull message = scene->add_line_strips();
    *message = *strips.second;
    message->set_path(std::string(strips.first));
  }
  humanoid_mpc_msgs::Spheres* absl_nonnull footholds = scene->add_spheres();
  *footholds = planFootholds_;
  footholds->set_path(std::string(scene::kPlanFootholds));
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::visualization
