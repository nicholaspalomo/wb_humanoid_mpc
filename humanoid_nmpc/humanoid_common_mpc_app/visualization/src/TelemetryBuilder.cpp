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
#include <pinocchio/fwd.hpp>

#include "humanoid_common_mpc_app/visualization/TelemetryBuilder.h"

#include <algorithm>
#include <string>
#include <vector>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc_app/visualization/EulerAngles.h"
#include "humanoid_mpc_msgs/scalar_group.pb.h"

namespace ocs2::humanoid::visualization {

namespace {

using Model = PinocchioInterface::Model;
using Data = PinocchioInterface::Data;
using Motion = pinocchio::MotionTpl<scalar_t>;

// The sources of sources_, in the order of the contract's frame groups.
constexpr size_t kMeasured = 0;
constexpr size_t kReference = 1;
constexpr size_t kPlan = 2;

// The kinds of the frame groups, in the contract's order.
constexpr size_t kPose = 0;
constexpr size_t kTwist = 1;
constexpr size_t kAcceleration = 2;
constexpr size_t kWrench = 3;

// joints_, in the contract's order.
constexpr size_t kJointPositionMeasured = 0;
constexpr size_t kJointPositionTarget = 1;
constexpr size_t kJointVelocityMeasured = 2;
constexpr size_t kJointVelocityTarget = 3;
constexpr size_t kJointEffortMeasured = 4;
constexpr size_t kJointEffortTarget = 5;

// The generalized coordinates the "Generalized Coordinates (Pinocchio)" panels plot: base_z, base_pitch, base_roll.
constexpr std::array<size_t, 3> kGeneralizedBaseDofs = {BASE_POS_Z_INDEX, BASE_TRANSLATION_DIM + BASE_ROT_PITCH_INDEX,
                                                        BASE_TRANSLATION_DIM + BASE_ROT_ROLL_INDEX};

/** "<prefix>0", "<prefix>1", ...: one name per component of a vector of `size`. */
std::vector<std::string> indexedNames(absl::string_view prefix, size_t size) {
  std::vector<std::string> names;
  names.reserve(size);
  for (size_t index = 0; index < size; ++index) {
    names.push_back(absl::StrCat(prefix, index));
  }
  return names;
}

}  // namespace

TelemetryBuilder::TelemetryBuilder(const VisualizationModel& model, const VisualizationConfig& /*config*/)
    : pinocchioInterface_(*model.pinocchioInterface),
      robotModel_(model.mpcRobotModel->clone()),
      modelSettings_(robotModel_->modelSettings) {
  for (Source& source : sources_) {
    source.data = std::make_unique<Data>(pinocchioInterface_.getModel());
  }
}

absl::StatusOr<std::unique_ptr<TelemetryBuilder>> TelemetryBuilder::Create(const VisualizationModel& model,
                                                                           const VisualizationConfig& config) {
  RETURN_IF_ERROR(checkVisualizationModel(model));
  std::unique_ptr<TelemetryBuilder> builder(new TelemetryBuilder(model, config));
  RETURN_IF_ERROR(builder->initialize(config));
  return builder;
}

int TelemetryBuilder::addGroup(const std::string& path, const std::vector<std::string>& names) {
  humanoid_mpc_msgs::ScalarGroup* group = series_.add_groups();
  group->set_path(path);
  for (const std::string& name : names) {
    group->add_names(name);
  }
  group->mutable_values()->Resize(static_cast<int>(names.size()), /*value=*/0.0);
  groupValues_.push_back(group->mutable_values());
  return static_cast<int>(groupValues_.size()) - 1;
}

void TelemetryBuilder::setValues(int group, const vector_t& vector) {
  // The layout is fixed at construction; a vector of another size (which the model's dimensions rule out) neither
  // overruns the group nor leaves stale values in it.
  google::protobuf::RepeatedField<double>* groupValues = groupValues_[static_cast<size_t>(group)];
  const Eigen::Index size = std::min<Eigen::Index>(vector.size(), groupValues->size());
  std::copy(vector.data(), vector.data() + size, groupValues->mutable_data());
  std::fill(groupValues->mutable_data() + size, groupValues->mutable_data() + groupValues->size(), 0.0);
}

absl::Status TelemetryBuilder::initialize(const VisualizationConfig& config) {
  const Model& model = pinocchioInterface_.getModel();
  if (modelSettings_.contactNames.size() < N_CONTACTS) {
    return absl::InvalidArgumentError(
        absl::StrCat("the model settings name ", modelSettings_.contactNames.size(), " contacts; the telemetry needs ", N_CONTACTS, "."));
  }
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    if (!model.existFrame(modelSettings_.contactNames[contact])) {
      return absl::InvalidArgumentError(
          absl::StrCat("the contact frame '", modelSettings_.contactNames[contact], "' is not a frame of the MPC's robot model."));
    }
    contactFrames_[contact] = model.getFrameId(modelSettings_.contactNames[contact]);
  }

  // The groups of the telemetry contract, in its order: the panel groups, then the complete groups.
  // LINT.IfChange(telemetry_groups)
  const std::vector<std::string> measuredReference = {"measured", "reference"};
  const std::vector<std::string> mpcMeasured = {"mpc", "measured"};
  const std::vector<std::string> tangential = {"mpc_x", "measured_x", "mpc_y", "measured_y"};
  const std::vector<std::string> poseNames = {"x", "y", "z", "roll", "pitch", "yaw"};
  const std::vector<std::string> twistNames = {"linear_x", "linear_y", "linear_z", "angular_x", "angular_y", "angular_z"};
  const std::vector<std::string> wrenchNames = {"force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"};
  const std::vector<std::string> forceNames = {"force_x", "force_y", "force_z"};
  const std::array<const char*, 3> axes = {"x", "y", "z"};
  const std::array<const char*, 3> angles = {"roll", "pitch", "yaw"};
  const std::array<const char*, N_CONTACTS> sides = {"left", "right"};
  const std::array<const char*, 3> sources = {"measured", "reference", "plan"};
  const std::array<const char*, 4> frameKinds = {"pose", "twist", "acceleration", "wrench"};

  for (size_t axis = 0; axis < 3; ++axis) {
    basePosition_[axis] = addGroup(absl::StrCat("base_pose/position_", axes[axis]), measuredReference);
  }
  for (size_t angle = 0; angle < 3; ++angle) {
    baseRollPitchYaw_[angle] = addGroup(absl::StrCat("base_pose/", angles[angle]), measuredReference);
  }
  for (size_t axis = 0; axis < 3; ++axis) {
    baseLinearVelocity_[axis] = addGroup(absl::StrCat("base_twist/linear_", axes[axis]), measuredReference);
  }
  for (size_t axis = 0; axis < 3; ++axis) {
    baseAngularVelocity_[axis] = addGroup(absl::StrCat("base_twist/angular_", axes[axis]), measuredReference);
  }
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    normalForce_[contact] = addGroup(absl::StrCat("contact_forces/", sides[contact], "_normal"), mpcMeasured);
  }
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    tangentialForce_[contact] = addGroup(absl::StrCat("contact_forces/", sides[contact], "_tangential"), tangential);
  }
  generalizedBaseCoordinate_ = {addGroup("generalized_base/position_z", measuredReference),
                                addGroup("generalized_base/pitch", measuredReference),
                                addGroup("generalized_base/roll", measuredReference)};
  generalizedBaseVelocity_ = {addGroup("generalized_base/velocity_z", measuredReference),
                              addGroup("generalized_base/pitch_rate", measuredReference),
                              addGroup("generalized_base/roll_rate", measuredReference)};
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    footAcceleration_[contact] = addGroup(absl::StrCat("foot_kinematics/", sides[contact], "_acceleration_z"), measuredReference);
  }
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    footVelocity_[contact] = addGroup(absl::StrCat("foot_kinematics/", sides[contact], "_velocity_z"), measuredReference);
  }

  const std::vector<std::string>& jointNames = modelSettings_.fullJointNames;
  joints_ = {addGroup("joints/position/measured", jointNames), addGroup("joints/position/target", jointNames),
             addGroup("joints/velocity/measured", jointNames), addGroup("joints/velocity/target", jointNames),
             addGroup("joints/effort/measured", jointNames),   addGroup("joints/effort/target", jointNames)};
  std::vector<std::string> dofNames = getBaseDofNames();
  dofNames.insert(dofNames.end(), modelSettings_.mpcModelJointNames.begin(), modelSettings_.mpcModelJointNames.end());
  for (size_t source = 0; source < 3; ++source) {
    dofPositions_[source] = addGroup(absl::StrCat("dofs/position/", sources[source]), dofNames);
  }
  for (size_t source = 0; source < 3; ++source) {
    dofVelocities_[source] = addGroup(absl::StrCat("dofs/velocity/", sources[source]), dofNames);
  }
  for (size_t source = 0; source < 2; ++source) {
    dofForces_[source] = addGroup(absl::StrCat("dofs/force/", sources[source]), dofNames);
  }
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    contactWrenchMpc_[contact] = addGroup(absl::StrCat("contact_wrenches/", sides[contact], "/mpc"), wrenchNames);
    contactWrenchMeasured_[contact] = addGroup(absl::StrCat("contact_wrenches/", sides[contact], "/measured"), forceNames);
  }
  observationState_ = addGroup("mpc_observation/state", indexedNames("x", robotModel_->getStateDim()));
  observationInput_ = addGroup("mpc_observation/input", indexedNames("u", robotModel_->getInputDim()));
  observationMode_ = addGroup("mpc_observation/mode", {"mode"});

  for (const std::string& frame : config.telemetryFrames) {
    if (!model.existFrame(frame)) {
      return absl::InvalidArgumentError(
          absl::StrCat(kTelemetryFramesKey, " names the frame '", frame, "', which the MPC's robot model does not have."));
    }
    FrameGroups groups{model.getFrameId(frame), -1, {}};
    for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
      if (modelSettings_.contactNames[contact] == frame) {
        groups.contact = static_cast<int>(contact);
      }
    }
    const std::array<const std::vector<std::string>*, 4> kindNames = {&poseNames, &twistNames, &twistNames, &wrenchNames};
    for (size_t kind = 0; kind < frameKinds.size(); ++kind) {
      for (size_t source = 0; source < sources.size(); ++source) {
        groups.groups[kind][source] =
            addGroup(absl::StrCat("frames/", frameKinds[kind], "/", frame, "/", sources[source]), *kindNames[kind]);
      }
    }
    frames_.push_back(groups);
  }
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/telemetry_contract.py:panel_groups, //humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/telemetry_contract.py:complete_groups)
  // clang-format on

  const size_t dofs = robotModel_->getGenCoordinatesDim();
  for (Source& source : sources_) {
    source.q.setZero(dofs);
    source.v.setZero(dofs);
    source.a.setZero(dofs);
  }
  return absl::OkStatus();
}

std::vector<std::string> TelemetryBuilder::groupPaths() const {
  std::vector<std::string> paths;
  paths.reserve(static_cast<size_t>(series_.groups_size()));
  for (const humanoid_mpc_msgs::ScalarGroup& group : series_.groups()) {
    paths.push_back(group.path());
  }
  return paths;
}

vector_t TelemetryBuilder::generalizedVelocities(const vector_t& state, const vector_t& input) {
  return robotModel_->getGeneralizedVelocities(state, input);
}

void TelemetryBuilder::updateKinematics(Source* source) {
  const Model& model = pinocchioInterface_.getModel();
  pinocchio::forwardKinematics(model, *source->data, source->q, source->v, source->a);
  pinocchio::updateFramePlacements(model, *source->data);
}

void TelemetryBuilder::computeReference(const DecodedRobotState& measured, const PolicySnapshot* policy) {
  Source& reference = sources_[kReference];
  const size_t stateDim = robotModel_->getStateDim();
  const size_t inputDim = robotModel_->getInputDim();
  const scalar_t time = measured.time;
  reference.contactWrenches = makeFeetArray<vector6_t>(vector6_t::Zero());
  reference.v.setZero();
  reference.a.setZero();

  vector_t state;
  vector_t input;
  const TargetSample sample =
      policy != nullptr ? sampleTarget(policy->target, time, stateDim, inputDim, &state, &input) : TargetSample::kNone;
  if (sample == TargetSample::kNone) {
    // No reference: the measured robot at rest, as PinocchioTelemetryPublisher fell back.
    reference.q = measured.generalizedCoordinates;
    referenceBasePosition_ = measured.basePosition;
    referenceRollPitchYaw_ = vector3_t(measured.baseEulerAnglesZyx(2), measured.baseEulerAnglesZyx(1), measured.baseEulerAnglesZyx(0));
    referenceBaseLinearVelocity_.setZero();
    return;
  }
  reference.q = robotModel_->getGeneralizedCoordinates(state);
  referenceBasePosition_ = robotModel_->getBasePosition(state);
  const vector3_t eulerAnglesZyx = robotModel_->getBaseOrientationEulerZYX(state);
  referenceRollPitchYaw_ = vector3_t(eulerAnglesZyx(2), eulerAnglesZyx(1), eulerAnglesZyx(0));
  referenceBaseLinearVelocity_ = robotModel_->getBaseComLinearVelocity(state);
  if (sample != TargetSample::kStateAndInput) {
    return;
  }
  reference.v = generalizedVelocities(state, input);
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    reference.contactWrenches[contact] = robotModel_->getContactWrenchInWorldFrame(state, input, contact);
  }
  const scalar_array_t& times = policy->target.timeTrajectory;
  const scalar_t earlier = std::max(time - kAccelerationHalfInterval, times.front());
  const scalar_t later = std::min(time + kAccelerationHalfInterval, times.back());
  if (later > earlier) {
    vector_t earlierState;
    vector_t earlierInput;
    vector_t laterState;
    vector_t laterInput;
    sampleTarget(policy->target, earlier, stateDim, inputDim, &earlierState, &earlierInput);
    sampleTarget(policy->target, later, stateDim, inputDim, &laterState, &laterInput);
    reference.a = (generalizedVelocities(laterState, laterInput) - generalizedVelocities(earlierState, earlierInput)) / (later - earlier);
  }
}

void TelemetryBuilder::computePlan(const DecodedRobotState& measured, const PolicySnapshot* policy) {
  Source& plan = sources_[kPlan];
  plan.contactWrenches = makeFeetArray<vector6_t>(vector6_t::Zero());
  plan.v.setZero();
  plan.a.setZero();
  if (policy == nullptr) {
    plan.q = measured.generalizedCoordinates;
    return;
  }
  const scalar_t time = measured.time;
  vector_t state;
  vector_t input;
  samplePlan(*policy, time, &state, &input);
  plan.q = robotModel_->getGeneralizedCoordinates(state);
  plan.v = generalizedVelocities(state, input);
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    plan.contactWrenches[contact] = robotModel_->getContactWrenchInWorldFrame(state, input, contact);
  }
  const scalar_t earlier = std::max(time - kAccelerationHalfInterval, policy->time.front());
  const scalar_t later = std::min(time + kAccelerationHalfInterval, policy->time.back());
  if (later > earlier) {
    vector_t earlierState;
    vector_t earlierInput;
    vector_t laterState;
    vector_t laterInput;
    samplePlan(*policy, earlier, &earlierState, &earlierInput);
    samplePlan(*policy, later, &laterState, &laterInput);
    plan.a = (generalizedVelocities(laterState, laterInput) - generalizedVelocities(earlierState, earlierInput)) / (later - earlier);
  }
}

const humanoid_mpc_msgs::TelemetrySeries& TelemetryBuilder::build(const DecodedRobotState& measured,
                                                                  const SystemObservation* observation,
                                                                  const PolicySnapshot* policy) {
  const Model& model = pinocchioInterface_.getModel();
  const size_t stateDim = robotModel_->getStateDim();
  const size_t inputDim = robotModel_->getInputDim();
  const bool hasObservation = observation != nullptr && static_cast<size_t>(observation->state.size()) == stateDim;
  const PolicySnapshot* plan = policy != nullptr && isConsistentPlan(*policy, stateDim, inputDim) ? policy : nullptr;
  const scalar_t time = measured.time;
  series_.set_time(time);

  Source& measuredSource = sources_[kMeasured];
  measuredSource.q = measured.generalizedCoordinates;
  measuredSource.v = measured.generalizedVelocities;
  measuredSource.a.setZero(measuredSource.v.size());
  if (hasPreviousSample_ && previousVelocities_.size() == measuredSource.v.size()) {
    const scalar_t interval = time - previousTime_;
    if (interval > kMinDifferenceInterval && interval < kMaxDifferenceInterval) {
      measuredSource.a = (measuredSource.v - previousVelocities_) / interval;
    }
  }
  hasPreviousSample_ = true;
  previousTime_ = time;
  previousVelocities_ = measuredSource.v;
  measuredSource.contactWrenches = measured.measuredContactWrenches;
  computeReference(measured, plan);
  computePlan(measured, plan);
  for (Source& source : sources_) {
    updateKinematics(&source);
  }
  const Source& referenceSource = sources_[kReference];
  const Source& planSource = sources_[kPlan];

  // Panel groups.
  const vector3_t measuredRollPitchYaw(measured.baseEulerAnglesZyx(2), measured.baseEulerAnglesZyx(1), measured.baseEulerAnglesZyx(0));
  for (size_t axis = 0; axis < 3; ++axis) {
    double* position = values(basePosition_[axis]);
    position[0] = measured.basePosition[axis];
    position[1] = referenceBasePosition_[axis];
    double* angle = values(baseRollPitchYaw_[axis]);
    angle[0] = measuredRollPitchYaw[axis];
    angle[1] = referenceRollPitchYaw_[axis];
    double* linear = values(baseLinearVelocity_[axis]);
    linear[0] = measured.baseLinearVelocity[axis];
    linear[1] = referenceBaseLinearVelocity_[axis];
    double* angular = values(baseAngularVelocity_[axis]);
    angular[0] = measured.baseAngularVelocity[axis];
    // The target trajectories carry no base angular velocity.
    angular[1] = 0.0;
  }
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    const vector6_t& mpc = planSource.contactWrenches[contact];
    const vector6_t& sensor = measured.measuredContactWrenches[contact];
    double* normal = values(normalForce_[contact]);
    normal[0] = mpc[WRENCH_FORCE_Z_INDEX];
    normal[1] = sensor[WRENCH_FORCE_Z_INDEX];
    double* tangentialForce = values(tangentialForce_[contact]);
    tangentialForce[0] = mpc[WRENCH_FORCE_X_INDEX];
    tangentialForce[1] = sensor[WRENCH_FORCE_X_INDEX];
    tangentialForce[2] = mpc[WRENCH_FORCE_Y_INDEX];
    tangentialForce[3] = sensor[WRENCH_FORCE_Y_INDEX];
  }
  for (size_t dof = 0; dof < kGeneralizedBaseDofs.size(); ++dof) {
    double* coordinate = values(generalizedBaseCoordinate_[dof]);
    coordinate[0] = measuredSource.q[kGeneralizedBaseDofs[dof]];
    coordinate[1] = referenceSource.q[kGeneralizedBaseDofs[dof]];
    double* velocity = values(generalizedBaseVelocity_[dof]);
    velocity[0] = measuredSource.v[kGeneralizedBaseDofs[dof]];
    velocity[1] = referenceSource.v[kGeneralizedBaseDofs[dof]];
  }
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    const pinocchio::FrameIndex frame = contactFrames_[contact];
    double* acceleration = values(footAcceleration_[contact]);
    double* velocity = values(footVelocity_[contact]);
    for (const size_t source : {kMeasured, kReference}) {
      const Data& data = *sources_[source].data;
      acceleration[source] = pinocchio::getFrameClassicalAcceleration(model, data, frame, pinocchio::LOCAL_WORLD_ALIGNED).linear().z();
      velocity[source] = pinocchio::getFrameVelocity(model, data, frame, pinocchio::LOCAL_WORLD_ALIGNED).linear().z();
    }
  }

  // Complete groups.
  setValues(joints_[kJointPositionMeasured], measured.jointPositions);
  setValues(joints_[kJointPositionTarget], measured.jointPositionTargets);
  setValues(joints_[kJointVelocityMeasured], measured.jointVelocities);
  setValues(joints_[kJointVelocityTarget], measured.jointVelocityTargets);
  setValues(joints_[kJointEffortMeasured], measured.jointEfforts);
  setValues(joints_[kJointEffortTarget], measured.jointFeedForwardEfforts);
  for (size_t source = 0; source < sources_.size(); ++source) {
    setValues(dofPositions_[source], sources_[source].q);
    setValues(dofVelocities_[source], sources_[source].v);
  }
  setValues(dofForces_[kMeasured], measured.generalizedForces);
  // The reference force of a joint is the feed-forward effort of the action applied; the base's is zero.
  double* referenceForce = values(dofForces_[kReference]);
  std::fill(referenceForce, referenceForce + FLOATING_BASE_DIM, 0.0);
  for (size_t joint = 0; joint < modelSettings_.mpc_joint_dim; ++joint) {
    const size_t fullJoint = modelSettings_.mpcModelToFullJointsIndices[joint];
    referenceForce[JOINT_COORDINATE_OFFSET + joint] =
        fullJoint < static_cast<size_t>(measured.jointFeedForwardEfforts.size()) ? measured.jointFeedForwardEfforts[fullJoint] : 0.0;
  }
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    setValues(contactWrenchMpc_[contact], planSource.contactWrenches[contact]);
    setValues(contactWrenchMeasured_[contact], measured.measuredContactWrenches[contact].head<3>());
  }
  double* observationState = values(observationState_);
  double* observationInput = values(observationInput_);
  std::fill(observationState, observationState + stateDim, 0.0);
  std::fill(observationInput, observationInput + inputDim, 0.0);
  values(observationMode_)[0] = 0.0;
  if (hasObservation) {
    setValues(observationState_, observation->state);
    values(observationMode_)[0] = static_cast<double>(observation->mode);
    if (plan != nullptr) {
      vector_t planState;
      vector_t planInput;
      samplePlan(*plan, observation->time, &planState, &planInput);
      setValues(observationInput_, planInput);
    }
  }

  for (const FrameGroups& frame : frames_) {
    for (size_t source = 0; source < sources_.size(); ++source) {
      const Data& data = *sources_[source].data;
      const pinocchio::SE3Tpl<scalar_t>& placement = data.oMf[frame.frame];
      const vector3_t rollPitchYaw = rollPitchYawFromRotation(placement.rotation());
      double* pose = values(frame.groups[kPose][source]);
      for (size_t axis = 0; axis < 3; ++axis) {
        pose[axis] = placement.translation()[axis];
        pose[3 + axis] = rollPitchYaw[axis];
      }
      const Motion twist = pinocchio::getFrameVelocity(model, data, frame.frame, pinocchio::LOCAL_WORLD_ALIGNED);
      setValues(frame.groups[kTwist][source], twist.toVector());
      const Motion acceleration = pinocchio::getFrameClassicalAcceleration(model, data, frame.frame, pinocchio::LOCAL_WORLD_ALIGNED);
      setValues(frame.groups[kAcceleration][source], acceleration.toVector());
      const vector6_t wrench =
          frame.contact >= 0 ? sources_[source].contactWrenches[static_cast<size_t>(frame.contact)] : vector6_t(vector6_t::Zero());
      setValues(frame.groups[kWrench][source], wrench);
    }
  }
  return series_;
}

}  // namespace ocs2::humanoid::visualization
