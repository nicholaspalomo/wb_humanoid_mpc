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

#include "humanoid_common_mpc_app/visualization/TelemetryBuilder.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/orientation/BaseOrientation.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"
#include "humanoid_common_mpc_app/visualization/BalanceQuantities.h"
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

// The entries of a tangential contact force group, in the contract's order.
constexpr size_t kTangentialMpcX = 0;
constexpr size_t kTangentialMeasuredX = 1;
constexpr size_t kTangentialMpcY = 2;
constexpr size_t kTangentialMeasuredY = 3;

// The entries of mpc_solution/performance, in the contract's order.
constexpr size_t kPerformanceMerit = 0;
constexpr size_t kPerformanceCost = 1;
constexpr size_t kPerformanceDynamicsViolation = 2;
constexpr size_t kPerformanceEqualityConstraints = 3;
constexpr size_t kPerformanceInequalityConstraints = 4;
constexpr size_t kPerformanceEqualityLagrangian = 5;
constexpr size_t kPerformanceInequalityLagrangian = 6;
constexpr size_t kPerformanceEntries = 7;

constexpr scalar_t kNotANumber = std::numeric_limits<scalar_t>::quiet_NaN();

// The base panels and the pose groups take an axis of space and the Euler angle
// about it in one loop.
static_assert(kEulerAnglesZyxDim == kPositionDim,
              "one angle per axis of space");

// The base orientation in q: the quaternion (x, y, z, w) of the
// Composite(Translation, Spherical) root after the position; in the display
// coordinates the ZYX Euler angles (yaw, pitch, roll) in its place
// (getBaseDofNames()).
static_assert(kBaseConfigurationDim ==
                  kBaseTranslationDim + kQuaternionCoefficients,
              "q holds the position, then the quaternion");
constexpr size_t kDisplayBaseDim = kBaseTranslationDim + kBaseRotationDim;

// The entries of the display coordinates the "Generalized Coordinates
// (Pinocchio)" panels plot: base_z, base_pitch, base_roll; and of v (or of the
// Euler rates) the velocity panels plot: base_vz, then the pitch and the roll
// rate.
constexpr std::array<size_t, 3> kGeneralizedBaseDofs = {
    kBasePosZIndex, kBaseTranslationDim + kBaseRotPitchIndex,
    kBaseTranslationDim + kBaseRotRollIndex};

/** The (roll, pitch, yaw) order of the plots of ZYX Euler angles or rates (yaw,
 * pitch, roll). */
vector3_t rollPitchYawOf(const vector3_t& eulerZyx) {
  return vector3_t(eulerZyx(kBaseRotRollIndex), eulerZyx(kBaseRotPitchIndex),
                   eulerZyx(kBaseRotYawIndex));
}

/** The base orientation of the Pinocchio configuration `q`, as quaternion
 * coefficients (x, y, z, w), normalized. */
vector4_t baseOrientationOf(const vector_t& q) {
  return safelyNormalizedQuaternion(
      vector4_t(q.segment<kQuaternionCoefficients>(kBaseTranslationDim)));
}

/** The ZYX Euler angles (yaw, pitch, roll) of a rotation matrix, through its
 * quaternion (orientation/EulerBoundary.h). */
vector3_t eulerZyxOfRotation(const matrix3_t& rotation) {
  return eulerZyxFromQuaternion(
      safelyNormalizedQuaternion(quaternion_t(rotation).coeffs()));
}

/** "<prefix>0", "<prefix>1", ...: one name per component of a vector of `size`.
 */
std::vector<std::string> indexedNames(absl::string_view prefix, size_t size) {
  std::vector<std::string> names;
  names.reserve(size);
  for (size_t index = 0; index < size; ++index) {
    names.push_back(absl::StrCat(prefix, index));
  }
  return names;
}

}  // namespace

TelemetryBuilder::TelemetryBuilder(const VisualizationModel& model,
                                   const VisualizationConfig& /*config*/)
    : pinocchioInterface_(*model.pinocchioInterface),
      robotModel_(model.mpcRobotModel->clone()),
      stateManifold_(model.stateManifold) {
  for (Source& source : sources_) {
    source.data = std::make_unique<Data>(pinocchioInterface_.getModel());
  }
  if (model.nominalState.size() > 0) {
    // The pendulum of the DCM: the repository's one pendulum height, at the
    // nominal state, and the model's gravity.
    const scalar_t gravity =
        pinocchioInterface_.getModel().gravity.linear().norm();
    const scalar_t height = computeComHeightAboveFeet(
        robotModel_->getGeneralizedCoordinates(model.nominalState),
        pinocchioInterface_, *robotModel_);
    pendulumFrequency_ = pendulumFrequency(gravity, height);
    LOG(INFO) << "[TelemetryBuilder] the DCM's pendulum: height " << height
              << " m (the nominal state's CoM above its feet), gravity "
              << gravity << " m/s^2 (the Pinocchio model's), omega "
              << pendulumFrequency_.value_or(kNotANumber) << " 1/s.";
  }
}

absl::StatusOr<std::unique_ptr<TelemetryBuilder>> TelemetryBuilder::Create(
    const VisualizationModel& model,
    const VisualizationConfig& config,
    const SolutionTermNames& termNames) {
  RETURN_IF_ERROR(checkVisualizationModel(model));
  std::unique_ptr<TelemetryBuilder> builder =
      absl::WrapUnique(new TelemetryBuilder(model, config));
  RETURN_IF_ERROR(builder->initialize(config, termNames));
  return builder;
}

int TelemetryBuilder::addGroup(const std::string& path,
                               const std::vector<std::string>& names) {
  humanoid_mpc_msgs::ScalarGroup* absl_nonnull group = series_.add_groups();
  group->set_path(path);
  for (const std::string& name : names) {
    group->add_names(name);
  }
  group->mutable_values()->resize(static_cast<int>(names.size()),
                                  /*value=*/0.0);
  groupValues_.push_back(group->mutable_values());
  return static_cast<int>(groupValues_.size()) - 1;
}

void TelemetryBuilder::setValues(int group, const vector_t& vector) {
  // The layout is fixed at construction; a vector of another size (which the
  // model's dimensions rule out) neither overruns the group nor leaves stale
  // values in it.
  google::protobuf::RepeatedField<double>* absl_nonnull groupValues =
      groupValues_[static_cast<size_t>(group)];
  const Eigen::Index size =
      std::min<Eigen::Index>(vector.size(), groupValues->size());
  std::copy(vector.data(), vector.data() + size, groupValues->mutable_data());
  std::fill(groupValues->mutable_data() + size,
            groupValues->mutable_data() + groupValues->size(), 0.0);
}

absl::Status TelemetryBuilder::initialize(const VisualizationConfig& config,
                                          const SolutionTermNames& termNames) {
  static_assert(kPlan + 1 == kNumSources, "one source of the plots per index");
  static_assert(kWrench + 1 == kNumFrameKinds,
                "one kind of a frame's groups per index");
  static_assert(kJointEffortTarget + 1 == kNumJointGroups,
                "one joint group per index");
  const Model& model = pinocchioInterface_.getModel();
  if (modelSettings().contactNames.size() < kNumContacts) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the model settings name ", modelSettings().contactNames.size(),
        " contacts; the telemetry needs ", kNumContacts, "."));
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    const std::string& name =
        modelSettings().contactNames[footContact(contact)];
    if (!model.existFrame(name)) {
      return absl::InvalidArgumentError(
          absl::StrCat("the contact frame '", name,
                       "' is not a frame of the MPC's robot model."));
    }
    contactFrames_[contact] = model.getFrameId(name);
  }
  // The ACoM of the robot's network, the MPC's own (ComAndAcomTrackingCost); a
  // robot without one plots NaN.
  absl::StatusOr<std::unique_ptr<AngularCenterOfMass>> acom =
      AngularCenterOfMass::Create(modelSettings().robotName,
                                  modelSettings().mpcModelJointNames);
  if (acom.ok()) {
    acom_ = std::move(*acom);
  } else {
    LOG(INFO) << "[TelemetryBuilder] no ACoM plots (acom/*): "
              << acom.status().message();
  }

  // The groups of the telemetry contract, in its order: the panel groups, then
  // the complete groups.
  // LINT.IfChange(telemetry_groups)
  const std::vector<std::string> measuredReference = {"measured", "reference"};
  const std::vector<std::string> mpcMeasured = {"mpc", "measured"};
  const std::vector<std::string> tangential = {"mpc_x", "measured_x", "mpc_y",
                                               "measured_y"};
  const std::vector<std::string> poseNames = {"x",    "y",     "z",
                                              "roll", "pitch", "yaw"};
  const std::vector<std::string> twistNames = {"linear_x",  "linear_y",
                                               "linear_z",  "angular_x",
                                               "angular_y", "angular_z"};
  const std::vector<std::string> wrenchNames = {
      "force_x", "force_y", "force_z", "torque_x", "torque_y", "torque_z"};
  const std::vector<std::string> forceNames = {"force_x", "force_y", "force_z"};
  const std::array<const char* absl_nonnull, kPositionDim> axes = {"x", "y",
                                                                   "z"};
  const std::array<const char* absl_nonnull, kEulerAnglesZyxDim> angles = {
      "roll", "pitch", "yaw"};
  const std::array<const char* absl_nonnull, kNumContacts> sides = {"left",
                                                                    "right"};
  const std::array<const char* absl_nonnull, kNumSources> sources = {
      "measured", "reference", "plan"};
  const std::array<const char* absl_nonnull, kNumFrameKinds> frameKinds = {
      "pose", "twist", "acceleration", "wrench"};
  const std::vector<std::string> sourceNames(sources.begin(), sources.end());
  const std::array<const char* absl_nonnull, kHorizontalPositionDim>
      horizontalAxes = {"x", "y"};

  for (size_t axis = 0; axis < kPositionDim; ++axis) {
    basePosition_[axis] = addGroup(
        absl::StrCat("base_pose/position_", axes[axis]), measuredReference);
  }
  for (size_t angle = 0; angle < kEulerAnglesZyxDim; ++angle) {
    baseRollPitchYaw_[angle] =
        addGroup(absl::StrCat("base_pose/", angles[angle]), measuredReference);
  }
  for (size_t angle = 0; angle < kEulerAnglesZyxDim; ++angle) {
    baseRollPitchYawRates_[angle] = addGroup(
        absl::StrCat("base_pose/", angles[angle], "_rate"), measuredReference);
  }
  for (size_t axis = 0; axis < kRotationTangentDim; ++axis) {
    baseAngularVelocityLocal_[axis] =
        addGroup(absl::StrCat("base_pose/angular_velocity_", axes[axis]),
                 measuredReference);
  }
  for (size_t axis = 0; axis < kPositionDim; ++axis) {
    baseLinearVelocity_[axis] = addGroup(
        absl::StrCat("base_twist/linear_", axes[axis]), measuredReference);
  }
  for (size_t axis = 0; axis < kRotationTangentDim; ++axis) {
    baseAngularVelocity_[axis] = addGroup(
        absl::StrCat("base_twist/angular_", axes[axis]), measuredReference);
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    normalForce_[contact] =
        addGroup(absl::StrCat("contact_forces/", sides[contact], "_normal"),
                 mpcMeasured);
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    tangentialForce_[contact] =
        addGroup(absl::StrCat("contact_forces/", sides[contact], "_tangential"),
                 tangential);
  }
  generalizedBaseCoordinate_ = {
      addGroup("generalized_base/position_z", measuredReference),
      addGroup("generalized_base/pitch", measuredReference),
      addGroup("generalized_base/roll", measuredReference)};
  generalizedBaseVelocity_ = {
      addGroup("generalized_base/velocity_z", measuredReference),
      addGroup("generalized_base/pitch_rate", measuredReference),
      addGroup("generalized_base/roll_rate", measuredReference)};
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    footAcceleration_[contact] = addGroup(
        absl::StrCat("foot_kinematics/", sides[contact], "_acceleration_z"),
        measuredReference);
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    footVelocity_[contact] = addGroup(
        absl::StrCat("foot_kinematics/", sides[contact], "_velocity_z"),
        measuredReference);
  }
  for (size_t axis = 0; axis < kPositionDim; ++axis) {
    comPosition_[axis] =
        addGroup(absl::StrCat("com/position_", axes[axis]), sourceNames);
  }
  for (size_t axis = 0; axis < kHorizontalPositionDim; ++axis) {
    dcm_[axis] =
        addGroup(absl::StrCat("dcm/", horizontalAxes[axis]), sourceNames);
  }
  for (size_t axis = 0; axis < kHorizontalPositionDim; ++axis) {
    zmp_[axis] =
        addGroup(absl::StrCat("zmp/", horizontalAxes[axis]), sourceNames);
  }
  for (size_t angle = 0; angle < kEulerAnglesZyxDim; ++angle) {
    acomAngles_[angle] =
        addGroup(absl::StrCat("acom/", angles[angle]), sourceNames);
  }

  const std::vector<std::string>& jointNames = modelSettings().fullJointNames;
  joints_ = {addGroup("joints/position/measured", jointNames),
             addGroup("joints/position/target", jointNames),
             addGroup("joints/velocity/measured", jointNames),
             addGroup("joints/velocity/target", jointNames),
             addGroup("joints/effort/measured", jointNames),
             addGroup("joints/effort/target", jointNames)};
  // The display coordinates name the base's Euler angles; v and the forces dual
  // to it the base's twist.
  std::vector<std::string> dofNames = getBaseDofNames();
  dofNames.insert(dofNames.end(), modelSettings().mpcModelJointNames.begin(),
                  modelSettings().mpcModelJointNames.end());
  std::vector<std::string> velocityDofNames = getBaseVelocityDofNames();
  velocityDofNames.insert(velocityDofNames.end(),
                          modelSettings().mpcModelJointNames.begin(),
                          modelSettings().mpcModelJointNames.end());
  for (size_t source = 0; source < kNumSources; ++source) {
    dofPositions_[source] =
        addGroup(absl::StrCat("dofs/position/", sources[source]), dofNames);
  }
  for (size_t source = 0; source < kNumSources; ++source) {
    dofVelocities_[source] = addGroup(
        absl::StrCat("dofs/velocity/", sources[source]), velocityDofNames);
  }
  for (size_t source = 0; source < kNumForceSources; ++source) {
    dofForces_[source] = addGroup(absl::StrCat("dofs/force/", sources[source]),
                                  velocityDofNames);
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    contactWrenchMpc_[contact] = addGroup(
        absl::StrCat("contact_wrenches/", sides[contact], "/mpc"), wrenchNames);
    contactWrenchMeasured_[contact] =
        addGroup(absl::StrCat("contact_wrenches/", sides[contact], "/measured"),
                 forceNames);
  }
  observationState_ = addGroup("mpc_observation/state",
                               indexedNames("x", robotModel_->getStateDim()));
  observationInput_ = addGroup("mpc_observation/input",
                               indexedNames("u", robotModel_->getInputDim()));
  observationMode_ = addGroup("mpc_observation/mode", {"mode"});
  solutionPerformance_ =
      addGroup("mpc_solution/performance",
               {"merit", "cost", "dynamics_violation_sse",
                "equality_constraints_sse", "inequality_constraints_sse",
                "equality_lagrangian", "inequality_lagrangian"});
  solutionCosts_ = addGroup("mpc_solution/costs", termNames.costs);
  solutionConstraints_ =
      addGroup("mpc_solution/constraints", termNames.constraints);
  solutionCostCount_ = termNames.costs.size();
  solutionConstraintCount_ = termNames.constraints.size();

  for (const std::string& frame : config.telemetryFrames) {
    if (!model.existFrame(frame)) {
      return absl::InvalidArgumentError(
          absl::StrCat(kTelemetryFramesField, " names the frame '", frame,
                       "', which the MPC's robot model does not have."));
    }
    FrameGroups groups{
        .frame = model.getFrameId(frame), .contact = -1, .groups = {}};
    for (size_t contact = 0; contact < kNumContacts; ++contact) {
      if (modelSettings().contactNames[footContact(contact)] == frame) {
        groups.contact = static_cast<int>(contact);
      }
    }
    const std::array<const std::vector<std::string>* absl_nonnull,
                     kNumFrameKinds>
        kindNames = {&poseNames, &twistNames, &twistNames, &wrenchNames};
    for (size_t kind = 0; kind < frameKinds.size(); ++kind) {
      for (size_t source = 0; source < sources.size(); ++source) {
        groups.groups[kind][source] =
            addGroup(absl::StrCat("frames/", frameKinds[kind], "/", frame, "/",
                                  sources[source]),
                     *kindNames[kind]);
      }
    }
    frames_.push_back(groups);
  }
  // clang-format off
  // LINT.ThenChange(//humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/telemetry_contract.py:panel_groups, //humanoid_nmpc/humanoid_rerun_viewer/python/humanoid_rerun_viewer/telemetry_contract.py:complete_groups)
  // clang-format on

  for (Source& source : sources_) {
    source.q.setZero(robotModel_->getConfigurationDim());
    source.v.setZero(robotModel_->getVelocityDim());
    source.a.setZero(robotModel_->getVelocityDim());
    source.displayCoordinates.setZero(static_cast<Eigen::Index>(
        kDisplayBaseDim + robotModel_->getJointDim()));
    source.jointPositions.setZero(
        static_cast<Eigen::Index>(robotModel_->getJointDim()));
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

vector_t TelemetryBuilder::generalizedVelocities(const vector_t& state,
                                                 const vector_t& input) {
  return robotModel_->getGeneralizedVelocities(state, input);
}

void TelemetryBuilder::updateDisplayValues(Source* absl_nonnull source) {
  source->eulerZyxRates = eulerZyxRatesForDisplay(
      source->eulerAnglesZyx,
      source->v.segment<kRotationTangentDim>(kBaseTranslationDim));
  const Eigen::Index jointDim = source->displayCoordinates.size() -
                                static_cast<Eigen::Index>(kDisplayBaseDim);
  source->displayCoordinates.head<kBaseTranslationDim>() =
      source->q.head<kBaseTranslationDim>();
  source->displayCoordinates.segment<kBaseRotationDim>(kBaseTranslationDim) =
      source->eulerAnglesZyx;
  source->displayCoordinates.tail(jointDim) = source->q.tail(jointDim);
}

void TelemetryBuilder::updateBalance(Source* absl_nonnull source) {
  const Model& model = pinocchioInterface_.getModel();
  Data& data = *source->data;
  // The CoM and its velocity of the kinematics updateKinematics() computed
  // (world frame, data.com[0] and data.vcom[0]).
  pinocchio::centerOfMass(model, data, pinocchio::VELOCITY,
                          /*computeSubtreeComs=*/false);
  source->com = data.com[0];
  source->dcm = pendulumFrequency_.has_value()
                    ? divergentComponentOfMotion(data.com[0], data.vcom[0],
                                                 *pendulumFrequency_)
                    : vector2_t(vector2_t::Constant(kNotANumber));
  feet_array_t<vector3_t> contactPositions;
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    contactPositions[contact] = data.oMf[contactFrames_[contact]].translation();
  }
  const std::optional<vector3_t> zmp = zeroMomentPoint(
      contactPositions, source->contactWrenches, kMinNormalForceForZmp);
  source->zmp = zmp.has_value() ? vector2_t(zmp->head<kHorizontalPositionDim>())
                                : vector2_t(vector2_t::Constant(kNotANumber));
  if (acom_ == nullptr) {
    source->acomRollPitchYaw.setConstant(kNotANumber);
    return;
  }
  source->jointPositions = source->q.tail(source->jointPositions.size());
  source->acomRollPitchYaw =
      rollPitchYawOf(eulerZyxFromQuaternion(acom_->acomAttitude(
          baseOrientationOf(source->q), source->jointPositions)));
}

void TelemetryBuilder::writeSolutionTerms(
    const SolutionTerms* absl_nullable terms) {
  double* absl_nonnull performance = values(solutionPerformance_);
  static_assert(kPerformanceInequalityLagrangian + 1 == kPerformanceEntries,
                "one entry per performance index field");
  if (terms == nullptr) {
    std::fill(performance, performance + kPerformanceEntries, kNotANumber);
  } else {
    performance[kPerformanceMerit] = terms->performance.merit;
    performance[kPerformanceCost] = terms->performance.cost;
    performance[kPerformanceDynamicsViolation] =
        terms->performance.dynamicsViolationSSE;
    performance[kPerformanceEqualityConstraints] =
        terms->performance.equalityConstraintsSSE;
    performance[kPerformanceInequalityConstraints] =
        terms->performance.inequalityConstraintsSSE;
    performance[kPerformanceEqualityLagrangian] =
        terms->performance.equalityLagrangian;
    performance[kPerformanceInequalityLagrangian] =
        terms->performance.inequalityLagrangian;
  }
  // A group without series has no values to write (values() needs at least
  // one).
  if (solutionCostCount_ > 0) {
    if (terms != nullptr &&
        static_cast<size_t>(terms->costs.size()) == solutionCostCount_) {
      setValues(solutionCosts_, terms->costs);
    } else {
      std::fill(values(solutionCosts_),
                values(solutionCosts_) + solutionCostCount_, kNotANumber);
    }
  }
  if (solutionConstraintCount_ > 0) {
    if (terms != nullptr &&
        static_cast<size_t>(terms->constraintViolations.size()) ==
            solutionConstraintCount_) {
      setValues(solutionConstraints_, terms->constraintViolations);
    } else {
      std::fill(values(solutionConstraints_),
                values(solutionConstraints_) + solutionConstraintCount_,
                kNotANumber);
    }
  }
}

void TelemetryBuilder::updateKinematics(Source* absl_nonnull source) {
  const Model& model = pinocchioInterface_.getModel();
  pinocchio::forwardKinematics(model, *source->data, source->q, source->v,
                               source->a);
  pinocchio::updateFramePlacements(model, *source->data);
}

void TelemetryBuilder::computeReference(
    const DecodedRobotState& measured,
    const PolicySnapshot* absl_nullable policy) {
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
      policy != nullptr ? sampleTarget(policy->target, stateManifold_, time,
                                       stateDim, inputDim, &state, &input)
                        : TargetSample::kNone;
  if (sample == TargetSample::kNone) {
    // No reference: the measured robot at rest, as PinocchioTelemetryPublisher
    // fell back.
    reference.q = measured.generalizedCoordinates;
    reference.eulerAnglesZyx = measured.baseEulerAnglesZyx;
    referenceBasePosition_ = measured.basePosition;
    referenceBaseLinearVelocity_.setZero();
    return;
  }
  reference.q = robotModel_->getGeneralizedCoordinates(state);
  reference.eulerAnglesZyx = robotModel_->getEulerzyxBaseRtWorld(state);
  referenceBasePosition_ = robotModel_->getPosBaseRtWorldEwrtWorld(state);
  referenceBaseLinearVelocity_ = robotModel_->getBaseComLinearVelocity(state);
  if (sample != TargetSample::kStateAndInput) {
    return;
  }
  reference.v = generalizedVelocities(state, input);
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    reference.contactWrenches[contact] = robotModel_->getWrenchContactEwrtWorld(
        state, input, footContact(contact));
  }
  const scalar_array_t& times = policy->target.timeTrajectory;
  const scalar_t earlier =
      std::max(time - kAccelerationHalfInterval, times.front());
  const scalar_t later =
      std::min(time + kAccelerationHalfInterval, times.back());
  if (later > earlier) {
    vector_t earlierState;
    vector_t earlierInput;
    vector_t laterState;
    vector_t laterInput;
    sampleTarget(policy->target, stateManifold_, earlier, stateDim, inputDim,
                 &earlierState, &earlierInput);
    sampleTarget(policy->target, stateManifold_, later, stateDim, inputDim,
                 &laterState, &laterInput);
    reference.a = (generalizedVelocities(laterState, laterInput) -
                   generalizedVelocities(earlierState, earlierInput)) /
                  (later - earlier);
  }
}

void TelemetryBuilder::computePlan(const DecodedRobotState& measured,
                                   const PolicySnapshot* absl_nullable policy) {
  Source& plan = sources_[kPlan];
  plan.contactWrenches = makeFeetArray<vector6_t>(vector6_t::Zero());
  plan.v.setZero();
  plan.a.setZero();
  if (policy == nullptr) {
    plan.q = measured.generalizedCoordinates;
    plan.eulerAnglesZyx = measured.baseEulerAnglesZyx;
    return;
  }
  const scalar_t time = measured.time;
  vector_t state;
  vector_t input;
  samplePlan(*policy, stateManifold_, time, &state, &input);
  plan.q = robotModel_->getGeneralizedCoordinates(state);
  plan.eulerAnglesZyx = robotModel_->getEulerzyxBaseRtWorld(state);
  plan.v = generalizedVelocities(state, input);
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    plan.contactWrenches[contact] = robotModel_->getWrenchContactEwrtWorld(
        state, input, footContact(contact));
  }
  const scalar_t earlier =
      std::max(time - kAccelerationHalfInterval, policy->time.front());
  const scalar_t later =
      std::min(time + kAccelerationHalfInterval, policy->time.back());
  if (later > earlier) {
    vector_t earlierState;
    vector_t earlierInput;
    vector_t laterState;
    vector_t laterInput;
    samplePlan(*policy, stateManifold_, earlier, &earlierState, &earlierInput);
    samplePlan(*policy, stateManifold_, later, &laterState, &laterInput);
    plan.a = (generalizedVelocities(laterState, laterInput) -
              generalizedVelocities(earlierState, earlierInput)) /
             (later - earlier);
  }
}

void TelemetryBuilder::updateMeasuredSource(const DecodedRobotState& measured) {
  const scalar_t time = measured.time;
  Source& measuredSource = sources_[kMeasured];
  measuredSource.q = measured.generalizedCoordinates;
  measuredSource.v = measured.generalizedVelocities;
  measuredSource.a.setZero(measuredSource.v.size());
  if (hasPreviousSample_ &&
      previousVelocities_.size() == measuredSource.v.size()) {
    const scalar_t interval = time - previousTime_;
    if (interval > kMinDifferenceInterval &&
        interval < kMaxDifferenceInterval) {
      measuredSource.a = (measuredSource.v - previousVelocities_) / interval;
    }
  }
  hasPreviousSample_ = true;
  previousTime_ = time;
  previousVelocities_ = measuredSource.v;
  measuredSource.contactWrenches = measured.measuredContactWrenches;
  measuredSource.eulerAnglesZyx = measured.baseEulerAnglesZyx;
}

void TelemetryBuilder::writeBasePanels(const DecodedRobotState& measured) {
  const Model& model = pinocchioInterface_.getModel();
  const Source& measuredSource = sources_[kMeasured];
  const Source& referenceSource = sources_[kReference];
  const Source& planSource = sources_[kPlan];

  // Panel groups. The base's angular velocity: w_B is v's, the world-frame one
  // R(xi) w_B with xi the orientation of q; the reference's is zero unless the
  // target trajectory carries inputs too (TargetSample::kStateAndInput).
  const vector3_t measuredRollPitchYaw =
      rollPitchYawOf(measuredSource.eulerAnglesZyx);
  const vector3_t referenceRollPitchYaw =
      rollPitchYawOf(referenceSource.eulerAnglesZyx);
  const vector3_t measuredRollPitchYawRates =
      rollPitchYawOf(measuredSource.eulerZyxRates);
  const vector3_t referenceRollPitchYawRates =
      rollPitchYawOf(referenceSource.eulerZyxRates);
  const vector3_t referenceAngularVelocityLocal =
      referenceSource.v.segment<kRotationTangentDim>(kBaseTranslationDim);
  const vector3_t referenceAngularVelocityWorld =
      quaternion_t(baseOrientationOf(referenceSource.q)).toRotationMatrix() *
      referenceAngularVelocityLocal;
  for (size_t axis = 0; axis < kPositionDim; ++axis) {
    double* absl_nonnull position = values(basePosition_[axis]);
    position[0] = measured.basePosition[axis];
    position[1] = referenceBasePosition_[axis];
    double* absl_nonnull angle = values(baseRollPitchYaw_[axis]);
    angle[0] = measuredRollPitchYaw[axis];
    angle[1] = referenceRollPitchYaw[axis];
    double* absl_nonnull angleRate = values(baseRollPitchYawRates_[axis]);
    angleRate[0] = measuredRollPitchYawRates[axis];
    angleRate[1] = referenceRollPitchYawRates[axis];
    double* absl_nonnull angularLocal = values(baseAngularVelocityLocal_[axis]);
    angularLocal[0] = measured.baseAngularVelocityLocal[axis];
    angularLocal[1] = referenceAngularVelocityLocal[axis];
    double* absl_nonnull linear = values(baseLinearVelocity_[axis]);
    linear[0] = measured.baseLinearVelocity[axis];
    linear[1] = referenceBaseLinearVelocity_[axis];
    double* absl_nonnull angular = values(baseAngularVelocity_[axis]);
    angular[0] = measured.baseAngularVelocity[axis];
    angular[1] = referenceAngularVelocityWorld[axis];
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    const vector6_t& mpc = planSource.contactWrenches[contact];
    const vector6_t& sensor = measured.measuredContactWrenches[contact];
    double* absl_nonnull normal = values(normalForce_[contact]);
    normal[0] = mpc[kWrenchForceZIndex];
    normal[1] = sensor[kWrenchForceZIndex];
    double* absl_nonnull tangentialForce = values(tangentialForce_[contact]);
    tangentialForce[kTangentialMpcX] = mpc[kWrenchForceXIndex];
    tangentialForce[kTangentialMeasuredX] = sensor[kWrenchForceXIndex];
    tangentialForce[kTangentialMpcY] = mpc[kWrenchForceYIndex];
    tangentialForce[kTangentialMeasuredY] = sensor[kWrenchForceYIndex];
  }
  // The coordinates of the display buffers; the vertical velocity of v, and the
  // Euler rates of the angles.
  static_assert(kGeneralizedBaseDofs.size() == kNumGeneralizedBasePanels,
                "one panel per generalized base coordinate");
  for (size_t dof = 0; dof < kGeneralizedBaseDofs.size(); ++dof) {
    double* absl_nonnull coordinate = values(generalizedBaseCoordinate_[dof]);
    coordinate[0] =
        measuredSource.displayCoordinates[kGeneralizedBaseDofs[dof]];
    coordinate[1] =
        referenceSource.displayCoordinates[kGeneralizedBaseDofs[dof]];
    double* absl_nonnull velocity = values(generalizedBaseVelocity_[dof]);
    if (kGeneralizedBaseDofs[dof] < kBaseTranslationDim) {
      velocity[0] = measuredSource.v[kGeneralizedBaseDofs[dof]];
      velocity[1] = referenceSource.v[kGeneralizedBaseDofs[dof]];
    } else {
      velocity[0] =
          measuredSource
              .eulerZyxRates[kGeneralizedBaseDofs[dof] - kBaseTranslationDim];
      velocity[1] =
          referenceSource
              .eulerZyxRates[kGeneralizedBaseDofs[dof] - kBaseTranslationDim];
    }
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    const pinocchio::FrameIndex frame = contactFrames_[contact];
    double* absl_nonnull acceleration = values(footAcceleration_[contact]);
    double* absl_nonnull velocity = values(footVelocity_[contact]);
    for (const size_t source : {kMeasured, kReference}) {
      const Data& data = *sources_[source].data;
      acceleration[source] =
          pinocchio::getFrameClassicalAcceleration(
              model, data, frame, pinocchio::LOCAL_WORLD_ALIGNED)
              .linear()
              .z();
      velocity[source] = pinocchio::getFrameVelocity(
                             model, data, frame, pinocchio::LOCAL_WORLD_ALIGNED)
                             .linear()
                             .z();
    }
  }
}

void TelemetryBuilder::writeBalancePanels() {
  // The balance panels: measured, reference, plan.
  for (size_t source = 0; source < kNumSources; ++source) {
    for (size_t axis = 0; axis < kPositionDim; ++axis) {
      values(comPosition_[axis])[source] =
          sources_[source].com[static_cast<Eigen::Index>(axis)];
      values(acomAngles_[axis])[source] =
          sources_[source].acomRollPitchYaw[static_cast<Eigen::Index>(axis)];
    }
    for (size_t axis = 0; axis < kHorizontalPositionDim; ++axis) {
      values(dcm_[axis])[source] =
          sources_[source].dcm[static_cast<Eigen::Index>(axis)];
      values(zmp_[axis])[source] =
          sources_[source].zmp[static_cast<Eigen::Index>(axis)];
    }
  }
}

void TelemetryBuilder::writeCompleteGroups(const DecodedRobotState& measured) {
  const Source& planSource = sources_[kPlan];
  // Complete groups.
  setValues(joints_[kJointPositionMeasured], measured.jointPositions);
  setValues(joints_[kJointPositionTarget], measured.jointPositionTargets);
  setValues(joints_[kJointVelocityMeasured], measured.jointVelocities);
  setValues(joints_[kJointVelocityTarget], measured.jointVelocityTargets);
  setValues(joints_[kJointEffortMeasured], measured.jointEfforts);
  setValues(joints_[kJointEffortTarget], measured.jointFeedForwardEfforts);
  for (size_t source = 0; source < sources_.size(); ++source) {
    setValues(dofPositions_[source], sources_[source].displayCoordinates);
    setValues(dofVelocities_[source], sources_[source].v);
  }
  setValues(dofForces_[kMeasured], measured.generalizedForces);
  // The reference force of a joint is the feed-forward effort of the action
  // applied; the base's is zero. A force is dual to the velocity v.
  double* absl_nonnull referenceForce = values(dofForces_[kReference]);
  std::fill(referenceForce, referenceForce + kBaseVelocityDim, 0.0);
  for (size_t joint = 0; joint < modelSettings().mpc_joint_dim; ++joint) {
    const size_t fullJoint = modelSettings().mpcModelToFullJointsIndices[joint];
    referenceForce[kJointVelocityOffset + joint] =
        fullJoint < static_cast<size_t>(measured.jointFeedForwardEfforts.size())
            ? measured.jointFeedForwardEfforts[fullJoint]
            : 0.0;
  }
  for (size_t contact = 0; contact < kNumContacts; ++contact) {
    setValues(contactWrenchMpc_[contact], planSource.contactWrenches[contact]);
    setValues(contactWrenchMeasured_[contact],
              measured.measuredContactWrenches[contact].head<kForceDim>());
  }
}

void TelemetryBuilder::writeObservation(
    const SystemObservation* absl_nullable observation,
    const PolicySnapshot* absl_nullable plan) {
  const size_t stateDim = robotModel_->getStateDim();
  const size_t inputDim = robotModel_->getInputDim();
  const bool hasObservation =
      observation != nullptr &&
      static_cast<size_t>(observation->state.size()) == stateDim;
  double* absl_nonnull observationState = values(observationState_);
  double* absl_nonnull observationInput = values(observationInput_);
  std::fill(observationState, observationState + stateDim, 0.0);
  std::fill(observationInput, observationInput + inputDim, 0.0);
  values(observationMode_)[0] = 0.0;
  if (hasObservation) {
    setValues(observationState_, observation->state);
    values(observationMode_)[0] = static_cast<double>(observation->mode);
    if (plan != nullptr) {
      vector_t planState;
      vector_t planInput;
      samplePlan(*plan, stateManifold_, observation->time, &planState,
                 &planInput);
      setValues(observationInput_, planInput);
    }
  }
}

void TelemetryBuilder::writeFramePanels() {
  const Model& model = pinocchioInterface_.getModel();
  for (const FrameGroups& frame : frames_) {
    for (size_t source = 0; source < sources_.size(); ++source) {
      const Data& data = *sources_[source].data;
      const pinocchio::SE3Tpl<scalar_t>& placement = data.oMf[frame.frame];
      const vector3_t rollPitchYaw =
          rollPitchYawOf(eulerZyxOfRotation(placement.rotation()));
      double* absl_nonnull pose = values(frame.groups[kPose][source]);
      for (size_t axis = 0; axis < kPositionDim; ++axis) {
        pose[axis] = placement.translation()[axis];
        pose[kPositionDim + axis] = rollPitchYaw[axis];
      }
      const Motion twist = pinocchio::getFrameVelocity(
          model, data, frame.frame, pinocchio::LOCAL_WORLD_ALIGNED);
      setValues(frame.groups[kTwist][source], twist.toVector());
      const Motion acceleration = pinocchio::getFrameClassicalAcceleration(
          model, data, frame.frame, pinocchio::LOCAL_WORLD_ALIGNED);
      setValues(frame.groups[kAcceleration][source], acceleration.toVector());
      const vector6_t wrench =
          frame.contact >= 0
              ? sources_[source]
                    .contactWrenches[static_cast<size_t>(frame.contact)]
              : vector6_t(vector6_t::Zero());
      setValues(frame.groups[kWrench][source], wrench);
    }
  }
}

const humanoid_mpc_msgs::TelemetrySeries& TelemetryBuilder::build(
    const DecodedRobotState& measured,
    const SystemObservation* absl_nullable observation,
    const PolicySnapshot* absl_nullable policy,
    const SolutionTerms* absl_nullable solutionTerms) {
  const size_t stateDim = robotModel_->getStateDim();
  const size_t inputDim = robotModel_->getInputDim();
  const PolicySnapshot* absl_nullable plan =
      policy != nullptr && isConsistentPlan(*policy, stateDim, inputDim)
          ? policy
          : nullptr;
  series_.set_time(measured.time);

  updateMeasuredSource(measured);
  computeReference(measured, plan);
  computePlan(measured, plan);
  for (Source& source : sources_) {
    updateDisplayValues(&source);
    updateKinematics(&source);
    updateBalance(&source);
  }
  writeBasePanels(measured);
  writeBalancePanels();
  writeCompleteGroups(measured);
  writeObservation(observation, plan);
  writeSolutionTerms(solutionTerms);
  writeFramePanels();
  return series_;
}

}  // namespace ocs2::humanoid::visualization
