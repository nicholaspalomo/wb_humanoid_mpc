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

#include "humanoid_nmpc/humanoid_common_mpc_app/visualization/test/VisualizationTestRobot.h"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_centroidal_model/FactoryFunctions.h"
#include "ocs2_core/Types.h"
#include "ocs2_core/manifold/StateManifold.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/common/ContactMask.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/StateManifoldFactory.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/config/model/MpcFormulationFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/config/weights/StateInputWeightsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/gait/BipedModes.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_common_mpc_app/visualization/VisualizationModel.h"
#include "humanoid_mpc_config/model_settings_config.nproto.h"
#include "humanoid_mpc_config/reference_file.nproto.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_msgs/quaternion.pb.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"
#include "humanoid_mpc_msgs/vector3.pb.h"
#include "humanoid_mpc_msgs/wrench.pb.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::visualization::test {

namespace {

std::string resolve(absl::string_view path) {
  const absl::StatusOr<std::string> resolved = robot::resolveResourcePath(path);
  CHECK_OK(resolved.status());
  return *resolved;
}

void setVector3(const vector3_t& value,
                humanoid_mpc_msgs::Vector3* absl_nonnull message) {
  message->set_x(value.x());
  message->set_y(value.y());
  message->set_z(value.z());
}

}  // namespace

RobotFiles g1CentroidalFiles() {
  return {
      .taskFile =
          "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
      .referenceFile =
          "robot_models/unitree_g1/g1_centroidal_mpc/config/command/"
          "reference.textproto",
      .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"};
}

RobotFiles g1WholeBodyFiles() {
  return {
      .taskFile = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
      .referenceFile =
          "robot_models/unitree_g1/g1_wb_mpc/config/command/"
          "reference.textproto",
      .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"};
}

RobotFiles atlasFiles() {
  return {.taskFile =
              "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/"
              "task.textproto",
          .referenceFile =
              "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/"
              "reference.textproto",
          .urdfFile =
              "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"};
}

RobotFiles sa01Files() {
  return {.taskFile =
              "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/"
              "mpc/task.textproto",
          .referenceFile =
              "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/"
              "command/reference.textproto",
          .urdfFile =
              "robot_models/engineai_sa01/engineai_sa01_description/urdf/"
              "zq_sa01.urdf"};
}

RobotFiles r1Files() {
  return {.taskFile =
              "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/"
              "task.textproto",
          .referenceFile =
              "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/"
              "command/reference.textproto",
          .urdfFile =
              "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf"};
}

RobotFiles anymalBFiles() {
  return {.taskFile =
              "robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/config/"
              "mpc/task.textproto",
          .referenceFile =
              "robot_models/anybotics_anymal_b/anymal_b_centroidal_mpc/config/"
              "command/reference.textproto",
          .urdfFile =
              "robot_models/anybotics_anymal_b/anymal_b_description/urdf/"
              "anymal_b.urdf"};
}

RobotFiles tron1Files() {
  return {
      .taskFile =
          "robot_models/limx_tron1/tron1_centroidal_mpc/config/mpc/"
          "task.textproto",
      .referenceFile =
          "robot_models/limx_tron1/tron1_centroidal_mpc/"
          "config/command/reference.textproto",
      .urdfFile = "robot_models/limx_tron1/tron1_description/urdf/tron1.urdf"};
}

RobotFiles anymalCWheelsFiles() {
  return {.taskFile =
              "robot_models/anybotics_anymal_c_wheels/"
              "anymal_c_wheels_centroidal_mpc/config/mpc/task.textproto",
          .referenceFile =
              "robot_models/anybotics_anymal_c_wheels/"
              "anymal_c_wheels_centroidal_mpc/config/command/"
              "reference.textproto",
          .urdfFile =
              "robot_models/anybotics_anymal_c_wheels/"
              "anymal_c_wheels_description/urdf/anymal_c_wheels.urdf"};
}

std::vector<std::pair<RobotFiles, Formulation>> shippedConfigurations() {
  return {{g1CentroidalFiles(), Formulation::kCentroidal},
          {g1WholeBodyFiles(), Formulation::kWholeBody},
          {atlasFiles(), Formulation::kCentroidalBasisVectors},
          {sa01Files(), Formulation::kCentroidalBasisVectors},
          {r1Files(), Formulation::kCentroidal},
          {anymalBFiles(), Formulation::kCentroidal},
          {tron1Files(), Formulation::kCentroidal},
          {anymalCWheelsFiles(), Formulation::kCentroidal}};
}

quaternion_t unitQuaternionOfRollPitchYaw(const vector3_t& rollPitchYaw) {
  return quaternion_t(
      Eigen::AngleAxis<scalar_t>(rollPitchYaw.z(), vector3_t::UnitZ()) *
      Eigen::AngleAxis<scalar_t>(rollPitchYaw.y(), vector3_t::UnitY()) *
      Eigen::AngleAxis<scalar_t>(rollPitchYaw.x(), vector3_t::UnitX()));
}

std::unique_ptr<TestRobot> TestRobot::load(const RobotFiles& files,
                                           Formulation formulation) {
  std::unique_ptr<TestRobot> robot(new TestRobot());
  robot->formulation_ = formulation;
  robot->taskFile_ = resolve(files.taskFile);
  absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(robot->taskFile_);
  CHECK_OK(task.status());
  robot->task_ = *std::move(task);
  robot->referenceFile_ = resolve(files.referenceFile);
  robot->urdfFile_ = resolve(files.urdfFile);
  robot->build();
  return robot;
}

std::unique_ptr<TestRobot> TestRobot::loadWithPointContacts(
    const RobotFiles& files,
    Formulation formulation,
    std::vector<mpc_config::ModelSettingsConfig::PointContact> pointContacts) {
  std::unique_ptr<TestRobot> robot(new TestRobot());
  robot->formulation_ = formulation;
  robot->taskFile_ = resolve(files.taskFile);
  absl::StatusOr<mpc_config::TaskFile> task = loadTaskFile(robot->taskFile_);
  CHECK_OK(task.status());
  robot->task_ = *std::move(task);
  robot->task_.model_settings.contacts_3dof = std::move(pointContacts);
  robot->referenceFile_ = resolve(files.referenceFile);
  robot->urdfFile_ = resolve(files.urdfFile);
  robot->build();
  return robot;
}

void TestRobot::build() {
  modelSettings_ = std::make_unique<ModelSettings>(
      ModelSettings::Create(task_, urdfFile_, "visualization_test",
                            /*verbose=*/false)
          .value());
  pinocchioInterface_ =
      std::make_unique<PinocchioInterface>(makeReferencePinocchioInterface());
  const mpc_config::TaskFile& taskFile = task_;

  if (formulation_ == Formulation::kWholeBody) {
    robotModel_ =
        std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*modelSettings_);
  } else {
    const absl::StatusOr<mpc_config::ReferenceFile> reference =
        loadReferenceFile(referenceFile_);
    CHECK_OK(reference.status());
    const absl::StatusOr<CentroidalModelType> centroidalModelType =
        centroidalModelTypeFromConfig(taskFile);
    CHECK_OK(centroidalModelType.status());
    const absl::StatusOr<vector_t> defaultJointState =
        defaultJointStateFromConfig(*reference,
                                    modelSettings_->mpcModelJointNames,
                                    modelSettings_->fixedJointNames);
    CHECK_OK(defaultJointState.status());
    centroidalModelInfo_ = centroidal_model::createCentroidalModelInfo(
        *pinocchioInterface_, *centroidalModelType, *defaultJointState,
        modelSettings_->contactNames3DoF, modelSettings_->contactNames6DoF);
    std::unique_ptr<MpcRobotModelBase<scalar_t>> wrenchModel =
        std::make_unique<CentroidalMpcRobotModel<scalar_t>>(
            *modelSettings_, *pinocchioInterface_, centroidalModelInfo_);
    if (formulation_ == Formulation::kCentroidal) {
      robotModel_ = std::move(wrenchModel);
    } else {
      const absl::StatusOr<std::vector<ContactWrenchConeBasisMatrix>> bases =
          contactWrenchConeBasesFromConfig(taskFile.contacts, *modelSettings_);
      CHECK_OK(bases.status());
      absl::StatusOr<std::unique_ptr<BasisInputsModelDecorator<scalar_t>>>
          decorator = BasisInputsModelDecorator<scalar_t>::Create(
              std::move(wrenchModel), *bases, *pinocchioInterface_);
      CHECK_OK(decorator.status());
      robotModel_ = *std::move(decorator);
    }
  }
  const StateInputLayout::Mpc mpc = formulation_ == Formulation::kWholeBody
                                        ? StateInputLayout::Mpc::kWholeBody
                                        : StateInputLayout::Mpc::kCentroidal;
  const absl::StatusOr<vector_t> initialState = stateValuesFromConfig(
      taskFile.initial_state, stateInputLayout(*modelSettings_, mpc),
      /*fieldPath=*/"initial_state");
  CHECK_OK(initialState.status());
  // The one mapping of the tuning layout onto the state, as the MPC interfaces
  // build their initial state.
  absl::StatusOr<vector_t> nominalState =
      robotModel_->stateFromTuning(*initialState);
  CHECK_OK(nominalState.status());
  CHECK_EQ(nominalState->size(),
           static_cast<Eigen::Index>(robotModel_->getStateDim()));
  nominalState_ = *std::move(nominalState);
  absl::StatusOr<std::shared_ptr<const StateManifold>> stateManifold =
      stateManifoldOf(*robotModel_);
  CHECK_OK(stateManifold.status());
  stateManifold_ = *std::move(stateManifold);
}

PinocchioInterface TestRobot::makeReferencePinocchioInterface() const {
  return loadCustomPinocchioInterface(task_, urdfFile_, *modelSettings_,
                                      /*scaleTotalMass=*/false)
      .value();
}

VisualizationModel TestRobot::model() const {
  return VisualizationModel{.taskFile = taskFile_,
                            .urdfFile = urdfFile_,
                            .pinocchioInterface = pinocchioInterface_.get(),
                            .mpcRobotModel = robotModel_.get(),
                            .stateManifold = stateManifold_.get(),
                            .nominalState = nominalState_};
}

SystemObservation TestRobot::observation(scalar_t time, size_t mode) const {
  SystemObservation observation;
  observation.time = time;
  observation.mode = mode;
  observation.state = nominalState_;
  observation.input = vector_t::Zero(robotModel_->getInputDim());
  return observation;
}

void TestRobot::makePolicy(scalar_t startTime,
                           size_t nodes,
                           scalar_t forceFootEwrtWorldZ,
                           const vector2_t& posCopRtContactEwrtContactXy,
                           CommandData* absl_nonnull command,
                           PrimalSolution* absl_nonnull solution) const {
  CHECK_GE(nodes, 2u);
  const scalar_t duration = 1.0;
  solution->clear();
  solution->modeSchedule_ =
      ModeSchedule({startTime + 0.3, startTime + 0.6},
                   {kBipedModeStance, kBipedModeRf, kBipedModeStance});
  for (size_t node = 0; node < nodes; ++node) {
    const scalar_t fraction =
        static_cast<scalar_t>(node) / static_cast<scalar_t>(nodes - 1);
    const scalar_t time = startTime + fraction * duration;
    vector_t state = nominalState_;
    vector6_t poseeulerzyxBaseRtWorld =
        robotModel_->getPoseeulerzyxBaseRtWorld(state);
    poseeulerzyxBaseRtWorld[0] += 0.3 * fraction;
    poseeulerzyxBaseRtWorld[3] += 0.2 * fraction;
    robotModel_->setPoseeulerzyxBaseRtWorld(state, poseeulerzyxBaseRtWorld);
    vector_t joints = robotModel_->getJointAngles(state);
    for (Eigen::Index joint = 0; joint < joints.size(); ++joint) {
      joints[joint] +=
          0.05 * std::sin(3.0 * fraction + 0.3 * static_cast<scalar_t>(joint));
    }
    robotModel_->setJointAngles(state, joints);

    vector_t input = vector_t::Zero(robotModel_->getInputDim());
    const ContactMask stance = ContactMask::fromModeNumber(
        solution->modeSchedule_.modeAtTime(time),
        robotModel_->modelSettings.contactNames.size());
    for (size_t contact = 0; contact < stance.size(); ++contact) {
      if (!stance[contact]) {
        continue;
      }
      // The world wrench whose center of pressure is
      // posCopRtContactEwrtContactXy in the contact frame, for the nominal
      // orientation of the feet, which is close to level: tau_x = f_z * y,
      // tau_y = -f_z * x.
      vector6_t wrenchFootEwrtWorld = vector6_t::Zero();
      wrenchFootEwrtWorld[kWrenchForceXIndex] = 0.05 * forceFootEwrtWorldZ;
      wrenchFootEwrtWorld[kWrenchForceZIndex] = forceFootEwrtWorldZ;
      wrenchFootEwrtWorld[kWrenchTorqueXIndex] =
          forceFootEwrtWorldZ * posCopRtContactEwrtContactXy.y();
      wrenchFootEwrtWorld[kWrenchTorqueYIndex] =
          -forceFootEwrtWorldZ * posCopRtContactEwrtContactXy.x();
      robotModel_->setWrenchContactEwrtWorld(state, input, wrenchFootEwrtWorld,
                                             contact);
    }
    solution->timeTrajectory_.push_back(time);
    solution->stateTrajectory_.push_back(state);
    solution->inputTrajectory_.push_back(input);
  }
  command->mpcInitObservation_ = observation(startTime, kBipedModeStance);
  command->mpcTargetTrajectories_ =
      TargetTrajectories({startTime, startTime + duration},
                         {nominalState_, solution->stateTrajectory_.back()},
                         {vector_t::Zero(robotModel_->getInputDim()),
                          vector_t::Zero(robotModel_->getInputDim())});
}

humanoid_mpc_msgs::RobotStateSample TestRobot::robotState(
    scalar_t time,
    const vector3_t& posBaseRtWorldEwrtWorld,
    const vector3_t& rpyBaseRtWorld) const {
  humanoid_mpc_msgs::RobotStateSample sample;
  sample.set_time(time);
  sample.set_control_mode("WB_MPC");
  setVector3(posBaseRtWorldEwrtWorld,
             sample.mutable_pos_base_rt_world_ewrt_world());
  const quaternion_t quatBaseRtWorld =
      unitQuaternionOfRollPitchYaw(rpyBaseRtWorld);
  humanoid_mpc_msgs::Quaternion* absl_nonnull msgQuatBaseRtWorld =
      sample.mutable_quat_base_rt_world();
  msgQuatBaseRtWorld->set_w(quatBaseRtWorld.w());
  msgQuatBaseRtWorld->set_x(quatBaseRtWorld.x());
  msgQuatBaseRtWorld->set_y(quatBaseRtWorld.y());
  msgQuatBaseRtWorld->set_z(quatBaseRtWorld.z());
  sample.mutable_linvel_base_rt_world_ewrt_base();
  sample.mutable_angvel_base_rt_world_ewrt_base();
  const std::vector<std::string>& joints = modelSettings_->fullJointNames;
  for (size_t reversed = 0; reversed < joints.size(); ++reversed) {
    const size_t joint = joints.size() - 1 - reversed;
    sample.add_joint_names(joints[joint]);
    sample.add_joint_positions(0.01 * static_cast<scalar_t>(joint + 1));
    sample.add_joint_velocities(0.1);
    sample.add_joint_measured_efforts(0.0);
  }
  for (size_t contact = 0; contact < modelSettings_->numContacts(); ++contact) {
    sample.add_contact_flags(true);
    humanoid_mpc_msgs::Wrench* absl_nonnull measWrenchContactEwrtWorld =
        sample.add_meas_wrench_contacts_ewrt_world();
    // NOLINTBEGIN(frame-naming): the components of a Wrench message, whose frames its field names
    measWrenchContactEwrtWorld->mutable_force();
    measWrenchContactEwrtWorld->mutable_torque();
    // NOLINTEND(frame-naming)
  }
  return sample;
}

}  // namespace ocs2::humanoid::visualization::test
