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

#include "VisualizationTestRobot.h"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ocs2_centroidal_model/FactoryFunctions.h>
#include <ocs2_core/misc/LoadData.h>
#include <ocs2_core/reference/ModeSchedule.h>
#include <ocs2_core/reference/TargetTrajectories.h>

#include "absl/log/check.h"
#include "absl/status/statusor.h"

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/contact/ContactWrenchConeBasisMatrix.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid::visualization::test {

namespace {

std::string resolve(const std::string& path) {
  const absl::StatusOr<std::string> resolved = robot::resolveResourcePath(path);
  CHECK_OK(resolved.status());
  return *resolved;
}

void setVector3(const vector3_t& value, humanoid_mpc_msgs::Vector3* message) {
  message->set_x(value.x());
  message->set_y(value.y());
  message->set_z(value.z());
}

}  // namespace

RobotFiles g1CentroidalFiles() {
  return {"robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml",
          "robot_models/unitree_g1/g1_centroidal_mpc/config/command/reference.yaml",
          "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"};
}

RobotFiles g1WholeBodyFiles() {
  return {"robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml", "robot_models/unitree_g1/g1_wb_mpc/config/command/reference.yaml",
          "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"};
}

RobotFiles atlasFiles() {
  return {"robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml",
          "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml",
          "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"};
}

RobotFiles sa01Files() {
  return {"robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml",
          "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/command/reference.yaml",
          "robot_models/engineai_sa01/engineai_sa01_description/urdf/zq_sa01.urdf"};
}

RobotFiles r1Files() {
  return {"robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml",
          "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/command/reference.yaml",
          "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf"};
}

std::vector<std::pair<RobotFiles, Formulation>> shippedConfigurations() {
  return {{g1CentroidalFiles(), Formulation::kCentroidal},
          {g1WholeBodyFiles(), Formulation::kWholeBody},
          {atlasFiles(), Formulation::kCentroidalBasisVectors},
          {sa01Files(), Formulation::kCentroidalBasisVectors},
          {r1Files(), Formulation::kCentroidal}};
}

quaternion_t quaternionFromRollPitchYaw(const vector3_t& rollPitchYaw) {
  return quaternion_t(Eigen::AngleAxis<scalar_t>(rollPitchYaw.z(), vector3_t::UnitZ()) *
                      Eigen::AngleAxis<scalar_t>(rollPitchYaw.y(), vector3_t::UnitY()) *
                      Eigen::AngleAxis<scalar_t>(rollPitchYaw.x(), vector3_t::UnitX()));
}

std::unique_ptr<TestRobot> TestRobot::load(const RobotFiles& files, Formulation formulation) {
  std::unique_ptr<TestRobot> robot(new TestRobot());
  robot->formulation_ = formulation;
  robot->taskFile_ = resolve(files.taskFile);
  robot->referenceFile_ = resolve(files.referenceFile);
  robot->urdfFile_ = resolve(files.urdfFile);
  robot->modelSettings_ = std::make_unique<ModelSettings>(robot->taskFile_, robot->urdfFile_, "visualization_test", /*verbose=*/false);
  robot->pinocchioInterface_ = std::make_unique<PinocchioInterface>(robot->makeReferencePinocchioInterface());

  if (formulation == Formulation::kWholeBody) {
    robot->robotModel_ = std::make_unique<WBAccelMpcRobotModel<scalar_t>>(*robot->modelSettings_);
  } else {
    robot->centroidalModelInfo_ = centroidal_model::createCentroidalModelInfo(
        *robot->pinocchioInterface_, centroidal_model::loadCentroidalType(robot->taskFile_),
        centroidal_model::loadDefaultJointState(robot->pinocchioInterface_->getModel().nq - 6, robot->referenceFile_),
        robot->modelSettings_->contactNames3DoF, robot->modelSettings_->contactNames6DoF);
    std::unique_ptr<MpcRobotModelBase<scalar_t>> wrenchModel = std::make_unique<CentroidalMpcRobotModel<scalar_t>>(
        *robot->modelSettings_, *robot->pinocchioInterface_, robot->centroidalModelInfo_);
    if (formulation == Formulation::kCentroidal) {
      robot->robotModel_ = std::move(wrenchModel);
    } else {
      const absl::StatusOr<feet_array_t<ContactWrenchConeBasisMatrix>> bases =
          loadContactWrenchConeBases(robot->taskFile_, *robot->modelSettings_, /*verbose=*/false);
      CHECK_OK(bases.status());
      robot->robotModel_ =
          std::make_unique<BasisInputsModelDecorator<scalar_t>>(std::move(wrenchModel), *bases, *robot->pinocchioInterface_);
    }
  }
  robot->nominalState_.setZero(robot->robotModel_->getStateDim());
  loadData::loadEigenMatrix(robot->taskFile_, "initialState", robot->nominalState_);
  return robot;
}

PinocchioInterface TestRobot::makeReferencePinocchioInterface() const {
  return createCustomPinocchioInterface(taskFile_, urdfFile_, *modelSettings_, /*scaleTotalMass=*/false);
}

VisualizationModel TestRobot::model() const {
  return VisualizationModel{taskFile_, urdfFile_, pinocchioInterface_.get(), robotModel_.get()};
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
                           scalar_t normalForce,
                           const vector2_t& copOffset,
                           CommandData* command,
                           PrimalSolution* solution) const {
  CHECK_GE(nodes, 2u);
  const scalar_t duration = 1.0;
  solution->clear();
  solution->modeSchedule_ = ModeSchedule({startTime + 0.3, startTime + 0.6}, {ModeNumber::STANCE, ModeNumber::RF, ModeNumber::STANCE});
  for (size_t node = 0; node < nodes; ++node) {
    const scalar_t fraction = static_cast<scalar_t>(node) / static_cast<scalar_t>(nodes - 1);
    const scalar_t time = startTime + fraction * duration;
    vector_t state = nominalState_;
    vector6_t basePose = robotModel_->getBasePose(state);
    basePose[0] += 0.3 * fraction;
    basePose[3] += 0.2 * fraction;
    robotModel_->setBasePose(state, basePose);
    vector_t joints = robotModel_->getJointAngles(state);
    for (Eigen::Index joint = 0; joint < joints.size(); ++joint) {
      joints[joint] += 0.05 * std::sin(3.0 * fraction + 0.3 * static_cast<scalar_t>(joint));
    }
    robotModel_->setJointAngles(state, joints);

    vector_t input = vector_t::Zero(robotModel_->getInputDim());
    const contact_flag_t stance = modeNumber2StanceLeg(solution->modeSchedule_.modeAtTime(time));
    for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
      if (!stance[contact]) {
        continue;
      }
      // The world wrench whose center of pressure is copOffset in the contact frame, for the nominal orientation of the
      // feet, which is close to level: tau_x = f_z * y, tau_y = -f_z * x.
      vector6_t wrench = vector6_t::Zero();
      wrench[WRENCH_FORCE_X_INDEX] = 0.05 * normalForce;
      wrench[WRENCH_FORCE_Z_INDEX] = normalForce;
      wrench[WRENCH_TORQUE_X_INDEX] = normalForce * copOffset.y();
      wrench[WRENCH_TORQUE_Y_INDEX] = -normalForce * copOffset.x();
      robotModel_->setContactWrenchInWorldFrame(state, input, wrench, contact);
    }
    solution->timeTrajectory_.push_back(time);
    solution->stateTrajectory_.push_back(state);
    solution->inputTrajectory_.push_back(input);
  }
  command->mpcInitObservation_ = observation(startTime, ModeNumber::STANCE);
  command->mpcTargetTrajectories_ =
      TargetTrajectories({startTime, startTime + duration}, {nominalState_, solution->stateTrajectory_.back()},
                         {vector_t::Zero(robotModel_->getInputDim()), vector_t::Zero(robotModel_->getInputDim())});
}

humanoid_mpc_msgs::RobotStateSample TestRobot::robotState(scalar_t time, const vector3_t& position, const vector3_t& rollPitchYaw) const {
  humanoid_mpc_msgs::RobotStateSample sample;
  sample.set_time(time);
  sample.set_control_mode("WB_MPC");
  setVector3(position, sample.mutable_base_position_world());
  const quaternion_t orientation = quaternionFromRollPitchYaw(rollPitchYaw);
  humanoid_mpc_msgs::Quaternion* quaternion = sample.mutable_base_orientation_world();
  quaternion->set_w(orientation.w());
  quaternion->set_x(orientation.x());
  quaternion->set_y(orientation.y());
  quaternion->set_z(orientation.z());
  sample.mutable_base_linear_velocity_local();
  sample.mutable_base_angular_velocity_local();
  const std::vector<std::string>& joints = modelSettings_->fullJointNames;
  for (size_t reversed = 0; reversed < joints.size(); ++reversed) {
    const size_t joint = joints.size() - 1 - reversed;
    sample.add_joint_names(joints[joint]);
    sample.add_joint_positions(0.01 * static_cast<scalar_t>(joint + 1));
    sample.add_joint_velocities(0.1);
    sample.add_joint_measured_efforts(0.0);
  }
  for (size_t contact = 0; contact < N_CONTACTS; ++contact) {
    sample.add_contact_flags(true);
    humanoid_mpc_msgs::Wrench* wrench = sample.add_measured_contact_wrenches();
    wrench->mutable_force();
    wrench->mutable_torque();
  }
  return sample;
}

}  // namespace ocs2::humanoid::visualization::test
