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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_mpc/CommandData.h"
#include "ocs2_mpc/SystemObservation.h"
#include "ocs2_oc/oc_data/PrimalSolution.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/visualization/VisualizationModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"

namespace ocs2::humanoid::visualization::test {

/** The MPC formulations the visualization is tested with. */
enum class Formulation {
  /** CentroidalMpcRobotModel, world-frame wrench inputs. */
  kCentroidal,
  /** CentroidalMpcRobotModel behind BasisInputsModelDecorator: local-frame basis-vector inputs. */
  kCentroidalBasisVectors,
  /** WBAccelMpcRobotModel. */
  kWholeBody,
};

/** The files of one shipped robot configuration, relative to the repository root. */
struct RobotFiles {
  std::string taskFile;
  std::string referenceFile;
  std::string urdfFile;
};

/** The shipped configurations: G1 centroidal and whole-body, Atlas, SA01 and R1. */
RobotFiles g1CentroidalFiles();
RobotFiles g1WholeBodyFiles();
RobotFiles atlasFiles();
RobotFiles sa01Files();
RobotFiles r1Files();
/** Every shipped task file, with the formulation it runs. */
std::vector<std::pair<RobotFiles, Formulation>> shippedConfigurations();

/**
 * A robot's MPC model as the MPC node builds it, without the solver: the model settings, the MPC's Pinocchio model
 * (loadCustomPinocchioInterface) and the robot model of the formulation; plus synthetic observations, policies and
 * robot/state samples for it. Paths are resolved from the test's runfiles.
 */
class TestRobot {
 public:
  static std::unique_ptr<TestRobot> load(const RobotFiles& files, Formulation formulation);

  VisualizationModel model() const;
  const ModelSettings& modelSettings() const { return *modelSettings_; }
  const PinocchioInterface& pinocchioInterface() const { return *pinocchioInterface_; }
  const MpcRobotModelBase<scalar_t>& robotModel() const { return *robotModel_; }
  /** A second, independently built MPC Pinocchio model, for reference kinematics. */
  PinocchioInterface makeReferencePinocchioInterface() const;
  const std::string& taskFile() const { return taskFile_; }
  /** The task file, as loadTaskFile() read it. */
  const mpc_config::TaskFile& task() const { return task_; }
  const std::string& urdfFile() const { return urdfFile_; }
  Formulation formulation() const { return formulation_; }

  /** The task file's initial_state: a standing configuration. */
  const vector_t& nominalState() const { return nominalState_; }

  /** The nominal state at `time` in `mode`, its input zero. */
  SystemObservation observation(scalar_t time, size_t mode) const;

  /**
   * A policy from `startTime` over one second in `nodes` nodes: the nominal state moving forward and bending its
   * joints, each stance foot carrying `normalForce` with its center of pressure `copOffset` (in the contact frame) away
   * from the contact frame, written through the model's world-frame setter. The mode schedule is STANCE, then RF
   * (left foot in swing) from startTime + 0.3 s, then STANCE from startTime + 0.6 s; the target trajectories run from
   * the nominal state to the policy's last state, with zero inputs.
   */
  void makePolicy(scalar_t startTime,
                  size_t nodes,
                  scalar_t normalForce,
                  const vector2_t& copOffset,
                  CommandData* absl_nonnull command,
                  PrimalSolution* absl_nonnull solution) const;

  /**
   * A robot/state sample of every joint of fullJointNames (named, in reverse order, so that the order of the names is
   * exercised): joint i at 0.01 * (i + 1) rad, moving at 0.1 rad/s, the base at `position` with roll, pitch, yaw
   * `rollPitchYaw`, no joint action.
   */
  humanoid_mpc_msgs::RobotStateSample robotState(scalar_t time, const vector3_t& position, const vector3_t& rollPitchYaw) const;

 private:
  TestRobot() = default;

  Formulation formulation_ = Formulation::kCentroidal;
  std::string taskFile_;
  mpc_config::TaskFile task_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo centroidalModelInfo_;
  std::unique_ptr<MpcRobotModelBase<scalar_t>> robotModel_;
  vector_t nominalState_;
};

/** The quaternion of roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll). */
quaternion_t quaternionFromRollPitchYaw(const vector3_t& rollPitchYaw);

}  // namespace ocs2::humanoid::visualization::test
