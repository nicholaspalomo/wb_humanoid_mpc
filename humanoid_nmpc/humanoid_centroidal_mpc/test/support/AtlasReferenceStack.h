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

#include "absl/base/nullability.h"
#include "absl/strings/string_view.h"
#include "ocs2_centroidal_model/CentroidalModelInfo.h"
#include "ocs2_mpc/MPC_Settings.h"
#include "ocs2_mpc_test/ScriptedMpc.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_centroidal_mpc/CentroidalMpcConfig.h"
#include "humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h"
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"
#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/reference_manager/ProceduralMpcMotionManager.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid {

/** A path in the test's runfiles, or empty when it is not there. */
std::string atlasRunfilePath(absl::string_view relativePath);

/**
 * The DRC Atlas references of the centroidal MPC, wired as CentroidalMpcRobotSim wires them - reference manager, gait
 * schedule, swing trajectories, target calculator, procedural motion manager, and under contact planning the planner
 * module - around an ocs2::mpc_test::ScriptedMpc instead of the SQP solver. Every run() of that MPC goes through
 * SolverBase::run(), so the reference manager and the modules run exactly as they do before a real solve, and a reset of
 * the MPC resets them exactly as MPC_BASE::reset() does in the robot's controller. No CppAD model is built.
 */
class AtlasReferenceStack {
 public:
  enum class ScheduleSource { kGaitSchedule, kContactPlanner };

  /**
   * Under the contact planner, `plannerType` names the planner (ContactPlannerFactory, `planner.type`); empty: the one
   * the shipped file selects. Another one than the shipped runs on the grid and the solver limits
   * testContactPlanningIntegration gives the mixed-integer planner, with a solve-time limit no search reaches, so that
   * the node limit and not the speed of the machine ends every search and two stacks plan alike.
   */
  explicit AtlasReferenceStack(ScheduleSource scheduleSource = ScheduleSource::kGaitSchedule, absl::string_view plannerType = "");

  /** The typed DRC Atlas files the stack is built from: its task, reference and contact planner's file. */
  const CentroidalMpcConfig& config() const { return config_; }
  const ModelSettings& modelSettings() const { return *modelSettings_; }
  const PinocchioInterface& pinocchioInterface() const { return *pinocchioInterface_; }
  const CentroidalModelInfo& centroidalModelInfo() const { return info_; }
  const CentroidalMpcRobotModel<scalar_t>& model() const { return *model_; }
  /** The task file's initial_state: the robot standing at the origin. */
  const vector_t& initialState() const { return initialState_; }
  scalar_t horizon() const { return mpcSettings_.timeHorizon_; }

  SwitchedModelReferenceManager& referenceManager() { return *referenceManager_; }
  /** The reference manager, shared, for a component that keeps it as the stack's own ones do. */
  std::shared_ptr<SwitchedModelReferenceManager> referenceManagerPtr() const { return referenceManager_; }
  /** Null under the gait schedule. */
  ContactPlanningReferenceManager* absl_nullable planningReferenceManager() { return planningReferenceManager_.get(); }
  ContactPlannerModule* absl_nullable contactPlannerModule() { return contactPlannerModule_.get(); }
  ProceduralMpcMotionManager& motionManager() { return *motionManager_; }
  CentroidalMpcTargetTrajectoriesCalculator& targetCalculator() { return *targetCalculator_; }
  mpc_test::ScriptedMpc& mpc() { return *mpc_; }

  /** The velocity command of the operator: [v_x, v_y] in m/s and the yaw rate in rad/s, before the command limits. */
  void command(scalar_t forward, scalar_t lateral = 0.0, scalar_t yawRate = 0.0);

  /** The target the controllers reset the MPC to: `state` held still and upright, weight on both feet. */
  TargetTrajectories resetTarget(scalar_t time, const vector_t& state) const;

  /** `state` moved to base position (x, y) and heading `yaw`, standing still. */
  vector_t standingAt(const vector_t& state, scalar_t x, scalar_t y, scalar_t yaw) const;

  /** The paths of the DRC Atlas files in the runfiles: the task, reference and gait textprotos and the URDF. */
  const std::string& taskFile() const { return taskFile_; }
  const std::string& referenceFile() const { return referenceFile_; }
  const std::string& urdfFile() const { return urdfFile_; }
  const std::string& gaitFile() const { return gaitFile_; }

 private:
  std::string taskFile_;
  std::string referenceFile_;
  std::string urdfFile_;
  std::string gaitFile_;
  CentroidalMpcConfig config_;

  mpc::Settings mpcSettings_;
  std::unique_ptr<ModelSettings> modelSettings_;
  std::unique_ptr<PinocchioInterface> pinocchioInterface_;
  CentroidalModelInfo info_;
  std::unique_ptr<CentroidalMpcRobotModel<scalar_t>> model_;
  vector_t initialState_;

  std::shared_ptr<SwitchedModelReferenceManager> referenceManager_;
  std::shared_ptr<ContactPlanningReferenceManager> planningReferenceManager_;
  std::shared_ptr<ContactPlannerModule> contactPlannerModule_;
  std::unique_ptr<CentroidalMpcTargetTrajectoriesCalculator> targetCalculator_;
  std::shared_ptr<ProceduralMpcMotionManager> motionManager_;
  std::unique_ptr<mpc_test::ScriptedMpc> mpc_;
};

}  // namespace ocs2::humanoid
