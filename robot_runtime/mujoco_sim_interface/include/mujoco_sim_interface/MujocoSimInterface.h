/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.

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

#include <iostream>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <atomic>
#include <chrono>
#include <ctime>
#include <thread>

#include <Eigen/Dense>

#include <robot_model/RobotState.h>
#include "mujoco_sim_interface/MujocoRenderer.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "robot_core/FPSTracker.h"
#include "robot_core/TripleBuffer.h"
#include "robot_core/Types.h"
#include "robot_model/RobotHWInterfaceBase.h"

namespace robot::mujoco_sim_interface {

struct MujocoSimConfig {
  std::string scenePath;
  std::shared_ptr<model::RobotState> initStatePtr_;
  double dt{0.0005};
  double renderFrequencyHz{60.0};
  bool headless{false};
  bool verbose{false};
  bool enableGantry{true};
  bool isGantryLocked{true};
  double gantryHeight{0.0};
};

class MujocoSimInterface : public robot::model::RobotHWInterfaceBase {
 public:
  MujocoSimInterface(const MujocoSimConfig& config, const std::string& urdfPath);

  /** Destructor */
  ~MujocoSimInterface();

  void initSim();

  void startSim();

  void simulationStep();

  // Todo Manu also reset environment
  void reset();

  // Virtual Gantry Controls
  void lockGantry() { isGantryLocked_ = true; }
  void unlockGantry() { isGantryLocked_ = false; }
  double stepGantry(double delta) {
    gantryHeight_ = gantryHeight_.load() + delta;
    return gantryHeight_.load();
  }
  void setGantryHeight(double height) { gantryHeight_ = height; }
  bool isGantryLocked() const { return isGantryLocked_.load(); }
  double getGantryHeight() const { return gantryHeight_.load(); }

  /// Zero-torque mode: when enabled, all actuator commands are zeroed in simulationStep().
  /// The sim starts in zero-torque mode by default to allow the MPC solver to warm up.
  /// enableTorques/disableTorques also swap MuJoCo's dof_damping for smooth ragdoll behavior.
  bool isZeroTorqueMode() const { return zeroTorqueMode_.load(); }
  void enableTorques();
  void disableTorques();

  // Allows the renderer to read the latest sim state without blocking the sim thread.
  // Uses a lock-free triple buffer internally.
  void readLatestMjState(MjState& state) const;

  const mjModel* getModel() const { return mujocoModel_; }

  const MujocoSimConfig& getConfig() const { return config_; }

 private:
  void setupJointIndexMaps();

  void setSimState(const model::RobotState& robotState);

  void updateThreadSafeRobotState();

  void simulationLoop();

  void printModelInfo();

  void updateMetrics();

  MujocoSimConfig config_;

  model::RobotState robotStateInternal_;
  mjtNum* qpos_init_;  // position                                         (nq x 1)
  mjtNum* qvel_init_;
  model::RobotJointAction robotJointActionInternal_;

  size_t timeStepMicro_;
  double simStart_;
  size_t nActiveJoints_;
  size_t nActuators_;
  std::vector<std::string> activeMuJoCoJointNames_;
  std::vector<std::string> activeMuJoCoActuatorNames_;
  std::vector<joint_index_t> activeRobotJointStateIndices_;
  std::vector<joint_index_t> activeRobotActuatorIndices_;

  mjModel* mujocoModel_ = NULL;
  mjData* mujocoData_ = NULL;
  mjContact* mujocoContact_ = NULL;
  // mjfSensor mujocoSenor_;

  bool simInit_;
  const bool headless_;
  const bool verbose_;
  std::atomic<bool> terminate_{false};
  std::atomic<bool> guiInitialized_{false};

  std::thread simulate_thread_;
  std::unique_ptr<MujocoRenderer> renderer_;

  FPSTracker simFps_{"mujoco_sim", 0.02};
  std::chrono::steady_clock::time_point lastRealTime_;
  std::chrono::steady_clock::time_point loopStartTime_;
  double simTimeAtLoopStart_{0.0};
  Metrics metrics_{};

  size_t right_foot_sensor_addr_;
  size_t left_foot_sensor_addr_;

  size_t right_foot_touch_sensor_addr_;
  size_t left_foot_touch_sensor_addr_;

  std::atomic<bool> isGantryLocked_{true};
  std::atomic<double> gantryHeight_{0.0};
  std::atomic<bool> zeroTorqueMode_{true};  // Start in zero-torque mode by default
  std::vector<mjtNum> originalDofDamping_;  // Saved dof_damping values for restore on enableTorques

  /// Lock-free triple buffer for sim→render state transfer.
  /// Initialized lazily after MuJoCo model is loaded (requires mjModel* for MjState allocation).
  std::unique_ptr<TripleBuffer<MjState>> renderStateBuffer_;

  /// Throttle: only publish to the triple buffer every N sim steps (matches render Hz).
  size_t renderPublishInterval_{1};
  size_t renderPublishCounter_{0};
};

}  // namespace robot::mujoco_sim_interface
