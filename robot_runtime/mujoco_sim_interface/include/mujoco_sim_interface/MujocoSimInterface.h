/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.
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
#include <mutex>
#include <thread>
#include <vector>

#include <Eigen/Dense>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

#include <robot_model/RobotState.h>
#include "mujoco_sim_interface/MujocoContactPatch.h"
#include "mujoco_sim_interface/MujocoRenderer.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/visualization/VisualizationRegistry.h"
#include "robot_core/FPSTracker.h"
#include "robot_core/TripleBuffer.h"
#include "robot_core/Types.h"
#include "robot_model/RobotHWInterfaceBase.h"

namespace robot::mujoco_sim_interface {

/**
 * How the virtual gantry holds the floating base while it is locked.
 *
 * These are two different physical models rather than one feature switched on and off, so the robot task file names the
 * one it wants (`gantryHold`) and gantryHoldFromName resolves it.
 *
 * kWeldConstraint is what a real gantry does: a MuJoCo weld equality that the constraint solver satisfies INSIDE
 * mj_step, so the base is genuinely supported while the dynamics are integrated. A limb then needs exactly its own
 * gravity torque to hover, which is what GRAVITY_COMP commands.
 *
 * kKinematicTeleport is the legacy behaviour and is unphysical: it overwrites qpos and qvel immediately BEFORE mj_step
 * and so leaves the base unsupported during the step itself. The whole robot is then in free fall while integrating,
 * and a free-falling chain in uniform gravity needs ZERO relative joint torque to keep its shape - so the commanded
 * g_j(q) is entirely surplus torque in the lifting direction. Measured on the shipped scenes, that drives the limbs
 * through about 130 degrees in three seconds and into their joint stops, while the weld holds them under 0.06 degrees.
 * Kept only so that recorded runs can be reproduced.
 */
// LINT.IfChange(gantry_hold_names)
enum class GantryHold {
  kWeldConstraint,     ///< "weld_constraint": a real constraint solved inside mj_step (default)
  kKinematicTeleport,  ///< "kinematic_teleport": legacy qpos/qvel overwrite before mj_step, base unsupported during it
};

/// Resolves a task-file `gantryHold` name, naming the valid ones when it does not match.
absl::StatusOr<GantryHold> gantryHoldFromName(absl::string_view name);
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:gantry_hold, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:gantry_hold)
// clang-format on

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
  // Which base-hold implementation the gantry uses, by name (GantryHold above). An unknown name is rejected at start-up.
  std::string gantryHold{"weld_constraint"};

  // Contact points of the controller, in its order (URDF frame or link names). They drive the ground-truth contact
  // detection behind the viewer's contact timeline, the contact flags of the RobotState handed to the controller and
  // the CheaterSimContactEstimator.
  std::vector<std::string> contactFrameNames;
  // Joint carrying each contact frame (same order, may be shorter or hold empty strings). A contact frame that the
  // controller adds to its own kinematic model does not exist in the URDF or the MuJoCo scene; the MuJoCo body driven
  // by that joint is the contact body then.
  std::vector<std::string> contactParentJointNames;
  double contactForceThreshold{5.0};  // [N] normal force above which a contact point counts as touching
  double contactTimelineWindow{5.0};  // [s] sliding window of the contact timeline overlay

  // Contact patch of every contact point (same order) in its contact frame, drawn by the viewer at the target contact
  // pose the controller reports through setTargetContactPatches (MujocoContactPatch.h). A contact point without an
  // entry, or with an empty one, gets a generic outline.
  std::vector<ContactPatchCorners> contactPatchCorners;

  // Visualizations of the viewer, by registry name (task file `simVisualizations`, see VisualizationRegistry.h). Each
  // listed one starts enabled; its hotkey toggles it. Defaults to the historical set of the viewer.
  std::vector<std::string> visualizations = defaultVisualizationNames();
};

class MujocoSimInterface : public robot::model::RobotHWInterfaceBase {
 public:
  MujocoSimInterface(const MujocoSimConfig& config, const std::string& urdfPath);

  /** Destructor */
  ~MujocoSimInterface();

  void initSim();

  void startSim();

  std::thread& getSimulationThread() { return simulate_thread_; }

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

  vector3_t getLeftFootMeasuredForce() const;
  vector3_t getRightFootMeasuredForce() const;

  /// Ground-truth contact detection and the contact timeline (see MujocoUtils.h).
  bool hasContactDetection() const { return !contactBodyIds_.empty(); }
  const std::vector<std::string>& getContactNames() const { return config_.contactFrameNames; }
  /// Bit i set: no MuJoCo body could be resolved for contact point i (it is reported as touching, and drawn as unknown).
  uint32_t getUnresolvedContactMask() const { return unresolvedContactMask_; }
  double getContactTimelineWindow() const { return contactTimeline_.window(); }
  /// Planned contact state from the control thread; an empty vector means "unknown" (no policy yet).
  void setTargetContactFlags(const std::vector<bool>& flags);
  /// Ground truth of the last simulation step, one flag per contact point (false where unresolved).
  std::vector<bool> getGroundTruthContactFlags() const;
  /// Snapshot of the timeline for the render thread, oldest sample first.
  void copyContactTimeline(std::vector<ContactTimelineSample>& out) const;

  /// Target contact pose of every contact point from the control thread (MujocoContactPatch.h); an empty vector or
  /// invalid entries clear the viewer's patches.
  void setTargetContactPatches(const std::vector<TargetContactPatch>& patches);
  /// Snapshot of the target patches for the render thread.
  void copyTargetContactPatches(std::vector<TargetContactPatch>& out) const;

  void setTargetVelocities(double vx, double vy, double yawRate) {
    targetVelocityX_.store(vx, std::memory_order_relaxed);
    targetVelocityY_.store(vy, std::memory_order_relaxed);
    targetYawRate_.store(yawRate, std::memory_order_relaxed);
  }
  double getTargetVelocityX() const { return targetVelocityX_.load(std::memory_order_relaxed); }
  double getTargetVelocityY() const { return targetVelocityY_.load(std::memory_order_relaxed); }
  double getTargetYawRate() const { return targetYawRate_.load(std::memory_order_relaxed); }

 private:
  void setupContactDetection();
  void updateGroundTruthContacts();

  /**
   * Holds the floating base at the gantry height for one step, by whichever implementation gantryHold_ names.
   * Called once per step immediately before mj_step; a no-op when the gantry is disabled.
   */
  void applyGantryHold();

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

  bool simInit_ = false;
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

  // Sensor addresses, set only when the scene defines the sensor; the getters check for the "absent" value. They used
  // to be left uninitialized, which read sensordata at a garbage index (a segfault on scenes without sensors, such as
  // the DRC Atlas one) whenever the object's layout happened to leave a large value there.
  static constexpr size_t kNoSensor = static_cast<size_t>(-1);
  size_t right_foot_sensor_addr_{kNoSensor};
  size_t left_foot_sensor_addr_{kNoSensor};

  size_t right_foot_touch_sensor_addr_{kNoSensor};
  size_t left_foot_touch_sensor_addr_{kNoSensor};

  std::atomic<bool> isGantryLocked_{true};
  std::atomic<double> gantryHeight_{0.0};
  GantryHold gantryHold_{GantryHold::kWeldConstraint};
  /// Index of the scene's "gantry" weld equality, or -1 when the scene declares none.
  int gantryWeldEqId_{-1};
  std::atomic<bool> zeroTorqueMode_{true};  // Start in zero-torque mode by default
  std::vector<mjtNum> originalDofDamping_;  // Saved dof_damping values for restore on enableTorques

  /// Lock-free triple buffer for sim→render state transfer.
  /// Initialized lazily after MuJoCo model is loaded (requires mjModel* for MjState allocation).
  std::unique_ptr<TripleBuffer<MjState>> renderStateBuffer_;

  /// Throttle: only publish to the triple buffer every N sim steps (matches render Hz).
  size_t renderPublishInterval_{1};
  size_t renderPublishCounter_{0};

  /// Ground-truth contact detection (simulation thread) and the timeline read by the renderer.
  std::vector<int> contactBodyIds_;  // MuJoCo body per contact point, -1 if unresolved
  uint32_t unresolvedContactMask_{0};
  std::atomic<double> targetVelocityX_{0.0};  // commanded base velocities, from the control thread (see setTargetVelocities)
  std::atomic<double> targetVelocityY_{0.0};
  std::atomic<double> targetYawRate_{0.0};
  std::atomic<uint32_t> targetContactMask_{0};
  std::atomic<bool> targetContactKnown_{false};
  std::atomic<uint32_t> groundTruthContactMask_{0};
  mutable std::mutex contactTimelineMutex_;
  ContactTimeline contactTimeline_;
  mutable std::mutex targetPatchMutex_;
  std::vector<TargetContactPatch> targetContactPatches_;  // written by the control thread, drawn by the renderer
  size_t contactTimelineSampleInterval_{1};
  size_t contactTimelineSampleCounter_{0};
};

}  // namespace robot::mujoco_sim_interface
