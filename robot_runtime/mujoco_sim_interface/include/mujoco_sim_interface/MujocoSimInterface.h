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

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <optional>

#include <thread>
#include <vector>
#include "mujoco_sim_interface/Projectile.h"

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
 * kKinematicTeleport is the legacy behavior and is unphysical: it overwrites qpos and qvel immediately BEFORE mj_step
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
  /// Name of the projectile compiled into the scene, from the robot task file's `simProjectile`. Empty compiles the
  /// scene exactly as it is on disk, with no ball in it. See Projectile.h for the names.
  std::string projectile{};
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

  /**
   * Puts the robot back in its initial state, on the simulation thread's data. The simulation clock is NOT rewound:
   * every controller reading this simulator keys its plans on that clock, and a clock that jumps backwards leaves them
   * timestamped in the future (it is what MuJoCo's own automatic reset did, see mjDSBL_AUTORESET in simulationStep()).
   * Cancels any dodgeball in play and counts the reset in resetEpoch().
   */
  void reset();

  /**
   * How many times the simulator has put the robot back in its initial state: reset(), the automatic reset when the
   * base drops below the floor limit, and the recovery from a numerically unstable step. A change is a discontinuity of
   * the plant that no controller can observe from the state alone, so a control loop compares this against the value it
   * saw in its PREVIOUS cycle and resets its controller when it moves (SimFallRecovery in humanoid_common_mpc_app/robot).
   * Both automatic resets also lock the gantry, but they do it on the simulation thread at any moment of the control
   * cycle, and when the gantry was locked already they change nothing else a loop could see: watching the lock alone
   * misses them, and only this count reports every one.
   */
  uint64_t resetEpoch() const { return resetEpoch_.load(); }

  /**
   * Overwrites the floating base's pose (position, then the w-x-y-z quaternion) and velocity (linear, then angular).
   * For tests and tools that step the simulation themselves: it writes mjData without synchronization, so it must not
   * be called while the thread startSim() started is running.
   */
  void setBaseStateForTesting(const std::array<double, 7>& basePose, const std::array<double, 6>& baseVelocity);

  // Virtual Gantry Controls
  /**
   * Catches the robot on the gantry, WHERE IT IS: with the `weld_constraint` hold the weld holds the base at its
   * horizontal position and heading at the moment of the catch, at the gantry height, rather than at the scene's
   * anchor. A weld to a fixed anchor pulled a robot caught a long walk from the origin across the world, fast enough to
   * make the step numerically unstable. (The legacy `kinematic_teleport` hold still teleports to the origin; it is kept
   * only to reproduce recorded runs.)
   * Locking it also cancels any dodgeball still in play - the ball is parked and a
   * scheduled impulse is dropped - because this is how both the fall catch and the operator's LOCK_GANTRY hold the
   * robot, and a throw that lands on a robot that has just been caught is a disturbance nobody aimed at it. Only the
   * unlocked-to-locked transition cancels, so repeated calls cannot eat throws aimed at a robot already on the gantry.
   * Callable from any thread; the cancellation is carried out on the simulation thread.
   */
  void lockGantry() {
    if (!isGantryLocked_.exchange(true)) cancelDodgeballRequested_.store(true);
  }
  void unlockGantry() { isGantryLocked_ = false; }
  double stepGantry(double delta) {
    gantryHeight_ = gantryHeight_.load() + delta;
    return gantryHeight_.load();
  }
  void setGantryHeight(double height) { gantryHeight_ = height; }
  bool isGantryLocked() const { return isGantryLocked_.load(); }
  double getGantryHeight() const { return gantryHeight_.load(); }

  /**
   * Zero-torque mode: while it is on, simulationStep() commands no actuator. The sim starts in it, so that the MPC
   * solver can warm up. Switching it swaps MuJoCo's dof_damping as well (the boosted ragdoll damping while the torques
   * are off).
   *
   * Callable from any thread, and realtime-safe: the switch is an atomic, and the physics thread writes dof_damping
   * before its next step, so that no other thread writes the model while mj_step reads it. enableTorques() also discards
   * the joint action latched from before the torques went off (RobotHWInterfaceBase::discardAppliedJointAction()): the
   * actuators stay at zero until the next applyJointAction(), instead of executing an action of another mode for up to
   * a control period.
   */
  bool isZeroTorqueMode() const { return zeroTorqueMode_.load(); }
  void enableTorques();
  void disableTorques();

  // Allows the renderer to read the latest sim state without blocking the sim thread.
  // Uses a lock-free triple buffer internally.
  void readLatestMjState(MjState& state) const;

  const mjModel* getModel() const { return mujocoModel_; }
  /// Body id of the thrown ball, or -1 when `simProjectile` compiled none into this scene. For the viewer markers, which
  /// must not mistake a ball touching the robot for the ground.
  int projectileBodyId() const { return dodgeballBodyId_; }

  const MujocoSimConfig& getConfig() const { return config_; }

  /**
   * The forces the feet's force sensors measured in the newest physics step [N] (zero without a sensor), as the
   * physics thread published them. For one consumer thread (the robot process's realtime loop); lock-free, and it reads
   * no mjData, which the physics thread writes meanwhile.
   */
  void takeMeasuredFootForces(vector3_t& left, vector3_t& right);

  /**
   * The actuator commands (mjData::ctrl) of the last step. For tests that step the simulation themselves: it reads mjData
   * without synchronization, so it must not be called while the thread startSim() started is running.
   */
  std::vector<double> actuatorControlsForTesting() const;

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
  /// The same, as bits (bit i: contact point i touches), and the number of contact points it covers. Lock-free and
  /// allocation-free, for the control thread.
  uint32_t getGroundTruthContactMask() const { return groundTruthContactMask_.load(); }
  size_t getNumDetectedContactPoints() const { return contactBodyIds_.size(); }
  /// Snapshot of the timeline for the render thread, oldest sample first.
  void copyContactTimeline(std::vector<ContactTimelineSample>& out) const;

  /// Target contact pose of every contact point from the control thread (MujocoContactPatch.h); an empty vector or
  /// invalid entries clear the viewer's patches.
  void setTargetContactPatches(const std::vector<TargetContactPatch>& patches);
  /// Snapshot of the target patches for the render thread.
  void copyTargetContactPatches(std::vector<TargetContactPatch>& out) const;

  /**
   * One dodgeball, as the operator's GUI describes it: a spawn point and a launch velocity in the robot's own YAW
   * FRAME, the flight time to the base, and the ball's mass.
   *
   * The geometry is computed in the GUI and NOT recomputed here - see
   * humanoid_nmpc/remote_control/remote_control/tk_app/dodgeball.py, which owns the angle conventions and the
   * gravity compensation and is unit-tested. This struct is the wire format between the two.
   */
  struct DodgeballThrow {
    double spawnOffset[3]{0.0, 0.0, 0.0};     // [m] from the base, in the base's yaw frame
    double launchVelocity[3]{0.0, 0.0, 0.0};  // [m/s] in the base's yaw frame
    double flightTime{0.0};                   // [s] until it reaches the base
    double mass{0.0};                         // [kg] always sent by the GUI; clamped to kMin/kMaxProjectileMass on use
  };

  /**
   * Throws a dodgeball at the robot's base. Callable from one thread at a time (the robot process's realtime loop);
   * lock-free, and takes effect on the simulation thread.
   *
   * TWO PATHS, decided by whether `simProjectile` named a ball for this scene.
   *
   * WITH A BALL, which is the shipped configuration: the ball is a real free-floating body, appended to the scene
   * before MuJoCo compiled it. This retunes its mass to the one the operator asked for, places it at the spawn point
   * with the launch velocity - slid along its own path if that point is inside the robot or the floor, see
   * clearProjectileLaunch - and lets MuJoCo do the rest: the flight, the impact, the bounce and the roll to rest are
   * physics rather than a model of physics. What made a body in these scenes awkward is handled rather than avoided:
   * its dofs are skipped by every joint-damping write (setRobotJointDamping), its contacts are excluded from
   * `groundTruthContactMask` and from the ZMP marker so a ball against a swing foot is never mistaken for the ground,
   * and it is appended LAST so it cannot displace the robot as the first free-joint body.
   *
   * WITHOUT ONE: the flight is ballistic and known in closed form, so the momentum the ball carries on arrival,
   * m (v_launch - g t z), is scheduled for `flightTime` from now and applied to the base as a one-step external force.
   * It is a LOWER bound on the push rather than an equivalent of it: a ball that bounces off hands over up to (1 + e)
   * times its momentum. The force lasts one physics step, so the viewer, which samples the state about every thirty
   * steps, will usually not draw it.
   *
   * `mass` is clamped with clampProjectileMass on both paths, because a topic can be published by hand.
   */
  void throwDodgeball(const DodgeballThrow& throwCommand);

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
  /// Simulation thread only: carries out a requested cancellation, picks up a staged throw and launches the ball (or
  /// schedules the fallback impulse), parks a ball that has finished, and applies or clears the impulse. See
  /// throwDodgeball() for the two paths.
  void applyDodgeball();
  /// Simulation thread only: parks the ball and drops any staged throw and scheduled impulse. See lockGantry().
  void cancelDodgeball();
  /// True when a projectile was compiled into this scene, i.e. `simProjectile` named one.
  bool hasProjectile() const;
  /// Arms the ball (it collides, gravity acts on it) or parks it (neither).
  void setProjectileArmed(bool armed);
  /// Returns the ball to kProjectileParkPosition, disarmed and at rest.
  void parkProjectile();
  /// Body id of the robot's floating base, i.e. the first body carrying a free joint; -1 if the model has none.
  int robotRootBodyId() const;
  /// [rad] Heading of that base, from its free joint's quaternion at qpos[3..6].
  double baseYaw() const;

  void setupJointIndexMaps();

  void setSimState(const model::RobotState& robotState);

  void updateThreadSafeRobotState();

  /// Simulation thread only: brings dof_damping in line with the torque switch, when it changed since the last step.
  void applyTorqueSwitch();

  /// Simulation thread only: the force a foot force sensor at `sensorAddress` reads; zero for kNoSensor.
  vector3_t footSensorForce(size_t sensorAddress) const;

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
  /// Simulation thread only: whether the weld has been anchored at the base's pose for the current lock. Cleared on
  /// every unlock and every reset, so the next lock anchors at wherever the robot is then.
  bool gantryWeldAnchored_{false};
  /// See resetEpoch().
  std::atomic<uint64_t> resetEpoch_{0};
  /// Simulation thread only: MuJoCo's count of each state warning after the previous step, to tell a new one apart.
  int lastBadStateWarnings_{0};
  /// Puts the robot back in its initial state and catches it on the gantry; `reason` is logged. Simulation thread only.
  void resetAndCatch(absl::string_view reason);
  /// True when the step just taken produced a state MuJoCo flagged as bad, or one that is not finite.
  bool stepWentUnstable();
  std::atomic<bool> zeroTorqueMode_{true};  // Start in zero-torque mode by default
  /// Simulation thread only: whether dof_damping holds the ragdoll damping of zero-torque mode (applyTorqueSwitch()).
  bool ragdollDampingApplied_{true};
  std::vector<mjtNum> originalDofDamping_;  // Saved dof_damping values for restore on enableTorques

  /// The feet's measured forces of the newest step, from the simulation thread to takeMeasuredFootForces().
  struct MeasuredFootForces {
    vector3_t left = vector3_t::Zero();
    vector3_t right = vector3_t::Zero();
  };
  TripleBuffer<MeasuredFootForces> footForceBuffer_;

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
  /// Staged by throwDodgeball(), taken by the simulation thread on its next step: the newest throw replaces one that
  /// was not taken yet. Lock-free, so that the realtime loop never waits for the physics thread. Separate from the FSM
  /// command's own staging so that a throw and a mode change cannot overwrite one another.
  TripleBuffer<DodgeballThrow> dodgeballMailbox_;
  /// Resolved into the world frame and scheduled once the simulation thread has seen it. Only that thread touches
  /// these, so they need no lock.
  double scheduledImpulseWorld_[3]{0.0, 0.0, 0.0};  // [N s]
  double scheduledImpactTime_{-1.0};                // [s] of simulation time; negative means nothing is in flight
  /// Set for exactly the step on which the impulse is applied, so the next step can clear xfrc_applied again.
  bool dodgeballImpulseApplied_{false};
  /// The projectile compiled into the scene, and its addresses in the model. All -1 when `simProjectile` named
  /// none, in which case a throw falls back to a scheduled impulse on the base and there is nothing to look at.
  Projectile projectile_;
  int dodgeballBodyId_{-1};
  int dodgeballJointId_{-1};
  int dodgeballQposAdr_{-1};
  int dodgeballDofAdr_{-1};
  bool projectileArmed_{false};
  /// Decides when an armed ball has finished and may be parked again.
  ProjectileRestMonitor projectileRestMonitor_;
  /// Set by lockGantry() on any thread, carried out by applyDodgeball() on the simulation thread.
  std::atomic<bool> cancelDodgeballRequested_{false};
  /// The name the injected body carries in the compiled model.
  static constexpr const char* kProjectileBodyName = "sim_projectile";
  size_t contactTimelineSampleInterval_{1};
  size_t contactTimelineSampleCounter_{0};
};

}  // namespace robot::mujoco_sim_interface
