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

#include "mujoco_sim_interface/MujocoSimInterface.h"

#include <cmath>

#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace robot::mujoco_sim_interface {

MjState::MjState(const mjModel* model) : model(model), data(mj_makeData(model)) {}

MjState::MjState(const MjState& other)
    : model(other.model), timestamp(other.timestamp), data(other.model ? mj_makeData(other.model) : nullptr), metrics(other.metrics) {
  if (data && other.data && model) {
    mj_copyData(data, model, other.data);
  }
}

MjState& MjState::operator=(const MjState& other) {
  if (this != &other) {
    if (data) mj_deleteData(data);
    model = other.model;
    timestamp = other.timestamp;
    data = other.model ? mj_makeData(other.model) : nullptr;
    metrics = other.metrics;
    if (data && other.data && model) {
      mj_copyData(data, model, other.data);
    }
  }
  return *this;
}

MjState::MjState(MjState&& other) noexcept : model(other.model), timestamp(other.timestamp), data(other.data), metrics(other.metrics) {
  other.data = nullptr;
  other.model = nullptr;
}

MjState& MjState::operator=(MjState&& other) noexcept {
  if (this != &other) {
    if (data) mj_deleteData(data);
    model = other.model;
    timestamp = other.timestamp;
    data = other.data;
    metrics = other.metrics;
    other.data = nullptr;
    other.model = nullptr;
  }
  return *this;
}

MjState::~MjState() {
  if (data) mj_deleteData(data);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

MujocoSimInterface::MujocoSimInterface(const MujocoSimConfig& config, const std::string& urdfPath)
    : RobotHWInterfaceBase(urdfPath),
      config_(config),
      robotStateInternal_(model::RobotState(this->getRobotDescription())),
      robotJointActionInternal_(model::RobotJointAction(this->getRobotDescription())),
      headless_(config.headless),
      verbose_(config.verbose) {
  lastRealTime_ = std::chrono::steady_clock::now();
  const int errstr_sz = 1000;  // Define the size of the error buffer
  char errstr[errstr_sz];      // Declare the error string buffer

  // option 1: parse and compile XML from file
  mujocoModel_ = mj_loadXML(config.scenePath.c_str(), NULL, errstr, errstr_sz);
  if (!mujocoModel_) {
    std::cerr << "Could not load MuJoCo model: " << config.scenePath << ". Error: " << errstr << std::endl;
    throw std::runtime_error("Could not load MuJoCo: " + std::string(errstr));
  }

  // Create data
  mujocoData_ = mj_makeData(mujocoModel_);

  /* initialize random seed: */
  srand(time(NULL));

  mujocoContact_ = mujocoData_->contact;

  simStart_ = mujocoData_->time;

  // assert(nActiveJoints_ == neo_definitions::FULL_NEO_JOINT_DIM);

  mujocoModel_->opt.timestep = config_.dt;

  timeStepMicro_ = static_cast<size_t>(config_.dt * 1000000);

  if (verbose_) printModelInfo();

  setupJointIndexMaps();

  model::RobotState initRobotState(getRobotDescription(), 2);

  if (config_.initStatePtr_ != nullptr) {
    initRobotState = *config.initStatePtr_;
  } else {
    initRobotState.setConfigurationToZero();
    initRobotState.setRootPositionInWorldFrame(vector3_t(0.0, 0.0, 1.0));
  }
  setSimState(initRobotState);

  // Add default joint damping
  scalar_t defaultJointDamping = 10.0;

  for (int i = 6; i < mujocoModel_->nv; ++i) {
    std::string mjJointName(&mujocoModel_->names[mujocoModel_->name_jntadr[mujocoModel_->dof_jntid[i]]]);
    std::cerr << "mjJointName: " << mjJointName << std::endl;
    mujocoModel_->dof_damping[i] = defaultJointDamping;
  }

  for (int i = 0; i < mujocoModel_->nsensor; i++) {
    std::string sensorName(&mujocoModel_->names[mujocoModel_->name_sensoradr[i]]);

    if (sensorName == "right_foot_touch_sensor") {
      right_foot_touch_sensor_addr_ = mujocoModel_->sensor_adr[i];
    }
    if (sensorName == "left_foot_touch_sensor") {
      left_foot_touch_sensor_addr_ = mujocoModel_->sensor_adr[i];
    }
    if (sensorName == "right_foot_force_sensor") {
      right_foot_sensor_addr_ = mujocoModel_->sensor_adr[i];
    }
    if (sensorName == "left_foot_force_sensor") {
      left_foot_sensor_addr_ = mujocoModel_->sensor_adr[i];
    }
  }

  setupContactDetection();

  qpos_init_ = new mjtNum[mujocoModel_->nq];
  qvel_init_ = new mjtNum[mujocoModel_->nv];

  if (config_.gantryHeight > 0.0) {
    gantryHeight_ = config_.gantryHeight;
  } else if (config_.initStatePtr_) {
    gantryHeight_ = config_.initStatePtr_->getRootPositionInWorldFrame().z();
  } else {
    gantryHeight_ = mujocoData_->qpos[2];
  }
  isGantryLocked_ = config_.isGantryLocked;

  // Safe init state for resets
  memcpy(qpos_init_, mujocoData_->qpos, mujocoModel_->nq * sizeof(mjtNum));
  memcpy(qvel_init_, mujocoData_->qvel, mujocoModel_->nv * sizeof(mjtNum));

  // Make sure the init state is propagated throughout the RobotInterface.
  updateThreadSafeRobotState();
  updateInterfaceStateFromRobot();

  // Save original dof_damping and boost for zero-torque ragdoll mode at startup.
  originalDofDamping_.assign(mujocoModel_->dof_damping, mujocoModel_->dof_damping + mujocoModel_->nv);
  // Boost damping for smooth ragdoll settling (skip root 6 DOFs)
  for (int i = 6; i < mujocoModel_->nv; ++i) {
    mujocoModel_->dof_damping[i] = 20.0;
  }

  // Initialize lock-free triple buffer for sim→render state transfer.
  // Each slot holds an MjState with its own mjData copy.
  MjState initState(mujocoModel_);
  mj_copyData(initState.data, mujocoModel_, mujocoData_);
  initState.timestamp = mujocoData_->time;
  initState.metrics = metrics_;
  renderStateBuffer_ = std::make_unique<TripleBuffer<MjState>>(initState);

  // Throttle: publish to triple buffer at render frequency, not every sim step.
  // E.g. dt=0.0005 (2000 Hz sim), renderFrequencyHz=60 → publish every ~33 steps.
  if (config_.renderFrequencyHz > 0.0 && config_.dt > 0.0) {
    renderPublishInterval_ = std::max(static_cast<size_t>(1), static_cast<size_t>(1.0 / (config_.renderFrequencyHz * config_.dt)));
  }
}

/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::enableTorques() {
  // Restore original dof_damping for MPC active control
  if (!originalDofDamping_.empty()) {
    for (int i = 0; i < mujocoModel_->nv; ++i) {
      mujocoModel_->dof_damping[i] = originalDofDamping_[i];
    }
  }
  zeroTorqueMode_ = false;
}

void MujocoSimInterface::disableTorques() {
  // Boost joint damping for smooth ragdoll settling
  if (originalDofDamping_.empty()) {
    originalDofDamping_.assign(mujocoModel_->dof_damping, mujocoModel_->dof_damping + mujocoModel_->nv);
  }
  for (int i = 6; i < mujocoModel_->nv; ++i) {
    mujocoModel_->dof_damping[i] = 20.0;
  }
  zeroTorqueMode_ = true;
}

/******************************************************************************************************/

MujocoSimInterface::~MujocoSimInterface() {
  terminate_.store(true);
  if (simulate_thread_.joinable()) simulate_thread_.join();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::reset() {
  memcpy(mujocoData_->qpos, qpos_init_, mujocoModel_->nq * sizeof(mjtNum));
  memcpy(mujocoData_->qvel, qvel_init_, mujocoModel_->nv * sizeof(mjtNum));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::readLatestMjState(MjState& state) const {
  // Lock-free: acquire the latest published state from the triple buffer.
  // If new data is available, the internal read↔clean swap happens atomically.
  // Either way, copy from the current read slot into the caller's state.
  renderStateBuffer_->acquireRead();
  const MjState& latest = renderStateBuffer_->readSlot();
  state.timestamp = latest.timestamp;
  mj_copyData(state.data, mujocoModel_, latest.data);
  state.metrics = latest.metrics;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::setupJointIndexMaps() {
  // Mujoco to Robot joints
  for (int i = 1; i < mujocoModel_->njnt; ++i) {
    // Get the joint name
    const std::string jointName(&mujocoModel_->names[mujocoModel_->name_jntadr[i]]);
    if (getRobotDescription().containsJoint(jointName)) {
      activeMuJoCoJointNames_.emplace_back(jointName);
    } else {
      std::cerr << "WARNING: Joint contained in mujoco xml not exposed to RobotHWInterface: " << jointName << std::endl;
    }
  }

  activeRobotJointStateIndices_ = getRobotDescription().getJointIndices(activeMuJoCoJointNames_);

  // Mujoco to robot actuators
  // NOTE: MuJoCo actuator names (e.g. "back_bkz_motor") do NOT match robot description joint
  // names (e.g. "back_bkz"). We resolve the joint driven by each actuator via actuator_trnid
  // so that the robot description lookup succeeds. Only joint-type actuators (trntype == mjTRN_JOINT)
  // are considered; slide/site actuators are skipped.
  for (int i = 0; i < mujocoModel_->nu; ++i) {
    const std::string actuator_name = mj_id2name(mujocoModel_, mjOBJ_ACTUATOR, i);

    // Resolve the joint name driven by this actuator.
    std::string driven_joint_name;
    if (mujocoModel_->actuator_trntype[i] == mjTRN_JOINT) {
      const int jnt_id = mujocoModel_->actuator_trnid[2 * i];  // (nu x 2): [2*i]=primary target id, [2*i+1]=secondary
      const char* jnt_name_cstr = mj_id2name(mujocoModel_, mjOBJ_JOINT, jnt_id);
      if (jnt_name_cstr != nullptr) {
        driven_joint_name = jnt_name_cstr;
      }
    }

    if (!driven_joint_name.empty() && getRobotDescription().containsJoint(driven_joint_name)) {
      // Store the driven joint name so getJointIndices resolves correctly.
      activeMuJoCoActuatorNames_.emplace_back(driven_joint_name);
    } else {
      std::cerr << "WARNING: Actuator contained in mujoco xml not be commanded through RobotHWInterface: " << actuator_name << std::endl;
    }
  }

  activeRobotActuatorIndices_ = getRobotDescription().getJointIndices(activeMuJoCoActuatorNames_);

  nActiveJoints_ = activeRobotJointStateIndices_.size();
  nActuators_ = activeRobotActuatorIndices_.size();
  if (verbose_) {
    std::cerr << "Initialized " << nActiveJoints_ << " active Joints" << std::endl;
    std::cerr << "Initialized " << nActuators_ << " active Actuators" << std::endl;
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::printModelInfo() {
  std::cerr << "timeStepMicro_: " << timeStepMicro_ << std::endl;

  std::cerr << "njnt: " << mujocoModel_->njnt << std::endl;
  std::cerr << "nq: " << mujocoModel_->nq << std::endl;
  std::cerr << "nv: " << mujocoModel_->nv << std::endl;
  std::cerr << "nu: " << mujocoModel_->nu << std::endl;

  for (int i = 0; i < mujocoModel_->nbody; ++i) {
    std::string bodyName(&mujocoModel_->names[mujocoModel_->name_bodyadr[i]]);
    std::cerr << "Body " << i << ": " << bodyName << std::endl;

    std::cerr << "  Position: ";
    for (size_t j = 0; j < 3; ++j) {
      std::cerr << mujocoData_->xpos[i * 3 + j] << " ";
    }
    std::cerr << std::endl;

    // Print orientation quaternion
    std::cerr << "  Orientation (Quaternion): ";
    for (size_t j = 0; j < 4; ++j) {
      std::cerr << mujocoData_->xquat[i * 4 + j] << " ";
    }
    std::cerr << std::endl;
  }

  std::string jointName(&mujocoModel_->names[mujocoModel_->name_jntadr[0]]);

  // Print the information
  std::cerr << "Joint Name: " << jointName << std::endl;
  std::cerr << "Position: " << mujocoData_->qpos[0] << " " << mujocoData_->qpos[1] << " " << mujocoData_->qpos[2] << " "
            << mujocoData_->qpos[3] << " " << mujocoData_->qpos[4] << " " << mujocoData_->qpos[5] << " " << mujocoData_->qpos[6]
            << std::endl;
  std::cerr << "Velocity: " << mujocoData_->qvel[0] << " " << mujocoData_->qvel[1] << " " << mujocoData_->qvel[2] << " "
            << mujocoData_->qvel[3] << " " << mujocoData_->qvel[4] << " " << mujocoData_->qvel[5] << std::endl;

  // Print joint names, positions, and velocities
  for (int i = 1; i < mujocoModel_->njnt; ++i) {
    // Get the joint name
    std::string jointName(&mujocoModel_->names[mujocoModel_->name_jntadr[i]]);

    // Get the joint position and velocity
    double jointPos = mujocoData_->qpos[i + 6];
    double jointVel = mujocoData_->qvel[i + 5];

    // Print the information
    std::cerr << "Joint Name: " << jointName << ", Position: " << jointPos << ", Velocity: " << jointVel << std::endl;
  }

  // Calculate total mass
  scalar_t totalMass = 0.0;
  for (int i = 0; i < mujocoModel_->nbody; i++) {
    totalMass += mujocoModel_->body_mass[i];
  }
  std::cerr << "Total MuJoCo model mass: " << totalMass << std::endl;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::setSimState(const model::RobotState& robotState) {
  // Root Pose
  vector3_t rootPosition = robotState.getRootPositionInWorldFrame();
  quaternion_t quat_l_to_w = robotState.getRootRotationLocalToWorldFrame();

  mujocoData_->qpos[0] = rootPosition[0];
  mujocoData_->qpos[1] = rootPosition[1];
  mujocoData_->qpos[2] = rootPosition[2];
  mujocoData_->qpos[3] = quat_l_to_w.w();
  mujocoData_->qpos[4] = quat_l_to_w.x();
  mujocoData_->qpos[5] = quat_l_to_w.y();
  mujocoData_->qpos[6] = quat_l_to_w.z();

  // Root Velocity

  vector3_t root_vel_lin_world_frame = quat_l_to_w * robotState.getRootLinearVelocityInLocalFrame();
  vector3_t root_vel_ang_local_frame = robotState.getRootAngularVelocityInLocalFrame();

  mujocoData_->qvel[0] = root_vel_lin_world_frame[0];
  mujocoData_->qvel[1] = root_vel_lin_world_frame[1];
  mujocoData_->qvel[2] = root_vel_lin_world_frame[2];
  mujocoData_->qvel[3] = root_vel_ang_local_frame[0];
  mujocoData_->qvel[4] = root_vel_ang_local_frame[1];
  mujocoData_->qvel[5] = root_vel_ang_local_frame[2];

  // Joint State
  for (size_t i = 0; i < nActiveJoints_; ++i) {
    mujocoData_->qpos[i + 7] = robotState.getJointPosition(activeRobotJointStateIndices_[i]);
    mujocoData_->qvel[i + 6] = robotState.getJointVelocity(activeRobotJointStateIndices_[i]);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::updateThreadSafeRobotState() {
  // Update mujoco joint angles
  for (size_t i = 0; i < nActiveJoints_; ++i) {
    robotStateInternal_.setJointPosition(activeRobotJointStateIndices_[i], mujocoData_->qpos[i + 7]);
    robotStateInternal_.setJointVelocity(activeRobotJointStateIndices_[i], mujocoData_->qvel[i + 6]);
  }

  // Initialize in order w, x,y ,z
  quaternion_t quat_l_to_w = quaternion_t(mujocoData_->qpos[3], mujocoData_->qpos[4], mujocoData_->qpos[5], mujocoData_->qpos[6]);
  vector3_t pelvisAngularVelLocal = vector3_t(mujocoData_->qvel[3], mujocoData_->qvel[4], mujocoData_->qvel[5]);

  // Contact flags handed to the controller. By default every contact point is reported as touching (the historical
  // behaviour: the controller then treats its own schedule as the measured contact state). With
  // reportGroundTruthContacts the physics decides, see updateGroundTruthContacts(); a contact point whose MuJoCo body
  // could not be resolved keeps the default.
  const bool useGroundTruth = config_.reportGroundTruthContacts && hasContactDetection();
  const uint32_t touching = groundTruthContactMask_.load() | unresolvedContactMask_;

  robotStateInternal_.setRootPositionInWorldFrame(vector3_t(mujocoData_->qpos[0], mujocoData_->qpos[1], mujocoData_->qpos[2]));
  robotStateInternal_.setRootRotationLocalToWorldFrame(quat_l_to_w);
  // Rotate the angular velocity from world frame to local frame.
  robotStateInternal_.setRootLinearVelocityInLocalFrame(quat_l_to_w.inverse() *
                                                        vector3_t(mujocoData_->qvel[0], mujocoData_->qvel[1], mujocoData_->qvel[2]));
  robotStateInternal_.setRootAngularVelocityInLocalFrame(pelvisAngularVelLocal);
  const size_t numFlags = robotStateInternal_.getContactFlags().size();
  for (size_t i = 0; i < numFlags; ++i) {
    const bool flag = !useGroundTruth || i >= contactBodyIds_.size() || ((touching >> i) & 1u) != 0u;
    robotStateInternal_.setContactFlag(i, flag);
  }

  robotStateInternal_.setTime(mujocoData_->time);  // Todo Manu: should mujoco be the source of time?

  threadSafeRobotState_.set(robotStateInternal_);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::updateMetrics() {
  simFps_.tick();

  metrics_.fpsSim = simFps_.fps();

  auto nowRealTime = std::chrono::steady_clock::now();
  auto realElapsedTime = std::chrono::duration<double>(nowRealTime - lastRealTime_).count();
  lastRealTime_ = nowRealTime;

  metrics_.driftTick = config_.dt - realElapsedTime;
  metrics_.driftCumulative += metrics_.driftTick;

  metrics_.rtfTick = (realElapsedTime > 1e-7) ? (config_.dt / realElapsedTime) : 1.0;

  // Window-based RTF: total sim time elapsed / total wall time elapsed.
  // This is the true real-time factor, immune to per-tick scheduling noise.
  double wallElapsed = std::chrono::duration<double>(nowRealTime - loopStartTime_).count();
  double simElapsed = mujocoData_->time - simTimeAtLoopStart_;
  metrics_.rtfSmoothed = (wallElapsed > 0.1) ? (simElapsed / wallElapsed) : 1.0;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::simulationStep() {
  if (zeroTorqueMode_.load()) {
    // Zero-torque / ragdoll mode: zero all actuator commands.
    // Joint damping is handled by MuJoCo's native dof_damping (boosted on entry).
    for (int i = 0; i < mujocoModel_->nu; ++i) {
      mujocoData_->ctrl[i] = 0.0;
    }
  } else {
    threadSafeRobotJointAction_.copy_value(robotJointActionInternal_);
    for (size_t i = 0; i < nActuators_; ++i) {
      joint_index_t idx = activeRobotActuatorIndices_[i];
      const robot::model::JointAction& jointAction = robotJointActionInternal_.at(idx).value();
      double totalTorque =
          jointAction.getTotalFeedbackTorque(robotStateInternal_.getJointPosition(idx), robotStateInternal_.getJointVelocity(idx));

      // Clamp total torque to the actuator force range from the MuJoCo model.
      // This enforces physical actuator limits on the combined PD + feedforward torque,
      // preventing the unclamped PD term from exceeding actuator capabilities.
      if (mujocoModel_->actuator_forcelimited[i]) {
        totalTorque =
            std::clamp(totalTorque, (double)mujocoModel_->actuator_forcerange[2 * i], (double)mujocoModel_->actuator_forcerange[2 * i + 1]);
      } else {
        // Fallback: use the joint's actuatorfrcrange if the actuator itself isn't force-limited.
        // Look up the joint index from the actuator's transmission target.
        int mj_joint_id = mujocoModel_->actuator_trnid[2 * i];
        if (mj_joint_id >= 0 && mujocoModel_->jnt_actfrclimited[mj_joint_id]) {
          totalTorque = std::clamp(totalTorque, (double)mujocoModel_->jnt_actfrcrange[2 * mj_joint_id],
                                   (double)mujocoModel_->jnt_actfrcrange[2 * mj_joint_id + 1]);
        }
      }
      mujocoData_->ctrl[i] = totalTorque;
    }
  }

  // Apply virtual gantry constraint if locked to suspend robot at ground-touch stance height
  if (config_.enableGantry && isGantryLocked_.load()) {
    mujocoData_->qpos[0] = 0.0;
    mujocoData_->qpos[1] = 0.0;
    mujocoData_->qpos[2] = gantryHeight_.load();
    mujocoData_->qpos[3] = 1.0;
    mujocoData_->qpos[4] = 0.0;
    mujocoData_->qpos[5] = 0.0;
    mujocoData_->qpos[6] = 0.0;

    mujocoData_->qvel[0] = 0.0;
    mujocoData_->qvel[1] = 0.0;
    mujocoData_->qvel[2] = 0.0;
    mujocoData_->qvel[3] = 0.0;
    mujocoData_->qvel[4] = 0.0;
    mujocoData_->qvel[5] = 0.0;
  }

  mj_step(mujocoModel_, mujocoData_);
  updateGroundTruthContacts();
  updateThreadSafeRobotState();
  updateMetrics();

  // Publish state to the lock-free triple buffer for the render thread.
  // Throttled to render frequency to avoid unnecessary mj_copyData overhead.
  if (++renderPublishCounter_ >= renderPublishInterval_) {
    renderPublishCounter_ = 0;
    MjState& writeSlot = renderStateBuffer_->writeSlot();
    writeSlot.timestamp = mujocoData_->time;
    mj_copyData(writeSlot.data, mujocoModel_, mujocoData_);
    writeSlot.metrics = metrics_;
    renderStateBuffer_->publishWrite();
  }

  // Auto reset logic.
  if (mujocoData_->qpos[2] < 0.2) {
    reset();
    for (size_t i = 0; i < nActuators_; ++i) {
      mujocoData_->ctrl[i] = 0.0;
    }
    mj_step(mujocoModel_, mujocoData_);
    updateThreadSafeRobotState();
    simFps_.reset();
    metrics_.reset();
    auto resetNow = std::chrono::steady_clock::now();
    lastRealTime_ = resetNow;
    loopStartTime_ = resetNow;
    simTimeAtLoopStart_ = mujocoData_->time;
    updateMetrics();

    // Publish reset state to triple buffer
    {
      MjState& writeSlot = renderStateBuffer_->writeSlot();
      writeSlot.timestamp = mujocoData_->time;
      mj_copyData(writeSlot.data, mujocoModel_, mujocoData_);
      writeSlot.metrics = metrics_;
      renderStateBuffer_->publishWrite();
    }

    // Sleep to let controller update and adjust;
    std::this_thread::sleep_until(std::chrono::steady_clock::now() + std::chrono::microseconds(1000000));
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::simulationLoop() {
  simFps_.reset();
  metrics_.reset();
  auto now = std::chrono::steady_clock::now();
  lastRealTime_ = now;
  loopStartTime_ = now;
  simTimeAtLoopStart_ = mujocoData_->time;
  auto nextWakeup = now;
  while (!terminate_.load()) {
    simulationStep();

    // Advance the wakeup target by one sim timestep.
    nextWakeup += std::chrono::microseconds(timeStepMicro_);

    auto now = std::chrono::steady_clock::now();
    // Allow up to 10ms of catchup budget for minor OS scheduling jitter.
    // If we fell behind by more than 10ms (e.g. auto-reset sleep), rebase nextWakeup.
    if (now - nextWakeup > std::chrono::milliseconds(10)) {
      nextWakeup = now;
    } else if (nextWakeup > now) {
      // Sleep until 50µs before target, then busy-spin for sub-microsecond precision.
      auto spinThreshold = nextWakeup - std::chrono::microseconds(50);
      if (std::chrono::steady_clock::now() < spinThreshold) {
        std::this_thread::sleep_until(spinThreshold);
      }
      while (std::chrono::steady_clock::now() < nextWakeup) {
        // Busy spin
      }
    }
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::initSim() {
  simulationStep();
  simInit_ = true;

  if (!headless_) {
    renderer_.reset(new MujocoRenderer(this));
    renderer_->launchRenderThread();
  }
}

void MujocoSimInterface::startSim() {
  if (!simInit_) initSim();
  // Simulate in simulate_thread thread while rendering in this thread
  simulate_thread_ = std::thread(&MujocoSimInterface::simulationLoop, this);
}

void MujocoSimInterface::setupContactDetection() {
  contactBodyIds_.clear();
  unresolvedContactMask_ = 0;
  if (config_.contactFrameNames.empty()) return;
  if (config_.contactFrameNames.size() > 32) {
    std::cerr << "[MujocoSimInterface] contact detection supports at most 32 contact points, got " << config_.contactFrameNames.size()
              << "; the detection stays off." << std::endl;
    return;
  }
  std::vector<std::string> errors;
  contactBodyIds_ = resolveContactBodies(mujocoModel_, getRobotDescription().getURDFPath(), config_.contactFrameNames,
                                         config_.contactParentJointNames, &errors);
  for (const std::string& error : errors) {
    std::cerr << "[MujocoSimInterface] contact detection: " << error << std::endl;
  }
  for (size_t i = 0; i < contactBodyIds_.size(); ++i) {
    if (contactBodyIds_[i] < 0) {
      unresolvedContactMask_ |= (1u << i);
    } else if (verbose_) {
      std::cerr << "[MujocoSimInterface] contact point '" << config_.contactFrameNames[i] << "' -> MuJoCo body '"
                << mj_id2name(mujocoModel_, mjOBJ_BODY, contactBodyIds_[i]) << "'" << std::endl;
    }
  }
  // The timeline is sampled at a fixed rate; the physics step is finer.
  constexpr double kTimelineSampleRateHz = 500.0;
  const double timestep = mujocoModel_->opt.timestep > 0.0 ? mujocoModel_->opt.timestep : config_.dt;
  contactTimelineSampleInterval_ = std::max<size_t>(1, static_cast<size_t>(std::lround(1.0 / (kTimelineSampleRateHz * timestep))));
  contactTimeline_ = ContactTimeline(config_.contactTimelineWindow);
  if (config_.reportGroundTruthContacts) {
    std::cerr << "[MujocoSimInterface] reporting ground-truth contacts to the controller (normal force > " << config_.contactForceThreshold
              << " N)." << std::endl;
  }
}

void MujocoSimInterface::updateGroundTruthContacts() {
  if (contactBodyIds_.empty()) return;
  const uint32_t actual = groundTruthContactMask(mujocoModel_, mujocoData_, contactBodyIds_, config_.contactForceThreshold);
  groundTruthContactMask_.store(actual);
  if (++contactTimelineSampleCounter_ < contactTimelineSampleInterval_) return;
  contactTimelineSampleCounter_ = 0;
  ContactTimelineSample sample;
  sample.time = mujocoData_->time;
  sample.actual = actual;
  sample.target = targetContactMask_.load();
  sample.targetKnown = targetContactKnown_.load();
  std::lock_guard<std::mutex> lock(contactTimelineMutex_);
  contactTimeline_.append(sample);
}

void MujocoSimInterface::setTargetContactFlags(const std::vector<bool>& flags) {
  uint32_t mask = 0;
  for (size_t i = 0; i < flags.size() && i < 32; ++i) {
    if (flags[i]) mask |= (1u << i);
  }
  targetContactMask_.store(mask);
  targetContactKnown_.store(!flags.empty());
}

std::vector<bool> MujocoSimInterface::getGroundTruthContactFlags() const {
  const uint32_t mask = groundTruthContactMask_.load();
  std::vector<bool> flags(contactBodyIds_.size(), false);
  for (size_t i = 0; i < flags.size(); ++i) {
    flags[i] = ((mask >> i) & 1u) != 0u;
  }
  return flags;
}

void MujocoSimInterface::copyContactTimeline(std::vector<ContactTimelineSample>& out) const {
  std::lock_guard<std::mutex> lock(contactTimelineMutex_);
  out.assign(contactTimeline_.samples().begin(), contactTimeline_.samples().end());
}

vector3_t MujocoSimInterface::getLeftFootMeasuredForce() const {
  if (left_foot_sensor_addr_ != static_cast<size_t>(-1) && mujocoData_ != nullptr) {
    return vector3_t(mujocoData_->sensordata[left_foot_sensor_addr_], mujocoData_->sensordata[left_foot_sensor_addr_ + 1],
                     mujocoData_->sensordata[left_foot_sensor_addr_ + 2]);
  }
  return vector3_t::Zero();
}

vector3_t MujocoSimInterface::getRightFootMeasuredForce() const {
  if (right_foot_sensor_addr_ != static_cast<size_t>(-1) && mujocoData_ != nullptr) {
    return vector3_t(mujocoData_->sensordata[right_foot_sensor_addr_], mujocoData_->sensordata[right_foot_sensor_addr_ + 1],
                     mujocoData_->sensordata[right_foot_sensor_addr_ + 2]);
  }
  return vector3_t::Zero();
}

}  // namespace robot::mujoco_sim_interface
