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

#include "mujoco_sim_interface/MujocoSimInterface.h"

#include <cmath>

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace robot::mujoco_sim_interface {

namespace {
/// [m/s^2] Gravity the dodgeball's flight was computed under. It matches the default the GUI uses
/// (remote_control/tk_app/dodgeball.py GRAVITY) rather than mjModel::opt.gravity, because the two have to agree for
/// the impact velocity to be the one the GUI aimed with, and the GUI cannot see the model.
constexpr double kGravity = 9.81;
}  // namespace

namespace {
/// Name of the weld equality a scene declares for the virtual gantry.
constexpr absl::string_view kGantryWeldName = "gantry";
/// Index of the relpose translation z within an mjEQ_WELD eq_data row, whose layout is
/// [anchor(3), relpose position(3), relpose quaternion(4), torquescale(1)].
constexpr int kWeldRelposeZOffset = 5;
}  // namespace

absl::StatusOr<GantryHold> gantryHoldFromName(absl::string_view name) {
  if (name == "weld_constraint") return GantryHold::kWeldConstraint;
  if (name == "kinematic_teleport") return GantryHold::kKinematicTeleport;
  return absl::InvalidArgumentError(absl::StrCat("Unknown gantryHold '", name,
                                                 "'. Set gantryHold in the robot task file to one of: weld_constraint, "
                                                 "kinematic_teleport."));
}

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

  // The scene is compiled with a ball in it when the task file named a projectile, and exactly as it is on disk when
  // it did not. Going through mjSpec rather than editing the XML keeps the robot's own scene file untouched.
  if (config_.projectile.empty()) {
    mujocoModel_ = mj_loadXML(config.scenePath.c_str(), NULL, errstr, errstr_sz);
    if (!mujocoModel_) {
      LOG(ERROR) << "Could not load MuJoCo model: " << config.scenePath << ". Error: " << errstr;
      throw std::runtime_error("Could not load MuJoCo: " + std::string(errstr));
    }
  } else {
    const absl::StatusOr<Projectile> projectile = projectileFromName(config_.projectile);
    if (!projectile.ok()) {
      LOG(ERROR) << projectile.status().message();
      throw std::runtime_error(std::string(projectile.status().message()));
    }
    projectile_ = *projectile;
    mjSpec* spec = mj_parseXML(config.scenePath.c_str(), NULL, errstr, errstr_sz);
    if (spec == nullptr) {
      LOG(ERROR) << "Could not parse MuJoCo model: " << config.scenePath << ". Error: " << errstr;
      throw std::runtime_error("Could not parse MuJoCo: " + std::string(errstr));
    }
    const absl::Status added = addProjectileToSpec(spec, projectile_, kProjectileBodyName);
    if (!added.ok()) {
      mj_deleteSpec(spec);
      LOG(ERROR) << added.message();
      throw std::runtime_error(std::string(added.message()));
    }
    mujocoModel_ = mj_compile(spec, nullptr);
    if (mujocoModel_ == nullptr) {
      const std::string compileError(mjs_getError(spec));
      mj_deleteSpec(spec);
      LOG(ERROR) << "Could not compile MuJoCo model with the '" << projectile_.name << "' projectile: " << compileError;
      throw std::runtime_error("Could not compile MuJoCo: " + compileError);
    }
    mj_deleteSpec(spec);
  }

  // Create data
  mujocoData_ = mj_makeData(mujocoModel_);

  // Where the ball lives in the compiled model. Resolved once, here, because everything downstream - the damping
  // loops that must skip its dofs, the contact mask that must not mistake it for the ground, and the throw itself -
  // is written in terms of these indices.
  if (!config_.projectile.empty()) {
    dodgeballBodyId_ = mj_name2id(mujocoModel_, mjOBJ_BODY, kProjectileBodyName);
    if (dodgeballBodyId_ >= 0 && mujocoModel_->body_jntnum[dodgeballBodyId_] > 0) {
      dodgeballJointId_ = mujocoModel_->body_jntadr[dodgeballBodyId_];
      dodgeballQposAdr_ = mujocoModel_->jnt_qposadr[dodgeballJointId_];
      dodgeballDofAdr_ = mujocoModel_->jnt_dofadr[dodgeballJointId_];
    }
    if (!hasProjectile()) {
      LOG(ERROR) << "The '" << projectile_.name << "' projectile was compiled into the scene but could not be found in "
                 << "the model; throws will fall back to applying its impulse to the base.";
    } else {
      parkProjectile();
      // What the compiled model actually carries, so the first throw at the slider's default mass is free.
      appliedProjectileMass_ = mujocoModel_->body_mass[dodgeballBodyId_];
      LOG(INFO) << "Projectile '" << projectile_.name << "' ready: " << 2.0 * projectile_.radius << " m across, " << appliedProjectileMass_
                << " kg, restitution " << projectile_.restitution << ", mass adjustable from " << kMinProjectileMass << " to "
                << kMaxProjectileMass << " kg per throw.";
    }
  }

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
    if (isProjectileDof(i)) continue;  // the ball's free joint is not one of the robot's actuated joints
    std::string mjJointName(&mujocoModel_->names[mujocoModel_->name_jntadr[mujocoModel_->dof_jntid[i]]]);
    LOG(INFO) << "mjJointName: " << mjJointName;
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

  const absl::StatusOr<GantryHold> gantryHold = gantryHoldFromName(config_.gantryHold);
  if (!gantryHold.ok()) {
    throw std::invalid_argument(std::string(gantryHold.status().message()));
  }
  gantryHold_ = *gantryHold;
  gantryWeldEqId_ = mj_name2id(mujocoModel_, mjOBJ_EQUALITY, std::string(kGantryWeldName).c_str());
  if (gantryHold_ == GantryHold::kWeldConstraint && gantryWeldEqId_ < 0) {
    LOG(ERROR) << "gantryHold is 'weld_constraint' but the scene " << config_.scenePath << " declares no equality named '"
               << kGantryWeldName << "'. Add <equality><weld name=\"" << kGantryWeldName
               << "\" body1=\"world\" body2=\"<base body>\" relpose=\"0 0 <height> 1 0 0 0\" active=\"false\"/></equality> "
                  "to the scene. Falling back to 'kinematic_teleport', which leaves the base unsupported inside mj_step "
                  "and makes gravity compensation drive the limbs into their stops.";
    gantryHold_ = GantryHold::kKinematicTeleport;
  }
  if (gantryHold_ == GantryHold::kWeldConstraint) {
    // The weld carries the height, so the model's initial relpose is irrelevant; eq_active is driven every step.
    mujocoModel_->eq_active0[gantryWeldEqId_] = static_cast<mjtByte>(isGantryLocked_.load());
  }
  if (config_.verbose) {
    LOG(INFO) << "Virtual gantry base hold: " << (gantryHold_ == GantryHold::kWeldConstraint ? "weld_constraint" : "kinematic_teleport")
              << " (gantryHold).";
  }

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
    if (isProjectileDof(i)) continue;  // the thrown ball is not a joint to be settled; damping it makes it fall in syrup
    mujocoModel_->dof_damping[i] = 20.0;
  }
  zeroTorqueMode_ = true;
}

/******************************************************************************************************/

MujocoSimInterface::~MujocoSimInterface() {
  terminate_.store(true);
  if (simulate_thread_.joinable()) simulate_thread_.join();
  // The viewer has to go before any member it reads. renderer_ is declared ahead of renderStateBuffer_, the contact
  // timeline and the two mutexes, so destruction in reverse declaration order would tear those down while
  // MujocoRenderer::renderLoop is still running - a use-after-free on every shutdown that the render thread loses.
  // ~MujocoRenderer joins that thread, so resetting it here orders the teardown correctly.
  renderer_.reset();
  if (mujocoData_ != nullptr) mj_deleteData(mujocoData_);
  if (mujocoModel_ != nullptr) mj_deleteModel(mujocoModel_);
  delete[] qpos_init_;
  delete[] qvel_init_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::reset() {
  memcpy(mujocoData_->qpos, qpos_init_, mujocoModel_->nq * sizeof(mjtNum));
  memcpy(mujocoData_->qvel, qvel_init_, mujocoModel_->nv * sizeof(mjtNum));

  // Any dodgeball in flight is cancelled with the rest of the state. A reset is how the gantry catches a fallen
  // robot, and an impulse scheduled before the fall arriving on the freshly caught robot would look exactly like the
  // recovery having failed - a disturbance nobody threw at the robot that is now standing there.
  {
    std::lock_guard<std::mutex> lock(dodgeballMutex_);
    pendingDodgeball_.reset();
  }
  scheduledImpactTime_ = -1.0;
  // A ball in flight is put back in its box too: reset() is how the gantry catches a fallen robot, and a ball still
  // bouncing around the freshly caught robot is a disturbance nobody threw at it.
  parkProjectile();
  if (dodgeballImpulseApplied_ && robotRootBodyId() >= 0) {
    for (int axis = 0; axis < 3; ++axis) {
      mujocoData_->xfrc_applied[6 * robotRootBodyId() + axis] = 0.0;
    }
  }
  dodgeballImpulseApplied_ = false;
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
      LOG(WARNING) << "Joint contained in mujoco xml not exposed to RobotHWInterface: " << jointName;
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
      LOG(WARNING) << "Actuator contained in mujoco xml not be commanded through RobotHWInterface: " << actuator_name;
    }
  }

  activeRobotActuatorIndices_ = getRobotDescription().getJointIndices(activeMuJoCoActuatorNames_);

  nActiveJoints_ = activeRobotJointStateIndices_.size();
  nActuators_ = activeRobotActuatorIndices_.size();
  if (verbose_) {
    LOG(INFO) << "Initialized " << nActiveJoints_ << " active Joints";
    LOG(INFO) << "Initialized " << nActuators_ << " active Actuators";
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::printModelInfo() {
  LOG(INFO) << "timeStepMicro_: " << timeStepMicro_;

  LOG(INFO) << "njnt: " << mujocoModel_->njnt;
  LOG(INFO) << "nq: " << mujocoModel_->nq;
  LOG(INFO) << "nv: " << mujocoModel_->nv;
  LOG(INFO) << "nu: " << mujocoModel_->nu;

  for (int i = 0; i < mujocoModel_->nbody; ++i) {
    std::string bodyName(&mujocoModel_->names[mujocoModel_->name_bodyadr[i]]);
    LOG(INFO) << "Body " << i << ": " << bodyName;

    LOG(INFO) << "  Position: ";
    for (size_t j = 0; j < 3; ++j) {
      LOG(INFO) << mujocoData_->xpos[i * 3 + j] << " ";
    }
    LOG(INFO);

    // Print orientation quaternion
    LOG(INFO) << "  Orientation (Quaternion): ";
    for (size_t j = 0; j < 4; ++j) {
      LOG(INFO) << mujocoData_->xquat[i * 4 + j] << " ";
    }
    LOG(INFO);
  }

  std::string jointName(&mujocoModel_->names[mujocoModel_->name_jntadr[0]]);

  // Print the information
  LOG(INFO) << "Joint Name: " << jointName;
  LOG(INFO) << "Position: " << mujocoData_->qpos[0] << " " << mujocoData_->qpos[1] << " " << mujocoData_->qpos[2] << " "
            << mujocoData_->qpos[3] << " " << mujocoData_->qpos[4] << " " << mujocoData_->qpos[5] << " " << mujocoData_->qpos[6];
  LOG(INFO) << "Velocity: " << mujocoData_->qvel[0] << " " << mujocoData_->qvel[1] << " " << mujocoData_->qvel[2] << " "
            << mujocoData_->qvel[3] << " " << mujocoData_->qvel[4] << " " << mujocoData_->qvel[5];

  // Print joint names, positions, and velocities
  for (int i = 1; i < mujocoModel_->njnt; ++i) {
    // Get the joint name
    std::string jointName(&mujocoModel_->names[mujocoModel_->name_jntadr[i]]);

    // Get the joint position and velocity
    double jointPos = mujocoData_->qpos[i + 6];
    double jointVel = mujocoData_->qvel[i + 5];

    // Print the information
    LOG(INFO) << "Joint Name: " << jointName << ", Position: " << jointPos << ", Velocity: " << jointVel;
  }

  // Calculate total mass
  scalar_t totalMass = 0.0;
  for (int i = 0; i < mujocoModel_->nbody; i++) {
    totalMass += mujocoModel_->body_mass[i];
  }
  LOG(INFO) << "Total MuJoCo model mass: " << totalMass;
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

  // Contact flags handed to the controller: the ground truth of the physics, see updateGroundTruthContacts(). Without
  // contact detection (no contact frame names configured), and for a contact point whose MuJoCo body could not be
  // resolved, the point is reported as touching. Which contact state the controller actually uses is the choice of
  // its contact estimator (task file `contactEstimator`).
  const bool useGroundTruth = hasContactDetection();
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

  // Suspend the robot at the gantry height, by whichever implementation the task file named.
  if (config_.enableGantry) {
    applyGantryHold();
  }

  // A dodgeball in flight lands as a one-step external force on the base. Free when none is in flight.
  applyDodgeball();

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
    LOG(INFO) << "[MujocoSimInterface] contact detection supports at most 32 contact points, got " << config_.contactFrameNames.size()
              << "; the detection stays off.";
    return;
  }
  std::vector<std::string> errors;
  contactBodyIds_ = resolveContactBodies(mujocoModel_, getRobotDescription().getURDFPath(), config_.contactFrameNames,
                                         config_.contactParentJointNames, &errors);
  for (const std::string& error : errors) {
    LOG(INFO) << "[MujocoSimInterface] contact detection: " << error;
  }
  for (size_t i = 0; i < contactBodyIds_.size(); ++i) {
    if (contactBodyIds_[i] < 0) {
      unresolvedContactMask_ |= (1u << i);
    } else if (verbose_) {
      LOG(INFO) << "[MujocoSimInterface] contact point '" << config_.contactFrameNames[i] << "' -> MuJoCo body '"
                << mj_id2name(mujocoModel_, mjOBJ_BODY, contactBodyIds_[i]) << "'";
    }
  }
  // The timeline is sampled at a fixed rate; the physics step is finer.
  constexpr double kTimelineSampleRateHz = 500.0;
  const double timestep = mujocoModel_->opt.timestep > 0.0 ? mujocoModel_->opt.timestep : config_.dt;
  contactTimelineSampleInterval_ = std::max<size_t>(1, static_cast<size_t>(std::lround(1.0 / (kTimelineSampleRateHz * timestep))));
  contactTimeline_ = ContactTimeline(config_.contactTimelineWindow);
  if (verbose_) {
    LOG(INFO) << "[MujocoSimInterface] ground-truth contact detection: normal force > " << config_.contactForceThreshold << " N.";
  }
}

void MujocoSimInterface::applyGantryHold() {
  const bool locked = isGantryLocked_.load();

  if (gantryHold_ == GantryHold::kWeldConstraint) {
    // A real constraint: mj_step solves it, so the base is supported DURING the integration rather than corrected
    // afterwards. The height is carried in the weld's relpose, which the operator can move while the sim runs.
    mujocoData_->eq_active[gantryWeldEqId_] = static_cast<mjtByte>(locked);
    mujocoModel_->eq_data[mjNEQDATA * gantryWeldEqId_ + kWeldRelposeZOffset] = gantryHeight_.load();
    return;
  }

  // Legacy kinematic teleport. Unphysical - see the GantryHold comment - and kept only to reproduce recorded runs.
  if (!locked) return;
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

void MujocoSimInterface::updateGroundTruthContacts() {
  if (contactBodyIds_.empty()) return;
  const uint32_t actual =
      groundTruthContactMask(mujocoModel_, mujocoData_, contactBodyIds_, config_.contactForceThreshold, dodgeballBodyId_);
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

void MujocoSimInterface::throwDodgeball(const DodgeballThrow& throwCommand) {
  std::lock_guard<std::mutex> lock(dodgeballMutex_);
  // A second throw before the first has been picked up replaces it rather than queueing: the button is a one-shot
  // and an operator pressing it twice in one control cycle means "throw now", not "throw twice".
  pendingDodgeball_ = throwCommand;
}

bool MujocoSimInterface::hasDodgeballInFlight() const {
  std::lock_guard<std::mutex> lock(dodgeballMutex_);
  return pendingDodgeball_.has_value() || scheduledImpactTime_ >= 0.0;
}

void MujocoSimInterface::applyDodgeball() {
  // Simulation thread only. mjData has no lock - this thread owns it - so everything that touches qpos, qvel or
  // xfrc_applied happens here rather than in the ROS callback that staged the command.

  // The impulse from the previous step has done its work; take it off again, or one throw would become a constant
  // force. Only the no-ball fallback path below ever sets it.
  if (dodgeballImpulseApplied_) {
    const int previousBase = robotRootBodyId();
    if (previousBase >= 0) {
      for (int axis = 0; axis < 3; ++axis) {
        mujocoData_->xfrc_applied[6 * previousBase + axis] = 0.0;
      }
    }
    dodgeballImpulseApplied_ = false;
  }

  std::optional<DodgeballThrow> staged;
  {
    std::lock_guard<std::mutex> lock(dodgeballMutex_);
    staged.swap(pendingDodgeball_);
  }

  if (staged.has_value()) {
    // The base pose is read HERE rather than when the command was staged: the ball is thrown at the robot as it
    // stands at the moment of the throw, and a robot that turns during the flight does not drag the ball with it.
    const int baseBody = robotRootBodyId();
    const double yaw = baseYaw();
    const double cosYaw = std::cos(yaw);
    const double sinYaw = std::sin(yaw);
    const double flight = std::max(staged->flightTime, 0.0);

    // The GUI computed the whole throw in the robot's own yaw frame, so rotating it into the world is all that is
    // left to do here - see remote_control/tk_app/dodgeball.py, which owns the geometry.
    const double offsetWorld[3] = {cosYaw * staged->spawnOffset[0] - sinYaw * staged->spawnOffset[1],
                                   sinYaw * staged->spawnOffset[0] + cosYaw * staged->spawnOffset[1], staged->spawnOffset[2]};
    const double velocityWorld[3] = {cosYaw * staged->launchVelocity[0] - sinYaw * staged->launchVelocity[1],
                                     sinYaw * staged->launchVelocity[0] + cosYaw * staged->launchVelocity[1], staged->launchVelocity[2]};

    if (hasProjectile() && baseBody >= 0) {
      // A REAL BALL. Put it at the spawn point with the launch velocity and let MuJoCo fly it: the flight, the
      // impact and the bounce are then the physics rather than a model of it, which is the whole point of having a
      // body in the scene at all.
      //
      // The operator's mass first, and ONLY when it has changed: retuning it rebuilds the model's mass-derived
      // constants, which costs about twenty simulation steps. A sweep that varies the direction and leaves the mass
      // alone pays nothing.
      const double requestedMass = std::clamp(staged->mass, kMinProjectileMass, kMaxProjectileMass);
      if (std::abs(requestedMass - appliedProjectileMass_) > 1e-9) {
        const absl::Status retuned = setProjectileMass(mujocoModel_, dodgeballBodyId_, requestedMass, projectile_.radius);
        if (retuned.ok()) {
          appliedProjectileMass_ = requestedMass;
        } else {
          // The throw still happens, at whatever mass the ball already had. Saying so is the point: a disturbance
          // that silently weighed something other than what the slider said would invalidate the whole experiment.
          LOG(ERROR) << "Could not retune the dodgeball to " << requestedMass << " kg (" << retuned.message() << "); throwing it at "
                     << appliedProjectileMass_ << " kg instead.";
        }
      }

      const double* basePosition = &mujocoData_->xpos[3 * baseBody];
      for (int axis = 0; axis < 3; ++axis) {
        mujocoData_->qpos[dodgeballQposAdr_ + axis] = basePosition[axis] + offsetWorld[axis];
      }
      // Identity orientation. A sphere has no preferred one, and leaving whatever the last throw ended at would make
      // a re-thrown ball's spin depend on where the previous one came to rest.
      mujocoData_->qpos[dodgeballQposAdr_ + 3] = 1.0;
      mujocoData_->qpos[dodgeballQposAdr_ + 4] = 0.0;
      mujocoData_->qpos[dodgeballQposAdr_ + 5] = 0.0;
      mujocoData_->qpos[dodgeballQposAdr_ + 6] = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        mujocoData_->qvel[dodgeballDofAdr_ + axis] = velocityWorld[axis];
        mujocoData_->qvel[dodgeballDofAdr_ + 3 + axis] = 0.0;  // no spin: the GUI does not aim one
      }
      setProjectileArmed(true);
      LOG(INFO) << "Dodgeball thrown: " << appliedProjectileMass_ << " kg, " << 2.0 * projectile_.radius << " m across, arriving in "
                << flight << " s with " << appliedProjectileMass_ * std::hypot(velocityWorld[0], velocityWorld[1], velocityWorld[2])
                << " N s.";
      return;
    }

    // NO BALL IN THE SCENE, because the task file named no projectile. The flight is ballistic and known in closed
    // form, so the momentum it would have delivered is scheduled as a one-step force on the base instead. Nothing to
    // look at, but the disturbance is the same one. v_impact = v_launch - (0, 0, g t), because the launch was lifted
    // by g t / 2 to beat the drop.
    for (int axis = 0; axis < 3; ++axis) {
      scheduledImpulseWorld_[axis] = staged->mass * velocityWorld[axis];
    }
    scheduledImpulseWorld_[2] -= staged->mass * kGravity * flight;
    scheduledImpactTime_ = mujocoData_->time + flight;
    LOG(INFO) << "Dodgeball thrown (no projectile compiled into this scene, so its impulse is simulated instead): " << staged->mass
              << " kg arriving in " << flight << " s with an impulse of (" << scheduledImpulseWorld_[0] << ", " << scheduledImpulseWorld_[1]
              << ", " << scheduledImpulseWorld_[2] << ") N s.";
  }

  // A ball that has come to rest is parked again, so that it stops being something the robot can trip over and the
  // next throw starts from a clean state rather than from wherever the last one rolled to.
  if (projectileArmed_ && projectileAtRest()) {
    parkProjectile();
    return;
  }

  if (scheduledImpactTime_ < 0.0 || mujocoData_->time < scheduledImpactTime_) return;
  const int baseBody = robotRootBodyId();
  if (baseBody < 0) {
    scheduledImpactTime_ = -1.0;
    return;
  }

  // xfrc_applied is a FORCE held over the step, so the impulse is divided by the step it is spread across. One step
  // is the shortest a collision can be represented here and is already 1-2 ms, the right order for a foam ball
  // flattening against a torso.
  const double timestep = mujocoModel_->opt.timestep > 0.0 ? mujocoModel_->opt.timestep : 1e-3;
  for (int axis = 0; axis < 3; ++axis) {
    mujocoData_->xfrc_applied[6 * baseBody + axis] = scheduledImpulseWorld_[axis] / timestep;
  }
  dodgeballImpulseApplied_ = true;
  scheduledImpactTime_ = -1.0;
}

bool MujocoSimInterface::hasProjectile() const {
  return dodgeballBodyId_ >= 0 && dodgeballQposAdr_ >= 0 && dodgeballDofAdr_ >= 0;
}

bool MujocoSimInterface::isProjectileDof(int dof) const {
  return dodgeballDofAdr_ >= 0 && dof >= dodgeballDofAdr_ && dof < dodgeballDofAdr_ + 6;
}

void MujocoSimInterface::setProjectileArmed(bool armed) {
  if (!hasProjectile()) return;
  // Armed: it collides, and gravity acts on it. Parked: neither, so a ball nobody has thrown can neither be hit nor
  // fall for the whole session accumulating the velocity MuJoCo eventually warns about.
  setProjectileCollisionEnabled(mujocoModel_, dodgeballBodyId_, armed);
  if (mujocoModel_->body_gravcomp != nullptr) {
    mujocoModel_->body_gravcomp[dodgeballBodyId_] = armed ? 0.0 : 1.0;
  }
  projectileArmed_ = armed;
  projectileRestSteps_ = 0;
}

bool MujocoSimInterface::projectileAtRest() {
  if (!hasProjectile()) return false;
  // "At rest" is a slow ball for a while, not a stationary one: a ball lying on the floor is never exactly still,
  // and one balanced on a foot for an instant on its way past must not be parked out from under the robot.
  double speedSquared = 0.0;
  for (int axis = 0; axis < 3; ++axis) {
    const double component = mujocoData_->qvel[dodgeballDofAdr_ + axis];
    speedSquared += component * component;
  }
  projectileRestSteps_ = (speedSquared < kProjectileRestSpeed * kProjectileRestSpeed) ? projectileRestSteps_ + 1 : 0;
  return projectileRestSteps_ >= kProjectileRestSteps;
}

void MujocoSimInterface::parkProjectile() {
  if (!hasProjectile()) return;
  setProjectileArmed(false);
  mujocoData_->qpos[dodgeballQposAdr_ + 0] = 0.0;
  mujocoData_->qpos[dodgeballQposAdr_ + 1] = 0.0;
  mujocoData_->qpos[dodgeballQposAdr_ + 2] = kProjectileParkHeight;
  mujocoData_->qpos[dodgeballQposAdr_ + 3] = 1.0;
  for (int index = 4; index < 7; ++index) mujocoData_->qpos[dodgeballQposAdr_ + index] = 0.0;
  for (int dof = 0; dof < 6; ++dof) mujocoData_->qvel[dodgeballDofAdr_ + dof] = 0.0;
}

int MujocoSimInterface::robotRootBodyId() const {
  // The FIRST body carrying a free joint, which is the convention MujocoContactUtils::robotCentroidalState already
  // relies on for the centre of mass and the viewer's camera.
  for (int body = 1; body < mujocoModel_->nbody; ++body) {
    if (mujocoModel_->body_jntnum[body] > 0 && mujocoModel_->jnt_type[mujocoModel_->body_jntadr[body]] == mjJNT_FREE) {
      return body;
    }
  }
  return -1;
}

double MujocoSimInterface::baseYaw() const {
  // qpos[3..6] is the base free joint's quaternion (w, x, y, z), the same slice setSimState writes.
  const double w = mujocoData_->qpos[3];
  const double x = mujocoData_->qpos[4];
  const double y = mujocoData_->qpos[5];
  const double z = mujocoData_->qpos[6];
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

void MujocoSimInterface::setTargetContactPatches(const std::vector<TargetContactPatch>& patches) {
  std::lock_guard<std::mutex> lock(targetPatchMutex_);
  targetContactPatches_ = patches;
}

void MujocoSimInterface::copyTargetContactPatches(std::vector<TargetContactPatch>& out) const {
  std::lock_guard<std::mutex> lock(targetPatchMutex_);
  out = targetContactPatches_;
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
