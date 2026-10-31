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

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
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
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/PhysicalConstants.h"

#include "mujoco_sim_interface/MujocoUtils.h"
#include "mujoco_sim_interface/viewer/MujocoViewerRegistry.h"
#include "mujoco_sim_interface/viewer/ViewerNames.h"
#include "mujoco_sim_interface/viewer/ViewerSettings.h"

namespace robot::mujoco_sim_interface {

using ::ocs2::kPositionDim;
using ::ocs2::kXAxisIndex;
using ::ocs2::kYAxisIndex;
using ::ocs2::kZAxisIndex;

namespace {
/// The contact points of the initial state when the configuration gives none:
/// the two feet.
constexpr int kInitialStateContactPoints = 2;
/// [m] The height of the base in the initial state when the configuration gives
/// none.
constexpr double kDefaultInitialBaseHeight = 1.0;
/// The projectile whose restitution the no-ball fallback impulse models: the
/// ball the GUI's Dodgeball tab throws.
constexpr char kFallbackImpactProjectile[] = "dodgeball";

/// [us] How long the physics pauses after an automatic reset, so that the
/// controller can catch up.
constexpr int kAutoResetPauseMicroseconds = 1'000'000;
/// [ms] How far the physics loop may fall behind its schedule (scheduling
/// jitter) and still catch up; beyond it, it starts the schedule over.
constexpr int kMaxCatchUpMilliseconds = 10;
/// [us] How long before a wake-up the physics loop stops sleeping and spins,
/// for a precise step.
constexpr int kBusySpinMicroseconds = 50;
/// [s] The real time per step below which a step's real-time factor is reported
/// as 1 rather than divided by ~0.
constexpr double kMinRealTimePerStep = 1.0e-7;
/// [s] The wall time the smoothed real-time factor waits for before it is
/// measured.
constexpr double kMinRealTimeFactorWindow = 0.1;
/// The samples of the contact timeline beyond those one window spans: one for
/// its second end, one for rounding.
constexpr int kExtraTimelineSamples = 2;
/// [s] Minimum spawn shift along a dodgeball's path to trigger a warning; a
/// shorter one is rounding.
constexpr double kMinReportedSpawnShift = 1.0e-12;
/// [s] The step a dodgeball's fallback impulse is spread over when the model
/// has no timestep.
constexpr double kFallbackImpulseTimestep = 1.0e-3;

/// Six numbers per actuator in actuator_gear (mjmodel.h); the first scales a
/// joint transmission.
constexpr int kActuatorGearSize = 6;
/// Two numbers per actuator in actuator_trnid: its primary transmission target,
/// then a secondary one.
constexpr int kActuatorTransmissionTargets = 2;
/// Two numbers per actuator or joint in a range (actuator_forcerange,
/// jnt_actfrcrange): the lower bound, then the upper.
constexpr int kRangeBounds = 2;
}  // namespace

namespace {
/// Name of the weld equality a scene declares for the virtual gantry.
constexpr absl::string_view kGantryWeldName = "gantry";
/// Indices within an mjEQ_WELD eq_data row, whose layout is [anchor(3), relpose
/// position(3), relpose quaternion(4), torquescale(1)]: the relpose position
/// (x, y, z), then its quaternion (w, x, y, z).
constexpr int kWeldRelposeXIndex = 3;
constexpr int kWeldRelposeYIndex = 4;
constexpr int kWeldRelposeZIndex = 5;
constexpr int kWeldRelposeQuaternionWIndex = 6;
constexpr int kWeldRelposeQuaternionXIndex = 7;
constexpr int kWeldRelposeQuaternionYIndex = 8;
constexpr int kWeldRelposeQuaternionZIndex = 9;
/// The name of the MuJoCo object `id` of type `type`, or "" for an unnamed one.
/// mj_id2name returns null for an object the scene gives no name, such as
/// `<motor joint="..."/>`, and neither std::string nor a log stream accepts
/// null.
std::string mujocoName(const mjModel* absl_nonnull model, mjtObj type, int id) {
  const char* absl_nullable name = mj_id2name(model, type, id);
  return name != nullptr ? std::string(name) : std::string();
}

/// The body of the robot's free joint: the joint at qpos 0, whose pose
/// applyGantryHold() reads and writes. None when the scene's first coordinates
/// are not a free joint.
std::optional<int> robotBaseBody(const mjModel* absl_nonnull model) {
  for (int joint = 0; joint < model->njnt; ++joint) {
    if (model->jnt_type[joint] == mjJNT_FREE && model->jnt_qposadr[joint] == 0)
      return model->jnt_bodyid[joint];
  }
  return std::nullopt;
}

/// The gantry weld a scene declares to hold `baseBody`, as the messages that
/// ask for it spell it.
std::string gantryWeldDeclaration(absl::string_view baseBody) {
  return absl::StrCat(
      R"(<equality><weld name=")", kGantryWeldName,
      R"(" body1="world" body2=")", baseBody,
      R"(" relpose="0 0 <height> 1 0 0 0" active="false"/></equality>)");
}

/**
 * The weld half of checkSceneSupportsGantryHold(): a weld named kGantryWeldName
 * between bodies, from the world (body1, so that its relpose is the pose of the
 * base in the world, as applyGantryHold() writes it) to the robot's base
 * (body2).
 */
absl::Status checkGantryWeld(const mjModel* absl_nonnull model) {
  const std::optional<int> base = robotBaseBody(model);
  if (!base.has_value())
    return absl::FailedPreconditionError(
        "the scene has no free joint at qpos 0 for the gantry weld to hold");
  const std::string declaration =
      gantryWeldDeclaration(mujocoName(model, mjOBJ_BODY, *base));
  const int weld =
      mj_name2id(model, mjOBJ_EQUALITY, std::string(kGantryWeldName).c_str());
  if (weld < 0) {
    return absl::FailedPreconditionError(
        absl::StrCat("the scene declares no equality named '", kGantryWeldName,
                     "'; add ", declaration));
  }
  if (model->eq_type[weld] != mjEQ_WELD ||
      model->eq_objtype[weld] != mjOBJ_BODY) {
    return absl::FailedPreconditionError(absl::StrCat(
        "the scene's '", kGantryWeldName,
        "' equality is not a weld between two bodies; declare it as ",
        declaration));
  }
  if (model->eq_obj1id[weld] != 0 || model->eq_obj2id[weld] != *base) {
    return absl::FailedPreconditionError(absl::StrCat(
        "the scene's '", kGantryWeldName, "' weld joins body1 '",
        mujocoName(model, mjOBJ_BODY, model->eq_obj1id[weld]), "' to body2 '",
        mujocoName(model, mjOBJ_BODY, model->eq_obj2id[weld]),
        "'; it must hold the robot's base from the world: ", declaration));
  }
  return absl::OkStatus();
}
}  // namespace

std::vector<std::string> gantryHoldNames() {
  return {kWeldConstraintGantryHoldName, kKinematicTeleportGantryHoldName};
}

absl::StatusOr<GantryHold> gantryHoldFromName(absl::string_view name) {
  if (name == kWeldConstraintGantryHoldName) return GantryHold::kWeldConstraint;
  if (name == kKinematicTeleportGantryHoldName)
    return GantryHold::kKinematicTeleport;
  return absl::InvalidArgumentError(
      absl::StrCat("Unknown gantry_hold '", name,
                   "'. Set gantry_hold in the robot task file to one of: ",
                   absl::StrJoin(gantryHoldNames(), ", "), "."));
}

absl::Status checkSceneSupportsGantryHold(const mjModel* absl_nonnull model,
                                          GantryHold hold) {
  switch (hold) {
    case GantryHold::kWeldConstraint:
      return checkGantryWeld(model);
    case GantryHold::kKinematicTeleport:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "GantryHold ", static_cast<int>(hold), " is none of the gantry holds"));
}

namespace {

/** Compiles the scene of `config`, with its projectile when it names one (the
 * projectile is stored in `projectile`). */
absl::StatusOr<MjModelPtr> compileScene(const MujocoSimConfig& config,
                                        Projectile& projectile,
                                        absl::string_view projectileBodyName) {
  constexpr int kErrorSize = 1000;
  char error[kErrorSize] = "";
  // The scene is compiled with a ball in it when the task file named a
  // projectile, and exactly as it is on disk when it did not. Going through
  // mjSpec rather than editing the XML keeps the robot's own scene file
  // untouched.
  if (config.projectile.empty()) {
    MjModelPtr model(mj_loadXML(config.scenePath.c_str(), /*vfs=*/nullptr,
                                error, kErrorSize));
    if (model == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Could not load the MuJoCo scene ", config.scenePath, ": ", error));
    }
    return model;
  }
  absl::StatusOr<Projectile> named = projectileFromName(config.projectile);
  if (!named.ok()) return named.status();
  projectile = *std::move(named);
  const MjSpecPtr spec(mj_parseXML(config.scenePath.c_str(), /*vfs=*/nullptr,
                                   error, kErrorSize));
  if (spec == nullptr)
    return absl::InvalidArgumentError(absl::StrCat(
        "Could not parse the MuJoCo scene ", config.scenePath, ": ", error));
  if (absl::Status added =
          addProjectileToSpec(spec.get(), projectile, projectileBodyName);
      !added.ok()) {
    return added;
  }
  MjModelPtr model(mj_compile(spec.get(), /*vfs=*/nullptr));
  if (model == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Could not compile the MuJoCo scene ", config.scenePath, " with the '",
        projectile.name, "' projectile: ", mjs_getError(spec.get())));
  }
  return model;
}

/** Whether `actuator` turns its ctrl into exactly that torque on its joint: a
 * `<motor>` of gear 1. */
bool isUnitTorqueMotor(const mjModel* absl_nonnull model, int actuator) {
  const mjtNum* absl_nonnull gear =
      model->actuator_gear + kActuatorGearSize * actuator;
  const mjtNum* absl_nonnull gain =
      model->actuator_gainprm + mjNGAIN * actuator;
  return model->actuator_gaintype[actuator] == mjGAIN_FIXED && gain[0] == 1.0 &&
         model->actuator_biastype[actuator] == mjBIAS_NONE &&
         model->actuator_dyntype[actuator] == mjDYN_NONE && gear[0] == 1.0;
}

}  // namespace

absl::StatusOr<MujocoSimInterface::ProjectileAddresses>
MujocoSimInterface::findProjectile(const mjModel* absl_nonnull model) {
  ProjectileAddresses addresses;
  addresses.bodyId = mj_name2id(model, mjOBJ_BODY, kProjectileBodyName);
  if (addresses.bodyId < 0 || model->body_jntnum[addresses.bodyId] != 1) {
    return absl::InternalError(
        absl::StrCat("the projectile was compiled into the scene, but the "
                     "model has no body '",
                     kProjectileBodyName, "' with one joint"));
  }
  addresses.jointId = model->body_jntadr[addresses.bodyId];
  addresses.qposAddress = model->jnt_qposadr[addresses.jointId];
  addresses.dofAddress = model->jnt_dofadr[addresses.jointId];
  return addresses;
}

absl::StatusOr<MujocoSimInterface::LoadedScene> MujocoSimInterface::loadScene(
    const MujocoSimConfig& config,
    const robot::model::RobotDescription& description) {
  LoadedScene scene;
  const absl::StatusOr<GantryHold> gantryHold =
      gantryHoldFromName(config.gantryHold);
  if (!gantryHold.ok()) return gantryHold.status();
  scene.gantryHold = *gantryHold;
  absl::StatusOr<MjModelPtr> model =
      compileScene(config, scene.projectile, kProjectileBodyName);
  if (!model.ok()) return model.status();
  scene.model = *std::move(model);
  if (!config.projectile.empty()) {
    const absl::StatusOr<ProjectileAddresses> addresses =
        findProjectile(scene.model.get());
    if (!addresses.ok()) return addresses.status();
    scene.projectileAddresses = *addresses;
  }
  const absl::StatusOr<Projectile> fallbackProjectile =
      projectileFromName(kFallbackImpactProjectile);
  if (!fallbackProjectile.ok()) return fallbackProjectile.status();
  scene.fallbackRestitution = fallbackProjectile->restitution;
  if (const absl::Status supported =
          checkSceneSupportsGantryHold(scene.model.get(), scene.gantryHold);
      !supported.ok()) {
    LOG(ERROR) << "gantry_hold is 'weld_constraint' but " << supported.message()
               << " (" << config.scenePath
               << "). Falling back to 'kinematic_teleport', which leaves the "
                  "base unsupported inside mj_step and makes gravity "
                  "compensation drive the limbs into their stops.";
    scene.gantryHold = GantryHold::kKinematicTeleport;
  }
  absl::StatusOr<JointMap> jointMap = buildJointMap(
      scene.model.get(), description, scene.projectileAddresses.bodyId);
  if (!jointMap.ok()) return jointMap.status();
  scene.jointMap = *std::move(jointMap);
  absl::StatusOr<MjDataPtr> data = makeMjData(scene.model.get());
  if (!data.ok()) return data.status();
  scene.data = *std::move(data);
  return scene;
}

absl::StatusOr<MujocoSimInterface::JointMap> MujocoSimInterface::buildJointMap(
    const mjModel* absl_nonnull model,
    const robot::model::RobotDescription& description,
    int projectileBodyId) {
  if (model->njnt < 1 || model->jnt_type[0] != mjJNT_FREE ||
      model->jnt_qposadr[0] != 0 || model->jnt_dofadr[0] != 0) {
    return absl::InvalidArgumentError(
        "the scene's first joint must be the robot's free joint, at qpos 0 and "
        "dof 0");
  }
  JointMap map;
  for (int joint = 1; joint < model->njnt; ++joint) {
    if (projectileBodyId >= 0 && model->jnt_bodyid[joint] == projectileBodyId)
      continue;
    const std::string name = mujocoName(model, mjOBJ_JOINT, joint);
    if (!description.containsJoint(name)) {
      LOG(WARNING)
          << "Joint contained in mujoco xml not exposed to RobotHWInterface: "
          << name;
      continue;
    }
    if (model->jnt_type[joint] != mjJNT_HINGE &&
        model->jnt_type[joint] != mjJNT_SLIDE) {
      return absl::InvalidArgumentError(
          absl::StrCat("the scene's joint '", name,
                       "' of the robot description is not a hinge or a slide "
                       "(MuJoCo joint type ",
                       model->jnt_type[joint],
                       "); the simulator maps one coordinate per robot joint"));
    }
    map.joints.push_back(
        MappedJoint{.robotJoint = description.getJointIndex(name),
                    .qposAddress = model->jnt_qposadr[joint],
                    .dofAddress = model->jnt_dofadr[joint]});
  }
  std::vector<bool> jointActuated(model->njnt, false);
  for (int actuator = 0; actuator < model->nu; ++actuator) {
    // Unnamed actuators are common (`<motor joint="..."/>`): they are matched
    // through their joint like any other.
    const int joint =
        model->actuator_trntype[actuator] == mjTRN_JOINT
            ? model->actuator_trnid[kActuatorTransmissionTargets * actuator]
            : -1;
    const std::string jointName =
        joint >= 0 ? mujocoName(model, mjOBJ_JOINT, joint) : std::string();
    if (joint < 0 || !description.containsJoint(jointName)) {
      LOG(WARNING) << "Actuator " << actuator << " ('"
                   << mujocoName(model, mjOBJ_ACTUATOR, actuator)
                   << "') contained in mujoco xml cannot be commanded through "
                      "RobotHWInterface.";
      continue;
    }
    if (!isUnitTorqueMotor(model, actuator)) {
      return absl::InvalidArgumentError(
          absl::StrCat("the scene's actuator ", actuator, " ('",
                       mujocoName(model, mjOBJ_ACTUATOR, actuator),
                       "') on joint '", jointName,
                       "' is not a <motor> of gear 1; the simulator writes the "
                       "joint's torque into its ctrl"));
    }
    if (jointActuated[joint]) {
      return absl::InvalidArgumentError(
          absl::StrCat("the scene's joint '", jointName,
                       "' has more than one actuator; each would get the "
                       "joint's whole torque"));
    }
    jointActuated[joint] = true;
    map.actuators.push_back(
        MappedActuator{.actuatorId = actuator,
                       .mujocoJointId = joint,
                       .robotJoint = description.getJointIndex(jointName)});
  }
  return map;
}

absl::Status MujocoSimInterface::checkConfigNumbers(
    const MujocoSimConfig& config) {
  if (!(config.dt > 0.0) || !std::isfinite(config.dt)) {
    return absl::InvalidArgumentError(
        absl::StrCat("the physics step dt must be positive and finite, got ",
                     config.dt, " s"));
  }
  if (!(config.ragdollJointDamping >= 0.0) ||
      !std::isfinite(config.ragdollJointDamping)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "sim_ragdoll_joint_damping must be non-negative and finite, got ",
        config.ragdollJointDamping, " N m s/rad"));
  }
  if (!std::isfinite(config.autoResetBaseHeight)) {
    return absl::InvalidArgumentError(
        absl::StrCat("sim_auto_reset_min_pos_base_rt_world_ewrt_world_z must "
                     "be finite, got ",
                     config.autoResetBaseHeight, " m"));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<MujocoSimInterface>> MujocoSimInterface::Create(
    const MujocoSimConfig& config, const std::string& urdfPath) {
  if (absl::Status viewer = validateViewerSettings(config.viewer);
      !viewer.ok()) {
    return viewer;
  }
  if (absl::Status numbers = checkConfigNumbers(config); !numbers.ok()) {
    return numbers;
  }
  absl::StatusOr<robot::model::RobotDescription> robotDescription =
      robot::model::RobotDescription::Create(urdfPath);
  if (!robotDescription.ok()) return robotDescription.status();
  absl::StatusOr<LoadedScene> scene = loadScene(config, *robotDescription);
  if (!scene.ok()) return scene.status();
  return absl::WrapUnique(new MujocoSimInterface(
      config, *std::move(robotDescription), *std::move(scene)));
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

MujocoSimInterface::MujocoSimInterface(
    const MujocoSimConfig& config,
    robot::model::RobotDescription robotDescription,
    LoadedScene scene)
    : RobotHWInterfaceBase(std::move(robotDescription)),
      config_(config),
      robotStateInternal_(model::RobotState(this->getRobotDescription())),
      robotJointActionInternal_(
          model::RobotJointAction(this->getRobotDescription())),
      timeStepMicro_(
          static_cast<size_t>(config.dt * ::ocs2::kMicrosecondsPerSecond)),
      jointMap_(std::move(scene.jointMap)),
      mujocoModel_(std::move(scene.model)),
      mujocoData_(std::move(scene.data)),
      verbose_(config.verbose),
      lastRealTime_(std::chrono::steady_clock::now()),
      gantryHold_(scene.gantryHold),
      // The scene's own damping, before anything below writes dof_damping: what
      // the joints have while the torques are on.
      sceneDofDamping_(mujocoModel_->dof_damping,
                       mujocoModel_->dof_damping + mujocoModel_->nv),
      fallbackRestitution_(scene.fallbackRestitution),
      impactJacobian_(static_cast<size_t>(kPositionDim) * mujocoModel_->nv),
      impactGeneralizedDirection_(mujocoModel_->nv),
      impactMobility_(mujocoModel_->nv),
      projectile_(std::move(scene.projectile)),
      dodgeballBodyId_(scene.projectileAddresses.bodyId),
      dodgeballJointId_(scene.projectileAddresses.jointId),
      dodgeballQposAdr_(scene.projectileAddresses.qposAddress),
      dodgeballDofAdr_(scene.projectileAddresses.dofAddress) {
  if (hasProjectile()) {
    parkProjectile();
    LOG(INFO) << "Projectile '" << projectile_.name
              << "' ready: " << 2.0 * projectile_.radius << " m across, "
              << mujocoModel_->body_mass[dodgeballBodyId_]
              << " kg, restitution " << projectile_.restitution
              << ", mass adjustable from " << kMinProjectileMass << " to "
              << kMaxProjectileMass << " kg per throw.";
  }
  configureSolverOptions();
  if (verbose_) printModelInfo();
  setInitialState();
  findFootSensors();
  setupContactDetection();
  setupGantry();

  // Safe init state for resets
  initialQpos_.assign(mujocoData_->qpos, mujocoData_->qpos + mujocoModel_->nq);
  initialQvel_.assign(mujocoData_->qvel, mujocoData_->qvel + mujocoModel_->nv);

  // Make sure the init state is propagated throughout the RobotInterface.
  updateThreadSafeRobotState();
  updateInterfaceStateFromRobot();

  // The simulator starts in zero-torque mode, with the ragdoll damping on the
  // robot's joints (the root's six dofs and the ball's are skipped).
  setRobotJointDamping(mujocoModel_.get(), dodgeballBodyId_,
                       config_.ragdollJointDamping);
  ragdollDampingApplied_ = true;

  setupRenderBuffer();
}

void MujocoSimInterface::configureSolverOptions() {
  mujocoModel_->opt.timestep = config_.dt;
  // MuJoCo's automatic reset on a numerically bad state calls mj_resetData,
  // which rewinds the clock to zero and puts the robot back without the
  // controller being told - its plans are then timestamped hundreds of seconds
  // in the future, and every later solve fails. The simulator detects the bad
  // state itself (stepWentUnstable) and recovers without rewinding the clock
  // (resetAndCatch).
  mujocoModel_->opt.disableflags |= mjDSBL_AUTORESET;
}

void MujocoSimInterface::setInitialState() {
  if (config_.initialState.has_value()) {
    setSimState(*config_.initialState);
    return;
  }
  model::RobotState initialState(getRobotDescription(),
                                 /*contactSize=*/kInitialStateContactPoints);
  initialState.setConfigurationToZero();
  initialState.setPosBaseRtWorldEwrtWorld(
      vector3_t(0.0, 0.0, kDefaultInitialBaseHeight));
  setSimState(initialState);
}

void MujocoSimInterface::findFootSensors() {
  for (int i = 0; i < mujocoModel_->nsensor; ++i) {
    const std::string sensorName =
        mujocoName(mujocoModel_.get(), mjOBJ_SENSOR, i);
    if (sensorName == "right_foot_touch_sensor")
      rightFootTouchSensorAddress_ = mujocoModel_->sensor_adr[i];
    if (sensorName == "left_foot_touch_sensor")
      leftFootTouchSensorAddress_ = mujocoModel_->sensor_adr[i];
    if (sensorName != "right_foot_force_sensor" &&
        sensorName != "left_foot_force_sensor")
      continue;
    // NOLINTBEGIN(frame-naming): the ids of the feet's force sensors, not forces
    if (!isFootForceSensor(mujocoModel_.get(), i)) {
      LOG(WARNING) << "[MujocoSimInterface] '" << sensorName
                   << "' is not a <force> sensor on a site; the foot's "
                      "measured force is zero";
    } else if (sensorName == "right_foot_force_sensor") {
      rightFootForceSensor_ = i;
    } else {
      leftFootForceSensor_ = i;
    }
    // NOLINTEND(frame-naming)
  }
}

void MujocoSimInterface::setupGantry() {
  if (config_.gantryHeight > 0.0) {
    gantryHeight_ = config_.gantryHeight;
  } else {
    // The height of the initial state, which setInitialState() has written.
    gantryHeight_ = mujocoData_->qpos[kZAxisIndex];
  }
  isGantryLocked_ = config_.isGantryLocked;
  gantryWeldEqId_ = mj_name2id(mujocoModel_.get(), mjOBJ_EQUALITY,
                               std::string(kGantryWeldName).c_str());
  if (gantryHold_ == GantryHold::kWeldConstraint) {
    // The weld carries the height, so the model's initial relpose is
    // irrelevant; eq_active is driven every step.
    mujocoModel_->eq_active0[gantryWeldEqId_] =
        static_cast<mjtByte>(isGantryLocked_.load());
  }
  if (verbose_) {
    LOG(INFO) << "Virtual gantry base hold: "
              << (gantryHold_ == GantryHold::kWeldConstraint
                      ? "weld_constraint"
                      : "kinematic_teleport")
              << " (gantry_hold).";
  }
}

void MujocoSimInterface::setupRenderBuffer() {
  // Initialize lock-free triple buffer for sim→render state transfer.
  // Each slot holds an MjState with its own mjData copy.
  MjState initState(mujocoModel_.get());
  mj_copyData(initState.data, mujocoModel_.get(), mujocoData_.get());
  initState.timestamp = mujocoData_->time;
  initState.metrics = metrics_;
  renderStateBuffer_ = std::make_unique<TripleBuffer<MjState>>(initState);

  // Throttle: publish to triple buffer at render frequency, not every sim step.
  // E.g. dt=0.0005 (2000 Hz sim), renderFrequencyHz=60 → publish every ~33
  // steps.
  if (config_.renderFrequencyHz > 0.0) {
    renderPublishInterval_ = std::max(
        static_cast<size_t>(1),
        static_cast<size_t>(1.0 / (config_.renderFrequencyHz * config_.dt)));
  }
}

/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::enableTorques() {
  // First the latched action: an action applied before the torques went off
  // must never reach the actuators.
  discardAppliedJointAction();
  zeroTorqueMode_.store(false);
}

void MujocoSimInterface::disableTorques() {
  zeroTorqueMode_.store(true);
}

void MujocoSimInterface::applyJointTorqueSwitch() {
  const bool zeroJointTorque = zeroTorqueMode_.load();
  if (zeroJointTorque == ragdollDampingApplied_) return;
  if (zeroJointTorque) {
    // The ragdoll damping, for a passive robot that settles rather than swings.
    setRobotJointDamping(mujocoModel_.get(), dodgeballBodyId_,
                         config_.ragdollJointDamping);
  } else {
    // Active control: the scene's own damping, as it was loaded.
    std::copy(sceneDofDamping_.begin(), sceneDofDamping_.end(),
              mujocoModel_->dof_damping);
  }
  ragdollDampingApplied_ = zeroJointTorque;
}

/******************************************************************************************************/

MujocoSimInterface::~MujocoSimInterface() {
  terminate_.store(true);
  if (simulationThread_.joinable()) simulationThread_.join();
  // The viewer has to go before any member it reads. viewer_ is declared ahead
  // of renderStateBuffer_, the contact timeline and the two mutexes, so
  // destruction in reverse declaration order would tear those down while the
  // viewer's render thread is still running - a use-after-free on every
  // shutdown that the render thread loses. Every viewer's destructor joins its
  // threads, so resetting it here orders the teardown correctly.
  viewer_.reset();
  // mujocoData_ and mujocoModel_ go with the members, after every thread that
  // read them has stopped.
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::reset() {
  // mj_resetData clears everything a bad step can leave behind - accelerations,
  // the solver's warm start, activations, applied forces - but also rewinds the
  // clock, which is put back: see the header.
  const mjtNum time = mujocoData_->time;
  mj_resetData(mujocoModel_.get(), mujocoData_.get());
  mujocoData_->time = time;
  std::copy(initialQpos_.begin(), initialQpos_.end(), mujocoData_->qpos);
  std::copy(initialQvel_.begin(), initialQvel_.end(), mujocoData_->qvel);
  mj_forward(mujocoModel_.get(), mujocoData_.get());
  lastBadStateWarnings_ = 0;
  gantryWeldAnchored_ = false;
  resetEpoch_.fetch_add(1);

  // Any dodgeball in play is canceled with the rest of the state: a ball still
  // bouncing around the robot that has just been put back on its feet, or an
  // impulse scheduled before the reset landing after it, is a disturbance
  // nobody threw at the robot now standing there. (A reset is the auto-reset
  // when the base drops below 0.2 m; the gantry catch on excessive tilt goes
  // through lockGantry(), which cancels the same way.)
  cancelDodgeball();
}

void MujocoSimInterface::resetAndCatch(absl::string_view reason) {
  LOG(WARNING) << "Resetting the simulation because " << reason
               << ". The robot is put back in its initial state and caught on "
                  "the gantry; the simulation clock keeps "
                  "running (reset epoch "
               << resetEpoch_.load() + 1 << ").";
  reset();
  for (int i = 0; i < mujocoModel_->nu; ++i) mujocoData_->ctrl[i] = 0.0;
  lockGantry();
  lastPush_ = AppliedPush{};
}

bool MujocoSimInterface::stepWentUnstable() {
  // With mjDSBL_AUTORESET set MuJoCo still detects a bad position, velocity or
  // acceleration; it counts a warning and leaves the state as it is. A new
  // warning since the previous step, or a state that is not finite, is a bad
  // step.
  const int warnings = mujocoData_->warning[mjWARN_BADQPOS].number +
                       mujocoData_->warning[mjWARN_BADQVEL].number +
                       mujocoData_->warning[mjWARN_BADQACC].number;
  const bool newWarning = warnings != lastBadStateWarnings_;
  lastBadStateWarnings_ = warnings;
  if (newWarning) return true;
  for (int i = 0; i < mujocoModel_->nq; ++i) {
    if (!std::isfinite(mujocoData_->qpos[i])) return true;
  }
  for (int i = 0; i < mujocoModel_->nv; ++i) {
    if (!std::isfinite(mujocoData_->qvel[i])) return true;
  }
  return false;
}

void MujocoSimInterface::setBaseStateForTesting(
    const std::array<double, kMjFreeJointQposSize>& baseQpos,
    const std::array<double, kMjFreeJointDofs>& baseQvel) {
  for (int i = 0; i < kMjFreeJointQposSize; ++i)
    mujocoData_->qpos[i] = baseQpos[i];
  for (int i = 0; i < kMjFreeJointDofs; ++i) mujocoData_->qvel[i] = baseQvel[i];
  mj_forward(mujocoModel_.get(), mujocoData_.get());
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
  // LINT.IfChange(mj_state_fields)
  state.timestamp = latest.timestamp;
  mj_copyData(state.data, mujocoModel_.get(), latest.data);
  state.metrics = latest.metrics;
  state.modelFields = latest.modelFields;  // same sizes: copied in place
  state.push = latest.push;
  // LINT.ThenChange(//robot_runtime/mujoco_sim_interface/include/mujoco_sim_interface/MujocoUtils.h:mj_state_fields)
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

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
    for (size_t j = 0; j < kPositionDim; ++j) {
      LOG(INFO) << mujocoData_->xpos[i * kPositionDim + j] << " ";
    }
    LOG(INFO);

    // Print orientation quaternion
    LOG(INFO) << "  Orientation (Quaternion): ";
    for (size_t j = 0; j < kMjQuaternionCoefficients; ++j) {
      LOG(INFO) << mujocoData_->xquat[i * kMjQuaternionCoefficients + j] << " ";
    }
    LOG(INFO);
  }

  std::string jointName(&mujocoModel_->names[mujocoModel_->name_jntadr[0]]);

  // Print the information
  LOG(INFO) << "Joint Name: " << jointName;
  LOG(INFO) << "Position: " << mujocoData_->qpos[kXAxisIndex] << " "
            << mujocoData_->qpos[kYAxisIndex] << " "
            << mujocoData_->qpos[kZAxisIndex] << " "
            << mujocoData_->qpos[kMjFreeJointQuaternionWIndex] << " "
            << mujocoData_->qpos[kMjFreeJointQuaternionXIndex] << " "
            << mujocoData_->qpos[kMjFreeJointQuaternionYIndex] << " "
            << mujocoData_->qpos[kMjFreeJointQuaternionZIndex];
  LOG(INFO)
      << "Velocity: " << mujocoData_->qvel[kXAxisIndex] << " "
      << mujocoData_->qvel[kYAxisIndex] << " " << mujocoData_->qvel[kZAxisIndex]
      << " "
      << mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kXAxisIndex]
      << " "
      << mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kYAxisIndex]
      << " "
      << mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kZAxisIndex];

  // The first coordinate and velocity of every other joint, at its own address.
  for (int i = 1; i < mujocoModel_->njnt; ++i) {
    LOG(INFO) << "Joint Name: "
              << mujocoName(mujocoModel_.get(), mjOBJ_JOINT, i)
              << ", Position: "
              << mujocoData_->qpos[mujocoModel_->jnt_qposadr[i]]
              << ", Velocity: "
              << mujocoData_->qvel[mujocoModel_->jnt_dofadr[i]];
  }
  LOG(INFO) << "Initialized " << jointMap_.joints.size() << " active Joints";
  LOG(INFO) << "Initialized " << jointMap_.actuators.size()
            << " active Actuators";

  // Calculate total mass
  scalar_t totalMass = 0.0;
  for (int i = 0; i < mujocoModel_->nbody; ++i) {
    totalMass += mujocoModel_->body_mass[i];
  }
  LOG(INFO) << "Total MuJoCo model mass: " << totalMass;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::setSimState(const model::RobotState& robotState) {
  // Root Pose
  vector3_t posBaseRtWorldEwrtWorld = robotState.getPosBaseRtWorldEwrtWorld();
  quaternion_t quat_base_rt_world = robotState.getQuatBaseRtWorld();

  mujocoData_->qpos[kXAxisIndex] = posBaseRtWorldEwrtWorld[kXAxisIndex];
  mujocoData_->qpos[kYAxisIndex] = posBaseRtWorldEwrtWorld[kYAxisIndex];
  mujocoData_->qpos[kZAxisIndex] = posBaseRtWorldEwrtWorld[kZAxisIndex];
  mujocoData_->qpos[kMjFreeJointQuaternionWIndex] = quat_base_rt_world.w();
  mujocoData_->qpos[kMjFreeJointQuaternionXIndex] = quat_base_rt_world.x();
  mujocoData_->qpos[kMjFreeJointQuaternionYIndex] = quat_base_rt_world.y();
  mujocoData_->qpos[kMjFreeJointQuaternionZIndex] = quat_base_rt_world.z();

  // Root Velocity

  vector3_t linvel_base_rt_world_ewrt_world =
      quat_base_rt_world * robotState.getLinvelBaseRtWorldEwrtBase();
  vector3_t angvel_base_rt_world_ewrt_base =
      robotState.getAngvelBaseRtWorldEwrtBase();

  mujocoData_->qvel[kXAxisIndex] = linvel_base_rt_world_ewrt_world[kXAxisIndex];
  mujocoData_->qvel[kYAxisIndex] = linvel_base_rt_world_ewrt_world[kYAxisIndex];
  mujocoData_->qvel[kZAxisIndex] = linvel_base_rt_world_ewrt_world[kZAxisIndex];
  mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kXAxisIndex] =
      angvel_base_rt_world_ewrt_base[kXAxisIndex];
  mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kYAxisIndex] =
      angvel_base_rt_world_ewrt_base[kYAxisIndex];
  mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kZAxisIndex] =
      angvel_base_rt_world_ewrt_base[kZAxisIndex];

  // Joint State
  for (const MappedJoint& joint : jointMap_.joints) {
    mujocoData_->qpos[joint.qposAddress] =
        robotState.getJointPosition(joint.robotJoint);
    mujocoData_->qvel[joint.dofAddress] =
        robotState.getJointVelocity(joint.robotJoint);
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::updateThreadSafeRobotState() {
  // Update mujoco joint angles
  for (const MappedJoint& joint : jointMap_.joints) {
    robotStateInternal_.setJointPosition(joint.robotJoint,
                                         mujocoData_->qpos[joint.qposAddress]);
    robotStateInternal_.setJointVelocity(joint.robotJoint,
                                         mujocoData_->qvel[joint.dofAddress]);
  }

  // The free joint's qpos holds the base quaternion w first; Eigen's
  // constructor takes (w, x, y, z) as well. Its qvel holds the linear velocity
  // of the base frame's origin in the world frame, then the angular velocity in
  // the base frame (mujoco.readthedocs.io, "Floating objects").
  quaternion_t quat_base_rt_world =
      quaternion_t(mujocoData_->qpos[kMjFreeJointQuaternionWIndex],
                   mujocoData_->qpos[kMjFreeJointQuaternionXIndex],
                   mujocoData_->qpos[kMjFreeJointQuaternionYIndex],
                   mujocoData_->qpos[kMjFreeJointQuaternionZIndex]);
  vector3_t angvelBaseRtWorldEwrtBase = vector3_t(
      mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kXAxisIndex],
      mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kYAxisIndex],
      mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kZAxisIndex]);

  // Contact flags handed to the controller: the ground truth of the physics,
  // see updateGroundTruthContacts(). Without contact detection (no contact
  // frame names configured), and for a contact point whose MuJoCo body could
  // not be resolved, the point is reported as touching. Which contact state the
  // controller actually uses is the choice of its contact estimator (task file
  // `contact_estimator`).
  const bool useGroundTruth = hasContactDetection();
  const uint32_t touching =
      groundTruthContactMask_.load() | unresolvedContactMask_;

  robotStateInternal_.setPosBaseRtWorldEwrtWorld(
      vector3_t(mujocoData_->qpos[kXAxisIndex], mujocoData_->qpos[kYAxisIndex],
                mujocoData_->qpos[kZAxisIndex]));
  robotStateInternal_.setQuatBaseRtWorld(quat_base_rt_world);
  // Rotate the linear velocity from the world frame into the base frame; the
  // angular one is in the base frame already.
  robotStateInternal_.setLinvelBaseRtWorldEwrtBase(
      quat_base_rt_world.inverse() * vector3_t(mujocoData_->qvel[kXAxisIndex],
                                               mujocoData_->qvel[kYAxisIndex],
                                               mujocoData_->qvel[kZAxisIndex]));
  robotStateInternal_.setAngvelBaseRtWorldEwrtBase(angvelBaseRtWorldEwrtBase);
  // The base's accelerometer: the specific force R^T (a - g) of its origin in
  // its own frame, a the free joint's linear acceleration of the last step in
  // the world frame (qacc, the derivative of the world-frame qvel above) and g
  // the model's gravity. The input of the robot process's `kalman` state
  // estimator.
  const vector3_t linaccBaseRtWorldEwrtWorld(mujocoData_->qacc[kXAxisIndex],
                                             mujocoData_->qacc[kYAxisIndex],
                                             mujocoData_->qacc[kZAxisIndex]);
  const vector3_t gravity(mujocoModel_->opt.gravity[kXAxisIndex],
                          mujocoModel_->opt.gravity[kYAxisIndex],
                          mujocoModel_->opt.gravity[kZAxisIndex]);
  robotStateInternal_.setSpecforceBaseRtWorldEwrtBase(
      quat_base_rt_world.inverse() * (linaccBaseRtWorldEwrtWorld - gravity));
  const size_t numFlags = robotStateInternal_.getContactFlags().size();
  for (size_t i = 0; i < numFlags; ++i) {
    const bool flag = !useGroundTruth || i >= contactBodyIds_.size() ||
                      ((touching >> i) & 1u) != 0u;
    robotStateInternal_.setContactFlag(i, flag);
  }

  robotStateInternal_.setTime(
      mujocoData_->time);  // Todo Manu: should mujoco be the source of time?

  publishRobotState(robotStateInternal_);

  // The foot sensors of the same step, for takeMeasForceFeetEwrtWorld().
  MeasuredFootForces& measForceFeetEwrtWorld =
      bufferedMeasForceFeetEwrtWorld_.writeSlot();
  // NOLINTBEGIN(frame-naming): the ids of the feet's force sensors, not forces
  measForceFeetEwrtWorld.left = getMeasForceFootEwrtWorld(leftFootForceSensor_);
  measForceFeetEwrtWorld.right =
      getMeasForceFootEwrtWorld(rightFootForceSensor_);
  // NOLINTEND(frame-naming)
  bufferedMeasForceFeetEwrtWorld_.publishWrite();
}

vector3_t MujocoSimInterface::getMeasForceFootEwrtWorld(int sensorId) const {
  if (sensorId == kNoFootForceSensor)  // NOLINT(frame-naming): a sensor id
    return vector3_t::Zero();
  const std::array<mjtNum, kPositionDim> measForceFootEwrtWorld =
      computeMeasForceFootEwrtWorld(mujocoModel_.get(), mujocoData_.get(),
                                    sensorId);
  return vector3_t(measForceFootEwrtWorld[kXAxisIndex],
                   measForceFootEwrtWorld[kYAxisIndex],
                   measForceFootEwrtWorld[kZAxisIndex]);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::updateMetrics() {
  simFps_.tick();

  metrics_.fpsSim = simFps_.fps();

  const std::chrono::steady_clock::time_point nowRealTime =
      std::chrono::steady_clock::now();
  const double realElapsedTime =
      std::chrono::duration<double>(nowRealTime - lastRealTime_).count();
  lastRealTime_ = nowRealTime;

  metrics_.driftTick = config_.dt - realElapsedTime;
  metrics_.driftCumulative += metrics_.driftTick;

  metrics_.rtfTick = (realElapsedTime > kMinRealTimePerStep)
                         ? (config_.dt / realElapsedTime)
                         : 1.0;

  // Window-based RTF: total sim time elapsed / total wall time elapsed.
  // This is the true real-time factor, immune to per-tick scheduling noise.
  double wallElapsed =
      std::chrono::duration<double>(nowRealTime - loopStartTime_).count();
  double simElapsed = mujocoData_->time - simTimeAtLoopStart_;
  metrics_.rtfSmoothed = (wallElapsed > kMinRealTimeFactorWindow)
                             ? (simElapsed / wallElapsed)
                             : 1.0;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::simulationStep() {
  // The torque switch of another thread takes effect here, on the thread that
  // owns the model.
  applyJointTorqueSwitch();
  // The newest action, unless the torques are off or it was applied before they
  // came back on (enableTorques()).
  const bool actionCurrent = takeJointAction(robotJointActionInternal_);
  if (ragdollDampingApplied_ || !actionCurrent) {
    // Zero-torque / ragdoll mode: zero all actuator commands.
    // Joint damping is handled by MuJoCo's native dof_damping (boosted on
    // entry).
    for (int i = 0; i < mujocoModel_->nu; ++i) {
      mujocoData_->ctrl[i] = 0.0;
    }
  } else {
    for (const MappedActuator& actuator : jointMap_.actuators) {
      // A joint of the robot description by construction (buildJointMap()), so
      // the index is valid.
      const joint_index_t idx = actuator.robotJoint;
      const std::optional<robot::model::JointAction>& jointAction =
          robotJointActionInternal_[idx];
      if (!jointAction.has_value()) {
        // A RobotJointAction holds an action for every joint of its
        // description, but a controller can reset one. Its actuator is then
        // commanded nothing, as in zero-torque mode, rather than ending the
        // physics thread.
        mujocoData_->ctrl[actuator.actuatorId] = 0.0;
        continue;
      }
      double totalJointTorque = jointAction->getTotalFeedbackJointTorque(
          robotStateInternal_.getJointPosition(idx),
          robotStateInternal_.getJointVelocity(idx));

      // The actuator's and the joint's force ranges, as mj_step would apply
      // them (the actuator is a unit-gain, gear-1 motor, buildJointMap(), so
      // its force is its ctrl and its joint's torque): the clamped torque is
      // what the joint gets, and is what is commanded.
      const int id = actuator.actuatorId;
      if (mujocoModel_->actuator_forcelimited[id]) {
        totalJointTorque = std::clamp(
            totalJointTorque,
            mujocoModel_->actuator_forcerange[kRangeBounds * id],
            mujocoModel_->actuator_forcerange[kRangeBounds * id + 1]);
      }
      const int joint = actuator.mujocoJointId;
      if (mujocoModel_->jnt_actfrclimited[joint]) {
        totalJointTorque =
            std::clamp(totalJointTorque,
                       mujocoModel_->jnt_actfrcrange[kRangeBounds * joint],
                       mujocoModel_->jnt_actfrcrange[kRangeBounds * joint + 1]);
      }
      mujocoData_->ctrl[id] = totalJointTorque;
    }
  }

  // Suspend the robot at the gantry height, by whichever implementation the
  // task file named.
  if (config_.enableGantry) {
    applyGantryHold();
  }

  // Launches a staged throw, parks a finished ball, and - only when no ball is
  // compiled in - applies the fallback impulse. Free when nothing is in play.
  applyDodgeball();

  mj_step(mujocoModel_.get(), mujocoData_.get());
  if (stepWentUnstable()) {
    resetAndCatch(absl::StrCat(
        "the simulation became numerically unstable at t = ", mujocoData_->time,
        " s"));
  }
  updateGroundTruthContacts();
  updateThreadSafeRobotState();
  updateMetrics();

  // Publish state to the lock-free triple buffer for the render thread.
  // Throttled to render frequency to avoid unnecessary mj_copyData overhead.
  if (++renderPublishCounter_ >= renderPublishInterval_) {
    renderPublishCounter_ = 0;
    MjState& writeSlot = renderStateBuffer_->writeSlot();
    writeSlot.timestamp = mujocoData_->time;
    mj_copyData(writeSlot.data, mujocoModel_.get(), mujocoData_.get());
    writeSlot.metrics = metrics_;
    // The model arrays this thread writes (damping, gantry weld, projectile),
    // for a viewer with a private model copy.
    writeSlot.modelFields.captureFrom(mujocoModel_.get());
    writeSlot.push = lastPush_;
    renderStateBuffer_->publishWrite();
  }

  // Auto reset logic.
  if (mujocoData_->qpos[kZAxisIndex] < config_.autoResetBaseHeight) {
    resetAndCatch(absl::StrCat("the base dropped to ",
                               mujocoData_->qpos[kZAxisIndex], " m, below the ",
                               config_.autoResetBaseHeight, " m floor limit"));
    // resetAndCatch() has zeroed every actuator command.
    mj_step(mujocoModel_.get(), mujocoData_.get());
    updateThreadSafeRobotState();
    simFps_.reset();
    metrics_.reset();
    const std::chrono::steady_clock::time_point resetNow =
        std::chrono::steady_clock::now();
    lastRealTime_ = resetNow;
    loopStartTime_ = resetNow;
    simTimeAtLoopStart_ = mujocoData_->time;
    updateMetrics();

    // Publish reset state to triple buffer
    {
      MjState& writeSlot = renderStateBuffer_->writeSlot();
      writeSlot.timestamp = mujocoData_->time;
      mj_copyData(writeSlot.data, mujocoModel_.get(), mujocoData_.get());
      writeSlot.metrics = metrics_;
      writeSlot.modelFields.captureFrom(mujocoModel_.get());
      writeSlot.push = lastPush_;
      renderStateBuffer_->publishWrite();
    }

    // Sleep to let controller update and adjust;
    std::this_thread::sleep_until(
        std::chrono::steady_clock::now() +
        std::chrono::microseconds(kAutoResetPauseMicroseconds));
  }
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/

void MujocoSimInterface::simulationLoop() {
  simFps_.reset();
  metrics_.reset();
  const std::chrono::steady_clock::time_point now =
      std::chrono::steady_clock::now();
  lastRealTime_ = now;
  loopStartTime_ = now;
  simTimeAtLoopStart_ = mujocoData_->time;
  std::chrono::steady_clock::time_point nextWakeup = now;
  while (!terminate_.load()) {
    simulationStep();

    // Advance the wakeup target by one sim timestep.
    nextWakeup += std::chrono::microseconds(timeStepMicro_);

    const std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    stepLateness_.record(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - nextWakeup));
    // Allow up to 10ms of catchup budget for minor OS scheduling jitter.
    // If we fell behind by more than 10ms (e.g. auto-reset sleep), rebase
    // nextWakeup.
    if (now - nextWakeup > std::chrono::milliseconds(kMaxCatchUpMilliseconds)) {
      nextWakeup = now;
    } else if (nextWakeup > now) {
      // Sleep until 50µs before target, then busy-spin for sub-microsecond
      // precision.
      const std::chrono::steady_clock::time_point spinThreshold =
          nextWakeup - std::chrono::microseconds(kBusySpinMicroseconds);
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
  startViewer();
}

void MujocoSimInterface::startViewer() {
  const ViewerSettings& settings = config_.viewer;
  std::vector<std::string> failures;
  for (size_t index = 0; index < settings.names.size(); ++index) {
    const std::string& name = settings.names[index];
    absl::Status started = absl::OkStatus();
    {
      absl::StatusOr<std::unique_ptr<MujocoViewer>> viewer =
          createViewer(name, this);
      started = viewer.ok() ? (*viewer)->start() : viewer.status();
      if (started.ok()) {
        viewer_ = *std::move(viewer);
        viewerStatus_ = absl::OkStatus();
        // LINT.IfChange(web_viewer_announcement)
        LOG(INFO) << "[MujocoSimInterface] viewer: " << name;
        // LINT.ThenChange(//tools/deploy/session.sh:web_viewer_announcement)
        return;
      }
    }  // A viewer that did not start is destroyed here, before the next one is
       // created: a port it bound is free again.
    failures.push_back(absl::StrCat(name, ": ", started.message()));
    if (index + 1 < settings.names.size()) {
      LOG(WARNING) << "[MujocoSimInterface] the '" << name
                   << "' viewer did not start (" << started.message()
                   << "); trying '" << settings.names[index + 1] << "'.";
    }
  }
  viewerStatus_ = absl::FailedPreconditionError(
      absl::StrCat("no viewer of '", absl::StrJoin(settings.names, ","),
                   "' started (", absl::StrJoin(failures, "; "), "); list ",
                   kNoViewerName, " after it to run without a viewer"));
  LOG(ERROR) << "[MujocoSimInterface] " << viewerStatus_.message();
}

absl::string_view MujocoSimInterface::activeViewerName() const {
  return viewer_ != nullptr ? viewer_->name() : absl::string_view();
}

std::optional<int> MujocoSimInterface::webViewerPort() const {
  return viewer_ != nullptr ? viewer_->port() : std::nullopt;
}

void MujocoSimInterface::startSim() {
  if (!simInit_) initSim();
  // The physics runs on simulationThread_; the viewer renders on threads of its
  // own.
  simulationThread_ = std::thread(&MujocoSimInterface::simulationLoop, this);
}

void MujocoSimInterface::setupContactDetection() {
  contactBodyIds_.clear();
  unresolvedContactMask_ = 0;
  if (config_.contactFrameNames.empty()) return;
  if (config_.contactFrameNames.size() > kContactMaskCapacity) {
    LOG(INFO) << "[MujocoSimInterface] contact detection supports at most 32 "
                 "contact points, got "
              << config_.contactFrameNames.size()
              << "; the detection stays off.";
    return;
  }
  std::vector<std::string> errors;
  contactBodyIds_ = resolveContactBodies(
      mujocoModel_.get(), getRobotDescription().getURDFPath(),
      config_.contactFrameNames, config_.contactParentJointNames, &errors);
  for (const std::string& error : errors) {
    LOG(INFO) << "[MujocoSimInterface] contact detection: " << error;
  }
  for (size_t i = 0; i < contactBodyIds_.size(); ++i) {
    if (contactBodyIds_[i] < 0) {
      unresolvedContactMask_ |= (1u << i);
    } else if (verbose_) {
      LOG(INFO) << "[MujocoSimInterface] contact point '"
                << config_.contactFrameNames[i] << "' -> MuJoCo body '"
                << mujocoName(mujocoModel_.get(), mjOBJ_BODY,
                              contactBodyIds_[i])
                << "'";
    }
  }
  // The timeline is sampled at a fixed rate; the physics step is finer.
  constexpr double kTimelineSampleRateHz = 500.0;
  const double timestep = mujocoModel_->opt.timestep > 0.0
                              ? mujocoModel_->opt.timestep
                              : config_.dt;
  contactTimelineSampleInterval_ = std::max<size_t>(
      1, static_cast<size_t>(
             std::lround(1.0 / (kTimelineSampleRateHz * timestep))));
  contactTimeline_ = ContactTimeline(config_.contactTimelineWindow);
  // Samples within one window, both ends included, and one for rounding.
  contactTimelineCapacity_ =
      static_cast<size_t>(std::ceil(
          contactTimeline_.window() /
          (static_cast<double>(contactTimelineSampleInterval_) * timestep))) +
      kExtraTimelineSamples;
  if (verbose_) {
    LOG(INFO) << "[MujocoSimInterface] ground-truth contact detection: normal "
                 "force > "
              << config_.minForceMjcontactEwrtMjcontactX << " N.";
  }
}

void MujocoSimInterface::applyGantryHold() {
  const bool locked = isGantryLocked_.load();

  if (gantryHold_ == GantryHold::kWeldConstraint) {
    // A real constraint: mj_step solves it, so the base is supported DURING the
    // integration rather than corrected afterwards. The height is carried in
    // the weld's relpose, which the operator can move while the sim runs.
    mjtNum* absl_nonnull weld =
        mujocoModel_->eq_data + mjNEQDATA * gantryWeldEqId_;
    if (!locked) {
      gantryWeldAnchored_ = false;
    } else if (!gantryWeldAnchored_) {
      // Anchored where the robot is caught: its horizontal position, and the
      // nearest upright attitude (the twist of its attitude about the vertical,
      // design D9), so the catch lifts it upright in place, swinging it about a
      // horizontal axis only, instead of dragging it to the scene's anchor and
      // turning it to face +x. The ZYX yaw would turn a robot with a mixed tilt
      // about the vertical as well. The height stays the operator's gantry
      // height below.
      const mjtNum* absl_nonnull base = mujocoData_->qpos;
      const VerticalTwist quatHeadingRtWorld =
          verticalTwistFromQuaternion(base[kMjFreeJointQuaternionWIndex],
                                      base[kMjFreeJointQuaternionZIndex]);
      weld[kWeldRelposeXIndex] = base[0];
      weld[kWeldRelposeYIndex] = base[1];
      weld[kWeldRelposeQuaternionWIndex] = quatHeadingRtWorld.w;
      weld[kWeldRelposeQuaternionXIndex] = 0.0;
      weld[kWeldRelposeQuaternionYIndex] = 0.0;
      weld[kWeldRelposeQuaternionZIndex] = quatHeadingRtWorld.z;
      gantryWeldAnchored_ = true;
    }
    mujocoData_->eq_active[gantryWeldEqId_] = static_cast<mjtByte>(locked);
    weld[kWeldRelposeZIndex] = gantryHeight_.load();
    return;
  }

  // Legacy kinematic teleport. Unphysical - see the GantryHold comment - and
  // kept only to reproduce recorded runs.
  if (!locked) return;
  mujocoData_->qpos[kXAxisIndex] = 0.0;
  mujocoData_->qpos[kYAxisIndex] = 0.0;
  mujocoData_->qpos[kZAxisIndex] = gantryHeight_.load();
  mujocoData_->qpos[kMjFreeJointQuaternionWIndex] = 1.0;
  mujocoData_->qpos[kMjFreeJointQuaternionXIndex] = 0.0;
  mujocoData_->qpos[kMjFreeJointQuaternionYIndex] = 0.0;
  mujocoData_->qpos[kMjFreeJointQuaternionZIndex] = 0.0;

  mujocoData_->qvel[kXAxisIndex] = 0.0;
  mujocoData_->qvel[kYAxisIndex] = 0.0;
  mujocoData_->qvel[kZAxisIndex] = 0.0;
  mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kXAxisIndex] = 0.0;
  mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kYAxisIndex] = 0.0;
  mujocoData_->qvel[kMjFreeJointAngularVelocityOffset + kZAxisIndex] = 0.0;
}

void MujocoSimInterface::updateGroundTruthContacts() {
  if (contactBodyIds_.empty()) return;
  const uint32_t actual = groundTruthContactMask(
      mujocoModel_.get(), mujocoData_.get(), contactBodyIds_,
      config_.minForceMjcontactEwrtMjcontactX, dodgeballBodyId_);
  groundTruthContactMask_.store(actual);
  if (++contactTimelineSampleCounter_ < contactTimelineSampleInterval_) return;
  contactTimelineSampleCounter_ = 0;
  ContactTimelineSample sample;
  sample.time = mujocoData_->time;
  sample.actual = actual;
  sample.target = targetContactMask_.load();
  sample.targetKnown = targetContactKnown_.load();
  // The physics thread never waits for the viewer, which copies the timeline
  // under this lock: a sample that finds the lock held waits for the next one,
  // and of two that find it held the older is dropped
  // (droppedContactTimelineSamples()).
  std::unique_lock<std::mutex> lock(contactTimelineMutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    if (pendingTimelineSample_.has_value())
      droppedTimelineSamples_.fetch_add(1, std::memory_order_relaxed);
    pendingTimelineSample_ = sample;
    return;
  }
  if (pendingTimelineSample_.has_value()) {
    contactTimeline_.append(*pendingTimelineSample_);
    pendingTimelineSample_.reset();
  }
  contactTimeline_.append(sample);
}

void MujocoSimInterface::setTargetContactFlags(const std::vector<bool>& flags) {
  uint32_t mask = 0;
  for (size_t i = 0; i < flags.size() && i < kContactMaskCapacity; ++i) {
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

void MujocoSimInterface::copyContactTimeline(
    std::vector<ContactTimelineSample>& out) const {
  out.clear();
  out.reserve(contactTimelineCapacity_);  // before the lock, and a no-op once
                                          // `out` has been reserved
  std::lock_guard<std::mutex> lock(contactTimelineMutex_);
  out.assign(contactTimeline_.samples().begin(),
             contactTimeline_.samples().end());
}

void MujocoSimInterface::throwDodgeball(const DodgeballThrow& throwCommand) {
  // A second throw before the first has been picked up replaces it rather than
  // queueing: the button is a one-shot and an operator pressing it twice in one
  // control cycle means "throw now", not "throw twice".
  dodgeballMailbox_.writeSlot() = throwCommand;
  dodgeballMailbox_.publishWrite();
}

void MujocoSimInterface::cancelDodgeball() {
  // Taking a staged throw without using it drops it.
  dodgeballMailbox_.acquireRead();
  scheduledImpactTime_ = -1.0;
  parkProjectile();
  if (dodgeballImpulseApplied_ && robotRootBodyId() >= 0) {
    // The force and its moment.
    for (int axis = 0; axis < kMjSpatialVectorSize; ++axis) {
      mujocoData_
          ->xfrc_applied[kMjSpatialVectorSize * robotRootBodyId() + axis] = 0.0;
    }
  }
  dodgeballImpulseApplied_ = false;
}

void MujocoSimInterface::applyDodgeball() {
  // Simulation thread only. mjData has no lock - this thread owns it - so
  // everything that touches qpos, qvel or xfrc_applied happens here rather than
  // in throwDodgeball(), which stages the command on its caller's thread.

  if (cancelDodgeballRequested_.exchange(false)) {
    cancelDodgeball();
    LOG(INFO) << "Dodgeball canceled: the robot was caught on the gantry.";
  }

  // The impulse from the previous step has done its work; take it off again, or
  // one throw would become a constant force. Only the no-ball fallback path
  // below ever sets it.
  if (dodgeballImpulseApplied_) {
    const int previousBase = robotRootBodyId();
    if (previousBase >= 0) {
      // The force and its moment.
      for (int axis = 0; axis < kMjSpatialVectorSize; ++axis) {
        mujocoData_->xfrc_applied[kMjSpatialVectorSize * previousBase + axis] =
            0.0;
      }
    }
    dodgeballImpulseApplied_ = false;
  }

  std::optional<DodgeballThrow> staged;
  if (dodgeballMailbox_.acquireRead()) {
    staged = dodgeballMailbox_.readSlot();
  }

  if (staged.has_value() && throwStagedDodgeball(*staged)) return;

  // A ball that has finished is parked again, so that it stops being something
  // the robot can trip over and the next throw starts from a clean state rather
  // than from wherever the last one rolled to.
  if (projectileArmed_ && hasProjectile()) {
    const double speed =
        std::hypot(mujocoData_->qvel[dodgeballDofAdr_ + 0],
                   mujocoData_->qvel[dodgeballDofAdr_ + 1],
                   mujocoData_->qvel[dodgeballDofAdr_ + kZAxisIndex]);
    if (projectileRestMonitor_.update(speed, mujocoModel_->opt.timestep)) {
      parkProjectile();
      return;
    }
  }

  if (scheduledImpactTime_ < 0.0 || mujocoData_->time < scheduledImpactTime_)
    return;
  const int baseBody = robotRootBodyId();
  if (baseBody < 0) {
    scheduledImpactTime_ = -1.0;
    return;
  }

  applyFallbackImpulse(baseBody);
  scheduledImpactTime_ = -1.0;
}

bool MujocoSimInterface::throwStagedDodgeball(const DodgeballThrow& staged) {
  // Clamped once, for both paths: a topic can be published by hand, and a ball
  // and an impulse built from the same message must weigh the same.
  const double mass = clampProjectileMass(staged.mass);
  // The base pose is read HERE rather than when the command was staged: the
  // ball is thrown at the robot as it stands at the moment of the throw, and a
  // robot that turns during the flight does not drag the ball with it. Forward
  // kinematics first, because the base position and every geom pose
  // clearProjectileLaunch measures against come from mjData's derived
  // quantities, and before the first mj_step - a throw staged while the
  // simulator starts - nothing has computed them yet. It only reads qpos, which
  // the next mj_step recomputes from anyway.
  mj_kinematics(mujocoModel_.get(), mujocoData_.get());
  const int baseBody = robotRootBodyId();
  const double yaw = baseYaw();
  const double cosYaw = std::cos(yaw);
  const double sinYaw = std::sin(yaw);
  const double flight = std::max(staged.flightTime, 0.0);
  // [m/s^2] The gravity MuJoCo flies the ball under, the scene's, so that the
  // spawn search and the scheduled impact follow the path the ball actually
  // takes. The GUI aims with its own constant
  // (remote_control/tk_app/dodgeball.py GRAVITY), which every shipped scene
  // matches (humanoid_common_mpc_app/robot testShippedRobotGravity).
  const double gravity = -mujocoModel_->opt.gravity[kZAxisIndex];

  // The GUI computed the whole throw in the robot's own yaw frame, so rotating
  // it into the world is all that is left to do here - see
  // remote_control/tk_app/dodgeball.py, which owns the geometry.
  const std::array<double, kPositionDim> spawnPosBallRtBaseEwrtWorld{
      {cosYaw * staged.spawnPosBallRtBaseEwrtHeading[0] -
           sinYaw * staged.spawnPosBallRtBaseEwrtHeading[1],
       sinYaw * staged.spawnPosBallRtBaseEwrtHeading[0] +
           cosYaw * staged.spawnPosBallRtBaseEwrtHeading[1],
       staged.spawnPosBallRtBaseEwrtHeading[kZAxisIndex]}};
  const std::array<double, kPositionDim> launchLinvelBallRtWorldEwrtWorld{
      {cosYaw * staged.launchLinvelBallRtWorldEwrtHeading[0] -
           sinYaw * staged.launchLinvelBallRtWorldEwrtHeading[1],
       sinYaw * staged.launchLinvelBallRtWorldEwrtHeading[0] +
           cosYaw * staged.launchLinvelBallRtWorldEwrtHeading[1],
       staged.launchLinvelBallRtWorldEwrtHeading[kZAxisIndex]}};

  if (hasProjectile() && baseBody >= 0) {
    // A REAL BALL. Retune it to the operator's mass - a few stores, so every
    // throw does it - then place it at the spawn point with the launch velocity
    // and let MuJoCo fly it: the flight, the impact and the bounce are then
    // simulated directly by the physics engine rather than approximated.
    const absl::Status retuned =
        setProjectileMass(mujocoModel_.get(), dodgeballBodyId_, mass);
    if (!retuned.ok()) {
      // The throw still happens, at whatever mass the ball already had. Saying
      // so is the point: a disturbance that silently weighed something other
      // than what the slider said would invalidate the whole experiment.
      LOG(ERROR) << "Could not retune the dodgeball to " << mass << " kg ("
                 << retuned.message() << "); throwing it at "
                 << mujocoModel_->body_mass[dodgeballBodyId_] << " kg instead.";
    }

    ProjectileLaunch requested;
    const double* absl_nonnull posBaseRtWorldEwrtWorld =
        &mujocoData_->xpos[kPositionDim * baseBody];
    for (int axis = 0; axis < kPositionDim; ++axis) {
      requested.posBallRtWorldEwrtWorld[axis] =
          posBaseRtWorldEwrtWorld[axis] + spawnPosBallRtBaseEwrtWorld[axis];
      requested.linvelBallRtWorldEwrtWorld[axis] =
          launchLinvelBallRtWorldEwrtWorld[axis];
    }
    requested.flightTime = flight;
    // A spawn point inside the robot or under the floor is slid along the
    // ball's own path until it is clear, which keeps the aim; see
    // clearProjectileLaunch. The ball is still parked here, so measuring cannot
    // hit it.
    const absl::StatusOr<ProjectileLaunch> launch =
        clearProjectileLaunch(mujocoModel_.get(), mujocoData_.get(),
                              dodgeballBodyId_, requested, gravity);
    if (!launch.ok()) {
      LOG(WARNING) << "Dodgeball not thrown: " << launch.status().message();
      return true;
    }
    const double shift = requested.flightTime - launch->flightTime;
    if (std::abs(shift) > kMinReportedSpawnShift) {
      LOG(WARNING) << "The dodgeball's spawn point was inside the robot or the "
                      "floor, so it starts "
                   << std::abs(shift) << " s "
                   << (shift < 0.0 ? "earlier" : "later")
                   << " along the same path instead.";
    }

    for (int axis = 0; axis < kPositionDim; ++axis) {
      mujocoData_->qpos[dodgeballQposAdr_ + axis] =
          launch->posBallRtWorldEwrtWorld[axis];
      mujocoData_->qvel[dodgeballDofAdr_ + axis] =
          launch->linvelBallRtWorldEwrtWorld[axis];
      mujocoData_
          ->qvel[dodgeballDofAdr_ + kMjFreeJointAngularVelocityOffset + axis] =
          0.0;  // no spin: the GUI does not aim one
    }
    // Identity orientation. A sphere has no preferred one, and leaving whatever
    // the last throw ended at would make a re-thrown ball's spin depend on
    // where the previous one came to rest.
    mujocoData_->qpos[dodgeballQposAdr_ + kMjFreeJointQuaternionWIndex] = 1.0;
    mujocoData_->qpos[dodgeballQposAdr_ + kMjFreeJointQuaternionXIndex] = 0.0;
    mujocoData_->qpos[dodgeballQposAdr_ + kMjFreeJointQuaternionYIndex] = 0.0;
    mujocoData_->qpos[dodgeballQposAdr_ + kMjFreeJointQuaternionZIndex] = 0.0;
    setProjectileArmed(true);
    const double appliedMass = mujocoModel_->body_mass[dodgeballBodyId_];
    LOG(INFO) << "Dodgeball thrown: " << appliedMass << " kg, "
              << 2.0 * projectile_.radius << " m across, arriving in "
              << launch->flightTime << " s with "
              << appliedMass *
                     std::hypot(
                         launch->linvelBallRtWorldEwrtWorld[kXAxisIndex],
                         launch->linvelBallRtWorldEwrtWorld[kYAxisIndex],
                         launch->linvelBallRtWorldEwrtWorld[kZAxisIndex] -
                             gravity * launch->flightTime)
              << " N s of momentum.";
    return true;
  }

  // NO BALL IN THE SCENE, because the task file named no projectile. The flight
  // is ballistic and known in closed form, so the ball's arrival is scheduled,
  // and its impact is applied to the base as a one-step push then.
  scheduledArrivalLinvelBallRtWorldEwrtWorld_ =
      computeArrivalLinvelBallRtWorldEwrtWorld(launchLinvelBallRtWorldEwrtWorld,
                                               flight, gravity);
  scheduledBallMass_ = mass;
  scheduledImpactTime_ = mujocoData_->time + flight;
  LOG(INFO) << "Dodgeball thrown (no projectile compiled into this scene, so "
               "its impact is applied instead): "
            << mass << " kg arriving in " << flight << " s at ("
            << scheduledArrivalLinvelBallRtWorldEwrtWorld_[kXAxisIndex] << ", "
            << scheduledArrivalLinvelBallRtWorldEwrtWorld_[kYAxisIndex] << ", "
            << scheduledArrivalLinvelBallRtWorldEwrtWorld_[kZAxisIndex]
            << ") m/s.";
  return false;
}

void MujocoSimInterface::applyFallbackImpulse(int baseBody) {
  // The ball strikes the base's origin, which the GUI aimed it at, head on: the
  // impulse is along its arrival velocity n = v / |v| (a frictionless impact of
  // a sphere aimed at the point). Its size is that of an impact of the
  // dodgeball with the robot's effective mass at that point along n, M_eff = 1
  // / (n^T J M^-1 J^T n), from the configuration of this step: everything that
  // depends on qpos, the mass matrix's factorization included, is computed
  // first.
  const std::array<double, kPositionDim>& arrivalLinvelBallRtWorldEwrtWorld =
      scheduledArrivalLinvelBallRtWorldEwrtWorld_;
  const double speed =
      std::hypot(arrivalLinvelBallRtWorldEwrtWorld[kXAxisIndex],
                 arrivalLinvelBallRtWorldEwrtWorld[kYAxisIndex],
                 arrivalLinvelBallRtWorldEwrtWorld[kZAxisIndex]);
  if (!(speed > 0.0)) return;
  const std::array<double, kPositionDim> direction{
      {arrivalLinvelBallRtWorldEwrtWorld[kXAxisIndex] / speed,
       arrivalLinvelBallRtWorldEwrtWorld[kYAxisIndex] / speed,
       arrivalLinvelBallRtWorldEwrtWorld[kZAxisIndex] / speed}};
  mj_fwdPosition(mujocoModel_.get(), mujocoData_.get());
  const mjtNum* absl_nonnull point =
      &mujocoData_->xpos[kPositionDim * baseBody];
  mj_jac(mujocoModel_.get(), mujocoData_.get(), impactJacobian_.data(),
         /*jacr=*/nullptr, point, baseBody);
  const int nv = mujocoModel_->nv;
  double baseSpeedAlongNormal = 0.0;
  for (int dof = 0; dof < nv; ++dof) {
    double projected = 0.0;
    for (int axis = 0; axis < kPositionDim; ++axis)
      projected += direction[axis] * impactJacobian_[axis * nv + dof];
    impactGeneralizedDirection_[dof] = projected;
    baseSpeedAlongNormal += projected * mujocoData_->qvel[dof];
  }
  mj_solveM(mujocoModel_.get(), mujocoData_.get(), impactMobility_.data(),
            impactGeneralizedDirection_.data(), /*n=*/1);
  double inverseEffectiveMass = 0.0;
  for (int dof = 0; dof < nv; ++dof)
    inverseEffectiveMass +=
        impactGeneralizedDirection_[dof] * impactMobility_[dof];
  const double effectiveMass = inverseEffectiveMass > 0.0
                                   ? 1.0 / inverseEffectiveMass
                                   : std::numeric_limits<double>::infinity();
  const double impulse = projectileImpactImpulse(
      scheduledBallMass_, effectiveMass, fallbackRestitution_,
      speed - baseSpeedAlongNormal);

  // xfrc_applied is a FORCE and a moment about the body's center of mass held
  // over the step, so the impulse is divided by the step it is spread across,
  // and its moment about the center of mass is (p - c) x F for the force F at
  // the point p: together the generalized impulse J^T (impulse n) of a push at
  // p. One step is the shortest a collision can be represented here and is
  // already 0.5-1 ms, the order of a foam ball flattening against a torso.
  const double timestep = mujocoModel_->opt.timestep > 0.0
                              ? mujocoModel_->opt.timestep
                              : kFallbackImpulseTimestep;
  const mjtNum* absl_nonnull centerOfMass =
      &mujocoData_->xipos[kPositionDim * baseBody];
  std::array<mjtNum, kPositionDim> pushForceBaseEwrtWorld{};
  std::array<mjtNum, kPositionDim> lever{};
  for (int axis = 0; axis < kPositionDim; ++axis) {
    pushForceBaseEwrtWorld[axis] = impulse * direction[axis] / timestep;
    lever[axis] = point[axis] - centerOfMass[axis];
  }
  mjtNum* absl_nonnull applied =
      &mujocoData_->xfrc_applied[kMjSpatialVectorSize * baseBody];
  mju_cross(applied + kPositionDim, lever.data(),
            pushForceBaseEwrtWorld.data());
  for (int axis = 0; axis < kPositionDim; ++axis)
    applied[axis] = pushForceBaseEwrtWorld[axis];
  lastPush_ = AppliedPush{.bodyId = baseBody,
                          .impulseWorld = {impulse * direction[kXAxisIndex],
                                           impulse * direction[kYAxisIndex],
                                           impulse * direction[kZAxisIndex]},
                          .simTimeApplied = mujocoData_->time};
  dodgeballImpulseApplied_ = true;
  LOG(INFO) << "Dodgeball impact: " << impulse
            << " N s on an effective mass of " << effectiveMass << " kg.";
}

bool MujocoSimInterface::hasProjectile() const {
  return dodgeballBodyId_ >= 0 && dodgeballQposAdr_ >= 0 &&
         dodgeballDofAdr_ >= 0;
}

void MujocoSimInterface::setProjectileArmed(bool armed) {
  if (!hasProjectile()) return;
  // Armed: it collides, and gravity acts on it. Parked: neither. The gravity
  // half only works because the ball was compiled with gravcomp - see
  // addProjectileToSpec - so that mjModel::ngravcomp counts it.
  setProjectileCollisionEnabled(mujocoModel_.get(), dodgeballBodyId_, armed);
  if (mujocoModel_->body_gravcomp != nullptr) {
    mujocoModel_->body_gravcomp[dodgeballBodyId_] = armed ? 0.0 : 1.0;
  }
  projectileArmed_ = armed;
  if (armed) projectileRestMonitor_.start();
}

void MujocoSimInterface::parkProjectile() {
  if (!hasProjectile()) return;
  setProjectileArmed(false);
  for (int axis = 0; axis < kPositionDim; ++axis) {
    mujocoData_->qpos[dodgeballQposAdr_ + axis] =
        kParkPosBallRtWorldEwrtWorld[axis];
  }
  mujocoData_->qpos[dodgeballQposAdr_ + kMjFreeJointQuaternionWIndex] = 1.0;
  for (int index = kMjFreeJointQuaternionXIndex; index < kMjFreeJointQposSize;
       ++index)
    mujocoData_->qpos[dodgeballQposAdr_ + index] = 0.0;
  for (int dof = 0; dof < kMjFreeJointDofs; ++dof)
    mujocoData_->qvel[dodgeballDofAdr_ + dof] = 0.0;
}

int MujocoSimInterface::robotRootBodyId() const {
  // The FIRST body carrying a free joint, which is the convention
  // MujocoContactUtils::robotCentroidalState already relies on for the center
  // of mass and the viewer's camera.
  for (int body = 1; body < mujocoModel_->nbody; ++body) {
    if (mujocoModel_->body_jntnum[body] > 0 &&
        mujocoModel_->jnt_type[mujocoModel_->body_jntadr[body]] == mjJNT_FREE) {
      return body;
    }
  }
  return -1;
}

double MujocoSimInterface::baseYaw() const {
  // qpos[3..6] is the base free joint's quaternion (w, x, y, z), the same slice
  // setSimState writes. The twist heading (design D9), the gantry anchor's: the
  // angle of the twist about the vertical, in (-pi, pi].
  return yawOf(mujocoData_->qpos[kMjFreeJointQuaternionWIndex],
               mujocoData_->qpos[kMjFreeJointQuaternionZIndex]);
}

void MujocoSimInterface::setTargetContactPatches(
    const std::vector<TargetContactPatch>& patches) {
  std::lock_guard<std::mutex> lock(targetPatchMutex_);
  targetContactPatches_ = patches;
}

void MujocoSimInterface::copyTargetContactPatches(
    std::vector<TargetContactPatch>& out) const {
  std::lock_guard<std::mutex> lock(targetPatchMutex_);
  out = targetContactPatches_;
}

void MujocoSimInterface::takeMeasForceFeetEwrtWorld(
    vector3_t& measForceLfootEwrtWorld, vector3_t& measForceRfootEwrtWorld) {
  bufferedMeasForceFeetEwrtWorld_.acquireRead();
  const MeasuredFootForces& measForceFeetEwrtWorld =
      bufferedMeasForceFeetEwrtWorld_.readSlot();
  measForceLfootEwrtWorld = measForceFeetEwrtWorld.left;
  measForceRfootEwrtWorld = measForceFeetEwrtWorld.right;
}

std::vector<double> MujocoSimInterface::actuatorControlsForTesting() const {
  return std::vector<double>(mujocoData_->ctrl,
                             mujocoData_->ctrl + mujocoModel_->nu);
}

}  // namespace robot::mujoco_sim_interface
