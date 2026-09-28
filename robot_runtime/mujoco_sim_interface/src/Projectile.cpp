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

#include "mujoco_sim_interface/Projectile.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace robot::mujoco_sim_interface {

namespace {

/** [s] Resolution of the search along the ball's path in clearProjectileLaunch: 5 ms is a few cm at throwing speed. */
constexpr double kClearanceSearchStep = 0.005;
/** [s] How far back along its approach clearProjectileLaunch may move a spawn point. */
constexpr double kClearanceSearchBackward = 1.0;
/** [-] Fraction of the flight clearProjectileLaunch may skip forwards: the ball must still be arriving, not arrived. */
constexpr double kClearanceSearchForwardFraction = 0.8;

/** OK when `bodyId` is exactly what addProjectileToSpec builds; see setProjectileMass for why this is strict. */
absl::Status checkIsProjectile(const mjModel* model, int bodyId, absl::string_view caller) {
  if (model == nullptr) return absl::InvalidArgumentError(absl::StrCat(caller, ": the model is null."));
  if (bodyId <= 0 || bodyId >= model->nbody) {
    return absl::InvalidArgumentError(absl::StrCat(caller, ": body ", bodyId, " is not a body of a model with ", model->nbody,
                                                   " bodies (the world, body 0, is never a projectile)."));
  }
  const bool oneFreeJoint = model->body_jntnum[bodyId] == 1 && model->jnt_type[model->body_jntadr[bodyId]] == mjJNT_FREE;
  const bool oneSphere = model->body_geomnum[bodyId] == 1 && model->geom_type[model->body_geomadr[bodyId]] == mjGEOM_SPHERE;
  const bool ownTree = model->body_parentid[bodyId] == 0;
  const mjtNum* centerOfMass = &model->body_ipos[3 * bodyId];
  const bool centered = std::abs(centerOfMass[0]) < 1e-9 && std::abs(centerOfMass[1]) < 1e-9 && std::abs(centerOfMass[2]) < 1e-9;
  if (!oneFreeJoint || !oneSphere || !ownTree || !centered) {
    return absl::InvalidArgumentError(absl::StrCat(
        caller, ": body ", bodyId, " is not a projectile. A projectile is one free joint and one sphere, a direct child of the world, ",
        "centered on its body frame; this body has ", model->body_jntnum[bodyId], " joint(s), ", model->body_geomnum[bodyId],
        " geom(s), parent ", model->body_parentid[bodyId], (centered ? "" : " and an off-center center of mass"), "."));
  }
  return absl::OkStatus();
}

bool isFinite(const std::array<double, 3>& vector) {
  return std::isfinite(vector[0]) && std::isfinite(vector[1]) && std::isfinite(vector[2]);
}

/** Point `tau` seconds along the ballistic path through `launch` (negative tau is before the launch). */
ProjectileLaunch alongPath(const ProjectileLaunch& launch, double tau, double gravity) {
  ProjectileLaunch shifted = launch;
  for (int axis = 0; axis < 3; ++axis) {
    shifted.position[axis] = launch.position[axis] + launch.velocity[axis] * tau;
  }
  shifted.position[2] -= 0.5 * gravity * tau * tau;
  shifted.velocity[2] -= gravity * tau;
  shifted.flightTime = launch.flightTime - tau;
  return shifted;
}

/** True when a ball centered at `position` is at least kProjectileSpawnClearance from every geom it could collide with. */
bool isClearAt(const mjModel* model, mjData* data, int ballGeom, const std::array<double, 3>& position) {
  const int ballBody = model->geom_bodyid[ballGeom];
  const double radius = model->geom_size[3 * ballGeom];
  for (int axis = 0; axis < 3; ++axis) {
    data->geom_xpos[3 * ballGeom + axis] = position[axis];
  }
  for (int entry = 0; entry < 9; ++entry) {
    data->geom_xmat[9 * ballGeom + entry] = (entry % 4 == 0) ? 1.0 : 0.0;
  }
  for (int geom = 0; geom < model->ngeom; ++geom) {
    if (model->geom_bodyid[geom] == ballBody) continue;
    // The armed ball is contype = conaffinity = 1; this is MuJoCo's own pair filter against those bits.
    const bool couldCollide = (model->geom_conaffinity[geom] & 1) != 0 || (model->geom_contype[geom] & 1) != 0;
    if (!couldCollide) continue;
    // Bounding-sphere rejection first, so a throw costs a handful of exact distance queries rather than one per geom.
    // A plane has rbound 0, meaning unbounded, and is always measured.
    const double bound = model->geom_rbound[geom];
    if (bound > 0.0) {
      const mjtNum* center = &data->geom_xpos[3 * geom];
      const double separation = std::hypot(position[0] - center[0], position[1] - center[1], position[2] - center[2]);
      if (separation - bound - radius > kProjectileSpawnClearance) continue;
    }
    mjtNum fromTo[6];
    if (mj_geomDistance(model, data, ballGeom, geom, kProjectileSpawnClearance, fromTo) < kProjectileSpawnClearance) {
      return false;
    }
  }
  return true;
}

}  // namespace

const std::vector<std::string>& availableProjectiles() {
  // LINT.IfChange(projectile_names)
  static const std::vector<std::string> names{"dodgeball"};
  // clang-format off
  // LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml:sim_projectile, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.yaml:sim_projectile, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.yaml:sim_projectile, //robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.yaml:sim_projectile, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.yaml:sim_projectile)
  // clang-format on
  return names;
}

absl::StatusOr<Projectile> projectileFromName(absl::string_view name) {
  if (name == "dodgeball") {
    Projectile projectile;
    projectile.name = "dodgeball";
    // A regulation dodgeball is 8.5 inches across, so 0.108 m of radius, and a foam one of that size weighs a little
    // under half a kilogram. Restitution 0.80 is a rubber-skinned ball off a hard surface: it comes back at four
    // fifths of the speed it arrived at. A rolling resistance of 5 mm - a rolling coefficient of about 0.05, a soft
    // ball on a gym floor - brings it to rest from 3 m/s in about seven seconds.
    // LINT.IfChange(dodgeball_properties)
    projectile.radius = 0.108;
    projectile.mass = 0.45;
    projectile.restitution = 0.80;
    projectile.friction = 0.60;
    projectile.rollingFriction = 0.005;
    projectile.rgba = {{0.85f, 0.15f, 0.15f, 1.0f}};
    // LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/tk_app/dodgeball.py:dodgeball_mass)
    return projectile;
  }
  return absl::InvalidArgumentError(absl::StrCat("Unknown simProjectile '", name, "'. Set simProjectile in the robot task file to one of: ",
                                                 absl::StrJoin(availableProjectiles(), ", "),
                                                 "; or remove the key to compile the scene with no ball in it."));
}

double contactDampRatioForRestitution(double restitution) {
  // Clamped inside (0, 1): a restitution of 1 is a contact that never loses energy and never settles, and one of 0
  // would divide by a log of zero. The lower bound is the critically damped contact MuJoCo defaults to.
  const double clamped = std::clamp(restitution, 1e-4, 0.999);
  const double logarithm = std::log(clamped);
  return -logarithm / std::sqrt(M_PI * M_PI + logarithm * logarithm);
}

absl::Status addProjectileToSpec(mjSpec* spec, const Projectile& projectile, absl::string_view bodyName) {
  if (spec == nullptr) return absl::InvalidArgumentError("addProjectileToSpec: the spec is null.");
  if (!(projectile.radius > 0.0) || !(projectile.mass > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("addProjectileToSpec: '", projectile.name, "' needs a positive radius and mass, got ",
                                                   projectile.radius, " m and ", projectile.mass, " kg."));
  }
  if (!(projectile.restitution >= 0.0 && projectile.restitution < 1.0) || !(projectile.friction >= 0.0) ||
      !(projectile.rollingFriction >= 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "addProjectileToSpec: '", projectile.name, "' needs a restitution in [0, 1) and non-negative friction, got restitution ",
        projectile.restitution, ", friction ", projectile.friction, " and rolling friction ", projectile.rollingFriction, "."));
  }

  mjsBody* world = mjs_findBody(spec, "world");
  if (world == nullptr) return absl::NotFoundError("addProjectileToSpec: the scene has no 'world' body to attach to.");

  // Appended last. See the header: the FIRST free-joint body in this scene is taken to be the robot by the centroidal
  // state, the viewer's camera and every read of qpos[3..6].
  mjsBody* ball = mjs_addBody(world, /*def=*/nullptr);
  if (ball == nullptr) return absl::InternalError("addProjectileToSpec: mjs_addBody failed.");
  const std::string name(bodyName);
  mjs_setName(ball->element, name.c_str());
  for (int axis = 0; axis < 3; ++axis) {
    ball->pos[axis] = kProjectileParkPosition[axis];
  }
  // Compiled in, so that mjModel::ngravcomp counts this body; see the header. The simulator switches it off on a throw
  // and back on when it parks the ball.
  ball->gravcomp = 1.0;

  mjsJoint* joint = mjs_addFreeJoint(ball);
  if (joint == nullptr) return absl::InternalError("addProjectileToSpec: mjs_addFreeJoint failed.");
  // Named so that setupJointIndexMaps()'s warning about a MuJoCo joint the robot description does not know reads as
  // what it is rather than as a URDF/MJCF mismatch.
  mjs_setName(joint->element, absl::StrCat(name, "_free_joint").c_str());

  mjsGeom* geom = mjs_addGeom(ball, /*def=*/nullptr);
  if (geom == nullptr) return absl::InternalError("addProjectileToSpec: mjs_addGeom failed.");
  mjs_setName(geom->element, absl::StrCat(name, "_geom").c_str());
  geom->type = mjGEOM_SPHERE;
  geom->size[0] = projectile.radius;
  // The MASS is set rather than the density, because the default density of 1000 kg/m^3 would make a ball this size
  // weigh 5.3 kg - a medicine ball, and one that would knock over anything it hit.
  geom->mass = projectile.mass;
  // (slide, spin, roll), with condim 6 so that the last two act at all.
  geom->condim = 6;
  geom->friction[0] = projectile.friction;
  geom->friction[1] = projectile.rollingFriction;
  geom->friction[2] = projectile.rollingFriction;
  // The ball's contact parameters win every contact it makes; see the header for what averaging them costs.
  geom->priority = 1;
  // solref = (timeconst, dampratio). The damping ratio is what carries the restitution; see
  // contactDampRatioForRestitution for the inversion and kProjectileContactTimeConstant for why it is 20 ms.
  geom->solref[0] = kProjectileContactTimeConstant;
  geom->solref[1] = contactDampRatioForRestitution(projectile.restitution);
  for (size_t channel = 0; channel < projectile.rgba.size(); ++channel) {
    geom->rgba[channel] = projectile.rgba[channel];
  }
  // COLLIDING, even though the ball starts parked out of play: see setProjectileCollisionEnabled. The compiler folds
  // these into body_contype / body_conaffinity once, and a body whose aggregates are zero is pruned by the broadphase
  // for the rest of the session no matter what the geom flags are later set to.
  geom->contype = 1;
  geom->conaffinity = 1;
  return absl::OkStatus();
}

void setProjectileCollisionEnabled(mjModel* model, int bodyId, bool enabled) {
  if (model == nullptr || bodyId < 0 || bodyId >= model->nbody) return;
  const int flag = enabled ? 1 : 0;
  for (int geom = 0; geom < model->ngeom; ++geom) {
    if (model->geom_bodyid[geom] != bodyId) continue;
    model->geom_contype[geom] = flag;
    model->geom_conaffinity[geom] = flag;
  }
  // The body-level aggregates too, or the broadphase keeps its own answer. See the header.
  model->body_contype[bodyId] = flag;
  model->body_conaffinity[bodyId] = flag;
}

double clampProjectileMass(double mass) {
  if (std::isnan(mass)) return kMinProjectileMass;
  return std::clamp(mass, kMinProjectileMass, kMaxProjectileMass);
}

absl::Status setProjectileMass(mjModel* model, int bodyId, double mass) {
  const absl::Status isProjectile = checkIsProjectile(model, bodyId, "setProjectileMass");
  if (!isProjectile.ok()) return isProjectile;
  const double radius = model->geom_size[3 * model->body_geomadr[bodyId]];
  if (!(radius > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("setProjectileMass: the projectile's sphere has radius ", radius, " m."));
  }

  const double newMass = clampProjectileMass(mass);
  // A solid sphere, so the inertia tensor is isotropic: I = 2/5 m r^2 about every axis through the center.
  const double inertia = 0.4 * newMass * radius * radius;
  const double previousMass = model->body_mass[bodyId];

  model->body_mass[bodyId] = newMass;
  for (int axis = 0; axis < 3; ++axis) {
    model->body_inertia[3 * bodyId + axis] = inertia;
  }
  // The constants the compiler derived at qpos0. For a centered sphere on a free joint the joint-space inertia is
  // diag(m, m, m, I, I, I), so each is its diagonal or the inverse of it. See the header for why not mj_setConst.
  model->body_invweight0[2 * bodyId + 0] = 1.0 / newMass;
  model->body_invweight0[2 * bodyId + 1] = 1.0 / inertia;
  const int firstDof = model->jnt_dofadr[model->body_jntadr[bodyId]];
  for (int axis = 0; axis < 3; ++axis) {
    model->dof_invweight0[firstDof + axis] = 1.0 / newMass;
    model->dof_invweight0[firstDof + 3 + axis] = 1.0 / inertia;
    model->dof_M0[firstDof + axis] = newMass;
    model->dof_M0[firstDof + 3 + axis] = inertia;
  }
  // The ball is its own subtree, and the world's subtree holds everything.
  model->body_subtreemass[bodyId] = newMass;
  model->body_subtreemass[0] += newMass - previousMass;
  return absl::OkStatus();
}

bool isProjectileDof(const mjModel* model, int projectileBodyId, int dof) {
  if (model == nullptr || projectileBodyId < 0 || projectileBodyId >= model->nbody) return false;
  if (model->body_jntnum[projectileBodyId] <= 0) return false;
  const int firstDof = model->jnt_dofadr[model->body_jntadr[projectileBodyId]];
  return dof >= firstDof && dof < firstDof + 6;
}

void setRobotJointDamping(mjModel* model, int projectileBodyId, double damping) {
  if (model == nullptr) return;
  for (int dof = 6; dof < model->nv; ++dof) {
    if (isProjectileDof(model, projectileBodyId, dof)) continue;
    model->dof_damping[dof] = damping;
  }
}

absl::StatusOr<ProjectileLaunch> clearProjectileLaunch(
    const mjModel* model, mjData* data, int projectileBodyId, const ProjectileLaunch& requested, double gravity) {
  const absl::Status isProjectile = checkIsProjectile(model, projectileBodyId, "clearProjectileLaunch");
  if (!isProjectile.ok()) return isProjectile;
  if (data == nullptr) return absl::InvalidArgumentError("clearProjectileLaunch: the data is null.");
  if (!isFinite(requested.position) || !isFinite(requested.velocity) || !std::isfinite(requested.flightTime) || !std::isfinite(gravity)) {
    return absl::InvalidArgumentError("clearProjectileLaunch: the requested throw is not finite.");
  }
  const int ballGeom = model->body_geomadr[projectileBodyId];

  // The nearest point on the path in time, alternating backwards and forwards so that neither direction is favored.
  const double maxForward = kClearanceSearchForwardFraction * std::max(requested.flightTime, 0.0);
  const int steps = static_cast<int>(std::ceil(std::max(kClearanceSearchBackward, maxForward) / kClearanceSearchStep));
  for (int step = 0; step <= steps; ++step) {
    for (const double direction : {-1.0, 1.0}) {
      if (step == 0 && direction > 0.0) continue;  // tau = 0 is tried once
      const double tau = direction * step * kClearanceSearchStep;
      if (tau < -kClearanceSearchBackward || tau > maxForward) continue;
      const ProjectileLaunch candidate = alongPath(requested, tau, gravity);
      if (isClearAt(model, data, ballGeom, candidate.position)) return candidate;
    }
  }
  return absl::FailedPreconditionError(
      absl::StrCat("The dodgeball would start inside the robot or the floor, and no point on its path from ", kClearanceSearchBackward,
                   " s before the throw to ", maxForward,
                   " s after it is clear of them. Increase the spawn distance, or raise the elevation, on the GUI's Dodgeball tab."));
}

std::array<double, 3> projectileArrivalMomentum(const std::array<double, 3>& launchVelocity,
                                                double flightTime,
                                                double mass,
                                                double gravity) {
  const double clamped = clampProjectileMass(mass);
  return {{clamped * launchVelocity[0], clamped * launchVelocity[1], clamped * (launchVelocity[2] - gravity * flightTime)}};
}

void ProjectileRestMonitor::start() {
  slowTime_ = 0.0;
  timeInPlay_ = 0.0;
}

bool ProjectileRestMonitor::update(double speed, double dt) {
  timeInPlay_ += dt;
  slowTime_ = (speed < kRestSpeed) ? slowTime_ + dt : 0.0;
  return slowTime_ >= kRestDuration || timeInPlay_ >= kLifetime;
}

}  // namespace robot::mujoco_sim_interface
