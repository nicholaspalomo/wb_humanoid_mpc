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

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace robot::mujoco_sim_interface {

namespace {

/** [s] Contact time constant of the projectile. Two milliseconds is a foam ball flattening, not a steel bearing. */
constexpr double kContactTimeConstant = 0.002;

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
    // fifths of the speed it arrived at, which is lively enough to be worth watching and is what makes the second
    // bounce, rather than the first, the one that catches a robot mid-step.
    // LINT.IfChange(dodgeball_properties)
    projectile.radius = 0.108;
    projectile.mass = 0.45;
    projectile.restitution = 0.80;
    projectile.friction = 0.60;
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

  mjsBody* world = mjs_findBody(spec, "world");
  if (world == nullptr) return absl::NotFoundError("addProjectileToSpec: the scene has no 'world' body to attach to.");

  // Appended last. See the header: the FIRST free-joint body in this scene is taken to be the robot by the centroidal
  // state, the viewer's camera and every read of qpos[3..6].
  mjsBody* ball = mjs_addBody(world, nullptr);
  if (ball == nullptr) return absl::InternalError("addProjectileToSpec: mjs_addBody failed.");
  const std::string name(bodyName);
  mjs_setName(ball->element, name.c_str());
  ball->pos[0] = 0.0;
  ball->pos[1] = 0.0;
  ball->pos[2] = kProjectileParkHeight;

  mjsJoint* joint = mjs_addFreeJoint(ball);
  if (joint == nullptr) return absl::InternalError("addProjectileToSpec: mjs_addFreeJoint failed.");
  // Named so that setupJointIndexMaps()'s warning about a MuJoCo joint the robot description does not know reads as
  // what it is rather than as a URDF/MJCF mismatch.
  const std::string jointName = name + "_free_joint";
  mjs_setName(joint->element, jointName.c_str());

  mjsGeom* geom = mjs_addGeom(ball, nullptr);
  if (geom == nullptr) return absl::InternalError("addProjectileToSpec: mjs_addGeom failed.");
  mjs_setName(geom->element, (name + "_geom").c_str());
  geom->type = mjGEOM_SPHERE;
  geom->size[0] = projectile.radius;
  // The MASS is set rather than the density, because the default density of 1000 kg/m^3 would make a ball this size
  // weigh 5.3 kg - a medicine ball, and one that would knock over anything it hit.
  geom->mass = projectile.mass;
  geom->friction[0] = projectile.friction;
  // solref = (timeconst, dampratio). The damping ratio is what carries the restitution; see
  // contactDampRatioForRestitution for the inversion.
  geom->solref[0] = kContactTimeConstant;
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

absl::Status setProjectileMass(mjModel* model, int bodyId, double mass, double radius) {
  if (model == nullptr) return absl::InvalidArgumentError("setProjectileMass: the model is null.");
  if (bodyId < 0 || bodyId >= model->nbody) {
    return absl::InvalidArgumentError(absl::StrCat("setProjectileMass: body ", bodyId, " is not in a model of ", model->nbody, " bodies."));
  }
  // A PROJECTILE, precisely: one free joint and one sphere, which is exactly what addProjectileToSpec builds. A
  // looser check would let a mistargeted call silently retune a robot link's mass, and nothing downstream would
  // notice until the robot's own dynamics were quietly wrong.
  const bool oneFreeJoint = model->body_jntnum[bodyId] == 1 && model->jnt_type[model->body_jntadr[bodyId]] == mjJNT_FREE;
  const bool oneSphere = model->body_geomnum[bodyId] == 1 && model->geom_type[model->body_geomadr[bodyId]] == mjGEOM_SPHERE;
  if (!oneFreeJoint || !oneSphere) {
    return absl::InvalidArgumentError(absl::StrCat("setProjectileMass: body ", bodyId,
                                                   " is not a projectile - a projectile is one free joint and one sphere, and "
                                                   "this body has ",
                                                   model->body_jntnum[bodyId], " joint(s) and ", model->body_geomnum[bodyId], " geom(s)."));
  }
  if (!(radius > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat("setProjectileMass: the radius must be positive, got ", radius, " m."));
  }

  const double clamped = std::clamp(mass, kMinProjectileMass, kMaxProjectileMass);
  model->body_mass[bodyId] = clamped;
  // A solid sphere, so the inertia tensor is isotropic: I = 2/5 m r^2 about every axis through the centre.
  const double inertia = 0.4 * clamped * radius * radius;
  for (int axis = 0; axis < 3; ++axis) {
    model->body_inertia[3 * bodyId + axis] = inertia;
  }

  // The mass-derived constants, on a THROWAWAY mjData. See the header: mj_setConst overwrites the qpos of whatever
  // mjData it is given, so the live simulation state must never be handed to it.
  mjData* scratch = mj_makeData(model);
  if (scratch == nullptr) {
    return absl::InternalError("setProjectileMass: mj_makeData failed, so the mass-derived constants are now stale.");
  }
  mj_setConst(model, scratch);
  mj_deleteData(scratch);
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

}  // namespace robot::mujoco_sim_interface
