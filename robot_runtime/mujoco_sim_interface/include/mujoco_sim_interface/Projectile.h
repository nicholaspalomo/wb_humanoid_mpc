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

#include <array>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace robot::mujoco_sim_interface {

/**
 * A ball that can be thrown at the robot: the physical properties of one projectile, selected by name in the robot's
 * task file (`simProjectile`).
 *
 * A value type rather than a class family, because a projectile has no behaviour of its own - the simulator throws
 * it and MuJoCo does the rest. What it has is the handful of numbers that decide what being hit by it feels like.
 */
struct Projectile {
  std::string name;
  double radius{0.108};      // [m]
  double mass{0.45};         // [kg]
  double restitution{0.80};  // [-] coefficient of restitution against a rigid surface, in [0, 1)
  double friction{0.60};     // [-] sliding friction of the ball's surface
  std::array<float, 4> rgba{{0.85f, 0.15f, 0.15f, 1.0f}};
};

/** Names `simProjectile` accepts, in registry order, for the error messages and the task-file comments. */
const std::vector<std::string>& availableProjectiles();

/**
 * The projectile a task file named, or an InvalidArgumentError listing the valid names.
 *
 * An empty name is NOT an error and NOT a projectile: it means the scene is compiled exactly as it is on disk, with
 * no ball in it at all, which is what a robot that has not opted in gets. Callers check for the empty name before
 * calling this.
 */
absl::StatusOr<Projectile> projectileFromName(absl::string_view name);

/**
 * The contact damping ratio that produces a given coefficient of restitution.
 *
 * MuJoCo has no restitution parameter. A contact is a spring-damper whose `solref` is (timeconst, dampratio), and a
 * dampratio below 1 is what makes a contact bounce; the two are related by the logarithmic decrement of a damped
 * oscillator,
 *
 *     e = exp(-pi zeta / sqrt(1 - zeta^2))   =>   zeta = -ln(e) / sqrt(pi^2 + ln(e)^2).
 *
 * So the task file can name the number an engineer actually knows - a rubber ball bounces back at about 0.8 of the
 * speed it arrived at - and this turns it into the number MuJoCo wants. e = 0.8 gives zeta = 0.0708.
 *
 * Clamped to a hair below 1 and above 0: a restitution of exactly 1 is a contact that never settles, and one of 0 is
 * the critically damped default, which is what MuJoCo uses when it is given nothing.
 */
double contactDampRatioForRestitution(double restitution);

/**
 * Appends `projectile` to `spec` as a free-floating sphere, parked out of play and unable to collide until thrown.
 *
 * APPENDED, AND THAT MATTERS. It must be the LAST body in the world, because several things in this simulator take
 * the FIRST body carrying a free joint to be the robot: the centre of mass, the ZMP and DCM markers
 * (MujocoContactUtils::robotCentroidalState), the viewer's tracking camera, and every read of `qpos[3..6]` as the
 * base quaternion. A ball inserted ahead of the robot would quietly become the robot.
 *
 * The ball is compiled as a NORMALLY COLLIDING geom and parked out of play below the floor; the simulator calls
 * setProjectileCollisionEnabled to take it out of and back into the world. See that function for why it cannot be
 * compiled with its collisions already switched off.
 */
absl::Status addProjectileToSpec(mjSpec* spec, const Projectile& projectile, absl::string_view bodyName);

/**
 * Switches a projectile body's collisions on or off in a COMPILED model, so a parked ball can neither be hit nor rest
 * on anything until it is thrown.
 *
 * This sets both the geom flags and `body_contype` / `body_conaffinity`, and BOTH are load-bearing. MuJoCo's
 * broadphase prunes whole bodies by the body-level aggregates, which the compiler folds from the geoms once, at
 * compile time. A ball compiled with `contype = 0` therefore has a body aggregate of zero for the rest of the
 * session: restoring the geom flags when it is thrown would not bring it back, the broadphase would go on skipping
 * it, and the ball would sail straight through the robot without ever generating a contact. Hence the ball is
 * compiled colliding and parked through here instead.
 *
 * Does nothing for a negative body id, so callers need not check whether the scene has a projectile at all.
 */
void setProjectileCollisionEnabled(mjModel* model, int bodyId, bool enabled);

/**
 * Retunes a projectile's mass on a COMPILED model, so the GUI's mass slider changes the ball that actually flies.
 *
 * Sets `body_mass`, the diagonal `body_inertia` of a solid sphere ($I = \tfrac{2}{5} m r^{2}$) and - the part that is
 * easy to miss - refreshes the mass-derived constants with `mj_setConst`.
 *
 * WHY mj_setConst IS NOT OPTIONAL. MuJoCo normalises a contact's reference acceleration by `body_invweight0`, an
 * inverse mass the compiler computes once. Leaving it stale does not make the ball behave like its old mass, and it
 * does not make it behave like its new one either: it changes the CONTACT, so the coefficient of restitution starts
 * depending on the mass. Measured on a ball dropped from the same height, refreshing the constants gives a rebound
 * apex identical to four decimal places at 0.45, 2.0 and 5.0 kg, which is the correct physics - restitution is a
 * property of the contact, not of the ball. Skipping it collapses that apex by more than half and makes it wander
 * non-monotonically with mass. Nothing anywhere reports an error either way.
 *
 * AND WHY THIS TAKES NO mjData. `mj_setConst` uses the mjData it is handed as scratch space and leaves its `qpos`
 * overwritten - on the real Atlas scene, by thousands of radians. Handing it the live simulation state would silently
 * reset the walking robot mid-stride. So this function allocates a throwaway mjData, uses that, and frees it; the
 * live state is not reachable from here and cannot be corrupted by a future caller either. It costs about 0.4 ms on
 * the Atlas scene - two dozen simulation steps - which is why callers should only invoke it when the mass has
 * actually changed, and never inside the stepping loop.
 *
 * `mass` is clamped to kProjectileMassRange. Returns an error for a null model, a body id outside the model, or a
 * body whose id names no projectile-shaped body (no geoms), and does nothing to the model in those cases.
 */
absl::Status setProjectileMass(mjModel* model, int bodyId, double mass, double radius);

/** [m] Where a parked projectile sits: far enough under the floor to be out of sight and out of every camera. */
inline constexpr double kProjectileParkHeight = -50.0;

/**
 * [kg] What the mass slider is allowed to ask for, clamped by setProjectileMass because a topic can be published by
 * hand. The lower bound stays clear of zero: the inertia of a sphere is proportional to its mass and MuJoCo rejects a
 * body with a zero-inertia free joint.
 */
// LINT.IfChange(projectile_mass_range)
inline constexpr double kMinProjectileMass = 0.1;
inline constexpr double kMaxProjectileMass = 5.0;
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/tk_app/dodgeball.py:dodgeball_mass_range)

}  // namespace robot::mujoco_sim_interface
