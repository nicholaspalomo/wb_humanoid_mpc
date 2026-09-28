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
 * A value type rather than a class family, because a projectile has no behavior of its own - the simulator throws
 * it and MuJoCo does the rest. What it has is the handful of numbers that decide what being hit by it feels like.
 * Every field is set by projectileFromName; a default-constructed Projectile has a zero radius and mass and is
 * deliberately NOT a ball, so that addProjectileToSpec rejects one that was never looked up.
 */
struct Projectile {
  std::string name;
  double radius{0.0};           // [m]
  double mass{0.0};             // [kg] nominal; the GUI's mass slider overrides it per throw
  double restitution{0.0};      // [-] coefficient of restitution against a rigid surface, in [0, 1)
  double friction{0.0};         // [-] sliding friction of the ball's surface
  double rollingFriction{0.0};  // [m] rolling (and torsional) resistance, which is what brings a rolling ball to rest
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
 * That relation only holds for the contact the ball ACTUALLY gets, which is why addProjectileToSpec also gives the
 * ball contact priority and a time constant the integrator can resolve; see kProjectileContactTimeConstant.
 *
 * Clamped to a hair below 1 and above 0: a restitution of exactly 1 is a contact that never settles, and one of 0 is
 * the critically damped default, which is what MuJoCo uses when it is given nothing.
 */
double contactDampRatioForRestitution(double restitution);

/**
 * [s] Contact time constant of a projectile, the first element of its solref.
 *
 * It has to be long compared with the simulator's step, or the logarithmic-decrement mapping above is not what the
 * integrator produces. A lightly damped contact has a natural frequency of about 1 / (timeconst * dampratio), so at
 * the 0.07 damping ratio of a lively ball a 2 ms time constant rings at ~7000 rad/s - seven radians per 1 ms step, and
 * the discrete contact then ADDS energy (measured: rebounds of 4 to 50 times the arrival speed). At 20 ms it is ~740
 * rad/s, well resolved at both the 0.5 ms default step and 1 ms, and the measured restitution is 0.797 and 0.842
 * against a target of 0.80 on the shipped Atlas and SA01 scenes. Twenty milliseconds is also about the contact time
 * of a foam ball hitting something hard.
 */
inline constexpr double kProjectileContactTimeConstant = 0.02;

/**
 * Appends `projectile` to `spec` as a free-floating sphere, parked out of play until thrown.
 *
 * APPENDED, AND THAT MATTERS. It must be the LAST body in the world, because several things in this simulator take
 * the FIRST body carrying a free joint to be the robot: the center of mass, the ZMP and DCM markers
 * (MujocoContactUtils::robotCentroidalState), the viewer's tracking camera, and every read of `qpos[3..6]` as the
 * base quaternion. A ball inserted ahead of the robot would quietly become the robot.
 *
 * Three properties are set at COMPILE time because MuJoCo only honors them if the compiler saw them, and toggling
 * them afterwards silently does nothing:
 *
 * - `gravcomp = 1`, so the compiler counts the body in `mjModel::ngravcomp`. MuJoCo skips gravity compensation
 *   entirely when that count is zero, so a ball compiled without it and given `body_gravcomp = 1` at runtime free-falls
 *   anyway - for hours, until its position overflows and MuJoCo resets the whole simulation.
 * - `contype = conaffinity = 1`; see setProjectileCollisionEnabled for the broadphase aggregates this feeds.
 * - `priority = 1`, so the BALL's solref, solimp, friction and condim govern every contact it makes. Without it MuJoCo
 *   averages the ball's solref with the floor's and the robot's (0.004, 1.0), and the contact that results has a
 *   restitution of about 0.25 whatever the ball was configured with.
 *
 * condim 6 with rolling and torsional friction, so that a ball rolling across the floor comes to rest (in about seven
 * seconds from 3 m/s) instead of rolling on forever at 5/7 of its speed, as a condim-3 sphere does.
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
 * [kg] What the mass slider is allowed to ask for. A topic can be published by hand, so everything on the simulator
 * side goes through clampProjectileMass. The lower bound stays clear of zero because a sphere's inertia is
 * proportional to its mass, and MuJoCo cannot integrate a free body with zero inertia.
 */
// LINT.IfChange(projectile_mass_range)
inline constexpr double kMinProjectileMass = 0.1;
inline constexpr double kMaxProjectileMass = 5.0;
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/tk_app/dodgeball.py:dodgeball_mass_range)

/** `mass` clamped to [kMinProjectileMass, kMaxProjectileMass]; NaN, which every comparison lets through, maps to the minimum. */
double clampProjectileMass(double mass);

/**
 * Retunes a projectile's mass on a COMPILED model, so the GUI's mass slider changes the ball that actually flies.
 *
 * `mass` is clamped with clampProjectileMass. Sets `body_mass`, the diagonal `body_inertia` of a solid sphere
 * (I = 2/5 m r^2, with r read from the ball's geom) and every constant the compiler derived from them:
 * `body_invweight0`, `dof_invweight0`, `dof_M0` and `body_subtreemass` (the ball's own and the world's).
 *
 * WHY THE DERIVED CONSTANTS MATTER. MuJoCo normalizes a contact's reference acceleration by `body_invweight0`, an
 * inverse mass the compiler computed once. Leaving it stale does not make the ball behave like its old mass or its new
 * one: it changes the CONTACT, so the coefficient of restitution starts depending on the mass. With the constants
 * refreshed the rebound is identical at 0.1, 0.45 and 5 kg, which is the correct physics.
 *
 * WHY THEY ARE WRITTEN IN CLOSED FORM rather than by mj_setConst. For a free sphere that is its own kinematic tree,
 * centered on its body frame, each constant is a one-line function of m and I, and a direct store is what the compiler
 * would have produced (tested against a fresh compile). mj_setConst, by contrast, (a) overwrites the qpos of the mjData
 * it is handed as scratch, so it must never see the live state; (b) recomputes `mjModel::stat` from the qpos0 bounding
 * box, discarding the scene's own <statistic extent/center> and moving the viewer's near clip plane from 1.5 cm to
 * half a meter; and (c) rebuilds several fields by accumulating into them in place, so the render thread, which
 * reads the same model, would see partial sums. Here every field is a single store, the robot's constants and
 * `mjModel::stat` are untouched, and nothing is allocated, so the simulator can call this on every throw.
 *
 * Returns an error, and changes nothing, for a null model, a body id outside the model, or a body that is not a
 * projectile - exactly one free joint and one sphere, a direct child of the world, centered on its body frame, which is
 * what addProjectileToSpec builds. That check is strict on purpose: a mistargeted call would otherwise retune a robot
 * link and nothing downstream would notice.
 */
absl::Status setProjectileMass(mjModel* model, int bodyId, double mass);

/** True when `dof` is one of the six free dofs of the projectile body `projectileBodyId` (false for a negative id). */
bool isProjectileDof(const mjModel* model, int projectileBodyId, int dof);

/**
 * Sets the joint damping of every ROBOT joint - dofs 6 to nv, after the robot's own free joint - to `damping`, and
 * leaves the projectile's six dofs alone.
 *
 * The one place the simulator changes joint damping, so that the skip cannot drift out of step between the start-up
 * default, the zero-torque ragdoll boost and disableTorques(), as it once did: 20 N s/m on a thrown ball's free joint
 * brings it to a 0.2 m/s terminal speed within a few centimeters of where it was thrown from.
 */
void setRobotJointDamping(mjModel* model, int projectileBodyId, double damping);

/** A throw in world coordinates: where the ball starts, how fast, and how long until it reaches the base. */
struct ProjectileLaunch {
  std::array<double, 3> position{{0.0, 0.0, 0.0}};  // [m] world frame
  std::array<double, 3> velocity{{0.0, 0.0, 0.0}};  // [m/s] world frame
  double flightTime{0.0};                           // [s] until it reaches the base it was aimed at
};

/** [m] How far a spawning ball must be from every other collidable geom, the floor included. */
inline constexpr double kProjectileSpawnClearance = 0.02;

/**
 * The launch actually used for a requested throw: the requested one when the ball would start clear of everything,
 * otherwise the nearest point ON ITS OWN BALLISTIC PATH where it does.
 *
 * The GUI aims at the base without knowing the robot's shape or where the floor is, so a requested spawn point can
 * lie inside the robot (a short throw from the side starts inside an outstretched arm) or under the floor (a
 * negative elevation from far away). MuJoCo resolves a ball that starts inside something as a stiff spring and fires
 * it out at tens of meters per second, whatever speed was asked for. Sliding the start point along the ball's own
 * path, p(tau) = p0 + v0 tau - g tau^2 / 2 z and v(tau) = v0 - g tau z, keeps the throw aimed exactly where the GUI
 * aimed it: backwards in time (tau < 0) moves the start out of an arm along the approach, forwards (tau > 0) lifts it
 * out of the floor along the climb. The smallest |tau| that clears is taken, backwards up to one second and forwards
 * up to 80% of the flight, and `flightTime` is shortened or lengthened by tau accordingly.
 *
 * "Clear" is `kProjectileSpawnClearance` from every geom the armed ball could collide with, measured with
 * mj_geomDistance against the geom poses in `data`, which must be current (as they are after mj_step or mj_forward).
 * To measure, this writes the ball geom's position in `data->geom_xpos`; the caller places the ball for real
 * afterwards, and the next mj_step recomputes it either way.
 *
 * Returns FailedPreconditionError, naming the slider to change, when no point in that window clears, and
 * InvalidArgumentError when `projectileBodyId` is not a projectile.
 */
absl::StatusOr<ProjectileLaunch> clearProjectileLaunch(
    const mjModel* model, mjData* data, int projectileBodyId, const ProjectileLaunch& requested, double gravity);

/**
 * The momentum a ball carries when it reaches the base: m (v_launch - g t z), with the mass clamped by
 * clampProjectileMass. What the simulator applies to the base when no ball is compiled into the scene. A real ball
 * that bounces off delivers more than this - up to (1 + e) times as much - so the fallback is a lower bound on the
 * push, not an equivalent of it.
 */
std::array<double, 3> projectileArrivalMomentum(const std::array<double, 3>& launchVelocity,
                                                double flightTime,
                                                double mass,
                                                double gravity);

/**
 * Decides when a thrown ball is finished with and should be parked again, so the robot cannot trip over it later and
 * the next throw starts clean.
 *
 * Two conditions, either of which parks it: the ball has been slower than kRestSpeed for kRestDuration, or it has
 * been in play for kLifetime. The second is not a formality. Speed alone is not a reliable signal - a ball that
 * lodges against a moving foot, or rolls on a slope, may never be slow for long - and a ball left in play forever
 * pulls the world's center of mass around and can trip the robot minutes later.
 */
class ProjectileRestMonitor {
 public:
  static constexpr double kRestSpeed = 0.15;     // [m/s]
  static constexpr double kRestDuration = 0.25;  // [s]
  static constexpr double kLifetime = 10.0;      // [s] a few bounces and a roll to rest, with margin

  /** Starts timing a newly thrown ball. */
  void start();

  /** Advances by `dt` at linear speed `speed`, and returns true once the ball should be parked. */
  bool update(double speed, double dt);

 private:
  double slowTime_{0.0};
  double timeInPlay_{0.0};
};

/**
 * [m] Where a parked projectile sits: 100 m ABOVE the origin, held there by its compiled-in gravity compensation.
 *
 * Above rather than below the floor, because every floor in these scenes is a MuJoCo plane and a plane collides as a
 * half-space: a ball parked underneath it is 50 m deep inside the ground. That is harmless while its collisions are
 * off, but the viewer renders the live model against a state snapshot that can be a frame old, and in the frame
 * after a throw arms the model it would see a ball 50 m under the floor with collisions on - a contact of millions of
 * newtons, drawn as a kilometers-long force arrow. Up here nothing is within reach, and it is beyond the far clipping
 * plane of every camera looking at the robot.
 */
inline constexpr std::array<double, 3> kProjectileParkPosition{{0.0, 0.0, 100.0}};

}  // namespace robot::mujoco_sim_interface
