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

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "mujoco/mujoco.h"
#include "ocs2_core/Dimensions.h"

#include "mujoco_sim_interface/MjRuntimeModelFields.h"

namespace robot::mujoco_sim_interface {

// The sizes and indices of the vectors the simulator reads and writes, and of the layouts of MuJoCo's model and data
// arrays (mjmodel.h, mjdata.h), which MuJoCo documents but names no constant for. Each is an `int`, the type of MuJoCo's
// own sizes and indices. MuJoCo's world frame has z up. A vector of space and its axes are those of ocs2_core/Dimensions.h:
// ocs2::kPositionDim (a position, a velocity, a force or a moment, and one row of the MuJoCo arrays of them),
// ocs2::kHorizontalPositionDim (a point of the ground plane) and ocs2::kZAxisIndex.

/** A row-major 3x3 matrix: one row of xmat and geom_xmat, a contact frame, a decor geom's orientation. */
inline constexpr int kMjMatrixSize = ocs2::kPositionDim * ocs2::kPositionDim;
/**
 * A quaternion as MuJoCo stores it, (w, x, y, z): one row of xquat. Eigen and the MPC store the same coefficients as
 * (x, y, z, w) instead, so only the count is ocs2's.
 */
inline constexpr int kMjQuaternionSize = ocs2::kQuaternionCoefficients;
/**
 * A MuJoCo 6D vector of two vectors of space: a force and a torque (xfrc_applied, mj_contactForce), or an angular and a
 * linear velocity (mj_objectVelocity).
 */
inline constexpr int kMjSpatialVectorSize = ocs2::kTwistDim;
/**
 * A MuJoCo motion vector (mj_objectVelocity, a row of mjData::cvel) is (angular, linear): its linear velocity starts
 * after the angular one.
 */
inline constexpr int kMjMotionLinearOffset = ocs2::kRotationTangentDim;
/**
 * The rows of a contact frame (mjContact::frame, a row-major 3x3 matrix whose rows are the frame's axes in the world
 * frame), and with them the components of a contact force in that frame (mj_contactForce): the normal, then the two
 * tangents.
 */
inline constexpr int kMjContactNormalRow = 0;
inline constexpr int kMjContactFirstTangentRow = 1;
inline constexpr int kMjContactSecondTangentRow = 2;
/** The red, green, blue and alpha channels of a MuJoCo color (mjvGeom::rgba, mjsGeom::rgba, geom_rgba), at these indices. */
inline constexpr int kMjRgbaChannels = 4;
inline constexpr int kRedChannel = 0;
inline constexpr int kGreenChannel = 1;
inline constexpr int kBlueChannel = 2;
inline constexpr int kAlphaChannel = 3;
/**
 * The size parameters of a geom (geom_size, mjvGeom::size, mjsGeom::size), whose meaning its type gives: a sphere's
 * radius is the first.
 */
inline constexpr int kMjGeomSizeParameters = 3;

/** The coordinates of a free joint in qpos: its position, then its quaternion (w, x, y, z) at the indices below. */
inline constexpr int kMjFreeJointQposSize = ocs2::kPositionDim + kMjQuaternionSize;
inline constexpr int kMjFreeJointQuaternionWIndex = ocs2::kPositionDim;
inline constexpr int kMjFreeJointQuaternionXIndex = kMjFreeJointQuaternionWIndex + 1;
inline constexpr int kMjFreeJointQuaternionYIndex = kMjFreeJointQuaternionXIndex + 1;
inline constexpr int kMjFreeJointQuaternionZIndex = kMjFreeJointQuaternionYIndex + 1;
/**
 * The degrees of freedom of a free joint in qvel: its linear velocity (world frame), then, from
 * kMjFreeJointAngularVelocityOffset, its angular velocity (body frame). The joints of a robot on a free joint follow
 * them.
 */
inline constexpr int kMjFreeJointDofs = ocs2::kPositionDim + ocs2::kRotationTangentDim;
inline constexpr int kMjFreeJointAngularVelocityOffset = ocs2::kPositionDim;
/** A body's mean inverse inertias at qpos0 in body_invweight0: the translational one, then the rotational one. */
inline constexpr int kMjBodyInvweightSize = 2;

/** The most contact points a contact mask (groundTruthContactMask()) covers: one bit of a uint32_t each. */
inline constexpr int kContactMaskCapacity = std::numeric_limits<uint32_t>::digits;

/** Deletes an mjModel, so that a std::unique_ptr owns it (MjModelPtr). */
struct MjModelDeleter {
  void operator()(mjModel* absl_nonnull model) const { mj_deleteModel(model); }
};

/** Deletes an mjData, so that a std::unique_ptr owns it (MjDataPtr). */
struct MjDataDeleter {
  void operator()(mjData* absl_nonnull data) const { mj_deleteData(data); }
};

/** Deletes an mjSpec, so that a std::unique_ptr owns it (MjSpecPtr). */
struct MjSpecDeleter {
  void operator()(mjSpec* absl_nonnull spec) const { mj_deleteSpec(spec); }
};

using MjModelPtr = std::unique_ptr<mjModel, MjModelDeleter>;
using MjDataPtr = std::unique_ptr<mjData, MjDataDeleter>;
using MjSpecPtr = std::unique_ptr<mjSpec, MjSpecDeleter>;

/**
 * The simulation data of `model`, owned. Fails with InternalError where mj_makeData returns null. MuJoCo 3.3 aborts by
 * itself when it cannot allocate, so that is not expected, but its API does not promise a non-null result.
 */
absl::StatusOr<MjDataPtr> makeMjData(const mjModel* absl_nonnull model);

/**
 * Timing of the simulation loop, as the viewer's metrics overlay shows it. Passive data, written by the simulation
 * thread and copied to the render thread with the state it belongs to (MjState).
 */
struct Metrics {
  /// FPS of Simulation::step
  double fpsSim = 0.0;
  /// Real time factor for current sim step: RTF = dt_sim / dt_real
  double rtfTick = 0.0;
  /// Smoothed RTF (exponential moving average)
  double rtfSmoothed = 0.0;
  /// Time drift per-tick.
  double driftTick = 0.0;
  /// Total time drift since starting the sim.
  double driftCumulative = 0.0;

  void reset() {
    fpsSim = 0.0;
    rtfTick = 0.0;
    rtfSmoothed = 1.0;  // Start at ideal value
    driftTick = 0.0;
    driftCumulative = 0.0;
  }
};

/**
 * A snapshot of the physics state for the render thread: its own mjData of `model`, which it owns, plus the clock and the
 * metrics of the step it was taken at, and the model arrays the simulator writes while it runs (MjRuntimeModelFields),
 * for a viewer that draws with a private copy of the model. Copies are deep (a new mjData, with the contents copied); a
 * moved-from MjState has no model, no data and no model fields. Not thread-safe: the simulator hands snapshots between
 * threads through a triple buffer.
 *
 * `data` is never null in an MjState constructed from a model or copied from one that has data. A null from mj_makeData
 * is not expected (see makeMjData(), which MujocoSimInterface::Create() uses for the simulator's own data), so it aborts
 * here (ABSL_DIE_IF_NULL) rather than leaving a snapshot whose first reader dereferences it.
 */
/**
 * The last push the simulator applied to a body through mjData::xfrc_applied (the dodgeball's impulse when no ball is
 * compiled into the scene), kept for the viewer: the force itself lasts one physics step, which a render snapshot
 * rarely holds. Passive data.
 */
struct AppliedPush {
  int bodyId = -1;                                                        // the pushed body, -1 for none yet
  std::array<double, ocs2::kPositionDim> impulseWorld = {0.0, 0.0, 0.0};  // [N s] world frame
  double simTimeApplied = 0.0;                                            // [s] simulation time of the push
};

// LINT.IfChange(mj_state_fields)
struct MjState {
  explicit MjState(const mjModel* absl_nonnull model);

  // Deep-copy: allocate new mjData and copy contents
  MjState(const MjState& other);
  MjState& operator=(const MjState& other);

  // Move: transfer ownership of mjData
  MjState(MjState&& other) noexcept;
  MjState& operator=(MjState&& other) noexcept;

  ~MjState();

  const mjModel* absl_nullable model = nullptr;
  int64_t timestamp = 0;
  mjData* absl_nullable data = nullptr;  // owned; null only when moved from
  Metrics metrics;
  MjRuntimeModelFields modelFields;  // sized for `model` at construction; empty only when moved from
  AppliedPush push;                  // the simulator's last push, at the step of the snapshot
};
// clang-format off
// LINT.ThenChange(//robot_runtime/mujoco_sim_interface/src/MujocoContactUtils.cpp:mj_state_fields, //robot_runtime/mujoco_sim_interface/src/MujocoSimInterface.cpp:mj_state_fields)
// clang-format on

/** One sample of the contact timeline: the contact state the controller planned and the one the physics produced. */
struct ContactTimelineSample {
  double time = 0.0;         // [s] simulation time of the sample
  uint32_t target{0};        // bit i set: contact point i is planned to be in contact
  uint32_t actual{0};        // bit i set: contact point i carries contact force against something outside the robot
  bool targetKnown = false;  // false until the controller has provided a planned contact state
};

/**
 * Sliding window of contact samples, oldest first. The simulation thread appends and the render thread copies; the
 * owner provides the locking.
 */
class ContactTimeline {
 public:
  /** [s] The window of a timeline constructed without one, or with one that is not positive. */
  static constexpr double kDefaultWindowSeconds = 5.0;

  explicit ContactTimeline(double windowSeconds = kDefaultWindowSeconds)
      : window_(windowSeconds > 0.0 ? windowSeconds : kDefaultWindowSeconds) {}

  double window() const { return window_; }

  /** Appends a sample and drops the ones older than the window. A time that runs backwards (simulation reset) clears the history. */
  void append(const ContactTimelineSample& sample);

  void clear() { samples_.clear(); }

  const std::deque<ContactTimelineSample>& samples() const { return samples_; }

 private:
  double window_;
  std::deque<ContactTimelineSample> samples_;
};

/** True if `bodyId` is `ancestorId` or lies in its kinematic subtree. */
bool isInBodySubtree(const mjModel* absl_nonnull model, int bodyId, int ancestorId);

/**
 * MuJoCo body of every contact point, in the order of `contactFrameNames`, or -1 where none could be found (with a
 * message appended to `errors`). Resolution order per contact point:
 *  1. the body driven by `contactParentJointNames[i]` when that entry is given (the controller's contact frames are
 *     usually added to its own kinematic model on such a joint and exist in neither the URDF nor the scene);
 *  2. a MuJoCo body named like the frame;
 *  3. the URDF walked up through fixed joints from the frame's link, since the MJCF conversion merges fixed-joint
 *     children into their parent body. A link that hangs on a movable joint but is no body of the scene is an error,
 *     not a guess.
 */
std::vector<int> resolveContactBodies(const mjModel* absl_nonnull model,
                                      const std::string& urdfPath,
                                      const std::vector<std::string>& contactFrameNames,
                                      const std::vector<std::string>& contactParentJointNames,
                                      std::vector<std::string>* absl_nullable errors);

/**
 * Ground-truth contact mask: bit i is set when a geom of contact body i (or of its subtree) is in a contact that carries
 * more than `forceThreshold` [N] of normal force against a geom outside the robot's kinematic tree (the world, or any
 * body that does not share the contact body's root). Requires the constraint forces of the current state, i.e. call
 * after mj_step() or mj_forward(). Contacts beyond the 32nd are ignored.
 */
/**
 * Bit per contact point, set where that point is carrying more than `forceThreshold` of normal force against
 * something that is not the robot.
 *
 * `ignoreBodyId` is left out of that "something": a body listed there is neither the robot nor the ground, so a
 * contact with it sets no bit. It exists for the thrown projectile. Without it a ball resting against a SWING foot
 * would be reported to the controller as that foot being planted - the mask feeds the RobotState's contact flags and
 * the cheater_sim estimator - which is a far worse disturbance than the ball itself, and an invisible one.
 */
uint32_t groundTruthContactMask(const mjModel* absl_nonnull model,
                                const mjData* absl_nonnull data,
                                const std::vector<int>& contactBodyIds,
                                double forceThreshold,
                                int ignoreBodyId = -1);

/** Whole-body centroidal quantities of the floating-base robot of the scene, for the viewer's markers. */
struct RobotCentroidalState {
  bool valid = false;
  int rootBodyId = -1;                                    // the first body hanging on a free joint
  double mass = 0.0;                                      // [kg] of the root's subtree
  double com[ocs2::kPositionDim]{0.0, 0.0, 0.0};          // [m] world frame
  double comVelocity[ocs2::kPositionDim]{0.0, 0.0, 0.0};  // [m/s] world frame, mass-weighted mean of the body velocities
};

/**
 * Center of mass and its velocity of the robot: the subtree of the first body on a free joint. Requires a state on which
 * mj_forward() has run. `valid` is false without such a body or with zero mass.
 */
RobotCentroidalState robotCentroidalState(const mjModel* absl_nullable model, const mjData* absl_nullable data);

/** Ground reaction of the physics contacts against the robot's root subtree, and the zero moment point it defines. */
struct GroundReaction {
  bool valid = false;                                  // a vertical force above `minNormalForce` was found
  double force[ocs2::kPositionDim]{0.0, 0.0, 0.0};     // [N] total contact force on the robot, world frame
  double moment[ocs2::kPositionDim]{0.0, 0.0, 0.0};    // [N m] of those forces about the world origin
  double zmp[ocs2::kHorizontalPositionDim]{0.0, 0.0};  // [m] point of the ground plane z = 0 about which the horizontal moment vanishes
};

/**
 * Sums the active contact forces between the root subtree of `rootBodyId` and anything outside it (the ground, obstacles)
 * and returns the zero moment point on the plane z = 0: zmp_x = -M_y / F_z, zmp_y = M_x / F_z. Requires the constraint
 * forces of the current state (after mj_step() or mj_forward()). `valid` is false while F_z <= minNormalForce.
 *
 * Contacts with `ignoreBodyId` are left out, as groundTruthContactMask leaves them out: a thrown ball striking the
 * robot is not a ground reaction, and at 25 m/s its contact force alone exceeds the robot's weight.
 */
GroundReaction groundReaction(
    const mjModel* absl_nullable model, const mjData* absl_nullable data, int rootBodyId, double minNormalForce, int ignoreBodyId = -1);

/**
 * Divergent component of motion (capture point) of a linear inverted pendulum of natural frequency sqrt(gravity / height)
 * on the ground plane: com_xy + v_xy / omega. `height` is clamped to at least 0.05 m.
 */
void divergentComponentOfMotion(const double com[absl_nonnull ocs2::kPositionDim],
                                const double comVelocity[absl_nonnull ocs2::kPositionDim],
                                double height,
                                double gravity,
                                double dcm[absl_nonnull ocs2::kHorizontalPositionDim]);

// LINT.IfChange(foot_force_frame)
/**
 * The force that the foot force sensor `sensorId` (a `<force>` sensor on a site of the foot) measures, as the force the
 * foot carries in the world frame [N]: the ground reaction on the foot less the foot's own weight and inertia, +z up
 * while the foot stands, the sign of the MPC's contact forces. MuJoCo reads the force between the site's body and its
 * parent in the site frame, pointing from the parent's side (sensor.md): it is rotated into the world frame and negated.
 * Requires the sensor data of the current state (after mj_step() or mj_forward()); `sensorId` must name a force sensor
 * on a site (isFootForceSensor()).
 */
std::array<mjtNum, ocs2::kPositionDim> footForceInWorldFrame(const mjModel* absl_nonnull model,
                                                             const mjData* absl_nonnull data,
                                                             int sensorId);
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_msgs/robot_state_sample.proto:measured_contact_wrenches)

/** Whether `sensorId` is a `<force>` sensor on a site, as footForceInWorldFrame() needs. */
bool isFootForceSensor(const mjModel* absl_nonnull model, int sensorId);

/**
 * The rotation of a base attitude about the world vertical: the twist q_t = (w_t, 0, 0, z_t) of the twist-swing split
 * q = q_t (x) s (the twist heading of design decision D9), the nearest upright attitude. Passive data.
 */
struct VerticalTwist {
  double w = 1.0;  // cos(heading / 2), >= 0
  double z = 0.0;  // sin(heading / 2)
};

/**
 * The twist about the world z axis of a MuJoCo quaternion (w, x, y, z), from its w and z (the tilt components x, y do
 * not enter it): the rotation headingQuaternion() of
 * humanoid_common_mpc/orientation/BaseOrientation.h gives (design section 2.8.1), brought to w >= 0 (w = 0 < z at a
 * heading of pi): (w, z) / |(w, z)| for w > 0. It is computed from the angle atan2(z, w) rather than by the division, so
 * it is continuous down to an upside-down attitude |(w, z)| = 0, where the heading is undefined and the result is the
 * identity (heading 0), as for the MPC. Invariant to the scale of (w, z).
 */
VerticalTwist verticalTwistOf(double w, double z);

/** [rad] The twist heading 2 atan2(z_t, w_t) of verticalTwistOf(w, z), in (-pi, pi]. */
double twistHeadingOf(double w, double z);

}  // namespace robot::mujoco_sim_interface
