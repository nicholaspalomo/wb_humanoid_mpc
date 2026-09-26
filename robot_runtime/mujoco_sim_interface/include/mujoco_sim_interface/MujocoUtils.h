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

#include <mujoco/mujoco.h>

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace robot::mujoco_sim_interface {

struct Metrics {
  /// FPS of Simulation::step
  double fpsSim;
  /// Real time factor for current sim step: RTF = dt_sim / dt_real
  double rtfTick;
  /// Smoothed RTF (exponential moving average)
  double rtfSmoothed;
  /// Time drift per-tick.
  double driftTick;
  /// Total time drift since starting the sim.
  double driftCumulative;

  void reset() {
    fpsSim = 0.0;
    rtfTick = 0.0;
    rtfSmoothed = 1.0;  // Start at ideal value
    driftTick = 0.0;
    driftCumulative = 0.0;
  }
};

struct MjState {
  explicit MjState(const mjModel* model);

  // Deep-copy: allocate new mjData and copy contents
  MjState(const MjState& other);
  MjState& operator=(const MjState& other);

  // Move: transfer ownership of mjData
  MjState(MjState&& other) noexcept;
  MjState& operator=(MjState&& other) noexcept;

  ~MjState();

  const mjModel* model{nullptr};
  int64_t timestamp{0};
  mjData* data{nullptr};
  Metrics metrics;
};

/** One sample of the contact timeline: the contact state the controller planned and the one the physics produced. */
struct ContactTimelineSample {
  double time{0.0};         // [s] simulation time of the sample
  uint32_t target{0};       // bit i set: contact point i is planned to be in contact
  uint32_t actual{0};       // bit i set: contact point i carries contact force against something outside the robot
  bool targetKnown{false};  // false until the controller has provided a planned contact state
};

/**
 * Sliding window of contact samples, oldest first. The simulation thread appends and the render thread copies; the
 * owner provides the locking.
 */
class ContactTimeline {
 public:
  explicit ContactTimeline(double windowSeconds = 5.0) : window_(windowSeconds > 0.0 ? windowSeconds : 5.0) {}

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
bool isInBodySubtree(const mjModel* model, int bodyId, int ancestorId);

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
std::vector<int> resolveContactBodies(const mjModel* model,
                                      const std::string& urdfPath,
                                      const std::vector<std::string>& contactFrameNames,
                                      const std::vector<std::string>& contactParentJointNames,
                                      std::vector<std::string>* errors);

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
uint32_t groundTruthContactMask(
    const mjModel* model, const mjData* data, const std::vector<int>& contactBodyIds, double forceThreshold, int ignoreBodyId = -1);

/** Whole-body centroidal quantities of the floating-base robot of the scene, for the viewer's markers. */
struct RobotCentroidalState {
  bool valid{false};
  int rootBodyId{-1};                    // the first body hanging on a free joint
  double mass{0.0};                      // [kg] of the root's subtree
  double com[3]{0.0, 0.0, 0.0};          // [m] world frame
  double comVelocity[3]{0.0, 0.0, 0.0};  // [m/s] world frame, mass-weighted mean of the body velocities
};

/**
 * Centre of mass and its velocity of the robot: the subtree of the first body on a free joint. Requires a state on which
 * mj_forward() has run. `valid` is false without such a body or with zero mass.
 */
RobotCentroidalState robotCentroidalState(const mjModel* model, const mjData* data);

/** Ground reaction of the physics contacts against the robot's root subtree, and the zero moment point it defines. */
struct GroundReaction {
  bool valid{false};                // a vertical force above `minNormalForce` was found
  double force[3]{0.0, 0.0, 0.0};   // [N] total contact force on the robot, world frame
  double moment[3]{0.0, 0.0, 0.0};  // [N m] of those forces about the world origin
  double zmp[2]{0.0, 0.0};          // [m] point of the ground plane z = 0 about which the horizontal moment vanishes
};

/**
 * Sums the active contact forces between the root subtree of `rootBodyId` and anything outside it (the ground, obstacles)
 * and returns the zero moment point on the plane z = 0: zmp_x = -M_y / F_z, zmp_y = M_x / F_z. Requires the constraint
 * forces of the current state (after mj_step() or mj_forward()). `valid` is false while F_z <= minNormalForce.
 */
GroundReaction groundReaction(const mjModel* model, const mjData* data, int rootBodyId, double minNormalForce);

/**
 * Divergent component of motion (capture point) of a linear inverted pendulum of natural frequency sqrt(gravity / height)
 * on the ground plane: com_xy + v_xy / omega. `height` is clamped to at least 0.05 m.
 */
void divergentComponentOfMotion(const double com[3], const double comVelocity[3], double height, double gravity, double dcm[2]);

}  // namespace robot::mujoco_sim_interface
