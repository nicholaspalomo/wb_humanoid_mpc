/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

class ModelSettings {
 public:
  struct FootConstraintConfig {
    scalar_t positionErrorGain_z{1.0};
    scalar_t orientationErrorGain{1.0};
    scalar_t linearVelocityErrorGain_z{1.0};
    scalar_t linearVelocityErrorGain_xy{1.0};
    scalar_t angularVelocityErrorGain{1.0};
    scalar_t linearAccelerationErrorGain_z{1.0};
    scalar_t linearAccelerationErrorGain_xy{1.0};
    scalar_t angularAccelerationErrorGain{1.0};
    scalar_t softConstraintWeight{10.0};
    /**
     * Weight of the SOFT `normal_velocity` term, when it is listed in soft_constraints.
     *
     * The residual is the same row the hard constraint imposed,
     *     r = v_z - zdot_ref(t) - positionErrorGain_z * (z_ref(t) - z),
     * so this one weight buys the swing-foot vertical servo as a cost. That matters because the two channels the hard
     * row combined are otherwise split across task_space_foot_cost_weights.pos_z and .lin_velocity_z, which are faded
     * differently (only the velocity rows of the foot cost are scaled by the impact-proximity factor) and so cannot
     * reproduce the servo at any pair of values.
     *
     * Sizing: the residual is a velocity in m/s and is essentially affine in the input, because the joint velocities
     * ARE inputs of the centroidal model - so a large Gauss-Newton weight here is well conditioned. A foot stuck on
     * the ground at mid-swing has r = -positionErrorGain_z * swingHeight = -0.16 m/s, and the term must outbid the
     * leg-joint regularization that holds the swing leg in its standing crouch. Sweep it; the shipped value is a
     * starting point, not a tuned one.
     */
    scalar_t normalVelocitySoftConstraintWeight{500.0};
    bool constrainOrientation{true};  // When true, constraint is 6D (position+orientation); when false, 3D (position-only)
    // The orientation error with respect to the ground plane only measures the tilt of the foot normal, so a 6D
    // constraint built from it leaves the rotation about the contact normal free and the last row is identically zero.
    // Setting this adds that rate to the plane-normal row, which stops a stance foot pivoting on the spot. It is off by
    // default because it removes one input degree of freedom per stance foot from a controller that was tuned without
    // it; enable it and re-check the yaw behaviour.
    bool constrainYawRateAboutContactNormal{false};
  };

  /**
   * The nominal foothold reference, used only when no contact planner supplies one.
   *
   * Without a planner the swing foot has no horizontal target at all: the foot cost switches its xy position weights
   * off, and the only thing that ever fixed foot placement was the stance foot being pinned by the zero_velocity
   * constraint. The contact-implicit formulation removes that pin by design, so with both off nothing in the problem
   * has an opinion about where the feet go sideways and they drift together until the robot falls.
   *
   * A positive `stepWidth` here restores a horizontal target: the foot is placed that far to its own side of the
   * reference base pose, and follows that pose as it advances, so forward placement still comes from the commanded
   * motion rather than from a second heuristic. It is a foot placement heuristic, which is exactly what the
   * reduced-order planner exists to avoid, so it is off by default and is meant for testing the contact-implicit
   * formulation on its own before the planner is enabled on top of it.
   */
  struct NominalFootholdConfig {
    scalar_t stepWidth{0.0};  // [m] lateral distance between the feet; 0 disables the nominal reference
  };

  /**
   * Weights of the relaxed complementarity formulation of contact, active when contact_complementarity,
   * force_weighted_slip and ground_penetration are listed in the task file's soft_constraints
   * (humanoid_nmpc/docs/contact_implicit_mpc/README.md). They price the three conditions of rigid contact rather than
   * imposing them, which is what lets the solver choose the contact sequence.
   */
  struct ContactImplicitConfig {
    // Both residuals are normalised before they are penalised - (f_n / f_ref)(h / h_ref) and (f_n / f_ref)(v / v_ref) -
    // so these two weights are dimensionless and directly comparable with the task-space weights they compete against.
    // Each is the cost of the worst configuration its term can describe: a foot at the reference height, or sliding at
    // the reference speed, while carrying the reference force. See the class comment on ContactComplementarityConstraint
    // for why the un-normalised products could not be weighted sensibly at all.
    scalar_t complementarityWeight{100.0};
    // What holds a loaded foot still, in place of the mode-scheduled stance constraint.
    scalar_t slipWeight{100.0};
    // The references the two residuals are measured in. The force reference is not here: it is the robot's own weight,
    // taken from the model, because a value that has to agree with the URDF should not be maintained by hand.
    scalar_t heightReference{0.08};          // [m] normally the swing apex, swing_trajectory_config.swingHeight
    scalar_t velocityReference{0.3};         // [m/s] a sliding speed that would already be a failure
    scalar_t angularVelocityReference{1.0};  // [rad/s] a pivot rate that would already be a failure
    /**
     * Weight of the one-sided quadratic hinge on h >= 0: the penalty is `penetrationWeight * h^2 / 2` below the ground
     * and exactly zero on or above it.
     *
     * This used to be a relaxed log barrier, which is the wrong object for a unilateral condition whose solution lies
     * ON the boundary. A log barrier never reaches zero: with the values that shipped here (mu = 0.1, delta = 0.01)
     * its derivative at h = 0 is -2*mu/delta = -20, constant and upward, so every foot was pushed off the ground and
     * the only term pulling it back was the complementarity penalty, whose gradient in h is
     * `complementarityWeight * (f_n/f_ref)^2 * h / heightReference^2`. Balancing the two put a fully loaded foot 2.6 mm
     * above the ground and a foot at half body weight - i.e. BOTH feet, throughout double support - a centimetre above
     * it, with a permanent complementarity residual that no weight could tune away. The hinge is zero in value and in
     * gradient at h = 0, so it has no such equilibrium: it does nothing at all until the foot is actually below ground.
     */
    scalar_t penetrationWeight{5.0e4};
    /**
     * [m] The length scale over which the gap - the height of the lowest point of the footprint - is smoothed.
     *
     * The complementarity term measures the gap at the FOOTPRINT CORNERS, the same points the penetration hinge uses,
     * because a foot rocked onto its heel is carrying load while its sole centre is well clear of the ground. The
     * exact minimum of the four corner heights is not differentiable at a flat foot, which is where the robot spends
     * most of its stance, so smoothMinimumHeight() blends it over this scale. The bound it returns never falls below
     * the true minimum and exceeds it by `gapSmoothing * log(4 / k)` where k is how many corners sit at the minimum:
     * zero for a flat foot, 0.69 mm for an edge down (the ordinary heel strike), and 1.39 mm in the worst case of a
     * single corner. Against the penetration hinge those settle at about 0.09 mm and 0.19 mm of equilibrium
     * penetration under full body weight - the price of a residual the SQP solver can linearise consistently.
     */
    scalar_t gapSmoothing{1.0e-3};
    // Where the ground is, is NOT here: it is ModelSettings::terrainHeight, so that the complementarity conditions and
    // the swing trajectories cannot disagree about it.
  };

  ModelSettings(const std::string& configFile, const std::string& urdfFile, const std::string& mpcName, bool verbose = false);

  ModelSettings() = delete;

  ModelSettings(const ModelSettings&) = delete;

 public:
  std::string robotName;

  bool verboseCppAd = true;
  bool recompileLibrariesCppAd = true;
  std::string modelFolderCppAd = "build/cppad_autocode_gen";

  scalar_t phaseTransitionStanceTime;

  // Fixed joints , add from the fullJointNames to consider them as fixed in the MPC
  std::vector<std::string> fullJointNames;
  std::vector<std::string> fixedJointNames;

  std::vector<std::string> contactNames6DoF;
  std::vector<std::string> contactNames3DoF{};
  std::vector<std::string> contactParentJointNames;

  std::vector<std::string> mpcModelJointNames;      // Active joints (all joints except the fixed ones)
  std::vector<size_t> mpcModelToFullJointsIndices;  // an Array of indices mapping the active joints to the full joints
  std::unordered_map<std::string, size_t> jointIndexMap;
  std::vector<std::string> contactNames;  // containing all 3Dof and 6Dof contacts

  bool useContactBasisVectorInputs = false;
  bool useComAndAcomTracking = false;
  bool useContactPlanning = false;  // mode schedule and footholds from the mixed-integer contact planner instead of the gait schedule
  bool useDcmTerminalCost = false;  // DCM viability terminal cost instead of the quadratic Q_final terminal cost

  size_t mpc_joint_dim;
  size_t full_joint_dim;

  std::string j_l_shoulder_y_name;
  std::string j_r_shoulder_y_name;
  std::string j_l_elbow_y_name;
  std::string j_r_elbow_y_name;

  size_t j_l_shoulder_y_index;
  size_t j_r_shoulder_y_index;
  size_t j_l_elbow_y_index;
  size_t j_r_elbow_y_index;

  FootConstraintConfig footConstraintConfig;
  ContactImplicitConfig contactImplicitConfig;
  NominalFootholdConfig nominalFootholdConfig;

  /**
   * [m] Where the ground is, for every part of the controller that needs to know.
   *
   * There used to be two answers. SwitchedModelReferenceManager::adaptToCurrentGroundHeight() computed an estimate
   * from the stance feet and then threw it away on the next line, silently returning 0 to the swing trajectory planner
   * and the landing targets; the contact-implicit terms took their own `contact_implicit.terrainHeight`, also 0. They
   * agreed only because both were pinned to the same constant, and nothing said they had to.
   *
   * This key is now the single source, read by the reference manager, the swing trajectories, the landing targets and
   * the complementarity and ground-penetration terms alike. It is a constant because every part of this stack assumes
   * flat ground - the reduced-order contact planner most of all. If a measured estimator is ever wanted, it belongs
   * here as a named selection (a registry resolving e.g. `fixed` / `from_contacts`), not as a second constant
   * somewhere else.
   */
  scalar_t terrainHeight{0.0};
};

}  // namespace ocs2::humanoid
