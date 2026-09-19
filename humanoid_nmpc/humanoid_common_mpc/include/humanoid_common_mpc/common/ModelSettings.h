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
    // Relaxed barrier on h >= 0: `mu` is the barrier weight, `delta` the width of the quadratic relaxation [m].
    scalar_t penetrationMu{1.0e-2};
    scalar_t penetrationDelta{1.0e-3};
    // [m] height of the ground under the contact frames. The reduced-order planner assumes flat terrain; the MPC only
    // needs the ground to be where this says for the complementarity conditions to mean what they should.
    scalar_t terrainHeight{0.0};
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
};

}  // namespace ocs2::humanoid
