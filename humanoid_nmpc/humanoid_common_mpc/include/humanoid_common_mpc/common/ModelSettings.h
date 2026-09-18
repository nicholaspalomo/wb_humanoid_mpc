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
   * Weights of the relaxed complementarity formulation of contact, active when contact_complementarity,
   * force_weighted_slip and ground_penetration are listed in the task file's soft_constraints
   * (humanoid_nmpc/docs/contact_implicit_mpc/README.md). They price the three conditions of rigid contact rather than
   * imposing them, which is what lets the solver choose the contact sequence.
   */
  struct ContactImplicitConfig {
    // Penalty on f_n * h [N m]. Large enough that carrying load at a height is never worth it, small enough that the
    // linearised product stays a well-conditioned quadratic.
    scalar_t complementarityWeight{100.0};
    // Penalty on f_n * v_xy [N m/s]. This is what holds a loaded foot still, in place of the stance constraint.
    scalar_t slipWeight{10.0};
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
};

}  // namespace ocs2::humanoid
