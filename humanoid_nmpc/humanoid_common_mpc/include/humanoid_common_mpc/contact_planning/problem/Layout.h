/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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

#include <string>
#include <vector>

#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/** Variable names of the reduced model. Feet are labelled L and R for a biped, by index otherwise. */
namespace var {
std::string footLabel(size_t foot);
// LIP block (lip_com): states c_x c_y v_x v_y, input zmp_x zmp_y.
inline constexpr const char* kComX = "c_x";
inline constexpr const char* kComY = "c_y";
inline constexpr const char* kVelX = "v_x";
inline constexpr const char* kVelY = "v_y";
inline constexpr const char* kZmpX = "zmp_x";
inline constexpr const char* kZmpY = "zmp_y";
// Foothold block (foothold_integrator): states p_<foot>x p_<foot>y, inputs dp_<foot>x dp_<foot>y c_<foot>.
std::string footX(size_t foot);
std::string footY(size_t foot);
std::string footDeltaX(size_t foot);
std::string footDeltaY(size_t foot);
std::string contact(size_t foot);
// Heading block (heading_double_integrator): states theta omega psi_<foot>, inputs tau_<foot> dpsi_<foot>.
inline constexpr const char* kHeading = "theta";
inline constexpr const char* kHeadingRate = "omega";
std::string footYaw(size_t foot);
std::string yawTorque(size_t foot);
std::string footYawDelta(size_t foot);
}  // namespace var

/**
 * Variable layout of the planner's OCP: the states and inputs of every node, in the order the model blocks declared
 * them. The LIP block and the foothold block always come first, so that indices 0..7 are the StateIndex / InputIndex
 * enums of LipContactPlanner and the mixed-integer solver finds the contact binaries where it always did; the heading
 * block, when listed, appends its variables. The well-known indices below are resolved by name after the blocks have
 * declared their variables (-1 when the block is absent).
 */
struct Layout {
  int nx = 0;
  int nu = 0;
  std::vector<std::string> stateNames;
  std::vector<std::string> inputNames;

  // Heading block, -1 without it.
  bool hasHeading = false;
  int heading = -1;        // state: whole-body heading [rad]
  int headingRate = -1;    // state: heading rate [rad/s]
  int footYaw0 = -1;       // states: foot yaws, one per foot
  int yawTorque0 = -1;     // inputs: yaw torque per foot [N m]
  int footYawDelta0 = -1;  // inputs: foot yaw displacement per foot [rad]
  // The three accessors honour the sentinel of the members they read: without the heading block they return -1 for
  // EVERY foot. Plain offset arithmetic did not, and that was a trap rather than a live bug. With footYaw0 at -1,
  // footYaw(0) came out as -1 as documented while footYaw(1) came out as 0 - an in-bounds, valid index that aliases
  // c_x, because LipComDynamics forces the LIP block to declare its variables first; yawTorque(1) and footYawDelta(1)
  // likewise aliased zmp_x. Code written against the documented contract, `if (layout.footYaw(foot) < 0) continue;`,
  // therefore read or wrote the centre-of-mass state of the second foot on any formulation without
  // heading_double_integrator - which is the shipped EngineAI SA01 configuration - with no error and no crash. Nothing
  // was wrong in production only because LipIndices::bind, the single caller, worked around the hazard with its own
  // `hasHeading ? ... : -1`; the guard belongs here, where the invariant is stated.
  int footYaw(size_t foot) const { return hasHeading ? footYaw0 + static_cast<int>(foot) : -1; }
  int yawTorque(size_t foot) const { return hasHeading ? yawTorque0 + static_cast<int>(foot) : -1; }
  int footYawDelta(size_t foot) const { return hasHeading ? footYawDelta0 + static_cast<int>(foot) : -1; }

  /** Index of a state / input by name; throws std::out_of_range for an unknown name. */
  int state(const std::string& name) const;
  int input(const std::string& name) const;
  bool hasState(const std::string& name) const;
  bool hasInput(const std::string& name) const;

  /** One line: x = [...] (nx), u = [...] (nu). */
  std::string describe() const;
};

}  // namespace ocs2::humanoid
