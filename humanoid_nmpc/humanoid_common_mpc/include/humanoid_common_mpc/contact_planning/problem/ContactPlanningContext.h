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

#include <array>
#include <utility>
#include <vector>

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"

namespace ocs2::humanoid {

/** Nominal trajectory the heading frame of the foothold constraints is linearised around (heading model), per node. */
struct HeadingNominal {
  std::vector<scalar_t> heading;
  std::vector<vector2_t> com;
  std::vector<feet_array_t<vector2_t>> feet;
};

/**
 * Per-plan pre-computation, the analogue of ocs2::PreComputation: everything the terms read that depends on the input
 * and not on the node alone. Built once in LipContactPlanner::plan() (and once per heading re-linearisation pass);
 * every term reads from it and none of them recomputes it.
 */
struct ContactPlanningContext {
  const ContactPlannerInput* input = nullptr;
  const Layout* layout = nullptr;
  const HeadingNominal* nominal = nullptr;  // per node; the commanded ramp without the heading model
  const ContactPlan* previousPlan = nullptr;
  const ContactPlanningConfig* config = nullptr;
  int previousPlanShift = -1;  // nodes between the previous plan's start and input.time; -1: no usable previous plan
  scalar_t yawInertia = 1.0;   // [kg m^2] from the input (heading model), 1 otherwise
  scalar_t dt = 0.1;
  int numNodes = 0;
  scalar_t omega = 0.0;
  scalar_t bigM = 0.0;
  // Yaw-aligned constraint frame of every node k = 0..N: components of the world-frame unit vectors of the local x and y
  // axes. With the heading model the frame of node k is the nominal heading of node k, the base yaw otherwise.
  std::vector<std::array<vector2_t, 2>> axes;

  bool hasHeading() const { return layout != nullptr && layout->hasHeading; }
  const std::array<vector2_t, 2>& axesAt(int node) const { return axes[static_cast<size_t>(node)]; }

  /**
   * First-order heading term of the projection e_axis(theta) . d around the nominal (theta_n, d_n): the coefficient g
   * on the heading state and the constant offset -g theta_n, so that g theta + offset = g (theta - theta_n), with
   * d e_x / d theta = e_y and d e_y / d theta = -e_x. Zero without the heading model.
   */
  std::pair<scalar_t, scalar_t> frameTerm(int node, int axis, const vector2_t& dNominal) const;

  /** Fills `axes` from the nominal heading (or the input yaw). */
  void computeAxes();
};

/** The nominal heading trajectory read off a solution: its heading, CoM and foot positions per node (successive linearisation). */
HeadingNominal nominalFromSolution(const Layout& layout, const std::vector<vector_t>& x);

}  // namespace ocs2::humanoid
