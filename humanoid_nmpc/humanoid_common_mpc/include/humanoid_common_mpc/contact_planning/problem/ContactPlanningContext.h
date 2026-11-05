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
#include <cstddef>
#include <optional>
#include <vector>

#include "absl/base/nullability.h"
#include "ocs2_core/Dimensions.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/contact_planning/ContactPlan.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/problem/Layout.h"

namespace ocs2::humanoid {

/** Nominal trajectory the heading frame of the foothold constraints is
 * linearized around (heading model), per node. */
struct HeadingNominal {
  std::vector<scalar_t> nomHeadingModelYawTraj;
  std::vector<vector2_t> nomPosComRtWorldEwrtWorldXyTraj;
  std::vector<feet_array_t<vector2_t>> nomPosFeetRtWorldEwrtWorldXyTraj;
  // [rad] per node, the nominal ZYX yaw of every foot, which the footprint
  // frames of the ZMP support region are taken at
  // (ContactPlanningContext::rotmatFeetRtWorldTraj). Empty, or not covering
  // every node: every foot is taken at the nominal heading.
  std::vector<feet_array_t<scalar_t>> nomEulerzyxFeetRtWorldYawTraj;
};

/**
 * Per-plan pre-computation, the analog of ocs2::PreComputation: everything the
 * terms read that depends on the input and not on the node alone. Built once in
 * LipContactPlanner::plan() (and once per heading re-linearization pass); every
 * term reads from it and none of them recomputes it.
 */
struct ContactPlanningContext {
  const ContactPlannerInput* absl_nullable input = nullptr;
  const Layout* absl_nullable layout = nullptr;
  const HeadingNominal* absl_nullable nominal =
      nullptr;  // per node; the commanded ramp without the heading model
  const ContactPlan* absl_nullable previousPlan = nullptr;
  const ContactPlanningConfig* absl_nullable config = nullptr;
  std::optional<int>
      previousPlanShift;  // nodes between the previous plan's start and
                          // input.time; empty: no usable previous plan
  scalar_t inertiaComEwrtWorldZz =
      1.0;  // [kg m^2] from the input (heading model), 1 otherwise
  scalar_t dt = PlannerSettings::kDefaultNodeDuration;
  int numNodes = 0;
  scalar_t omega = 0.0;
  scalar_t bigM = 0.0;
  // Yaw-aligned constraint frame of every node k = 0..N, the planar
  // world_R_heading as its two columns: the world-frame unit vectors of the
  // heading frame's x and y axes. With the heading model the frame of node k is
  // the nominal heading of node k. Without it, it is the commanded heading of
  // node k: the input yaw turned by getCmdErrYawHeadingRtWorld(k), the frame
  // getCmdLinvelBaseRtWorldEwrtWorldXy() turns the command into.
  std::vector<std::array<vector2_t, kHorizontalPositionDim>>
      rotmatHeadingRtWorldTraj;
  // The yaw-only planar world_R_foot of every foot at every node k = 0..N, the
  // same columns at the foot's nominal yaw
  // (HeadingNominal::nomEulerzyxFeetRtWorldYawTraj);
  // equal to `rotmatHeadingRtWorldTraj` without the heading model or without
  // nominal foot yaws.
  std::vector<feet_array_t<std::array<vector2_t, kHorizontalPositionDim>>>
      rotmatFeetRtWorldTraj;

  bool hasHeading() const { return layout != nullptr && layout->hasHeading; }
  const std::array<vector2_t, kHorizontalPositionDim>& getRotmatHeadingRtWorld(
      int node) const {
    return rotmatHeadingRtWorldTraj[static_cast<size_t>(node)];
  }
  const std::array<vector2_t, kHorizontalPositionDim>& getRotmatFootRtWorld(
      int node, size_t foot) const {
    return rotmatFeetRtWorldTraj[static_cast<size_t>(node)][foot];
  }

  /**
   * Returns the turn of the commanded heading by node `node`,
   * omega_cmd node dt [rad], from the commanded yaw rate captured at the
   * planning instant. Requires `input`.
   */
  scalar_t getCmdErrYawHeadingRtWorld(int node) const;

  /**
   * Returns the commanded base velocity at node `node`: the command captured at
   * the planning instant (ContactPlannerInput::cmdLinvelBaseRtWorldEwrtWorldXy)
   * rotated by R_z(omega_cmd node dt), the turn of the commanded heading by
   * that node, as the operator's target and heading_tracking turn it. Applies
   * with or without the heading model. Requires `input`.
   */
  vector2_t getCmdLinvelBaseRtWorldEwrtWorldXy(int node) const;

  /**
   * First-order heading term of the projection e_axis(theta) . d around the
   * nominal (theta_n, d_n): the coefficient g on the heading state and the
   * constant offset -g theta_n, so that g theta + offset = g (theta - theta_n),
   * with d e_x / d theta = e_y and d e_y / d theta = -e_x. Zero without the
   * heading model.
   */
  struct HeadingFrameTerm {
    scalar_t gain = 0.0;    // g, the coefficient on the heading state
    scalar_t offset = 0.0;  // -g theta_n
  };
  HeadingFrameTerm frameTerm(int node,
                             int axis,
                             const vector2_t& dNominal) const;

  /** Fills `rotmatHeadingRtWorldTraj` from the nominal heading (or, without
   * the heading model, the commanded heading ramp), and
   * `rotmatFeetRtWorldTraj` from the nominal foot yaws. */
  void computeAxes();
};

/**
 * The nominal heading trajectory read off a solution: its heading, CoM, foot
 * positions and foot yaws per node (successive linearization).
 */
HeadingNominal nominalFromSolution(const Layout& layout,
                                   const std::vector<vector_t>& x);

}  // namespace ocs2::humanoid
