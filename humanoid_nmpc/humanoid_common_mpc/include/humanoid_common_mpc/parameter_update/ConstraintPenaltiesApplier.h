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

#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/HotUpdateTarget.h"

namespace ocs2::humanoid {

/**
 * Applies the penalties of the soft constraints: the mu and delta of the contact wrench cone, the friction force cone and
 * the contact moment of every foot, and of the joint limits and the foot collision. Only these are hot: the geometric
 * coefficients beside them (friction_coefficient, min_normal_force, ...) are baked into the constraints when the
 * problem is built.
 *
 * WHICH delta is written into a cone depends on the penalty that is actually installed, not on the file, and the
 * factory's own contactConePenaltyParameters() - the function it built the penalty with - decides it. A schedule-gated
 * cone is wrapped in a RelaxedBarrierPenalty, whose `delta` is the width of its quadratic relaxation and is read straight
 * from the file. An UN-gated cone - the contact-implicit formulation - is wrapped in a SquaredHingePenalty built with
 * delta = 0, because a hinge's delta is the OFFSET of its zero, and the whole point of the hinge there is that its zero
 * sits exactly on the cone that a foot at zero wrench lies on. Writing the barrier's delta into it would move that zero
 * into the interior and reinstate the very force floor that dropping `min_normal_force` and the friction cone's parabolic
 * margin exists to remove.
 *
 * Not thread-safe (HotFieldApplier).
 */
class ConstraintPenaltiesApplier final : public HotFieldApplier {
 public:
  /**
   * The fields it applies: the mu and delta of contacts.contact_wrench_cone_soft_constraint and
   * contacts.friction_force_cone_soft_constraint, contacts.contact_moment_xy_soft_constraint, joint_limits and the mu and
   * delta of collision_constraint.
   */
  static absl::Span<const absl::string_view> staticFields();

  ConstraintPenaltiesApplier() = default;

  absl::string_view name() const override { return "ConstraintPenaltiesApplier"; }
  absl::Span<const absl::string_view> fields() const final { return staticFields(); }
  void apply(HotUpdateTarget& target) override;
};

}  // namespace ocs2::humanoid
