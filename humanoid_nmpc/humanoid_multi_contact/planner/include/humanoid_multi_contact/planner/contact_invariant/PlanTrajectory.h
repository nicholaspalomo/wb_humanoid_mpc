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

#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_multi_contact/contact_set/ContactSet.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_multi_contact/scenarios/Scenario.h"
#include "humanoid_multi_contact/stability/RobotStanceModel.h"

namespace ocs2::humanoid::multi_contact {

/** A motion of the centroidal MPC's model and its stances, knot by knot.
 * Passive data. */
struct MpcTrajectory {
  // [s] Increasing.
  std::vector<double> times;
  // The MPC's states and inputs, one per time.
  vector_array_t states;
  vector_array_t inputs;
  // The stance of every knot, over the candidates of the contact set.
  std::vector<ContactMask> masks;
};

/** What a plan is converted with. Passive data; the pointers are not
 * retained. */
struct PlanConversion {
  const LoadedScenario* absl_nonnull scenario;
  // The MPC's robot model (its state and input layout) and its contacts by
  // name, in the MPC's order (ModelSettings::contactNames).
  const MpcRobotModelBase<scalar_t>* absl_nonnull model;
  std::vector<std::string> contactNames;
  // [kg]
  double mass = 0.0;
  // The kinematics of the pads, at every knot.
  RobotStanceModel* absl_nonnull stanceModel;
  std::string planner;
};

/**
 * The plan of `trajectory` (multi_contact_plan.proto): the knots' base pose,
 * joints (the MPC's, by name), joint velocities, momenta (the MPC's
 * normalized momenta times the mass) and the wrenches of the active
 * contacts; one contact phase per run of knots a candidate is active in
 * (until the knot that breaks it, where the mode changes), on
 * the patch under the run's mean pad position (TerrainModel::patchFrameAt()),
 * its target that position projected onto the patch; the scenario's hashes.
 * Not certified. InvalidArgument for inconsistent sizes, a mask of a contact
 * the MPC does not have, or a configuration the stance model refuses.
 */
absl::StatusOr<config::MultiContactPlan> planOfTrajectory(
    const MpcTrajectory& trajectory, const PlanConversion& conversion);

/** The contact-set index of each MPC contact of `contactNames`; -1 for a
 * contact that is not a candidate. */
std::vector<int> candidateIndicesOf(
    const std::vector<std::string>& contactNames, const ContactSet& contactSet);

}  // namespace ocs2::humanoid::multi_contact
