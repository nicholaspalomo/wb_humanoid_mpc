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

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "ocs2_mpc/MPC_BASE.h"

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/MpcParameterUpdaterModule.h"

namespace ocs2::humanoid {

/**
 * A consumer of the reference file, called with its command limits whenever it changes; returns why it refused them
 * (MpcParameterUpdaterModule::ReferenceFileReloader).
 */
using ReferenceFileReloader = MpcParameterUpdaterModule::ReferenceFileReloader;

/**
 * The hot-field appliers of the centroidal MPC `interface` built, in the order a reload applies them: the shared ones
 * (QuadraticCostWeightsApplier, ComAndAcomWeightsApplier, TrackingCostWeightsApplier, ConstraintPenaltiesApplier,
 * FootConstraintSoftWeightsApplier, ContactImplicitApplier, ReferenceManagerApplier, SqpSettingsApplier,
 * LocomotionHeuristicsApplier on the interface's layer) and the centroidal ones (BasisInputsCostApplier,
 * CentroidalCostWeightsApplier, ZeroVelocityGainsApplier). The list is the one centroidalHotFieldNames() is built from,
 * so the two cannot disagree. Returns the InvalidArgument of a basis-space transform that does not fit the interface's
 * input (BasisInputsCostApplier::Create()).
 */
absl::StatusOr<std::vector<std::unique_ptr<HotFieldApplier>>> centroidalHotFieldAppliers(const CentroidalMpcInterface& interface);

/**
 * The task-file fields a reload of the centroidal MPC applies, sorted: the union of the static fields of the appliers of
 * centroidalHotFieldAppliers(), known without an interface or an MPC (the reload-class coverage test,
 * humanoid_nmpc/humanoid_mpc_validation/test/testHotFieldCoverage.cpp). Equals the appliedFields() of the updater
 * makeCentroidalMpcParameterUpdater() builds.
 */
std::vector<std::string> centroidalHotFieldNames();

/**
 * The parameter updater of a node that runs a centroidal MPC, wired to everything `interface` built that a hot reload
 * has to reach:
 *  - the appliers of centroidalHotFieldAppliers(), sized to the OCP input, with the basis-space transform of
 *    input_weights when the contact inputs are basis vectors;
 *  - the configuration `interface` was built from, which a reload's start-up fields are compared with;
 *  - the reference manager (swing trajectories, the ground);
 *  - the contact planner module, under contact_schedule_source: "contact_planner" (contact_planning.textproto);
 *  - the locomotion-heuristic layer, whose coefficients are otherwise launch-time only;
 *  - `referenceFileReloaders`, the consumers of the reference file the node built (makeCommandLimitsReloaders(): the
 *    target trajectories calculator and the procedural motion manager), so that the Command Limits tab of the remote
 *    control reaches the running node: each is handed the command limits of the reloaded file.
 * It watches `taskFile`, the contact planner's file beside it and `referenceFile`.
 *
 * Every node and driver that runs the centroidal MPC builds its updater with this one function, so they cannot drift
 * apart in what they register. The node still subscribes the updater to the parameter topic (operator/mpc_parameters,
 * on its bus) and registers it with the solver (addSynchronizedModule, which takes it shared), which need the node's bus
 * and its solver. Returns the InvalidArgument of an updater that cannot be sized to the interface's input.
 */
absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> makeCentroidalMpcParameterUpdater(
    MPC_BASE* absl_nullable mpc,
    const CentroidalMpcInterface& interface,
    const std::string& taskFile,
    const std::string& referenceFile,
    std::vector<ReferenceFileReloader> referenceFileReloaders);

}  // namespace ocs2::humanoid
