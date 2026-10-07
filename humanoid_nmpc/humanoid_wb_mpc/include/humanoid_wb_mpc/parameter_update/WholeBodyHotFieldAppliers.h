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

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"

#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"

namespace ocs2::humanoid {

/**
 * Returns the hot-field appliers of the whole-body MPC `interface` built, in the order a reload applies them (none keeps
 * the interface, and none is refused): the quadratic cost weights,
 * the soft-constraint penalties, the foot constraint's soft weights, the reference manager (the ground and the swing
 * trajectories), the SQP settings, the weights of the CppAD-taped costs and the foot constraints' gains
 * (WholeBodyCostWeightsApplier, StanceFootAccelerationGainsApplier). They are exactly what a fresh start of the
 * whole-body MPC reads of the task file's RELOAD_HOT fields: the blocks WBMpcInterface refuses or never builds a term for
 * (the CoM + ACoM, capture-point, link and leg torque costs, the contact-implicit terms, the basis-vector inputs and the
 * locomotion heuristics) have no applier here.
 */
absl::StatusOr<std::vector<std::unique_ptr<HotFieldApplier>>> wholeBodyHotFieldAppliers(const WBMpcInterface& interface);

/**
 * Returns the task-file fields wholeBodyHotFieldAppliers() applies, by their path from the file message (a block's path
 * stands for every field below it), sorted and each once: the union of the appliers' staticFields(), read off the same
 * list the appliers are made from, without an MPC. MpcParameterUpdaterModule::appliedFields() of the whole-body
 * updater equals it.
 */
std::vector<std::string> wholeBodyHotFieldNames();

}  // namespace ocs2::humanoid
