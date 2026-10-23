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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include "humanoid_wb_mpc/parameter_update/WholeBodyHotFieldAppliers.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "humanoid_common_mpc/parameter_update/ConstraintPenaltiesApplier.h"
#include "humanoid_common_mpc/parameter_update/FootConstraintSoftWeightsApplier.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplier.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplierList.h"
#include "humanoid_common_mpc/parameter_update/QuadraticCostWeightsApplier.h"
#include "humanoid_common_mpc/parameter_update/ReferenceManagerApplier.h"
#include "humanoid_common_mpc/parameter_update/SqpSettingsApplier.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/parameter_update/StanceFootAccelerationGainsApplier.h"
#include "humanoid_wb_mpc/parameter_update/WholeBodyCostWeightsApplier.h"

namespace ocs2::humanoid {

namespace {

absl::StatusOr<std::unique_ptr<HotFieldApplier>> makeQuadraticCostWeightsApplier(const WBMpcInterface& /*interface*/) {
  // The whole-body inputs are the contact wrenches and the joint accelerations input_weights is indexed on.
  return std::make_unique<QuadraticCostWeightsApplier>(QuadraticCostWeightsApplier::InputCost::kInputWeights);
}

using Entry = HotFieldApplierEntry<WBMpcInterface>;

// The one list: wholeBodyHotFieldAppliers() makes the appliers from it, in this order, and wholeBodyHotFieldNames()
// reads their fields off it, so that the two cannot disagree.
constexpr std::array<Entry, 7> kWholeBodyAppliers = {{
    {.staticFields = &QuadraticCostWeightsApplier::staticFields, .make = &makeQuadraticCostWeightsApplier},
    {.staticFields = &ConstraintPenaltiesApplier::staticFields, .make = &makeHotFieldApplier<ConstraintPenaltiesApplier, WBMpcInterface>},
    {.staticFields = &FootConstraintSoftWeightsApplier::staticFields,
     .make = &makeHotFieldApplier<FootConstraintSoftWeightsApplier, WBMpcInterface>},
    {.staticFields = &ReferenceManagerApplier::staticFields, .make = &makeHotFieldApplier<ReferenceManagerApplier, WBMpcInterface>},
    {.staticFields = &SqpSettingsApplier::staticFields, .make = &makeHotFieldApplier<SqpSettingsApplier, WBMpcInterface>},
    {.staticFields = &WholeBodyCostWeightsApplier::staticFields, .make = &makeHotFieldApplier<WholeBodyCostWeightsApplier, WBMpcInterface>},
    {.staticFields = &StanceFootAccelerationGainsApplier::staticFields,
     .make = &makeHotFieldApplier<StanceFootAccelerationGainsApplier, WBMpcInterface>},
}};

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<HotFieldApplier>>> wholeBodyHotFieldAppliers(const WBMpcInterface& interface) {
  return makeHotFieldAppliers(absl::MakeConstSpan(kWholeBodyAppliers), interface);
}

std::vector<std::string> wholeBodyHotFieldNames() {
  return hotFieldNames(absl::MakeConstSpan(kWholeBodyAppliers));
}

}  // namespace ocs2::humanoid
