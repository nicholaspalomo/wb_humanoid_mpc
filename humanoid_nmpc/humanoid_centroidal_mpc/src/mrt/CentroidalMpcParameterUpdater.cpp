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

#include "humanoid_centroidal_mpc/mrt/CentroidalMpcParameterUpdater.h"

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

#include "humanoid_centroidal_mpc/parameter_update/BasisInputsCostApplier.h"
#include "humanoid_centroidal_mpc/parameter_update/CentroidalCostWeightsApplier.h"
#include "humanoid_centroidal_mpc/parameter_update/ZeroVelocityGainsApplier.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/parameter_update/ComAndAcomWeightsApplier.h"
#include "humanoid_common_mpc/parameter_update/ConstraintPenaltiesApplier.h"
#include "humanoid_common_mpc/parameter_update/ContactImplicitApplier.h"
#include "humanoid_common_mpc/parameter_update/FootConstraintSoftWeightsApplier.h"
#include "humanoid_common_mpc/parameter_update/HotFieldApplierList.h"
#include "humanoid_common_mpc/parameter_update/LocomotionHeuristicsApplier.h"
#include "humanoid_common_mpc/parameter_update/QuadraticCostWeightsApplier.h"
#include "humanoid_common_mpc/parameter_update/ReferenceManagerApplier.h"
#include "humanoid_common_mpc/parameter_update/SqpSettingsApplier.h"
#include "humanoid_common_mpc/parameter_update/TrackingCostWeightsApplier.h"

namespace ocs2::humanoid {

namespace {

absl::StatusOr<std::unique_ptr<HotFieldApplier>> makeQuadraticCostWeightsApplier(const CentroidalMpcInterface& interface) {
  // With basis-vector contact inputs R is BasisInputsCostApplier's, transformed exactly as the OCP factory did.
  return std::make_unique<QuadraticCostWeightsApplier>(interface.getBasisInputsCostTransformConfig().has_value()
                                                           ? QuadraticCostWeightsApplier::InputCost::kBasisVectorInputs
                                                           : QuadraticCostWeightsApplier::InputCost::kInputWeights);
}

absl::StatusOr<std::unique_ptr<HotFieldApplier>> makeBasisInputsCostApplier(const CentroidalMpcInterface& interface) {
  // The input layout the solver optimizes over is the effective model's: in basis-vector mode [lambda, joint velocities].
  ASSIGN_OR_RETURN(
      std::unique_ptr<BasisInputsCostApplier> applier,
      BasisInputsCostApplier::Create(interface.getBasisInputsCostTransformConfig(), interface.getEffectiveMpcRobotModel().getInputDim()));
  return std::unique_ptr<HotFieldApplier>(std::move(applier));
}

absl::StatusOr<std::unique_ptr<HotFieldApplier>> makeContactImplicitApplier(const CentroidalMpcInterface& interface) {
  return std::make_unique<ContactImplicitApplier>(interface.modelSettings().contactNames);
}

absl::StatusOr<std::unique_ptr<HotFieldApplier>> makeLocomotionHeuristicsApplier(const CentroidalMpcInterface& interface) {
  // Without it the locomotion_heuristics coefficients are launch-time only and the tuning GUI's sliders for them write
  // the file without reaching the running controller.
  return std::make_unique<LocomotionHeuristicsApplier>(interface.getLocomotionHeuristicLayerPtr());
}

using Entry = HotFieldApplierEntry<CentroidalMpcInterface>;

// The one list of the centroidal MPC's appliers: centroidalHotFieldAppliers() makes them from it, in this order, and
// centroidalHotFieldNames() reads their fields off it.
constexpr std::array<Entry, 12> kCentroidalAppliers = {{
    {.staticFields = &QuadraticCostWeightsApplier::staticFields, .make = &makeQuadraticCostWeightsApplier},
    {.staticFields = &BasisInputsCostApplier::staticFields, .make = &makeBasisInputsCostApplier},
    {.staticFields = &ComAndAcomWeightsApplier::staticFields,
     .make = &makeHotFieldApplier<ComAndAcomWeightsApplier, CentroidalMpcInterface>},
    {.staticFields = &TrackingCostWeightsApplier::staticFields,
     .make = &makeHotFieldApplier<TrackingCostWeightsApplier, CentroidalMpcInterface>},
    {.staticFields = &ConstraintPenaltiesApplier::staticFields,
     .make = &makeHotFieldApplier<ConstraintPenaltiesApplier, CentroidalMpcInterface>},
    {.staticFields = &FootConstraintSoftWeightsApplier::staticFields,
     .make = &makeHotFieldApplier<FootConstraintSoftWeightsApplier, CentroidalMpcInterface>},
    {.staticFields = &ContactImplicitApplier::staticFields, .make = &makeContactImplicitApplier},
    {.staticFields = &ReferenceManagerApplier::staticFields, .make = &makeHotFieldApplier<ReferenceManagerApplier, CentroidalMpcInterface>},
    {.staticFields = &SqpSettingsApplier::staticFields, .make = &makeHotFieldApplier<SqpSettingsApplier, CentroidalMpcInterface>},
    {.staticFields = &LocomotionHeuristicsApplier::staticFields, .make = &makeLocomotionHeuristicsApplier},
    {.staticFields = &CentroidalCostWeightsApplier::staticFields,
     .make = &makeHotFieldApplier<CentroidalCostWeightsApplier, CentroidalMpcInterface>},
    {.staticFields = &ZeroVelocityGainsApplier::staticFields,
     .make = &makeHotFieldApplier<ZeroVelocityGainsApplier, CentroidalMpcInterface>},
}};

}  // namespace

absl::StatusOr<std::vector<std::unique_ptr<HotFieldApplier>>> centroidalHotFieldAppliers(const CentroidalMpcInterface& interface) {
  return makeHotFieldAppliers(absl::MakeConstSpan(kCentroidalAppliers), interface);
}

std::vector<std::string> centroidalHotFieldNames() {
  return hotFieldNames(absl::MakeConstSpan(kCentroidalAppliers));
}

absl::StatusOr<std::shared_ptr<MpcParameterUpdaterModule>> makeCentroidalMpcParameterUpdater(
    MPC_BASE* absl_nullable mpc,
    const CentroidalMpcInterface& interface,
    const std::string& taskFile,
    const std::string& referenceFile,
    std::vector<ReferenceFileReloader> referenceFileReloaders) {
  // A reload is compared with the configuration the interface was built from.
  MpcParameterUpdaterModule::Options options;
  options.taskFile = taskFile;
  options.referenceFile = referenceFile;
  options.contactPlanningFile = contactPlanningFileBeside(taskFile);
  options.runningTask = interface.config().task;
  options.layout = stateInputLayout(interface.modelSettings(), StateInputLayout::Mpc::kCentroidal);
  options.inputDim = interface.getEffectiveMpcRobotModel().getInputDim();
  options.referenceManager = interface.getSwitchedModelReferenceManagerPtr().get();
  ASSIGN_OR_RETURN(options.appliers, centroidalHotFieldAppliers(interface));
  ASSIGN_OR_RETURN(std::unique_ptr<MpcParameterUpdaterModule> created, MpcParameterUpdaterModule::Create(mpc, std::move(options)));
  std::shared_ptr<MpcParameterUpdaterModule> updater = std::move(created);
  // Null when contact planning is off, which the updater takes as "no contact planner to reload".
  updater->setContactPlannerModule(interface.getContactPlannerModulePtr());
  // The command limits and ramps come from the reference file, which its consumers read once at construction. Each
  // consumer must outlive the updater, and is responsible for the thread safety of what it writes.
  for (ReferenceFileReloader& reloader : referenceFileReloaders) {
    updater->addReferenceFileReloader(std::move(reloader));
  }
  return updater;
}

}  // namespace ocs2::humanoid
