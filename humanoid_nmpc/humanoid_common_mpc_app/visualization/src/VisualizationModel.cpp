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

// Pinocchio forward declarations must be included first.
#include "pinocchio/fwd.hpp"

#include "humanoid_common_mpc_app/visualization/VisualizationModel.h"

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "pinocchio/multibody/model.hpp"

namespace ocs2::humanoid::visualization {

absl::Status checkVisualizationModel(const VisualizationModel& model) {
  if (model.pinocchioInterface == nullptr || model.mpcRobotModel == nullptr) {
    return absl::InvalidArgumentError("the visualization needs the MPC's Pinocchio model and its robot model.");
  }
  const PinocchioInterface::Model& pinocchioModel = model.pinocchioInterface->getModel();
  const int dofs = static_cast<int>(model.mpcRobotModel->getGenCoordinatesDim());
  if (pinocchioModel.nq != dofs || pinocchioModel.nv != dofs) {
    return absl::InvalidArgumentError(absl::StrCat("the Pinocchio model has nq = ", pinocchioModel.nq, " and nv = ", pinocchioModel.nv,
                                                   "; the visualization expects the MPC's model, whose configuration and velocity "
                                                   "are the 6 base coordinates and the ",
                                                   model.mpcRobotModel->getJointDim(), " MPC joints (", dofs, ")."));
  }
  return absl::OkStatus();
}

}  // namespace ocs2::humanoid::visualization
