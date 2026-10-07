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

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

namespace ocs2::humanoid::visualization {

/**
 * The robot as the visualization sees it: the MPC's model and the files it was built from. The objects are borrowed
 * only while a builder or the publisher is constructed, which copies the Pinocchio model and clones the robot model, so
 * construct them before the solver thread starts using `mpcRobotModel`.
 */
struct VisualizationModel {
  /** The robot's task file: the visualization fields, the contact polygons and the collision spheres. */
  std::string taskFile;
  /** The robot's URDF: the measured robot is drawn with every joint of it. */
  std::string urdfFile;
  /**
   * The MPC's Pinocchio model, as loadCustomPinocchioInterface() builds it: the joints that are not MPC joints fixed,
   * the contact frames and the contact polygon corners added, and a Translation + SphericalZYX root.
   */
  const PinocchioInterface* absl_nullable pinocchioInterface = nullptr;
  /**
   * The MPC's robot model, the one its dynamics use (CentroidalMpcRobotModel, its BasisInputsModelDecorator, or
   * WBAccelMpcRobotModel); its ModelSettings must outlive the visualization.
   */
  const MpcRobotModelBase<scalar_t>* absl_nullable mpcRobotModel = nullptr;
};

/**
 * InvalidArgument when a pointer is null, or when the Pinocchio model is not the MPC's: its configuration and velocity
 * must both be the base (3 translations, 3 Euler ZYX angles) followed by the MPC joints.
 */
absl::Status checkVisualizationModel(const VisualizationModel& model);

}  // namespace ocs2::humanoid::visualization
