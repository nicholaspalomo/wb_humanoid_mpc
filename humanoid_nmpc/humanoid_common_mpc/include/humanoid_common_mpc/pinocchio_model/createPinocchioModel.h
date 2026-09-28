/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

// Pinocchio forward declarations must be included first
#include <pinocchio/fwd.hpp>

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include <ocs2_core/misc/LoadData.h>
#include <ocs2_pinocchio_interface/urdf.h>

#include "absl/status/statusor.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/contact/ContactPolygon.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"

namespace ocs2::humanoid {

///
/// \brief Creates a standard pinocchio model from the urdf
///
/// \param[in] urdfFilePath: The absolute path to the URDF file for the robot.
///

PinocchioInterface createDefaultPinocchioInterface(const std::string& urdfFilePath);

///
/// \brief Creates the MPC's Pinocchio model from the URDF: every joint not in ModelSettings::mpcModelJointNames is set to
/// FIXED, and a frame is added at the center and at every corner of each contact polygon of the task file.
///
/// The model's actuated joints are then checked against mpcModelJointNames, in order (checkPinocchioJointNaming), in
/// every build: the MPC indexes its state, joint limits and weights by that order.
///
/// \param[in] taskFilePath: The task file, whose contact configuration places the contact frames.
/// \param[in] urdfFilePath: The URDF of the robot. It must be the one `modelSettings` was built from.
/// \param[in] modelSettings: The model settings; mpcModelJointNames names the actuated joints.
/// \return InvalidArgument when the URDF does not parse, or when the model's joints are not mpcModelJointNames in
///         order - naming the first joint that differs.
///

absl::StatusOr<PinocchioInterface> loadCustomPinocchioInterface(const std::string& taskFilePath,
                                                                const std::string& urdfFilePath,
                                                                const ModelSettings& modelSettings,
                                                                bool scaleTotalMass = false,
                                                                scalar_t totalMass = 1.0,
                                                                bool verbose = false);

///
/// \brief loadCustomPinocchioInterface() for the callers that cannot return a Status yet: throws std::invalid_argument
/// with its message instead. Prefer loadCustomPinocchioInterface().
///

PinocchioInterface createCustomPinocchioInterface(const std::string& taskFilePath,
                                                  const std::string& urdfFilePath,
                                                  const ModelSettings& modelSettings,
                                                  bool scaleTotalMass = false,
                                                  scalar_t totalMass = 1.0,
                                                  bool verbose = false);

}  // namespace ocs2::humanoid
