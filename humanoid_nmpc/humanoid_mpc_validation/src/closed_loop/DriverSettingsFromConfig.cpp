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

#include "humanoid_mpc_validation/closed_loop/DriverSettingsFromConfig.h"

#include <array>

#include "Eigen/Core"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/reference/ReferenceFromConfig.h"
#include "humanoid_common_mpc/config/reference/ReferenceSettings.h"
#include "humanoid_mpc_config/reference_file.nproto.h"

namespace ocs2::humanoid::validation {
namespace {

/** The fields of GuiCommandScaling::commandLimits, in its order. */
constexpr std::array<absl::string_view, 3> kCommandLimitFields = {"max_displacement_velocity_x", "max_displacement_velocity_y",
                                                                  "max_rotation_velocity"};

}  // namespace

absl::StatusOr<GuiCommandScaling> guiCommandScalingFromConfig(const mpc_config::ReferenceFile& file) {
  ASSIGN_OR_RETURN(const ReferenceSettings reference, referenceSettingsFromConfig(file));
  GuiCommandScaling scaling;
  scaling.commandLimits =
      Eigen::Vector3d(reference.maxDisplacementVelocityX, reference.maxDisplacementVelocityY, reference.maxRotationVelocity);
  scaling.defaultPelvisHeight = reference.defaultBaseHeight;
  // The GUI sends a command as its fraction of these: a limit that is not positive leaves no command to send. Every
  // value is finite (referenceSettingsFromConfig()).
  for (Eigen::Index i = 0; i < scaling.commandLimits.size(); ++i) {
    if (scaling.commandLimits(i) <= 0.0) {
      return absl::InvalidArgumentError(absl::StrCat(kCommandLimitFields[static_cast<size_t>(i)], " is ", scaling.commandLimits(i),
                                                     ", but the GUI's command limits must be positive."));
    }
  }
  return scaling;
}

}  // namespace ocs2::humanoid::validation
