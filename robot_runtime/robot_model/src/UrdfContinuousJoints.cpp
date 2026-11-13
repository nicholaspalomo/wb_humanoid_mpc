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

#include "robot_model/UrdfContinuousJoints.h"

#include <exception>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "urdf_model/joint.h"
#include "urdf_model/model.h"
#include "urdf_model/types.h"
#include "urdf_world/types.h"
#include "urdfdom/urdf_parser/urdf_parser.h"

namespace robot::model {

std::vector<std::string> makeContinuousJointsUnboundedRevolute(
    urdf::ModelInterface& model) {
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  std::vector<std::string> converted;
  // joints_ is a std::map, so the names come in name order.
  for (const std::pair<const std::string, urdf::JointSharedPtr>& entry :
       model.joints_) {
    urdf::Joint& joint = *entry.second;
    if (joint.type != urdf::Joint::CONTINUOUS) continue;
    joint.type = urdf::Joint::REVOLUTE;
    if (joint.limits == nullptr) {
      joint.limits = std::make_shared<urdf::JointLimits>();
      joint.limits->effort = kInfinity;
      // NOLINTNEXTLINE(frame-naming): urdfdom JointLimits field, a joint-space bound
      joint.limits->velocity = kInfinity;
    }
    joint.limits->lower = -kInfinity;
    joint.limits->upper = kInfinity;
    converted.push_back(entry.first);
  }
  return converted;
}

absl::StatusOr<urdf::ModelInterfaceSharedPtr> parseUrdfFileForPinocchio(
    absl::string_view urdfPath) {
  urdf::ModelInterfaceSharedPtr model;
  try {  // NOLINT(exceptions): urdfdom is not documented not to throw from its parser; converted to a Status at once
    model = urdf::parseURDFFile(std::string(urdfPath));
    // NOLINTNEXTLINE(exceptions): urdfdom; converted to a Status at once
  } catch (const std::exception& error) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the file '", urdfPath, "' does not parse as a URDF: ", error.what()));
  }
  if (model == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the file '", urdfPath, "' does not contain a valid URDF model."));
  }
  makeContinuousJointsUnboundedRevolute(*model);
  return model;
}

}  // namespace robot::model
