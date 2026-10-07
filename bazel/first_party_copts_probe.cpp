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

#include "bazel/first_party_copts_probe.h"

// Pinocchio's forward declarations come first: they set the Boost MPL limits the rest of Pinocchio needs.
#include "pinocchio/fwd.hpp"

#include <cstddef>
#include <string>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "cppad/cg.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "hpipm_d_ocp_qp_ipm.h"  // NOLINT(build/include_subdir): HPIPM installs its headers in no directory
#include "mujoco/mujoco.h"
#include "ocs2_core/Types.h"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "urdf_parser/urdf_parser.h"
#include "zmq.hpp"  // NOLINT(build/include_subdir): cppzmq installs its header in no directory

#include "humanoid_mpc_msgs/robot_state_sample.nproto.h"
#include "humanoid_mpc_msgs/robot_state_sample.pb.h"

namespace wb_humanoid_mpc::copts_probe {

int instantiateThirdPartyTemplates() {
  absl::flat_hash_map<std::string, int> counts;
  counts[absl::StrCat("probe", counts.size())] = 1;
  const absl::StatusOr<int> parsed = absl::InvalidArgumentError("probe");
  const absl::Status& status = parsed.status();

  const Eigen::Quaterniond rotation(Eigen::AngleAxisd(/*angle=*/0.5, Eigen::Vector3d::UnitZ()));
  const Eigen::Vector3d rotated = rotation * Eigen::Vector3d::UnitX();
  const ocs2::vector_t state = ocs2::vector_t::Zero(/*size=*/3);

  pinocchio::Model model;
  pinocchio::Data data(model);
  pinocchio::updateFramePlacements(model, data);

  const humanoid_mpc_msgs::RobotStateSample message;
  const std::string serialized = message.SerializeAsString();
  ocs2::humanoid::msgs::RobotStateSample sample;

  const zmq::message_t frame(serialized.data(), serialized.size());
  const CppAD::cg::CG<double> constant(1.0);
  const CppAD::AD<CppAD::cg::CG<double>> taped(constant);
  const ::testing::Matcher<int> matcher = ::testing::Eq(1);

  const size_t total = counts.size() + serialized.size() + frame.size() + static_cast<size_t>(sample.joint_positions.size());
  return static_cast<int>(total) + static_cast<int>(rotated.x() + state.sum()) + (status.ok() ? 1 : 0) + (matcher.Matches(1) ? 1 : 0) +
         (CppAD::Variable(taped) ? 1 : 0) + model.nq;
}

}  // namespace wb_humanoid_mpc::copts_probe
