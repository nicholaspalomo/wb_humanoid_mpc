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
#include "absl/strings/string_view.h"

#include "humanoid_common_mpc_app/robot/TelemetrySink.h"
#include "robot_ipc/Bus.h"

namespace ocs2::humanoid {

/** What a telemetry sink can be built from. */
struct TelemetrySinkContext {
  robot::ipc::Bus* absl_nullable bus = nullptr;
};

// LINT.IfChange(telemetry_sink_names)
/** Publishes robot/state on the bus (BusTelemetrySink). */
inline constexpr absl::string_view kBusTelemetrySinkName = "bus";
// clang-format off
// LINT.ThenChange(//robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.textproto:telemetry_sinks, //robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/config/mpc/task.textproto:telemetry_sinks, //robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto:telemetry_sinks, //robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto:telemetry_sinks, //robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/task.textproto:telemetry_sinks, //humanoid_nmpc/humanoid_common_mpc_app/robot/README.md:telemetry_sinks, //humanoid_nmpc/humanoid_mpc_config/task_file.proto:telemetry_sinks)
// clang-format on

/**
 * The telemetry sinks by name: the task file's `telemetry_sinks` lists the ones a robot process feeds (an empty list
 * turns its telemetry off). The only place a new sink is added; an unknown name is refused with the available ones.
 */
class TelemetrySinkRegistry {
 public:
  using Factory = std::function<absl::StatusOr<std::unique_ptr<TelemetrySink>>(const TelemetrySinkContext& context)>;

  /** A registry with the built-in sinks. */
  TelemetrySinkRegistry();

  void add(const std::string& name, const std::string& description, Factory factory);
  bool has(absl::string_view name) const;
  /** The sink `name` names; InvalidArgument listing the available names for an unknown one. */
  absl::StatusOr<std::unique_ptr<TelemetrySink>> create(absl::string_view name, const TelemetrySinkContext& context) const;
  /** "bus (publishes robot/state on the IPC bus), ...". */
  std::string availableNames() const;
  /** The registered names, in registration order. */
  std::vector<std::string> names() const;

 private:
  struct Entry {
    std::string name;
    std::string description;
    Factory factory;
  };
  std::vector<Entry> entries_;
};

}  // namespace ocs2::humanoid
