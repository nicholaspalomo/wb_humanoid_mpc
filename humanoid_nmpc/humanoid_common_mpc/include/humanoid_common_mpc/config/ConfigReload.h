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
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"

namespace ocs2::humanoid {

/**
 * Who applies a change of a field of a configuration file (the `reload` of its (humanoid_mpc_config.tuning) option,
 * humanoid_nmpc/humanoid_mpc_config/tuning_options.proto): the running MPC or robot (kHot), the next start-up
 * (kStartUp), or neither said (kUnspecified, the fields the tuning GUI excludes).
 */
enum class ConfigReload {
  kUnspecified,
  kHot,
  kStartUp,
};

// The names of the MPC formulations, as the `formulations` of the tuning options name them (an empty list: every
// formulation reads the field).
// clang-format off
// LINT.IfChange(formulation_names)
inline constexpr char kCentroidalFormulation[] = "centroidal";
inline constexpr char kWholeBodyFormulation[] = "whole_body";
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/config_schema.py:formulation_names, //humanoid_nmpc/humanoid_mpc_config/tuning_options.proto:formulations)
// clang-format on

// Who reads a field, as the `consumer` of the tuning options names it (an empty one: the MPC).
// LINT.IfChange(consumer_names)
inline constexpr char kMpcConsumer[] = "mpc";
inline constexpr char kRobotConsumer[] = "robot";
inline constexpr char kGuiConsumer[] = "gui";
// LINT.ThenChange(//humanoid_nmpc/humanoid_mpc_config/tuning_options.proto:consumer)

/** Whether the MPC reads a field whose tuning options name `consumer`: an empty one, or kMpcConsumer. */
bool isReadByTheMpc(absl::string_view consumer);

/** Whether the MPC formulation `formulation` reads a field whose tuning options name `formulations` (empty: every one). */
bool isReadByFormulation(absl::Span<const std::string> formulations, absl::string_view formulation);

/** A field of a configuration file's schema that holds values (no message), with the tuning options in effect for it. */
struct ConfigLeaf {
  // The path from the file message: "state_weights.base_position.x", and "[*]" after a repeated block for every element
  // of it ("task_space_costs[*].weights.pos_x"), as remote_control/config_schema.py writes it.
  std::string path;
  ConfigReload reload = ConfigReload::kUnspecified;
  // Whether the tuning GUI renders it: a number, a bool, an enum, a name list (a repeated string) or a string naming a
  // registry entry, without an exclude_reason.
  bool tunable = false;
  // Who reads it (the `consumer` of its tuning options, inherited like `reload`): "gui", "robot" or "mpc"; empty: the
  // MPC.
  std::string consumer;
  // The MPC formulations that read it, by name (kCentroidalFormulation, kWholeBodyFormulation; the `formulations` of its
  // tuning options, inherited like `reload`); empty: every formulation.
  std::vector<std::string> formulations;
};

/**
 * Every value field of the file message `file`, depth first in declaration order, with the options the schema gives it:
 * a block's options apply to every field below it unless that field sets its own, except the registry, which is each
 * field's own (tuning_options.proto; remote_control/config_schema.py's inherit()). Deprecated fields and maps are
 * skipped, as by the GUI's walk, and a message reachable from itself is walked once on each path.
 */
std::vector<ConfigLeaf> configLeaves(const google::protobuf::Descriptor& file);

/** A field whose values differ between two versions of a file and that a hot reload does not apply. */
struct ConfigChange {
  // Its path from the file message, an element of a repeated block named by its index ("task_space_costs[1].link_name").
  std::string path;
  // The tuning options in effect for it, as ConfigLeaf has them; never kHot.
  ConfigReload reload = ConfigReload::kUnspecified;
  std::string consumer;
  std::vector<std::string> formulations;
};

/**
 * The fields whose values differ between `running` and `reloaded`, two messages of one file message type, and whose
 * reload is not kHot: what a hot reload of `reloaded` leaves for the next start-up, or for no one (kUnspecified), with
 * who reads each. An element of a repeated block is named by its index ("task_space_costs[1].link_name"). A repeated
 * field whose sizes differ is named whole, unless it is a block whose own reload is kHot: then an element only one side
 * has is compared with the block's defaults, so that an added or removed entry is named by the start-up fields it sets
 * ("task_space_costs[2].name"). An (nproto.optional_message) block or a scalar without a default that is present in one
 * message only is named whole too. Any other absent field is its default, as the parser reads it. Allocates; for the
 * threads that apply a reload, not the realtime one.
 */
std::vector<ConfigChange> changedStartUpFields(const google::protobuf::Message& running, const google::protobuf::Message& reloaded);

}  // namespace ocs2::humanoid
