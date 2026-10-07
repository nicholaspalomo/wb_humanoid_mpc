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

#include "humanoid_common_mpc/config/ConfigReload.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"

#include "humanoid_mpc_config/config_options.pb.h"
#include "humanoid_mpc_config/tuning_options.pb.h"
#include "nproto/options.pb.h"

namespace ocs2::humanoid {

namespace {

using google::protobuf::Descriptor;
using google::protobuf::FieldDescriptor;
using google::protobuf::Message;
using google::protobuf::Reflection;

/**
 * The options in effect for a field: the reload, the exclusion, the consumer and the formulations its block passes on,
 * and the field's own registry.
 */
struct FieldTuning {
  ConfigReload reload = ConfigReload::kUnspecified;
  bool excluded = false;
  std::string consumer;
  // Empty: every formulation reads it.
  std::vector<std::string> formulations;
  // Each field's own, never inherited.
  std::string registry;
};

ConfigReload reloadOf(humanoid_mpc_config::TuningOptions::Reload reload) {
  // Proto enums are open: a number this code does not know is no reload class.
  if (reload == humanoid_mpc_config::TuningOptions::RELOAD_HOT) return ConfigReload::kHot;
  if (reload == humanoid_mpc_config::TuningOptions::RELOAD_START_UP) return ConfigReload::kStartUp;
  return ConfigReload::kUnspecified;
}

// LINT.IfChange(tuning_inheritance)
/** The tuning of `field` below a block whose tuning is `parent` (tuning_options.proto, config_schema.inherit()). */
FieldTuning inherit(const FieldTuning& parent, const FieldDescriptor& field) {
  FieldTuning tuning = parent;
  tuning.registry.clear();
  if (!field.options().HasExtension(humanoid_mpc_config::tuning)) {
    return tuning;
  }
  const humanoid_mpc_config::TuningOptions& own = field.options().GetExtension(humanoid_mpc_config::tuning);
  if (own.has_reload()) tuning.reload = reloadOf(own.reload());
  if (own.has_exclude_reason()) tuning.excluded = !own.exclude_reason().empty();
  if (own.has_consumer()) tuning.consumer = own.consumer();
  // A field that names formulations replaces its block's; one that names none keeps them (config_schema.inherit()).
  if (own.formulations_size() > 0) tuning.formulations.assign(own.formulations().begin(), own.formulations().end());
  tuning.registry = own.registry();
  return tuning;
}
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/config_schema.py:tuning_inheritance)

/** Whether the walks visit `field`: a file may not set a deprecated field, and the GUI edits no map. */
bool isVisited(const FieldDescriptor& field) {
  return !field.options().deprecated() && !field.is_map();
}

// LINT.IfChange(tunable_fields)
/** Whether the GUI renders the value field `field` with `tuning` (config_schema.kind_of() and renders). */
bool isTunable(const FieldDescriptor& field, const FieldTuning& tuning) {
  if (tuning.excluded) return false;
  if (field.type() == FieldDescriptor::TYPE_BYTES) return false;
  if (field.type() == FieldDescriptor::TYPE_STRING) {
    // A name list, or a name of a registry entry; any other string is text, which is not edited.
    return field.is_repeated() || !tuning.registry.empty();
  }
  return field.cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE;
}
// clang-format off
// LINT.ThenChange(//humanoid_nmpc/remote_control/remote_control/config_schema.py:tunable_fields, //humanoid_nmpc/remote_control/remote_control/config_schema.py:rendered_fields)
// clang-format on

std::string childPath(absl::string_view prefix, const FieldDescriptor& field) {
  return prefix.empty() ? std::string(field.name()) : absl::StrCat(prefix, ".", field.name());
}

void walkLeaves(const Descriptor& descriptor,
                const std::string& prefix,
                const FieldTuning& tuning,
                std::vector<std::string>& stack,
                std::vector<ConfigLeaf>& found) {
  for (int i = 0; i < descriptor.field_count(); ++i) {
    const FieldDescriptor& field = *descriptor.field(i);
    if (!isVisited(field)) continue;
    const FieldTuning fieldTuning = inherit(tuning, field);
    std::string path = childPath(prefix, field);
    const Descriptor* absl_nullable block = field.message_type();
    if (block == nullptr) {
      found.push_back(ConfigLeaf{.path = std::move(path),
                                 .reload = fieldTuning.reload,
                                 .tunable = isTunable(field, fieldTuning),
                                 .consumer = fieldTuning.consumer,
                                 .formulations = fieldTuning.formulations});
      continue;
    }
    if (std::find(stack.begin(), stack.end(), block->full_name()) != stack.end()) continue;
    if (field.is_repeated()) absl::StrAppend(&path, "[*]");
    stack.emplace_back(descriptor.full_name());
    walkLeaves(*block, path, fieldTuning, stack, found);
    stack.pop_back();
  }
}

/**
 * Whether a field's presence is a value of its own: a block that is an (nproto.optional_message), whose absence means
 * something else than its defaults, and a scalar with presence and no default (std::optional in its struct). Any other
 * field that is absent is its default.
 */
bool presenceMatters(const FieldDescriptor& field) {
  if (field.is_repeated()) return false;
  if (field.cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) return field.options().GetExtension(nproto::optional_message);
  return field.has_presence() && !field.has_default_value();
}

/** Whether the value field `field` holds the same values in `running` and `reloaded`, presence included where it matters. */
bool sameValues(const Message& running, const Message& reloaded, const FieldDescriptor& field) {
  const Reflection& left = *running.GetReflection();
  const Reflection& right = *reloaded.GetReflection();
  if (presenceMatters(field) && left.HasField(running, &field) != right.HasField(reloaded, &field)) return false;
  const int size = field.is_repeated() ? left.FieldSize(running, &field) : 1;
  if (field.is_repeated() && size != right.FieldSize(reloaded, &field)) return false;
  for (int index = 0; index < size; ++index) {
    const bool repeated = field.is_repeated();
    bool same = true;
    if (field.cpp_type() == FieldDescriptor::CPPTYPE_INT32) {
      same = repeated ? left.GetRepeatedInt32(running, &field, index) == right.GetRepeatedInt32(reloaded, &field, index)
                      : left.GetInt32(running, &field) == right.GetInt32(reloaded, &field);
    } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_INT64) {
      same = repeated ? left.GetRepeatedInt64(running, &field, index) == right.GetRepeatedInt64(reloaded, &field, index)
                      : left.GetInt64(running, &field) == right.GetInt64(reloaded, &field);
    } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_UINT32) {
      same = repeated ? left.GetRepeatedUInt32(running, &field, index) == right.GetRepeatedUInt32(reloaded, &field, index)
                      : left.GetUInt32(running, &field) == right.GetUInt32(reloaded, &field);
    } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_UINT64) {
      same = repeated ? left.GetRepeatedUInt64(running, &field, index) == right.GetRepeatedUInt64(reloaded, &field, index)
                      : left.GetUInt64(running, &field) == right.GetUInt64(reloaded, &field);
    } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_DOUBLE) {
      same = repeated ? left.GetRepeatedDouble(running, &field, index) == right.GetRepeatedDouble(reloaded, &field, index)
                      : left.GetDouble(running, &field) == right.GetDouble(reloaded, &field);
    } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_FLOAT) {
      same = repeated ? left.GetRepeatedFloat(running, &field, index) == right.GetRepeatedFloat(reloaded, &field, index)
                      : left.GetFloat(running, &field) == right.GetFloat(reloaded, &field);
    } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_BOOL) {
      same = repeated ? left.GetRepeatedBool(running, &field, index) == right.GetRepeatedBool(reloaded, &field, index)
                      : left.GetBool(running, &field) == right.GetBool(reloaded, &field);
    } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_ENUM) {
      same = repeated ? left.GetRepeatedEnumValue(running, &field, index) == right.GetRepeatedEnumValue(reloaded, &field, index)
                      : left.GetEnumValue(running, &field) == right.GetEnumValue(reloaded, &field);
    } else if (field.cpp_type() == FieldDescriptor::CPPTYPE_STRING) {
      same = repeated ? left.GetRepeatedString(running, &field, index) == right.GetRepeatedString(reloaded, &field, index)
                      : left.GetString(running, &field) == right.GetString(reloaded, &field);
    }
    if (!same) return false;
  }
  return true;
}

/** Appends the field at `path` with `tuning` to `changed`, unless a hot reload applies it. */
void recordChange(const std::string& path, const FieldTuning& tuning, std::vector<ConfigChange>& changed) {
  if (tuning.reload == ConfigReload::kHot) return;
  changed.push_back(ConfigChange{.path = path, .reload = tuning.reload, .consumer = tuning.consumer, .formulations = tuning.formulations});
}

void walkChanges(const Message& running,
                 const Message& reloaded,
                 const std::string& prefix,
                 const FieldTuning& tuning,
                 std::vector<ConfigChange>& changed);

/**
 * Compares the elements of the repeated block `field` of `running` and `reloaded` (with `tuning`) one by one, an element
 * only one side has with the block's defaults.
 */
void walkElementChanges(const Message& running,
                        const Message& reloaded,
                        const FieldDescriptor& field,
                        const std::string& path,
                        const FieldTuning& tuning,
                        std::vector<ConfigChange>& changed) {
  const Reflection& left = *running.GetReflection();
  const Reflection& right = *reloaded.GetReflection();
  const int runningSize = left.FieldSize(running, &field);
  const int reloadedSize = right.FieldSize(reloaded, &field);
  const Message& defaults = *left.GetMessageFactory()->GetPrototype(field.message_type());
  for (int index = 0; index < std::max(runningSize, reloadedSize); ++index) {
    walkChanges(index < runningSize ? left.GetRepeatedMessage(running, &field, index) : defaults,
                index < reloadedSize ? right.GetRepeatedMessage(reloaded, &field, index) : defaults, absl::StrCat(path, "[", index, "]"),
                tuning, changed);
  }
}

void walkChanges(const Message& running,
                 const Message& reloaded,
                 const std::string& prefix,
                 const FieldTuning& tuning,
                 std::vector<ConfigChange>& changed) {
  const Descriptor& descriptor = *running.GetDescriptor();
  const Reflection& left = *running.GetReflection();
  const Reflection& right = *reloaded.GetReflection();
  for (int i = 0; i < descriptor.field_count(); ++i) {
    const FieldDescriptor& field = *descriptor.field(i);
    if (!isVisited(field)) continue;
    const FieldTuning fieldTuning = inherit(tuning, field);
    const std::string path = childPath(prefix, field);
    if (field.cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
      if (!sameValues(running, reloaded, field)) recordChange(path, fieldTuning, changed);
      continue;
    }
    if (field.is_repeated()) {
      // A block a reload applies keeps its elements' start-up fields: an added or removed element is named by them.
      if (left.FieldSize(running, &field) != right.FieldSize(reloaded, &field) && fieldTuning.reload != ConfigReload::kHot) {
        recordChange(path, fieldTuning, changed);
        continue;
      }
      walkElementChanges(running, reloaded, field, path, fieldTuning, changed);
      continue;
    }
    if (presenceMatters(field) && left.HasField(running, &field) != right.HasField(reloaded, &field)) {
      recordChange(path, fieldTuning, changed);
      continue;
    }
    // An absent block is its defaults: GetMessage() is the default instance then.
    walkChanges(left.GetMessage(running, &field), right.GetMessage(reloaded, &field), path, fieldTuning, changed);
  }
}

}  // namespace

std::vector<ConfigLeaf> configLeaves(const google::protobuf::Descriptor& file) {
  std::vector<ConfigLeaf> found;
  std::vector<std::string> stack;
  walkLeaves(file, /*prefix=*/"", FieldTuning{}, stack, found);
  return found;
}

bool isReadByTheMpc(absl::string_view consumer) {
  return consumer.empty() || consumer == kMpcConsumer;
}

bool isReadByFormulation(absl::Span<const std::string> formulations, absl::string_view formulation) {
  return formulations.empty() || std::find(formulations.begin(), formulations.end(), formulation) != formulations.end();
}

std::vector<ConfigChange> changedStartUpFields(const google::protobuf::Message& running, const google::protobuf::Message& reloaded) {
  std::vector<ConfigChange> changed;
  if (running.GetDescriptor() != reloaded.GetDescriptor()) {
    // Not two versions of one file: every field differs, which the file message's name says best.
    changed.push_back(ConfigChange{.path = std::string(running.GetDescriptor()->full_name()), .reload = ConfigReload::kStartUp});
    return changed;
  }
  walkChanges(running, reloaded, /*prefix=*/"", FieldTuning{}, changed);
  return changed;
}

}  // namespace ocs2::humanoid
