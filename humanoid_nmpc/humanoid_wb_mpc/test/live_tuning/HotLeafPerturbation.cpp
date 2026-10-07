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

#include "humanoid_nmpc/humanoid_wb_mpc/test/live_tuning/HotLeafPerturbation.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "humanoid_wb_mpc/parameter_update/WholeBodyHotFieldAppliers.h"

namespace ocs2::humanoid::live_tuning_test {
namespace {

using google::protobuf::FieldDescriptor;
using google::protobuf::Message;
using google::protobuf::Reflection;

// Moved on its own (appliedNumericLeaves()).
constexpr char kTerrainHeight[] = "terrain_height";

/** Whether the whole-body updater applies the field at `path` (the path without list indices): at or below one of `names`. */
bool isApplied(absl::string_view path, const std::vector<std::string>& names) {
  for (const std::string& name : names) {
    if (path == name || absl::StartsWith(path, absl::StrCat(name, "."))) return true;
  }
  return false;
}

bool isNumeric(const FieldDescriptor& field) {
  switch (field.cpp_type()) {
    case FieldDescriptor::CPPTYPE_DOUBLE:
    case FieldDescriptor::CPPTYPE_FLOAT:
    case FieldDescriptor::CPPTYPE_INT32:
    case FieldDescriptor::CPPTYPE_INT64:
      return true;
    default:
      return false;
  }
}

/**
 * A walk over the set leaves of a task file's message: collects the paths of the applied numeric ones and, with `move`,
 * moves those below `prefix` (`onlyLeaf`: that one alone). Not thread-safe; one walk per call.
 */
class LeafWalk {
 public:
  LeafWalk(absl::string_view prefix, LeafStep step, std::optional<std::string> onlyLeaf, bool move)
      : names_(wholeBodyHotFieldNames()), prefix_(prefix), step_(step), onlyLeaf_(std::move(onlyLeaf)), move_(move) {}

  void walk(Message& message, const std::string& path, const std::string& plainPath) {
    const Reflection& reflection = *message.GetReflection();
    std::vector<const FieldDescriptor* absl_nonnull> fields;
    reflection.ListFields(message, &fields);
    for (const FieldDescriptor* absl_nonnull field : fields) {
      const std::string fieldPath = path.empty() ? std::string(field->name()) : absl::StrCat(path, ".", field->name());
      const std::string fieldPlainPath = plainPath.empty() ? std::string(field->name()) : absl::StrCat(plainPath, ".", field->name());
      if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
        if (field->is_repeated()) {
          for (int i = 0; i < reflection.FieldSize(message, field); ++i) {
            walk(*reflection.MutableRepeatedMessage(&message, field, i), absl::StrCat(fieldPath, "[", i, "]"), fieldPlainPath);
          }
        } else {
          walk(*reflection.MutableMessage(&message, field), fieldPath, fieldPlainPath);
        }
        continue;
      }
      if (field->is_repeated() || !isNumeric(*field) || fieldPlainPath == kTerrainHeight || !isApplied(fieldPlainPath, names_)) continue;
      leaves_.push_back(fieldPath);
      if (move_ && absl::StartsWith(fieldPath, prefix_) && (!onlyLeaf_.has_value() || *onlyLeaf_ == fieldPath)) moveLeaf(message, *field);
    }
  }

  const std::vector<std::string>& leaves() const { return leaves_; }

 private:
  /** The factor of the next scaled leaf: 1.05 to 1.2 by leaf, so that two leaves are rarely scaled alike. */
  double nextFactor() { return 1.05 + 0.01 * static_cast<double>(numMoved_++ % 16); }

  double moved(double value) {
    if (step_ == LeafStep::kScaled) return value * nextFactor();
    return value == 0.0 ? 0.1 : 1.1 * value;
  }

  void moveLeaf(Message& message, const FieldDescriptor& field) {
    const Reflection& reflection = *message.GetReflection();
    switch (field.cpp_type()) {
      case FieldDescriptor::CPPTYPE_DOUBLE:
        reflection.SetDouble(&message, &field, moved(reflection.GetDouble(message, &field)));
        break;
      case FieldDescriptor::CPPTYPE_FLOAT:
        reflection.SetFloat(&message, &field, static_cast<float>(moved(reflection.GetFloat(message, &field))));
        break;
      case FieldDescriptor::CPPTYPE_INT32:
        reflection.SetInt32(&message, &field, reflection.GetInt32(message, &field) + 1);
        break;
      case FieldDescriptor::CPPTYPE_INT64:
        reflection.SetInt64(&message, &field, reflection.GetInt64(message, &field) + 1);
        break;
      default:
        ADD_FAILURE() << "not a numeric leaf: " << field.full_name();
        break;
    }
  }

  const std::vector<std::string> names_;
  const std::string prefix_;
  const LeafStep step_;
  const std::optional<std::string> onlyLeaf_;
  const bool move_;
  int numMoved_ = 0;
  std::vector<std::string> leaves_;
};

humanoid_mpc_config::TaskFile toProto(const mpc_config::TaskFile& task) {
  humanoid_mpc_config::TaskFile proto;
  mpc_config::ToProto(task, &proto);
  return proto;
}

}  // namespace

std::vector<std::string> appliedNumericLeaves(const mpc_config::TaskFile& task) {
  humanoid_mpc_config::TaskFile proto = toProto(task);
  LeafWalk walk(/*prefix=*/"", LeafStep::kScaled, /*onlyLeaf=*/std::nullopt, /*move=*/false);
  walk.walk(proto, /*path=*/"", /*plainPath=*/"");
  return walk.leaves();
}

mpc_config::TaskFile withMovedLeaves(const mpc_config::TaskFile& task,
                                     absl::string_view prefix,
                                     LeafStep step,
                                     std::optional<std::string> onlyLeaf) {
  humanoid_mpc_config::TaskFile proto = toProto(task);
  LeafWalk walk(prefix, step, std::move(onlyLeaf), /*move=*/true);
  walk.walk(proto, /*path=*/"", /*plainPath=*/"");
  mpc_config::TaskFile moved;
  const absl::Status converted = mpc_config::FromProto(proto, &moved);
  EXPECT_TRUE(converted.ok()) << converted;
  return moved;
}

}  // namespace ocs2::humanoid::live_tuning_test
