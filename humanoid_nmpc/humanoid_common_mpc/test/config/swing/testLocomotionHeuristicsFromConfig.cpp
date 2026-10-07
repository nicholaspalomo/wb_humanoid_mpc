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

// The locomotion heuristics from the typed task file (LocomotionHeuristicsFromConfig.h): the schema's defaults are the
// layer's no-op, its parameter blocks are the registry's heuristics, every parameter reaches a member of its own, the
// lists are copied as written and the layer's own checks still apply, and a value that is not finite is refused.

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/swing/LocomotionHeuristicsFromConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicConfig.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFormulation.h"
#include "humanoid_mpc_config/locomotion_heuristics_config.nproto.h"
#include "humanoid_mpc_config/locomotion_heuristics_config.nproto.pb.h"
#include "humanoid_mpc_config/locomotion_heuristics_config.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

using google::protobuf::Descriptor;
using google::protobuf::FieldDescriptor;
using HeuristicsProto = humanoid_mpc_config::LocomotionHeuristicsConfig;

/**
 * The bits of every member of a heuristic's parameter struct, in declaration order. The structs hold nothing but
 * scalar_t members, so their bytes are those members one after the other; the test of every parameter checks that their
 * number is the schema block's.
 */
template <typename Parameters>
std::vector<uint64_t> memberBits(const Parameters& parameters) {
  static_assert(std::is_trivially_copyable_v<Parameters> && sizeof(Parameters) % sizeof(double) == 0);
  std::array<double, sizeof(Parameters) / sizeof(double)> members{};
  std::memcpy(members.data(), &parameters, sizeof(Parameters));
  std::vector<uint64_t> bits;
  for (const double member : members) bits.push_back(std::bit_cast<uint64_t>(member));
  return bits;
}

/** The bits of the parameter struct of the heuristic `heuristic` in `config`; empty for a name that has none. */
std::vector<uint64_t> parameterBits(const LocomotionHeuristicConfig& config, absl::string_view heuristic) {
  if (heuristic == heuristic::kOrientationCompensation) return memberBits(config.orientationCompensation);
  if (heuristic == heuristic::kPeriodicOrientation) return memberBits(config.periodicOrientation);
  if (heuristic == heuristic::kHeightCompensation) return memberBits(config.heightCompensation);
  if (heuristic == heuristic::kHipCenteredStepping) return memberBits(config.hipCenteredStepping);
  if (heuristic == heuristic::kCapturePoint) return memberBits(config.capturePoint);
  if (heuristic == heuristic::kTranslationalStepping) return memberBits(config.translationalStepping);
  if (heuristic == heuristic::kInPlaceTurning) return memberBits(config.inPlaceTurning);
  if (heuristic == heuristic::kHighSpeedTurning) return memberBits(config.highSpeedTurning);
  if (heuristic == heuristic::kImpulseScaling) return memberBits(config.impulseScaling);
  if (heuristic == heuristic::kCentripetalAcceleration) return memberBits(config.centripetalAcceleration);
  return {};
}

/** The parameter blocks of the schema: the message fields of LocomotionHeuristicsConfig. */
std::vector<const FieldDescriptor* absl_nonnull> blockFields() {
  std::vector<const FieldDescriptor* absl_nonnull> blocks;
  const Descriptor* absl_nonnull descriptor = HeuristicsProto::descriptor();
  for (int i = 0; i < descriptor->field_count(); ++i) {
    if (descriptor->field(i)->message_type() != nullptr) blocks.push_back(descriptor->field(i));
  }
  return blocks;
}

/** The conversion of `proto`, a block that must convert. */
LocomotionHeuristicConfig convert(const HeuristicsProto& proto) {
  mpc_config::LocomotionHeuristicsConfig config;
  EXPECT_TRUE(mpc_config::FromProto(proto, &config).ok());
  const absl::StatusOr<LocomotionHeuristicConfig> heuristics = locomotionHeuristicConfigFromConfig(config);
  EXPECT_TRUE(heuristics.ok()) << heuristics.status();
  return heuristics.ok() ? *heuristics : LocomotionHeuristicConfig{};
}

/** The conversion of the textproto block `text`, written inside `locomotion_heuristics { ... }` of a task file. */
absl::StatusOr<LocomotionHeuristicConfig> convertText(absl::string_view text) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> file =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(absl::StrCat("locomotion_heuristics {\n", text, "\n}\n"), "task.textproto");
  if (!file.ok()) return file.status();
  mpc_config::LocomotionHeuristicsConfig config;
  absl::Status converted = mpc_config::FromProto(file->locomotion_heuristics(), &config);
  if (!converted.ok()) return converted;
  return locomotionHeuristicConfigFromConfig(config);
}

TEST(LocomotionHeuristicsFromConfigTest, TheDefaultsAreTheLayersOwnNoOp) {
  const LocomotionHeuristicConfig defaults;
  const absl::StatusOr<LocomotionHeuristicConfig> fromStruct =
      locomotionHeuristicConfigFromConfig(mpc_config::LocomotionHeuristicsConfig{});
  ASSERT_TRUE(fromStruct.ok()) << fromStruct.status();
  // A task file without the block, as the parser reads it: the defaults of the protobuf descriptor, not of the struct.
  const absl::StatusOr<humanoid_mpc_config::TaskFile> file =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(/*text=*/"", "task.textproto");
  ASSERT_TRUE(file.ok()) << file.status();
  for (const LocomotionHeuristicConfig& converted : {*fromStruct, convert(file->locomotion_heuristics())}) {
    EXPECT_TRUE(converted.formulation.empty()) << "every list ships empty";
    for (const FieldDescriptor* absl_nonnull block : blockFields()) {
      EXPECT_EQ(parameterBits(converted, block->name()), parameterBits(defaults, block->name())) << block->name();
    }
  }
}

TEST(LocomotionHeuristicsFromConfigTest, TheBlocksAndListsAreTheRegistrysHeuristics) {
  std::set<std::string> registered;
  std::set<std::string> kinds;
  for (const HeuristicKind kind : allHeuristicKinds()) {
    kinds.emplace(heuristicKindName(kind));
    for (const std::string& name : knownHeuristicNames(kind)) registered.insert(name);
  }
  std::set<std::string> blocks;
  for (const FieldDescriptor* absl_nonnull block : blockFields()) blocks.emplace(block->name());
  EXPECT_EQ(blocks, registered) << "one parameter block per heuristic, named after it (LocomotionHeuristicFormulation.cpp)";

  std::set<std::string> lists;
  const Descriptor* absl_nonnull descriptor = HeuristicsProto::descriptor();
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const FieldDescriptor* absl_nonnull field = descriptor->field(i);
    if (field->message_type() != nullptr) continue;
    EXPECT_TRUE(field->is_repeated() && field->cpp_type() == FieldDescriptor::CPPTYPE_STRING) << field->name();
    lists.emplace(field->name());
  }
  EXPECT_EQ(lists, kinds) << "one name list per kind of heuristic";
}

TEST(LocomotionHeuristicsFromConfigTest, EveryParameterReachesAMemberOfItsOwn) {
  const LocomotionHeuristicConfig defaults;
  size_t parameters = 0;
  for (const FieldDescriptor* absl_nonnull block : blockFields()) {
    const Descriptor* absl_nonnull message = block->message_type();
    const std::vector<uint64_t> blockDefaults = parameterBits(defaults, block->name());
    ASSERT_EQ(blockDefaults.size(), static_cast<size_t>(message->field_count())) << block->name() << ": one member per field";
    std::set<size_t> reached;
    for (int i = 0; i < message->field_count(); ++i) {
      const FieldDescriptor* absl_nonnull field = message->field(i);
      ASSERT_EQ(field->cpp_type(), FieldDescriptor::CPPTYPE_DOUBLE) << field->full_name();
      HeuristicsProto proto;
      google::protobuf::Message* absl_nonnull blockProto = proto.GetReflection()->MutableMessage(&proto, block);
      // Away from the default, of its sign, and inside every range the layer checks.
      const double value = 0.5 * field->default_value_double() + 0.0625;
      blockProto->GetReflection()->SetDouble(blockProto, field, value);
      const LocomotionHeuristicConfig converted = convert(proto);

      for (const FieldDescriptor* absl_nonnull other : blockFields()) {
        if (other != block) {
          EXPECT_EQ(parameterBits(converted, other->name()), parameterBits(defaults, other->name())) << field->full_name();
        }
      }
      const std::vector<uint64_t> bits = parameterBits(converted, block->name());
      std::vector<size_t> changed;
      for (size_t member = 0; member < bits.size(); ++member) {
        if (bits[member] != blockDefaults[member]) changed.push_back(member);
      }
      ASSERT_EQ(changed.size(), 1u) << field->full_name() << " must set exactly one member";
      EXPECT_EQ(bits[changed.front()], std::bit_cast<uint64_t>(value)) << field->full_name();
      EXPECT_TRUE(reached.insert(changed.front()).second) << field->full_name() << " sets the member of another field";
      ++parameters;
    }
  }
  EXPECT_GT(parameters, 0u) << "the walk found no parameter, so nothing above was checked";
}

TEST(LocomotionHeuristicsFromConfigTest, TheListsAreCopiedAsWritten) {
  const absl::StatusOr<LocomotionHeuristicConfig> heuristics = convertText(
      "base_pose: \"periodicOrientation\"\n"
      "foothold: \"hip_centered_stepping\"\n"
      "foothold: \"capture_point\"\n"
      "wrench: \"impulse_scaling\"\n"
      "capture_point { gain: 0.5 }");
  ASSERT_TRUE(heuristics.ok()) << heuristics.status();
  EXPECT_EQ(heuristics->formulation.basePose, std::vector<std::string>{"periodicOrientation"});
  EXPECT_EQ(heuristics->formulation.foothold, (std::vector<std::string>{"hip_centered_stepping", "capture_point"}));
  EXPECT_EQ(heuristics->formulation.wrench, std::vector<std::string>{"impulse_scaling"});
  EXPECT_TRUE(heuristics->formulation.listed(HeuristicKind::kBasePose, heuristic::kPeriodicOrientation));
  EXPECT_EQ(heuristics->capturePoint.gain, 0.5);
}

TEST(LocomotionHeuristicsFromConfigTest, TheLayersOwnChecksStillApply) {
  const std::array<std::pair<absl::string_view, absl::string_view>, 6> refused = {{
      {"foothold: \"lean_forward\"", "unknown foothold heuristic 'lean_forward'"},
      {"foothold: \"periodic_orientation\"", "is a base_pose heuristic"},
      {"wrench: \"impulse_scaling\"\nwrench: \"impulseScaling\"", "listed more than once"},
      {"foothold: \"\"", "unknown foothold heuristic ''"},
      {"capture_point { gravity: 0 }", "capture_point.gravity must be positive"},
      {"centripetal_acceleration { maximum_force: 0 maximum_force_ratio_of_weight: 0 }", "needs a force clamp"},
  }};
  for (const std::pair<absl::string_view, absl::string_view>& entry : refused) {
    const absl::StatusOr<LocomotionHeuristicConfig> heuristics = convertText(entry.first);
    ASSERT_EQ(heuristics.status().code(), absl::StatusCode::kInvalidArgument) << entry.first;
    EXPECT_TRUE(absl::StrContains(heuristics.status().message(), entry.second)) << heuristics.status();
  }
}

TEST(LocomotionHeuristicsFromConfigTest, AParameterThatIsNotFiniteIsRefusedByName) {
  for (const FieldDescriptor* absl_nonnull block : blockFields()) {
    const Descriptor* absl_nonnull message = block->message_type();
    for (int i = 0; i < message->field_count(); ++i) {
      const FieldDescriptor* absl_nonnull field = message->field(i);
      for (const double value : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        HeuristicsProto proto;
        google::protobuf::Message* absl_nonnull blockProto = proto.GetReflection()->MutableMessage(&proto, block);
        blockProto->GetReflection()->SetDouble(blockProto, field, value);
        mpc_config::LocomotionHeuristicsConfig config;
        ASSERT_TRUE(mpc_config::FromProto(proto, &config).ok());
        const absl::StatusOr<LocomotionHeuristicConfig> heuristics = locomotionHeuristicConfigFromConfig(config);
        ASSERT_EQ(heuristics.status().code(), absl::StatusCode::kInvalidArgument) << field->full_name() << " = " << value;
        EXPECT_TRUE(absl::StrContains(heuristics.status().message(),
                                      absl::StrCat("locomotion_heuristics.", block->name(), ".", field->name(), " is ")))
            << heuristics.status();
      }
    }
  }
}

TEST(LocomotionHeuristicsFromConfigTest, TheCamelCaseSpellingOfAKeyIsRefusedWithItsNewName) {
  const absl::StatusOr<LocomotionHeuristicConfig> heuristics = convertText("capture_point {\n  comHeightOverride: 1.0805\n}");
  ASSERT_EQ(heuristics.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StartsWith(heuristics.status().message(), "task.textproto:3:3: ")) << heuristics.status();
  EXPECT_TRUE(absl::StrContains(heuristics.status().message(), "Did you mean \"com_height_override\"?")) << heuristics.status();
}

}  // namespace
}  // namespace ocs2::humanoid
