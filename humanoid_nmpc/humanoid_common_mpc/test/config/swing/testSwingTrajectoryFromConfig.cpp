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

// The swing trajectory settings from the typed task file (SwingTrajectoryFromConfig.h): the schema's defaults are the
// planner's, every field of the schema reaches its own setting, a value that is not finite is refused, and the old
// camelCase spelling of a key is refused by the parser.

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <type_traits>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/swing/SwingTrajectoryFromConfig.h"
#include "humanoid_common_mpc/swing_foot_planner/SwingTrajectoryPlanner.h"
#include "humanoid_mpc_config/swing_trajectory_config.nproto.h"
#include "humanoid_mpc_config/swing_trajectory_config.nproto.pb.h"
#include "humanoid_mpc_config/swing_trajectory_config.pb.h"
#include "humanoid_mpc_config/task_file.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

using Settings = SwingTrajectoryPlanner::Config;

/**
 * The bits of every member of `settings`, in declaration order. Settings holds nothing but scalar_t members, so its
 * bytes are those members one after the other; the test of every field checks that their number is the schema's.
 */
std::vector<uint64_t> memberBits(const Settings& settings) {
  static_assert(std::is_trivially_copyable_v<Settings> && sizeof(Settings) % sizeof(double) == 0);
  std::array<double, sizeof(Settings) / sizeof(double)> members{};
  std::memcpy(members.data(), &settings, sizeof(Settings));
  std::vector<uint64_t> bits;
  for (const double member : members) bits.push_back(std::bit_cast<uint64_t>(member));
  return bits;
}

/** The settings of the block `proto`, which must convert. */
Settings convert(const humanoid_mpc_config::SwingTrajectoryConfig& proto) {
  mpc_config::SwingTrajectoryConfig config;
  EXPECT_TRUE(mpc_config::FromProto(proto, &config).ok());
  const absl::StatusOr<Settings> settings = swingTrajectorySettingsFromConfig(config);
  EXPECT_TRUE(settings.ok()) << settings.status();
  return settings.ok() ? *settings : Settings{};
}

TEST(SwingTrajectoryFromConfigTest, TheDefaultsAreThePlannersOwn) {
  const absl::StatusOr<Settings> fromStruct = swingTrajectorySettingsFromConfig(mpc_config::SwingTrajectoryConfig{});
  ASSERT_TRUE(fromStruct.ok()) << fromStruct.status();
  EXPECT_EQ(memberBits(*fromStruct), memberBits(Settings{}));

  // A task file without the block, as the parser reads it: the defaults of the protobuf descriptor, not of the struct.
  const absl::StatusOr<humanoid_mpc_config::TaskFile> file =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(/*text=*/"", "task.textproto");
  ASSERT_TRUE(file.ok()) << file.status();
  EXPECT_EQ(memberBits(convert(file->swing_trajectory_config())), memberBits(Settings{}));
}

TEST(SwingTrajectoryFromConfigTest, EveryFieldReachesASettingOfItsOwn) {
  const google::protobuf::Descriptor* absl_nonnull descriptor = humanoid_mpc_config::SwingTrajectoryConfig::descriptor();
  const std::vector<uint64_t> defaults = memberBits(Settings{});
  ASSERT_EQ(defaults.size(), static_cast<size_t>(descriptor->field_count())) << "one setting per field of the schema";

  std::set<size_t> reached;
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const google::protobuf::FieldDescriptor* absl_nonnull field = descriptor->field(i);
    ASSERT_EQ(field->cpp_type(), google::protobuf::FieldDescriptor::CPPTYPE_DOUBLE) << field->name();
    humanoid_mpc_config::SwingTrajectoryConfig proto;
    // Away from the default, and of the default's sign.
    const double value = 0.5 * field->default_value_double() + 0.0625;
    proto.GetReflection()->SetDouble(&proto, field, value);

    const std::vector<uint64_t> bits = memberBits(convert(proto));
    std::vector<size_t> changed;
    for (size_t member = 0; member < bits.size(); ++member) {
      if (bits[member] != defaults[member]) changed.push_back(member);
    }
    ASSERT_EQ(changed.size(), 1u) << field->name() << " must set exactly one setting";
    EXPECT_EQ(bits[changed.front()], std::bit_cast<uint64_t>(value)) << field->name();
    EXPECT_TRUE(reached.insert(changed.front()).second) << field->name() << " sets the setting of another field";
  }
}

TEST(SwingTrajectoryFromConfigTest, AValueThatIsNotFiniteIsRefusedByName) {
  const google::protobuf::Descriptor* absl_nonnull descriptor = humanoid_mpc_config::SwingTrajectoryConfig::descriptor();
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const google::protobuf::FieldDescriptor* absl_nonnull field = descriptor->field(i);
    for (const double value :
         {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
      humanoid_mpc_config::SwingTrajectoryConfig proto;
      proto.GetReflection()->SetDouble(&proto, field, value);
      mpc_config::SwingTrajectoryConfig config;
      ASSERT_TRUE(mpc_config::FromProto(proto, &config).ok());
      const absl::StatusOr<Settings> settings = swingTrajectorySettingsFromConfig(config);
      ASSERT_EQ(settings.status().code(), absl::StatusCode::kInvalidArgument) << field->name() << " = " << value;
      EXPECT_TRUE(absl::StrContains(settings.status().message(), absl::StrCat("swing_trajectory_config.", field->name(), " is ")))
          << settings.status();
    }
  }
}

TEST(SwingTrajectoryFromConfigTest, TheCamelCaseSpellingOfAKeyIsRefusedWithItsNewName) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> file =
      nproto::ParseTextproto<humanoid_mpc_config::TaskFile>("swing_trajectory_config {\n  swingTimeScale: 0.25\n}\n", "task.textproto");
  ASSERT_EQ(file.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StartsWith(file.status().message(), "task.textproto:2:3: ")) << file.status();
  EXPECT_TRUE(absl::StrContains(file.status().message(), "Did you mean \"swing_time_scale\"?")) << file.status();
}

TEST(SwingTrajectoryFromConfigTest, AWrittenValueIsTheSetting) {
  const absl::StatusOr<humanoid_mpc_config::TaskFile> file = nproto::ParseTextproto<humanoid_mpc_config::TaskFile>(
      "swing_trajectory_config {\n  swing_height: 0.08\n  touch_down_velocity: -0.0\n}\n", "task.textproto");
  ASSERT_TRUE(file.ok()) << file.status();
  const Settings settings = convert(file->swing_trajectory_config());
  EXPECT_EQ(settings.swingHeight, 0.08);
  // The negative zero the G1 files write survives as one.
  EXPECT_TRUE(std::signbit(settings.touchDownVelocity));
  EXPECT_EQ(settings.swingTimeScale, Settings{}.swingTimeScale);
}

}  // namespace
}  // namespace ocs2::humanoid
