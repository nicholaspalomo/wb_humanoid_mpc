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

// The conversion of the typed contact-planning file into ContactPlanningConfig (contactPlanningConfigFromConfig()), by
// its properties (the tests T3, T4 and T6 of humanoid_nmpc/humanoid_mpc_config/README.md): the schema's defaults are the struct's, every
// value of the schema reaches the configuration, the term blocks are named as the registry's terms, the slack penalties inherit the shared
// one, and the strict parser answers a stale file with what replaced its keys.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningFormulation.h"
#include "humanoid_mpc_config/config_options.pb.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.pb.h"
#include "humanoid_mpc_config/contact_planning_file.pb.h"
#include "humanoid_mpc_config/tuning_options.pb.h"
#include "nproto/Textproto.h"

namespace ocs2::humanoid {
namespace {

using FieldDescriptor = google::protobuf::FieldDescriptor;
using FieldPath = std::vector<const FieldDescriptor* absl_nonnull>;
using File = mpc_config::ContactPlanningFile;
using FileProto = humanoid_mpc_config::ContactPlanningFile;

// The blocks of the file that hold the planner and the shared parameters, not the parameters of one term.
constexpr std::array<absl::string_view, 3> kSettingsBlocks = {"planner", "shared", "hlip"};

// The terms with no parameters in the file, each with the reason it has no block.
constexpr std::array<std::pair<absl::string_view, absl::string_view>, 13> kTermsWithoutBlock = {{
    {"lip_com", "a model block; its variables are the problem's"},
    {"foothold_integrator", "a model block; its variables are the problem's"},
    {"heading_double_integrator", "a model block; the heading costs carry its weights"},
    {"no_flight", "a hard rule on the contact binaries"},
    {"foot_motion_in_swing_only", "a hard constraint without a parameter"},
    {"yaw_torque_budget", "derived from the model and the wrench cone (ContactPlanningModelParameters)"},
    {"foot_yaw_pinned_in_contact", "a hard constraint without a parameter"},
    {"phase_durations", "reads shared.gait_limits"},
    {"minimum_double_support", "reads shared.gait_limits"},
    {"alternating_feet", "a logic rule without a parameter"},
    {"warm_start_previous_plan", "a search stage without a parameter"},
    {"planned_heading_override", "an execution rule without a parameter"},
    {"planned_com_override", "an execution rule without a parameter"},
}};

/** Every value of a ContactPlanningConfig as text, doubles by their bits, so that two configurations compare exactly. */
class Fingerprint {
 public:
  void add(absl::string_view name, double value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    absl::StrAppend(&text_, name, " = ", bits, "\n");
  }
  void add(absl::string_view name, int value) { absl::StrAppend(&text_, name, " = ", value, "\n"); }
  void add(absl::string_view name, bool value) { absl::StrAppend(&text_, name, " = ", value ? "true" : "false", "\n"); }
  void add(absl::string_view name, const std::string& value) { absl::StrAppend(&text_, name, " = \"", value, "\"\n"); }
  void add(absl::string_view name, const std::vector<std::string>& values) {
    absl::StrAppend(&text_, name, " = [", absl::StrJoin(values, ", "), "]\n");
  }
  void add(absl::string_view name, std::optional<SlackPenalty> slack) {
    if (!slack.has_value()) {
      absl::StrAppend(&text_, name, " = none\n");
      return;
    }
    add(absl::StrCat(name, ".quadratic"), slack->quadratic);
    add(absl::StrCat(name, ".linear"), slack->linear);
  }
  void add(absl::string_view name, const feet_array_t<scalar_t>& values) {
    for (size_t foot = 0; foot < values.size(); ++foot) add(absl::StrCat(name, "[", foot, "]"), values[foot]);
  }

  const std::string& text() const { return text_; }

 private:
  std::string text_;
};

std::string fingerprint(const ContactPlanningConfig& c) {
  Fingerprint f;
  f.add("planner.type", c.planner.type);
  f.add("planner.dt", c.planner.dt);
  f.add("planner.numNodes", c.planner.numNodes);
  f.add("planner.commitTime", c.planner.commitTime);
  f.add("planner.maxCommitExtension", c.planner.maxCommitExtension);
  f.add("planner.maxBranchAndBoundNodes", c.planner.maxBranchAndBoundNodes);
  f.add("planner.maxSolveTime", c.planner.maxSolveTime);
  f.add("planner.maxQpIterations", c.planner.maxQpIterations);
  f.add("planner.runInBackgroundThread", c.planner.runInBackgroundThread);
  f.add("planner.planningFrequency", c.planner.planningFrequency);
  f.add("planner.verbose", c.planner.verbose);
  f.add("planner.logPlans", c.planner.logPlans);
  f.add("shared.gravity", c.shared.gravity);
  f.add("shared.comHeight", c.shared.comHeight.value_or(0.0));
  f.add("shared.bigM", c.shared.bigM);
  f.add("shared.slackPenalty", std::optional<SlackPenalty>(c.shared.slackPenalty));
  f.add("shared.gaitLimits.minSwingDuration", c.shared.gaitLimits.minSwingDuration);
  f.add("shared.gaitLimits.maxSwingDuration", c.shared.gaitLimits.maxSwingDuration);
  f.add("shared.gaitLimits.minContactDuration", c.shared.gaitLimits.minContactDuration);
  f.add("shared.gaitLimits.maxContactDuration", c.shared.gaitLimits.maxContactDuration);
  f.add("shared.gaitLimits.minDoubleSupportDuration", c.shared.gaitLimits.minDoubleSupportDuration);
  for (const TermKind kind : allTermKinds()) f.add(termKindName(kind), c.formulation.list(kind));
  f.add("hlip.sspDuration", c.hlip.sspDuration);
  f.add("hlip.dspDuration", c.hlip.dspDuration);
  f.add("hlip.stepWidth", c.hlip.stepWidth);
  f.add("hlip.maxStepLength", c.hlip.maxStepLength);
  f.add("hlip.maxStepWidth", c.hlip.maxStepWidth);
  f.add("hlip.minStepWidth", c.hlip.minStepWidth);
  f.add("hlip.blend.sharpness", c.hlip.blend.sharpness);
  f.add("hlip.blend.threshold", c.hlip.blend.threshold);
  f.add("hlip.blend.maxCommandedVelocityX", c.hlip.blend.maxCommandedVelocityX);
  f.add("hlip.blend.maxCommandedVelocityY", c.hlip.blend.maxCommandedVelocityY);
  f.add("hlip.blend.maxCommandedYawRate", c.hlip.blend.maxCommandedYawRate);
  f.add("hlip.blend.maxComVelocityX", c.hlip.blend.maxComVelocityX);
  f.add("hlip.blend.maxComVelocityY", c.hlip.blend.maxComVelocityY);
  f.add("regularization.state", c.regularization.state);
  f.add("regularization.input", c.regularization.input);
  f.add("previousFootholdConsistency.weight", c.previousFootholdConsistency.weight);
  f.add("velocityTracking.weight", c.velocityTracking.weight);
  f.add("stepWidth.weight", c.stepWidth.weight);
  f.add("stepWidth.nominalStepWidth", c.stepWidth.nominalStepWidth);
  f.add("headingRateTracking.weight", c.headingRateTracking.weight);
  f.add("headingTracking.weight", c.headingTracking.weight);
  f.add("footYawTracking.weight", c.footYawTracking.weight);
  f.add("yawTorqueRegularization.weight", c.yawTorqueRegularization.weight);
  f.add("footYawRegularization.weight", c.footYawRegularization.weight);
  f.add("zmpRegularization.weight", c.zmpRegularization.weight);
  f.add("footholdRegularization.weight", c.footholdRegularization.weight);
  f.add("stepLength.weight", c.stepLength.weight);
  f.add("terminalDcm.weight", c.terminalDcm.weight);
  f.add("terminalDcm.trackCommandedVelocity", c.terminalDcm.trackCommandedVelocity);
  f.add("zmpSupportRegion.halfWidthX", c.zmpSupportRegion.halfWidthX);
  f.add("zmpSupportRegion.halfWidthY", c.zmpSupportRegion.halfWidthY);
  f.add("zmpSupportRegion.slack", c.zmpSupportRegion.slack);
  f.add("reachability.reachX", c.reachability.reachX);
  f.add("reachability.reachYInner", c.reachability.reachYInner);
  f.add("reachability.reachYOuter", c.reachability.reachYOuter);
  f.add("reachability.slack", c.reachability.slack);
  f.add("footSeparation.maxStepLength", c.footSeparation.maxStepLength);
  f.add("footSeparation.minStepWidth", c.footSeparation.minStepWidth);
  f.add("footSeparation.maxStepWidth", c.footSeparation.maxStepWidth);
  f.add("footSeparation.slack", c.footSeparation.slack);
  f.add("hipYawRange.lower", c.hipYawRange.lower);
  f.add("hipYawRange.upper", c.hipYawRange.upper);
  f.add("hipYawRange.slack", c.hipYawRange.slack);
  f.add("yawTorqueBudget.torsionalFrictionTorque", c.yawTorqueBudget.torsionalFrictionTorque);
  f.add("yawTorqueBudget.doubleSupportYawCouple", c.yawTorqueBudget.doubleSupportYawCouple);
  f.add("contactSwitch.cost", c.contactSwitch.cost);
  f.add("planConsistency.cost", c.planConsistency.cost);
  f.add("doubleSupportPenalty.cost", c.doubleSupportPenalty.cost);
  f.add("diving.maxDiveIterations", c.diving.maxDiveIterations);
  f.add("eventShiftLocalSearch.iterations", c.eventShiftLocalSearch.iterations);
  f.add("eventShiftLocalSearch.maxTime", c.eventShiftLocalSearch.maxTime);
  f.add("cadenceStretch.samples", c.cadenceStretch.samples);
  f.add("cadenceStretch.maxStretch", c.cadenceStretch.maxStretch);
  f.add("headingRelinearization.passes", c.headingRelinearization.passes);
  f.add("phaseResetting.earlyTouchdownMinSwingRatio", c.phaseResetting.earlyTouchdownMinSwingRatio);
  f.add("phaseResetting.earlyTouchdownMinContactDuration", c.phaseResetting.earlyTouchdownMinContactDuration);
  f.add("phaseResetting.earlyTouchdownMinAdvance", c.phaseResetting.earlyTouchdownMinAdvance);
  f.add("phaseResetting.maxLateTouchdownExtension", c.phaseResetting.maxLateTouchdownExtension);
  f.add("phaseResetting.lateTouchdownExtensionStep", c.phaseResetting.lateTouchdownExtensionStep);
  f.add("phaseResetting.lateTouchdownSearchVelocity", c.phaseResetting.lateTouchdownSearchVelocity);
  f.add("energyCadenceModulation.gain", c.energyCadenceModulation.gain);
  f.add("energyCadenceModulation.deadband", c.energyCadenceModulation.deadband);
  f.add("dcmStepAdjustment.gain", c.dcmStepAdjustment.gain);
  f.add("dcmStepAdjustment.maxOffset", c.dcmStepAdjustment.maxOffset);
  return f.text();
}

/** A file whose eight term lists are the library's default formulation, every other value absent. */
File defaultFormulationFile() {
  const ContactPlanningFormulation formulation;
  File file;
  file.dynamics = formulation.dynamics;
  file.costs = formulation.costs;
  file.soft_constraints = formulation.softConstraints;
  file.hard_constraints = formulation.hardConstraints;
  file.logic_rules = formulation.logicRules;
  file.assignment_costs = formulation.assignmentCosts;
  file.search = formulation.search;
  file.execution = formulation.execution;
  return file;
}

ContactPlanningConfig convertOrDie(const File& file) {
  const absl::StatusOr<ContactPlanningConfig> config =
      contactPlanningConfigFromConfig(file, ContactPlanningValidation::kDeferUntilModelParametersApplied);
  EXPECT_TRUE(config.ok()) << config.status();
  return config.ok() ? *config : ContactPlanningConfig{};
}

/** The configuration of the file `proto`, through its struct, as a loader makes it. */
ContactPlanningConfig convertProtoOrDie(const FileProto& proto) {
  File file;
  const absl::Status converted = FromProto(proto, &file);
  EXPECT_TRUE(converted.ok()) << converted;
  return convertOrDie(file);
}

/** Every scalar and repeated string field below `descriptor`, as the path of fields that leads to it. */
void collectLeaves(const google::protobuf::Descriptor& descriptor, FieldPath& path, std::vector<FieldPath>& leaves) {
  for (int i = 0; i < descriptor.field_count(); ++i) {
    const FieldDescriptor* absl_nonnull field = descriptor.field(i);
    path.push_back(field);
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      collectLeaves(*field->message_type(), path, leaves);
    } else {
      leaves.push_back(path);
    }
    path.pop_back();
  }
}

std::string pathName(const FieldPath& path) {
  std::vector<std::string> names;
  names.reserve(path.size());
  for (const FieldDescriptor* absl_nonnull field : path) names.emplace_back(field->name());
  return absl::StrJoin(names, ".");
}

/**
 * A value of the string `field` other than `current`: another name of its registry where the conversion resolves the
 * name (planner.threading, terminal_dcm.target), otherwise `current` renamed, which the unvalidated conversion keeps.
 */
std::string otherName(const FieldDescriptor& field, const std::string& current) {
  const std::string& registry = field.options().GetExtension(humanoid_mpc_config::tuning).registry();
  std::vector<std::string> names;
  if (registry == "contact_planner_threading") names = plannerThreadingNames();
  if (registry == "terminal_dcm_target") names = terminalDcmTargetNames();
  for (const std::string& name : names) {
    if (name != current) return name;
  }
  return absl::StrCat(current, "_changed");
}

/** Changes the leaf at `path` of `proto` from what it reads (its default when absent): x 1.5 + 0.25, +1, flipped, renamed. */
void mutate(const FieldPath& path, FileProto& proto) {
  google::protobuf::Message* absl_nonnull message = &proto;
  for (size_t i = 0; i + 1 < path.size(); ++i) {
    message = message->GetReflection()->MutableMessage(message, path[i]);
  }
  const FieldDescriptor* absl_nonnull field = path.back();
  const google::protobuf::Reflection* absl_nonnull reflection = message->GetReflection();
  if (field->is_repeated()) {
    ASSERT_EQ(field->cpp_type(), FieldDescriptor::CPPTYPE_STRING) << pathName(path);
    reflection->AddString(message, field, "a_term");
    return;
  }
  switch (field->cpp_type()) {
    case FieldDescriptor::CPPTYPE_DOUBLE:
      reflection->SetDouble(message, field, reflection->GetDouble(*message, field) * 1.5 + 0.25);
      return;
    case FieldDescriptor::CPPTYPE_INT32:
      reflection->SetInt32(message, field, reflection->GetInt32(*message, field) + 1);
      return;
    case FieldDescriptor::CPPTYPE_BOOL:
      reflection->SetBool(message, field, !reflection->GetBool(*message, field));
      return;
    case FieldDescriptor::CPPTYPE_STRING:
      reflection->SetString(message, field, otherName(*field, reflection->GetString(*message, field)));
      return;
    default:
      FAIL() << pathName(path) << " is of a kind this test does not mutate; add it";
  }
}

TEST(ContactPlanningFromConfigTest, TheSchemaDefaultsAreTheLibraryDefaults) {
  // An empty file is the library's configuration except for the term lists, which are empty (an absent list is empty).
  const ContactPlanningConfig empty = convertOrDie(File{});
  for (const TermKind kind : allTermKinds()) {
    EXPECT_TRUE(empty.formulation.list(kind).empty()) << termKindName(kind);
  }
  // Given the default formulation's lists, it is the library's configuration exactly.
  EXPECT_EQ(fingerprint(convertOrDie(defaultFormulationFile())), fingerprint(ContactPlanningConfig{}));
}

TEST(ContactPlanningFromConfigTest, EveryValueOfTheSchemaReachesTheConfiguration) {
  std::vector<FieldPath> leaves;
  FieldPath path;
  collectLeaves(*FileProto::descriptor(), path, leaves);
  ASSERT_FALSE(leaves.empty());

  const std::string base = fingerprint(convertProtoOrDie(FileProto{}));
  for (const FieldPath& leaf : leaves) {
    FileProto proto;
    mutate(leaf, proto);
    EXPECT_NE(fingerprint(convertProtoOrDie(proto)), base) << pathName(leaf) << " does not reach the configuration";
  }
}

TEST(ContactPlanningFromConfigTest, EachThreadingAndTargetNameIsTheBooleanItReplaced) {
  // The parity of the retired booleans: planner.run_in_background_thread true is "background_thread" and false
  // "pre_solve_hook"; terminal_dcm.track_commanded_velocity true is "commanded_velocity" and false "rest". The defaults
  // are the booleans' defaults.
  EXPECT_EQ(plannerThreadingNames(), (std::vector<std::string>{"background_thread", "pre_solve_hook"}));
  EXPECT_EQ(terminalDcmTargetNames(), (std::vector<std::string>{"rest", "commanded_velocity"}));
  for (const std::string& name : plannerThreadingNames()) {
    File file;
    file.planner.threading = name;
    EXPECT_EQ(convertOrDie(file).planner.runInBackgroundThread, name == "background_thread") << name;
  }
  for (const std::string& name : terminalDcmTargetNames()) {
    File file;
    file.terminal_dcm.target = name;
    EXPECT_EQ(convertOrDie(file).terminalDcm.trackCommandedVelocity, name == "commanded_velocity") << name;
  }
  EXPECT_EQ(convertOrDie(File{}).planner.runInBackgroundThread, PlannerSettings{}.runInBackgroundThread);
  EXPECT_EQ(convertOrDie(File{}).terminalDcm.trackCommandedVelocity, TerminalDcmParameters{}.trackCommandedVelocity);

  // Another name is refused, unvalidated too, listing the names.
  File threading;
  threading.planner.threading = "synchronous";
  const absl::StatusOr<ContactPlanningConfig> refusedThreading =
      contactPlanningConfigFromConfig(threading, ContactPlanningValidation::kDeferUntilModelParametersApplied);
  EXPECT_EQ(refusedThreading.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refusedThreading.status().message(), "planner.threading is 'synchronous'")) << refusedThreading.status();
  EXPECT_TRUE(absl::StrContains(refusedThreading.status().message(), "background_thread, pre_solve_hook")) << refusedThreading.status();
  File target;
  target.terminal_dcm.target = "walk";
  const absl::StatusOr<ContactPlanningConfig> refusedTarget =
      contactPlanningConfigFromConfig(target, ContactPlanningValidation::kDeferUntilModelParametersApplied);
  EXPECT_EQ(refusedTarget.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(refusedTarget.status().message(), "terminal_dcm.target is 'walk'")) << refusedTarget.status();
  EXPECT_TRUE(absl::StrContains(refusedTarget.status().message(), "rest, commanded_velocity")) << refusedTarget.status();
}

TEST(ContactPlanningFromConfigTest, AComHeightIsPositiveOrLeftOutForTheModels) {
  // Left out, it is the model's pendulum: unset, for ContactPlanningModelParameters::applyTo() to fill in.
  EXPECT_FALSE(convertOrDie(File{}).shared.comHeight.has_value());
  EXPECT_EQ(convertOrDie(File{}).shared.comHeight, SharedParameters{}.comHeight);
  File given;
  given.shared.com_height = 1.0805;
  EXPECT_EQ(convertOrDie(given).shared.comHeight, 1.0805);
  // 0, which stood for the model's before the field was optional, is refused like any other height that is no height.
  for (const double height : {0.0, -0.5, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
    File refused;
    refused.shared.com_height = height;
    const absl::StatusOr<ContactPlanningConfig> config =
        contactPlanningConfigFromConfig(refused, ContactPlanningValidation::kDeferUntilModelParametersApplied);
    EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument) << height;
    EXPECT_TRUE(absl::StrContains(config.status().message(), "shared.com_height is")) << config.status();
    EXPECT_TRUE(absl::StrContains(config.status().message(), "leave shared.com_height out for the model's")) << config.status();
  }
}

TEST(ContactPlanningFromConfigTest, TheTermBlocksAndListsAreNamedAsTheRegistrysTerms) {
  const google::protobuf::Descriptor* absl_nonnull descriptor = FileProto::descriptor();
  absl::flat_hash_set<std::string> blocks;
  absl::flat_hash_set<std::string> lists;
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const FieldDescriptor* absl_nonnull field = descriptor->field(i);
    if (field->cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      if (std::find(kSettingsBlocks.begin(), kSettingsBlocks.end(), field->name()) == kSettingsBlocks.end()) {
        blocks.emplace(field->name());
      }
    } else {
      EXPECT_TRUE(field->is_repeated()) << field->name() << " is neither a block nor a term list";
      lists.emplace(field->name());
    }
  }

  absl::flat_hash_set<std::string> kindNames;
  for (const TermKind kind : allTermKinds()) kindNames.insert(termKindName(kind));
  EXPECT_EQ(lists, kindNames);

  // Every term has its block or a reason not to, and every block is a term's, named exactly as the term.
  absl::flat_hash_set<std::string> termsWithoutBlock;
  for (const std::pair<absl::string_view, absl::string_view>& term : kTermsWithoutBlock) termsWithoutBlock.emplace(term.first);
  absl::flat_hash_set<std::string> terms;
  for (const TermKind kind : allTermKinds()) {
    for (const std::string& name : knownTermNames(kind)) {
      terms.insert(name);
      EXPECT_NE(blocks.contains(name), termsWithoutBlock.contains(name))
          << name << " needs exactly one of: a block of ContactPlanningFile, an entry of kTermsWithoutBlock";
    }
  }
  for (const std::string& block : blocks) EXPECT_TRUE(terms.contains(block)) << block << " is the block of no term";
  for (const std::string& term : termsWithoutBlock) EXPECT_TRUE(terms.contains(term)) << term << " is no term";
}

TEST(ContactPlanningFromConfigTest, ASlackBlockInheritsTheHalfItOmitsFromTheSharedPenalty) {
  File file = defaultFormulationFile();
  file.shared.slack_penalty.quadratic = 7.0;
  file.zmp_support_region.slack = File::TermSlackConfig{};
  file.zmp_support_region.slack->linear = 3.0;
  file.foot_separation.slack = File::TermSlackConfig{};
  const ContactPlanningConfig config = convertOrDie(file);

  // A penalty no conversion writes, for the slacks that are missing.
  const SlackPenalty missing{.quadratic = -1.0, .linear = -1.0};
  EXPECT_TRUE(config.zmpSupportRegion.slack.has_value());
  const SlackPenalty zmp = config.zmpSupportRegion.slack.value_or(missing);
  EXPECT_EQ(zmp.quadratic, 7.0);
  EXPECT_EQ(zmp.linear, 3.0);
  EXPECT_TRUE(config.footSeparation.slack.has_value());
  const SlackPenalty separation = config.footSeparation.slack.value_or(missing);
  EXPECT_EQ(separation.quadratic, 7.0);
  EXPECT_EQ(separation.linear, SlackPenalty{}.linear);
  EXPECT_FALSE(config.reachability.slack.has_value()) << "a term without a slack block uses the shared penalty, not one of its own";
  EXPECT_FALSE(config.hipYawRange.slack.has_value());
}

TEST(ContactPlanningFromConfigTest, AnEmptyTermNameIsRefusedWithItsList) {
  File file = defaultFormulationFile();
  file.costs = {"regularization", ""};
  const absl::StatusOr<ContactPlanningConfig> config =
      contactPlanningConfigFromConfig(file, ContactPlanningValidation::kDeferUntilModelParametersApplied);
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(config.status().message(), "costs[1] is empty")) << config.status();
}

TEST(ContactPlanningFromConfigTest, ValidationIsTheConfigurationsOwn) {
  File file = defaultFormulationFile();
  // The library's pendulum height is 0, "from the model", which validation refuses until it has been filled in.
  const absl::StatusOr<ContactPlanningConfig> unresolved = contactPlanningConfigFromConfig(file, ContactPlanningValidation::kValidate);
  EXPECT_EQ(unresolved.status().code(), absl::StatusCode::kInvalidArgument) << unresolved.status();
  EXPECT_TRUE(contactPlanningConfigFromConfig(file, ContactPlanningValidation::kDeferUntilModelParametersApplied).ok());

  file.shared.com_height = 1.0;
  const absl::StatusOr<ContactPlanningConfig> validated = contactPlanningConfigFromConfig(file, ContactPlanningValidation::kValidate);
  EXPECT_TRUE(validated.ok()) << validated.status();

  file.dynamics.clear();
  EXPECT_FALSE(contactPlanningConfigFromConfig(file, ContactPlanningValidation::kValidate).ok())
      << "a formulation without its mandatory model blocks";
}

TEST(ContactPlanningFromConfigTest, AnUnknownPlannerOrTermFailsTheValidatedConversionAsAWhole) {
  // A typo must fail a reload atomically, leaving the running planner and its configuration in force: the parameter
  // updater applies nothing of a configuration that does not convert.
  File file = defaultFormulationFile();
  file.shared.com_height = 0.85;
  file.planner.type = "hilp";
  const absl::StatusOr<ContactPlanningConfig> planner = contactPlanningConfigFromConfig(file, ContactPlanningValidation::kValidate);
  EXPECT_EQ(planner.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(planner.status().message(), "planner.type")) << planner.status();

  file = defaultFormulationFile();
  file.shared.com_height = 0.85;
  file.costs = {"regularization", "gravity_compensation"};
  const absl::StatusOr<ContactPlanningConfig> term = contactPlanningConfigFromConfig(file, ContactPlanningValidation::kValidate);
  EXPECT_EQ(term.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(term.status().message(), "costs")) << term.status();
  EXPECT_TRUE(absl::StrContains(term.status().message(), "gravity_compensation")) << term.status();
}

TEST(ContactPlanningFileTest, AFileThatCannotBeReadIsNotFoundNamingIt) {
  const std::string missing = absl::StrCat(::testing::TempDir(), "/no_such_contact_planning.textproto");
  const absl::StatusOr<File> file = loadContactPlanningFile(missing);
  EXPECT_EQ(file.status().code(), absl::StatusCode::kNotFound);
  EXPECT_TRUE(absl::StrContains(file.status().message(), missing)) << file.status();
}

TEST(ContactPlanningFileTest, TheWrapperBlockOfTheOldFileIsRetired) {
  const absl::StatusOr<FileProto> wrapped =
      nproto::ParseTextproto<FileProto>("contact_planning {\n  planner {\n  }\n}\n", "contact_planning.textproto");
  EXPECT_EQ(wrapped.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(wrapped.status().message(), "contact_planning.textproto:1:1: 'contact_planning' is retired: the file is"))
      << wrapped.status();
}

TEST(ContactPlanningFileTest, AKeyOfTheFlatLayoutGetsTheLayoutHint) {
  const absl::StatusOr<FileProto> flat = nproto::ParseTextproto<FileProto>("useAcomDynamics: true\n", "contact_planning.textproto");
  EXPECT_EQ(flat.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(flat.status().message(), "contact_planning.textproto:1:1:")) << flat.status();
  EXPECT_TRUE(absl::StrContains(flat.status().message(), "The flat layout of the previous planner")) << flat.status();
  EXPECT_TRUE(absl::StrContains(flat.status().message(), "useAcomDynamics -> heading_double_integrator")) << flat.status();
}

TEST(ContactPlanningFileTest, ACamelCaseSpellingOfAKeyIsAnErrorThatSuggestsItsField) {
  const absl::StatusOr<FileProto> camel = nproto::ParseTextproto<FileProto>("planner {\n  numNodes: 12\n}\n", "contact_planning.textproto");
  EXPECT_EQ(camel.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(camel.status().message(), "contact_planning.textproto:2:3:")) << camel.status();
  EXPECT_TRUE(absl::StrContains(camel.status().message(), "Did you mean \"num_nodes\"?")) << camel.status();
}

}  // namespace
}  // namespace ocs2::humanoid
