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

// Every value of the configuration schemas reaches what the stack makes of the files: the permanent test T4 of
// humanoid_nmpc/humanoid_mpc_config/README.md ("Tests"). For each scalar of the task, reference, PD gains and gait
// files - every field of every block, every element of every list the shipped files fill, each block the files leave
// out at its defaults - the test changes the value (a double x 1.5 + 0.25, an integer + 1, a bool flipped, a registry
// name to another of its registry, any other string renamed, a list grown by one), runs every typed conversion
// (dumpConversions()) and requires that what they make changes, on the shipped files of a centroidal or of the
// whole-body configuration. A field added to a schema without its conversion fails here, naming it. The contact
// planner's file has a walk of its own (humanoid_common_mpc/test/config/contact_planning).

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <array>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "gmock/gmock.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_mpc_config/config_options.pb.h"
#include "humanoid_mpc_config/config_registries.nproto.h"
#include "humanoid_mpc_config/gait_file.nproto.pb.h"
#include "humanoid_mpc_config/joint_pd_gains_file.nproto.pb.h"
#include "humanoid_mpc_config/reference_file.nproto.pb.h"
#include "humanoid_mpc_config/task_file.nproto.pb.h"
#include "humanoid_mpc_config/tuning_options.pb.h"
#include "humanoid_mpc_validation/closed_loop/RobotConfiguration.h"
#include "nproto/Textproto.h"
#include "tools/config_dump/ConversionDump.h"
#include "tools/config_registries/ConfigRegistries.h"

namespace ocs2::humanoid::config_dump {
namespace {

using google::protobuf::Descriptor;
using google::protobuf::FieldDescriptor;
using google::protobuf::Message;
using google::protobuf::Reflection;

// The configurations whose shipped files the walk starts from: the centroidal MPC with the most features switched on,
// and the whole-body MPC.
constexpr std::array<absl::string_view, 2> kConfigurations = {"drc_atlas", "unitree_g1_wb"};

// The fields no conversion of the C++ stack reads, each with what does (by its path, elements as [*]).
const std::map<std::string, std::string>& notReadByTheConversions() {
  static const std::map<std::string, std::string>& kFields = *new std::map<std::string, std::string>{
      {"TaskFile.enable_online_tuning", "the tuning GUI's own switch (consumer gui): remote_control/config_files.py reads it"},
  };
  return kFields;
}

/** One step of the path to a value: a field, and the element of it for a repeated field (-1 for a singular one). */
struct Step {
  const FieldDescriptor* absl_nonnull field;
  int index = -1;
};

/** A value of a file: the path to it; `grow` for a list that the walk grows by one element instead of changing one. */
struct Leaf {
  std::vector<Step> path;
  bool grow = false;
};

/** `Message.a.b[*].c`: the path of a leaf with its elements as [*], the key of notReadByTheConversions(). */
std::string schemaPath(const Descriptor& file, const Leaf& leaf) {
  std::string path(file.name());
  for (const Step& step : leaf.path) {
    absl::StrAppend(&path, ".", step.field->name(), step.index >= 0 ? "[*]" : "");
  }
  return path;
}

/** `a.b[3].c`: the path of a leaf with its elements, for the messages. */
std::string elementPath(const Leaf& leaf) {
  std::vector<std::string> steps;
  for (const Step& step : leaf.path) {
    steps.push_back(step.index >= 0 ? absl::StrCat(step.field->name(), "[", step.index, "]") : std::string(step.field->name()));
  }
  return absl::StrJoin(steps, ".");
}

/** Every value below `message`: its scalars and lists, each element of its lists of blocks, and each block it leaves out. */
void collectLeaves(const Message& message, std::vector<Step>& path, std::vector<Leaf>& leaves) {
  const Descriptor* absl_nonnull descriptor = message.GetDescriptor();
  const Reflection* absl_nonnull reflection = message.GetReflection();
  for (int i = 0; i < descriptor->field_count(); ++i) {
    const FieldDescriptor* absl_nonnull field = descriptor->field(i);
    if (field->cpp_type() != FieldDescriptor::CPPTYPE_MESSAGE) {
      path.push_back({.field = field});
      leaves.push_back({.path = path});
      path.pop_back();
    } else if (field->is_repeated()) {
      path.push_back({.field = field});
      leaves.push_back({.path = path, .grow = true});
      path.pop_back();
      for (int element = 0; element < reflection->FieldSize(message, field); ++element) {
        path.push_back({.field = field, .index = element});
        collectLeaves(reflection->GetRepeatedMessage(message, field, element), path, leaves);
        path.pop_back();
      }
    } else {
      path.push_back({.field = field});
      collectLeaves(reflection->GetMessage(message, field), path, leaves);
      path.pop_back();
    }
  }
}

/** The names of the registry `registry`, from the registries themselves. */
std::vector<std::string> registryNames(const std::string& registry) {
  for (const mpc_config::ConfigRegistries::Registry& entry : config_registries::collectConfigRegistries().registries) {
    if (entry.name == registry) return entry.names;
  }
  return {};
}

/** A name of `field`'s registry other than `current`, or `current` renamed for a field without one. */
std::string otherName(const FieldDescriptor& field, const std::string& current) {
  const std::string& registry = field.options().GetExtension(humanoid_mpc_config::tuning).registry();
  for (const std::string& name : registryNames(registry)) {
    if (name != current) return name;
  }
  return absl::StrCat(current, "_changed");
}

/** Changes the singular scalar `field` of `message` from what it reads (its default when absent). */
void changeSingular(Message& message, const FieldDescriptor& field) {
  const Reflection* absl_nonnull reflection = message.GetReflection();
  switch (field.cpp_type()) {
    case FieldDescriptor::CPPTYPE_DOUBLE:
      reflection->SetDouble(&message, &field, reflection->GetDouble(message, &field) * 1.5 + 0.25);
      return;
    case FieldDescriptor::CPPTYPE_INT32:
      reflection->SetInt32(&message, &field, reflection->GetInt32(message, &field) + 1);
      return;
    case FieldDescriptor::CPPTYPE_INT64:
      reflection->SetInt64(&message, &field, reflection->GetInt64(message, &field) + 1);
      return;
    case FieldDescriptor::CPPTYPE_BOOL:
      reflection->SetBool(&message, &field, !reflection->GetBool(message, &field));
      return;
    case FieldDescriptor::CPPTYPE_STRING:
      reflection->SetString(&message, &field, otherName(field, reflection->GetString(message, &field)));
      return;
    default:
      ADD_FAILURE() << field.full_name() << " is of a kind this test does not change; add it";
  }
}

/** Changes the first element of the list of scalars `field` of `message`. */
void changeFirstElement(Message& message, const FieldDescriptor& field) {
  const Reflection* absl_nonnull reflection = message.GetReflection();
  switch (field.cpp_type()) {
    case FieldDescriptor::CPPTYPE_DOUBLE:
      reflection->SetRepeatedDouble(&message, &field, /*index=*/0,
                                    reflection->GetRepeatedDouble(message, &field, /*index=*/0) * 1.5 + 0.25);
      return;
    case FieldDescriptor::CPPTYPE_INT32:
      reflection->SetRepeatedInt32(&message, &field, /*index=*/0, reflection->GetRepeatedInt32(message, &field, /*index=*/0) + 1);
      return;
    case FieldDescriptor::CPPTYPE_STRING:
      reflection->SetRepeatedString(&message, &field, /*index=*/0,
                                    otherName(field, reflection->GetRepeatedString(message, &field, /*index=*/0)));
      return;
    default:
      ADD_FAILURE() << field.full_name() << " is a list of a kind this test does not change; add it";
  }
}

/** Adds a value to the list `field` of `message`: a name of its registry not in it, a renamed string, or a number. */
void growList(Message& message, const FieldDescriptor& field) {
  const Reflection* absl_nonnull reflection = message.GetReflection();
  switch (field.cpp_type()) {
    case FieldDescriptor::CPPTYPE_MESSAGE:
      reflection->AddMessage(&message, &field);
      return;
    case FieldDescriptor::CPPTYPE_DOUBLE:
      reflection->AddDouble(&message, &field, /*value=*/0.25);
      return;
    case FieldDescriptor::CPPTYPE_INT32:
      reflection->AddInt32(&message, &field, /*value=*/1);
      return;
    case FieldDescriptor::CPPTYPE_STRING: {
      std::set<std::string> present;
      for (int i = 0; i < reflection->FieldSize(message, &field); ++i) present.insert(reflection->GetRepeatedString(message, &field, i));
      std::string added = "a_name_of_no_registry";
      for (const std::string& name : registryNames(field.options().GetExtension(humanoid_mpc_config::tuning).registry())) {
        if (!present.contains(name)) {
          added = name;
          break;
        }
      }
      reflection->AddString(&message, &field, added);
      return;
    }
    default:
      ADD_FAILURE() << field.full_name() << " is a list of a kind this test does not grow; add it";
  }
}

/** Changes the value `leaf` of `file`. */
void change(const Leaf& leaf, Message& file) {
  Message* absl_nonnull message = &file;
  for (size_t i = 0; i + 1 < leaf.path.size(); ++i) {
    const Step& step = leaf.path[i];
    message = step.index >= 0 ? message->GetReflection()->MutableRepeatedMessage(message, step.field, step.index)
                              : message->GetReflection()->MutableMessage(message, step.field);
  }
  const Step& last = leaf.path.back();
  if (leaf.grow || (last.field->is_repeated() && message->GetReflection()->FieldSize(*message, last.field) == 0)) {
    growList(*message, *last.field);
  } else if (last.field->is_repeated()) {
    changeFirstElement(*message, *last.field);
  } else {
    changeSingular(*message, *last.field);
  }
}

/** The shipped files of a configuration, as protobuf messages: what the walk changes one value of at a time. */
struct Files {
  humanoid_mpc_config::TaskFile task;
  humanoid_mpc_config::ReferenceFile reference;
  humanoid_mpc_config::JointPdGainsFile pdGains;
  humanoid_mpc_config::GaitFile gait;
};

absl::StatusOr<Files> loadFiles(const validation::RobotConfiguration& configuration) {
  Files files;
  absl::StatusOr<humanoid_mpc_config::TaskFile> task = nproto::ParseTextprotoFile<humanoid_mpc_config::TaskFile>(configuration.taskFile);
  if (!task.ok()) return task.status();
  files.task = *std::move(task);
  absl::StatusOr<humanoid_mpc_config::ReferenceFile> reference =
      nproto::ParseTextprotoFile<humanoid_mpc_config::ReferenceFile>(configuration.referenceFile);
  if (!reference.ok()) return reference.status();
  files.reference = *std::move(reference);
  absl::StatusOr<humanoid_mpc_config::JointPdGainsFile> pdGains =
      nproto::ParseTextprotoFile<humanoid_mpc_config::JointPdGainsFile>(configuration.pdGainsFile);
  if (!pdGains.ok()) return pdGains.status();
  files.pdGains = *std::move(pdGains);
  absl::StatusOr<humanoid_mpc_config::GaitFile> gait = nproto::ParseTextprotoFile<humanoid_mpc_config::GaitFile>(configuration.gaitFile);
  if (!gait.ok()) return gait.status();
  files.gait = *std::move(gait);
  return files;
}

/**
 * Weighs every joint of the whole-body MPC's model with 1 in `files`' joint_torque_weights where the file weighs none:
 * the shipped file does not list joint_torque_cost, and a block that names no joint is refused before its scaling is
 * read, so the walk could not see the scaling reach the weights.
 */
absl::Status weighEveryJointTorque(const validation::RobotConfiguration& configuration, Files& files) {
  if (files.task.joint_torque_weights().joints_size() > 0) return absl::OkStatus();
  mpc_config::TaskFile task;
  absl::Status converted = mpc_config::FromProto(files.task, &task);
  if (!converted.ok()) return converted;
  const absl::StatusOr<ModelSettings> model = ModelSettings::Create(task, configuration.urdfFile, "wb_mpc_", /*verbose=*/false);
  if (!model.ok()) return model.status();
  for (const std::string& joint : model->mpcModelJointNames) {
    humanoid_mpc_config::JointValue* absl_nonnull weight = files.task.mutable_joint_torque_weights()->add_joints();
    weight->set_joint(joint);
    weight->set_value(1.0);
  }
  return absl::OkStatus();
}

/** What the conversions of `configuration` make of `files`; an error of the structs' conversion is part of the dump. */
std::string dump(const validation::RobotConfiguration& configuration, const Files& files) {
  ConversionInputs inputs;
  absl::Status converted = mpc_config::FromProto(files.task, &inputs.task);
  converted.Update(mpc_config::FromProto(files.reference, &inputs.reference));
  converted.Update(mpc_config::FromProto(files.pdGains, &inputs.pdGains));
  converted.Update(mpc_config::FromProto(files.gait, &inputs.gait));
  if (!converted.ok()) return converted.ToString();
  const StateInputLayout::Mpc mpc = configuration.formulation == validation::MpcFormulation::kWholeBody
                                        ? StateInputLayout::Mpc::kWholeBody
                                        : StateInputLayout::Mpc::kCentroidal;
  return dumpConversions(inputs, configuration.urdfFile, mpc);
}

/** The lines of a dump that are a conversion's error. */
std::vector<std::string> errorLines(absl::string_view dump) {
  std::vector<std::string> errors;
  for (const absl::string_view line : absl::StrSplit(dump, '\n')) {
    if (absl::StrContains(line, ".error = ") || absl::StartsWith(line, "error = ")) errors.emplace_back(line);
  }
  return errors;
}

/** The file of `files` a walk changes: the message with descriptor `descriptor`. */
Message& fileOf(Files& files, const Descriptor& descriptor) {
  if (&descriptor == humanoid_mpc_config::ReferenceFile::descriptor()) return files.reference;
  if (&descriptor == humanoid_mpc_config::JointPdGainsFile::descriptor()) return files.pdGains;
  if (&descriptor == humanoid_mpc_config::GaitFile::descriptor()) return files.gait;
  return files.task;
}

TEST(EveryFieldIsConsumedTest, EveryValueOfTheFileSchemasChangesWhatTheConversionsMake) {
  // A schema path is consumed when changing one of its values changes the dump of a configuration.
  std::map<std::string, bool> consumed;
  std::map<std::string, std::string> example;
  size_t values = 0;
  for (const absl::string_view name : kConfigurations) {
    const absl::StatusOr<validation::RobotConfiguration> configuration = validation::findRobotConfiguration(name);
    ASSERT_TRUE(configuration.ok()) << configuration.status();
    absl::StatusOr<Files> files = loadFiles(*configuration);
    ASSERT_TRUE(files.ok()) << files.status();
    if (configuration->formulation == validation::MpcFormulation::kWholeBody) {
      const absl::Status weighed = weighEveryJointTorque(*configuration, *files);
      ASSERT_TRUE(weighed.ok()) << weighed;
    }
    // The dump of the shipped files, where a block its MPC does not read may be refused (the whole-body MPC's file has
    // no com_weights): what a conversion refuses is part of the dump, so that a value which makes it convert changes it.
    const std::string base = dump(*configuration, *files);
    ASSERT_THAT(errorLines(base), ::testing::Not(::testing::Contains(::testing::StartsWith("model_settings.error"))))
        << name << ": the model settings convert, which the weights and the contacts are built on";
    for (const Descriptor* absl_nonnull schema :
         {humanoid_mpc_config::TaskFile::descriptor(), humanoid_mpc_config::ReferenceFile::descriptor(),
          humanoid_mpc_config::JointPdGainsFile::descriptor(), humanoid_mpc_config::GaitFile::descriptor()}) {
      Files walked = *files;
      std::vector<Leaf> leaves;
      std::vector<Step> path;
      collectLeaves(fileOf(walked, *schema), path, leaves);
      ASSERT_FALSE(leaves.empty()) << schema->name();
      for (const Leaf& leaf : leaves) {
        Files changed = *files;
        change(leaf, fileOf(changed, *schema));
        const std::string key = schemaPath(*schema, leaf);
        const bool changes = dump(*configuration, changed) != base;
        consumed[key] = consumed[key] || changes;
        ++values;
        if (!changes) example[key] = absl::StrCat(name, ": ", elementPath(leaf));
      }
    }
  }
  // Every block of every file is walked, the elements of the lists the shipped files fill among them.
  ASSERT_GT(consumed.size(), 300) << "the walk found fewer fields than the schemas have";
  ASSERT_GT(values, 1000) << "the walk found fewer values than the shipped files have";
  for (const std::pair<const std::string, bool>& entry : consumed) {
    const bool exempt = notReadByTheConversions().contains(entry.first);
    if (exempt) {
      EXPECT_FALSE(entry.second) << entry.first << " is read by a conversion: take it out of notReadByTheConversions()";
    } else {
      EXPECT_TRUE(entry.second) << entry.first << " changes nothing any conversion makes (" << example[entry.first]
                                << "): add its conversion, or say in notReadByTheConversions() what reads it";
    }
  }
}

}  // namespace
}  // namespace ocs2::humanoid::config_dump
