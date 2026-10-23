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

#include "pinocchio/fwd.hpp"  // forward declarations must be included first.

#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"

#include "humanoid_common_mpc/config/ConfigReload.h"
#include "humanoid_common_mpc/config/contact_planning/ContactPlanningFromConfig.h"
#include "humanoid_common_mpc/contact_planning/ContactPlannerModule.h"
#include "humanoid_common_mpc/contact_planning/ContactPlanningReferenceManager.h"
#include "humanoid_common_mpc/contact_planning/TargetContactPose.h"
#include "humanoid_common_mpc/gait/MotionPhaseDefinition.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.h"
#include "humanoid_mpc_config/contact_planning_file.nproto.pb.h"
#include "humanoid_mpc_config/contact_planning_file.pb.h"
#include "support/AtlasReferenceStack.h"

/*
 * A reload of the contact planner's file is a fresh start (the contact-planning half of test L4 of the live-tuning
 * design, humanoid_nmpc/humanoid_mpc_config/README.md): every RELOAD_HOT field of the file is perturbed into the file E -
 * every real number it sets scaled by a few percent, and every name list of terms (costs, soft and hard constraints,
 * logic rules, assignment costs, the searches and the execution rules; not the dynamics, which the terms need) left one
 * entry short - and a DRC Atlas reference
 * stack started from the file and handed E as MpcParameterUpdaterModule hands a reloaded file over (converted without
 * the model's parameters, then ContactPlannerModule::setConfig()) plans exactly as a stack started from E: the same mode
 * schedule, target trajectories, contact flags, swing references and target contact poses, solve for solve, standing
 * and then walking. Under the mixed-integer planner, which reads every list, on the grid testMpcResetState gives it so
 * that the node limit, not the speed of the machine, ends every search.
 *
 * No CppAD model is built.
 */
namespace ocs2::humanoid {
namespace {

using google::protobuf::Descriptor;
using google::protobuf::FieldDescriptor;
using google::protobuf::Message;
using google::protobuf::Reflection;
using ContactPlanningProto = humanoid_mpc_config::ContactPlanningFile;

constexpr scalar_t kSolvePeriod = 0.01;  // [s]
constexpr int kStandingSolves = 20;
constexpr int kWalkingSolves = 150;
constexpr scalar_t kWalkingSpeed = 0.6;   // [m/s]
constexpr scalar_t kCommandLimitX = 1.2;  // [m/s] the shipped reference file's max_displacement_velocity_x
constexpr scalar_t kScale = 1.04;         // the factor of every real number of E
constexpr scalar_t kTolerance = 1.0e-9;

/**
 * The shipped DRC Atlas contact planner's file on the mixed-integer planner's grid of testMpcResetState, planned in the
 * pre-solve hook so that both stacks plan synchronously.
 */
ContactPlanningProto baseFile() {
  const AtlasReferenceStack shipped(AtlasReferenceStack::ScheduleSource::kGaitSchedule);
  ContactPlanningProto proto;
  EXPECT_TRUE(shipped.config().contactPlanning.has_value()) << "the DRC Atlas ships a contact planner's file";
  if (shipped.config().contactPlanning.has_value()) mpc_config::ToProto(*shipped.config().contactPlanning, &proto);
  const bool merged = google::protobuf::TextFormat::MergeFromString(
      R"pb(
        planner { type: "lip_miqp" dt: 0.1 num_nodes: 12 commit_time: 0.3 max_solve_time: 1000 threading: "pre_solve_hook" }
        shared { gait_limits { min_swing_duration: 0.4 max_swing_duration: 0.5 } }
        event_shift_local_search { max_time: 1000 }
      )pb",
      &proto);
  EXPECT_TRUE(merged);
  return proto;
}

/** Appends to `perturbed` the paths of the fields below `message` (at `prefix`) that perturb() changed. */
void perturb(Message& message, const std::string& prefix, std::vector<std::string>& perturbed) {
  const Descriptor& descriptor = *message.GetDescriptor();
  const Reflection& reflection = *message.GetReflection();
  for (int i = 0; i < descriptor.field_count(); ++i) {
    const FieldDescriptor& field = *descriptor.field(i);
    const std::string path = prefix.empty() ? std::string(field.name()) : absl::StrCat(prefix, ".", field.name());
    if (field.cpp_type() == FieldDescriptor::CPPTYPE_MESSAGE) {
      if (field.is_repeated()) {
        for (int index = 0; index < reflection.FieldSize(message, &field); ++index) {
          perturb(*reflection.MutableRepeatedMessage(&message, &field, index), absl::StrCat(path, "[", index, "]"), perturbed);
        }
      } else if (reflection.HasField(message, &field)) {
        perturb(*reflection.MutableMessage(&message, &field), path, perturbed);
      }
      continue;
    }
    if (field.cpp_type() == FieldDescriptor::CPPTYPE_DOUBLE && !field.is_repeated() && reflection.HasField(message, &field)) {
      const double value = reflection.GetDouble(message, &field);
      if (value != 0.0) {
        reflection.SetDouble(&message, &field, value * kScale);
        perturbed.push_back(path);
      }
      continue;
    }
    // A name list of terms (the file's top-level lists): one entry short (the last), so that a term the reload drops is
    // dropped.
    // The dynamics list is left whole: the terms of the other lists name the blocks of the planner's state they act on.
    if (prefix.empty() && field.name() != "dynamics" && field.cpp_type() == FieldDescriptor::CPPTYPE_STRING && field.is_repeated() &&
        reflection.FieldSize(message, &field) > 1) {
      reflection.RemoveLast(&message, &field);
      perturbed.push_back(absl::StrCat(path, "[last]"));
    }
  }
}

/** The leaves of the contact planner's file a reload applies are all of them: the whole file is RELOAD_HOT. */
TEST(ContactPlanningReload, EveryTunableFieldOfTheFileIsReloadHot) {
  for (const ConfigLeaf& leaf : configLeaves(*ContactPlanningProto::descriptor())) {
    if (!leaf.tunable) continue;
    EXPECT_EQ(leaf.reload, ConfigReload::kHot) << leaf.path << ": the perturbation below covers the RELOAD_HOT fields only";
  }
}

/** Everything the solver would read from the two stacks, compared. */
void expectSameReferences(AtlasReferenceStack& reloaded, AtlasReferenceStack& fresh, scalar_t time, const std::string& when) {
  SwitchedModelReferenceManager& a = reloaded.referenceManager();
  SwitchedModelReferenceManager& b = fresh.referenceManager();
  ASSERT_EQ(a.getModeSchedule().modeSequence, b.getModeSchedule().modeSequence) << when;
  ASSERT_EQ(a.getModeSchedule().eventTimes, b.getModeSchedule().eventTimes) << when;
  const TargetTrajectories& targetA = a.getTargetTrajectories();
  const TargetTrajectories& targetB = b.getTargetTrajectories();
  ASSERT_EQ(targetA.timeTrajectory, targetB.timeTrajectory) << when;
  for (size_t k = 0; k < targetA.stateTrajectory.size(); ++k) {
    EXPECT_LT((targetA.stateTrajectory[k] - targetB.stateTrajectory[k]).cwiseAbs().maxCoeff(), kTolerance) << when << ", knot " << k;
  }
  constexpr int kQueries = 40;
  constexpr scalar_t kQueryStep = 0.05;  // [s]
  for (int step = 0; step <= kQueries; ++step) {
    const scalar_t query = time + kQueryStep * step;
    EXPECT_EQ(a.getContactFlags(query), b.getContactFlags(query)) << when << ", t = " << query;
    for (size_t foot = 0; foot < kNumContacts; ++foot) {
      const std::optional<SwingFootReference> referenceA = a.getSwingFootReference(foot, query);
      const std::optional<SwingFootReference> referenceB = b.getSwingFootReference(foot, query);
      ASSERT_EQ(referenceA.has_value(), referenceB.has_value()) << when << ", foot " << foot << ", t = " << query;
      if (!referenceA.has_value() || !referenceB.has_value()) continue;
      EXPECT_LT((referenceA->position - referenceB->position).norm(), kTolerance) << when << ", foot " << foot << ", t = " << query;
      EXPECT_LT((referenceA->linearVelocity - referenceB->linearVelocity).norm(), kTolerance) << when << ", foot " << foot;
    }
  }
  ASSERT_NE(reloaded.planningReferenceManager(), nullptr);
  ASSERT_NE(fresh.planningReferenceManager(), nullptr);
  EXPECT_EQ(reloaded.planningReferenceManager()->hasActivePlan(), fresh.planningReferenceManager()->hasActivePlan()) << when;
  const feet_array_t<TargetContactPose> posesA = reloaded.planningReferenceManager()->getTargetContactPoses();
  const feet_array_t<TargetContactPose> posesB = fresh.planningReferenceManager()->getTargetContactPoses();
  for (size_t foot = 0; foot < kNumContacts; ++foot) {
    EXPECT_LT((posesA[foot].position - posesB[foot].position).norm(), kTolerance) << when << ", foot " << foot;
  }
}

TEST(ContactPlanningReloadEqualsFreshStart, OnTheDrcAtlasUnderTheMixedIntegerPlanner) {
  const ContactPlanningProto base = baseFile();
  ContactPlanningProto edited = base;
  std::vector<std::string> perturbed;
  perturb(edited, /*prefix=*/"", perturbed);
  ASSERT_GT(perturbed.size(), 10U) << "the perturbation reached almost no field: " << absl::StrJoin(perturbed, ", ");
  RecordProperty("perturbed_fields", absl::StrJoin(perturbed, " "));
  mpc_config::ContactPlanningFile baseFileStruct;
  mpc_config::ContactPlanningFile editedFileStruct;
  ASSERT_TRUE(mpc_config::FromProto(base, &baseFileStruct).ok());
  ASSERT_TRUE(mpc_config::FromProto(edited, &editedFileStruct).ok());

  AtlasReferenceStack reloaded(baseFileStruct, /*plannerType=*/"");
  AtlasReferenceStack fresh(editedFileStruct, /*plannerType=*/"");
  // As MpcParameterUpdaterModule::applyContactPlanningFile() hands a reloaded file over, before the first solve.
  const absl::StatusOr<ContactPlanningConfig> reload =
      contactPlanningConfigFromConfig(editedFileStruct, ContactPlanningValidation::kDeferUntilModelParametersApplied);
  ASSERT_TRUE(reload.ok()) << reload.status();
  ASSERT_NE(reloaded.contactPlannerModule(), nullptr);
  const absl::Status applied = reloaded.contactPlannerModule()->setConfig(*reload);
  ASSERT_TRUE(applied.ok()) << applied;

  scalar_t time = 0.0;
  vector_t state = reloaded.initialState();
  for (AtlasReferenceStack* absl_nonnull stack : {&reloaded, &fresh}) {
    stack->referenceManager().setTargetTrajectories(stack->resetTarget(time, state));
  }
  for (int solve = 0; solve < kStandingSolves + kWalkingSolves; ++solve) {
    if (solve == kStandingSolves) {
      reloaded.command(kWalkingSpeed / kCommandLimitX);
      fresh.command(kWalkingSpeed / kCommandLimitX);
    }
    reloaded.mpc().run(time, state, ModeNumber::kStance);
    fresh.mpc().run(time, state, ModeNumber::kStance);
    expectSameReferences(reloaded, fresh, time, absl::StrCat("solve ", solve));
    if (HasFatalFailure()) return;
    time += kSolvePeriod;
  }
  // Positive control: the edit matters, so the equality above is not two stacks that ignore the file.
  AtlasReferenceStack unedited(baseFileStruct, /*plannerType=*/"");
  EXPECT_NE(unedited.contactPlannerModule()->getConfig().planner.commitTime,
            reloaded.contactPlannerModule()->getConfig().planner.commitTime);
}

}  // namespace
}  // namespace ocs2::humanoid
