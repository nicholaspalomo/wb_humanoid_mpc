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

// The contacts' surface heights the robot process gives the state estimator:
// the flat ground sets every contact; a multi-contact plan's patches give a
// contact on a seat, a box top or a tilted ramp the patch's plane height at its
// x and y, and none on a wall, far from every patch or without an estimated
// position; the selection by the task file's contact_schedule_source; and
// fill() allocates nothing.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/Types.h"

#include "legged_common_mpc/common/ModelSettings.h"
#include "legged_common_mpc/common/Types.h"
#include "legged_common_mpc/config/ConfigFiles.h"
#include "legged_common_mpc_app/robot/ContactSurfaceHeightSource.h"
#include "legged_common_mpc_app/robot/FlatGroundSurfaceHeights.h"
#include "legged_common_mpc_app/robot/PlannedSurfaceHeights.h"
#include "legged_common_mpc_app/robot/RobotProcessSettings.h"
#include "legged_common_mpc_app/robot/RobotSetUpSteps.h"
#include "legged_mpc_config/task_file.nproto.h"
#include "legged_multi_contact/plan/multi_contact_plan.nproto.h"
#include "legged_state_estimator/ContactEstimationInputs.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

namespace ocs2::legged {
namespace {

constexpr char kG1Task[] =
    "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto";
constexpr char kG1Urdf[] =
    "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
constexpr char kAtlasMultiContactTask[] =
    "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/multi_contact/"
    "task.textproto";
constexpr char kAtlasUrdf[] =
    "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";
constexpr char kLeftFoot[] = "foot_l_contact";
constexpr char kRightFoot[] = "foot_r_contact";

ModelSettings modelSettingsOf(const mpc_config::TaskFile& task,
                              absl::string_view urdfFile) {
  return ModelSettings::Create(task, urdfFile, "surface_height_source_test_",
                               /*verbose=*/false)
      .value();
}

using ContactPhase = multi_contact::config::MultiContactPlan::ContactPhase;

ContactPhase phaseOn(absl::string_view contact,
                     const vector3_t& posSurfaceRtWorldEwrtWorld,
                     const vector3_t& normalSurfaceEwrtWorld) {
  ContactPhase phase;
  phase.contact = std::string(contact);
  phase.des_pose_pad_rt_world = vector_t(6);
  phase.des_pose_pad_rt_world << posSurfaceRtWorldEwrtWorld,
      normalSurfaceEwrtWorld;
  return phase;
}

/** [rad] A ramp the height row accepts, and one too steep for it. */
constexpr scalar_t kRampTilt = 0.2;
constexpr scalar_t kSteepTilt = 0.8;

/** The centers of the patches of seatAndWallPlan(). */
vector3_t seatCenter() {
  return vector3_t(0.3, 0.1, 0.45);
}
vector3_t wallCenter() {
  return vector3_t(0.5, -0.1, 0.9);
}
vector3_t rampCenter() {
  return vector3_t(1.0, -0.1, 0.2);
}
vector3_t steepRampCenter() {
  return vector3_t(2.0, -0.1, 0.2);
}

/**
 * The G1's feet on a plan: the left foot on a seat (twice, the same patch)
 * and the floor, the right one on a wall, a ramp and a steep ramp, and a hand
 * the G1 does not have.
 */
multi_contact::config::MultiContactPlan seatAndWallPlan() {
  multi_contact::config::MultiContactPlan plan;
  plan.phases.push_back(phaseOn(kLeftFoot, seatCenter(), vector3_t::UnitZ()));
  plan.phases.push_back(phaseOn(kLeftFoot, seatCenter(), vector3_t::UnitZ()));
  plan.phases.push_back(
      phaseOn(kLeftFoot, vector3_t(-1.0, 0.1, 0.0), vector3_t::UnitZ()));
  plan.phases.push_back(phaseOn(kRightFoot, wallCenter(), -vector3_t::UnitX()));
  plan.phases.push_back(
      phaseOn(kRightFoot, rampCenter(),
              vector3_t(std::sin(kRampTilt), 0.0, std::cos(kRampTilt))));
  plan.phases.push_back(
      phaseOn(kRightFoot, steepRampCenter(),
              vector3_t(std::sin(kSteepTilt), 0.0, std::cos(kSteepTilt))));
  plan.phases.push_back(
      phaseOn("hand_l_contact", vector3_t(0.0, 0.0, 1.0), vector3_t::UnitZ()));
  return plan;
}

TEST(ContactSurfaceHeightSourceTest, TheFlatGroundSetsEveryContact) {
  const FlatGroundSurfaceHeights flat(/*posSurfaceRtWorldEwrtWorldZ=*/0.07);
  estimation::ContactEstimationInputs inputs(/*numContacts=*/3);
  inputs.clearNomPosContactRtWorldEwrtWorldZ(1);
  flat.fill({}, inputs);
  for (size_t contact = 0; contact < inputs.numContacts(); ++contact) {
    ASSERT_TRUE(inputs.hasNomPosContactRtWorldEwrtWorldZ(contact));
    EXPECT_EQ(inputs.nomPosContactRtWorldEwrtWorldZ(contact), 0.07);
  }
}

TEST(ContactSurfaceHeightSourceTest,
     APlannedPatchGivesItsPlaneHeightAndAWallOrAFarContactNone) {
  const ModelSettings modelSettings =
      modelSettingsOf(loadTaskFile(kG1Task).value(), kG1Urdf);
  ASSERT_EQ(modelSettings.contactNames,
            (std::vector<std::string>{kLeftFoot, kRightFoot}));
  absl::StatusOr<std::unique_ptr<PlannedSurfaceHeights>> created =
      PlannedSurfaceHeights::Create(seatAndWallPlan(), modelSettings);
  ASSERT_TRUE(created.ok()) << created.status();
  const PlannedSurfaceHeights& planned = **created;
  EXPECT_EQ(planned.numPatches(0), 2u) << "the seat once, and the floor";
  EXPECT_EQ(planned.numPatches(1), 3u);
  estimation::ContactEstimationInputs inputs(modelSettings.numContacts());

  // The left foot on the seat, 2 cm off its center; the right one on the wall.
  std::vector<vector3_t> estPosContactsRtWorldEwrtWorld = {
      seatCenter() + vector3_t(0.02, 0.02, 0.01),
      wallCenter() + vector3_t(0.01, 0.0, 0.03)};
  planned.fill(estPosContactsRtWorldEwrtWorld, inputs);
  ASSERT_TRUE(inputs.hasNomPosContactRtWorldEwrtWorldZ(0));
  EXPECT_DOUBLE_EQ(inputs.nomPosContactRtWorldEwrtWorldZ(0), seatCenter().z());
  EXPECT_FALSE(inputs.hasNomPosContactRtWorldEwrtWorldZ(1))
      << "a foot against a wall has no height row";

  // The right foot on the ramp, 5 cm uphill of its center: the ramp's plane
  // there. The left foot far from every patch of it.
  estPosContactsRtWorldEwrtWorld[0] = vector3_t(3.0, 3.0, 0.45);
  estPosContactsRtWorldEwrtWorld[1] =
      rampCenter() + vector3_t(0.05, 0.02, -0.01);
  planned.fill(estPosContactsRtWorldEwrtWorld, inputs);
  EXPECT_FALSE(inputs.hasNomPosContactRtWorldEwrtWorldZ(0));
  ASSERT_TRUE(inputs.hasNomPosContactRtWorldEwrtWorldZ(1));
  EXPECT_NEAR(inputs.nomPosContactRtWorldEwrtWorldZ(1),
              rampCenter().z() - 0.05 * std::tan(kRampTilt), 1.0e-12);

  // A ramp steeper than kMaxSurfaceTilt gives none.
  estPosContactsRtWorldEwrtWorld[1] = steepRampCenter();
  planned.fill(estPosContactsRtWorldEwrtWorld, inputs);
  EXPECT_FALSE(inputs.hasNomPosContactRtWorldEwrtWorldZ(1));

  // Without estimated positions (the ground truth keeps none), no contact
  // has a surface.
  inputs.setAllNomPosContactsRtWorldEwrtWorldZ(0.0);
  planned.fill({}, inputs);
  EXPECT_FALSE(inputs.hasNomPosContactRtWorldEwrtWorldZ(0));
  EXPECT_FALSE(inputs.hasNomPosContactRtWorldEwrtWorldZ(1));
}

TEST(ContactSurfaceHeightSourceTest, RefusesAPhaseWithoutAPadPose) {
  const ModelSettings modelSettings =
      modelSettingsOf(loadTaskFile(kG1Task).value(), kG1Urdf);
  multi_contact::config::MultiContactPlan plan;
  plan.phases.push_back(
      phaseOn(kRightFoot, vector3_t::Zero(), vector3_t(0.0, 0.0, 2.0)));
  const absl::StatusOr<std::unique_ptr<PlannedSurfaceHeights>> notUnit =
      PlannedSurfaceHeights::Create(plan, modelSettings);
  EXPECT_EQ(notUnit.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(notUnit.status().message(),
                                "phase 0 of 'foot_r_contact'"))
      << notUnit.status();
  plan.phases[0].des_pose_pad_rt_world = vector_t::Zero(3);
  EXPECT_EQ(PlannedSurfaceHeights::Create(plan, modelSettings).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ContactSurfaceHeightSourceTest, TheTaskFileSelectsTheSource) {
  RobotProcessSettings settings;
  settings.kalmanStateEstimator.nomPosContactRtWorldEwrtWorldZ = 0.07;
  // Walking: the flat ground at the configured height.
  const mpc_config::TaskFile g1Task = loadTaskFile(kG1Task).value();
  const ModelSettings g1 = modelSettingsOf(g1Task, kG1Urdf);
  absl::StatusOr<std::unique_ptr<const ContactSurfaceHeightSource>> flat =
      contactSurfaceHeightSource(g1Task, g1, settings, /*plan=*/nullptr);
  ASSERT_TRUE(flat.ok()) << flat.status();
  estimation::ContactEstimationInputs inputs(g1.numContacts());
  inputs.clearNomPosContactRtWorldEwrtWorldZ(0);
  (*flat)->fill({}, inputs);
  ASSERT_TRUE(inputs.hasNomPosContactRtWorldEwrtWorldZ(0));
  EXPECT_EQ(inputs.nomPosContactRtWorldEwrtWorldZ(0), 0.07);

  // A multi-contact task file: the plan's patches, and a refusal without one.
  const mpc_config::TaskFile atlasTask =
      loadTaskFile(kAtlasMultiContactTask).value();
  const ModelSettings atlas = modelSettingsOf(atlasTask, kAtlasUrdf);
  const absl::StatusOr<std::unique_ptr<const ContactSurfaceHeightSource>>
      noPlan = contactSurfaceHeightSource(atlasTask, atlas, settings,
                                          /*plan=*/nullptr);
  EXPECT_EQ(noPlan.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(
      absl::StrContains(noPlan.status().message(), "multi_contact_plan"))
      << noPlan.status();
  multi_contact::config::MultiContactPlan plan;
  plan.phases.push_back(
      phaseOn("seat_contact", seatCenter(), vector3_t::UnitZ()));
  absl::StatusOr<std::unique_ptr<const ContactSurfaceHeightSource>> planned =
      contactSurfaceHeightSource(atlasTask, atlas, settings, &plan);
  ASSERT_TRUE(planned.ok()) << planned.status();
  estimation::ContactEstimationInputs atlasInputs(atlas.numContacts());
  std::vector<vector3_t> estPosContactsRtWorldEwrtWorld(
      atlas.numContacts(), vector3_t(5.0, 5.0, 0.0));
  const size_t seat =
      static_cast<size_t>(std::find(atlas.contactNames.begin(),
                                    atlas.contactNames.end(), "seat_contact") -
                          atlas.contactNames.begin());
  ASSERT_LT(seat, atlas.numContacts());
  estPosContactsRtWorldEwrtWorld[seat] = seatCenter();
  (*planned)->fill(estPosContactsRtWorldEwrtWorld, atlasInputs);
  for (size_t contact = 0; contact < atlas.numContacts(); ++contact) {
    EXPECT_EQ(atlasInputs.hasNomPosContactRtWorldEwrtWorldZ(contact),
              contact == seat)
        << atlas.contactNames[contact];
  }
  EXPECT_EQ(atlasInputs.nomPosContactRtWorldEwrtWorldZ(seat), seatCenter().z());
}

TEST(ContactSurfaceHeightSourceTest, FillAllocatesNothing) {
  const ModelSettings modelSettings =
      modelSettingsOf(loadTaskFile(kG1Task).value(), kG1Urdf);
  const std::unique_ptr<PlannedSurfaceHeights> planned =
      PlannedSurfaceHeights::Create(seatAndWallPlan(), modelSettings).value();
  const FlatGroundSurfaceHeights flat(/*posSurfaceRtWorldEwrtWorldZ=*/0.0);
  estimation::ContactEstimationInputs inputs(modelSettings.numContacts());
  const std::vector<vector3_t> onSurfaces = {seatCenter(), rampCenter()};
  const std::vector<vector3_t> offSurfaces = {vector3_t(3.0, 3.0, 0.0),
                                              wallCenter()};
  const size_t before = robot::realtime::heapAllocationCountOnThisThread();
  for (int cycle = 0; cycle < 100; ++cycle) {
    planned->fill(cycle % 2 == 0 ? onSurfaces : offSurfaces, inputs);
    planned->fill({}, inputs);
    flat.fill(onSurfaces, inputs);
  }
  EXPECT_EQ(robot::realtime::heapAllocationCountOnThisThread() - before, 0u);
}

}  // namespace
}  // namespace ocs2::legged
