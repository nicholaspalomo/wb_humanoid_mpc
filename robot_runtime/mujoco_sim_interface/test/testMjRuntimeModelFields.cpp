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

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "gtest/gtest.h"
#include "mujoco/mujoco.h"

#include "mujoco_sim_interface/MjRuntimeModelFields.h"
#include "mujoco_sim_interface/MujocoUtils.h"
#include "robot_runtime/robot_realtime/test/AllocationCounter.h"

/*
 * MjRuntimeModelFields on a scene with what the simulator changes while it runs: a damped robot joint, a gantry weld and
 * a parked ball. A start-up copy of the model with the fields written into it is the model again, byte for byte; nothing
 * is copied between models of other sizes; capturing, applying and handing the fields over allocate nothing; and an
 * MjState carries them through its copies and moves.
 */

namespace robot::mujoco_sim_interface {
namespace {

constexpr char kScene[] = R"(
<mujoco>
  <worldbody>
    <geom type="plane" size="2 2 0.1"/>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="2"/>
      <body name="arm">
        <joint type="hinge" axis="0 1 0" damping="0.5"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.3"/>
      </body>
    </body>
    <body name="ball" pos="1 0 1" gravcomp="1">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="0.5" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
  <equality>
    <weld name="gantry" body1="world" body2="base"/>
  </equality>
</mujoco>
)";

constexpr char kOtherScene[] = R"(
<mujoco>
  <worldbody>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
)";

MjModelPtr compile(const char* absl_nonnull xml) {
  char error[1000] = "";
  const MjSpecPtr spec(mj_parseXMLString(xml, /*vfs=*/nullptr, error, sizeof(error)));
  EXPECT_NE(spec, nullptr) << error;
  if (spec == nullptr) return nullptr;
  MjModelPtr model(mj_compile(spec.get(), /*vfs=*/nullptr));
  EXPECT_NE(model, nullptr);
  return model;
}

MjModelPtr copyOf(const mjModel* absl_nonnull model) {
  return MjModelPtr(mj_copyModel(/*dest=*/nullptr, model));
}

/** True when the arrays of the two models, which have one layout, hold the same bytes. */
bool sameArrays(const mjModel* absl_nonnull first, const mjModel* absl_nonnull second) {
  return first->nbuffer == second->nbuffer && std::memcmp(first->buffer, second->buffer, static_cast<size_t>(first->nbuffer)) == 0;
}

/** Changes `model` as the simulator does while it runs: the ragdoll damping, the gantry's weld, a throw of the ball. */
void changeAsTheSimulatorDoes(mjModel* absl_nonnull model) {
  model->dof_damping[6] = 8.0;  // the arm's hinge, after the base's six
  const int weld = mj_name2id(model, mjOBJ_EQUALITY, "gantry");
  model->eq_data[mjNEQDATA * weld + 5] = 0.25;  // the relpose's height
  model->eq_data[mjNEQDATA * weld + 9] = 0.5;   // and the z of its quaternion
  const int ball = mj_name2id(model, mjOBJ_BODY, "ball");
  const int ballGeom = model->body_geomadr[ball];
  const int ballDof = model->jnt_dofadr[model->body_jntadr[ball]];
  model->geom_contype[ballGeom] = 0;
  model->geom_conaffinity[ballGeom] = 0;
  model->body_contype[ball] = 0;
  model->body_conaffinity[ball] = 0;
  model->body_mass[ball] = 1.5;
  for (int axis = 0; axis < 3; ++axis) model->body_inertia[3 * ball + axis] = 0.006;
  model->body_invweight0[2 * ball] = 1.0 / 1.5;
  model->body_invweight0[2 * ball + 1] = 1.0 / 0.006;
  model->body_subtreemass[ball] = 1.5;
  model->body_subtreemass[0] += 1.0;
  model->body_gravcomp[ball] = 0.0;
  for (int dof = ballDof; dof < ballDof + 3; ++dof) {
    model->dof_invweight0[dof] = 1.0 / 1.5;
    model->dof_M0[dof] = 1.5;
  }
}

TEST(MjRuntimeModelFields, AStartUpCopyWithTheFieldsWrittenIntoItIsTheModelAgain) {
  const MjModelPtr model = compile(kScene);
  ASSERT_NE(model, nullptr);
  const MjModelPtr startUp = copyOf(model.get());
  ASSERT_NE(startUp, nullptr);
  MjRuntimeModelFields fields(startUp.get());
  ASSERT_TRUE(fields.fits(model.get()));

  changeAsTheSimulatorDoes(model.get());
  ASSERT_FALSE(sameArrays(startUp.get(), model.get())) << "the changes reach nothing the comparison sees";
  fields.captureFrom(model.get());
  fields.applyTo(startUp.get());
  EXPECT_TRUE(sameArrays(startUp.get(), model.get())) << "a run-time change that MjRuntimeModelFields does not carry";
}

TEST(MjRuntimeModelFields, TheFieldsOfAModelHoldItsValuesAtConstruction) {
  const MjModelPtr model = compile(kScene);
  ASSERT_NE(model, nullptr);
  changeAsTheSimulatorDoes(model.get());
  const MjRuntimeModelFields fields(model.get());
  const MjModelPtr target = compile(kScene);
  ASSERT_NE(target, nullptr);
  fields.applyTo(target.get());
  EXPECT_TRUE(sameArrays(target.get(), model.get()));
}

TEST(MjRuntimeModelFields, NothingIsCopiedBetweenModelsOfOtherSizes) {
  const MjModelPtr model = compile(kScene);
  const MjModelPtr other = compile(kOtherScene);
  ASSERT_NE(model, nullptr);
  ASSERT_NE(other, nullptr);
  MjRuntimeModelFields fields(model.get());
  EXPECT_FALSE(fields.fits(other.get()));

  const MjModelPtr otherBefore = copyOf(other.get());
  ASSERT_NE(otherBefore, nullptr);
  fields.applyTo(other.get());
  EXPECT_TRUE(sameArrays(other.get(), otherBefore.get())) << "applyTo wrote into a model of other sizes";

  other->body_mass[1] = 9.0;
  fields.captureFrom(other.get());
  const MjModelPtr target = compile(kScene);
  ASSERT_NE(target, nullptr);
  fields.applyTo(target.get());
  EXPECT_TRUE(sameArrays(target.get(), model.get())) << "captureFrom read a model of other sizes";
}

TEST(MjRuntimeModelFields, EmptyFieldsFitNoModel) {
  const MjModelPtr model = compile(kScene);
  ASSERT_NE(model, nullptr);
  const MjRuntimeModelFields empty;
  EXPECT_FALSE(empty.fits(model.get()));
  const MjModelPtr before = copyOf(model.get());
  ASSERT_NE(before, nullptr);
  empty.applyTo(model.get());
  EXPECT_TRUE(sameArrays(model.get(), before.get()));
}

TEST(MjRuntimeModelFields, CapturingApplyingAndHandingOverAllocateNothing) {
  const MjModelPtr model = compile(kScene);
  ASSERT_NE(model, nullptr);
  const MjModelPtr startUp = copyOf(model.get());
  ASSERT_NE(startUp, nullptr);
  MjRuntimeModelFields published(model.get());
  MjRuntimeModelFields received(model.get());
  changeAsTheSimulatorDoes(model.get());

  const size_t before = realtime::heapAllocationCountOnThisThread();
  published.captureFrom(model.get());  // the physics thread, at every snapshot
  received = published;                // MujocoSimInterface::readLatestMjState
  received.applyTo(startUp.get());     // the render thread, at every frame
  const size_t allocations = realtime::heapAllocationCountOnThisThread() - before;
  EXPECT_EQ(allocations, 0u);
  EXPECT_TRUE(sameArrays(startUp.get(), model.get()));
}

TEST(MjRuntimeModelFields, AnMjStateCarriesItsFieldsThroughCopiesAndMoves) {
  const MjModelPtr model = compile(kScene);
  ASSERT_NE(model, nullptr);
  MjState original(model.get());
  ASSERT_TRUE(original.modelFields.fits(model.get()));
  changeAsTheSimulatorDoes(model.get());
  original.modelFields.captureFrom(model.get());

  const MjState copied(original);
  MjState assigned(model.get());
  assigned = original;
  for (const MjState* absl_nonnull state : std::vector<const MjState* absl_nonnull>{&copied, &assigned}) {
    const MjModelPtr target = compile(kScene);
    ASSERT_NE(target, nullptr);
    state->modelFields.applyTo(target.get());
    EXPECT_TRUE(sameArrays(target.get(), model.get()));
  }

  MjState moved(std::move(original));
  EXPECT_TRUE(moved.modelFields.fits(model.get()));
  EXPECT_FALSE(original.modelFields.fits(model.get()));  // NOLINT(bugprone-use-after-move): the moved-from state is what is tested.
  MjState moveAssigned(model.get());
  moveAssigned = std::move(moved);
  EXPECT_TRUE(moveAssigned.modelFields.fits(model.get()));
  EXPECT_FALSE(moved.modelFields.fits(model.get()));  // NOLINT(bugprone-use-after-move): the moved-from state is what is tested.
}

}  // namespace
}  // namespace robot::mujoco_sim_interface
