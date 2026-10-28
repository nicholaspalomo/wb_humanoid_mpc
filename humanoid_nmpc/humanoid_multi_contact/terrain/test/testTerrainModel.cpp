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

// TerrainModel (design section 3.2): closed-form distances, nearest points and
// normals of a box and a capsule, the smooth minimum at an inside stair corner,
// the patch frames, the scene validation, and the derivatives of a CppAD tape
// against central differences.

#include <cmath>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "Eigen/Geometry"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "cppad/cg.hpp"
#include "cppad/cppad.hpp"
#include "gtest/gtest.h"
#include "ocs2_core/automatic_differentiation/Types.h"

#include "humanoid_multi_contact/terrain/PatchFrame.h"
#include "humanoid_multi_contact/terrain/TerrainModel.h"
#include "humanoid_multi_contact/terrain/TerrainPrimitives.h"
#include "humanoid_multi_contact/terrain/terrain_scene.nproto.h"

namespace ocs2::humanoid::multi_contact {
namespace {

using Primitive = config::TerrainScene::Primitive;

constexpr double kTolerance = 1.0e-12;

Primitive boxPrimitive(const std::string& name,
                       const Eigen::Vector3d& center,
                       const Eigen::Vector3d& halfExtents,
                       const Eigen::Vector4d& orientationXyzw =
                           Eigen::Vector4d(0.0, 0.0, 0.0, 1.0)) {
  Primitive primitive;
  primitive.name = name;
  primitive.friction = 0.8;
  config::TerrainScene::Box box;
  box.center = center;
  box.half_extents = halfExtents;
  box.orientation_xyzw = orientationXyzw;
  primitive.shape = box;
  return primitive;
}

Primitive capsulePrimitive(const std::string& name,
                           const Eigen::Vector3d& start,
                           const Eigen::Vector3d& end,
                           double radius) {
  Primitive primitive;
  primitive.name = name;
  primitive.friction = 0.5;
  config::TerrainScene::Capsule capsule;
  capsule.start = start;
  capsule.end = end;
  capsule.radius = radius;
  primitive.shape = capsule;
  return primitive;
}

TerrainModel modelOf(const std::vector<Primitive>& primitives) {
  config::TerrainScene scene;
  scene.name = "test";
  scene.primitives = primitives;
  absl::StatusOr<TerrainModel> model = TerrainModel::Create(scene);
  EXPECT_TRUE(model.ok()) << model.status();
  return *std::move(model);
}

void expectQuery(const PrimitiveQuery<double>& query,
                 double distance,
                 const Eigen::Vector3d& nearest,
                 const Eigen::Vector3d& normal) {
  EXPECT_NEAR(query.distance, distance, kTolerance);
  EXPECT_LT((query.nearestPoint - nearest).norm(), kTolerance)
      << query.nearestPoint.transpose();
  EXPECT_LT((query.normal - normal).norm(), kTolerance)
      << query.normal.transpose();
}

TEST(TerrainModelTest, BoxFacesEdgesCornersAndInside) {
  const TerrainModel model = modelOf({boxPrimitive(
      "box", Eigen::Vector3d::Zero(), Eigen::Vector3d(1.0, 2.0, 3.0))});
  // Outside a face, an edge and a corner.
  expectQuery(model.primitiveQuery(/*index=*/0, Eigen::Vector3d(1.5, 0.5, 0.0)),
              /*distance=*/0.5, Eigen::Vector3d(1.0, 0.5, 0.0),
              Eigen::Vector3d::UnitX());
  expectQuery(model.primitiveQuery(/*index=*/0, Eigen::Vector3d(2.0, 3.0, 0.0)),
              std::sqrt(2.0), Eigen::Vector3d(1.0, 2.0, 0.0),
              Eigen::Vector3d(1.0, 1.0, 0.0).normalized());
  expectQuery(
      model.primitiveQuery(/*index=*/0, Eigen::Vector3d(-2.0, -3.0, 4.0)),
      std::sqrt(3.0), Eigen::Vector3d(-1.0, -2.0, 3.0),
      Eigen::Vector3d(-1.0, -1.0, 1.0).normalized());
  // Inside, nearest to the +x face and to the -z face.
  expectQuery(model.primitiveQuery(/*index=*/0, Eigen::Vector3d(0.5, 0.2, 0.0)),
              /*distance=*/-0.5, Eigen::Vector3d(1.0, 0.2, 0.0),
              Eigen::Vector3d::UnitX());
  expectQuery(
      model.primitiveQuery(/*index=*/0, Eigen::Vector3d(0.1, 0.2, -2.9)),
      /*distance=*/-0.1, Eigen::Vector3d(0.1, 0.2, -3.0),
      -Eigen::Vector3d::UnitZ());
  // On the surface.
  EXPECT_NEAR(model.signedDistance(Eigen::Vector3d(0.3, 2.0, 1.0)), 0.0,
              kTolerance);
}

TEST(TerrainModelTest, RotatedBoxIsTheRotatedQuery) {
  const Eigen::Quaterniond rotation(
      Eigen::AngleAxisd(0.4, Eigen::Vector3d(1.0, 2.0, -0.5).normalized()));
  const Eigen::Vector3d center(0.3, -0.2, 1.0);
  const TerrainModel rotated = modelOf({boxPrimitive(
      "box", center, Eigen::Vector3d(0.5, 0.25, 0.75), rotation.coeffs())});
  const TerrainModel aligned = modelOf({boxPrimitive(
      "box", Eigen::Vector3d::Zero(), Eigen::Vector3d(0.5, 0.25, 0.75))});
  for (const Eigen::Vector3d& local :
       {Eigen::Vector3d(0.9, 0.1, 0.0), Eigen::Vector3d(0.7, 0.4, -1.0),
        Eigen::Vector3d(0.1, 0.05, 0.2)}) {
    const PrimitiveQuery<double> expected =
        aligned.primitiveQuery(/*index=*/0, local);
    expectQuery(rotated.primitiveQuery(
                    /*index=*/0, Eigen::Vector3d(center + rotation * local)),
                expected.distance, center + rotation * expected.nearestPoint,
                rotation * expected.normal);
  }
}

TEST(TerrainModelTest, CapsuleSideCapInsideAndAxis) {
  const TerrainModel model = modelOf(
      {capsulePrimitive("rail", Eigen::Vector3d::Zero(),
                        Eigen::Vector3d(0.0, 0.0, 2.0), /*radius=*/0.5)});
  expectQuery(model.primitiveQuery(/*index=*/0, Eigen::Vector3d(0.0, 1.0, 1.0)),
              /*distance=*/0.5, Eigen::Vector3d(0.0, 0.5, 1.0),
              Eigen::Vector3d::UnitY());
  expectQuery(model.primitiveQuery(/*index=*/0, Eigen::Vector3d(0.0, 0.0, 3.0)),
              /*distance=*/0.5, Eigen::Vector3d(0.0, 0.0, 2.5),
              Eigen::Vector3d::UnitZ());
  expectQuery(
      model.primitiveQuery(/*index=*/0, Eigen::Vector3d(-0.2, 0.0, 0.5)),
      /*distance=*/-0.3, Eigen::Vector3d(-0.5, 0.0, 0.5),
      -Eigen::Vector3d::UnitX());
  const PrimitiveQuery<double> onAxis =
      model.primitiveQuery(/*index=*/0, Eigen::Vector3d(0.0, 0.0, 1.0));
  EXPECT_NEAR(onAxis.distance, -0.5, 1.0e-8);
  EXPECT_NEAR(onAxis.normal.norm(), 1.0, kTolerance);
  EXPECT_NEAR(onAxis.normal.z(), 0.0, kTolerance);
  // Coinciding ends: a sphere.
  const TerrainModel sphere = modelOf(
      {capsulePrimitive("ball", Eigen::Vector3d(1.0, 0.0, 0.0),
                        Eigen::Vector3d(1.0, 0.0, 0.0), /*radius=*/0.25)});
  EXPECT_NEAR(sphere.signedDistance(Eigen::Vector3d(1.0, 1.0, 0.0)), 0.75,
              kTolerance);
}

TerrainModel stairCorner() {
  return modelOf({boxPrimitive("floor", Eigen::Vector3d(0.0, 0.0, -0.5),
                               Eigen::Vector3d(10.0, 10.0, 0.5)),
                  boxPrimitive("step", Eigen::Vector3d(0.64, 0.0, 0.09),
                               Eigen::Vector3d(0.14, 0.6, 0.09))});
}

TEST(TerrainModelTest, SmoothMinimumAtAnInsideStairCorner) {
  const TerrainModel model = stairCorner();
  // 1 cm from the floor and 1 cm from the riser at x = 0.5.
  const Eigen::Vector3d corner(0.49, 0.0, 0.01);
  EXPECT_NEAR(model.signedDistance(corner),
              0.01 - kSmoothMinimumScale * std::log(2.0), kTolerance);
  EXPECT_LT(
      (model.normal(corner) - Eigen::Vector3d(-1.0, 0.0, 1.0).normalized())
          .norm(),
      1.0e-9);
  // Far from the riser, the floor alone.
  const Eigen::Vector3d open(-1.0, 0.0, 0.2);
  EXPECT_NEAR(model.signedDistance(open), 0.2, 1.0e-12);
  EXPECT_LT((model.nearestPoint(open) - Eigen::Vector3d(-1.0, 0.0, 0.0)).norm(),
            1.0e-12);
  EXPECT_NEAR(model.sphereDistance(open, /*radius=*/0.05), 0.15, 1.0e-12);
}

TEST(TerrainModelTest, OnePrimitiveIsExact) {
  const TerrainModel floor =
      modelOf({boxPrimitive("floor", Eigen::Vector3d(0.0, 0.0, -0.5),
                            Eigen::Vector3d(5.0, 5.0, 0.5))});
  for (const double height : {-0.3, 0.0, 0.01, 0.7}) {
    EXPECT_NEAR(floor.signedDistance(Eigen::Vector3d(0.2, -0.1, height)),
                height, kTolerance);
  }
}

TEST(TerrainModelTest, BoxFaceFrames) {
  const Eigen::Quaterniond rotation(
      Eigen::AngleAxisd(0.3, Eigen::Vector3d::UnitZ()));
  const TerrainModel model = modelOf(
      {boxPrimitive("box", Eigen::Vector3d(1.0, 0.0, 0.3),
                    Eigen::Vector3d(0.5, 0.4, 0.3), rotation.coeffs()),
       capsulePrimitive("rail", Eigen::Vector3d(0.0, 2.0, 1.0),
                        Eigen::Vector3d(1.0, 2.0, 1.0), /*radius=*/0.05)});
  for (int face = 0; face < kBoxFaceCount; ++face) {
    const absl::StatusOr<PatchFrame> frame =
        model.boxFaceFrame(/*index=*/0, static_cast<BoxFace>(face));
    ASSERT_TRUE(frame.ok()) << frame.status();
    EXPECT_NEAR(frame->rotation.determinant(), 1.0, kTolerance);
    EXPECT_LT((frame->rotation.transpose() * frame->rotation -
               Eigen::Matrix3d::Identity())
                  .norm(),
              kTolerance);
    // The face center is on the surface, and the normal is the gradient there.
    const PrimitiveQuery<double> query = model.primitiveQuery(
        /*index=*/0, Eigen::Vector3d(frame->origin + 0.01 * frame->normal()));
    EXPECT_NEAR(query.distance, 0.01, kTolerance);
    EXPECT_LT((query.normal - frame->normal()).norm(), kTolerance);
    // Its polygon's corners are box corners, counterclockwise about the normal.
    ASSERT_EQ(frame->polygon.size(), 4u);
    double area = 0.0;
    for (size_t i = 0; i < frame->polygon.size(); ++i) {
      const Eigen::Vector2d& a = frame->polygon[i];
      const Eigen::Vector2d& b =
          frame->polygon[(i + 1) % frame->polygon.size()];
      area += 0.5 * (a.x() * b.y() - a.y() * b.x());
      const Eigen::Vector3d corner =
          frame->origin + frame->rotation.leftCols<2>() * a;
      EXPECT_NEAR(model.primitiveQuery(/*index=*/0, corner).distance, 0.0,
                  kTolerance);
    }
    EXPECT_GT(area, 0.0);
    EXPECT_DOUBLE_EQ(frame->friction, 0.8);
  }
  EXPECT_FALSE(model.boxFaceFrame(/*index=*/1, BoxFace::kPositiveZ).ok());
  EXPECT_FALSE(model.boxFaceFrame(/*index=*/2, BoxFace::kPositiveZ).ok());

  // A point above the top face, just past an edge, rests on the top face.
  const Eigen::Vector3d top = Eigen::Vector3d(1.0, 0.0, 0.3) +
                              rotation * Eigen::Vector3d(0.501, 0.1, 0.31);
  const PatchFrame topFrame = model.patchFrameAt(top);
  EXPECT_EQ(topFrame.face, BoxFace::kPositiveZ);
  EXPECT_EQ(topFrame.primitiveIndex, 0);
  // A point beside the rail rests on its tangent plane.
  const PatchFrame railFrame =
      model.patchFrameAt(Eigen::Vector3d(0.5, 2.0, 1.2));
  EXPECT_FALSE(railFrame.face.has_value());
  EXPECT_EQ(railFrame.primitiveIndex, 1);
  EXPECT_LT((railFrame.normal() - Eigen::Vector3d::UnitZ()).norm(), kTolerance);
  EXPECT_LT((railFrame.origin - Eigen::Vector3d(0.5, 2.0, 1.05)).norm(),
            kTolerance);
  EXPECT_NEAR(railFrame.rotation.determinant(), 1.0, kTolerance);
  EXPECT_DOUBLE_EQ(railFrame.friction, 0.5);
}

void expectRefused(std::vector<Primitive> primitives,
                   absl::string_view message) {
  config::TerrainScene scene;
  scene.name = "bad";
  scene.primitives = std::move(primitives);
  const absl::StatusOr<TerrainModel> model = TerrainModel::Create(scene);
  ASSERT_FALSE(model.ok());
  EXPECT_EQ(model.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(model.status().message(), message))
      << model.status();
}

TEST(TerrainModelTest, RefusesInvalidScenes) {
  expectRefused(/*primitives=*/{}, "lists no primitive");
  const Primitive box =
      boxPrimitive("a", Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones());
  expectRefused({box, box}, "listed twice");
  Primitive unnamed = box;
  unnamed.name = "";
  expectRefused({unnamed}, "has no name");
  Primitive frictionless = box;
  frictionless.friction = 0.0;
  expectRefused({frictionless}, "friction");
  expectRefused(
      {boxPrimitive("a", Eigen::Vector3d::Zero(), Eigen::Vector3d::Ones(),
                    Eigen::Vector4d(0.0, 0.0, 0.0, 2.0))},
      "unit quaternion");
  expectRefused({boxPrimitive("a", Eigen::Vector3d::Zero(),
                              Eigen::Vector3d(1.0, 0.0, 1.0))},
                "half_extents must be positive");
  Primitive shapeless = box;
  shapeless.shape = std::monostate();
  expectRefused({shapeless}, "neither box nor capsule");
  expectRefused({capsulePrimitive("c", Eigen::Vector3d::Zero(),
                                  Eigen::Vector3d::Ones(), /*radius=*/-0.1)},
                "radius");
}

/** A scene with every kind of region: a floor, a stair, a rotated box and a
 * capsule. */
TerrainModel mixedScene() {
  const Eigen::Quaterniond rotation(
      Eigen::AngleAxisd(0.5, Eigen::Vector3d(0.2, 1.0, 0.3).normalized()));
  return modelOf(
      {boxPrimitive("floor", Eigen::Vector3d(0.0, 0.0, -0.5),
                    Eigen::Vector3d(10.0, 10.0, 0.5)),
       boxPrimitive("step", Eigen::Vector3d(0.64, 0.0, 0.09),
                    Eigen::Vector3d(0.14, 0.6, 0.09)),
       boxPrimitive("crate", Eigen::Vector3d(-1.0, 1.0, 0.6),
                    Eigen::Vector3d(0.3, 0.2, 0.25), rotation.coeffs()),
       capsulePrimitive("rail", Eigen::Vector3d(0.5, 0.7, 0.9),
                        Eigen::Vector3d(1.4, 0.7, 1.6), /*radius=*/0.025)});
}

/** The points where the derivatives are checked: a face, an edge, a corner, the
 * inside of a box, a stair's inside corner, a capsule and a blend. */
std::vector<Eigen::Vector3d> probePoints() {
  return {Eigen::Vector3d(0.3, -0.2, 0.15),   Eigen::Vector3d(0.81, 0.65, 0.22),
          Eigen::Vector3d(0.85, 0.7, 0.25),   Eigen::Vector3d(0.6, 0.1, 0.12),
          Eigen::Vector3d(0.495, 0.0, 0.004), Eigen::Vector3d(1.0, 0.75, 1.3),
          Eigen::Vector3d(-0.8, 1.1, 0.95),   Eigen::Vector3d(-1.0, 1.0, 0.62)};
}

TEST(TerrainModelTest, CppAdJacobianMatchesCentralDifferences) {
  using AdScalar = CppAD::AD<double>;
  const TerrainModel model = mixedScene();
  // Recorded once, far from every probe point: every branch is a conditional
  // expression, so the tape is valid in every region.
  std::vector<AdScalar> x = {AdScalar(2.0), AdScalar(-2.0), AdScalar(3.0)};
  CppAD::Independent(x);
  const Vector3<AdScalar> point(x[0], x[1], x[2]);
  std::vector<AdScalar> y = {model.signedDistance(point),
                             model.sphereDistance(point, AdScalar(0.03))};
  CppAD::ADFun<double> tape(x, y);

  constexpr double kStep = 1.0e-6;
  for (const Eigen::Vector3d& probe : probePoints()) {
    const std::vector<double> at = {probe.x(), probe.y(), probe.z()};
    const std::vector<double> value = tape.Forward(/*q=*/0, at);
    EXPECT_NEAR(value[0], model.signedDistance(probe), 1.0e-12);
    EXPECT_NEAR(value[1], model.sphereDistance(probe, /*radius=*/0.03),
                1.0e-12);
    const std::vector<double> jacobian = tape.Jacobian(at);
    for (int axis = 0; axis < kPositionDim; ++axis) {
      const Eigen::Vector3d step = kStep * Eigen::Vector3d::Unit(axis);
      const double difference =
          (model.signedDistance(Eigen::Vector3d(probe + step)) -
           model.signedDistance(Eigen::Vector3d(probe - step))) /
          (2.0 * kStep);
      EXPECT_NEAR(jacobian[axis], difference, 1.0e-6)
          << "at " << probe.transpose() << ", axis " << axis;
      EXPECT_NEAR(jacobian[kPositionDim + axis], difference, 1.0e-6);
    }
    EXPECT_TRUE(std::isfinite(jacobian[0]));
  }
}

TEST(TerrainModelTest, GradientIsTheNormalWhereOnePrimitiveIsNearest) {
  using AdScalar = CppAD::AD<double>;
  const TerrainModel model = mixedScene();
  std::vector<AdScalar> x = {AdScalar(0.0), AdScalar(0.0), AdScalar(1.0)};
  CppAD::Independent(x);
  std::vector<AdScalar> y = {
      model.signedDistance(Vector3<AdScalar>(x[0], x[1], x[2]))};
  CppAD::ADFun<double> tape(x, y);
  for (const Eigen::Vector3d& probe :
       {Eigen::Vector3d(0.0, -0.2, 0.15), Eigen::Vector3d(0.85, 0.7, 0.25),
        Eigen::Vector3d(0.6, 0.1, 0.12)}) {
    const std::vector<double> jacobian =
        tape.Jacobian(std::vector<double>{probe.x(), probe.y(), probe.z()});
    const Eigen::Vector3d gradient(jacobian[0], jacobian[1], jacobian[2]);
    EXPECT_LT((gradient - model.normal(probe)).norm(), 1.0e-9)
        << probe.transpose();
  }
}

TEST(TerrainModelTest, TapeRunsOnCodeGenVariables) {
  const TerrainModel model = mixedScene();
  ad_vector_t x = ad_vector_t::Ones(kPositionDim);
  CppAD::Independent(x);
  ad_vector_t y(1);
  y(0) = model.signedDistance(Vector3<ad_scalar_t>(x));
  CppAD::ADFun<ad_base_t> tape(x, y);
  CppAD::cg::CodeHandler<scalar_t> handler;
  CppAD::vector<ad_base_t> variables(kPositionDim);
  handler.makeVariables(variables);
  // Code generation throws on a comparison between variables, which a plain
  // `if` on a tape would need.
  EXPECT_NO_THROW(tape.Forward(/*q=*/0, variables));
}

}  // namespace
}  // namespace ocs2::humanoid::multi_contact
