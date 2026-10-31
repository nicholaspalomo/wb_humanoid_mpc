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

#include "pinocchio/fwd.hpp"

#include "humanoid_nmpc/humanoid_common_mpc/test/equivalence/RootJointModelPair.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "gtest/gtest.h"
#include "pinocchio/multibody/joint/joint-composite.hpp"
#include "pinocchio/multibody/joint/joint-spherical-ZYX.hpp"
#include "pinocchio/multibody/joint/joint-spherical.hpp"
#include "pinocchio/multibody/joint/joint-translation.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "urdf_parser/urdf_parser.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/config/weights/StateInputLayout.h"
#include "humanoid_common_mpc/orientation/EulerBoundary.h"
#include "humanoid_mpc_validation/chart/EulerChart.h"
#include "robot_core/ResourcePaths.h"

namespace ocs2::humanoid {
namespace {

/** Where Pinocchio's composite root stores the rotation: after the
 * translation's 3 coordinates, in q and in v. */
constexpr Eigen::Index kRootRotationStart = 3;

// LINT.IfChange(equivalence_robots)
constexpr EquivalenceRobot kEquivalenceRobots[] = {
    {.name = "drc_atlas_centroidal_mpc",
     .taskFile = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/"
                 "task.textproto",
     .urdfFile =
         "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf"},
    {.name = "engineai_sa01_centroidal_mpc",
     .taskFile = "robot_models/engineai_sa01/engineai_sa01_centroidal_mpc/"
                 "config/mpc/task.textproto",
     .urdfFile = "robot_models/engineai_sa01/engineai_sa01_description/urdf/"
                 "zq_sa01.urdf"},
    {.name = "g1_centroidal_mpc",
     .taskFile =
         "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"},
    {.name = "g1_wb_mpc",
     .taskFile = "robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto",
     .urdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf"},
    {.name = "unitree_r1_centroidal_mpc",
     .taskFile = "robot_models/unitree_r1/unitree_r1_centroidal_mpc/config/mpc/"
                 "task.textproto",
     .urdfFile = "robot_models/unitree_r1/unitree_r1_description/urdf/R1.urdf"},
};
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/test/equivalence/BUILD.bazel:equivalence_robot_data)

/** The root joint of the quaternion formulation (design decision D1). */
pinocchio::JointModelComposite quaternionRoot() {
  pinocchio::JointModelComposite root(2);
  root.addJoint(pinocchio::JointModelTranslation());
  root.addJoint(pinocchio::JointModelSpherical());
  return root;
}

/** The Euler root joint the quaternion root is compared with:
 * Composite(Translation, SphericalZYX). */
pinocchio::JointModelComposite eulerRoot() {
  pinocchio::JointModelComposite root(2);
  root.addJoint(pinocchio::JointModelTranslation());
  // NOLINTNEXTLINE(spherical-zyx): the Euler root of the oracle the quaternion root is compared with
  root.addJoint(pinocchio::JointModelSphericalZYX());
  return root;
}

/** The URDF with every joint not in `keptJointNames` fixed: the reduction
 * loadCustomPinocchioInterface() applies. */
urdf::ModelInterfaceSharedPtr reducedUrdf(
    const urdf::ModelInterface& urdfTree,
    const std::vector<std::string>& keptJointNames) {
  urdf::ModelInterfaceSharedPtr reduced =
      std::make_shared<urdf::ModelInterface>(urdfTree);
  for (std::pair<const std::string, std::shared_ptr<urdf::Joint>>& joint :
       reduced->joints_) {
    if (std::find(keptJointNames.begin(), keptJointNames.end(), joint.first) ==
        keptJointNames.end()) {
      joint.second->type = urdf::Joint::FIXED;
    }
  }
  return reduced;
}

/** OK when the two models differ in their root joint only: same joints, bodies
 * and frames, in the same order. */
absl::Status checkSameRobot(const pinocchio::Model& quaternionModel,
                            const pinocchio::Model& eulerModel,
                            const ModelSettings& settings) {
  if (quaternionModel.njoints != eulerModel.njoints ||
      quaternionModel.nframes != eulerModel.nframes ||
      quaternionModel.nv != eulerModel.nv ||
      quaternionModel.nq != eulerModel.nq + 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the two models of ", settings.robotName,
        " differ beyond the root: njoints ", quaternionModel.njoints, " / ",
        eulerModel.njoints, ", nframes ", quaternionModel.nframes, " / ",
        eulerModel.nframes, ", nq ", quaternionModel.nq, " / ", eulerModel.nq,
        ", nv ", quaternionModel.nv, " / ", eulerModel.nv));
  }
  if (static_cast<size_t>(eulerModel.nv - 6) !=
      settings.mpcModelJointNames.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the models of ", settings.robotName, " have ", eulerModel.nv - 6,
        " joints, the task file keeps ", settings.mpcModelJointNames.size()));
  }
  for (int joint = 0; joint < quaternionModel.njoints; ++joint) {
    if (quaternionModel.names[joint] != eulerModel.names[joint]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "joint ", joint, " of ", settings.robotName, ": ",
          quaternionModel.names[joint], " / ", eulerModel.names[joint]));
    }
  }
  for (int frame = 0; frame < quaternionModel.nframes; ++frame) {
    if (quaternionModel.frames[frame].name != eulerModel.frames[frame].name) {
      return absl::InvalidArgumentError(
          absl::StrCat("frame ", frame, " of ", settings.robotName, ": ",
                       quaternionModel.frames[frame].name, " / ",
                       eulerModel.frames[frame].name));
    }
  }
  return absl::OkStatus();
}

/** OK when the whole-body Euler layout is [p_W, angles, q_j, pd_W, rates, qd_j]
 * of the Euler model's sizes. */
absl::Status checkPinocchioStateLayout(
    const validation::EulerStateLayout& layout,
    const pinocchio::Model& eulerModel) {
  const bool anglesAtTheRoot =
      layout.segments.size() == 6 && layout.segments[0].dim == 3 &&
      layout.segments[1].kind ==
          validation::EulerStateSegmentKind::kEulerZyxAngles &&
      layout.segments[4].kind ==
          validation::EulerStateSegmentKind::kEulerZyxRates;
  if (!anglesAtTheRoot ||
      validation::getEulerStateDim(layout) !=
          static_cast<size_t>(eulerModel.nq + eulerModel.nv)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "the whole-body Euler layout of ", validation::getEulerStateDim(layout),
        " rows is not [q_e; v_e] of the Euler model (nq ", eulerModel.nq,
        ", nv ", eulerModel.nv, ")"));
  }
  return absl::OkStatus();
}

/** [top; bottom]. */
vector_t stacked(const vector_t& top, const vector_t& bottom) {
  vector_t result(top.size() + bottom.size());
  result << top, bottom;
  return result;
}

scalar_t uniformScalar(scalar_t low, scalar_t high, std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> distribution(low, high);
  return distribution(generator);
}

::testing::AssertionResult isClose(const matrix_t& actual,
                                   const matrix_t& expected,
                                   scalar_t bound,
                                   absl::string_view what) {
  if (actual.rows() != expected.rows() || actual.cols() != expected.cols()) {
    return ::testing::AssertionFailure()
           << "size " << actual.rows() << " x " << actual.cols()
           << ", expected " << expected.rows() << " x " << expected.cols();
  }
  if (actual.size() == 0) return ::testing::AssertionSuccess();
  Eigen::Index row = 0;
  Eigen::Index column = 0;
  const scalar_t difference =
      (actual - expected).cwiseAbs().maxCoeff(&row, &column);
  if (difference <= bound)
    return ::testing::AssertionSuccess();  // A NaN difference fails.
  return ::testing::AssertionFailure() << absl::StrFormat(
             "max |difference| %.3e at (%d, %d) above the %s bound %.3e: %.17g "
             "vs %.17g",
             difference, row, column, what, bound, actual(row, column),
             expected(row, column));
}

}  // namespace

absl::Span<const EquivalenceRobot> equivalenceRobots() {
  return kEquivalenceRobots;
}

std::string equivalenceRobotName(
    const ::testing::TestParamInfo<EquivalenceRobot>& info) {
  return info.param.name;
}

RootJointModelPair::RootJointModelPair(
    const pinocchio::Model& quaternionModel,
    const pinocchio::Model& eulerModel,
    validation::EulerStateLayout pinocchioStateLayout,
    validation::EulerStateLayout centroidalStateLayout)
    : quaternionModel_(quaternionModel),
      eulerModel_(eulerModel),
      pinocchioStateLayout_(std::move(pinocchioStateLayout)),
      centroidalStateLayout_(std::move(centroidalStateLayout)) {}

absl::StatusOr<std::unique_ptr<RootJointModelPair>> RootJointModelPair::Create(
    const EquivalenceRobot& robot) {
  ASSIGN_OR_RETURN(const std::string taskFile,
                   robot::resolveResourcePath(robot.taskFile));
  ASSIGN_OR_RETURN(const std::string urdfFile,
                   robot::resolveResourcePath(robot.urdfFile));
  ASSIGN_OR_RETURN(const ModelSettings settings,
                   ModelSettings::Create(taskFile, urdfFile,
                                         /*mpcName=*/"testRootJointEquivalence",
                                         /*verbose=*/false));

  const urdf::ModelInterfaceSharedPtr urdfTree = urdf::parseURDFFile(urdfFile);
  if (urdfTree == nullptr)
    return absl::InvalidArgumentError(
        absl::StrCat("the URDF ", urdfFile, " does not parse"));
  const urdf::ModelInterfaceSharedPtr reduced =
      reducedUrdf(*urdfTree, settings.mpcModelJointNames);

  pinocchio::Model quaternionModel;
  pinocchio::urdf::buildModel(reduced, quaternionRoot(), quaternionModel);
  pinocchio::Model eulerModel;
  pinocchio::urdf::buildModel(reduced, eulerRoot(), eulerModel);
  RETURN_IF_ERROR(checkSameRobot(quaternionModel, eulerModel, settings));

  validation::EulerStateLayout pinocchioStateLayout =
      validation::eulerStateLayoutOf(
          stateInputLayout(settings, StateInputLayout::Mpc::kWholeBody));
  RETURN_IF_ERROR(checkPinocchioStateLayout(pinocchioStateLayout, eulerModel));
  validation::EulerStateLayout centroidalStateLayout =
      validation::eulerStateLayoutOf(
          stateInputLayout(settings, StateInputLayout::Mpc::kCentroidal));

  return absl::WrapUnique(new RootJointModelPair(
      quaternionModel, eulerModel, std::move(pinocchioStateLayout),
      std::move(centroidalStateLayout)));
}

absl::StatusOr<PinocchioMotion> quaternionMotionOf(
    const PinocchioMotion& eulerMotion,
    const validation::EulerStateLayout& layout) {
  const vector_t eulerState = stacked(eulerMotion.q, eulerMotion.v);
  ASSIGN_OR_RETURN(const vector_t physicalState,
                   validation::physicalStateFromEulerState(eulerState, layout));
  ASSIGN_OR_RETURN(const matrix_t tangentJacobian,
                   validation::physicalTangentJacobian(eulerState, layout));
  const Eigen::Index nv = eulerMotion.v.size();
  const Eigen::Index nq = physicalState.size() - nv;
  PinocchioMotion motion;
  motion.q = physicalState.head(nq);
  motion.v = physicalState.tail(nv);
  // (C3): [v; a] = P_e [v_e; a_e]; the top half repeats (C2), which the tests
  // check against Phi's velocity.
  motion.a =
      tangentJacobian.bottomRows(nv) * stacked(eulerMotion.v, eulerMotion.a);
  return motion;
}

absl::StatusOr<ChartVelocityMaps> chartVelocityMapsAt(
    const PinocchioMotion& eulerMotion,
    const validation::EulerStateLayout& layout) {
  ASSIGN_OR_RETURN(const matrix_t tangentJacobian,
                   validation::physicalTangentJacobian(
                       stacked(eulerMotion.q, eulerMotion.v), layout));
  const Eigen::Index nv = eulerMotion.v.size();
  ChartVelocityMaps maps;
  maps.velocityMap = tangentJacobian.topLeftCorner(nv, nv);
  maps.velocityMapRate = matrix_t::Zero(nv, nv);
  const vector3_t angles = eulerMotion.q.segment<3>(kRootRotationStart);
  const vector3_t rates = eulerMotion.v.segment<3>(kRootRotationStart);
  for (Eigen::Index column = 0; column < 3; ++column) {
    // (C3): column i of dT_B/dt is d(T_B(Theta) e_i)/dTheta times dTheta/dt.
    maps.velocityMapRate.block<3, 1>(kRootRotationStart,
                                     kRootRotationStart + column) =
        bodyRateEulerZyxJacobian(angles, vector3_t::Unit(column)) * rates;
  }
  return maps;
}

PinocchioMotion randomEulerMotion(const pinocchio::Model& eulerModel,
                                  const MotionSamplingBounds& bounds,
                                  std::mt19937& generator) {
  const Eigen::Index nj = eulerModel.nv - 6;
  PinocchioMotion motion;
  motion.q.resize(eulerModel.nq);
  motion.q(0) = uniformScalar(-bounds.maxAbsBasePosition,
                              bounds.maxAbsBasePosition, generator);
  motion.q(1) = uniformScalar(-bounds.maxAbsBasePosition,
                              bounds.maxAbsBasePosition, generator);
  motion.q(2) = uniformScalar(
      /*low=*/0.3, /*high=*/0.3 + 2.0 * bounds.maxAbsBasePosition, generator);
  motion.q(3) = uniformScalar(-M_PI, M_PI, generator);
  motion.q(4) =
      uniformScalar(-bounds.maxAbsPitch, bounds.maxAbsPitch, generator);
  motion.q(5) = uniformScalar(-M_PI, M_PI, generator);
  for (Eigen::Index joint = 0; joint < nj; ++joint) {
    const Eigen::Index index = 6 + joint;
    scalar_t lower = eulerModel.lowerPositionLimit(index);
    scalar_t upper = eulerModel.upperPositionLimit(index);
    if (!std::isfinite(lower) || !std::isfinite(upper) ||
        upper - lower > 2.0 * M_PI) {
      lower = -M_PI;
      upper = M_PI;
    }
    motion.q(index) = uniformScalar(lower, upper, generator);
  }

  motion.v.resize(eulerModel.nv);
  motion.v << uniformVector(/*size=*/3, bounds.maxAbsBaseLinearVelocity,
                            generator),
      uniformVector(/*size=*/3, bounds.maxAbsEulerRate, generator),
      uniformVector(nj, bounds.maxAbsJointVelocity, generator);
  motion.a.resize(eulerModel.nv);
  motion.a << uniformVector(/*size=*/3, bounds.maxAbsBaseLinearAcceleration,
                            generator),
      uniformVector(/*size=*/3, bounds.maxAbsEulerAcceleration, generator),
      uniformVector(nj, bounds.maxAbsJointAcceleration, generator);
  return motion;
}

vector_t uniformVector(Eigen::Index size,
                       scalar_t halfWidth,
                       std::mt19937& generator) {
  vector_t vector(size);
  for (Eigen::Index i = 0; i < size; ++i)
    vector(i) = uniformScalar(-halfWidth, halfWidth, generator);
  return vector;
}

scalar_t maxAbs(const matrix_t& m) {
  return m.size() == 0 ? 0.0 : m.cwiseAbs().maxCoeff();
}

::testing::AssertionResult isCloseRelative(const matrix_t& actual,
                                           const matrix_t& expected,
                                           scalar_t relativeTolerance) {
  return isClose(actual, expected,
                 relativeTolerance * std::max(1.0, maxAbs(expected)),
                 "relative");
}

::testing::AssertionResult isCloseAbsolute(const matrix_t& actual,
                                           const matrix_t& expected,
                                           scalar_t absoluteTolerance) {
  return isClose(actual, expected, absoluteTolerance, "absolute");
}

matrix_t symmetricFromUpper(const matrix_t& upper) {
  matrix_t symmetric = upper;
  symmetric.triangularView<Eigen::StrictlyLower>() = upper.transpose();
  return symmetric;
}

}  // namespace ocs2::humanoid
