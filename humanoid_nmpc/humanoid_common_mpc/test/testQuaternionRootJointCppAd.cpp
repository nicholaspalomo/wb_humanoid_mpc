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

// Pinocchio's CppADCodeGen support comes before every other Pinocchio header, as in the vendored OCS2's
// ocs2_pinocchio_interface/implementation/PinocchioInterface.h.
#include "pinocchio/codegen/cppadcg.hpp"

#include "pinocchio/fwd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_centroidal_model/ModelHelperFunctions.h"
#include "ocs2_core/automatic_differentiation/CppAdInterface.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"
#include "ocs2_robotic_tools/common/RotationTransforms.h"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "urdf_parser/urdf_parser.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/config/ConfigFiles.h"
#include "humanoid_common_mpc/config/costs/ContactsFromConfig.h"
#include "humanoid_common_mpc/contact/ContactCenterPoint.h"
#include "humanoid_common_mpc/pinocchio_model/createPinocchioModel.h"
#include "humanoid_mpc_config/task_file.nproto.h"

/*
 * Step 0 of the quaternion base orientation design (humanoid_nmpc/docs/quaternion_base_orientation/README.md): the
 * go/no-go for its decision D1, the root joint JointModelComposite{JointModelTranslation, JointModelSpherical}.
 *
 * The G1 model is built here exactly as createPinocchioModel.cpp builds the MPC model - the URDF with every joint the
 * task file does not name fixed, plus the contact center frames - with only the root joint changed, and nothing in the
 * production code is touched. The kinds of Pinocchio computation the MPCs tape are then taped and code-generated with
 * CppAD / CppADCodeGen through CppAdInterface itself (tape point x = ones, fun.optimize(), gcc with -ffast-math),
 * reading the quaternion only through the safe normalization of design section 2.6, and compared with double-precision
 * Pinocchio:
 *
 *   - computeCentroidalMap, forward kinematics with the contact frame placements, the frame Jacobians in the LOCAL and
 *     LOCAL_WORLD_ALIGNED frames, the CoM and its Jacobian (values);
 *   - crba, nonLinearEffects and ccrba (values);
 *   - the closed-form momentum-matrix inverse path of design section 2.5, v_b = A_b^-1 (m h - A_j qd_j), with the
 *     quaternion row 1/2 G(xi) w_B, generated with its Jacobian;
 *   - forwardKinematics(q, v, a) with getFrameVelocity and getFrameClassicalAcceleration of the contact frames, as the
 *     whole-body MPC's end-effector terms tape them (PinocchioEndEffectorDynamicsCppAd), generated with its Jacobian,
 *     and a scalar of it generated to second order with respect to the root's quaternion, angular velocity and angular
 *     acceleration, as OCS2's former StateInputCostCppAd / StateCostCppAd generated their libraries;
 *
 * at random attitudes and at a pitch of 89.9, 90 and 90.1 degrees, for xi, -xi, xi / 2, 2 xi and a zero quaternion.
 * Each tape is also checked to record no comparison between variables - the reason the repository never calls
 * Eigen's normalized() on an AD scalar (DynamicsHelperFunctions.cpp, computeBaseAcceleration): CppADCodeGen cannot
 * generate code for one. The model itself is checked against the production SphericalZYX model (same joints, mass,
 * contact frames and centroidal momentum for the same physical motion), and for the premises of decision D1:
 * v_base = [pd_W, w_B], A_b = [m I, *; 0, I_G R], and 1/2 G(xi) w_B as pinocchio::integrate moves the quaternion.
 *
 * Outcome, recorded when this test was written (Pinocchio 4.1.0, CppAD 20190200.5, the vendored CppADCodeGen):
 *   - `-c opt` (the repository default): PASS, all 11 tests of the spike as first written.
 *   - `-c fastbuild` (assertions compiled in): PASS, the same 11 tests (the NDEBUG-only positive control at the end is
 *     compiled out there). Pinocchio 4.1's JointModelSpherical::calc and JointModelFreeFlyer::calc assert nothing
 *     about the norm of the quaternion (only their forwardKinematics(M, q) helpers do, which no algorithm here calls),
 *     and the composite's own assertions compare sizes, not scalars, so no comparison is recorded in either mode.
 *     Fastbuild-with-assertions is still never relied on: the CppAD-taping tests run in the default opt mode.
 *   - The norm guard of design section 2.6 as written, CondExpGt(|xi|, epsilon, |xi|, 1), has a NaN reverse-mode
 *     Jacobian at a zero quaternion (SquaredNormGuardIsDifferentiableAtAZeroQuaternion); the guard on the squared norm,
 *     sqrt(CondExpGt(xi^T xi, epsilon^2, xi^T xi, 1)), has the same values and finite derivatives everywhere. The
 *     production normalization should use the squared-norm guard.
 *   - The frame velocity / classical acceleration tests (first and second order) were added after this record, in
 *     answer to a review, and were run under `-c opt` only.
 * Tape sizes are logged by TapesRecordNoComparisonBetweenVariables ("[tape]" lines).
 *
 * What this spike does NOT verify, for Step 8 to cover with the production terms: the whole-body terms as taped (with
 * their state-input mappings and the Schur-complement base acceleration), second-order libraries over all of their
 * inputs, rnea with external forces on a tape (production calls it in double precision only), and the derivatives of
 * the value-only libraries above (centroidal map and frames, crba, nonLinearEffects, ccrba).
 */

namespace ocs2::humanoid {
namespace {

using ad_fun_t = CppAD::ADFun<ad_base_t>;
using AdModel = PinocchioInterfaceCppAd::Model;
using AdData = PinocchioInterfaceCppAd::Data;
using ReferenceFunction = std::function<vector_t(const vector_t&)>;
using AdConfigurationMap = std::function<ad_vector_t(const ad_vector_t&)>;

// LINT.IfChange(robot_files)
constexpr absl::string_view kTaskFile = "robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task.textproto";
constexpr absl::string_view kUrdfFile = "robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
// LINT.ThenChange(//humanoid_nmpc/humanoid_common_mpc/BUILD.bazel:quaternion_root_test_data)

/// q = [p_W(3), xi(4), q_j] for the quaternion root; v = [pd_W(3), w_B(3), qd_j].
constexpr Eigen::Index kQuaternionStart = 3;
/// The composite root joint's index in the model (index 0 is the universe).
constexpr pinocchio::JointIndex kRootJoint = 1;
/// The normalized centroidal momentum that leads the input of the momentum path, h = [cd, L / m].
constexpr Eigen::Index kMomentumDim = 6;

/// epsilon of design section 2.6: a quaternion shorter than this is read as the zero quaternion.
constexpr scalar_t kNormGuard = 1.0e-6;

/// Generated code (-O3 -ffast-math) against double-precision Pinocchio, relative to the largest reference entry.
constexpr scalar_t kValueTolerance = 1.0e-9;
/// Generated Jacobians against central differences of double-precision Pinocchio.
constexpr scalar_t kFiniteDifferenceStep = 1.0e-6;
constexpr scalar_t kJacobianTolerance = 1.0e-6;

constexpr int kRandomAttitudes = 8;

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  return std::string();
}

std::string libraryFolder() {
  return absl::StrCat(testing::TempDir(), "cppad_quaternion_root_joint");
}

/******************************************************************************************************/
/* Quaternion algebra and the safe normalization of design section 2.6                                */
/******************************************************************************************************/

/** G(xi) with xi (x) (w, 0) = G(xi) w, for coefficients (x, y, z, w) (design section 2.1). */
template <typename Scalar>
Eigen::Matrix<Scalar, 4, 3> quaternionRateMatrix(const VECTOR4_T<Scalar>& xi) {
  const Scalar& x = xi(0);
  const Scalar& y = xi(1);
  const Scalar& z = xi(2);
  const Scalar& w = xi(3);
  Eigen::Matrix<Scalar, 4, 3> rateMatrix;
  // clang-format off
  rateMatrix <<  w, -z,  y,
                 z,  w, -x,
                -y,  x,  w,
                -x, -y, -z;
  // clang-format on
  return rateMatrix;
}

/** Where the comparison against epsilon sits in the safe normalization. */
enum class NormGuard {
  // Design section 2.6 as written: n = sqrt(xi^T xi), n_safe = CondExpGt(n, epsilon, n, 1).
  kOnTheNorm,
  // n_safe = sqrt(CondExpGt(xi^T xi, epsilon^2, xi^T xi, 1)): the same values, but the square root is taken of the
  // selected branch, so neither branch has an unbounded derivative at a zero quaternion.
  kOnTheSquaredNorm,
};

absl::string_view guardName(NormGuard guard) {
  return guard == NormGuard::kOnTheNorm ? "norm_guard" : "squared_norm_guard";
}

/** xi / n_safe on the tape, with a conditional expression and no comparison (design section 2.6). */
ad_vector4_t safelyNormalizedQuaternion(const ad_vector4_t& xi, NormGuard guard) {
  const ad_scalar_t squaredNorm = xi.squaredNorm();
  if (guard == NormGuard::kOnTheNorm) {
    const ad_scalar_t norm = CppAD::sqrt(squaredNorm);
    return xi / CppAD::CondExpGt(norm, ad_scalar_t(kNormGuard), norm, ad_scalar_t(1.0));
  }
  const ad_scalar_t selected = CppAD::CondExpGt(squaredNorm, ad_scalar_t(kNormGuard * kNormGuard), squaredNorm, ad_scalar_t(1.0));
  return xi / CppAD::sqrt(selected);
}

/** The same rule in double precision, with an if. */
vector4_t safelyNormalizedQuaternion(const vector4_t& xi) {
  const scalar_t norm = xi.norm();
  return xi / (norm > kNormGuard ? norm : 1.0);
}

/** Pinocchio's q = [p_W, xi_hat, q_j] from the ambient configuration [p_W, xi, q_j] on the tape. */
ad_vector_t safelyNormalizedConfiguration(const ad_vector_t& ambientConfiguration, NormGuard guard) {
  ad_vector_t q = ambientConfiguration;
  q.segment<4>(kQuaternionStart) = safelyNormalizedQuaternion(ambientConfiguration.segment<4>(kQuaternionStart), guard);
  return q;
}

vector_t safelyNormalizedConfiguration(const vector_t& ambientConfiguration) {
  vector_t q = ambientConfiguration;
  q.segment<4>(kQuaternionStart) = safelyNormalizedQuaternion(vector4_t(ambientConfiguration.segment<4>(kQuaternionStart)));
  return q;
}

/** The composite root of decision D1: world-frame translation, then a quaternion rotation. */
pinocchio::JointModelComposite translationSphericalRoot() {
  pinocchio::JointModelComposite root(2);
  root.addJoint(pinocchio::JointModelTranslation());
  root.addJoint(pinocchio::JointModelSpherical());
  return root;
}

/******************************************************************************************************/
/* The computations the MPCs tape, written once for double and AD scalars                            */
/******************************************************************************************************/

template <typename Derived>
void append(const Eigen::MatrixBase<Derived>& block, std::vector<typename Derived::Scalar>& values) {
  for (Eigen::Index col = 0; col < block.cols(); ++col) {
    for (Eigen::Index row = 0; row < block.rows(); ++row) {
      values.push_back(block(row, col));
    }
  }
}

template <typename Scalar>
VECTOR_T<Scalar> toVector(const std::vector<Scalar>& values) {
  return Eigen::Map<const VECTOR_T<Scalar>>(values.data(), static_cast<Eigen::Index>(values.size()));
}

/** computeCentroidalMap, the frame placements and LOCAL / LOCAL_WORLD_ALIGNED frame Jacobians, the CoM and its Jacobian. */
template <typename Scalar>
VECTOR_T<Scalar> centroidalMapAndFrames(const pinocchio::ModelTpl<Scalar>& model,
                                        pinocchio::DataTpl<Scalar>& data,
                                        const VECTOR_T<Scalar>& q,
                                        const std::vector<pinocchio::FrameIndex>& frames) {
  std::vector<Scalar> values;
  append(pinocchio::computeCentroidalMap(model, data, q), values);
  pinocchio::computeJointJacobians(model, data, q);
  pinocchio::updateFramePlacements(model, data);
  for (const pinocchio::FrameIndex frame : frames) {
    append(data.oMf[frame].translation(), values);
    append(data.oMf[frame].rotation(), values);
    for (const pinocchio::ReferenceFrame referenceFrame :
         {pinocchio::ReferenceFrame::LOCAL, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED}) {
      MATRIX_T<Scalar> jacobian = MATRIX_T<Scalar>::Zero(6, model.nv);
      pinocchio::getFrameJacobian(model, data, frame, referenceFrame, jacobian);
      append(jacobian, values);
    }
  }
  append(pinocchio::jacobianCenterOfMass(model, data, q), values);
  append(data.com[0], values);
  return toVector(values);
}

/** The upper triangle of crba()'s mass matrix, the part it writes. */
template <typename Scalar>
VECTOR_T<Scalar> massMatrixUpperTriangle(const pinocchio::ModelTpl<Scalar>& model,
                                         pinocchio::DataTpl<Scalar>& data,
                                         const VECTOR_T<Scalar>& q) {
  const MATRIX_T<Scalar>& massMatrix = pinocchio::crba(model, data, q);
  std::vector<Scalar> values;
  for (Eigen::Index col = 0; col < massMatrix.cols(); ++col) {
    for (Eigen::Index row = 0; row <= col; ++row) {
      values.push_back(massMatrix(row, col));
    }
  }
  return toVector(values);
}

template <typename Scalar>
VECTOR_T<Scalar> nonlinearEffects(const pinocchio::ModelTpl<Scalar>& model,
                                  pinocchio::DataTpl<Scalar>& data,
                                  const VECTOR_T<Scalar>& q,
                                  const VECTOR_T<Scalar>& v) {
  return pinocchio::nonLinearEffects(model, data, q, v);
}

/** ccrba: the centroidal momentum matrix, the centroidal momentum and the centroidal inertia. */
template <typename Scalar>
VECTOR_T<Scalar> centroidalCompositeRigidBody(const pinocchio::ModelTpl<Scalar>& model,
                                              pinocchio::DataTpl<Scalar>& data,
                                              const VECTOR_T<Scalar>& q,
                                              const VECTOR_T<Scalar>& v) {
  std::vector<Scalar> values;
  append(pinocchio::ccrba(model, data, q, v), values);
  append(data.hg.toVector(), values);
  append(data.Ig.matrix(), values);
  return toVector(values);
}

/**
 * forwardKinematics(q, v, a), then the LOCAL_WORLD_ALIGNED spatial velocity and classical acceleration of each frame, as
 * PinocchioEndEffectorDynamicsCppAd tapes them for the whole-body MPC.
 */
template <typename Scalar>
VECTOR_T<Scalar> frameVelocitiesAndAccelerations(const pinocchio::ModelTpl<Scalar>& model,
                                                 pinocchio::DataTpl<Scalar>& data,
                                                 const VECTOR_T<Scalar>& q,
                                                 const VECTOR_T<Scalar>& v,
                                                 const VECTOR_T<Scalar>& a,
                                                 const std::vector<pinocchio::FrameIndex>& frames) {
  pinocchio::forwardKinematics(model, data, q, v, a);
  std::vector<Scalar> values;
  for (const pinocchio::FrameIndex frame : frames) {
    append(pinocchio::getFrameVelocity(model, data, frame, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED).toVector(), values);
    append(pinocchio::getFrameClassicalAcceleration(model, data, frame, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED).toVector(), values);
  }
  return toVector(values);
}

/******************************************************************************************************/
/* Taped functions, their double-precision references, and the G1 models                            */
/******************************************************************************************************/

/** One computation: taped on AD scalars from an input x, and the same computation in double precision. */
struct PinocchioFunction {
  std::string name;
  Eigen::Index variableDim = 0;
  // Where the quaternion sits in x, for the tests that scale it.
  Eigen::Index quaternionStart = 0;
  CppAdInterface::ad_function_t taped;
  ReferenceFunction reference;
};

/** The G1 MPC model on the composite quaternion root, beside the production model on today's SphericalZYX root. */
class G1Models {
 public:
  G1Models() {
    const std::string taskFile = runfilePath(kTaskFile);
    urdfFile_ = runfilePath(kUrdfFile);
    CHECK(!taskFile.empty() && !urdfFile_.empty()) << "the G1 files are not in the runfiles";
    task_ = loadTaskFile(taskFile).value();
    settings_ = std::make_unique<ModelSettings>(
        ModelSettings::Create(task_, urdfFile_, "testQuaternionRootJointCppAd_", /*verbose=*/false).value());

    // What loadCustomPinocchioInterface (createPinocchioModel.cpp) builds, with the root joint of decision D1.
    const urdf::ModelInterfaceSharedPtr urdfTree = urdf::parseURDFFile(urdfFile_);
    CHECK(urdfTree != nullptr) << urdfFile_;
    const urdf::ModelInterfaceSharedPtr reducedTree = std::make_shared<urdf::ModelInterface>(*urdfTree);
    const std::vector<std::string>& mpcJointNames = settings_->mpcModelJointNames;
    for (std::pair<const std::string, std::shared_ptr<urdf::Joint>>& joint : reducedTree->joints_) {
      if (std::find(mpcJointNames.begin(), mpcJointNames.end(), joint.first) == mpcJointNames.end()) {
        joint.second->type = urdf::Joint::FIXED;
      }
    }
    pinocchio::Model quaternionModel;
    pinocchio::urdf::buildModel(reducedTree, translationSphericalRoot(), quaternionModel);
    for (size_t contact = 0; contact < kNumContacts; ++contact) {
      const ContactCenterPoint center = contactCenterPointFromConfig(task_.contacts, *settings_, static_cast<int>(contact)).value();
      quaternionModel.addFrame(pinocchio::Frame(
          center.frameName, quaternionModel.getJointId(center.parentJointName), quaternionModel.getFrameId(center.parentJointName),
          pinocchio::SE3(matrix3_t::Identity(), center.translationFromParent), pinocchio::FIXED_JOINT));
      contactFrames_.push_back(quaternionModel.getFrameId(center.frameName));
    }
    quaternionRoot_ = std::make_unique<PinocchioInterface>(quaternionModel, urdfTree);
    quaternionRootAd_ = std::make_shared<const PinocchioInterfaceCppAd>(quaternionRoot_->toCppAd());

    eulerRoot_ = std::make_unique<PinocchioInterface>(loadCustomPinocchioInterface(task_, urdfFile_, *settings_).value());
    eulerRootAd_ = std::make_shared<const PinocchioInterfaceCppAd>(eulerRoot_->toCppAd());
    mass_ = pinocchio::computeTotalMass(quaternionRoot_->getModel());
  }

  const pinocchio::Model& model() const { return quaternionRoot_->getModel(); }
  const pinocchio::Model& eulerModel() const { return eulerRoot_->getModel(); }
  const std::vector<pinocchio::FrameIndex>& contactFrames() const { return contactFrames_; }
  Eigen::Index numJoints() const { return static_cast<Eigen::Index>(settings_->mpc_joint_dim); }
  scalar_t mass() const { return mass_; }

  /** The quaternion root's functions, reading the quaternion through `guard`. */
  std::vector<PinocchioFunction> quaternionRootFunctions(NormGuard guard) const {
    const AdConfigurationMap toConfiguration = [guard](const ad_vector_t& ambient) {
      return safelyNormalizedConfiguration(ambient, guard);
    };
    const ReferenceFunction toReferenceConfiguration = [](const vector_t& ambient) { return safelyNormalizedConfiguration(ambient); };
    return functions(quaternionRootAd_, quaternionRoot_->getModel(), toConfiguration, toReferenceConfiguration,
                     /*appendQuaternionRate=*/true);
  }

  /** The same functions on the production SphericalZYX root, taped for their sizes only. */
  std::vector<PinocchioFunction> eulerRootFunctions() const {
    const AdConfigurationMap toConfiguration = [](const ad_vector_t& ambient) { return ambient; };
    const ReferenceFunction toReferenceConfiguration = [](const vector_t& ambient) { return ambient; };
    return functions(eulerRootAd_, eulerRoot_->getModel(), toConfiguration, toReferenceConfiguration, /*appendQuaternionRate=*/false);
  }

  /** The function of the quaternion root called `name`, reading the quaternion through `guard`. */
  PinocchioFunction quaternionRootFunction(absl::string_view name, NormGuard guard = NormGuard::kOnTheNorm) const {
    for (const PinocchioFunction& function : quaternionRootFunctions(guard)) {
      if (function.name == name) return function;
    }
    LOG(FATAL) << "no function named " << name;
    return PinocchioFunction();
  }

 private:
  std::vector<PinocchioFunction> functions(std::shared_ptr<const PinocchioInterfaceCppAd> adInterface,
                                           const pinocchio::Model& model,
                                           const AdConfigurationMap& toConfiguration,
                                           const ReferenceFunction& toReferenceConfiguration,
                                           bool appendQuaternionRate) const {
    const Eigen::Index nq = model.nq;
    const Eigen::Index nv = model.nv;
    const Eigen::Index nj = numJoints();
    const std::vector<pinocchio::FrameIndex> frames = contactFrames_;
    const scalar_t mass = mass_;
    std::vector<PinocchioFunction> result;

    result.push_back(PinocchioFunction{
        .name = "centroidal_map_and_frames",
        .variableDim = nq,
        .quaternionStart = kQuaternionStart,
        .taped =
            [adInterface, toConfiguration, frames](const ad_vector_t& x, ad_vector_t& y) {
              AdData data(adInterface->getModel());
              y = centroidalMapAndFrames<ad_scalar_t>(adInterface->getModel(), data, toConfiguration(x), frames);
            },
        .reference =
            [&model, toReferenceConfiguration, frames](const vector_t& x) {
              pinocchio::Data data(model);
              return centroidalMapAndFrames<scalar_t>(model, data, toReferenceConfiguration(x), frames);
            },
    });

    result.push_back(PinocchioFunction{
        .name = "crba",
        .variableDim = nq,
        .quaternionStart = kQuaternionStart,
        .taped =
            [adInterface, toConfiguration](const ad_vector_t& x, ad_vector_t& y) {
              AdData data(adInterface->getModel());
              y = massMatrixUpperTriangle<ad_scalar_t>(adInterface->getModel(), data, toConfiguration(x));
            },
        .reference =
            [&model, toReferenceConfiguration](const vector_t& x) {
              pinocchio::Data data(model);
              return massMatrixUpperTriangle<scalar_t>(model, data, toReferenceConfiguration(x));
            },
    });

    result.push_back(PinocchioFunction{
        .name = "non_linear_effects",
        .variableDim = nq + nv,
        .quaternionStart = kQuaternionStart,
        .taped =
            [adInterface, toConfiguration, nq, nv](const ad_vector_t& x, ad_vector_t& y) {
              AdData data(adInterface->getModel());
              y = nonlinearEffects<ad_scalar_t>(adInterface->getModel(), data, toConfiguration(x.head(nq)), x.segment(nq, nv));
            },
        .reference =
            [&model, toReferenceConfiguration, nq, nv](const vector_t& x) {
              pinocchio::Data data(model);
              return nonlinearEffects<scalar_t>(model, data, toReferenceConfiguration(x.head(nq)), x.segment(nq, nv));
            },
    });

    result.push_back(PinocchioFunction{
        .name = "ccrba",
        .variableDim = nq + nv,
        .quaternionStart = kQuaternionStart,
        .taped =
            [adInterface, toConfiguration, nq, nv](const ad_vector_t& x, ad_vector_t& y) {
              AdData data(adInterface->getModel());
              y = centroidalCompositeRigidBody<ad_scalar_t>(adInterface->getModel(), data, toConfiguration(x.head(nq)), x.segment(nq, nv));
            },
        .reference =
            [&model, toReferenceConfiguration, nq, nv](const vector_t& x) {
              pinocchio::Data data(model);
              return centroidalCompositeRigidBody<scalar_t>(model, data, toReferenceConfiguration(x.head(nq)), x.segment(nq, nv));
            },
    });

    // x = [q_ambient(nq), v(nv), a(nv)] -> the contact frames' velocities and classical accelerations.
    result.push_back(PinocchioFunction{
        .name = "frame_velocity_and_acceleration",
        .variableDim = nq + 2 * nv,
        .quaternionStart = kQuaternionStart,
        .taped =
            [adInterface, toConfiguration, frames, nq, nv](const ad_vector_t& x, ad_vector_t& y) {
              AdData data(adInterface->getModel());
              y = frameVelocitiesAndAccelerations<ad_scalar_t>(adInterface->getModel(), data, toConfiguration(x.head(nq)),
                                                               x.segment(nq, nv), x.segment(nq + nv, nv), frames);
            },
        .reference =
            [&model, toReferenceConfiguration, frames, nq, nv](const vector_t& x) {
              pinocchio::Data data(model);
              return frameVelocitiesAndAccelerations<scalar_t>(model, data, toReferenceConfiguration(x.head(nq)), x.segment(nq, nv),
                                                               x.segment(nq + nv, nv), frames);
            },
    });

    // x = [h(6), q_ambient(nq), qd_j(nj)] -> [v_b, 1/2 G(xi) w_B]: the closed-form inverse of design section 2.5, against
    // a pivoting solve in double precision. The quaternion row uses the raw xi (design decision D5).
    const Eigen::Index outputDim = appendQuaternionRate ? 10 : 6;
    result.push_back(PinocchioFunction{
        .name = "momentum_inverse_path",
        .variableDim = kMomentumDim + nq + nj,
        .quaternionStart = kMomentumDim + kQuaternionStart,
        .taped =
            [adInterface, toConfiguration, nq, nj, mass, outputDim](const ad_vector_t& x, ad_vector_t& y) {
              const AdModel& adModel = adInterface->getModel();
              AdData data(adModel);
              const ad_vector_t q = toConfiguration(x.segment(kMomentumDim, nq));
              const AdData::Matrix6x& centroidalMap = pinocchio::computeCentroidalMap(adModel, data, q);
              const ad_matrix6_t baseBlock = centroidalMap.leftCols<6>();
              const ad_matrix6_t baseBlockInverse = computeFloatingBaseCentroidalMomentumMatrixInverse<ad_scalar_t>(baseBlock);
              const ad_vector6_t baseVelocity =
                  baseBlockInverse * (ad_scalar_t(mass) * x.head<kMomentumDim>() - centroidalMap.rightCols(nj) * x.tail(nj));
              y.resize(outputDim);
              y.head<6>() = baseVelocity;
              if (outputDim > 6) {
                const ad_vector4_t rawQuaternion = x.segment<4>(kMomentumDim + kQuaternionStart);
                y.tail<4>() = ad_scalar_t(0.5) * quaternionRateMatrix<ad_scalar_t>(rawQuaternion) * baseVelocity.tail<3>();
              }
            },
        .reference =
            [&model, toReferenceConfiguration, nq, nj, mass, outputDim](const vector_t& x) {
              pinocchio::Data data(model);
              const vector_t q = toReferenceConfiguration(x.segment(kMomentumDim, nq));
              const pinocchio::Data::Matrix6x& centroidalMap = pinocchio::computeCentroidalMap(model, data, q);
              const matrix6_t baseBlock = centroidalMap.leftCols<6>();
              const vector6_t baseVelocity =
                  baseBlock.partialPivLu().solve(mass * x.head<kMomentumDim>() - centroidalMap.rightCols(nj) * x.tail(nj));
              vector_t y(outputDim);
              y.head<6>() = baseVelocity;
              if (outputDim > 6) {
                const vector4_t rawQuaternion = x.segment<4>(kMomentumDim + kQuaternionStart);
                y.tail<4>() = 0.5 * quaternionRateMatrix<scalar_t>(rawQuaternion) * baseVelocity.tail<3>();
              }
              return y;
            },
    });
    return result;
  }

  mpc_config::TaskFile task_;
  std::string urdfFile_;
  std::unique_ptr<ModelSettings> settings_;
  std::vector<pinocchio::FrameIndex> contactFrames_;
  std::unique_ptr<PinocchioInterface> quaternionRoot_;
  std::shared_ptr<const PinocchioInterfaceCppAd> quaternionRootAd_;
  std::unique_ptr<PinocchioInterface> eulerRoot_;
  std::shared_ptr<const PinocchioInterfaceCppAd> eulerRootAd_;
  scalar_t mass_ = 0.0;
};

/** The models, built once for the whole test program (never destroyed, as a function-local static). */
const G1Models& g1() {
  static const G1Models* absl_nonnull const kModels = new G1Models();
  return *kModels;
}

/******************************************************************************************************/
/* Inputs                                                                                             */
/******************************************************************************************************/

struct Attitude {
  std::string label;
  vector4_t coefficients;  // (x, y, z, w), unit norm
};

vector4_t eulerZyxQuaternion(scalar_t yaw, scalar_t pitch, scalar_t roll) {
  return getQuaternionFromEulerAnglesZyx<scalar_t>(vector3_t(yaw, pitch, roll)).coeffs();
}

/** Random attitudes, the pitch singularity of the Euler formulation and its neighbors, and the identity. */
std::vector<Attitude> testAttitudes() {
  std::vector<Attitude> attitudes;
  std::mt19937 generator(11);
  std::normal_distribution<scalar_t> normal(0.0, 1.0);
  for (int sample = 0; sample < kRandomAttitudes; ++sample) {
    vector4_t coefficients;
    for (Eigen::Index i = 0; i < 4; ++i) coefficients(i) = normal(generator);
    attitudes.push_back(Attitude{.label = absl::StrCat("random attitude ", sample), .coefficients = coefficients.normalized()});
  }
  for (const scalar_t pitchDegrees : {89.9, 90.0, 90.1, -90.0}) {
    const scalar_t pitch = pitchDegrees * M_PI / 180.0;
    attitudes.push_back(Attitude{.label = absl::StrCat("pitch ", pitchDegrees, " deg"),
                                 .coefficients = eulerZyxQuaternion(/*yaw=*/0.0, pitch, /*roll=*/0.0)});
    attitudes.push_back(Attitude{.label = absl::StrCat("yaw 0.7, pitch ", pitchDegrees, " deg, roll -0.4"),
                                 .coefficients = eulerZyxQuaternion(/*yaw=*/0.7, pitch, /*roll=*/-0.4)});
  }
  attitudes.push_back(Attitude{.label = "identity", .coefficients = vector4_t(0.0, 0.0, 0.0, 1.0)});
  return attitudes;
}

vector_t uniformVector(Eigen::Index size, scalar_t halfWidth, std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> distribution(-halfWidth, halfWidth);
  vector_t result(size);
  for (Eigen::Index i = 0; i < size; ++i) result(i) = distribution(generator);
  return result;
}

/** A random input of `function`: positions, joint angles, velocities and momenta of order one. */
vector_t randomInput(const PinocchioFunction& function, std::mt19937& generator) {
  vector_t x = uniformVector(function.variableDim, /*halfWidth=*/0.8, generator);
  x.segment<4>(function.quaternionStart) = vector4_t(0.0, 0.0, 0.0, 1.0);
  return x;
}

vector_t withQuaternion(const vector_t& x, Eigen::Index quaternionStart, const vector4_t& coefficients) {
  vector_t result = x;
  result.segment<4>(quaternionStart) = coefficients;
  return result;
}

/******************************************************************************************************/
/* Taping and code generation                                                                        */
/******************************************************************************************************/

struct TapeSize {
  size_t operations = 0;
  size_t optimizedOperations = 0;
  size_t optimizedVariables = 0;
};

/** Tapes `function` as CppAdInterface::createModels does (at x = ones), recording comparisons or not. */
TapeSize tape(const CppAdInterface::ad_function_t& function, Eigen::Index variableDim, bool recordCompare) {
  ad_vector_t x = ad_vector_t::Ones(variableDim);
  CppAD::Independent(x, /*abort_op_index=*/0, recordCompare);
  ad_vector_t y;
  function(x, y);
  ad_fun_t fun(x, y);
  TapeSize size;
  size.operations = fun.size_op();
  fun.optimize();
  size.optimizedOperations = fun.size_op();
  size.optimizedVariables = fun.size_var();
  return size;
}

/** The comparisons the tape of `function` records: the operations that are only there when comparisons are recorded. */
size_t recordedComparisons(const CppAdInterface::ad_function_t& function, Eigen::Index variableDim) {
  return tape(function, variableDim, /*recordCompare=*/true).operations - tape(function, variableDim, /*recordCompare=*/false).operations;
}

/**
 * Runs the tape of `function` forward on CppADCodeGen variables, the step at which code generation throws on a
 * comparison between variables ("GreaterThanZero cannot be called for non-parameters"). Empty when it does not throw.
 */
std::string codeGenerationError(const CppAdInterface::ad_function_t& function, Eigen::Index variableDim) {
  ad_vector_t x = ad_vector_t::Ones(variableDim);
  CppAD::Independent(x);
  ad_vector_t y;
  function(x, y);
  ad_fun_t fun(x, y);
  CppAD::cg::CodeHandler<scalar_t> handler;
  CppAD::vector<ad_base_t> variables(static_cast<size_t>(variableDim));
  handler.makeVariables(variables);
  try {
    fun.Forward(/*q=*/0, variables);
  } catch (const std::exception& error) {
    return error.what();
  }
  return std::string();
}

std::unique_ptr<CppAdInterface> generate(const PinocchioFunction& function, CppAdInterface::ApproximationOrder order) {
  std::unique_ptr<CppAdInterface> library =
      std::make_unique<CppAdInterface>(function.taped, static_cast<size_t>(function.variableDim),
                                       absl::StrCat("testQuaternionRootJointCppAd_", function.name), libraryFolder());
  library->createModels(order, /*verbose=*/false);
  return library;
}

/******************************************************************************************************/
/* Comparisons                                                                                       */
/******************************************************************************************************/

/** max |a - b| relative to the largest entry of b (at least one). */
scalar_t relativeError(const vector_t& a, const vector_t& b) {
  CHECK_EQ(a.size(), b.size());
  const scalar_t scale = std::max(1.0, b.cwiseAbs().maxCoeff());
  return (a - b).cwiseAbs().maxCoeff() / scale;
}

scalar_t relativeError(const matrix_t& a, const matrix_t& b) {
  return relativeError(vector_t(Eigen::Map<const vector_t>(a.data(), a.size())), vector_t(Eigen::Map<const vector_t>(b.data(), b.size())));
}

matrix_t centralDifferenceJacobian(const ReferenceFunction& function, const vector_t& x) {
  const vector_t value = function(x);
  matrix_t jacobian(value.size(), x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    vector_t plus = x;
    vector_t minus = x;
    plus(i) += kFiniteDifferenceStep;
    minus(i) -= kFiniteDifferenceStep;
    jacobian.col(i) = (function(plus) - function(minus)) / (2.0 * kFiniteDifferenceStep);
  }
  return jacobian;
}

/**
 * The generated library of a configuration function equals double-precision Pinocchio at every test attitude, and sees
 * the quaternion only through the safe normalization: -xi, xi / 2 and 2 xi give the values of xi (double cover and
 * radial invariance), and a zero quaternion gives the values of the identity rather than NaN.
 */
void expectGeneratedMatchesPinocchio(const PinocchioFunction& function, const CppAdInterface& library) {
  std::mt19937 generator(5);
  for (const Attitude& attitude : testAttitudes()) {
    const vector_t x = withQuaternion(randomInput(function, generator), function.quaternionStart, attitude.coefficients);
    const vector_t expected = function.reference(x);
    ASSERT_TRUE(expected.allFinite()) << attitude.label;
    for (const scalar_t scale : {1.0, -1.0, 0.5, 2.0}) {
      const vector_t generated = library.getFunctionValue(withQuaternion(x, function.quaternionStart, scale * attitude.coefficients));
      ASSERT_TRUE(generated.allFinite()) << function.name << ", " << attitude.label << ", quaternion scaled by " << scale;
      EXPECT_LE(relativeError(generated, expected), kValueTolerance)
          << function.name << ", " << attitude.label << ", quaternion scaled by " << scale;
    }
    const vector_t atZero = library.getFunctionValue(withQuaternion(x, function.quaternionStart, vector4_t::Zero()));
    const vector_t atIdentity = function.reference(withQuaternion(x, function.quaternionStart, vector4_t(0.0, 0.0, 0.0, 1.0)));
    ASSERT_TRUE(atZero.allFinite()) << function.name << ": a zero quaternion";
    EXPECT_LE(relativeError(atZero, atIdentity), kValueTolerance) << function.name << ": a zero quaternion is not read as the identity";
  }
}

/**
 * The momentum path of design section 2.5, generated with its Jacobian, once per guard for the whole test program
 * (never destroyed, as a function-local static). The Jacobian is generated in reverse mode (10 outputs of 59 inputs).
 */
const CppAdInterface& momentumPathLibrary(NormGuard guard) {
  static const CppAdInterface* absl_nonnull const kNormGuardLibrary =
      generate(g1().quaternionRootFunction("momentum_inverse_path", NormGuard::kOnTheNorm), CppAdInterface::ApproximationOrder::First)
          .release();
  static const CppAdInterface* absl_nonnull const kSquaredNormGuardLibrary =
      generate(g1().quaternionRootFunction("momentum_inverse_path", NormGuard::kOnTheSquaredNorm),
               CppAdInterface::ApproximationOrder::First)
          .release();
  return guard == NormGuard::kOnTheNorm ? *kNormGuardLibrary : *kSquaredNormGuardLibrary;
}

/******************************************************************************************************/
/* Tests                                                                                              */
/******************************************************************************************************/

TEST(QuaternionRootJointModel, IsTheProductionRobotWithAQuaternionRoot) {
  const pinocchio::Model& model = g1().model();
  const pinocchio::Model& eulerModel = g1().eulerModel();
  const Eigen::Index nj = g1().numJoints();
  ASSERT_EQ(model.nq, 7 + nj);
  ASSERT_EQ(model.nv, 6 + nj);
  ASSERT_EQ(eulerModel.nq, 6 + nj);
  ASSERT_EQ(eulerModel.njoints, model.njoints);
  for (pinocchio::JointIndex joint = kRootJoint + 1; joint < static_cast<pinocchio::JointIndex>(model.njoints); ++joint) {
    EXPECT_EQ(model.names[joint], eulerModel.names[joint]);
  }
  EXPECT_NEAR(pinocchio::computeTotalMass(model), pinocchio::computeTotalMass(eulerModel), 1.0e-12);

  // The same physical pose and motion: q = [p, quaternion(yaw, pitch, roll), q_j] against [p, (yaw, pitch, roll), q_j],
  // and v = [pd, w_B, qd_j] against [pd, Euler rates, qd_j], where w_B is the body angular velocity of those rates.
  std::mt19937 generator(3);
  pinocchio::Data data(model);
  pinocchio::Data eulerData(eulerModel);
  for (const scalar_t pitch : {0.3, M_PI / 2.0 - 0.01}) {
    const vector3_t eulerZyx(0.7, pitch, -0.4);
    vector_t eulerQ = uniformVector(eulerModel.nq, /*halfWidth=*/0.6, generator);
    eulerQ.segment<3>(3) = eulerZyx;
    const vector_t eulerV = uniformVector(eulerModel.nv, /*halfWidth=*/1.0, generator);
    pinocchio::forwardKinematics(eulerModel, eulerData, eulerQ, eulerV);
    pinocchio::updateFramePlacements(eulerModel, eulerData);
    vector_t q(model.nq);
    q << eulerQ.head<3>(), eulerZyxQuaternion(eulerZyx(0), eulerZyx(1), eulerZyx(2)), eulerQ.tail(nj);
    vector_t v = eulerV;
    v.segment<3>(3) = eulerData.v[kRootJoint].angular();
    pinocchio::forwardKinematics(model, data, q, v);
    pinocchio::updateFramePlacements(model, data);
    for (const pinocchio::FrameIndex frame : g1().contactFrames()) {
      const pinocchio::FrameIndex eulerFrame = eulerModel.getFrameId(model.frames[frame].name);
      EXPECT_LE((data.oMf[frame].translation() - eulerData.oMf[eulerFrame].translation()).norm(), 1.0e-12) << model.frames[frame].name;
      EXPECT_LE((data.oMf[frame].rotation() - eulerData.oMf[eulerFrame].rotation()).norm(), 1.0e-12) << model.frames[frame].name;
    }
    const vector6_t momentum = pinocchio::computeCentroidalMap(model, data, q) * v;
    const vector6_t eulerMomentum = pinocchio::computeCentroidalMap(eulerModel, eulerData, eulerQ) * eulerV;
    EXPECT_LE((momentum - eulerMomentum).norm(), 1.0e-9 * std::max(1.0, eulerMomentum.norm())) << "pitch " << pitch;
  }
}

TEST(QuaternionRootJointModel, RootVelocityIsWorldLinearAndBodyAngular) {
  // v_base = [pd_W, w_B] (design section 2.2), and the quaternion moves as xi_dot = 1/2 G(xi) w_B under
  // pinocchio::integrate - the kinematic row of the quaternion dynamics.
  const pinocchio::Model& model = g1().model();
  pinocchio::Data data(model);
  std::mt19937 generator(17);
  for (const Attitude& attitude : testAttitudes()) {
    vector_t q = uniformVector(model.nq, /*halfWidth=*/0.6, generator);
    q.segment<4>(kQuaternionStart) = attitude.coefficients;
    const vector_t v = uniformVector(model.nv, /*halfWidth=*/1.0, generator);
    pinocchio::forwardKinematics(model, data, q, v);
    const matrix3_t rotation = data.oMi[kRootJoint].rotation();
    EXPECT_LE((data.v[kRootJoint].linear() - rotation.transpose() * v.head<3>()).norm(), 1.0e-12) << attitude.label;
    EXPECT_LE((data.v[kRootJoint].angular() - v.segment<3>(3)).norm(), 1.0e-12) << attitude.label;

    vector_t baseRotation = vector_t::Zero(model.nv);
    baseRotation.segment<3>(3) = v.segment<3>(3);
    const vector_t forward = pinocchio::integrate(model, q, kFiniteDifferenceStep * baseRotation);
    const vector_t backward = pinocchio::integrate(model, q, -kFiniteDifferenceStep * baseRotation);
    const vector4_t quaternionRate =
        (forward.segment<4>(kQuaternionStart) - backward.segment<4>(kQuaternionStart)) / (2.0 * kFiniteDifferenceStep);
    const vector4_t expected = 0.5 * quaternionRateMatrix<scalar_t>(attitude.coefficients) * v.segment<3>(3);
    EXPECT_LE((quaternionRate - expected).norm(), 1.0e-8) << attitude.label;
  }
}

TEST(QuaternionRootJointModel, MomentumMatrixKeepsTheStructureOfTheClosedFormInverse) {
  // A_b = [m I, -m [c - p]x R; 0, I_G R] (design section 2.5): the blocks computeFloatingBaseCentroidalMomentumMatrixInverse
  // assumes, and a rotational block with the singular values of I_G at every attitude, the 90 degree pitch included.
  const pinocchio::Model& model = g1().model();
  pinocchio::Data data(model);
  std::mt19937 generator(23);
  for (const Attitude& attitude : testAttitudes()) {
    vector_t q = uniformVector(model.nq, /*halfWidth=*/0.6, generator);
    q.segment<4>(kQuaternionStart) = attitude.coefficients;
    const vector_t v = uniformVector(model.nv, /*halfWidth=*/1.0, generator);
    pinocchio::ccrba(model, data, q, v);
    const matrix6_t baseBlock = data.Ag.leftCols<6>();
    const matrix3_t translationalBlock = baseBlock.topLeftCorner<3, 3>();
    const matrix3_t momentOfTranslationBlock = baseBlock.bottomLeftCorner<3, 3>();
    EXPECT_LE((translationalBlock - g1().mass() * matrix3_t::Identity()).norm(), 1.0e-9 * g1().mass()) << attitude.label;
    EXPECT_LE(momentOfTranslationBlock.norm(), 1.0e-9) << attitude.label;
    const vector3_t singularValues = Eigen::JacobiSVD<matrix3_t>(baseBlock.bottomRightCorner<3, 3>()).singularValues();
    const vector3_t inertiaSingularValues = Eigen::JacobiSVD<matrix3_t>(data.Ig.inertia().matrix()).singularValues();
    EXPECT_LE((singularValues - inertiaSingularValues).norm(), 1.0e-9 * inertiaSingularValues.norm()) << attitude.label;
  }

  // Positive control: on today's SphericalZYX root the rotational block is singular at a pitch of 90 degrees.
  const pinocchio::Model& eulerModel = g1().eulerModel();
  pinocchio::Data eulerData(eulerModel);
  vector_t eulerQ = uniformVector(eulerModel.nq, /*halfWidth=*/0.6, generator);
  eulerQ.segment<3>(3) = vector3_t(0.7, M_PI / 2.0, -0.4);
  const matrix3_t eulerRotationalBlock = pinocchio::computeCentroidalMap(eulerModel, eulerData, eulerQ).block<3, 3>(3, 3);
  EXPECT_LT(Eigen::JacobiSVD<matrix3_t>(eulerRotationalBlock).singularValues()(2), 1.0e-9);
}

TEST(QuaternionRootJointCppAd, TapesRecordNoComparisonBetweenVariables) {
  // Positive control: the detectors see the comparison of Eigen's normalized() on an AD scalar.
  const CppAdInterface::ad_function_t eigenNormalized = [](const ad_vector_t& x, ad_vector_t& y) {
    y = ad_vector4_t(x.head<4>()).normalized();
  };
  ASSERT_GE(recordedComparisons(eigenNormalized, /*variableDim=*/4), 1u);
  const std::string eigenNormalizedError = codeGenerationError(eigenNormalized, /*variableDim=*/4);
  EXPECT_NE(eigenNormalizedError.find("non-parameters"), std::string::npos) << eigenNormalizedError;

  for (const NormGuard guard : {NormGuard::kOnTheNorm, NormGuard::kOnTheSquaredNorm}) {
    for (const PinocchioFunction& function : g1().quaternionRootFunctions(guard)) {
      EXPECT_EQ(recordedComparisons(function.taped, function.variableDim), 0u) << function.name << ", " << guardName(guard);
      EXPECT_EQ(codeGenerationError(function.taped, function.variableDim), "") << function.name << ", " << guardName(guard);
    }
  }

  // The tape sizes, against today's SphericalZYX root, for the real-time budget of the switch (design risk R12).
  const std::vector<PinocchioFunction> quaternionFunctions = g1().quaternionRootFunctions(NormGuard::kOnTheNorm);
  const std::vector<PinocchioFunction> eulerFunctions = g1().eulerRootFunctions();
  ASSERT_EQ(quaternionFunctions.size(), eulerFunctions.size());
  for (size_t i = 0; i < quaternionFunctions.size(); ++i) {
    const TapeSize quaternionSize = tape(quaternionFunctions[i].taped, quaternionFunctions[i].variableDim, /*recordCompare=*/true);
    const TapeSize eulerSize = tape(eulerFunctions[i].taped, eulerFunctions[i].variableDim, /*recordCompare=*/true);
    EXPECT_EQ(recordedComparisons(eulerFunctions[i].taped, eulerFunctions[i].variableDim), 0u) << eulerFunctions[i].name;
    LOG(INFO) << "[tape] " << quaternionFunctions[i].name << ": Translation+Spherical " << quaternionSize.operations << " ops ("
              << quaternionSize.optimizedOperations << " optimized, " << quaternionSize.optimizedVariables << " variables), SphericalZYX "
              << eulerSize.operations << " ops (" << eulerSize.optimizedOperations << " optimized, " << eulerSize.optimizedVariables
              << " variables)";
  }
}

TEST(QuaternionRootJointCppAd, GeneratedCentroidalMapAndFramesMatchPinocchio) {
  const PinocchioFunction function = g1().quaternionRootFunction("centroidal_map_and_frames");
  const std::unique_ptr<CppAdInterface> library = generate(function, CppAdInterface::ApproximationOrder::Zero);
  expectGeneratedMatchesPinocchio(function, *library);
}

TEST(QuaternionRootJointCppAd, GeneratedCrbaMatchesPinocchio) {
  const PinocchioFunction function = g1().quaternionRootFunction("crba");
  const std::unique_ptr<CppAdInterface> library = generate(function, CppAdInterface::ApproximationOrder::Zero);
  expectGeneratedMatchesPinocchio(function, *library);
}

TEST(QuaternionRootJointCppAd, GeneratedNonLinearEffectsMatchPinocchio) {
  const PinocchioFunction function = g1().quaternionRootFunction("non_linear_effects");
  const std::unique_ptr<CppAdInterface> library = generate(function, CppAdInterface::ApproximationOrder::Zero);
  expectGeneratedMatchesPinocchio(function, *library);
}

TEST(QuaternionRootJointCppAd, GeneratedCcrbaMatchesPinocchio) {
  const PinocchioFunction function = g1().quaternionRootFunction("ccrba");
  const std::unique_ptr<CppAdInterface> library = generate(function, CppAdInterface::ApproximationOrder::Zero);
  expectGeneratedMatchesPinocchio(function, *library);
}

TEST(QuaternionRootJointCppAd, GeneratedMomentumInversePathAndItsJacobianMatchPinocchio) {
  for (const NormGuard guard : {NormGuard::kOnTheNorm, NormGuard::kOnTheSquaredNorm}) {
    const PinocchioFunction function = g1().quaternionRootFunction("momentum_inverse_path", guard);
    const CppAdInterface& library = momentumPathLibrary(guard);
    const Eigen::Index start = function.quaternionStart;
    std::mt19937 generator(29);
    for (const Attitude& attitude : testAttitudes()) {
      const std::string context = absl::StrCat(guardName(guard), ", ", attitude.label);
      const vector_t x = withQuaternion(randomInput(function, generator), start, attitude.coefficients);
      const vector_t expected = function.reference(x);
      ASSERT_TRUE(expected.allFinite()) << context;

      // The momentum the base velocity was solved for is reproduced by the centroidal map: A_g [v_b; qd_j] = m h.
      {
        const pinocchio::Model& model = g1().model();
        pinocchio::Data data(model);
        const vector_t q = safelyNormalizedConfiguration(vector_t(x.segment(kMomentumDim, model.nq)));
        vector_t velocity(model.nv);
        velocity << library.getFunctionValue(x).head<6>(), x.tail(g1().numJoints());
        const vector6_t momentum = pinocchio::computeCentroidalMap(model, data, q) * velocity;
        EXPECT_LE((momentum - g1().mass() * x.head<kMomentumDim>()).norm(), 1.0e-9 * g1().mass()) << context;
      }

      // Values: v_b sees the quaternion only through its normalization, and the quaternion row is 1/2 G(xi) w_B with
      // the raw xi, so it scales with xi.
      for (const scalar_t scale : {1.0, -1.0, 0.5, 2.0}) {
        const vector_t generated = library.getFunctionValue(withQuaternion(x, start, scale * attitude.coefficients));
        ASSERT_TRUE(generated.allFinite()) << context << ", quaternion scaled by " << scale;
        vector_t scaledExpected = expected;
        scaledExpected.tail<4>() *= scale;
        EXPECT_LE(relativeError(generated, scaledExpected), kValueTolerance) << context << ", quaternion scaled by " << scale;
      }

      // The Jacobian against central differences of double-precision Pinocchio, and the radial identities of design
      // decision D5: d(v_b)/d(xi) xi = 0 (degree zero), d(xi_dot)/d(xi) xi = xi_dot (degree one).
      const matrix_t jacobian = library.getJacobian(x);
      ASSERT_TRUE(jacobian.allFinite()) << context;
      EXPECT_LE(relativeError(jacobian, centralDifferenceJacobian(function.reference, x)), kJacobianTolerance) << context;
      const vector_t radial = jacobian.middleCols<4>(start) * attitude.coefficients;
      EXPECT_LE(radial.head<6>().cwiseAbs().maxCoeff(), kValueTolerance * std::max(1.0, expected.head<6>().cwiseAbs().maxCoeff()))
          << context;
      EXPECT_LE((radial.tail<4>() - expected.tail<4>()).cwiseAbs().maxCoeff(),
                kValueTolerance * std::max(1.0, expected.cwiseAbs().maxCoeff()))
          << context;
    }

    // A zero quaternion: the base velocity of the identity attitude, and a zero quaternion rate.
    const vector_t x = randomInput(function, generator);
    const vector_t atZero = library.getFunctionValue(withQuaternion(x, start, vector4_t::Zero()));
    ASSERT_TRUE(atZero.allFinite()) << guardName(guard);
    EXPECT_LE(relativeError(vector_t(atZero.head<6>()), vector_t(function.reference(x).head<6>())), kValueTolerance) << guardName(guard);
    EXPECT_LE(atZero.tail<4>().cwiseAbs().maxCoeff(), 1.0e-15) << guardName(guard);
  }
}

TEST(QuaternionRootJointCppAd, GeneratedFrameVelocityAndAccelerationAndTheirJacobianMatchPinocchio) {
  // The whole-body end-effector terms tape forwardKinematics with velocities and accelerations: the composite root's
  // calc(q, v) and its bias terms, which the configuration-only functions above never reach.
  const PinocchioFunction function = g1().quaternionRootFunction("frame_velocity_and_acceleration", NormGuard::kOnTheSquaredNorm);
  const std::unique_ptr<CppAdInterface> library = generate(function, CppAdInterface::ApproximationOrder::First);
  expectGeneratedMatchesPinocchio(function, *library);

  std::mt19937 generator(37);
  for (const Attitude& attitude : testAttitudes()) {
    const vector_t x = withQuaternion(randomInput(function, generator), function.quaternionStart, attitude.coefficients);
    const matrix_t jacobian = library->getJacobian(x);
    ASSERT_TRUE(jacobian.allFinite()) << attitude.label;
    EXPECT_LE(relativeError(jacobian, centralDifferenceJacobian(function.reference, x)), kJacobianTolerance) << attitude.label;
    // The frame motion sees the quaternion only through its normalization: no radial derivative.
    const vector_t radial = jacobian.middleCols<4>(function.quaternionStart) * attitude.coefficients;
    EXPECT_LE(radial.cwiseAbs().maxCoeff(), kValueTolerance * std::max(1.0, function.reference(x).cwiseAbs().maxCoeff())) << attitude.label;
  }
  const vector_t x = randomInput(function, generator);
  EXPECT_TRUE(library->getJacobian(withQuaternion(x, function.quaternionStart, vector4_t::Zero())).allFinite());
}

TEST(QuaternionRootJointCppAd, GeneratedSecondOrderThroughTheRootIsConsistentWithItsJacobian) {
  // Second order, as OCS2's former StateInputCostCppAd / StateCostCppAd generated their libraries: c = 1/2 |frame motion|^2,
  // taped as a function of the root's quaternion, body angular velocity and body angular acceleration, z = [xi(4), w_B(3),
  // wd_B(3)], with every other entry of [q, v, a] a constant on the tape. The Hessian stays small, but every derivative
  // runs through the composite root's computations.
  const PinocchioFunction frames = g1().quaternionRootFunction("frame_velocity_and_acceleration", NormGuard::kOnTheSquaredNorm);
  const Eigen::Index nq = g1().model().nq;
  const Eigen::Index nv = g1().model().nv;
  const Eigen::Index angularVelocityStart = nq + 3;
  const Eigen::Index angularAccelerationStart = nq + nv + 3;
  constexpr Eigen::Index kRootInputDim = 10;
  std::mt19937 generator(41);
  const vector_t sample = randomInput(frames, generator);

  const CppAdInterface::ad_function_t taped = [frames, sample, angularVelocityStart, angularAccelerationStart](const ad_vector_t& z,
                                                                                                               ad_vector_t& y) {
    ad_vector_t x = sample.cast<ad_scalar_t>();
    x.segment<4>(kQuaternionStart) = z.head<4>();
    x.segment<3>(angularVelocityStart) = z.segment<3>(4);
    x.segment<3>(angularAccelerationStart) = z.tail<3>();
    ad_vector_t motion;
    frames.taped(x, motion);
    y.resize(1);
    y(0) = ad_scalar_t(0.5) * motion.squaredNorm();
  };
  const ReferenceFunction reference = [frames, sample, angularVelocityStart, angularAccelerationStart](const vector_t& z) {
    vector_t x = sample;
    x.segment<4>(kQuaternionStart) = z.head<4>();
    x.segment<3>(angularVelocityStart) = z.segment<3>(4);
    x.segment<3>(angularAccelerationStart) = z.tail<3>();
    vector_t y(1);
    y(0) = 0.5 * frames.reference(x).squaredNorm();
    return y;
  };
  EXPECT_EQ(recordedComparisons(taped, kRootInputDim), 0u);
  const PinocchioFunction rootCost{
      .name = "root_frame_motion_cost",
      .variableDim = kRootInputDim,
      .quaternionStart = 0,
      .taped = taped,
      .reference = reference,
  };
  const std::unique_ptr<CppAdInterface> library = generate(rootCost, CppAdInterface::ApproximationOrder::Second);

  for (const Attitude& attitude : testAttitudes()) {
    vector_t z = uniformVector(kRootInputDim, /*halfWidth=*/0.8, generator);
    z.head<4>() = attitude.coefficients;
    const scalar_t expected = reference(z)(0);
    EXPECT_LE(std::abs(library->getFunctionValue(z)(0) - expected), kValueTolerance * std::max(1.0, std::abs(expected))) << attitude.label;
    const matrix_t gradient = library->getJacobian(z);
    EXPECT_LE(relativeError(gradient, centralDifferenceJacobian(reference, z)), kJacobianTolerance) << attitude.label;

    // The generated Hessian against central differences of the generated gradient, and its symmetry.
    const matrix_t hessian = library->getHessian(/*outputIndex=*/0, z);
    ASSERT_TRUE(hessian.allFinite()) << attitude.label;
    const ReferenceFunction generatedGradient = [&library](const vector_t& point) {
      return vector_t(library->getJacobian(point).transpose());
    };
    EXPECT_LE(relativeError(hessian, centralDifferenceJacobian(generatedGradient, z)), kJacobianTolerance) << attitude.label;
    EXPECT_LE(relativeError(hessian, matrix_t(hessian.transpose())), kValueTolerance) << attitude.label;
  }
  vector_t atZero = uniformVector(kRootInputDim, /*halfWidth=*/0.8, generator);
  atZero.head<4>().setZero();
  EXPECT_TRUE(library->getHessian(/*outputIndex=*/0, atZero).allFinite());
}

/** The output of a normalization probe, which decides how CppADCodeGen differentiates it. */
enum class ProbeOutput {
  // [xi_hat, R(xi_hat)]: 13 outputs of 4 inputs, differentiated in forward mode.
  kWide,
  // A weighted sum of those 13 entries: one output, differentiated in reverse mode, as every cost is.
  kScalar,
};

/** x = xi -> [xi_hat, R(xi_hat)] (or their weighted sum) with `guard`, generated with its Jacobian. */
std::unique_ptr<CppAdInterface> generateNormalization(NormGuard guard, ProbeOutput output) {
  const CppAdInterface::ad_function_t taped = [guard, output](const ad_vector_t& x, ad_vector_t& y) {
    const ad_vector4_t normalized = safelyNormalizedQuaternion(ad_vector4_t(x.head<4>()), guard);
    const ad_matrix3_t rotation = ad_quaternion_t(normalized).toRotationMatrix();
    ad_vector_t entries(13);
    entries << normalized, Eigen::Map<const ad_vector_t>(rotation.data(), /*size=*/9);
    if (output == ProbeOutput::kWide) {
      y = entries;
      return;
    }
    y.resize(1);
    y(0) = ad_scalar_t(0.0);
    for (Eigen::Index i = 0; i < entries.size(); ++i) y(0) += ad_scalar_t(1.0 + 0.1 * static_cast<scalar_t>(i)) * entries(i);
  };
  const PinocchioFunction function{
      .name = absl::StrCat("normalization_", guardName(guard), output == ProbeOutput::kWide ? "_wide" : "_scalar"),
      .variableDim = 4,
      .quaternionStart = 0,
      .taped = taped,
      .reference = ReferenceFunction(),
  };
  return generate(function, CppAdInterface::ApproximationOrder::First);
}

TEST(QuaternionRootJointCppAd, SafeNormalizationHasTheDerivativeOfTheNormalizationAwayFromZero) {
  std::mt19937 generator(31);
  for (const NormGuard guard : {NormGuard::kOnTheNorm, NormGuard::kOnTheSquaredNorm}) {
    const std::unique_ptr<CppAdInterface> library = generateNormalization(guard, ProbeOutput::kWide);
    for (int sample = 0; sample < 20; ++sample) {
      const vector4_t xi = uniformVector(/*size=*/4, /*halfWidth=*/2.0, generator);
      const scalar_t norm = xi.norm();
      const vector4_t xiHat = xi / norm;
      const vector_t value = library->getFunctionValue(xi);
      EXPECT_LE((value.head<4>() - xiHat).norm(), 1.0e-12) << guardName(guard);
      const matrix3_t rotation = quaternion_t(xiHat).toRotationMatrix();
      EXPECT_LE((value.tail<9>() - Eigen::Map<const vector_t>(rotation.data(), /*size=*/9)).norm(), 1.0e-12) << guardName(guard);
      // d(xi / |xi|) / d(xi) = (I - xi_hat xi_hat^T) / |xi|.
      const matrix_t jacobian = library->getJacobian(xi);
      const matrix4_t expected = (matrix4_t::Identity() - xiHat * xiHat.transpose()) / norm;
      EXPECT_LE((jacobian.topRows<4>() - expected).norm(), 1.0e-10) << guardName(guard);
    }
  }
}

TEST(QuaternionRootJointCppAd, SquaredNormGuardIsDifferentiableAtAZeroQuaternion) {
  // A zero quaternion has the identity rotation under both guards (the tests above). Only the squared-norm guard also
  // gives it a finite Jacobian in reverse mode, the mode of every cost: there the partial of |xi| from the branch the
  // conditional expression does not select is 0, and the square root's reverse sweep multiplies it by 1 / (2 |xi|) =
  // inf, which CppADCodeGen generates as a plain product (its azmul is not an absolute zero for a variable), so
  // 0 * inf = NaN. The squared-norm guard takes the square root after the selection, of 1.
  const vector4_t zero = vector4_t::Zero();
  {
    const std::unique_ptr<CppAdInterface> wide = generateNormalization(NormGuard::kOnTheSquaredNorm, ProbeOutput::kWide);
    const vector_t value = wide->getFunctionValue(zero);
    const matrix3_t identity = matrix3_t::Identity();
    EXPECT_LE(value.head<4>().norm(), 0.0);
    EXPECT_LE((value.tail<9>() - Eigen::Map<const vector_t>(identity.data(), /*size=*/9)).norm(), 0.0);
    // At a zero quaternion n_safe = 1 is a constant, so d(xi_hat)/d(xi) = I, and R(xi_hat) is quadratic in xi_hat.
    matrix_t expected = matrix_t::Zero(13, 4);
    expected.topRows<4>().setIdentity();
    const matrix_t jacobian = wide->getJacobian(zero);
    EXPECT_LE((jacobian - expected).norm(), 1.0e-15) << jacobian;

    const std::unique_ptr<CppAdInterface> scalar = generateNormalization(NormGuard::kOnTheSquaredNorm, ProbeOutput::kScalar);
    const matrix_t gradient = scalar->getJacobian(zero);
    ASSERT_TRUE(gradient.allFinite()) << gradient;
    EXPECT_LE((gradient.transpose() - vector4_t(1.0, 1.1, 1.2, 1.3)).norm(), 1.0e-12) << gradient;
  }

  // The momentum path: finite, the v_b rows independent of xi at zero (R(xi_hat) is quadratic in xi_hat), and their
  // other columns those of the identity attitude.
  const PinocchioFunction function = g1().quaternionRootFunction("momentum_inverse_path", NormGuard::kOnTheSquaredNorm);
  const CppAdInterface& library = momentumPathLibrary(NormGuard::kOnTheSquaredNorm);
  const Eigen::Index start = function.quaternionStart;
  std::mt19937 generator(37);
  const vector_t x = randomInput(function, generator);
  const matrix_t atZero = library.getJacobian(withQuaternion(x, start, zero));
  ASSERT_TRUE(atZero.allFinite()) << atZero;
  const matrix_t atIdentity = library.getJacobian(withQuaternion(x, start, vector4_t(0.0, 0.0, 0.0, 1.0)));
  const matrix_t baseRowsAtZero = atZero.topRows<6>();
  EXPECT_LE(baseRowsAtZero.middleCols<4>(start).cwiseAbs().maxCoeff(), 1.0e-12);
  const matrix_t baseRowsAtIdentity = atIdentity.topRows<6>();
  EXPECT_LE(relativeError(matrix_t(baseRowsAtZero.leftCols(start)), matrix_t(baseRowsAtIdentity.leftCols(start))), kValueTolerance);
  const Eigen::Index tail = function.variableDim - start - 4;
  EXPECT_LE(relativeError(matrix_t(baseRowsAtZero.rightCols(tail)), matrix_t(baseRowsAtIdentity.rightCols(tail))), kValueTolerance);

#ifdef NDEBUG
  // Positive control, with assertions off (CppAdInterface::getJacobian asserts a finite Jacobian otherwise): the norm
  // guard of design section 2.6 as written has no finite reverse-mode Jacobian at a zero quaternion, in the probe and
  // in the momentum path, while its forward-mode Jacobian (the wide probe) is finite - the mode, not the guard alone,
  // decides, so a forward-mode test of the norm guard proves nothing.
  EXPECT_FALSE(generateNormalization(NormGuard::kOnTheNorm, ProbeOutput::kScalar)->getJacobian(zero).allFinite());
  EXPECT_TRUE(generateNormalization(NormGuard::kOnTheNorm, ProbeOutput::kWide)->getJacobian(zero).allFinite());
  EXPECT_FALSE(momentumPathLibrary(NormGuard::kOnTheNorm).getJacobian(withQuaternion(x, start, zero)).allFinite());
#endif
}

}  // namespace
}  // namespace ocs2::humanoid
