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

/**
 * End-to-end integration test of the basis-vector contact input formulation through CentroidalMpcInterface.
 *
 * Two full interfaces are built from the DRC Atlas configuration, one with useContactBasisVectorInputs: true and one
 * with false. Constructing an interface JIT-compiles all CppAD models, which takes minutes, so each mode is built at
 * most once per test process and shared by all test cases (see BasisInputsFormulationTest::holder).
 */

#include <gtest/gtest.h>

#include <pinocchio/fwd.hpp>  // forward declarations must be included first.

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>

#include <ocs2_core/PreComputation.h>
#include <ocs2_core/cost/QuadraticStateInputCost.h>
#include <ocs2_core/misc/LoadData.h>

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "humanoid_centroidal_mpc/CentroidalMpcInterface.h"
#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsAD.h"
#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsBasisInputsAD.h"
#include "humanoid_common_mpc/common/BasisInputsCostTransform.h"
#include "humanoid_common_mpc/common/BasisInputsModelDecorator.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc/constraint/ZeroWrenchConstraint.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid {
namespace {

constexpr size_t kWrenchDimPerContact = 6;
// Centroidal state layout: [h(6), p_base(3), euler_zyx(3), q_joints]. Used to sanity check the model accessors.
constexpr Eigen::Index kBaseEulerZyxStartIndex = 9;
constexpr Eigen::Index kJointAnglesStartIndex = 12;

/******************************************************************************************************/
/*                                       File location helpers                                        */
/******************************************************************************************************/

struct AtlasFiles {
  std::string taskFile;
  std::string urdfFile;
  std::string referenceFile;
};

std::optional<std::string> firstExistingPath(const std::vector<std::string>& candidates) {
  for (const auto& candidate : candidates) {
    std::error_code ec;
    if (std::filesystem::exists(candidate, ec)) {
      return candidate;
    }
  }
  return std::nullopt;
}

/**
 * Locates the DRC Atlas task/URDF/reference files.
 *
 * Under `bazel test` the data files live in the runfiles tree, which is also the working directory, so those copies
 * are preferred: they are symlinks into the checkout and therefore always current. The ament index (populated by
 * setup_env.sh from a *copy* of the source tree) is only a fallback because that copy can be stale.
 */
AtlasFiles locateAtlasFiles() {
  const std::string taskRel = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/mpc/task.yaml";
  const std::string referenceRel = "robot_models/drc_atlas/drc_atlas_centroidal_mpc/config/command/reference.yaml";
  const std::string urdfRel = "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf";

  std::vector<std::string> roots;
  if (const char* srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::string(srcDir) + "/_main");
    roots.emplace_back(std::string(srcDir) + "/wb_humanoid_mpc");
  }
  roots.emplace_back(std::filesystem::current_path().string());
  roots.emplace_back("/wb_humanoid_mpc_ws/workspace/wb_humanoid_mpc");

  auto resolve = [&roots](const std::string& relativePath) {
    std::vector<std::string> candidates;
    for (const auto& root : roots) {
      candidates.emplace_back(root + "/" + relativePath);
    }
    return firstExistingPath(candidates);
  };

  const auto taskFile = resolve(taskRel);
  const auto referenceFile = resolve(referenceRel);
  const auto urdfFile = resolve(urdfRel);
  if (taskFile && referenceFile && urdfFile) {
    return AtlasFiles{*taskFile, *urdfFile, *referenceFile};
  }

  // Fallback: installed / ament-indexed packages (requires AMENT_PREFIX_PATH, which .bazelrc forwards to tests).
  const std::string configDir = ament_index_cpp::get_package_share_directory("drc_atlas_centroidal_mpc");
  const std::string descriptionDir = ament_index_cpp::get_package_share_directory("drc_atlas_description");
  return AtlasFiles{configDir + "/config/mpc/task.yaml", descriptionDir + "/urdf/atlas.urdf", configDir + "/config/command/reference.yaml"};
}

/******************************************************************************************************/
/*                                        Task file rewriting                                         */
/******************************************************************************************************/

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("[testBasisInputsFormulation] Cannot read file: " + path);
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

/**
 * Replaces the value of every `<indent>key: value  # comment` line with `<indent>key: newValue` and returns the number
 * of lines that were rewritten. The key must match a whole YAML key (e.g. "verbose" does not match "verboseCppAd:").
 */
size_t replaceYamlScalar(std::string& content, const std::string& key, const std::string& newValue) {
  const std::regex lineRegex("(^|\\n)([ \\t]*" + key + ":)[ \\t]*[^\\n]*");
  const size_t numMatches = std::distance(std::sregex_iterator(content.begin(), content.end(), lineRegex), std::sregex_iterator());
  content = std::regex_replace(content, lineRegex, "$1$2 " + newValue);
  return numMatches;
}

/**
 * Writes a copy of the Atlas task file for the requested input formulation into `dir`.
 *
 * The runfiles tree is not guaranteed to be writable and the CppAD folder in the task file is a relative path, so the
 * copy redirects CppAD code generation into a mode-specific temp folder. Each mode gets its own folder because the two
 * modes generate CppAD models with the same names but different input dimensions; forcing recompilation guards against
 * ever loading a stale library.
 */
std::string writeTaskFileForMode(const AtlasFiles& files, bool useBasisInputs, const std::filesystem::path& dir) {
  const std::string modeName = useBasisInputs ? "basis" : "wrench";
  const std::filesystem::path cppAdFolder = dir / ("cppad_autocode_gen_" + modeName);

  std::string content = readFile(files.taskFile);
  auto replaceExactlyOnce = [&content](const std::string& key, const std::string& value) {
    const size_t numReplaced = replaceYamlScalar(content, key, value);
    if (numReplaced != 1) {
      throw std::runtime_error("[testBasisInputsFormulation] Expected exactly one '" + key + ":' line in the task file, found " +
                               std::to_string(numReplaced));
    }
  };
  replaceExactlyOnce("useContactBasisVectorInputs", useBasisInputs ? "true" : "false");
  replaceExactlyOnce("modelFolderCppAd", cppAdFolder.string());
  replaceExactlyOnce("recompileLibrariesCppAd", "true");

  const std::filesystem::path taskFile = dir / ("task_" + modeName + ".yaml");
  std::ofstream out(taskFile);
  if (!out) {
    throw std::runtime_error("[testBasisInputsFormulation] Cannot write file: " + taskFile.string());
  }
  out << content;
  return taskFile.string();
}

/******************************************************************************************************/
/*                                         Numerical helpers                                          */
/******************************************************************************************************/

/** Central finite-difference Jacobian of f around v. */
template <typename Func>
matrix_t centralDifferenceJacobian(Func&& f, const vector_t& v, size_t outputDim, scalar_t eps) {
  matrix_t jacobian(outputDim, v.size());
  for (Eigen::Index j = 0; j < v.size(); ++j) {
    vector_t vPlus = v;
    vector_t vMinus = v;
    vPlus(j) += eps;
    vMinus(j) -= eps;
    jacobian.col(j) = (f(vPlus) - f(vMinus)) / (2.0 * eps);
  }
  return jacobian;
}

/** max_ij |A_ij - B_ij| / max(1, |B_ij|): absolute error for small entries, relative error for large ones. */
scalar_t maxNormalizedError(const matrix_t& A, const matrix_t& B) {
  return ((A - B).cwiseAbs().array() / B.cwiseAbs().array().max(1.0)).maxCoeff();
}

/** Initial state with a rotated base (yaw 0.7, pitch 0.1, roll -0.05) and slightly perturbed joint angles. */
vector_t makeRotatedTestState(const CentroidalMpcInterface& interface, std::mt19937& gen) {
  const MpcRobotModelBase<scalar_t>& model = interface.getEffectiveMpcRobotModel();
  vector_t x = interface.getInitialState();
  model.setBaseOrientationEulerZYX(x, VECTOR3_T<scalar_t>(0.7, 0.1, -0.05));

  std::uniform_real_distribution<scalar_t> jointPerturbation(-0.05, 0.05);
  vector_t jointAngles = model.getJointAngles(x);
  for (Eigen::Index i = 0; i < jointAngles.size(); ++i) {
    jointAngles(i) += jointPerturbation(gen);
  }
  model.setJointAngles(x, jointAngles);
  return x;
}

/** Basis-vector input with λ ~ U[0, 50] (non-negative, inside the cone) and joint velocities ~ U[-1, 1]. */
vector_t makeRandomBasisInput(size_t numBasisInputs, size_t jointDim, std::mt19937& gen) {
  std::uniform_real_distribution<scalar_t> lambdaDist(0.0, 50.0);
  std::uniform_real_distribution<scalar_t> jointVelocityDist(-1.0, 1.0);
  vector_t u(numBasisInputs + jointDim);
  for (size_t i = 0; i < numBasisInputs; ++i) {
    u(i) = lambdaDist(gen);
  }
  for (size_t i = numBasisInputs; i < numBasisInputs + jointDim; ++i) {
    u(i) = jointVelocityDist(gen);
  }
  return u;
}

/******************************************************************************************************/
/*                                            Test fixture                                            */
/******************************************************************************************************/

class BasisInputsFormulationTest : public ::testing::Test {
 protected:
  struct InterfaceHolder {
    std::unique_ptr<CentroidalMpcInterface> interface;
    std::string taskFile;
    std::string error;
    bool attempted = false;
  };

  static void SetUpTestSuite() {
    tmpDir_ = std::filesystem::path(testing::TempDir()) / ("basis_inputs_formulation_" + std::to_string(::getpid()));
    std::filesystem::create_directories(tmpDir_);
    files_ = locateAtlasFiles();
  }

  static void TearDownTestSuite() {
    // Release the interfaces (and the CppAD libraries they dlopen'ed) before deleting the generated files.
    basisHolder_ = InterfaceHolder{};
    wrenchHolder_ = InterfaceHolder{};
    std::error_code ec;
    std::filesystem::remove_all(tmpDir_, ec);
  }

  /**
   * Lazily constructs the interface for the requested mode exactly once per process. Construction errors are stored
   * instead of thrown so that every test reports the same actionable message via ASSERT on `interface`.
   */
  static InterfaceHolder& holder(bool useBasisInputs) {
    InterfaceHolder& h = useBasisInputs ? basisHolder_ : wrenchHolder_;
    if (h.attempted) {
      return h;
    }
    h.attempted = true;
    try {
      h.taskFile = writeTaskFileForMode(files_, useBasisInputs, tmpDir_);
      auto statusOr = CentroidalMpcInterface::Create(h.taskFile, files_.urdfFile, files_.referenceFile);
      if (!statusOr.ok()) {
        h.error = std::string(statusOr.status().message());
      } else {
        h.interface = std::move(*statusOr);
      }
    } catch (const std::exception& e) {
      h.error = e.what();
    }
    return h;
  }

  static CentroidalDynamicsBasisInputsAD* basisDynamics(const CentroidalMpcInterface& interface) {
    return dynamic_cast<CentroidalDynamicsBasisInputsAD*>(interface.getOptimalControlProblem().dynamicsPtr.get());
  }

  static CentroidalDynamicsAD* wrenchDynamics(const CentroidalMpcInterface& interface) {
    return dynamic_cast<CentroidalDynamicsAD*>(interface.getOptimalControlProblem().dynamicsPtr.get());
  }

  static std::filesystem::path tmpDir_;
  static AtlasFiles files_;
  static InterfaceHolder basisHolder_;
  static InterfaceHolder wrenchHolder_;
};

std::filesystem::path BasisInputsFormulationTest::tmpDir_;
AtlasFiles BasisInputsFormulationTest::files_;
BasisInputsFormulationTest::InterfaceHolder BasisInputsFormulationTest::basisHolder_;
BasisInputsFormulationTest::InterfaceHolder BasisInputsFormulationTest::wrenchHolder_;

/******************************************************************************************************/
/* (1) Dimensions and wiring of the basis-vector formulation                                          */
/******************************************************************************************************/
TEST_F(BasisInputsFormulationTest, BasisMode_Dimensions) {
  InterfaceHolder& h = holder(/*useBasisInputs=*/true);
  ASSERT_TRUE(h.interface) << "Failed to construct the basis-mode interface: " << h.error;
  const CentroidalMpcInterface& interface = *h.interface;

  EXPECT_TRUE(interface.usesContactBasisVectorInputs());

  const BasisInputsModelDecorator<scalar_t>* decorator = interface.getBasisDecoratorPtr();
  ASSERT_NE(decorator, nullptr);
  EXPECT_NE(dynamic_cast<const BasisInputsModelDecorator<scalar_t>*>(&interface.getEffectiveMpcRobotModel()), nullptr);
  EXPECT_NE(dynamic_cast<const BasisInputsModelDecorator<ad_scalar_t>*>(&interface.getEffectiveMpcRobotModelAD()), nullptr);

  // numBasisPerFoot = numBasisVectors (friction pyramid) + 1 normal + 4 CoP corner + 2 torsional rays.
  size_t numBasisVectorsFromYaml = 0;
  loadData::loadCppDataType(h.taskFile, "contacts.contactWrenchConeSoftConstraint.numBasisVectors", numBasisVectorsFromYaml);
  const size_t numBasisPerFoot = decorator->getNumBasisPerFoot();
  EXPECT_EQ(numBasisPerFoot, numBasisVectorsFromYaml + 7);

  const size_t jointDim = interface.modelSettings().mpc_joint_dim;
  const size_t wrenchInputDim = kWrenchDimPerContact * N_CONTACTS + jointDim;
  const size_t basisInputDim = N_CONTACTS * numBasisPerFoot + jointDim;

  EXPECT_EQ(interface.getEffectiveMpcRobotModel().getInputDim(), basisInputDim);
  EXPECT_EQ(interface.getEffectiveMpcRobotModelAD().getInputDim(), basisInputDim);
  EXPECT_EQ(decorator->getWrenchInputDim(), wrenchInputDim);
  // The wrench-space model stays untouched so that consumers of physical wrenches keep working.
  EXPECT_EQ(interface.getMpcRobotModel().getInputDim(), wrenchInputDim);
  EXPECT_EQ(interface.getWrenchInputDim(), wrenchInputDim);
  EXPECT_EQ(interface.getWrenchInputDim(), static_cast<size_t>(interface.getCentroidalModelInfo().inputDim));
  EXPECT_EQ(interface.getNumBasisInputs(), N_CONTACTS * numBasisPerFoot);
  EXPECT_EQ(interface.getEffectiveMpcRobotModel().getJointVelocitiesStartindex(), N_CONTACTS * numBasisPerFoot);

  // M = blkdiag(B_0, B_1, I_joints) of size wrenchInputDim x basisInputDim.
  ASSERT_TRUE(interface.getBasisToWrenchMap().has_value());
  const matrix_t& M = *interface.getBasisToWrenchMap();
  ASSERT_EQ(static_cast<size_t>(M.rows()), interface.getWrenchInputDim());
  ASSERT_EQ(static_cast<size_t>(M.cols()), basisInputDim);
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    const matrix_t& B = decorator->getBasisMatrix(i);
    ASSERT_EQ(static_cast<size_t>(B.rows()), kWrenchDimPerContact);
    ASSERT_EQ(static_cast<size_t>(B.cols()), numBasisPerFoot);
    EXPECT_DOUBLE_EQ((M.block(kWrenchDimPerContact * i, numBasisPerFoot * i, kWrenchDimPerContact, numBasisPerFoot) - B).norm(), 0.0);
  }
  EXPECT_DOUBLE_EQ((M.bottomRightCorner(jointDim, jointDim) - matrix_t::Identity(jointDim, jointDim)).norm(), 0.0);
  EXPECT_DOUBLE_EQ(M.topRightCorner(kWrenchDimPerContact * N_CONTACTS, jointDim).norm(), 0.0);
  EXPECT_DOUBLE_EQ(M.bottomLeftCorner(jointDim, N_CONTACTS * numBasisPerFoot).norm(), 0.0);

  // The OCP uses the basis-input dynamics with a matching input dimension.
  CentroidalDynamicsBasisInputsAD* dynamics = basisDynamics(interface);
  ASSERT_NE(dynamics, nullptr);
  EXPECT_EQ(dynamics->getBasisInputDim(), basisInputDim);
  EXPECT_EQ(dynamics->getNumBasisPerFoot(), numBasisPerFoot);
  EXPECT_EQ(wrenchDynamics(interface), nullptr);

  const auto config = interface.getBasisInputsCostTransformConfig();
  ASSERT_TRUE(config.has_value());
  EXPECT_EQ(config->wrenchInputDim, wrenchInputDim);
  EXPECT_EQ(config->numBasisInputs, N_CONTACTS * numBasisPerFoot);
  EXPECT_EQ(config->basisInputDim(), basisInputDim);
  EXPECT_DOUBLE_EQ(config->lambdaRegularization, interface.getBasisScalingRegularization());
  EXPECT_DOUBLE_EQ((config->basisToWrenchMap - M).norm(), 0.0);
}

/******************************************************************************************************/
/* (2) Basis dynamics == wrench dynamics evaluated on the world-frame-rotated wrench                   */
/******************************************************************************************************/
TEST_F(BasisInputsFormulationTest, BasisDynamics_MatchesWrenchDynamicsWithRotatedWrench) {
  InterfaceHolder& hBasis = holder(/*useBasisInputs=*/true);
  ASSERT_TRUE(hBasis.interface) << "Failed to construct the basis-mode interface: " << hBasis.error;
  InterfaceHolder& hWrench = holder(/*useBasisInputs=*/false);
  ASSERT_TRUE(hWrench.interface) << "Failed to construct the wrench-mode interface: " << hWrench.error;
  CentroidalMpcInterface& basisInterface = *hBasis.interface;
  const CentroidalMpcInterface& wrenchInterface = *hWrench.interface;

  CentroidalDynamicsBasisInputsAD* basisDyn = basisDynamics(basisInterface);
  ASSERT_NE(basisDyn, nullptr);
  CentroidalDynamicsAD* wrenchDyn = wrenchDynamics(wrenchInterface);
  ASSERT_NE(wrenchDyn, nullptr);
  // Both interfaces are built from the same URDF/reference, so the underlying centroidal model is identical.
  ASSERT_EQ(basisInterface.getCentroidalModelInfo().stateDim, wrenchInterface.getCentroidalModelInfo().stateDim);
  ASSERT_EQ(basisInterface.getCentroidalModelInfo().inputDim, wrenchInterface.getCentroidalModelInfo().inputDim);

  const BasisInputsModelDecorator<scalar_t>& decorator = *basisInterface.getBasisDecoratorPtr();
  const MpcRobotModelBase<scalar_t>& effectiveModel = basisInterface.getEffectiveMpcRobotModel();
  const size_t numBasisPerFoot = decorator.getNumBasisPerFoot();
  const size_t jointDim = basisInterface.modelSettings().mpc_joint_dim;
  const size_t wrenchInputDim = basisInterface.getWrenchInputDim();

  std::mt19937 gen(42);
  const vector_t x = makeRotatedTestState(basisInterface, gen);
  const vector_t uBasis = makeRandomBasisInput(N_CONTACTS * numBasisPerFoot, jointDim, gen);
  ASSERT_EQ(static_cast<size_t>(uBasis.size()), effectiveModel.getInputDim());
  // Sanity check of the assumed state layout (euler ZYX at 9..11, joints from 12).
  EXPECT_DOUBLE_EQ(x(kBaseEulerZyxStartIndex), 0.7);
  EXPECT_DOUBLE_EQ(x(kBaseEulerZyxStartIndex + 1), 0.1);
  EXPECT_DOUBLE_EQ(x(kBaseEulerZyxStartIndex + 2), -0.05);
  EXPECT_DOUBLE_EQ((x.tail(x.size() - kJointAnglesStartIndex) - effectiveModel.getJointAngles(x)).norm(), 0.0);

  // Expected world-frame wrench input from the tested helper, using an independently updated pinocchio copy.
  PinocchioInterface pinocchioInterface = basisInterface.getPinocchioInterface();
  const vector_t q = effectiveModel.getGeneralizedCoordinates(x);
  updateFramePlacements<scalar_t>(q, pinocchioInterface);
  const vector_t uWrench = basisDyn->toWorldFrameWrenchInput<scalar_t>(pinocchioInterface, basisInterface.getCentroidalModelInfo(), uBasis);
  ASSERT_EQ(static_cast<size_t>(uWrench.size()), wrenchInputDim);

  const matrix_t& M = *basisInterface.getBasisToWrenchMap();
  const vector_t uWrenchLocal = M * uBasis;  // naive constant map: wrenches stay in the local contact frame

  for (size_t i = 0; i < N_CONTACTS; ++i) {
    const VECTOR6_T<scalar_t> wrenchWorldFromDynamics = uWrench.segment<kWrenchDimPerContact>(kWrenchDimPerContact * i);
    const VECTOR6_T<scalar_t> wrenchWorldFromDecorator = decorator.getContactWrenchInWorldFrame(x, uBasis, i);
    const VECTOR6_T<scalar_t> wrenchLocalFromDecorator = decorator.getContactWrench(uBasis, i);
    const VECTOR6_T<scalar_t> wrenchLocalFromMap = uWrenchLocal.segment<kWrenchDimPerContact>(kWrenchDimPerContact * i);

    // The dynamics and the decorator must agree on the world-frame wrench.
    EXPECT_LE((wrenchWorldFromDynamics - wrenchWorldFromDecorator).norm(), 1e-9) << "contact " << i;
    // The input-only accessor returns the local-frame wrench B * lambda.
    EXPECT_LE((wrenchLocalFromDecorator - wrenchLocalFromMap).norm(), 1e-12) << "contact " << i;
    // Rotating back must recover the local wrench (frame rotation is consistent).
    EXPECT_LE((decorator.rotateWrenchWorldToLocal(x, wrenchWorldFromDecorator, i) - wrenchLocalFromDecorator).norm(), 1e-9)
        << "contact " << i;
    // With a rotated foot (yaw 0.7) the naive constant map differs from the rotated one in the force x/y components.
    EXPECT_GT((wrenchWorldFromDynamics.head<2>() - wrenchLocalFromMap.head<2>()).norm(), 1e-6) << "contact " << i;
    // Rotations preserve force and moment magnitudes.
    EXPECT_NEAR(wrenchWorldFromDynamics.head<3>().norm(), wrenchLocalFromMap.head<3>().norm(), 1e-9) << "contact " << i;
    EXPECT_NEAR(wrenchWorldFromDynamics.tail<3>().norm(), wrenchLocalFromMap.tail<3>().norm(), 1e-9) << "contact " << i;
  }
  // Joint velocities pass through unchanged.
  EXPECT_DOUBLE_EQ((uWrench.tail(jointDim) - uBasis.tail(jointDim)).norm(), 0.0);

  // Flow maps must agree once the wrench is expressed in the world frame.
  const PreComputation preComp;
  const vector_t fBasis = basisDyn->computeFlowMap(0.0, x, uBasis, preComp);
  const vector_t fWrench = wrenchDyn->computeFlowMap(0.0, x, uWrench, preComp);
  ASSERT_EQ(fBasis.size(), fWrench.size());
  EXPECT_LE((fBasis - fWrench).norm(), 1e-8 * (1.0 + fWrench.norm()))
      << "||f_basis - f_wrench|| = " << (fBasis - fWrench).norm() << ", ||f_wrench|| = " << fWrench.norm();

  // Feeding the un-rotated local wrench into the wrench dynamics must NOT reproduce the basis dynamics: the rotation
  // inside the basis-input tape is what makes the formulation frame-correct.
  const vector_t fNaive = wrenchDyn->computeFlowMap(0.0, x, uWrenchLocal, preComp);
  EXPECT_GT((fBasis - fNaive).norm(), 1e-6);
}

/******************************************************************************************************/
/* (3) CppAD linear approximation of the basis dynamics vs central finite differences                 */
/******************************************************************************************************/
TEST_F(BasisInputsFormulationTest, BasisDynamics_LinearApproximationMatchesFiniteDifferences) {
  InterfaceHolder& h = holder(/*useBasisInputs=*/true);
  ASSERT_TRUE(h.interface) << "Failed to construct the basis-mode interface: " << h.error;
  const CentroidalMpcInterface& interface = *h.interface;

  CentroidalDynamicsBasisInputsAD* dynamics = basisDynamics(interface);
  ASSERT_NE(dynamics, nullptr);

  const size_t stateDim = interface.getEffectiveMpcRobotModel().getStateDim();
  const size_t inputDim = interface.getEffectiveMpcRobotModel().getInputDim();
  const size_t numBasisPerFoot = interface.getBasisDecoratorPtr()->getNumBasisPerFoot();
  const size_t jointDim = interface.modelSettings().mpc_joint_dim;

  std::mt19937 gen(1234);
  const vector_t x = makeRotatedTestState(interface, gen);
  const vector_t u = makeRandomBasisInput(N_CONTACTS * numBasisPerFoot, jointDim, gen);

  const PreComputation preComp;
  const VectorFunctionLinearApproximation approx = dynamics->linearApproximation(0.0, x, u, preComp);
  ASSERT_EQ(static_cast<size_t>(approx.f.size()), stateDim);
  ASSERT_EQ(static_cast<size_t>(approx.dfdx.rows()), stateDim);
  ASSERT_EQ(static_cast<size_t>(approx.dfdx.cols()), stateDim);
  ASSERT_EQ(static_cast<size_t>(approx.dfdu.rows()), stateDim);
  ASSERT_EQ(static_cast<size_t>(approx.dfdu.cols()), inputDim);

  const vector_t f = dynamics->computeFlowMap(0.0, x, u, preComp);
  EXPECT_LE((approx.f - f).norm(), 1e-12);

  constexpr scalar_t eps = 1e-6;
  constexpr scalar_t tolerance = 1e-4;
  const matrix_t dfdxFd = centralDifferenceJacobian(
      [&](const vector_t& xPerturbed) { return dynamics->computeFlowMap(0.0, xPerturbed, u, preComp); }, x, stateDim, eps);
  const matrix_t dfduFd = centralDifferenceJacobian(
      [&](const vector_t& uPerturbed) { return dynamics->computeFlowMap(0.0, x, uPerturbed, preComp); }, u, stateDim, eps);

  EXPECT_LE(maxNormalizedError(approx.dfdx, dfdxFd), tolerance) << "dfdx mismatch";
  EXPECT_LE(maxNormalizedError(approx.dfdu, dfduFd), tolerance) << "dfdu mismatch";

  // The contact-frame rotation depends on the configuration, so the *linear* momentum rate (rows 0..2) must depend on
  // the base orientation through dfdx. In the wrench-space dynamics this block is identically zero because the forces
  // are constant world-frame inputs; here it is the extra term contributed by the rotation inside the tape.
  EXPECT_GT(approx.dfdx.block(0, kBaseEulerZyxStartIndex, 3, 3).norm(), 1e-6);
}

/******************************************************************************************************/
/* (4) The quadratic input cost is R_basis = M^T R_wrench M + reg on the lambda diagonal              */
/******************************************************************************************************/
TEST_F(BasisInputsFormulationTest, InputCost_IsTransformedWithRegularization) {
  InterfaceHolder& h = holder(/*useBasisInputs=*/true);
  ASSERT_TRUE(h.interface) << "Failed to construct the basis-mode interface: " << h.error;
  const CentroidalMpcInterface& interface = *h.interface;

  const auto config = interface.getBasisInputsCostTransformConfig();
  ASSERT_TRUE(config.has_value());
  const size_t wrenchInputDim = interface.getWrenchInputDim();
  const size_t basisInputDim = interface.getEffectiveMpcRobotModel().getInputDim();
  const size_t numLambda = config->numBasisInputs;
  const size_t jointDim = interface.modelSettings().mpc_joint_dim;
  ASSERT_EQ(numLambda + jointDim, basisInputDim);

  // The regularization must be the value configured in the task file (and non-trivial for this test to be meaningful).
  scalar_t regularizationFromYaml = -1.0;
  loadData::loadCppDataType(h.taskFile, "contacts.basisScalingRegularization", regularizationFromYaml);
  EXPECT_DOUBLE_EQ(config->lambdaRegularization, regularizationFromYaml);
  EXPECT_GT(config->lambdaRegularization, 0.0);

  matrix_t R_wrench = matrix_t::Zero(wrenchInputDim, wrenchInputDim);
  loadData::loadEigenMatrix(h.taskFile, "R", R_wrench);
  const matrix_t R_expected = transformWrenchInputCostToBasisSpace(R_wrench, *config);
  ASSERT_EQ(static_cast<size_t>(R_expected.rows()), basisInputDim);
  ASSERT_EQ(static_cast<size_t>(R_expected.cols()), basisInputDim);

  // The Atlas formulation enables input_quadratic_cost; a state_input_quadratic_cost would be transformed identically.
  OptimalControlProblem& ocp = interface.getOptimalControlProblemRef();
  const std::vector<std::string> candidateCostNames = {"stateInputQuadraticCost", "inputQuadraticCost"};
  size_t numCostsChecked = 0;
  for (const std::string& costName : candidateCostNames) {
    size_t termIndex = 0;
    if (!ocp.costPtr->getTermIndex(costName, termIndex)) {
      continue;
    }
    auto* quadraticCost = dynamic_cast<QuadraticStateInputCost*>(&ocp.costPtr->get(costName));
    ASSERT_NE(quadraticCost, nullptr) << costName << " is not a QuadraticStateInputCost";
    matrix_t Q, R, P;
    quadraticCost->getGains(Q, R, P);
    ASSERT_EQ(static_cast<size_t>(R.rows()), basisInputDim) << costName;
    ASSERT_EQ(static_cast<size_t>(R.cols()), basisInputDim) << costName;

    EXPECT_LE((R - R_expected).cwiseAbs().maxCoeff(), 1e-12) << costName << ": R is not M^T R_wrench M + reg";
    EXPECT_LE((R - R.transpose()).cwiseAbs().maxCoeff(), 1e-12) << costName << ": R is not symmetric";
    // The joint-velocity block is unaffected by the transform.
    EXPECT_LE((R.bottomRightCorner(jointDim, jointDim) - R_wrench.bottomRightCorner(jointDim, jointDim)).cwiseAbs().maxCoeff(), 1e-12)
        << costName;

    // M^T R_wrench M is singular on the null space of M; the regularization must make the lambda block positive definite.
    Eigen::SelfAdjointEigenSolver<matrix_t> eigenSolver(matrix_t(R.topLeftCorner(numLambda, numLambda)));
    ASSERT_EQ(eigenSolver.info(), Eigen::Success) << costName;
    EXPECT_GE(eigenSolver.eigenvalues().minCoeff(), config->lambdaRegularization - 1e-12)
        << costName << ": lambda block of R is not positive definite";
    ++numCostsChecked;
  }
  EXPECT_GE(numCostsChecked, 1u) << "No quadratic input cost found in the OCP";

  // Without regularization the lambda block would be singular (more generators than wrench components).
  const matrix_t R_unregularized = transformWrenchInputCostToBasisSpace(R_wrench, config->basisToWrenchMap, numLambda, 0.0);
  Eigen::SelfAdjointEigenSolver<matrix_t> unregularizedSolver(matrix_t(R_unregularized.topLeftCorner(numLambda, numLambda)));
  ASSERT_EQ(unregularizedSolver.info(), Eigen::Success);
  EXPECT_LE(unregularizedSolver.eigenvalues().minCoeff(), 1e-12);
}

/******************************************************************************************************/
/* (5) ZeroWrenchConstraint on the effective (basis) model: Jacobian vs finite differences            */
/******************************************************************************************************/
TEST_F(BasisInputsFormulationTest, ZeroWrenchConstraint_JacobianMatchesFiniteDifferences) {
  InterfaceHolder& h = holder(/*useBasisInputs=*/true);
  ASSERT_TRUE(h.interface) << "Failed to construct the basis-mode interface: " << h.error;
  const CentroidalMpcInterface& interface = *h.interface;

  const BasisInputsModelDecorator<scalar_t>& decorator = *interface.getBasisDecoratorPtr();
  const MpcRobotModelBase<scalar_t>& effectiveModel = interface.getEffectiveMpcRobotModel();
  const size_t numBasisPerFoot = decorator.getNumBasisPerFoot();
  const size_t jointDim = interface.modelSettings().mpc_joint_dim;
  const size_t stateDim = effectiveModel.getStateDim();
  const size_t inputDim = effectiveModel.getInputDim();
  const PreComputation preComp;

  for (size_t contactIndex = 0; contactIndex < N_CONTACTS; ++contactIndex) {
    ZeroWrenchConstraint constraint(*interface.getSwitchedModelReferenceManagerPtr(), contactIndex, effectiveModel);

    std::mt19937 gen(7 + contactIndex);
    const vector_t x = makeRotatedTestState(interface, gen);
    const vector_t u = makeRandomBasisInput(N_CONTACTS * numBasisPerFoot, jointDim, gen);

    const vector_t value = constraint.getValue(0.0, x, u, preComp);
    ASSERT_EQ(static_cast<size_t>(value.size()), kWrenchDimPerContact) << "contact " << contactIndex;
    // The constraint acts on the local-frame wrench B * lambda; W_local = 0 <=> W_world = 0 since the rotation is invertible.
    EXPECT_LE((value - decorator.getContactWrench(u, contactIndex)).norm(), 1e-12) << "contact " << contactIndex;

    const VectorFunctionLinearApproximation approx = constraint.getLinearApproximation(0.0, x, u, preComp);
    ASSERT_EQ(static_cast<size_t>(approx.dfdu.rows()), kWrenchDimPerContact) << "contact " << contactIndex;
    ASSERT_EQ(static_cast<size_t>(approx.dfdu.cols()), inputDim) << "contact " << contactIndex;
    ASSERT_EQ(static_cast<size_t>(approx.dfdx.rows()), kWrenchDimPerContact) << "contact " << contactIndex;
    ASSERT_EQ(static_cast<size_t>(approx.dfdx.cols()), stateDim) << "contact " << contactIndex;
    EXPECT_LE((approx.f - value).norm(), 1e-12) << "contact " << contactIndex;
    EXPECT_DOUBLE_EQ(approx.dfdx.norm(), 0.0) << "contact " << contactIndex;

    constexpr scalar_t eps = 1e-6;
    const matrix_t dfduFd = centralDifferenceJacobian(
        [&](const vector_t& uPerturbed) { return constraint.getValue(0.0, x, uPerturbed, preComp); }, u, kWrenchDimPerContact, eps);
    EXPECT_LE(maxNormalizedError(approx.dfdu, dfduFd), 1e-6) << "contact " << contactIndex;

    // Structure: B at this contact's lambda columns, zero everywhere else.
    matrix_t expectedDfdu = matrix_t::Zero(kWrenchDimPerContact, inputDim);
    expectedDfdu.middleCols(numBasisPerFoot * contactIndex, numBasisPerFoot) = decorator.getBasisMatrix(contactIndex);
    EXPECT_LE((approx.dfdu - expectedDfdu).cwiseAbs().maxCoeff(), 1e-12) << "contact " << contactIndex;
  }
}

/******************************************************************************************************/
/* (6) The wrench-space formulation is unchanged                                                      */
/******************************************************************************************************/
TEST_F(BasisInputsFormulationTest, WrenchMode_Unchanged) {
  InterfaceHolder& h = holder(/*useBasisInputs=*/false);
  ASSERT_TRUE(h.interface) << "Failed to construct the wrench-mode interface: " << h.error;
  const CentroidalMpcInterface& interface = *h.interface;

  const size_t jointDim = interface.modelSettings().mpc_joint_dim;
  const size_t wrenchInputDim = kWrenchDimPerContact * N_CONTACTS + jointDim;

  EXPECT_FALSE(interface.usesContactBasisVectorInputs());
  EXPECT_EQ(interface.getEffectiveMpcRobotModel().getInputDim(), wrenchInputDim);
  EXPECT_EQ(interface.getEffectiveMpcRobotModelAD().getInputDim(), wrenchInputDim);
  EXPECT_EQ(interface.getMpcRobotModel().getInputDim(), wrenchInputDim);
  EXPECT_EQ(interface.getWrenchInputDim(), wrenchInputDim);
  // The effective model IS the wrench-space model.
  EXPECT_EQ(&interface.getEffectiveMpcRobotModel(), static_cast<const MpcRobotModelBase<scalar_t>*>(&interface.getMpcRobotModel()));
  EXPECT_EQ(&interface.getEffectiveMpcRobotModelAD(), static_cast<const MpcRobotModelBase<ad_scalar_t>*>(&interface.getMpcRobotModelAD()));
  EXPECT_EQ(interface.getEffectiveMpcRobotModel().getJointVelocitiesStartindex(), kWrenchDimPerContact * N_CONTACTS);

  EXPECT_FALSE(interface.getBasisInputsCostTransformConfig().has_value());
  EXPECT_FALSE(interface.getBasisToWrenchMap().has_value());
  EXPECT_EQ(interface.getBasisDecoratorPtr(), nullptr);
  EXPECT_EQ(interface.getNumBasisInputs(), 0u);

  EXPECT_NE(wrenchDynamics(interface), nullptr);
  EXPECT_EQ(basisDynamics(interface), nullptr);

  // The input cost keeps the raw wrench-space R.
  matrix_t R_wrench = matrix_t::Zero(wrenchInputDim, wrenchInputDim);
  loadData::loadEigenMatrix(h.taskFile, "R", R_wrench);
  OptimalControlProblem& ocp = interface.getOptimalControlProblemRef();
  const std::vector<std::string> candidateCostNames = {"stateInputQuadraticCost", "inputQuadraticCost"};
  size_t numCostsChecked = 0;
  for (const std::string& costName : candidateCostNames) {
    size_t termIndex = 0;
    if (!ocp.costPtr->getTermIndex(costName, termIndex)) {
      continue;
    }
    auto* quadraticCost = dynamic_cast<QuadraticStateInputCost*>(&ocp.costPtr->get(costName));
    ASSERT_NE(quadraticCost, nullptr) << costName;
    matrix_t Q, R, P;
    quadraticCost->getGains(Q, R, P);
    ASSERT_EQ(static_cast<size_t>(R.rows()), wrenchInputDim) << costName;
    ASSERT_EQ(static_cast<size_t>(R.cols()), wrenchInputDim) << costName;
    EXPECT_LE((R - R_wrench).cwiseAbs().maxCoeff(), 1e-12) << costName;
    ++numCostsChecked;
  }
  EXPECT_GE(numCostsChecked, 1u) << "No quadratic input cost found in the OCP";

  // For wrench-space models the world-frame accessors are the identity on the input (already world frame).
  std::mt19937 gen(99);
  const vector_t x = makeRotatedTestState(interface, gen);
  std::uniform_real_distribution<scalar_t> dist(-100.0, 100.0);
  vector_t u(wrenchInputDim);
  for (Eigen::Index i = 0; i < u.size(); ++i) {
    u(i) = dist(gen);
  }
  const MpcRobotModelBase<scalar_t>& model = interface.getEffectiveMpcRobotModel();
  for (size_t i = 0; i < N_CONTACTS; ++i) {
    EXPECT_DOUBLE_EQ((model.getContactWrenchInWorldFrame(x, u, i) - model.getContactWrench(u, i)).norm(), 0.0) << "contact " << i;
    EXPECT_DOUBLE_EQ((model.getContactWrench(u, i) - u.segment<kWrenchDimPerContact>(kWrenchDimPerContact * i)).norm(), 0.0)
        << "contact " << i;
  }
}

}  // namespace
}  // namespace ocs2::humanoid
