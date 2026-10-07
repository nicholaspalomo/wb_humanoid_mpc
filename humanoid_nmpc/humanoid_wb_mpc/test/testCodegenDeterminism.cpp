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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "gtest/gtest.h"
#include "ocs2_core/automatic_differentiation/Types.h"
#include "ocs2_pinocchio_interface/PinocchioInterface.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_wb_mpc/WBMpcInterface.h"
#include "humanoid_wb_mpc/common/WBAccelMpcRobotModel.h"
#include "humanoid_wb_mpc/dynamics/DynamicsHelperFunctions.h"

/*
 * The CppAD code generation of a production tape is deterministic: two generations of the G1's whole-body dynamics
 * (the flow map WBAccelDynamicsAD tapes) give the same C sources byte for byte, the second on a thread of its own after
 * the heap was moved, so that its constants land at other addresses. The whole-body libraries are where the generation
 * used to follow the heap (lib/ocs2/core/test/automatic_differentiation/testCppAdCodegenDeterminism.cpp shows the cause
 * on a small function); this pins the property on a real model at every test run. Nothing is compiled: the sources are
 * saved as CppAdInterface::createModels() would hand them to the compiler.
 */
namespace ocs2::humanoid {
namespace {

std::string runfilePath(absl::string_view relativePath) {
  std::vector<std::filesystem::path> roots;
  if (const char* absl_nullable srcDir = std::getenv("TEST_SRCDIR")) {
    roots.emplace_back(std::filesystem::path(srcDir) / "_main");
  }
  roots.emplace_back(std::filesystem::current_path());
  for (const std::filesystem::path& root : roots) {
    const std::filesystem::path candidate = root / std::string(relativePath);
    if (std::filesystem::exists(candidate)) return std::filesystem::absolute(candidate).string();
  }
  return std::string();
}

/** The files of `folder`, by name, with their contents. */
std::map<std::string, std::string> readFiles(const std::filesystem::path& folder) {
  std::map<std::string, std::string> files;
  for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(folder)) {
    std::ifstream file(entry.path(), std::ios::binary);
    files[entry.path().filename().string()] = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  }
  return files;
}

/**
 * Tapes the whole-body flow map of the robot `pinocchioInterface` with `modelSettings` as CppAdInterface::createModels()
 * does (at ones, optimized) and returns the sources of its zero- and first-order models, generated into `folder`.
 */
std::map<std::string, std::string> generateFlowMapSources(const PinocchioInterface& pinocchioInterface,
                                                          const ModelSettings& modelSettings,
                                                          const std::filesystem::path& folder) {
  const PinocchioInterfaceCppAd pinocchio = pinocchioInterface.toCppAd();
  WBAccelMpcRobotModel<ad_scalar_t> model(modelSettings);
  const Eigen::Index stateDim = static_cast<Eigen::Index>(model.getStateDim());
  const Eigen::Index inputDim = static_cast<Eigen::Index>(model.getInputDim());
  ad_vector_t stateInput = ad_vector_t::Ones(stateDim + inputDim);
  CppAD::Independent(stateInput);
  const ad_vector_t flow = computeStateDerivative<ad_scalar_t>(stateInput.head(stateDim), stateInput.tail(inputDim), pinocchio, model);
  CppAD::ADFun<ad_base_t> fun(stateInput, flow);
  fun.optimize();

  CppAD::cg::ModelCSourceGen<scalar_t> sourceGen(fun, "wb_flow_map");
  sourceGen.setCreateForwardZero(/*create=*/true);
  sourceGen.setCreateSparseJacobian(/*create=*/true);
  CppAD::cg::ModelLibraryCSourceGen<scalar_t> librarySourceGen(sourceGen);
  std::filesystem::remove_all(folder);
  CppAD::cg::SaveFilesModelLibraryProcessor<scalar_t>::saveLibrarySourcesTo(librarySourceGen, folder.string());
  return readFiles(folder);
}

TEST(CodegenDeterminism, TwoGenerationsOfTheWholeBodyDynamicsGiveTheSameSources) {
  const std::string taskFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/mpc/task.textproto");
  const std::string referenceFile = runfilePath("robot_models/unitree_g1/g1_wb_mpc/config/command/reference.textproto");
  const std::string urdfFile = runfilePath("robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf");
  ASSERT_FALSE(taskFile.empty() || referenceFile.empty() || urdfFile.empty()) << "the G1 whole-body files are not in the runfiles";
  // The controller models only: nothing is taped or compiled by the interface.
  absl::StatusOr<std::unique_ptr<WBMpcInterface>> interface = WBMpcInterface::CreateControllerModels(taskFile, urdfFile, referenceFile);
  ASSERT_TRUE(interface.ok()) << interface.status();
  const std::filesystem::path scratch = testing::TempDir();
  const PinocchioInterface& pinocchioInterface = (*interface)->getPinocchioInterface();
  const ModelSettings& modelSettings = (*interface)->modelSettings();

  const std::map<std::string, std::string> first = generateFlowMapSources(pinocchioInterface, modelSettings, scratch / "first");

  // The second on another thread, which allocates from an arena and a cache of its own, after blocks of every small size
  // were taken and kept: its constants live at other addresses than the first generation's.
  std::map<std::string, std::string> second;
  std::thread([&pinocchioInterface, &modelSettings, &scratch, &second]() {
    std::vector<std::unique_ptr<char[]>> kept;
    for (size_t size = 1; size <= 512; ++size) kept.push_back(std::make_unique<char[]>(size));
    second = generateFlowMapSources(pinocchioInterface, modelSettings, scratch / "second");
  }).join();

  ASSERT_FALSE(first.empty());
  EXPECT_NE(first.find("wb_flow_map_forward_zero.c"), first.end()) << "the zero-order model was not generated";
  size_t bytes = 0;
  for (const std::pair<const std::string, std::string>& source : first) bytes += source.second.size();
  RecordProperty("source_files", static_cast<int>(first.size()));
  RecordProperty("source_bytes", static_cast<int>(bytes));
  ASSERT_EQ(first.size(), second.size());
  for (const std::pair<const std::string, std::string>& source : first) {
    const std::map<std::string, std::string>::const_iterator other = second.find(source.first);
    ASSERT_NE(other, second.end()) << source.first;
    EXPECT_TRUE(source.second == other->second) << source.first << " differs between the two generations";
  }
}

}  // namespace
}  // namespace ocs2::humanoid
