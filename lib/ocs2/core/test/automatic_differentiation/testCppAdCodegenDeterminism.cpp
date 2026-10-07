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

/*
 * CppAD code generation is deterministic: the C sources generated from one function do not depend on where the heap
 * puts the values of its constants. CppAD's recorder files a constant under a hash code and shares a parameter index
 * only with the constant last filed under the same code; CppADCodeGen supplied no hash code for its CG type, so the
 * default hashed a CG's bytes, a heap address among them, and the deduplication of the tape's constants, optimize()'s
 * common subexpressions and the printed sources followed the heap's layout. The local change of this fork
 * (cppad/cg/identical.hpp, lib/ocs2/README.md) hashes a constant's value.
 *
 * The tests tape and generate sources as CppAdInterface::createModels() does, but save the sources instead of compiling
 * them.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/str_cat.h"

#include <ocs2_core/automatic_differentiation/Types.h>

namespace ocs2 {
namespace {

using ad_fun_t = CppAD::ADFun<ad_base_t>;

/** An empty folder under the test's temporary directory. */
std::filesystem::path testFolder(const std::string& name) {
  const char* absl_nullable tmp = std::getenv("TEST_TMPDIR");
  const std::filesystem::path folder = std::filesystem::path(tmp != nullptr ? tmp : "/tmp") / ("cppad_codegen_determinism_" + name);
  std::filesystem::remove_all(folder);
  return folder;
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

/** The size of a tape: its parameters as recorded, and its parameters and operations after optimize(). */
struct TapeSize {
  size_t recordedParameters = 0;
  size_t optimizedParameters = 0;
  size_t optimizedOperations = 0;
};

TapeSize optimize(ad_fun_t& fun) {
  TapeSize size{.recordedParameters = fun.size_par()};
  fun.optimize();
  size.optimizedParameters = fun.size_par();
  size.optimizedOperations = fun.size_op();
  return size;
}

TEST(CppAdCodegenDeterminism, ConstantsHashByValue) {
  const ad_base_t first(0.3);
  const ad_base_t second(0.3);
  ASSERT_NE(&first.getValue(), &second.getValue()) << "each CG keeps its value on the heap";
  EXPECT_EQ(CppAD::hash_code(first), CppAD::hash_code(second));
  const double value = 0.3;
  EXPECT_EQ(CppAD::hash_code(first), CppAD::hash_code(value)) << "a constant hashes as its value does";

  const ad_base_t zero(0.0);
  const ad_base_t negativeZero(-0.0);
  EXPECT_NE(CppAD::hash_code(zero), CppAD::hash_code(negativeZero)) << "so that +0 and -0 keep their signs";
}

/** y(k) = x0 * constants[k % constants.size()] for k < productCount, taped at x0 = 1 and optimized. */
TapeSize productsTapeSize(const std::vector<ad_scalar_t>& constants, size_t productCount) {
  ad_vector_t x = ad_vector_t::Ones(1);
  CppAD::Independent(x);
  ad_vector_t y(static_cast<Eigen::Index>(productCount));
  for (size_t k = 0; k < productCount; ++k) {
    y(static_cast<Eigen::Index>(k)) = x(0) * constants[k % constants.size()];
  }
  ad_fun_t fun(x, y);
  return optimize(fun);
}

TEST(CppAdCodegenDeterminism, EqualConstantsShareOneParameter) {
  constexpr size_t kProductCount = 64;
  // All alive at once, so each value has its own heap address.
  std::vector<ad_scalar_t> distinctConstants;
  distinctConstants.reserve(kProductCount);
  for (size_t k = 0; k < kProductCount; ++k) {
    distinctConstants.emplace_back(0.3);
  }
  const std::vector<ad_scalar_t> oneConstant(/*n=*/1, ad_scalar_t(0.3));

  const TapeSize distinct = productsTapeSize(distinctConstants, kProductCount);
  const TapeSize shared = productsTapeSize(oneConstant, kProductCount);
  EXPECT_EQ(distinct.recordedParameters, 2u) << "the NaN every tape starts with, and 0.3";
  EXPECT_EQ(distinct.recordedParameters, shared.recordedParameters);
  EXPECT_EQ(distinct.optimizedOperations, shared.optimizedOperations) << "the 64 products are one common subexpression";
}

constexpr Eigen::Index kVariableDim = 8;

/**
 * A function with the constants that showed the nondeterminism in the whole-body MPC's libraries: equal constants at
 * several call sites, 0 - x next to -0 - x, and sums that optimize() folds into cumulative sums with constant terms.
 * Every constant is a temporary, built, used and destroyed before the next one is built, and perturbHeap() runs after
 * each: with nothing in between, the allocator hands the next constant the address the last one freed, which the
 * default hash of a CG mistook for the same constant.
 */
ad_vector_t tapedFunction(const ad_vector_t& x, absl::FunctionRef<void()> perturbHeap) {
  const Eigen::Index n = x.size();
  ad_vector_t y(3 * n + 1);
  for (Eigen::Index i = 0; i < n; ++i) {
    y(i) = x(0) * ad_scalar_t(0.3) + x(i);
    perturbHeap();
  }
  for (Eigen::Index i = 0; i < n; ++i) {
    y(n + i) = ad_scalar_t(0.0) - x(i);
    perturbHeap();
    y(2 * n + i) = ad_scalar_t(-0.0) - x(i);
    perturbHeap();
  }
  ad_scalar_t sum = x(0);
  for (Eigen::Index i = 0; i < n; ++i) {
    sum = sum + x(i) * ad_scalar_t(0.5);
    perturbHeap();
    sum = sum - ad_scalar_t(0.25);
    perturbHeap();
    sum = ad_scalar_t(0.25) + sum - x((i + 1) % n);
    perturbHeap();
  }
  y(3 * n) = sum;
  return y;
}

/** What one generation of tapedFunction() gave: the tape's size and the generated sources, by file name. */
struct Generation {
  TapeSize tapeSize;
  std::map<std::string, std::string> sources;
};

/** Tapes tapedFunction() at x = 1 with `perturbHeap`, optimizes it and generates its sources into `folder`. */
Generation generate(absl::FunctionRef<void()> perturbHeap, const std::filesystem::path& folder) {
  ad_vector_t x = ad_vector_t::Ones(kVariableDim);
  CppAD::Independent(x);
  const ad_vector_t y = tapedFunction(x, perturbHeap);
  ad_fun_t fun(x, y);
  Generation generation{.tapeSize = optimize(fun)};

  CppAD::cg::ModelCSourceGen<scalar_t> sourceGen(fun, "model");
  sourceGen.setCreateForwardZero(/*create=*/true);
  sourceGen.setCreateSparseJacobian(/*create=*/true);
  CppAD::cg::ModelLibraryCSourceGen<scalar_t> librarySourceGen(sourceGen);
  // The sources DynamicModelLibraryProcessor compiles in createModels(), saved instead.
  CppAD::cg::SaveFilesModelLibraryProcessor<scalar_t>::saveLibrarySourcesTo(librarySourceGen, folder.string());
  generation.sources = readFiles(folder);
  return generation;
}

TEST(CppAdCodegenDeterminism, SourcesDoNotDependOnTheHeap) {
  const Generation reused = generate([] {}, testFolder("reused"));

  // Keeps one block of a constant's size per call, so that every constant lands at an address of its own.
  std::vector<std::unique_ptr<double>> keptBlocks;
  const Generation moved = generate([&keptBlocks] { keptBlocks.push_back(std::make_unique<double>(0.0)); }, testFolder("moved"));
  ASSERT_GT(keptBlocks.size(), 0u);

  EXPECT_EQ(reused.tapeSize.recordedParameters, moved.tapeSize.recordedParameters);
  EXPECT_EQ(reused.tapeSize.optimizedParameters, moved.tapeSize.optimizedParameters);
  EXPECT_EQ(reused.tapeSize.optimizedOperations, moved.tapeSize.optimizedOperations);

  ASSERT_FALSE(reused.sources.empty());
  ASSERT_EQ(reused.sources.size(), moved.sources.size());
  for (const std::pair<const std::string, std::string>& source : reused.sources) {
    const std::map<std::string, std::string>::const_iterator other = moved.sources.find(source.first);
    ASSERT_NE(other, moved.sources.end()) << source.first;
    EXPECT_EQ(source.second, other->second) << source.first;
  }
}

TEST(CppAdCodegenDeterminism, SignedZerosKeepTheirSigns) {
  // IdenticalEqualCon() counts +0 and -0 as equal, so only their hash codes keep them apart: each gets a parameter of
  // its own, and 0 - x and -0 - x print as written, not as whichever of the two the recorder met first.
  const Generation generation = generate([] {}, testFolder("signed_zeros"));
  const std::map<std::string, std::string>::const_iterator forwardZero = generation.sources.find("model_forward_zero.c");
  ASSERT_NE(forwardZero, generation.sources.end());
  for (Eigen::Index i = 0; i < kVariableDim; ++i) {
    EXPECT_NE(forwardZero->second.find(absl::StrCat("= 0 - x[", i, "];")), std::string::npos) << i;
    EXPECT_NE(forwardZero->second.find(absl::StrCat("= -0 - x[", i, "];")), std::string::npos) << i;
  }
}

}  // namespace
}  // namespace ocs2
