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
 * CppAdInterface refuses a library on disk that was generated for another function. Every robot ships
 * recompileLibrariesCppAd: false, so after a change of the state layout the libraries of the old layout would otherwise
 * be loaded and evaluated on vectors of the wrong size. The check compares the library's domain with the function's
 * variables + parameters, and its range with the function's: the range already known (from the interface a copy was
 * made of), or, for a fresh interface, the size of one off-tape evaluation.
 *
 * This test generates and compiles a few tiny libraries; it is tagged exclusive so that it runs alone.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "absl/base/nullability.h"

#include <ocs2_core/automatic_differentiation/CppAdInterface.h>

namespace ocs2 {
namespace {

std::string testFolder(const std::string& name) {
  const char* absl_nullable tmp = std::getenv("TEST_TMPDIR");
  const std::filesystem::path folder = std::filesystem::path(tmp != nullptr ? tmp : "/tmp") / ("cppad_dimension_check_" + name);
  std::filesystem::remove_all(folder);
  return folder.string();
}

/** y = [sum(x) * p0; x0 * x1; ...] with `rangeDim` outputs, for `variableDim` variables and one parameter. */
CppAdInterface::ad_parameterized_function_t makeFunction(size_t rangeDim) {
  return [rangeDim](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(rangeDim);
    for (size_t i = 0; i < rangeDim; ++i) {
      y(i) = x.sum() * p(0) + x(i % x.size()) * x((i + 1) % x.size());
    }
  };
}

const std::vector<std::string> kFastCompileFlags = {"-O1"};

TEST(CppAdLibraryDimensionCheck, AMatchingLibraryLoads) {
  const std::string folder = testFolder("matching");
  CppAdInterface created(makeFunction(/*rangeDim=*/2), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
  created.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);

  CppAdInterface loaded(makeFunction(/*rangeDim=*/2), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
  ASSERT_NO_THROW(loaded.loadModels(/*verbose=*/false));
  const vector_t x = vector_t::Random(3);
  const vector_t p = vector_t::Random(1);
  EXPECT_EQ(loaded.getFunctionValue(x, p), created.getFunctionValue(x, p));

  // A copy reloads the same library, checked against the range it knows.
  const CppAdInterface copy(created);
  EXPECT_EQ(copy.getFunctionValue(x, p), created.getFunctionValue(x, p));
}

TEST(CppAdLibraryDimensionCheck, AStaleDomainThrowsAndNamesTheFolder) {
  const std::string folder = testFolder("stale_domain");
  CppAdInterface oldLayout(makeFunction(/*rangeDim=*/2), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
  oldLayout.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);

  // The same model in the same folder, now with one more variable (a state that grew by one entry).
  CppAdInterface newLayout(makeFunction(/*rangeDim=*/2), /*variableDim=*/4, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
  try {
    newLayout.loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
    FAIL() << "A library generated for 3 + 1 inputs was loaded for 4 + 1.";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find(newLayout.getLibraryFolder()), std::string::npos) << message;
    EXPECT_NE(message.find("domain is 4"), std::string::npos) << message;
    EXPECT_NE(message.find("recompileLibrariesCppAd"), std::string::npos) << message;
  }
}

TEST(CppAdLibraryDimensionCheck, AStaleRangeThrowsOnReload) {
  const std::string folder = testFolder("stale_range");
  CppAdInterface threeOutputs(makeFunction(/*rangeDim=*/3), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
  threeOutputs.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);

  // Another process regenerates the library with two outputs; a copy of the three-output interface then reloads it.
  CppAdInterface twoOutputs(makeFunction(/*rangeDim=*/2), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
  twoOutputs.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
  try {
    const CppAdInterface copy(threeOutputs);
    FAIL() << "A library with 2 outputs was loaded for a function with 3.";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find(threeOutputs.getLibraryFolder()), std::string::npos) << message;
    EXPECT_NE(message.find("range 2"), std::string::npos) << message;
  }
}

TEST(CppAdLibraryDimensionCheck, AStaleRangeThrowsOnAFreshLoad) {
  // The case every robot meets with recompileLibrariesCppAd: false: a new process, a term whose residual grew by one
  // row with the same inputs, and the old library still on disk.
  const std::string folder = testFolder("stale_range_fresh");
  CppAdInterface oldTerm(makeFunction(/*rangeDim=*/2), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
  oldTerm.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);

  CppAdInterface newTerm(makeFunction(/*rangeDim=*/3), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
  try {
    newTerm.loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
    FAIL() << "A library with 2 outputs was loaded for a function with 3.";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find(newTerm.getLibraryFolder()), std::string::npos) << message;
    EXPECT_NE(message.find("range 2"), std::string::npos) << message;
    EXPECT_NE(message.find("returns 3"), std::string::npos) << message;
  }
}

TEST(CppAdLibraryDimensionCheck, TapeOperationCountNeedsNoLibrary) {
  const CppAdInterface small(makeFunction(/*rangeDim=*/1), /*variableDim=*/3, /*parameterDim=*/1, "small", testFolder("count_small"));
  const CppAdInterface large(makeFunction(/*rangeDim=*/5), /*variableDim=*/3, /*parameterDim=*/1, "large", testFolder("count_large"));
  EXPECT_GT(small.getTapeOperationCount(), 0u);
  EXPECT_GT(large.getTapeOperationCount(), small.getTapeOperationCount());
  EXPECT_EQ(small.getTapeOperationCount(), small.getTapeOperationCount());
}

}  // namespace
}  // namespace ocs2
