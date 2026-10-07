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
 * The library observer of CppAdInterface: the solve benchmark of humanoid_mpc_validation installs one to record the tape
 * operation count of every library a problem builds, which the problem's terms keep private. Without an observer
 * nothing happens; with one, every library made ready - generated, loaded, or reloaded by a copy - is handed to it,
 * and removing it stops the calls.
 *
 * This test generates and compiles two tiny libraries; it is tagged exclusive so that it runs alone.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include <ocs2_core/automatic_differentiation/CppAdInterface.h>

namespace ocs2 {
namespace {

std::string testFolder(const std::string& name) {
  const char* absl_nullable tmp = std::getenv("TEST_TMPDIR");
  const std::filesystem::path folder = std::filesystem::path(tmp != nullptr ? tmp : "/tmp") / ("cppad_library_observer_" + name);
  std::filesystem::remove_all(folder);
  return folder.string();
}

/** y = [sum(x) * p0; x0 * x1] for three variables and one parameter. */
CppAdInterface::ad_parameterized_function_t makeFunction() {
  return [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(2);
    y(0) = x.sum() * p(0);
    y(1) = x(0) * x(1);
  };
}

const std::vector<std::string> kFastCompileFlags = {"-O1"};

/** What the observer saw: the folder and the tape operation count of each library handed to it. */
struct Observation {
  std::string folder;
  size_t tapeOperationCount = 0;
};

/** Removes the observer at the end of a test, also when an assertion fails. */
class ObserverGuard {
 public:
  explicit ObserverGuard(std::vector<Observation>& observations) {
    CppAdInterface::setLibraryObserver([&observations](const CppAdInterface& library) {
      observations.push_back({library.getLibraryFolder(), library.getTapeOperationCount()});
    });
  }
  ~ObserverGuard() { CppAdInterface::setLibraryObserver(CppAdInterface::LibraryObserver()); }
};

TEST(CppAdLibraryObserver, EveryLibraryMadeReadyIsHandedToTheObserver) {
  const std::string folder = testFolder("ready");
  std::vector<Observation> observations;
  {
    const ObserverGuard guard(observations);
    CppAdInterface created(makeFunction(), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
    created.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
    ASSERT_EQ(observations.size(), 1u) << "the generated library";

    CppAdInterface loaded(makeFunction(), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
    loaded.loadModels(/*verbose=*/false);
    ASSERT_EQ(observations.size(), 2u) << "the loaded library";

    const CppAdInterface copy(created);
    ASSERT_EQ(observations.size(), 3u) << "the library a copy reloads";

    for (const Observation& observation : observations) {
      EXPECT_EQ(observation.folder, created.getLibraryFolder());
      EXPECT_EQ(observation.tapeOperationCount, created.getTapeOperationCount());
      EXPECT_GT(observation.tapeOperationCount, 0u);
    }
  }

  // Removed: a library made ready afterwards is not reported.
  CppAdInterface later(makeFunction(), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
  later.loadModels(/*verbose=*/false);
  EXPECT_EQ(observations.size(), 3u);
}

TEST(CppAdLibraryObserver, WithoutAnObserverTheLibraryIsUnchanged) {
  const std::string folder = testFolder("none");
  CppAdInterface created(makeFunction(), /*variableDim=*/3, /*parameterDim=*/1, "model", folder, kFastCompileFlags);
  created.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
  const vector_t x = vector_t::Random(3);
  const vector_t p = vector_t::Random(1);
  const vector_t expected = (vector_t(2) << x.sum() * p(0), x(0) * x(1)).finished();
  EXPECT_TRUE(created.getFunctionValue(x, p).isApprox(expected, 1e-12));
}

}  // namespace
}  // namespace ocs2
