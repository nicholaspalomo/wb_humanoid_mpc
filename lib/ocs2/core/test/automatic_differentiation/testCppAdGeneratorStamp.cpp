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
 * The generator stamp of CppAdInterface: every robot ships recompileLibrariesCppAd: false, so a library on disk is
 * loaded whenever it exists, and a library that an earlier code generator made (before the deterministic hashing of
 * this fork's CppADCodeGen, lib/ocs2/README.md) would be used silently. createModels() therefore stamps a library with
 * kCppAdGeneratorTag, and loadModelsIfAvailable() loads only a library whose stamp names it: one without a stamp, with
 * another tag or with a partly written stamp is regenerated, once, in place.
 *
 * This test generates and compiles tiny libraries; it is tagged exclusive so that it runs alone.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include <ocs2_core/automatic_differentiation/CppAdInterface.h>

namespace ocs2 {
namespace {

constexpr char kModelName[] = "model";

std::string testFolder(const std::string& name) {
  const char* absl_nullable tmp = std::getenv("TEST_TMPDIR");
  const std::filesystem::path folder = std::filesystem::path(tmp != nullptr ? tmp : "/tmp") / ("cppad_generator_stamp_" + name);
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

CppAdInterface makeInterface(const std::string& folder) {
  return CppAdInterface(makeFunction(), /*variableDim=*/3, /*parameterDim=*/1, kModelName, folder, kFastCompileFlags);
}

// LINT.IfChange(library_paths)
/** Where CppAdInterface keeps the library of kModelName in the library folder of `library`. */
std::filesystem::path libraryPath(const CppAdInterface& library) {
  return std::filesystem::path(library.getLibraryFolder()) / (std::string(kModelName) + "_lib.so");
}

/** Where CppAdInterface keeps the generator stamp of that library. */
std::filesystem::path stampPath(const CppAdInterface& library) {
  return std::filesystem::path(library.getLibraryFolder()) / (std::string(kModelName) + "_lib.generator");
}
// LINT.ThenChange(//lib/ocs2/core/src/automatic_differentation/CppAdInterface.cpp:generator_stamp_extension)

std::string readFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void writeFile(const std::filesystem::path& path, const std::string& content) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file << content;
}

/** Dates the library back by an hour, so that a regeneration shows as a newer modification time. */
std::filesystem::file_time_type ageLibrary(const std::filesystem::path& library) {
  const std::filesystem::file_time_type aged = std::filesystem::last_write_time(library) - std::chrono::hours(1);
  std::filesystem::last_write_time(library, aged);
  return aged;
}

/** Whether `library` evaluates the function of makeFunction(). */
bool evaluates(const CppAdInterface& library) {
  const vector_t x = vector_t::Random(3);
  const vector_t p = vector_t::Random(1);
  const vector_t expected = (vector_t(2) << x.sum() * p(0), x(0) * x(1)).finished();
  return library.getFunctionValue(x, p).isApprox(expected, 1e-12);
}

const std::string kCurrentStamp = std::string(kCppAdGeneratorTag) + "\n";

TEST(CppAdGeneratorStamp, AGeneratedLibraryIsStamped) {
  const std::string folder = testFolder("generated");
  CppAdInterface created = makeInterface(folder);
  created.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
  ASSERT_TRUE(std::filesystem::exists(libraryPath(created)));
  EXPECT_EQ(readFile(stampPath(created)), kCurrentStamp);

  // Nothing but the stamp is left next to the library: its temporary file was renamed.
  for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(created.getLibraryFolder())) {
    const std::string name = entry.path().filename().string();
    if (name.find(".generator") != std::string::npos) {
      EXPECT_EQ(name, stampPath(created).filename().string());
    }
  }
}

TEST(CppAdGeneratorStamp, AStampedLibraryLoadsWithoutRecompiling) {
  const std::string folder = testFolder("stamped");
  CppAdInterface created = makeInterface(folder);
  created.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
  const std::filesystem::file_time_type aged = ageLibrary(libraryPath(created));

  CppAdInterface loaded = makeInterface(folder);
  loaded.loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
  EXPECT_EQ(std::filesystem::last_write_time(libraryPath(loaded)), aged) << "loaded, not regenerated";
  EXPECT_TRUE(evaluates(loaded));

  // A copy reloads the library its source uses.
  const CppAdInterface copy(loaded);
  EXPECT_TRUE(evaluates(copy));
}

/** Generates the library, replaces its stamp by `prepare`, and checks that loadModelsIfAvailable() regenerates it. */
void expectRegenerated(const std::string& name, void (*absl_nonnull prepare)(const std::filesystem::path& stamp)) {
  SCOPED_TRACE(name);
  const std::string folder = testFolder(name);
  CppAdInterface created = makeInterface(folder);
  created.createModels(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
  prepare(stampPath(created));
  const std::filesystem::file_time_type aged = ageLibrary(libraryPath(created));

  CppAdInterface loaded = makeInterface(folder);
  loaded.loadModelsIfAvailable(CppAdInterface::ApproximationOrder::First, /*verbose=*/false);
  EXPECT_GT(std::filesystem::last_write_time(libraryPath(loaded)), aged) << "regenerated";
  EXPECT_EQ(readFile(stampPath(loaded)), kCurrentStamp) << "and stamped";
  EXPECT_TRUE(evaluates(loaded));
}

TEST(CppAdGeneratorStamp, ALibraryWithoutAStampIsRegenerated) {
  expectRegenerated("missing", [](const std::filesystem::path& stamp) { std::filesystem::remove(stamp); });
}

TEST(CppAdGeneratorStamp, ALibraryOfAnotherGeneratorIsRegenerated) {
  expectRegenerated("other", [](const std::filesystem::path& stamp) { writeFile(stamp, "cppadcg-another-generator\n"); });
}

TEST(CppAdGeneratorStamp, APartlyWrittenStampIsIgnored) {
  // A stamp cut short, as a writer that stopped midway would leave it if it wrote in place.
  expectRegenerated("truncated", [](const std::filesystem::path& stamp) {
    writeFile(stamp, std::string(kCppAdGeneratorTag).substr(/*pos=*/0, /*n=*/8));
  });
  // The temporary file of a writer that stopped before its rename, with the whole tag in it, and no stamp.
  expectRegenerated("temporary", [](const std::filesystem::path& stamp) {
    writeFile(std::filesystem::path(stamp.string() + ".cppadcg_tmp1234"), kCurrentStamp);
    std::filesystem::remove(stamp);
  });
}

}  // namespace
}  // namespace ocs2
