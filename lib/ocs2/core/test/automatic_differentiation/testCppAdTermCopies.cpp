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
 * The copies of the four CppAD term bases - StateInputCostGaussNewtonAd, StateInputConstraintCppAd,
 * StateConstraintCppAd and SystemDynamicsBaseAD - tape themselves. Their taped functions call a virtual function of
 * the term, so a copy whose library kept the function of the term it was copied from would tape that term, after it
 * was destroyed, wherever a copy re-tapes: the library observer of the solve benchmark, here one that re-tapes every
 * library made ready (CppAdInterface::getTapeOperationCount()). Each term logs the object that tapes it; the original
 * is destroyed after the first copy, and the copy of the copy must be taped by the first copy while it is built, and by
 * itself afterwards.
 *
 * This test generates and compiles five tiny libraries; it is tagged exclusive so that it runs alone.
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/nullability.h"

#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_core/constraint/StateConstraintCppAd.h>
#include <ocs2_core/constraint/StateInputConstraintCppAd.h>
#include <ocs2_core/cost/StateInputGaussNewtonCostAd.h>
#include <ocs2_core/dynamics/SystemDynamicsBaseAD.h>

namespace ocs2 {
namespace {

constexpr size_t kStateDim = 2;
constexpr size_t kInputDim = 1;

/** The objects that taped a term's functions, in the order they taped them. */
using TapeLog = std::vector<const void* absl_nonnull>;

std::string testFolder(const std::string& name) {
  const char* absl_nullable tmp = std::getenv("TEST_TMPDIR");
  const std::filesystem::path folder = std::filesystem::path(tmp != nullptr ? tmp : "/tmp") / ("cppad_term_copies_" + name);
  std::filesystem::remove_all(folder);
  return folder.string();
}

/** Re-tapes every library made ready while it is alive, as the solve benchmark's observer does for a new library. */
class RetapingObserver {
 public:
  RetapingObserver() {
    CppAdInterface::setLibraryObserver([](const CppAdInterface& library) { EXPECT_GT(library.getTapeOperationCount(), 0u); });
  }
  ~RetapingObserver() { CppAdInterface::setLibraryObserver(CppAdInterface::LibraryObserver()); }
  RetapingObserver(const RetapingObserver&) = delete;
  RetapingObserver& operator=(const RetapingObserver&) = delete;
};

class LoggingCost final : public StateInputCostGaussNewtonAd {
 public:
  LoggingCost(TapeLog* absl_nonnull log, const std::string& folder) : log_(log) {
    initialize(kStateDim, kInputDim, /*parameterDim=*/0, "cost", folder, /*recompileLibraries=*/true, /*verbose=*/false);
  }
  LoggingCost* absl_nonnull clone() const override { return new LoggingCost(*this); }

 private:
  LoggingCost(const LoggingCost& rhs) = default;
  ad_vector_t costVectorFunction(ad_scalar_t /*time*/, const ad_vector_t& state, const ad_vector_t& input,
                                 const ad_vector_t& /*parameters*/) override {
    log_->push_back(this);
    return state * input(0);
  }

  TapeLog* absl_nonnull log_;
};

class LoggingStateInputConstraint final : public StateInputConstraintCppAd {
 public:
  LoggingStateInputConstraint(TapeLog* absl_nonnull log, const std::string& folder)
      : StateInputConstraintCppAd(ConstraintOrder::Linear), log_(log) {
    initialize(kStateDim, kInputDim, /*parameterDim=*/0, "state_input_constraint", folder, /*recompileLibraries=*/true, /*verbose=*/false);
  }
  LoggingStateInputConstraint* absl_nonnull clone() const override { return new LoggingStateInputConstraint(*this); }
  size_t getNumConstraints(scalar_t /*time*/) const override { return kStateDim; }

 private:
  LoggingStateInputConstraint(const LoggingStateInputConstraint& rhs) = default;
  ad_vector_t constraintFunction(ad_scalar_t /*time*/, const ad_vector_t& state, const ad_vector_t& input,
                                 const ad_vector_t& /*parameters*/) const override {
    log_->push_back(this);
    return state * input(0);
  }

  TapeLog* absl_nonnull log_;
};

class LoggingStateConstraint final : public StateConstraintCppAd {
 public:
  LoggingStateConstraint(TapeLog* absl_nonnull log, const std::string& folder) : StateConstraintCppAd(ConstraintOrder::Linear), log_(log) {
    initialize(kStateDim, /*parameterDim=*/0, "state_constraint", folder, /*recompileLibraries=*/true, /*verbose=*/false);
  }
  LoggingStateConstraint* absl_nonnull clone() const override { return new LoggingStateConstraint(*this); }
  size_t getNumConstraints(scalar_t /*time*/) const override { return 1; }

 private:
  LoggingStateConstraint(const LoggingStateConstraint& rhs) = default;
  ad_vector_t constraintFunction(ad_scalar_t /*time*/, const ad_vector_t& state, const ad_vector_t& /*parameters*/) const override {
    log_->push_back(this);
    return state.head(1) * state(1);
  }

  TapeLog* absl_nonnull log_;
};

/** Logs both of its tapes: the flow map's and the jump map's. */
class LoggingDynamics final : public SystemDynamicsBaseAD {
 public:
  LoggingDynamics(TapeLog* absl_nonnull log, const std::string& folder) : log_(log) {
    initialize(kStateDim, kInputDim, "dynamics", folder, /*recompileLibraries=*/true, /*verbose=*/false);
  }
  LoggingDynamics* absl_nonnull clone() const override { return new LoggingDynamics(*this); }

 private:
  LoggingDynamics(const LoggingDynamics& rhs) = default;
  ad_vector_t systemFlowMap(ad_scalar_t /*time*/, const ad_vector_t& state, const ad_vector_t& input) const override {
    log_->push_back(this);
    return state * input(0);
  }
  ad_vector_t systemJumpMap(ad_scalar_t /*time*/, const ad_vector_t& state) const override {
    log_->push_back(this);
    return state;
  }

  TapeLog* absl_nonnull log_;
};

/**
 * Copies `original` with the re-taping observer installed, destroys it, and copies the copy twice more, each time
 * destroying the one copied from: every copy is taped by the term it is copied from while it is built, which is alive,
 * and never by a destroyed one. `tapesPerTerm` is the number of libraries of a term.
 */
template <typename Term>
void expectEachCopyTapesTheTermItIsCopiedFrom(std::unique_ptr<Term> original, TapeLog& log, size_t tapesPerTerm) {
  const RetapingObserver observer;
  std::unique_ptr<Term> source = std::move(original);
  for (int copyIndex = 0; copyIndex < 3; ++copyIndex) {
    log.clear();
    std::unique_ptr<Term> copy(source->clone());
    EXPECT_EQ(log, TapeLog(tapesPerTerm, source.get())) << "copy " << copyIndex;
    source = std::move(copy);  // destroys the term copied from
  }
}

TEST(CppAdTermCopies, TheCopiesOfAGaussNewtonCostTapeThemselves) {
  TapeLog log;
  std::unique_ptr<LoggingCost> original = std::make_unique<LoggingCost>(&log, testFolder("cost"));
  EXPECT_EQ(log, TapeLog(1, original.get())) << "the generated library";
  expectEachCopyTapesTheTermItIsCopiedFrom(std::move(original), log, /*tapesPerTerm=*/1);
}

TEST(CppAdTermCopies, TheCopiesOfAStateInputConstraintTapeThemselves) {
  TapeLog log;
  std::unique_ptr<LoggingStateInputConstraint> original =
      std::make_unique<LoggingStateInputConstraint>(&log, testFolder("state_input_constraint"));
  EXPECT_EQ(log, TapeLog(1, original.get())) << "the generated library";
  expectEachCopyTapesTheTermItIsCopiedFrom(std::move(original), log, /*tapesPerTerm=*/1);
}

TEST(CppAdTermCopies, TheCopiesOfAStateConstraintTapeThemselves) {
  TapeLog log;
  std::unique_ptr<LoggingStateConstraint> original = std::make_unique<LoggingStateConstraint>(&log, testFolder("state_constraint"));
  EXPECT_EQ(log, TapeLog(1, original.get())) << "the generated library";
  expectEachCopyTapesTheTermItIsCopiedFrom(std::move(original), log, /*tapesPerTerm=*/1);
}

TEST(CppAdTermCopies, TheCopiesOfTheDynamicsTapeThemselves) {
  TapeLog log;
  std::unique_ptr<LoggingDynamics> original = std::make_unique<LoggingDynamics>(&log, testFolder("dynamics"));
  EXPECT_EQ(log, TapeLog(2, original.get())) << "the generated flow and jump map libraries";
  expectEachCopyTapesTheTermItIsCopiedFrom(std::move(original), log, /*tapesPerTerm=*/2);
}

TEST(CppAdTermCopies, ACopyEvaluatesTheLibraryOfTheTermItIsCopiedFrom) {
  TapeLog log;
  const std::unique_ptr<LoggingCost> original = std::make_unique<LoggingCost>(&log, testFolder("cost_values"));
  const std::unique_ptr<LoggingCost> copy(original->clone());
  const vector_t state = (vector_t(kStateDim) << 0.3, -0.7).finished();
  const vector_t input = (vector_t(kInputDim) << 1.5).finished();
  const scalar_t expected = 0.5 * (state * input(0)).squaredNorm();
  EXPECT_DOUBLE_EQ(copy->getValue(/*time=*/0.0, state, input, TargetTrajectories(), PreComputation()), expected);
}

}  // namespace
}  // namespace ocs2
