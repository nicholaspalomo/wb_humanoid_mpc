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

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"

#include <ocs2_core/control/LinearController.h>
#include <ocs2_core/control/ManifoldLinearController.h>
#include <ocs2_core/manifold/EuclideanStateManifold.h>
#include <ocs2_core/manifold/ProductStateManifold.h>
#include <ocs2_core/manifold/UnitQuaternionMath.h>

namespace ocs2 {
namespace {

constexpr size_t kInputDim = 2;

/** [E(2), Q, E(1)]: ambient 7, tangent 6. */
std::shared_ptr<const StateManifold> attitudeManifold() {
  return ProductStateManifold::create(
             {StateManifoldSegment::euclidean(2), StateManifoldSegment::unitQuaternion(), StateManifoldSegment::euclidean(1)})
      .value();
}

vector_t randomState(const StateManifold& manifold) {
  vector_t x = vector_t::Random(manifold.getAmbientDim());
  manifold.project(x);
  return x;
}

/** A controller on `manifold` with random anchors, inputs and gains at the given times. */
ManifoldLinearController randomController(const std::shared_ptr<const StateManifold>& manifold, const scalar_array_t& times) {
  vector_array_t anchors;
  vector_array_t inputs;
  matrix_array_t gains;
  for (size_t k = 0; k < times.size(); ++k) {
    anchors.push_back(randomState(*manifold));
    inputs.push_back(vector_t::Random(kInputDim));
    gains.push_back(matrix_t::Random(kInputDim, manifold->getTangentDim()));
  }
  return ManifoldLinearController(times, anchors, inputs, gains, manifold);
}

vector_t nodeLaw(const ManifoldLinearController& controller, size_t k, const vector_t& x) {
  return controller.getNominalInputs()[k] +
         controller.getFeedbackGains()[k] * controller.getStateManifold()->difference(controller.getAnchorStates()[k], x);
}

TEST(ManifoldLinearController, IsExactAtTheNodes) {
  const std::shared_ptr<const StateManifold> manifold = attitudeManifold();
  ManifoldLinearController controller = randomController(manifold, {0.0, 0.1, 0.2, 0.3});
  for (size_t k = 0; k < 4; ++k) {
    const scalar_t t = controller.getTimeStamps()[k];
    // At the anchor the feedback vanishes.
    EXPECT_TRUE(controller.computeInput(t, controller.getAnchorStates()[k]).isApprox(controller.getNominalInputs()[k], 1e-14));
    const vector_t x = randomState(*manifold);
    EXPECT_TRUE(controller.computeInput(t, x).isApprox(nodeLaw(controller, k, x), 1e-14));
  }
  EXPECT_EQ(controller.getType(), ControllerType::MANIFOLD_LINEAR);
  EXPECT_EQ(controller.size(), 4);
}

TEST(ManifoldLinearController, BlendsTheNodeLawsBetweenNodes) {
  const std::shared_ptr<const StateManifold> manifold = attitudeManifold();
  ManifoldLinearController controller = randomController(manifold, {0.0, 0.1, 0.2});
  const vector_t x = randomState(*manifold);
  const scalar_t t = 0.13;
  const scalar_t fractionToNext = 0.3;  // t is 30 % of the way from node 1 to node 2
  const vector_t expected = (1.0 - fractionToNext) * nodeLaw(controller, /*k=*/1, x) + fractionToNext * nodeLaw(controller, /*k=*/2, x);
  EXPECT_TRUE(controller.computeInput(t, x).isApprox(expected, 1e-12));
  // Clamped outside the time range.
  EXPECT_TRUE(controller.computeInput(/*t=*/-1.0, x).isApprox(nodeLaw(controller, /*k=*/0, x), 1e-14));
  EXPECT_TRUE(controller.computeInput(/*t=*/5.0, x).isApprox(nodeLaw(controller, /*k=*/2, x), 1e-14));
  // The gain and the input interpolate linearly, the anchor along the manifold.
  EXPECT_TRUE(
      controller.getFeedbackGain(t).isApprox(0.7 * controller.getFeedbackGains()[1] + 0.3 * controller.getFeedbackGains()[2], 1e-12));
  EXPECT_TRUE(manifold
                  ->difference(controller.getAnchorState(t),
                               manifold->interpolate(controller.getAnchorStates()[1], controller.getAnchorStates()[2], fractionToNext))
                  .isZero(1e-12));
}

TEST(ManifoldLinearController, UsesTheNodeBeforeAnEvent) {
  // Repeated time = an event: at the event time, the left (pre-event) node is used, as in LinearInterpolation.
  const std::shared_ptr<const StateManifold> manifold = attitudeManifold();
  ManifoldLinearController controller = randomController(manifold, {0.0, 0.1, 0.1, 0.2});
  const vector_t x = randomState(*manifold);
  EXPECT_TRUE(controller.computeInput(/*t=*/0.1, x).isApprox(nodeLaw(controller, /*k=*/1, x), 1e-14));
  EXPECT_TRUE(controller.computeInput(0.1 + 1e-9, x).isApprox(nodeLaw(controller, /*k=*/2, x), 1e-6));
  EXPECT_EQ(controller.controllerEventTimes().size(), 2u);  // {0, 0.1}
}

TEST(ManifoldLinearController, IsInvariantUnderTheDoubleCover) {
  const std::shared_ptr<const StateManifold> manifold = attitudeManifold();
  ManifoldLinearController controller = randomController(manifold, {0.0, 0.1, 0.2});
  for (const scalar_t t : {0.0, 0.05, 0.1, 0.17}) {
    const vector_t x = randomState(*manifold);
    vector_t flipped = x;
    flipped.segment<4>(2) *= -1.0;
    EXPECT_EQ(controller.computeInput(t, x), controller.computeInput(t, flipped)) << t;
  }
  // Flipping an anchor does not change the law either.
  vector_array_t anchors = controller.getAnchorStates();
  anchors[1].segment<4>(2) *= -1.0;
  ManifoldLinearController flippedAnchors(controller.getTimeStamps(), anchors, controller.getNominalInputs(), controller.getFeedbackGains(),
                                          manifold);
  const vector_t x = randomState(*manifold);
  EXPECT_EQ(controller.computeInput(/*t=*/0.1, x), flippedAnchors.computeInput(/*t=*/0.1, x));
}

TEST(ManifoldLinearController, OnAFlatManifoldItIsTheLinearController) {
  const size_t stateDim = 3;
  const std::shared_ptr<const StateManifold> flat = std::make_shared<EuclideanStateManifold>(stateDim);
  ManifoldLinearController controller = randomController(flat, {0.0, 0.1, 0.2});
  vector_array_t bias;
  for (size_t k = 0; k < 3; ++k) {
    bias.push_back(controller.getNominalInputs()[k] - controller.getFeedbackGains()[k] * controller.getAnchorStates()[k]);
  }
  LinearController linear(controller.getTimeStamps(), bias, controller.getFeedbackGains());
  for (const scalar_t t : {0.0, 0.04, 0.1, 0.18, 0.2}) {
    const vector_t x = vector_t::Random(stateDim);
    EXPECT_TRUE(controller.computeInput(t, x).isApprox(linear.computeInput(t, x), 1e-12)) << t;
  }
}

TEST(ManifoldLinearController, FlattenUnFlattenRoundTrip) {
  // Distinct node times: at a repeated (event) time a time-indexed flatten, like LinearController's, writes the
  // pre-event node.
  const std::shared_ptr<const StateManifold> manifold = attitudeManifold();
  ManifoldLinearController controller = randomController(manifold, {0.0, 0.1, 0.15, 0.25});
  const scalar_array_t& times = controller.getTimeStamps();
  std::vector<std::vector<double>> storage(times.size());
  std::vector<std::vector<double>*> flatArrays;
  std::vector<std::vector<double> const*> constFlatArrays;
  for (std::vector<double>& array : storage) {
    flatArrays.push_back(&array);
    constFlatArrays.push_back(&array);
  }
  controller.flatten(times, flatArrays);
  const size_t nx = manifold->getAmbientDim();
  const size_t ndx = manifold->getTangentDim();
  EXPECT_EQ(storage.front().size(), kInputDim + nx + kInputDim * ndx);

  const size_array_t stateDims(times.size(), nx);
  const size_array_t inputDims(times.size(), kInputDim);
  const size_array_t tangentDims(times.size(), ndx);
  absl::StatusOr<ManifoldLinearController> decoded =
      ManifoldLinearController::unFlatten(stateDims, inputDims, tangentDims, times, constFlatArrays, manifold);
  ASSERT_TRUE(decoded.ok()) << decoded.status();
  for (size_t k = 0; k < times.size(); ++k) {
    EXPECT_EQ(decoded->getAnchorStates()[k], controller.getAnchorStates()[k]);
    EXPECT_EQ(decoded->getNominalInputs()[k], controller.getNominalInputs()[k]);
    EXPECT_EQ(decoded->getFeedbackGains()[k], controller.getFeedbackGains()[k]);
  }
  const vector_t x = randomState(*manifold);
  EXPECT_EQ(decoded->computeInput(/*t=*/0.17, x), controller.computeInput(/*t=*/0.17, x));

  // A wrong length, wrong dimensions or no manifold are rejected with a status.
  storage[1].pop_back();
  EXPECT_FALSE(ManifoldLinearController::unFlatten(stateDims, inputDims, tangentDims, times, constFlatArrays, manifold).ok());
  storage[1].push_back(0.0);
  EXPECT_FALSE(
      ManifoldLinearController::unFlatten(size_array_t(times.size(), nx + 1), inputDims, tangentDims, times, constFlatArrays, manifold)
          .ok());
  EXPECT_FALSE(
      ManifoldLinearController::unFlatten(stateDims, inputDims, tangentDims, times, constFlatArrays, /*stateManifold=*/nullptr).ok());
}

/** Flattens `controller` at `times` and reads it back as a controller whose nodes are those times. */
ManifoldLinearController resample(const ManifoldLinearController& controller, const scalar_array_t& times) {
  std::vector<std::vector<double>> storage(times.size());
  std::vector<std::vector<double>*> flatArrays;
  std::vector<std::vector<double> const*> constFlatArrays;
  for (std::vector<double>& array : storage) {
    flatArrays.push_back(&array);
    constFlatArrays.push_back(&array);
  }
  controller.flatten(times, flatArrays);
  const std::shared_ptr<const StateManifold>& manifold = controller.getStateManifold();
  absl::StatusOr<ManifoldLinearController> decoded = ManifoldLinearController::unFlatten(
      size_array_t(times.size(), manifold->getAmbientDim()), size_array_t(times.size(), kInputDim),
      size_array_t(times.size(), manifold->getTangentDim()), times, constFlatArrays, manifold);
  CHECK(decoded.ok()) << decoded.status();
  return *std::move(decoded);
}

TEST(ManifoldLinearController, ResamplingOffTheNodesKeepsTheLawOnAFlatManifold) {
  // An encoder may sample the policy on its own time grid. On a flat manifold the resampled law must be the solver's,
  // as LinearController's flatten is; interpolating u* instead of evaluating the blended law at the interpolated
  // anchor would be off by w (1 - w) (K_k - K_{k+1}) (xbar_k - xbar_{k+1}).
  const std::shared_ptr<const StateManifold> flat = std::make_shared<EuclideanStateManifold>(/*dimension=*/3);
  ManifoldLinearController controller = randomController(flat, {0.0, 0.1, 0.2, 0.3});
  const scalar_array_t sampleTimes{0.03, 0.07, 0.15, 0.26};
  ManifoldLinearController resampled = resample(controller, sampleTimes);
  for (const scalar_t t : sampleTimes) {
    for (int trial = 0; trial < 5; ++trial) {
      const vector_t x = 3.0 * vector_t::Random(3);
      EXPECT_TRUE(resampled.computeInput(t, x).isApprox(controller.computeInput(t, x), 1e-12)) << t;
    }
  }
}

TEST(ManifoldLinearController, ResamplingOffTheNodesIsExactAtTheAnchorAndBilinearAwayFromIt) {
  const std::shared_ptr<const StateManifold> manifold = attitudeManifold();
  ManifoldLinearController controller = randomController(manifold, {0.0, 0.1, 0.2});
  const scalar_array_t sampleTimes{0.04, 0.13};
  ManifoldLinearController resampled = resample(controller, sampleTimes);
  for (const scalar_t t : sampleTimes) {
    const vector_t anchor = controller.getAnchorState(t);
    EXPECT_TRUE(resampled.computeInput(t, anchor).isApprox(controller.computeInput(t, anchor), 1e-12)) << t;
    // Away from the anchor the mismatch is first order in the deviation (times the anchor distance), so it shrinks
    // in proportion to the deviation.
    const vector_t direction = vector_t::Random(manifold->getTangentDim()).normalized();
    const scalar_t largeStep = 1e-2;
    const scalar_t smallStep = 1e-4;
    vector_t far;
    vector_t near;
    manifold->retract(anchor, direction, /*alpha=*/largeStep, far);
    manifold->retract(anchor, direction, /*alpha=*/smallStep, near);
    const scalar_t farError = (resampled.computeInput(t, far) - controller.computeInput(t, far)).norm();
    const scalar_t nearError = (resampled.computeInput(t, near) - controller.computeInput(t, near)).norm();
    EXPECT_LT(nearError, 2.0 * (smallStep / largeStep) * farError + 1e-12) << t;
  }
  // At the node times nothing is interpolated: the resampled nodes are the controller's, bit for bit.
  ManifoldLinearController atNodes = resample(controller, controller.getTimeStamps());
  for (size_t k = 0; k < controller.getTimeStamps().size(); ++k) {
    EXPECT_EQ(atNodes.getNominalInputs()[k], controller.getNominalInputs()[k]);
    EXPECT_EQ(atNodes.getAnchorStates()[k], controller.getAnchorStates()[k]);
    EXPECT_EQ(atNodes.getFeedbackGains()[k], controller.getFeedbackGains()[k]);
  }
}

TEST(ManifoldLinearController, ConcatenateCloneAndClear) {
  const std::shared_ptr<const StateManifold> manifold = attitudeManifold();
  ManifoldLinearController first = randomController(manifold, {0.0, 0.1});
  const ManifoldLinearController second = randomController(manifold, {0.2, 0.3, 0.4});
  first.concatenate(&second, /*index=*/1, /*length=*/2);
  ASSERT_EQ(first.size(), 4);
  EXPECT_EQ(first.getTimeStamps()[2], 0.3);
  EXPECT_EQ(first.getAnchorStates()[3], second.getAnchorStates()[2]);

  std::unique_ptr<ManifoldLinearController> copy(first.clone());
  const vector_t x = randomState(*manifold);
  EXPECT_EQ(copy->computeInput(/*t=*/0.25, x), first.computeInput(/*t=*/0.25, x));

  // Another manifold object, or another controller type, cannot be appended.
  ManifoldLinearController other = randomController(attitudeManifold(), {0.5});
  EXPECT_THROW(first.concatenate(&other, /*index=*/0, /*length=*/1), std::runtime_error);
  LinearController linear;
  EXPECT_THROW(first.concatenate(&linear, /*index=*/0, /*length=*/0), std::runtime_error);

  first.clear();
  EXPECT_TRUE(first.empty());
}

}  // namespace
}  // namespace ocs2
