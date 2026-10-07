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

#pragma once

#include <memory>
#include <ostream>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"

#include <ocs2_core/Types.h>
#include <ocs2_core/control/ControllerBase.h>
#include <ocs2_core/manifold/StateManifold.h>

namespace ocs2 {

/**
 * The feedback policy of a solver whose state lives on a StateManifold:
 *
 *   u(t, x) = (1 - a) [u*_k + K_k (x (-) xbar_k)] + a [u*_{k+1} + K_{k+1} (x (-) xbar_{k+1})],  t in [t_k, t_{k+1}],
 *
 * with a the interpolation fraction of t in the interval, xbar_k the anchor state (the solver's state at node k), u*_k
 * the nominal input and K_k (inputs x tangent) the Riccati gain, and x (-) xbar = manifold.difference(xbar, x). A
 * LinearController's u = uff + K x cannot represent this: its K would multiply the ambient state, which has more entries
 * than the tangent the gains act on. At a node the law is exact; on a flat manifold it is LinearController's.
 *
 * The anchor of a pre-event node is copied from the node before it, together with its input and gain, as the flat
 * solver copies uff and K there. Because the difference takes the shortest path, the policy is invariant under the
 * double cover: u(t, x) = u(t, x') whenever x and x' differ by the sign of a quaternion block.
 */
class ManifoldLinearController final : public ControllerBase {
 public:
  /** Constructor, leaves the object empty. */
  ManifoldLinearController() = default;

  /**
   * @param [in] timeStamp: The node times.
   * @param [in] anchorStates: The anchor (nominal) state at each node, ambient.
   * @param [in] nominalInputs: The nominal input at each node.
   * @param [in] feedbackGains: The gain at each node, inputs x tangent.
   * @param [in] stateManifold: The manifold the states live on; not null.
   */
  ManifoldLinearController(scalar_array_t timeStamp,
                           vector_array_t anchorStates,
                           vector_array_t nominalInputs,
                           matrix_array_t feedbackGains,
                           std::shared_ptr<const StateManifold> stateManifold);

  ManifoldLinearController(const ManifoldLinearController& other) = default;
  ManifoldLinearController(ManifoldLinearController&& other) = default;
  ManifoldLinearController& operator=(const ManifoldLinearController& other) = default;
  ManifoldLinearController& operator=(ManifoldLinearController&& other) = default;
  ~ManifoldLinearController() override = default;

  ManifoldLinearController* absl_nonnull clone() const override { return new ManifoldLinearController(*this); }

  vector_t computeInput(scalar_t t, const vector_t& x) override;

  void concatenate(const ControllerBase* absl_nonnull nextController, int index, int length) override;

  int size() const override { return static_cast<int>(timeStamp_.size()); }

  ControllerType getType() const override { return ControllerType::MANIFOLD_LINEAR; }

  void clear() override;

  bool empty() const override { return timeStamp_.empty(); }

  void display() const override;

  scalar_array_t controllerEventTimes() const override;

  /** The gain at `time`, linearly interpolated between the nodes (inputs x tangent). */
  matrix_t getFeedbackGain(scalar_t time) const;

  /** The nominal input at `time`, linearly interpolated between the nodes. */
  vector_t getNominalInput(scalar_t time) const;

  /** The anchor state at `time`, interpolated along the manifold between the nodes. */
  vector_t getAnchorState(scalar_t time) const;

  const std::shared_ptr<const StateManifold>& getStateManifold() const { return stateManifoldPtr_; }

  /**
   * Writes the controller at each query time t as [u*; xbar; vec(K)] (inputs + ambient + inputs * tangent entries, K
   * in column-major order), so that a reader applies u*(t) + K(t) (x (-) xbar(t)). xbar(t) is the anchor interpolated
   * along the manifold, K(t) the interpolated gain, and u*(t) = computeInput(t, xbar(t)), the blended law at that
   * anchor (not the interpolated nominal input). Therefore:
   *  - at the node times of the controller the written law is the node's, bit for bit;
   *  - on a flat manifold it equals computeInput() at every time and state up to round-off, as LinearController's
   *    flatten does, so an encoder may sample on any time grid;
   *  - on a curved manifold it equals computeInput() at x = xbar(t), and elsewhere differs by a term bilinear in the
   *    distance between the two anchors and the deviation x (-) xbar(t).
   */
  void flatten(const scalar_array_t& timeArray, const std::vector<std::vector<double>* absl_nonnull>& flatArray2) const override;

  /**
   * Reads a controller written by flatten(). Fails when an array has the wrong length or the dimensions disagree with
   * the manifold.
   */
  static absl::StatusOr<ManifoldLinearController> unFlatten(const size_array_t& stateDim,
                                                            const size_array_t& inputDim,
                                                            const size_array_t& tangentDim,
                                                            const scalar_array_t& timeArray,
                                                            const std::vector<const std::vector<double>* absl_nullable>& flatArray2,
                                                            std::shared_ptr<const StateManifold> stateManifold);

  const scalar_array_t& getTimeStamps() const { return timeStamp_; }
  const vector_array_t& getAnchorStates() const { return anchorStateArray_; }
  const vector_array_t& getNominalInputs() const { return nominalInputArray_; }
  const matrix_array_t& getFeedbackGains() const { return gainArray_; }

 private:
  /** The law of node k at state x: u*_k + K_k (x (-) xbar_k). */
  vector_t nodeInput(size_t k, const vector_t& x) const;

  /** computeInput(), which the interface declares non-const. */
  vector_t blendedInput(scalar_t t, const vector_t& x) const;

  void flattenSingle(scalar_t time, std::vector<double>& flatArray) const;

  scalar_array_t timeStamp_;
  vector_array_t anchorStateArray_;
  vector_array_t nominalInputArray_;
  matrix_array_t gainArray_;
  std::shared_ptr<const StateManifold> stateManifoldPtr_;
};

std::ostream& operator<<(std::ostream& out, const ManifoldLinearController& controller);

}  // namespace ocs2
