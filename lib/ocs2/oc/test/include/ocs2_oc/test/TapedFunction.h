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

#include <cppad/cppad.hpp>

#include <functional>
#include <memory>
#include <vector>

#include <ocs2_core/Types.h>

namespace ocs2 {
namespace manifold_test {

/**
 * A function y = f(z) recorded once on a CppAD tape (plain AD<double>, no code generation) at z = ones, with its value,
 * Jacobian and per-output Hessian. The function must be branch-free (CppAD conditional expressions only), which the
 * UnitQuaternionMath functions are.
 */
class TapedFunction {
 public:
  using ad_t = CppAD::AD<scalar_t>;
  using ad_vector_t = Eigen::Matrix<ad_t, Eigen::Dynamic, 1>;
  using ad_function_t = std::function<ad_vector_t(const ad_vector_t&)>;

  TapedFunction(const ad_function_t& function, size_t inputDim) : inputDim_(inputDim) {
    ad_vector_t z = ad_vector_t::Ones(inputDim);
    CppAD::Independent(z);
    ad_vector_t y = function(z);
    outputDim_ = y.size();
    tape_ = std::make_unique<CppAD::ADFun<scalar_t>>(z, y);
  }

  size_t getInputDim() const { return inputDim_; }
  size_t getOutputDim() const { return outputDim_; }

  vector_t value(const vector_t& z) const { return toVector(tape_->Forward(/*q=*/0, toStd(z))); }

  /** outputs x inputs */
  matrix_t jacobian(const vector_t& z) const {
    const std::vector<scalar_t> values = tape_->Jacobian(toStd(z));
    return Eigen::Map<const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), outputDim_, inputDim_);
  }

  /** inputs x inputs, of output `outputIndex` */
  matrix_t hessian(const vector_t& z, size_t outputIndex) const {
    const std::vector<scalar_t> values = tape_->Hessian(toStd(z), outputIndex);
    return Eigen::Map<const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), inputDim_, inputDim_);
  }

  static vector_t stack(const vector_t& a, const vector_t& b) {
    vector_t ab(a.size() + b.size());
    ab << a, b;
    return ab;
  }

 private:
  static std::vector<scalar_t> toStd(const vector_t& z) { return std::vector<scalar_t>(z.data(), z.data() + z.size()); }
  static vector_t toVector(const std::vector<scalar_t>& values) { return Eigen::Map<const vector_t>(values.data(), values.size()); }

  size_t inputDim_;
  size_t outputDim_ = 0;
  std::unique_ptr<CppAD::ADFun<scalar_t>> tape_;  // its sweeps modify it, through the const accessors above
};

}  // namespace manifold_test
}  // namespace ocs2
