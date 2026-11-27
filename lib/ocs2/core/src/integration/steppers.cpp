/******************************************************************************
Copyright (c) 2017, Farbod Farshidian. All rights reserved.

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

// Ported from odeint (Boost Software License, Version 1.0); see the notice in ocs2_core/integration/steppers.h. The
// expressions below keep odeint's form, a coefficient times a vector summed left to right, so that the results match
// odeint's bit for bit: `1.0 * x` and the zero-coefficient terms of runge_kutta4 are kept for that reason.

#include <algorithm>
#include <cmath>

#include <ocs2_core/integration/DormandPrince5Tableau.h>
#include <ocs2_core/integration/steppers.h>

namespace ocs2 {
namespace steppers {

namespace {

/** The midpoint update x1 = x0 + 2 h dxdt of the modified midpoint method spans two substeps of size h. */
constexpr scalar_t kLeapfrogSpanInSubsteps = static_cast<scalar_t>(2);
/** The weight of the final smoothing step of the modified midpoint method, x = 1/2 (x0 + x1 + h dxdt). */
constexpr scalar_t kSmoothingStepWeight = static_cast<scalar_t>(1) / static_cast<scalar_t>(2);

/* The Butcher tableau of odeint's runge_kutta4 (rk4_coefficients_a1, _a2, _a3, _b and _c): kRk4A<i><j> is the weight
 * of stage j in the state of stage i, kRk4B<j> the weight of stage j in the solution and kRk4C<i> the node of stage i,
 * the fraction of dt at which it evaluates the system. */
constexpr scalar_t kRk4A10 = static_cast<scalar_t>(1) / static_cast<scalar_t>(2);
constexpr scalar_t kRk4A20 = static_cast<scalar_t>(0);
constexpr scalar_t kRk4A21 = static_cast<scalar_t>(1) / static_cast<scalar_t>(2);
constexpr scalar_t kRk4A30 = static_cast<scalar_t>(0);
constexpr scalar_t kRk4A31 = static_cast<scalar_t>(0);
constexpr scalar_t kRk4A32 = static_cast<scalar_t>(1);
constexpr scalar_t kRk4B0 = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);
constexpr scalar_t kRk4B1 = static_cast<scalar_t>(1) / static_cast<scalar_t>(3);
constexpr scalar_t kRk4B2 = static_cast<scalar_t>(1) / static_cast<scalar_t>(3);
constexpr scalar_t kRk4B3 = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);
constexpr scalar_t kRk4C1 = static_cast<scalar_t>(1) / static_cast<scalar_t>(2);
constexpr scalar_t kRk4C2 = static_cast<scalar_t>(1) / static_cast<scalar_t>(2);
constexpr scalar_t kRk4C3 = static_cast<scalar_t>(1);

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void Euler::doStep(const system_func_t& system, vector_t& x, scalar_t t, scalar_t dt) {
  system(x, dxdt_, t);
  x = 1.0 * x + dt * dxdt_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void ModifiedMidpoint::doStep(const system_func_t& system, vector_t& x, scalar_t t, scalar_t dt) {
  const scalar_t val1 = static_cast<scalar_t>(1);

  system(x, dxdt_, t);

  const scalar_t h = dt / static_cast<scalar_t>(kSteps);
  const scalar_t h2 = kLeapfrogSpanInSubsteps * h;
  scalar_t th = t + h;

  // x1 = x + h * dxdt
  x1_ = val1 * x + h * dxdt_;
  system(x1_, dxdtMidpoint_, th);
  x0_ = x;

  unsigned short i = 1;
  while (i != kSteps) {
    // tmp = x1; x1 = x0 + h2 * dxdt; x0 = tmp
    const vector_t tmp(x1_);
    x1_ = val1 * x0_ + h2 * dxdtMidpoint_;
    x0_ = tmp;
    th += h;
    system(x1_, dxdtMidpoint_, th);
    i++;
  }

  // x = 0.5 * (x0 + x1 + h * dxdt)
  x = kSmoothingStepWeight * x0_ + kSmoothingStepWeight * x1_ + (kSmoothingStepWeight * h) * dxdtMidpoint_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void RungeKutta4::doStep(const system_func_t& system, vector_t& x, scalar_t t, scalar_t dt) {
  system(x, dxdt_, t);

  xTmp_ = 1.0 * x + (kRk4A10 * dt) * dxdt_;
  system(xTmp_, f0_, t + kRk4C1 * dt);
  xTmp_ = 1.0 * x + (kRk4A20 * dt) * dxdt_ + (kRk4A21 * dt) * f0_;
  system(xTmp_, f1_, t + kRk4C2 * dt);
  xTmp_ = 1.0 * x + (kRk4A30 * dt) * dxdt_ + (kRk4A31 * dt) * f0_ + (kRk4A32 * dt) * f1_;
  system(xTmp_, f2_, t + kRk4C3 * dt);
  x = 1.0 * x + (kRk4B0 * dt) * dxdt_ + (kRk4B1 * dt) * f0_ + (kRk4B2 * dt) * f1_ + (kRk4B3 * dt) * f2_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void DormandPrince5::doStep(const system_func_t& system, vector_t& x, scalar_t t, scalar_t dt) {
  if (!initialized_) {
    system(x, dxdt_, t);
    initialized_ = true;
  }
  step(system, x, dxdt_, t, x, dxdt_, dt);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void DormandPrince5::step(
    const system_func_t& system, const vector_t& in, const vector_t& dxdtIn, scalar_t t, vector_t& out, vector_t& dxdtOut, scalar_t dt) {
  // xTmp = x + dt * b21 * dxdt
  xTmp_ = 1.0 * in + (dt * dormand_prince5::kB21) * dxdtIn;
  system(xTmp_, k2_, t + dt * dormand_prince5::kA2);
  // xTmp = x + dt * b31 * dxdt + dt * b32 * k2
  xTmp_ = 1.0 * in + (dt * dormand_prince5::kB31) * dxdtIn + (dt * dormand_prince5::kB32) * k2_;
  system(xTmp_, k3_, t + dt * dormand_prince5::kA3);
  // xTmp = x + dt * (b41 * dxdt + b42 * k2 + b43 * k3)
  xTmp_ = 1.0 * in + (dt * dormand_prince5::kB41) * dxdtIn + (dt * dormand_prince5::kB42) * k2_ + (dt * dormand_prince5::kB43) * k3_;
  system(xTmp_, k4_, t + dt * dormand_prince5::kA4);
  xTmp_ = 1.0 * in + (dt * dormand_prince5::kB51) * dxdtIn + (dt * dormand_prince5::kB52) * k2_ + (dt * dormand_prince5::kB53) * k3_ +
          (dt * dormand_prince5::kB54) * k4_;
  system(xTmp_, k5_, t + dt * dormand_prince5::kA5);
  xTmp_ = 1.0 * in + (dt * dormand_prince5::kB61) * dxdtIn + (dt * dormand_prince5::kB62) * k2_ + (dt * dormand_prince5::kB63) * k3_ +
          (dt * dormand_prince5::kB64) * k4_ + (dt * dormand_prince5::kB65) * k5_;
  system(xTmp_, k6_, t + dt);
  out = 1.0 * in + (dt * dormand_prince5::kC1) * dxdtIn + (dt * dormand_prince5::kC3) * k3_ + (dt * dormand_prince5::kC4) * k4_ +
        (dt * dormand_prince5::kC5) * k5_ + (dt * dormand_prince5::kC6) * k6_;

  // the new derivative
  system(out, dxdtOut, t + dt);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void DormandPrince5::stepWithError(const system_func_t& system,
                                   const vector_t& in,
                                   const vector_t& dxdtIn,
                                   scalar_t t,
                                   vector_t& out,
                                   vector_t& dxdtOut,
                                   scalar_t dt,
                                   vector_t& xErr) {
  step(system, in, dxdtIn, t, out, dxdtOut, dt);

  // error estimate
  xErr = (dt * dormand_prince5::kDc1) * dxdtIn + (dt * dormand_prince5::kDc3) * k3_ + (dt * dormand_prince5::kDc4) * k4_ +
         (dt * dormand_prince5::kDc5) * k5_ + (dt * dormand_prince5::kDc6) * k6_ + (dt * dormand_prince5::kDc7) * dxdtOut;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
StepResult ControlledDormandPrince5::tryStep(const system_func_t& system, vector_t& x, scalar_t& t, scalar_t& dt) {
  if (!initialized_) {
    system(x, dxdt_, t);
    initialized_ = true;
  }

  stepper_.stepWithError(system, x, dxdt_, t, xNew_, dxdtNew_, dt, xErr_);

  // this overwrites xErr_
  const scalar_t maxRelativeError = error(x, dxdt_, xErr_, dt);

  if (maxRelativeError > 1.0) {
    // error too big, decrease step size and reject this step
    dt = decreaseStep(dt, maxRelativeError, DormandPrince5::kErrorOrder);
    return StepResult::kFail;
  }
  // otherwise, increase step size and accept
  t += dt;
  dt = increaseStep(dt, maxRelativeError, DormandPrince5::kStepperOrder);
  x = xNew_;
  dxdt_ = dxdtNew_;
  return StepResult::kSuccess;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t ControlledDormandPrince5::error(const vector_t& xOld, const vector_t& dxdtOld, vector_t& xErr, scalar_t dt) const {
  // odeint's default_error_checker with a_x = 1 and a_dxdt = 1 (as make_controlled builds it).
  const scalar_t aX = static_cast<scalar_t>(1);
  const scalar_t aDxdt = static_cast<scalar_t>(1) * std::abs(dt);
  xErr = (xErr.array().abs() / (absTol_ + relTol_ * (aX * xOld.array().abs() + aDxdt * dxdtOld.array().abs()))).matrix();
  return xErr.lpNorm<Eigen::Infinity>();
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t ControlledDormandPrince5::decreaseStep(scalar_t dt, scalar_t error, int errorOrder) {
  dt *= std::max(
      static_cast<scalar_t>(dormand_prince5::kStepSizeSafetyFactor * std::pow(error, static_cast<scalar_t>(-1) / (errorOrder - 1))),
      dormand_prince5::kMinStepDecreaseFactor);
  return dt;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t ControlledDormandPrince5::increaseStep(scalar_t dt, scalar_t error, int stepperOrder) {
  if (error < dormand_prince5::kStepIncreaseErrorThreshold) {
    // error should be > 0
    error = std::max(static_cast<scalar_t>(std::pow(dormand_prince5::kStepIncreaseLimit, -static_cast<scalar_t>(stepperOrder))), error);
    // error too small - increase dt and keep the evolution and limit scaling factor to 5.0
    dt *= dormand_prince5::kStepSizeSafetyFactor * std::pow(error, static_cast<scalar_t>(-1) / stepperOrder);
  }
  return dt;
}

}  // namespace steppers
}  // namespace ocs2
