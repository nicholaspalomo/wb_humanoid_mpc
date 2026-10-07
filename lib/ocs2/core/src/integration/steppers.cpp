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

#include <ocs2_core/integration/steppers.h>

#include <algorithm>
#include <cmath>

namespace ocs2 {
namespace steppers {

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
  const scalar_t val05 = static_cast<scalar_t>(1) / static_cast<scalar_t>(2);

  system(x, dxdt_, t);

  const scalar_t h = dt / static_cast<scalar_t>(kSteps);
  const scalar_t h2 = static_cast<scalar_t>(2) * h;
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
  x = val05 * x0_ + val05 * x1_ + (val05 * h) * dxdtMidpoint_;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void RungeKutta4::doStep(const system_func_t& system, vector_t& x, scalar_t t, scalar_t dt) {
  // Butcher tableau of odeint's rk4_coefficients_a1, _a2, _a3, _b and _c.
  const scalar_t a1_0 = static_cast<scalar_t>(1) / static_cast<scalar_t>(2);
  const scalar_t a2_0 = static_cast<scalar_t>(0);
  const scalar_t a2_1 = static_cast<scalar_t>(1) / static_cast<scalar_t>(2);
  const scalar_t a3_0 = static_cast<scalar_t>(0);
  const scalar_t a3_1 = static_cast<scalar_t>(0);
  const scalar_t a3_2 = static_cast<scalar_t>(1);
  const scalar_t b0 = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);
  const scalar_t b1 = static_cast<scalar_t>(1) / static_cast<scalar_t>(3);
  const scalar_t b2 = static_cast<scalar_t>(1) / static_cast<scalar_t>(3);
  const scalar_t b3 = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);
  const scalar_t c1 = static_cast<scalar_t>(1) / static_cast<scalar_t>(2);
  const scalar_t c2 = static_cast<scalar_t>(1) / static_cast<scalar_t>(2);
  const scalar_t c3 = static_cast<scalar_t>(1);

  system(x, dxdt_, t);

  xTmp_ = 1.0 * x + (a1_0 * dt) * dxdt_;
  system(xTmp_, f0_, t + c1 * dt);
  xTmp_ = 1.0 * x + (a2_0 * dt) * dxdt_ + (a2_1 * dt) * f0_;
  system(xTmp_, f1_, t + c2 * dt);
  xTmp_ = 1.0 * x + (a3_0 * dt) * dxdt_ + (a3_1 * dt) * f0_ + (a3_2 * dt) * f1_;
  system(xTmp_, f2_, t + c3 * dt);
  x = 1.0 * x + (b0 * dt) * dxdt_ + (b1 * dt) * f0_ + (b2 * dt) * f1_ + (b3 * dt) * f2_;
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
  const scalar_t a2 = static_cast<scalar_t>(1) / static_cast<scalar_t>(5);
  const scalar_t a3 = static_cast<scalar_t>(3) / static_cast<scalar_t>(10);
  const scalar_t a4 = static_cast<scalar_t>(4) / static_cast<scalar_t>(5);
  const scalar_t a5 = static_cast<scalar_t>(8) / static_cast<scalar_t>(9);

  const scalar_t b21 = static_cast<scalar_t>(1) / static_cast<scalar_t>(5);

  const scalar_t b31 = static_cast<scalar_t>(3) / static_cast<scalar_t>(40);
  const scalar_t b32 = static_cast<scalar_t>(9) / static_cast<scalar_t>(40);

  const scalar_t b41 = static_cast<scalar_t>(44) / static_cast<scalar_t>(45);
  const scalar_t b42 = static_cast<scalar_t>(-56) / static_cast<scalar_t>(15);
  const scalar_t b43 = static_cast<scalar_t>(32) / static_cast<scalar_t>(9);

  const scalar_t b51 = static_cast<scalar_t>(19372) / static_cast<scalar_t>(6561);
  const scalar_t b52 = static_cast<scalar_t>(-25360) / static_cast<scalar_t>(2187);
  const scalar_t b53 = static_cast<scalar_t>(64448) / static_cast<scalar_t>(6561);
  const scalar_t b54 = static_cast<scalar_t>(-212) / static_cast<scalar_t>(729);

  const scalar_t b61 = static_cast<scalar_t>(9017) / static_cast<scalar_t>(3168);
  const scalar_t b62 = static_cast<scalar_t>(-355) / static_cast<scalar_t>(33);
  const scalar_t b63 = static_cast<scalar_t>(46732) / static_cast<scalar_t>(5247);
  const scalar_t b64 = static_cast<scalar_t>(49) / static_cast<scalar_t>(176);
  const scalar_t b65 = static_cast<scalar_t>(-5103) / static_cast<scalar_t>(18656);

  const scalar_t c1 = static_cast<scalar_t>(35) / static_cast<scalar_t>(384);
  const scalar_t c3 = static_cast<scalar_t>(500) / static_cast<scalar_t>(1113);
  const scalar_t c4 = static_cast<scalar_t>(125) / static_cast<scalar_t>(192);
  const scalar_t c5 = static_cast<scalar_t>(-2187) / static_cast<scalar_t>(6784);
  const scalar_t c6 = static_cast<scalar_t>(11) / static_cast<scalar_t>(84);

  // xTmp = x + dt * b21 * dxdt
  xTmp_ = 1.0 * in + (dt * b21) * dxdtIn;
  system(xTmp_, k2_, t + dt * a2);
  // xTmp = x + dt * b31 * dxdt + dt * b32 * k2
  xTmp_ = 1.0 * in + (dt * b31) * dxdtIn + (dt * b32) * k2_;
  system(xTmp_, k3_, t + dt * a3);
  // xTmp = x + dt * (b41 * dxdt + b42 * k2 + b43 * k3)
  xTmp_ = 1.0 * in + (dt * b41) * dxdtIn + (dt * b42) * k2_ + (dt * b43) * k3_;
  system(xTmp_, k4_, t + dt * a4);
  xTmp_ = 1.0 * in + (dt * b51) * dxdtIn + (dt * b52) * k2_ + (dt * b53) * k3_ + (dt * b54) * k4_;
  system(xTmp_, k5_, t + dt * a5);
  xTmp_ = 1.0 * in + (dt * b61) * dxdtIn + (dt * b62) * k2_ + (dt * b63) * k3_ + (dt * b64) * k4_ + (dt * b65) * k5_;
  system(xTmp_, k6_, t + dt);
  out = 1.0 * in + (dt * c1) * dxdtIn + (dt * c3) * k3_ + (dt * c4) * k4_ + (dt * c5) * k5_ + (dt * c6) * k6_;

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
  const scalar_t c1 = static_cast<scalar_t>(35) / static_cast<scalar_t>(384);
  const scalar_t c3 = static_cast<scalar_t>(500) / static_cast<scalar_t>(1113);
  const scalar_t c4 = static_cast<scalar_t>(125) / static_cast<scalar_t>(192);
  const scalar_t c5 = static_cast<scalar_t>(-2187) / static_cast<scalar_t>(6784);
  const scalar_t c6 = static_cast<scalar_t>(11) / static_cast<scalar_t>(84);

  const scalar_t dc1 = c1 - static_cast<scalar_t>(5179) / static_cast<scalar_t>(57600);
  const scalar_t dc3 = c3 - static_cast<scalar_t>(7571) / static_cast<scalar_t>(16695);
  const scalar_t dc4 = c4 - static_cast<scalar_t>(393) / static_cast<scalar_t>(640);
  const scalar_t dc5 = c5 - static_cast<scalar_t>(-92097) / static_cast<scalar_t>(339200);
  const scalar_t dc6 = c6 - static_cast<scalar_t>(187) / static_cast<scalar_t>(2100);
  const scalar_t dc7 = static_cast<scalar_t>(-1) / static_cast<scalar_t>(40);

  step(system, in, dxdtIn, t, out, dxdtOut, dt);

  // error estimate
  xErr = (dt * dc1) * dxdtIn + (dt * dc3) * k3_ + (dt * dc4) * k4_ + (dt * dc5) * k5_ + (dt * dc6) * k6_ + (dt * dc7) * dxdtOut;
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
  dt *= std::max(static_cast<scalar_t>(static_cast<scalar_t>(9) / static_cast<scalar_t>(10) *
                                       std::pow(error, static_cast<scalar_t>(-1) / (errorOrder - 1))),
                 static_cast<scalar_t>(static_cast<scalar_t>(1) / static_cast<scalar_t>(5)));
  return dt;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
scalar_t ControlledDormandPrince5::increaseStep(scalar_t dt, scalar_t error, int stepperOrder) {
  if (error < 0.5) {
    // error should be > 0
    error = std::max(static_cast<scalar_t>(std::pow(static_cast<scalar_t>(5.0), -static_cast<scalar_t>(stepperOrder))), error);
    // error too small - increase dt and keep the evolution and limit scaling factor to 5.0
    dt *= static_cast<scalar_t>(9) / static_cast<scalar_t>(10) * std::pow(error, static_cast<scalar_t>(-1) / stepperOrder);
  }
  return dt;
}

}  // namespace steppers
}  // namespace ocs2
