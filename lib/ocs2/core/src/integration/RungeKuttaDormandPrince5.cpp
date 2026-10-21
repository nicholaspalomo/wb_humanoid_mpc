/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <algorithm>
#include <limits>

#include <ocs2_core/integration/DormandPrince5Tableau.h>
#include <ocs2_core/integration/RungeKuttaDormandPrince5.h>

namespace ocs2 {

namespace {

/** Helper less comparison for both positive and negative dt case. */
bool lessWithSign(scalar_t t1, scalar_t t2, scalar_t dt) {
  if (dt > 0) {
    return t2 - t1 > std::numeric_limits<scalar_t>::epsilon();
  } else {
    return t1 - t2 > std::numeric_limits<scalar_t>::epsilon();
  }
}

/** Helper to get the min absolute value, t1 and t2 have same sign. */
scalar_t minAbs(scalar_t t1, scalar_t t2) {
  if (t1 > 0) {
    return std::min(t1, t2);
  } else {
    return std::max(t1, t2);
  }
}

/** Helper to get max absolute value, t1 and t2 have same sign. */
scalar_t maxAbs(scalar_t t1, scalar_t t2) {
  if (t1 > 0) {
    return std::max(t1, t2);
  } else {
    return std::min(t1, t2);
  }
}

/** Runge Kutta Dormand-Prince stepper */
class Stepper {
 public:
  using system_func_t = IntegratorBase::system_func_t;

  /**
   * Try to perform one step. If the step is accepted, then state (x), derivative (dxdt), time (t) and step size (dt) are updated.
   * Otherwise only the step size (dt) is updated and false is returned.
   *
   * @param [in] system: System function.
   * @param [in,out] x: current state, updated if step is taken.
   * @param [in,out] dxdt: current derivative wrt. time, updated if step is taken.
   * @param [in,out] t: current time, updated if step is taken.
   * @param [in,out] dt: step size, updated if step is taken.
   * @param [in] absTol: The absolute tolerance error for ode solver.
   * @param [in] relTol: The relative tolerance error for ode solver.
   * @return true if the step is taken, false otherwise..
   */
  bool tryStep(system_func_t& system, vector_t& x, vector_t& dxdt, scalar_t& t, scalar_t& dt, scalar_t absTol, scalar_t relTol) {
    vector_t x_out, dxdt_out;
    doStep(system, x, dxdt, t, dt, x_out, dxdt_out);

    // error estimate
    const vector_t x_err = dt * (dormand_prince5::kDc1 * k1_ + dormand_prince5::kDc3 * k3_ + dormand_prince5::kDc4 * k4_ +
                                 dormand_prince5::kDc5 * k5_ + dormand_prince5::kDc6 * k6_ + dormand_prince5::kDc7 * dxdt_out);

    const scalar_t error = maxError(x, dxdt, x_err, dt, absTol, relTol);
    if (error > 1.0) {
      dt = decreaseStep(dt, error);
      return false;
    } else {
      // accept the step
      t += dt;
      x = x_out;
      dxdt = dxdt_out;
      dt = increaseStep(dt, error);
      return true;
    }
  }

  /**
   * Perform one Dormand-Prince step.
   *
   * @param [in] system: System function.
   * @param [in] x0: current state.
   * @param [in] dxdt: current derivative wrt. time.
   * @param [in] t: current time.
   * @param [in] dt: step size.
   * @param [out] x_out: next state (can be same reference as x0).
   * @param [out] dxdt_out: derivative at next state (can be same reference as dxdt).
   */
  void doStep(
      system_func_t& system, const vector_t& x0, const vector_t& dxdt, scalar_t t, scalar_t dt, vector_t& x_out, vector_t& dxdt_out) {
    k1_ = dxdt;  // k1 = system(x, t) from previous iteration
    vector_t x = x0 + dt * dormand_prince5::kB21 * k1_;
    system(x, k2_, t + dt * dormand_prince5::kA2);
    x.noalias() = x0 + dt * dormand_prince5::kB31 * k1_ + dt * dormand_prince5::kB32 * k2_;
    system(x, k3_, t + dt * dormand_prince5::kA3);
    x.noalias() = x0 + dt * (dormand_prince5::kB41 * k1_ + dormand_prince5::kB42 * k2_ + dormand_prince5::kB43 * k3_);
    system(x, k4_, t + dt * dormand_prince5::kA4);
    x.noalias() =
        x0 + dt * (dormand_prince5::kB51 * k1_ + dormand_prince5::kB52 * k2_ + dormand_prince5::kB53 * k3_ + dormand_prince5::kB54 * k4_);
    system(x, k5_, t + dt * dormand_prince5::kA5);
    x.noalias() = x0 + dt * (dormand_prince5::kB61 * k1_ + dormand_prince5::kB62 * k2_ + dormand_prince5::kB63 * k3_ +
                             dormand_prince5::kB64 * k4_ + dormand_prince5::kB65 * k5_);
    system(x, k6_, t + dt);
    // update x_out and dxdt_out (x_out can be x0 and dxdt_out can be dxdt)
    x_out = x0 + dt * (dormand_prince5::kC1 * k1_ + dormand_prince5::kC3 * k3_ + dormand_prince5::kC4 * k4_ + dormand_prince5::kC5 * k5_ +
                       dormand_prince5::kC6 * k6_);
    system(x_out, dxdt_out, t + dt);
  }

 private:
  /** The orders of the error estimate and of the solution, the exponents of the step-size control. */
  static constexpr int kErrorOrder = 4;
  static constexpr int kStepperOrder = 5;

  /**
   * Estimate the maximal error value.
   *
   * @param [in] x_old: prevoius state.
   * @param [in] dxdt_old: prevoius derivative.
   * @param [in] error: step error estimate.
   * @param [in] dt: step size.
   * @param [in] absTol: The absolute error tolerance.
   * @param [in] relTol: The relative error tolerance.
   * @return maximal error value.
   */
  static scalar_t maxError(
      const vector_t& x_old, const vector_t& dxdt_old, const vector_t& x_err, scalar_t dt, scalar_t absTol, scalar_t relTol) {
    const vector_t err = x_err.array() / (absTol + relTol * (x_old.array().abs() + std::abs(dt) * dxdt_old.array().abs()));
    return err.lpNorm<Eigen::Infinity>();
  }

  /**
   * Decrease the step size
   *
   * @param [in] dt: step size.
   * @param [in] error: maximal error.
   * @return new step size dt.
   */
  static scalar_t decreaseStep(scalar_t dt, scalar_t error) {
    dt *= std::max(dormand_prince5::kStepSizeSafetyFactor * std::pow(error, -1.0 / (kErrorOrder - 1)),
                   dormand_prince5::kMinStepDecreaseFactor);
    return dt;
  }

  /**
   * Increase the step size
   *
   * @param [in] dt: step size.
   * @param [in] error: maximal error.
   * @return new step size dt.
   */
  static scalar_t increaseStep(scalar_t dt, scalar_t error) {
    if (error < dormand_prince5::kStepIncreaseErrorThreshold) {
      error = std::max(std::pow(dormand_prince5::kStepIncreaseLimit, -kStepperOrder), error);
      dt *= dormand_prince5::kStepSizeSafetyFactor * std::pow(error, -1.0 / kStepperOrder);
    }
    return dt;
  }

  /** intermediate derivatives during Runge-Kutta step. */
  vector_t k1_, k2_, k3_, k4_, k5_, k6_;
};

}  // namespace

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void RungeKuttaDormandPrince5::runIntegrateConst(
    system_func_t system, observer_func_t observer, const vector_t& initialState, scalar_t startTime, scalar_t finalTime, scalar_t dt) {
  // TODO(mspieler): This does one redundant system() evaluation at the end.

  // Ensure that finalTime is included by adding a fraction of dt such that: N * dt <= finalTime < (N + 1) * dt.
  finalTime += kFinalTimeSlackStepFraction * dt;

  Stepper stepper;
  scalar_t t = startTime;
  vector_t x = initialState;
  vector_t dxdt;
  system(x, dxdt, t);
  size_t step = 0;
  while (lessWithSign(t + dt, finalTime, dt)) {
    observer(x, t);
    stepper.doStep(system, x, dxdt, t, dt, x, dxdt);
    step++;
    t = startTime + step * dt;
  }
  observer(x, t);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void RungeKuttaDormandPrince5::runIntegrateAdaptive(system_func_t system,
                                                    observer_func_t observer,
                                                    const vector_t& initialState,
                                                    scalar_t startTime,
                                                    scalar_t finalTime,
                                                    scalar_t dtInitial,
                                                    scalar_t absTol,
                                                    scalar_t relTol) {
  Stepper stepper;
  scalar_t t = startTime;
  scalar_t dt = dtInitial;
  vector_t x = initialState;
  vector_t dxdt;
  system(x, dxdt, t);

  while (lessWithSign(t, finalTime, dt)) {
    observer(x, t);

    if (lessWithSign(finalTime, t + dt, dt)) {
      dt = finalTime - t;
    }

    size_t tries = 0;
    while (!stepper.tryStep(system, x, dxdt, t, dt, absTol, relTol)) {
      tries++;
      if (tries > kMaxStepRetries) {
        throw std::runtime_error("[RungeKuttaDormandPrince5] Max number of iterations exceeded");
      }
    }  // end of while loop
  }  // end of while loop
  observer(x, t);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
void RungeKuttaDormandPrince5::runIntegrateTimes(system_func_t system,
                                                 observer_func_t observer,
                                                 const vector_t& initialState,
                                                 typename scalar_array_t::const_iterator beginTimeItr,
                                                 typename scalar_array_t::const_iterator endTimeItr,
                                                 scalar_t dtInitial,
                                                 scalar_t absTol,
                                                 scalar_t relTol) {
  Stepper stepper;
  scalar_t dt = dtInitial;
  vector_t x = initialState;
  vector_t dxdt;
  system(x, dxdt, *beginTimeItr);

  while (true) {
    scalar_t t = *beginTimeItr++;
    observer(x, t);

    if (beginTimeItr == endTimeItr) {
      break;
    }

    size_t tries = 0;
    while (lessWithSign(t, *beginTimeItr, dt)) {
      // adjust stepsize to end up exactly at the observation point
      scalar_t dtCurrent = minAbs(dt, *beginTimeItr - t);
      if (stepper.tryStep(system, x, dxdt, t, dtCurrent, absTol, relTol)) {
        tries = 0;
        // continue with the original step size if dt was reduced due to observation
        dt = maxAbs(dt, dtCurrent);
      } else {
        tries++;
        dt = dtCurrent;
        if (tries > kMaxStepRetries) {
          throw std::runtime_error("[RungeKuttaDormandPrince5] Max number of iterations exceeded");
        }
      }
    }  // end of while loop
  }  // end of while loop
}

}  // namespace ocs2
