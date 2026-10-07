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

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include <ocs2_core/integration/Integrator.h>
#include <ocs2_core/integration/IntegratorBase.h>
#include <ocs2_core/integration/steppers.h>

namespace ocs2 {

/*
 * The integrate loops below are ports of odeint's integrate_const, integrate_adaptive and integrate_times (Boost
 * Software License, Version 1.0; see the notice in ocs2_core/integration/steppers.h), as OCS2 called them: which states
 * and times reach the observer, the number of evaluations of the system and the step-count checks are those of odeint.
 */
namespace integration_internal {

using system_func_t = IntegratorBase::system_func_t;
using observer_func_t = IntegratorBase::observer_func_t;

/** t1 < t2 for dt > 0 and t1 > t2 for dt < 0, with epsilon accuracy (odeint's less_with_sign). */
inline bool lessWithSign(scalar_t t1, scalar_t t2, scalar_t dt) {
  if (dt > 0) {
    return t2 - t1 > std::numeric_limits<scalar_t>::epsilon();
  } else {
    return t1 - t2 > std::numeric_limits<scalar_t>::epsilon();
  }
}

/** t1 <= t2 for dt > 0 and t1 >= t2 for dt < 0, with epsilon accuracy (odeint's less_eq_with_sign). */
inline bool lessEqWithSign(scalar_t t1, scalar_t t2, scalar_t dt) {
  if (dt > 0) {
    return t1 - t2 <= std::numeric_limits<scalar_t>::epsilon();
  } else {
    return t2 - t1 <= std::numeric_limits<scalar_t>::epsilon();
  }
}

/** The one of t1 and t2 that is smaller in magnitude, for t1 and t2 of the same sign (odeint's min_abs). */
inline scalar_t minAbs(scalar_t t1, scalar_t t2) {
  if (t1 > 0) {
    return std::min(t1, t2);
  } else {
    return std::max(t1, t2);
  }
}

/** The one of t1 and t2 that is larger in magnitude, for t1 and t2 of the same sign (odeint's max_abs). */
inline scalar_t maxAbs(scalar_t t1, scalar_t t2) {
  if (t1 > 0) {
    return std::max(t1, t2);
  } else {
    return std::min(t1, t2);
  }
}

/** Throws when more than `maxSteps` steps are taken between two observations (odeint's max_step_checker). */
class MaxStepChecker {
 public:
  explicit MaxStepChecker(int maxSteps = 500) : maxSteps_(maxSteps) {}

  void reset() { steps_ = 0; }

  void operator()() {
    if (steps_ >= maxSteps_) {
      throw std::runtime_error("Max number of iterations exceeded (" + std::to_string(maxSteps_) + ").");
    }
    ++steps_;
  }

 private:
  const int maxSteps_;
  int steps_ = 0;
};

/** Throws when a controlled stepper has tried more than `maxSteps` step sizes in a row (odeint's failed_step_checker). */
class FailedStepChecker {
 public:
  explicit FailedStepChecker(int maxSteps = 500) : maxSteps_(maxSteps) {}

  void reset() { steps_ = 0; }

  void operator()() {
    if (steps_ >= maxSteps_) {
      throw std::runtime_error("Max number of iterations exceeded (" + std::to_string(maxSteps_) + "). A new step size was not found.");
    }
    ++steps_;
  }

 private:
  const int maxSteps_;
  int steps_ = 0;
};

/**
 * Steps of constant size `dt` from `startTime` while the next step does not overshoot `endTime`, observing the state
 * before every step and once at the end (odeint's integrate_const for a stepper). Returns the number of steps.
 */
template <class Stepper>
size_t integrateConst(Stepper stepper,
                      const system_func_t& system,
                      vector_t& state,
                      scalar_t startTime,
                      scalar_t endTime,
                      scalar_t dt,
                      const observer_func_t& observer) {
  scalar_t time = startTime;
  int step = 0;
  while (lessEqWithSign(static_cast<scalar_t>(time + dt), endTime, dt)) {
    observer(state, time);
    stepper.doStep(system, state, time, dt);
    // direct computation of the time avoids error propagation happening when using time += dt
    ++step;
    time = startTime + static_cast<scalar_t>(step) * dt;
  }
  observer(state, time);

  return static_cast<size_t>(step);
}

/**
 * integrateConst(), then one last, shorter step that ends exactly at `endTime` when the constant steps fell short of it
 * (odeint's integrate_adaptive for a stepper without step-size control).
 */
template <class Stepper>
size_t integrateAdaptive(Stepper stepper,
                         const system_func_t& system,
                         vector_t& state,
                         scalar_t startTime,
                         scalar_t endTime,
                         scalar_t dt,
                         const observer_func_t& observer) {
  size_t steps = integrateConst(stepper, system, state, startTime, endTime, dt, observer);

  const scalar_t end = startTime + dt * static_cast<scalar_t>(steps);
  if (lessWithSign(end, endTime, dt)) {
    // make a last step to end exactly at endTime
    stepper.doStep(system, state, end, endTime - end);
    steps++;
    observer(state, endTime);
  }
  return steps;
}

/**
 * Adaptive steps from `startTime` to exactly `endTime`, observing the state before every accepted step and once at the
 * end (odeint's integrate_adaptive for a controlled stepper). Returns the number of accepted steps.
 */
inline size_t integrateAdaptiveControlled(steppers::ControlledDormandPrince5 stepper,
                                          const system_func_t& system,
                                          vector_t& state,
                                          scalar_t startTime,
                                          scalar_t endTime,
                                          scalar_t dt,
                                          const observer_func_t& observer) {
  FailedStepChecker failChecker;  // to throw a runtime_error if step size adjustment fails
  size_t count = 0;
  while (lessWithSign(startTime, endTime, dt)) {
    observer(state, startTime);
    if (lessWithSign(endTime, static_cast<scalar_t>(startTime + dt), dt)) {
      dt = endTime - startTime;
    }

    steppers::StepResult result = steppers::StepResult::kFail;
    do {
      result = stepper.tryStep(system, state, startTime, dt);
      failChecker();  // check number of failed steps
    } while (result == steppers::StepResult::kFail);
    failChecker.reset();  // if we reach here, the step was successful -> reset fail checker

    ++count;
  }
  observer(state, startTime);
  return count;
}

/**
 * Integrates through the time points [beginTimeItr, endTimeItr) and observes the state at each of them, stepping at most
 * `dt` at a time and shortening the step that would cross a time point (odeint's integrate_times for a stepper, with a
 * max_step_checker). An empty range observes nothing.
 */
template <class Stepper>
size_t integrateTimes(Stepper stepper,
                      const system_func_t& system,
                      vector_t& state,
                      scalar_array_t::const_iterator beginTimeItr,
                      scalar_array_t::const_iterator endTimeItr,
                      scalar_t dt,
                      const observer_func_t& observer,
                      MaxStepChecker& checker) {
  if (beginTimeItr == endTimeItr) {
    return 0;
  }
  size_t steps = 0;
  scalar_t currentDt = dt;
  while (true) {
    scalar_t currentTime = *beginTimeItr++;
    observer(state, currentTime);
    checker.reset();
    if (beginTimeItr == endTimeItr) {
      break;
    }
    while (lessWithSign(currentTime, static_cast<scalar_t>(*beginTimeItr), currentDt)) {
      currentDt = minAbs(dt, *beginTimeItr - currentTime);
      stepper.doStep(system, state, currentTime, currentDt);
      checker();
      currentTime += currentDt;
      steps++;
    }
  }
  return steps;
}

/**
 * Integrates through the time points [beginTimeItr, endTimeItr) with adaptive steps and observes the state at each of
 * them (odeint's integrate_times for a controlled stepper, with a max_step_checker). An empty range observes nothing.
 */
inline size_t integrateTimesControlled(steppers::ControlledDormandPrince5 stepper,
                                       const system_func_t& system,
                                       vector_t& state,
                                       scalar_array_t::const_iterator beginTimeItr,
                                       scalar_array_t::const_iterator endTimeItr,
                                       scalar_t dt,
                                       const observer_func_t& observer,
                                       MaxStepChecker& checker) {
  if (beginTimeItr == endTimeItr) {
    return 0;
  }
  FailedStepChecker failChecker;  // to throw a runtime_error if step size adjustment fails
  size_t steps = 0;
  while (true) {
    scalar_t currentTime = *beginTimeItr++;
    observer(state, currentTime);
    checker.reset();
    if (beginTimeItr == endTimeItr) {
      break;
    }
    while (lessWithSign(currentTime, static_cast<scalar_t>(*beginTimeItr), dt)) {
      // adjust stepsize to end up exactly at the observation point
      scalar_t currentDt = minAbs(dt, *beginTimeItr - currentTime);
      if (stepper.tryStep(system, state, currentTime, currentDt) == steppers::StepResult::kSuccess) {
        checker();
        ++steps;
        // successful step -> reset the fail counter
        failChecker.reset();
        // continue with the original step size if dt was reduced due to observation
        dt = maxAbs(dt, currentDt);
      } else {
        failChecker();  // check for possible overflow of failed steps in step size adjustment
        dt = currentDt;
      }
    }
  }
  return steps;
}

}  // namespace integration_internal

/**
 * Integrator with a stepper of constant step size (Euler, modified midpoint, RK4). integrateAdaptive() ignores the
 * tolerances, as odeint did for such steppers: it steps by dtInitial and shortens the last step to end at finalTime.
 * @tparam Stepper: one of the steppers of ocs2_core/integration/steppers.h without step-size control.
 */
template <class Stepper>
class Integrator final : public IntegratorBase {
 public:
  using observer_func_t = typename IntegratorBase::observer_func_t;
  using system_func_t = typename IntegratorBase::system_func_t;

  explicit Integrator(std::shared_ptr<SystemEventHandler> eventHandlerPtr = nullptr) : IntegratorBase(std::move(eventHandlerPtr)) {}

  ~Integrator() override = default;

 private:
  void runIntegrateConst(system_func_t system,
                         observer_func_t observer,
                         const vector_t& initialState,
                         scalar_t startTime,
                         scalar_t finalTime,
                         scalar_t dt) override {
    vector_t state = initialState;
    // Ensure that finalTime is included by adding a fraction of dt such that: N * dt <= finalTime < (N + 1) * dt.
    finalTime += 0.1 * dt;
    integration_internal::integrateConst(Stepper(), system, state, startTime, finalTime, dt, observer);
  }

  void runIntegrateAdaptive(system_func_t system,
                            observer_func_t observer,
                            const vector_t& initialState,
                            scalar_t startTime,
                            scalar_t finalTime,
                            scalar_t dtInitial,
                            scalar_t /*AbsTol*/,
                            scalar_t /*RelTol*/) override {
    vector_t state = initialState;
    integration_internal::integrateAdaptive(Stepper(), system, state, startTime, finalTime, dtInitial, observer);
  }

  void runIntegrateTimes(system_func_t system,
                         observer_func_t observer,
                         const vector_t& initialState,
                         typename scalar_array_t::const_iterator beginTimeItr,
                         typename scalar_array_t::const_iterator endTimeItr,
                         scalar_t dtInitial,
                         scalar_t /*AbsTol*/,
                         scalar_t /*RelTol*/) override {
    vector_t state = initialState;
    // maxNumSteps is already checked by the event handler.
    integration_internal::MaxStepChecker maxStepChecker(std::numeric_limits<int>::max());
    integration_internal::integrateTimes(Stepper(), system, state, beginTimeItr, endTimeItr, dtInitial, observer, maxStepChecker);
  }
};

/**
 * ode45: the Dormand-Prince 5(4) pair, with step-size control for integrateAdaptive() and integrateTimes() and constant
 * steps for integrateConst() (odeint's runge_kutta_dopri5, as OCS2 used it).
 */
class ODE45 final : public IntegratorBase {
 public:
  explicit ODE45(std::shared_ptr<SystemEventHandler> eventHandlerPtr = nullptr) : IntegratorBase(std::move(eventHandlerPtr)) {}

  ~ODE45() override = default;

 private:
  void runIntegrateConst(system_func_t system,
                         observer_func_t observer,
                         const vector_t& initialState,
                         scalar_t startTime,
                         scalar_t finalTime,
                         scalar_t dt) override {
    vector_t state = initialState;
    // Ensure that finalTime is included by adding a fraction of dt such that: N * dt <= finalTime < (N + 1) * dt.
    finalTime += 0.1 * dt;
    integration_internal::integrateConst(steppers::DormandPrince5(), system, state, startTime, finalTime, dt, observer);
  }

  void runIntegrateAdaptive(system_func_t system,
                            observer_func_t observer,
                            const vector_t& initialState,
                            scalar_t startTime,
                            scalar_t finalTime,
                            scalar_t dtInitial,
                            scalar_t AbsTol,
                            scalar_t RelTol) override {
    vector_t state = initialState;
    integration_internal::integrateAdaptiveControlled(steppers::ControlledDormandPrince5(AbsTol, RelTol), system, state, startTime,
                                                      finalTime, dtInitial, observer);
  }

  void runIntegrateTimes(system_func_t system,
                         observer_func_t observer,
                         const vector_t& initialState,
                         typename scalar_array_t::const_iterator beginTimeItr,
                         typename scalar_array_t::const_iterator endTimeItr,
                         scalar_t dtInitial,
                         scalar_t AbsTol,
                         scalar_t RelTol) override {
    vector_t state = initialState;
    // maxNumSteps is already checked by the event handler.
    integration_internal::MaxStepChecker maxStepChecker(std::numeric_limits<int>::max());
    integration_internal::integrateTimesControlled(steppers::ControlledDormandPrince5(AbsTol, RelTol), system, state, beginTimeItr,
                                                   endTimeItr, dtInitial, observer, maxStepChecker);
  }
};

/**
 * Euler integrator.
 */
using IntegratorEuler = Integrator<steppers::Euler>;

/**
 * Modified midpoint integrator.
 */
using IntegratorModifiedMidpoint = Integrator<steppers::ModifiedMidpoint>;

/**
 * RK4 integrator.
 */
using IntegratorRK4 = Integrator<steppers::RungeKutta4>;

}  // namespace ocs2
