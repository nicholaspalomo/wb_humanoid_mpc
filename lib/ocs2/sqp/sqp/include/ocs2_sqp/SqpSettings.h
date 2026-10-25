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

#pragma once

#include <ocs2_core/Types.h>
#include <ocs2_core/integration/SensitivityIntegrator.h>

#include <hpipm_interface/HpipmInterfaceSettings.h>
#include <ocs2_oc/search_strategy/FilterLinesearch.h>

namespace ocs2 {
namespace sqp {

struct Settings {
  // The defaults of the settings below.
  static constexpr size_t kDefaultMaxSqpIterations = 10;
  static constexpr scalar_t kDefaultDeltaTolerance = 1e-6;
  static constexpr scalar_t kDefaultCostTolerance = 1e-4;
  static constexpr scalar_t kDefaultStepSizeDecay = 0.5;
  static constexpr scalar_t kDefaultMinStepSize = 1e-4;
  static constexpr scalar_t kDefaultTimeStep = 0.01;  // [s]
  static constexpr size_t kDefaultLogSize = 1000;
  static constexpr size_t kDefaultNumThreads = 4;
  static constexpr int kDefaultThreadPriority = 50;

  // Sqp settings
  size_t sqpIteration = kDefaultMaxSqpIterations;  // Maximum number of SQP iterations
  scalar_t deltaTol = kDefaultDeltaTolerance;      // Termination condition : RMS update of x(t) and u(t) are both below this value
  scalar_t costTol = kDefaultCostTolerance;        // Termination condition : (cost{i+1} - (cost{i}) < costTol AND constraints{i+1} < g_min

  // Linesearch - step size rules
  scalar_t alpha_decay = kDefaultStepSizeDecay;  // multiply the step size by this factor every time a linesearch step is rejected.
  scalar_t alpha_min = kDefaultMinStepSize;      // terminate linesearch if the attempted step size is below this threshold

  // Linesearch - step acceptance criteria with c = costs, g = the norm of constraint violation, and w = [x; u]: the
  // FilterLinesearch's, whose defaults they have.
  // (1): IF g{i+1} > g_max REQUIRE g{i+1} < (1-gamma_c) * g{i}
  scalar_t g_max = FilterLinesearch::kDefaultMaxConstraintViolation;
  // (2): ELSE IF (g{i} < g_min AND g{i+1} < g_min AND dc/dw'{i} * delta_w < 0) REQUIRE armijo condition
  scalar_t g_min = FilterLinesearch::kDefaultMinConstraintViolation;
  // Armijo condition: c{i+1} < c{i} + armijoFactor * dc/dw'{i} * delta_w
  scalar_t armijoFactor = FilterLinesearch::kDefaultArmijoFactor;
  // (3): ELSE REQUIRE c{i+1} < (c{i} - gamma_c * g{i}) OR g{i+1} < (1-gamma_c) * g{i}
  scalar_t gamma_c = FilterLinesearch::kDefaultFilterMargin;

  // controller type
  bool useFeedbackPolicy = true;  // true to use feedback, false to use feedforward

  // QP subproblem solver settings
  hpipm_interface::Settings hpipmSettings = hpipm_interface::Settings();

  // Discretization method
  scalar_t dt = kDefaultTimeStep;  // user-defined time discretization
  SensitivityIntegratorType integratorType = SensitivityIntegratorType::RK2;

  bool projectStateInputEqualityConstraints = true;  // Use a projection method to resolve the state-input constraint Cx+Du+e

  // Printing
  bool printSolverStatus = false;      // Print HPIPM status after solving the QP subproblem
  bool printSolverStatistics = false;  // Print benchmarking of the multiple shooting method
  bool printLinesearch = false;        // Print linesearch information

  // Logging
  bool enableLogging = true;
  size_t logSize = kDefaultLogSize;                // the of the last N iterations will be stored
  std::string logFilePath = "/tmp/ocs2/sqp_log/";  // Folder the log will be written to

  // Threading
  size_t nThreads = kDefaultNumThreads;
  int threadPriority = kDefaultThreadPriority;
};

}  // namespace sqp
}  // namespace ocs2
