/******************************************************************************
Copyright (c) 2026. All rights reserved.

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
 * The integrators EULER, MODIFIED_MIDPOINT, RK4 and ODE45 are native ports of the odeint steppers and integrate loops
 * OCS2 used before. Before odeint was removed, a parity test ran both on the problems and cases below (and more: chaotic,
 * stiff, blowing-up and step-rejecting systems, event handlers, step limits) and required the evaluations of the
 * system and the observed states and times to be bit-identical, which they were. GOLDEN_ROWS are odeint's numbers from
 * that run; the other tests pin the properties of the loops that the observer and the rollouts rely on.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"

#include <ocs2_core/integration/Integrator.h>
#include <ocs2_core/integration/Observer.h>
#include <ocs2_core/integration/OdeBase.h>
#include <ocs2_core/integration/SystemEventHandler.h>

namespace ocs2 {
namespace {

using flow_t = std::function<vector_t(scalar_t, const vector_t&)>;

struct Problem {
  flow_t flow;
  vector_t x0;
  /** Flips the sign of the flow, scaled by 1e100, at every evaluation, so that no step size is ever accepted. */
  bool alternate = false;
};

/** Records every evaluation of the system. */
class RecordingOde final : public OdeBase {
 public:
  RecordingOde(flow_t flow, bool alternate) : flow_(std::move(flow)), alternate_(alternate) {}

  vector_t computeFlowMap(scalar_t t, const vector_t& x) override {
    times.push_back(t);
    if (alternate_) {
      return (times.size() % 2 == 0 ? 1e100 : -1e100) * flow_(t, x);
    }
    return flow_(t, x);
  }

  std::vector<scalar_t> times;

 private:
  flow_t flow_;
  bool alternate_;
};

/** Throws its event id once the first coordinate of the observed state exceeds a threshold. */
class ThresholdEventHandler final : public SystemEventHandler {
 public:
  explicit ThresholdEventHandler(scalar_t threshold) : threshold_(threshold) {}

  std::pair<bool, size_t> checkEvent(OdeBase& /*system*/, scalar_t /*time*/, const vector_t& state) override {
    return {state(0) > threshold_, kEventId};
  }

  static constexpr size_t kEventId = 7;

 private:
  scalar_t threshold_;
};

enum class Mode { kConst, kAdaptive, kTimes };

struct Case {
  Mode mode;
  scalar_t t0 = 0.0;
  scalar_t t1 = 1.0;
  scalar_t dt = 0.01;
  scalar_t absTol = 1e-6;
  scalar_t relTol = 1e-3;
  scalar_array_t times;
};

struct Trajectory {
  scalar_array_t times;
  vector_array_t states;
  size_t evaluations = 0;
};

Problem problem(const std::string& name) {
  if (name == "linear2") {
    const matrix_t A = (matrix_t(2, 2) << -2, -1, 1, 0).finished();
    const vector_t b = (vector_t(2) << 1, 0).finished();
    return {[A, b](scalar_t, const vector_t& x) -> vector_t { return A * x + b; }, vector_t::Zero(2)};
  }
  if (name == "vanDerPol1") {
    return {[](scalar_t, const vector_t& x) -> vector_t {
              vector_t d(2);
              d << x(1), 1.0 * (1.0 - x(0) * x(0)) * x(1) - x(0);
              return d;
            },
            (vector_t(2) << 2.0, 0.0).finished()};
  }
  if (name == "forced") {
    return {[](scalar_t t, const vector_t& x) -> vector_t {
              vector_t d(2);
              d << x(1), -x(0) - 0.1 * x(1) + std::sin(3.0 * t);
              return d;
            },
            (vector_t(2) << 0.3, -0.2).finished()};
  }
  if (name == "coupled13") {
    return {[](scalar_t t, const vector_t& x) -> vector_t {
              const Eigen::Index n = x.size();
              vector_t d(n);
              for (Eigen::Index i = 0; i < n; ++i) {
                d(i) = -0.5 * x(i) + std::sin(x((i + 1) % n)) + 0.1 * std::cos(t + static_cast<scalar_t>(i));
              }
              return d;
            },
            vector_t::LinSpaced(/*size=*/13, /*low=*/-1.0, /*high=*/1.5)};
  }
  if (name == "discontinuous") {
    return {[](scalar_t, const vector_t& x) -> vector_t {
              vector_t d(1);
              d << (x(0) > 0.5 ? -3.0 : 2.0);
              return d;
            },
            (vector_t(1) << 0.0).finished()};
  }
  if (name == "linearStiff3") {
    const matrix_t A = (matrix_t(3, 3) << -1.0, 0.5, 0.0, 0.0, -100.0, 10.0, 0.0, 0.0, -1000.0).finished();
    return {[A](scalar_t, const vector_t& x) -> vector_t { return A * x; }, vector_t::Ones(3)};
  }
  if (name == "alternating") {
    Problem alternating{[](scalar_t, const vector_t& x) -> vector_t { return vector_t::Ones(x.size()); }, vector_t::Zero(2)};
    alternating.alternate = true;
    return alternating;
  }
  throw std::invalid_argument("unknown problem " + name);
}

scalar_array_t uniformTimes() {
  scalar_array_t times;
  for (int i = 0; i <= 20; ++i) {
    times.push_back(0.1 * i);
  }
  return times;
}

Case testCase(const std::string& label) {
  if (label == "const 0..2 dt0.01") return {Mode::kConst, 0.0, 2.0, 0.01};
  if (label == "const 0..1 dt0.3") return {Mode::kConst, 0.0, 1.0, 0.3};
  if (label == "const backward") return {Mode::kConst, 2.0, 0.0, -0.05};
  if (label == "adaptive 1e-6/1e-3") return {Mode::kAdaptive, 0.0, 5.0, 0.01, 1e-6, 1e-3};
  if (label == "adaptive 1e-10/1e-8") return {Mode::kAdaptive, 0.0, 5.0, 0.1, 1e-10, 1e-8};
  if (label == "adaptive backward") return {Mode::kAdaptive, 1.0, 0.0, -0.01, 1e-6, 1e-6};
  if (label == "times uniform dt0.01") return {Mode::kTimes, 0.0, 0.0, 0.01, 1e-6, 1e-3, uniformTimes()};
  if (label == "times uniform dt0.5") return {Mode::kTimes, 0.0, 0.0, 0.5, 1e-6, 1e-3, uniformTimes()};
  if (label == "times irregular") {
    return {Mode::kTimes, 0.0, 0.0, 0.05, 1e-6, 1e-3, {0.0, 0.013, 0.05, 0.051, 0.2, 0.33, 0.7, 0.7000001, 1.1, 1.6, 2.05, 2.5, 3.0}};
  }
  if (label == "times duplicates") return {Mode::kTimes, 0.0, 0.0, 0.1, 1e-6, 1e-3, {0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 2.0}};
  if (label == "times backward") return {Mode::kTimes, 0.0, 0.0, -0.05, 1e-6, 1e-3, {2.0, 1.5, 1.2, 0.0}};
  throw std::invalid_argument("unknown case " + label);
}

Trajectory integrate(IntegratorBase& integrator, const Problem& problem, const Case& c, int maxNumSteps = std::numeric_limits<int>::max()) {
  Trajectory trajectory;
  RecordingOde ode(problem.flow, problem.alternate);
  Observer observer(&trajectory.states, &trajectory.times);
  switch (c.mode) {
    case Mode::kConst:
      integrator.integrateConst(ode, observer, problem.x0, c.t0, c.t1, c.dt, maxNumSteps);
      break;
    case Mode::kAdaptive:
      integrator.integrateAdaptive(ode, observer, problem.x0, c.t0, c.t1, c.dt, c.absTol, c.relTol, maxNumSteps);
      break;
    case Mode::kTimes:
      integrator.integrateTimes(ode, observer, problem.x0, c.times.begin(), c.times.end(), c.dt, c.absTol, c.relTol, maxNumSteps);
      break;
  }
  trajectory.evaluations = ode.times.size();
  return trajectory;
}

Trajectory integrate(IntegratorType type, const std::string& problemName, const std::string& caseLabel) {
  const std::unique_ptr<IntegratorBase> integrator = newIntegrator(type);
  return integrate(*integrator, problem(problemName), testCase(caseLabel));
}

/** |actual - expected| within `relative` of the larger of |expected| and 1. */
void expectClose(scalar_t actual, scalar_t expected, const std::string& what) {
  const scalar_t relative = 1e-12;
  EXPECT_NEAR(actual, expected, relative * std::max(std::abs(expected), 1.0)) << what;
}

struct GoldenRow {
  const char* absl_nonnull problem;
  IntegratorType type;
  const char* absl_nonnull label;
  size_t observations;
  size_t evaluations;
  scalar_t timeSum;
  scalar_t stateSum;
  std::vector<scalar_t> lastState;
};

// clang-format off
const std::vector<GoldenRow> GOLDEN_ROWS = {
    {"linear2", IntegratorType::EULER, "const 0..2 dt0.01", 201, 200, 201, 114.26398781093825, {0.27066600981406458, 0.59535431532797345}},
    {"linear2", IntegratorType::EULER, "const 0..1 dt0.3", 4, 3, 1.7999999999999998, 1.4670000000000001, {0.441, 0.216}},
    {"linear2", IntegratorType::EULER, "const backward", 41, 40, 41, -86.839762954617584, {-13.409502308808854, 7.3695135966842056}},
    {"linear2", IntegratorType::EULER, "adaptive 1e-6/1e-3", 501, 500, 1252.5000000000002, 401.65047782119905, {0.033184257789972853, 0.9602452591676125}},
    {"linear2", IntegratorType::EULER, "adaptive 1e-10/1e-8", 51, 50, 127.5, 41.04638397686589, {0.028632084485111724, 0.96621414030756825}},
    {"linear2", IntegratorType::EULER, "adaptive backward", 101, 100, 50.500000000000007, -72.186196771574132, {-2.678033494476757, 0.9732196650552325}},
    {"linear2", IntegratorType::EULER, "times uniform dt0.01", 21, 200, 21, 11.808928517075907, {0.2706660098140648, 0.59535431532797312}},
    {"linear2", IntegratorType::EULER, "times uniform dt0.5", 21, 20, 21, 12.094189891315127, {0.2701703435345984, 0.60825300187483233}},
    {"linear2", IntegratorType::EULER, "times irregular", 13, 66, 12.2940001, 5.8448269800892678, {0.14554412660506275, 0.80833125791490501}},
    {"linear2", IntegratorType::EULER, "times duplicates", 7, 20, 6, 3.6514080251094305, {0.27017034353459857, 0.608253001874832}},
    {"linear2", IntegratorType::EULER, "times backward", 4, 40, 4.7000000000000002, -7.8517579272840159, {-13.409502308808833, 7.3695135966841931}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "const 0..2 dt0.01", 201, 600, 201, 113.96489207872048, {0.27066943876275379, 0.59399415023595392}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "const 0..1 dt0.3", 4, 9, 1.7999999999999998, 1.3004679428964843, {0.36321129567773436, 0.22888928784374998}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "const backward", 41, 120, 41, -90.984028642040698, {-14.770424342584619, 8.3829063512566666}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-6/1e-3", 501, 1500, 1252.5000000000002, 401.16919813154203, {0.033690015741783388, 0.95957189688801059}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-10/1e-8", 51, 150, 127.5, 40.551744278306558, {0.033717798179516166, 0.95953024771200468}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "adaptive backward", 101, 300, 50.500000000000007, -72.688333327339436, {-2.7182365259509558, 0.99996602330493622}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.01", 21, 600, 21, 11.77845557853483, {0.27066943876275401, 0.59399415023595348}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.5", 21, 60, 21, 11.77596091572288, {0.27055870344572602, 0.59399354729814602}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "times irregular", 13, 198, 12.2940001, 5.7849727948713561, {0.14936122187587553, 0.80083651228092612}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "times duplicates", 7, 60, 6, 3.5471421911630054, {0.27055870344572619, 0.59399354729814557}},
    {"linear2", IntegratorType::MODIFIED_MIDPOINT, "times backward", 4, 120, 4.7000000000000002, -8.2615090650569858, {-14.770424342584587, 8.3829063512566488}},
    {"linear2", IntegratorType::RK4, "const 0..2 dt0.01", 201, 800, 201, 113.96514012431066, {0.27067056640480192, 0.59399415033584113}},
    {"linear2", IntegratorType::RK4, "const 0..1 dt0.3", 4, 12, 1.7999999999999998, 1.3037208958844471, {0.36577455222719529, 0.22762404506350192}},
    {"linear2", IntegratorType::RK4, "const backward", 41, 160, 41, -91.002261230274925, {-14.77810706048675, 8.3890516998561608}},
    {"linear2", IntegratorType::RK4, "adaptive 1e-6/1e-3", 501, 2000, 1252.5000000000002, 401.16959800134686, {0.033689734995403633, 0.95957231800267984}},
    {"linear2", IntegratorType::RK4, "adaptive 1e-10/1e-8", 51, 200, 127.5, 40.555725992185124, {0.033689732445938918, 0.95957229003730637}},
    {"linear2", IntegratorType::RK4, "adaptive backward", 101, 400, 50.500000000000007, -72.688755650894507, {-2.7182818271130564, 0.99999999887865298}},
    {"linear2", IntegratorType::RK4, "times uniform dt0.01", 21, 800, 21, 11.778480851569102, {0.27067056640480219, 0.59399415033584069}},
    {"linear2", IntegratorType::RK4, "times uniform dt0.5", 21, 80, 21, 11.778475357397628, {0.27066981043627614, 0.593994661141933}},
    {"linear2", IntegratorType::RK4, "times irregular", 13, 264, 12.2940001, 5.7852080349358443, {0.14936118903900214, 0.80085173472464433}},
    {"linear2", IntegratorType::RK4, "times duplicates", 7, 80, 6, 3.5479632794939526, {0.27066981043627614, 0.59399466114193278}},
    {"linear2", IntegratorType::RK4, "times backward", 4, 160, 4.7000000000000002, -8.2633174296902379, {-14.778107060486731, 8.3890516998561466}},
    {"linear2", IntegratorType::ODE45, "const 0..2 dt0.01", 201, 1201, 201, 113.96514012931154, {0.27067056647319498, 0.59399415029018454}},
    {"linear2", IntegratorType::ODE45, "const 0..1 dt0.3", 4, 19, 1.7999999999999998, 1.3037994749201636, {0.36591045524973936, 0.22751948275538048}},
    {"linear2", IntegratorType::ODE45, "const backward", 41, 241, 41, -91.002269999963445, {-14.778112207169583, 8.3890561070625278}},
    {"linear2", IntegratorType::ODE45, "adaptive 1e-6/1e-3", 12, 67, 22.632840015284923, 6.9402588105357648, {0.033653432313759966, 0.9595927912282517}},
    {"linear2", IntegratorType::ODE45, "adaptive 1e-10/1e-8", 65, 391, 127.47654810586646, 44.532746780250903, {0.033689734760295575, 0.95957231804748}},
    {"linear2", IntegratorType::ODE45, "adaptive backward", 9, 49, 4.9992276618861746, -6.0715873205854738, {-2.7182824173279427, 1.0000005005610437}},
    {"linear2", IntegratorType::ODE45, "times uniform dt0.01", 21, 133, 21, 11.77848083480462, {0.27067056295066044, 0.593994152966975}},
    {"linear2", IntegratorType::ODE45, "times uniform dt0.5", 21, 121, 21, 11.778480832143572, {0.27067056276815393, 0.59399415310567794}},
    {"linear2", IntegratorType::ODE45, "times irregular", 13, 73, 12.2940001, 5.7851973579023364, {0.1493553059505652, 0.80085603179945586}},
    {"linear2", IntegratorType::ODE45, "times duplicates", 7, 43, 6, 3.5479552992898769, {0.27064941598559444, 0.59401088439815752}},
    {"linear2", IntegratorType::ODE45, "times backward", 4, 37, 4.7000000000000002, -8.2633796120593157, {-14.778500496377122, 8.3893836257680796}},
    {"vanDerPol1", IntegratorType::EULER, "const 0..2 dt0.01", 201, 200, 201, 114.48847477957709, {0.33410789282358644, -1.8200689615488814}},
    {"vanDerPol1", IntegratorType::EULER, "const 0..1 dt0.3", 4, 3, 1.7999999999999998, 5.4338552000000009, {1.6220000000000001, -0.74814479999999994}},
    {"vanDerPol1", IntegratorType::EULER, "const backward", 41, 40, 41, 87.489075862763983, {-1.4665023422244696, 1.294728754367068}},
    {"vanDerPol1", IntegratorType::EULER, "adaptive 1e-6/1e-3", 501, 500, 1252.5000000000002, -416.42476587543064, {-0.87440291392283187, 1.273735873175349}},
    {"vanDerPol1", IntegratorType::EULER, "adaptive 1e-10/1e-8", 51, 50, 127.5, -46.312749867175491, {-1.2150492864499773, 0.99877636750971566}},
    {"vanDerPol1", IntegratorType::EULER, "adaptive backward", 101, 100, 50.500000000000007, 311.33374116767311, {0.38188061329539791, 2.5338938885446267}},
    {"vanDerPol1", IntegratorType::EULER, "times uniform dt0.01", 21, 200, 21, 11.665871637343342, {0.33410789282358805, -1.8200689615488794}},
    {"vanDerPol1", IntegratorType::EULER, "times uniform dt0.5", 21, 20, 21, 12.350754852371157, {0.42137510662500521, -1.7138310530685226}},
    {"vanDerPol1", IntegratorType::EULER, "times irregular", 13, 66, 12.2940001, 5.2552929170238256, {-1.9066119115510627, -1.2238176782830481}},
    {"vanDerPol1", IntegratorType::EULER, "times duplicates", 7, 20, 6, 5.4842717503674061, {0.42137510662500632, -1.7138310530685212}},
    {"vanDerPol1", IntegratorType::EULER, "times backward", 4, 40, 4.7000000000000002, 9.1062737785434091, {-1.4665023422244692, 1.2947287543670702}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "const 0..2 dt0.01", 201, 600, 201, 113.74511179096255, {0.32332308739270244, -1.832966436427637}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "const 0..1 dt0.3", 4, 9, 1.7999999999999998, 5.5873197888665196, {1.5831260458211183, -0.72910511208919426}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "const backward", 41, 120, 41, 83.098321610169066, {-1.2065673006000912, 0.95284256762183717}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-6/1e-3", 501, 1500, 1252.5000000000002, -410.6592067329492, {-0.83705092258388936, 1.3071125195125004}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-10/1e-8", 51, 150, 127.5, -39.830135823777582, {-0.83438435160124569, 1.3094867597716169}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "adaptive backward", 101, 300, 50.500000000000007, 309.32914808549106, {0.3797635671849704, 2.4770562690437519}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.01", 21, 600, 21, 11.580322565548508, {0.32332308739270477, -1.8329664364276346}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.5", 21, 60, 21, 11.586540156074175, {0.32396184475101891, -1.8321614089631992}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "times irregular", 13, 198, 12.2940001, 5.2967738210906745, {-1.8659022071454154, -1.0202492379053669}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "times duplicates", 7, 60, 6, 5.2826463915990729, {0.3239618447510188, -1.8321614089631988}},
    {"vanDerPol1", IntegratorType::MODIFIED_MIDPOINT, "times backward", 4, 120, 4.7000000000000002, 8.7307529429949113, {-1.2065673006000908, 0.95284256762183817}},
    {"vanDerPol1", IntegratorType::RK4, "const 0..2 dt0.01", 201, 800, 201, 113.74454761873791, {0.32331666855014723, -1.83297456597532}},
    {"vanDerPol1", IntegratorType::RK4, "const 0..1 dt0.3", 4, 12, 1.7999999999999998, 5.5882551330626038, {1.5839853250497766, -0.72908836584916625}},
    {"vanDerPol1", IntegratorType::RK4, "const backward", 41, 160, 41, 83.093035717509551, {-1.2053356597896163, 0.95148096007200211}},
    {"vanDerPol1", IntegratorType::RK4, "adaptive 1e-6/1e-3", 501, 2000, 1252.5000000000002, -410.66763165957212, {-0.83707745705448833, 1.3070889313943086}},
    {"vanDerPol1", IntegratorType::RK4, "adaptive 1e-10/1e-8", 51, 200, 127.5, -39.918993298908532, {-0.83714143327368773, 1.3070266462282583}},
    {"vanDerPol1", IntegratorType::RK4, "adaptive backward", 101, 400, 50.500000000000007, 309.32818961919844, {0.37978221409533514, 2.4770215344162332}},
    {"vanDerPol1", IntegratorType::RK4, "times uniform dt0.01", 21, 800, 21, 11.580259526742278, {0.32331666855014995, -1.8329745659753169}},
    {"vanDerPol1", IntegratorType::RK4, "times uniform dt0.5", 21, 80, 21, 11.580670508947691, {0.32333442537119089, -1.8329506568025957}},
    {"vanDerPol1", IntegratorType::RK4, "times irregular", 13, 264, 12.2940001, 5.2942713664135335, {-1.866072865474439, -1.0210645130131526}},
    {"vanDerPol1", IntegratorType::RK4, "times duplicates", 7, 80, 6, 5.2806326143555316, {0.32333442537119184, -1.8329506568025951}},
    {"vanDerPol1", IntegratorType::RK4, "times backward", 4, 160, 4.7000000000000002, 8.7296001529294269, {-1.2053356597896143, 0.95148096007200167}},
    {"vanDerPol1", IntegratorType::ODE45, "const 0..2 dt0.01", 201, 1201, 201, 113.74454729469703, {0.32331666704614098, -1.8329745679857157}},
    {"vanDerPol1", IntegratorType::ODE45, "const 0..1 dt0.3", 4, 19, 1.7999999999999998, 5.5791152596314841, {1.5832823619448426, -0.73092392153598496}},
    {"vanDerPol1", IntegratorType::ODE45, "const backward", 41, 241, 41, 83.09296151137562, {-1.2053389459495245, 0.95148176359680658}},
    {"vanDerPol1", IntegratorType::ODE45, "adaptive 1e-6/1e-3", 16, 121, 33.486789294631912, -5.1407961343362114, {-0.85718357593648642, 1.2891357889246406}},
    {"vanDerPol1", IntegratorType::ODE45, "adaptive 1e-10/1e-8", 106, 745, 254.84931184784136, -105.28556168540837, {-0.83707744329446632, 1.3070889411956106}},
    {"vanDerPol1", IntegratorType::ODE45, "adaptive backward", 12, 73, 6.5751402062677533, 34.298216831802925, {0.37978060391113705, 2.4770213048613199}},
    {"vanDerPol1", IntegratorType::ODE45, "times uniform dt0.01", 21, 133, 21, 11.580253253704447, {0.32331631219412826, -1.8329749471160186}},
    {"vanDerPol1", IntegratorType::ODE45, "times uniform dt0.5", 21, 121, 21, 11.58024761210164, {0.32331601957206901, -1.8329752585631605}},
    {"vanDerPol1", IntegratorType::ODE45, "times irregular", 13, 91, 12.2940001, 5.2892073385973113, {-1.8672766689224594, -1.0215999296346754}},
    {"vanDerPol1", IntegratorType::ODE45, "times duplicates", 7, 43, 6, 5.2789644148871755, {0.32300225521390247, -1.8333580025362697}},
    {"vanDerPol1", IntegratorType::ODE45, "times backward", 4, 49, 4.7000000000000002, 8.7289687050712885, {-1.2023017519887975, 0.94717733058432318}},
    {"forced", IntegratorType::EULER, "const 0..2 dt0.01", 201, 200, 201, 29.520791806656714, {0.078757174205765182, -0.68572347140358791}},
    {"forced", IntegratorType::EULER, "const 0..1 dt0.3", 4, 3, 1.7999999999999998, 0.35598031471977454, {0.12105542186647347, 0.13660681996505614}},
    {"forced", IntegratorType::EULER, "const backward", 41, 40, 41, 12.816518503399735, {-0.35774192968178659, -0.13287713886682573}},
    {"forced", IntegratorType::EULER, "adaptive 1e-6/1e-3", 501, 500, 1252.5000000000002, -59.209037428560229, {-0.15956573584603362, 0.5658108689159741}},
    {"forced", IntegratorType::EULER, "adaptive 1e-10/1e-8", 51, 50, 127.5, -6.3054012262484314, {-0.20971606176459859, 0.55607385965326461}},
    {"forced", IntegratorType::EULER, "adaptive backward", 101, 100, 50.500000000000007, 8.5898046891759741, {0.66369839664405239, -0.45286837799683483}},
    {"forced", IntegratorType::EULER, "times uniform dt0.01", 21, 200, 21, 2.7198394991758201, {0.078757174205765418, -0.68572347140358747}},
    {"forced", IntegratorType::EULER, "times uniform dt0.5", 21, 20, 21, 2.8078566964885914, {0.12885660979424907, -0.66086826421522249}},
    {"forced", IntegratorType::EULER, "times irregular", 13, 66, 12.2940001, 0.19402690152834373, {-0.30194594406696101, 0.13226981260241466}},
    {"forced", IntegratorType::EULER, "times duplicates", 7, 20, 6, 1.2589361424861782, {0.12885660979424951, -0.66086826421522182}},
    {"forced", IntegratorType::EULER, "times backward", 4, 40, 4.7000000000000002, 1.05628707823474, {-0.35774192968178564, -0.13287713886682512}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "const 0..2 dt0.01", 201, 600, 201, 29.345358915875355, {0.07273840049497371, -0.68729160657917254}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "const 0..1 dt0.3", 4, 9, 1.7999999999999998, 0.95279462701726203, {0.27048047350944271, 0.20808795113649919}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "const backward", 41, 120, 41, 11.596681865570243, {-0.30623109618115485, -0.16936877101428061}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-6/1e-3", 501, 1500, 1252.5000000000002, -58.543474754073543, {-0.15383729664263068, 0.56611663782977817}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-10/1e-8", 51, 150, 127.5, -5.6039542672706455, {-0.15319452109302542, 0.56586396389762805}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "adaptive backward", 101, 300, 50.500000000000007, 8.6370729619487925, {0.66643943000931827, -0.44874976983578163}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.01", 21, 600, 21, 2.6989346053379606, {0.072738400494974043, -0.68729160657917188}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.5", 21, 60, 21, 2.6875142762114579, {0.072232659046628192, -0.68683328627290086}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "times irregular", 13, 198, 12.2940001, 0.29209598676669257, {-0.28307621621082335, 0.15935015934355792}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "times duplicates", 7, 60, 6, 1.4331674435413893, {0.072232659046628733, -0.68683328627290019}},
    {"forced", IntegratorType::MODIFIED_MIDPOINT, "times backward", 4, 120, 4.7000000000000002, 1.0306521706857659, {-0.30623109618115429, -0.16936877101427983}},
    {"forced", IntegratorType::RK4, "const 0..2 dt0.01", 201, 800, 201, 29.346514276650527, {0.072743520227158479, -0.68729625646860837}},
    {"forced", IntegratorType::RK4, "const 0..1 dt0.3", 4, 12, 1.7999999999999998, 0.97553563518050712, {0.27446186855166188, 0.21368366480947074}},
    {"forced", IntegratorType::RK4, "const backward", 41, 160, 41, 11.598217252819037, {-0.30626411176126128, -0.16942107827450667}},
    {"forced", IntegratorType::RK4, "adaptive 1e-6/1e-3", 501, 2000, 1252.5000000000002, -58.543965672624019, {-0.15384379927978406, 0.56611919918785591}},
    {"forced", IntegratorType::RK4, "adaptive 1e-10/1e-8", 51, 200, 127.5, -5.6090620550248067, {-0.15384410614496344, 0.56611958005705931}},
    {"forced", IntegratorType::RK4, "adaptive backward", 101, 400, 50.500000000000007, 8.6369111074454104, {0.66644217116193227, -0.44875828221872416}},
    {"forced", IntegratorType::RK4, "times uniform dt0.01", 21, 800, 21, 2.699050244313856, {0.072743520227158617, -0.68729625646860781}},
    {"forced", IntegratorType::RK4, "times uniform dt0.5", 21, 80, 21, 2.6990704140401345, {0.072743993811757474, -0.68729705214274472}},
    {"forced", IntegratorType::RK4, "times irregular", 13, 264, 12.2940001, 0.29317402734999337, {-0.28297520406391308, 0.15927621723382585}},
    {"forced", IntegratorType::RK4, "times duplicates", 7, 80, 6, 1.4378665527651513, {0.072743993811758015, -0.68729705214274439}},
    {"forced", IntegratorType::RK4, "times backward", 4, 160, 4.7000000000000002, 1.0306501417490537, {-0.30626411176126078, -0.16942107827450606}},
    {"forced", IntegratorType::ODE45, "const 0..2 dt0.01", 201, 1201, 201, 29.346514255845864, {0.072743520164863407, -0.68729625639465686}},
    {"forced", IntegratorType::ODE45, "const 0..1 dt0.3", 4, 19, 1.7999999999999998, 0.97487441615979875, {0.2743158745321147, 0.21360217211441274}},
    {"forced", IntegratorType::ODE45, "const backward", 41, 241, 41, 11.59821569877645, {-0.30626410737173432, -0.1694210736907279}},
    {"forced", IntegratorType::ODE45, "adaptive 1e-6/1e-3", 13, 97, 26.623668123342846, -0.48423696518817783, {-0.15380818745115929, 0.56614261403440425}},
    {"forced", IntegratorType::ODE45, "adaptive 1e-10/1e-8", 95, 631, 237.07144068024229, -12.243842743729983, {-0.15384379892049477, 0.56611919910951258}},
    {"forced", IntegratorType::ODE45, "adaptive backward", 9, 55, 4.9649369843995723, 0.98282054223422599, {0.66644231075363258, -0.44875822535768756}},
    {"forced", IntegratorType::ODE45, "times uniform dt0.01", 21, 133, 21, 2.6990502110615653, {0.072743514073114884, -0.6872962585672352}},
    {"forced", IntegratorType::ODE45, "times uniform dt0.5", 21, 121, 21, 2.6990502230941553, {0.07274351456913275, -0.68729625861756305}},
    {"forced", IntegratorType::ODE45, "times irregular", 13, 73, 12.2940001, 0.29315544182070918, {-0.28297249929270402, 0.15927930814121355}},
    {"forced", IntegratorType::ODE45, "times duplicates", 7, 31, 6, 1.4378844460381888, {0.072788170110674316, -0.687350083448377}},
    {"forced", IntegratorType::ODE45, "times backward", 4, 37, 4.7000000000000002, 1.0304553428818366, {-0.30637915989918296, -0.16949963661909848}},
    {"coupled13", IntegratorType::EULER, "const 0..2 dt0.01", 201, 200, 201, 867.04861895714134, {-1.3799771007266286, -1.1518842092588577, -0.63691620580811847, 0.044406747757835936, 0.63009525129684774, 0.99683498996612563, 1.161933630445082, 1.2306763046268785, 1.3218633638566288, 1.4805985735297229, 1.571093598010987, 0.76790623190276963, -0.52940166790373144}},
    {"coupled13", IntegratorType::EULER, "const 0..1 dt0.3", 4, 3, 1.7999999999999998, 14.55942512711003, {-1.1125650245054977, -0.92006313700093867, -0.70852110699772197, -0.4084780303070471, -0.023600138021088748, 0.36415606006478851, 0.66947830772544648, 0.87127478453252871, 1.0181679464445768, 1.1780751710427151, 1.3767926260746959, 1.4395507591385424, 0.3295178920697065}},
    {"coupled13", IntegratorType::EULER, "const backward", 41, 40, 41, 172.01049871009855, {-0.79890687384585457, -0.54389381795524849, -0.26382266749594341, -0.10297657716021333, -0.12363971088344605, -0.22935111527741708, -0.26749541041140701, -0.089776471190163562, 0.072166272146002367, 0.69904754633184496, -0.17433115188165571, 2.6680921162603544, 6.3732166744350005}},
    {"coupled13", IntegratorType::EULER, "adaptive 1e-6/1e-3", 501, 500, 1252.5000000000002, 3018.209285453921, {-0.92788524534475425, 0.45839409031524747, 1.2570305507564861, 1.5848016576977564, 1.6666502219490802, 1.6676471464831693, 1.700266545600309, 1.7835270211797234, 1.8978194560310133, 1.0097292572164063, -0.48126049762940731, -1.3720300565906178, -1.7315449085023582}},
    {"coupled13", IntegratorType::EULER, "adaptive 1e-10/1e-8", 51, 50, 127.5, 301.5082690787122, {-1.074150881360878, 0.37618779948384651, 1.2339679672358097, 1.5852523439065151, 1.674114277932411, 1.6750701204145442, 1.7050511098493903, 1.7871870083950456, 1.912699923718205, 1.0709822678689036, -0.4727469929954915, -1.3853499893481189, -1.7748782393066693}},
    {"coupled13", IntegratorType::EULER, "adaptive backward", 101, 100, 50.500000000000007, 322.70730118626932, {-0.89526669683672311, -0.67607704932627688, -0.43887531037948896, -0.25757143685332851, -0.17035800621072558, -0.13787045552078919, -0.073431646101642573, 0.086376842667353446, 0.32449374174457701, 0.5615123183953683, 0.65813434869436249, 1.2127594350416437, 3.3941058773366106}},
    {"coupled13", IntegratorType::EULER, "times uniform dt0.01", 21, 200, 21, 90.649755981553085, {-1.3799771007266286, -1.1518842092588577, -0.63691620580811881, 0.044406747757835111, 0.63009525129684685, 0.99683498996612496, 1.1619336304450811, 1.230676304626878, 1.3218633638566284, 1.4805985735297225, 1.5710935980109872, 0.76790623190277074, -0.52940166790373056}},
    {"coupled13", IntegratorType::EULER, "times uniform dt0.5", 21, 20, 21, 89.820365689744222, {-1.3755568370311815, -1.1664332099276162, -0.67885866222758529, 0.0017982861930877295, 0.6065234737324523, 0.99231057182811777, 1.168437613196617, 1.2390781588315798, 1.3272448162920132, 1.4836093995612902, 1.5893306570650474, 0.78058410174378878, -0.54849675241782292}},
    {"coupled13", IntegratorType::EULER, "times irregular", 13, 66, 12.2940001, 55.516140150072957, {-1.622190567822674, -1.0916624876800574, -0.130157937099183, 0.69827255049599757, 1.1890054697950623, 1.3915862541645894, 1.4296714487637916, 1.4547901599404784, 1.5551171292066073, 1.7193636555127778, 1.2795955509048218, -0.086293084944827445, -1.1435049545283924}},
    {"coupled13", IntegratorType::EULER, "times duplicates", 7, 20, 6, 28.826748847404538, {-1.375556837031181, -1.1664332099276165, -0.67885866222758529, 0.0017982861930872784, 0.60652347373245186, 0.99231057182811755, 1.1684376131966172, 1.2390781588315796, 1.3272448162920132, 1.4836093995612902, 1.5893306570650472, 0.78058410174378923, -0.54849675241782236}},
    {"coupled13", IntegratorType::EULER, "times backward", 4, 40, 4.7000000000000002, 16.830087824929887, {-0.79890687384585479, -0.5438938179552486, -0.26382266749594346, -0.10297657716021372, -0.123639710883446, -0.22935111527741714, -0.26749541041140656, -0.089776471190163645, 0.072166272146003047, 0.69904754633184407, -0.17433115188165499, 2.6680921162603526, 6.3732166744349943}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "const 0..2 dt0.01", 201, 600, 201, 867.94432458011408, {-1.3803588968349791, -1.1500033729211412, -0.63212020854247797, 0.049052942407570702, 0.63259437945819386, 0.99726995918458816, 1.1611944132227363, 1.2297531161302493, 1.3212766278296848, 1.480264188609306, 1.5690346201359577, 0.7664954411226319, -0.52731634529920046}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "const 0..1 dt0.3", 4, 9, 1.7999999999999998, 14.759088413690389, {-1.1246920848822231, -0.93595573444993119, -0.70191773958242187, -0.36739132636192834, 0.032344381665306121, 0.40427476539259505, 0.68053900079898855, 0.86203391530772666, 1.0056865158161041, 1.1711500917456719, 1.3684990988302683, 1.378893997184784, 0.39462407782199982}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "const backward", 41, 120, 41, 175.83174612738745, {-0.80070654816003406, -0.55204673234689028, -0.27325489687093713, -0.10543046338573102, -0.11709348801106567, -0.21622531062703773, -0.26438008286699061, -0.07502814004782718, 0.031698742370795042, 0.75493520333415298, -0.15504028291407693, 2.7530788578229557, 6.4654281913769047}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-6/1e-3", 501, 1500, 1252.5000000000002, 3024.3366866644869, {-0.91189613906209244, 0.46703706808357748, 1.2594438792681273, 1.5847281149956292, 1.6658220807734867, 1.6668255533878231, 1.6997326826582821, 1.7831762153621284, 1.8958634536903671, 1.0030056912090579, -0.48224720698974349, -1.370658712322556, -1.7256545139580826}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-10/1e-8", 51, 150, 127.5, 307.68191143597852, {-0.91203604115100667, 0.46683775105292002, 1.2592786953564916, 1.5846073140895933, 1.665749535845424, 1.6667954533402747, 1.6997205261260548, 1.7831491487167697, 1.8956522780205025, 1.0024280039855993, -0.48243165283165657, -1.370604989751298, -1.725513111306564}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "adaptive backward", 101, 300, 50.500000000000007, 323.31642640165586, {-0.89544574893408191, -0.67686048936331134, -0.43967456229865581, -0.25761796526815678, -0.16940249562925336, -0.13642520559408333, -0.072379197067065748, 0.086563848494802156, 0.32415610678273538, 0.56247748937035014, 0.65639864299810846, 1.2224763277354831, 3.3998800560798164}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.01", 21, 600, 21, 90.743788595567537, {-1.3803588968349791, -1.1500033729211407, -0.63212020854247841, 0.049052942407569904, 0.63259437945819286, 0.99726995918458683, 1.1611944132227354, 1.2297531161302493, 1.3212766278296848, 1.4802641886093053, 1.5690346201359582, 0.76649544112263379, -0.52731634529919913}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.5", 21, 60, 21, 90.731637042633622, {-1.3804335580149083, -1.1501934405506884, -0.63229313770729934, 0.04897309778823964, 0.63255498392016862, 0.99723152066112986, 1.1611541016014737, 1.2297239269596199, 1.3212634650780239, 1.4802602646053384, 1.5688615381483375, 0.76619097600245878, -0.52722426656810562}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "times irregular", 13, 198, 12.2940001, 55.796741964802116, {-1.613597534086048, -1.0523976448979422, -0.091556322387673597, 0.71721118121604577, 1.1928808122340155, 1.3875310768817422, 1.4235412181675566, 1.4503667438948793, 1.5527280896047073, 1.715088168976951, 1.2585513305546854, -0.086348751606955806, -1.1334772502523633}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "times duplicates", 7, 60, 6, 29.079639196611634, {-1.3804335580149083, -1.1501934405506884, -0.63229313770729989, 0.048973097788239064, 0.63255498392016818, 0.99723152066112963, 1.1611541016014739, 1.2297239269596199, 1.3212634650780242, 1.4802602646053384, 1.5688615381483375, 0.76619097600245989, -0.52722426656810506}},
    {"coupled13", IntegratorType::MODIFIED_MIDPOINT, "times backward", 4, 120, 4.7000000000000002, 17.134211382466297, {-0.80070654816003362, -0.55204673234689006, -0.27325489687093768, -0.10543046338573113, -0.11709348801106574, -0.21622531062703781, -0.26438008286699022, -0.075028140047827374, 0.031698742370795806, 0.75493520333415254, -0.15504028291407621, 2.7530788578229548, 6.4654281913769012}},
    {"coupled13", IntegratorType::RK4, "const 0..2 dt0.01", 201, 800, 201, 867.9455031442227, {-1.3803581420962725, -1.150001455082893, -0.63211846360836144, 0.049053748078104913, 0.63259477703652445, 0.9972703472725003, 1.1611948202615878, 1.2297534108965424, 1.3212767607214657, 1.4802642277175464, 1.569036361847916, 0.76649852099076299, -0.52731727519214122}},
    {"coupled13", IntegratorType::RK4, "const 0..1 dt0.3", 4, 12, 1.7999999999999998, 14.769372253337252, {-1.1246499389836444, -0.93555195505031052, -0.7011235927810413, -0.36666985579910116, 0.032668625312688937, 0.40435534822763014, 0.68061728727109261, 0.86217334059846729, 1.0057973237107274, 1.1711633148950547, 1.3684977492573214, 1.3819390619906542, 0.39388752653025966}},
    {"coupled13", IntegratorType::RK4, "const backward", 41, 160, 41, 175.83698049925698, {-0.80069208344027654, -0.55202852629394561, -0.27325584187815177, -0.10544973663837988, -0.1171305142403621, -0.21621006194288928, -0.2644538215428095, -0.074830089778021788, 0.031394814161421397, 0.75520301338206774, -0.15520559482010871, 2.7535148525132058, 6.4656220098492092}},
    {"coupled13", IntegratorType::RK4, "adaptive 1e-6/1e-3", 501, 2000, 1252.5000000000002, 3024.3421132546719, {-0.91189472432992624, 0.46703907958350149, 1.2594455469992267, 1.5847293351793086, 1.6658228137643738, 1.6668258576268429, 1.6997328057273302, 1.7831764886308177, 1.8958655782568792, 1.0030115238619453, -0.48224534263975938, -1.3706592536710003, -1.7256559348089366}},
    {"coupled13", IntegratorType::RK4, "adaptive 1e-10/1e-8", 51, 200, 127.5, 307.73686275633497, {-0.91189479568670662, 0.46703899715155806, 1.2594454311679446, 1.5847292424515418, 1.6658227602323257, 1.6668258310097903, 1.699732770121847, 1.7831764011167903, 1.8958654639914161, 1.0030113076897587, -0.48224551926853543, -1.3706592168508025, -1.7256558332667598}},
    {"coupled13", IntegratorType::RK4, "adaptive backward", 101, 400, 50.500000000000007, 323.31627688937698, {-0.89544552916953957, -0.67686029001904813, -0.43967472900420812, -0.25761855870372236, -0.16940316944648814, -0.13642554113079255, -0.07237915739340596, 0.086563931779783193, 0.32415541112417068, 0.56247854952141263, 0.65639598394343979, 1.222478009305088, 3.3998823774716662}},
    {"coupled13", IntegratorType::RK4, "times uniform dt0.01", 21, 800, 21, 90.743911282201537, {-1.3803581420962721, -1.1500014550828925, -0.63211846360836166, 0.049053748078104094, 0.63259477703652345, 0.99727034727249975, 1.1611948202615876, 1.2297534108965424, 1.3212767607214657, 1.4802642277175462, 1.5690363618479162, 0.76649852099076454, -0.52731727519214022}},
    {"coupled13", IntegratorType::RK4, "times uniform dt0.5", 21, 80, 21, 90.743901455756259, {-1.3803581393387323, -1.1500012577238996, -0.63211836907664698, 0.049053709596515317, 0.6325947215103197, 0.99727031079635087, 1.1611948072154628, 1.2297534040937463, 1.3212767382268431, 1.4802641292720042, 1.5690362792161181, 0.7664980690414519, -0.52731725215660508}},
    {"coupled13", IntegratorType::RK4, "times irregular", 13, 264, 12.2940001, 55.798299825370563, {-1.6135597306678555, -1.0523400899292463, -0.091523887631071624, 0.71723228705509723, 1.1929005145881291, 1.3875484299105694, 1.4235528014440189, 1.450372094446142, 1.5527322885376429, 1.7151025852266049, 1.2586515267756455, -0.086311793610172868, -1.1335015308116712}},
    {"coupled13", IntegratorType::RK4, "times duplicates", 7, 80, 6, 29.083217074102098, {-1.3803581393387319, -1.1500012577238992, -0.63211836907664731, 0.049053709596514991, 0.63259472151031948, 0.99727031079635053, 1.1611948072154619, 1.2297534040937459, 1.3212767382268429, 1.4802641292720038, 1.5690362792161181, 0.76649806904145301, -0.52731725215660441}},
    {"coupled13", IntegratorType::RK4, "times backward", 4, 160, 4.7000000000000002, 17.134631218532533, {-0.80069208344027643, -0.55202852629394583, -0.27325584187815183, -0.10544973663838011, -0.11713051424036212, -0.21621006194288939, -0.26445382154280922, -0.074830089778021885, 0.031394814161422493, 0.75520301338206541, -0.15520559482010793, 2.7535148525132049, 6.4656220098492039}},
    {"coupled13", IntegratorType::ODE45, "const 0..2 dt0.01", 201, 1201, 201, 867.94550315385277, {-1.3803581420972391, -1.1500014551026669, -0.63211846361707857, 0.049053748082298614, 0.63259477704217515, 0.99727034727612873, 1.1611948202628692, 1.2297534108972432, 1.3212767607238276, 1.4802642277279254, 1.5690363618583036, 0.76649852103452654, -0.52731727519420313}},
    {"coupled13", IntegratorType::ODE45, "const 0..1 dt0.3", 4, 19, 1.7999999999999998, 14.769488614299311, {-1.1246467161454898, -0.93555042179920311, -0.70112967326593056, -0.3666753869874409, 0.032669883537854501, 0.40435810292271884, 0.68061820265470663, 0.86217310102724676, 1.0057975846864202, 1.171168179924754, 1.368512020279816, 1.3819767678353416, 0.39388521248299796}},
    {"coupled13", IntegratorType::ODE45, "const backward", 41, 241, 41, 175.83697641442612, {-0.80069208820767546, -0.55202853343692271, -0.2732558496398963, -0.10544971227394395, -0.11713055351202521, -0.2162099387226411, -0.26445407613841349, -0.074829772524697316, 0.031394418223741116, 0.75520359825094763, -0.15520611936831519, 2.7535148619343612, 6.4656220343532373}},
    {"coupled13", IntegratorType::ODE45, "adaptive 1e-6/1e-3", 9, 55, 16.08471912818062, 46.88669306153264, {-0.91164968439585714, 0.46742874840168425, 1.2595183922747086, 1.5848583182049425, 1.6659150790253336, 1.6668488524378615, 1.6997222637302125, 1.7831470800117397, 1.8958490949675888, 1.0045860197622871, -0.48199186838525776, -1.3707504014406859, -1.7262443568715828}},
    {"coupled13", IntegratorType::ODE45, "adaptive 1e-10/1e-8", 43, 259, 105.10614198741382, 256.60458487474381, {-0.9118947247204211, 0.4670390801117818, 1.259445547362412, 1.5847293352678549, 1.6658228137349138, 1.6668258575635591, 1.6997328056990777, 1.7831764886587571, 1.8958655822445223, 1.0030115288020796, -0.48224534201641056, -1.370659254182014, -1.7256559363575927}},
    {"coupled13", IntegratorType::ODE45, "adaptive backward", 7, 49, 4.3459805031660137, 22.78005689646254, {-0.89544552989874904, -0.67686029185252383, -0.43967473084749492, -0.25761855397313194, -0.16940314378730753, -0.13642553659844545, -0.072379192128750686, 0.086564018335506712, 0.32415523146574254, 0.56247881006284994, 0.6563948845900226, 1.2224778932316045, 3.3998824046436624}},
    {"coupled13", IntegratorType::ODE45, "times uniform dt0.01", 21, 133, 21, 90.743911311744313, {-1.3803581425034352, -1.1500014552262345, -0.63211846337353939, 0.049053748242790074, 0.63259477708892875, 0.99727034727389907, 1.1611948202666635, 1.229753410918119, 1.3212767607818845, 1.4802642279197136, 1.5690363646843786, 0.76649852056452283, -0.52731727513240179}},
    {"coupled13", IntegratorType::ODE45, "times uniform dt0.5", 21, 121, 21, 90.743911312042272, {-1.380358142514464, -1.1500014552496403, -0.6321184633803274, 0.049053748255561823, 0.63259477709994671, 0.99727034727503261, 1.161194820264885, 1.2297534109196901, 1.3212767607850326, 1.4802642279204088, 1.5690363646953571, 0.76649852058061607, -0.52731727512700199}},
    {"coupled13", IntegratorType::ODE45, "times irregular", 13, 73, 12.2940001, 55.798371756703304, {-1.6135623219961788, -1.0523405560887569, -0.091523093560322372, 0.71723276381330292, 1.1929008815413076, 1.3875486325972985, 1.4235528626540019, 1.450372124273414, 1.5527323997463616, 1.7151056914901526, 1.2586648251108026, -0.086313325144313266, -1.1335020200395041}},
    {"coupled13", IntegratorType::ODE45, "times duplicates", 7, 25, 6, 29.085143971490016, {-1.3804384075228842, -1.1502510637843815, -0.63210306950894002, 0.049056570676486291, 0.63263715605852078, 0.99728525383568878, 1.1611988372737849, 1.2297574052785378, 1.3212801792410087, 1.4803217915506659, 1.5712313181818969, 0.76633687551321783, -0.52730079337591196}},
    {"coupled13", IntegratorType::ODE45, "times backward", 4, 43, 4.7000000000000002, 17.132456759264894, {-0.80069402147803115, -0.55202740604923561, -0.27325292752593627, -0.10545244729154189, -0.11710726648088612, -0.21623477683535397, -0.26446963309321297, -0.07470055250188741, 0.030888571873148786, 0.75781736551097711, -0.15991958747986662, 2.7538323281845836, 6.4656257565232043}},
    {"discontinuous", IntegratorType::EULER, "const 0..2 dt0.01", 201, 200, 201, 92.249999999999986, {0.50000000000000011}},
    {"discontinuous", IntegratorType::EULER, "const 0..1 dt0.3", 4, 3, 1.7999999999999998, 0.60000000000000009, {0.30000000000000004}},
    {"discontinuous", IntegratorType::EULER, "const backward", 41, 40, 41, -82.000000000000028, {-4.0000000000000018}},
    {"discontinuous", IntegratorType::EULER, "adaptive 1e-6/1e-3", 501, 500, 1252.5000000000002, 239.24999999999949, {0.50000000000000011}},
    {"discontinuous", IntegratorType::EULER, "adaptive 1e-10/1e-8", 51, 50, 127.5, 24.5, {0.49999999999999983}},
    {"discontinuous", IntegratorType::EULER, "adaptive backward", 101, 100, 50.500000000000007, -101.00000000000006, {-2.0000000000000013}},
    {"discontinuous", IntegratorType::EULER, "times uniform dt0.01", 21, 200, 21, 9.5999999999999694, {0.49999999999999678}},
    {"discontinuous", IntegratorType::EULER, "times uniform dt0.5", 21, 20, 21, 8.9999999999999893, {0.499999999999998}},
    {"discontinuous", IntegratorType::EULER, "times irregular", 13, 66, 12.2940001, 4.8880001999999774, {0.59999999999998843}},
    {"discontinuous", IntegratorType::EULER, "times duplicates", 7, 20, 6, 2.9999999999999978, {0.49999999999999822}},
    {"discontinuous", IntegratorType::EULER, "times backward", 4, 40, 4.7000000000000002, -6.5999999999999979, {-4}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "const 0..2 dt0.01", 201, 600, 201, 92.649999999999636, {0.48749999999999644}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "const 0..1 dt0.3", 4, 9, 1.7999999999999998, 0.59999999999999987, {0.29999999999999993}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "const backward", 41, 120, 41, -81.999999999999943, {-3.9999999999999942}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-6/1e-3", 501, 1500, 1252.5000000000002, 240.39999999999733, {0.48749999999998977}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-10/1e-8", 51, 150, 127.5, 20.999999999999947, {0.37499999999999795}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "adaptive backward", 101, 300, 50.500000000000007, -101.00000000000006, {-2.0000000000000013}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.01", 21, 600, 21, 9.3749999999999769, {0.48749999999999738}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.5", 21, 60, 21, 8.0000000000000053, {0.37500000000000067}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "times irregular", 13, 198, 12.2940001, 4.3755008250000138, {0.46250012500000803}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "times duplicates", 7, 60, 6, 2.2499999999999991, {0.37499999999999978}},
    {"discontinuous", IntegratorType::MODIFIED_MIDPOINT, "times backward", 4, 120, 4.7000000000000002, -6.5999999999999908, {-3.9999999999999916}},
    {"discontinuous", IntegratorType::RK4, "const 0..2 dt0.01", 201, 800, 201, 92.983333333333135, {0.49166666666666475}},
    {"discontinuous", IntegratorType::RK4, "const 0..1 dt0.3", 4, 12, 1.7999999999999998, 1.0999999999999999, {0.54999999999999993}},
    {"discontinuous", IntegratorType::RK4, "const backward", 41, 160, 41, -81.999999999999957, {-3.9999999999999942}},
    {"discontinuous", IntegratorType::RK4, "adaptive 1e-6/1e-3", 501, 2000, 1252.5000000000002, 241.23333333333306, {0.49166666666666142}},
    {"discontinuous", IntegratorType::RK4, "adaptive 1e-10/1e-8", 51, 200, 127.5, 21.833333333333329, {0.5}},
    {"discontinuous", IntegratorType::RK4, "adaptive backward", 101, 400, 50.500000000000007, -101.00000000000004, {-2.0000000000000013}},
    {"discontinuous", IntegratorType::RK4, "times uniform dt0.01", 21, 800, 21, 9.4499999999999922, {0.49166666666666586}},
    {"discontinuous", IntegratorType::RK4, "times uniform dt0.5", 21, 80, 21, 8.5833333333333304, {0.49999999999999956}},
    {"discontinuous", IntegratorType::RK4, "times irregular", 13, 264, 12.2940001, 4.3213347833333371, {0.42500025000000552}},
    {"discontinuous", IntegratorType::RK4, "times duplicates", 7, 80, 6, 2.8333333333333326, {0.499999999999999}},
    {"discontinuous", IntegratorType::RK4, "times backward", 4, 160, 4.7000000000000002, -6.5999999999999908, {-3.999999999999992}},
    {"discontinuous", IntegratorType::ODE45, "const 0..2 dt0.01", 201, 1201, 201, 92.671826917677976, {0.48951805368374363}},
    {"discontinuous", IntegratorType::ODE45, "const 0..1 dt0.3", 4, 19, 1.7999999999999998, 2.0296874999999996, {0.46598930256064691}},
    {"discontinuous", IntegratorType::ODE45, "const backward", 41, 241, 41, -82.000000000000028, {-4.0000000000000027}},
    {"discontinuous", IntegratorType::ODE45, "adaptive 1e-6/1e-3", 1413, 18529, 3692.8180481474742, 699.99028049394224, {0.49679720803143024}},
    {"discontinuous", IntegratorType::ODE45, "adaptive backward", 5, 25, 3.6775000000000002, -2.6449999999999996, {-1.9999999999999998}},
    {"discontinuous", IntegratorType::ODE45, "times uniform dt0.01", 21, 6895, 21, 9.5511072880631698, {0.49654893984563214}},
    {"discontinuous", IntegratorType::ODE45, "times uniform dt0.5", 21, 6883, 21, 9.5511072880631698, {0.49654893984563214}},
    {"discontinuous", IntegratorType::ODE45, "times irregular", 13, 10735, 12.2940001, 4.6045899364019602, {0.49851419976786521}},
    {"discontinuous", IntegratorType::ODE45, "times duplicates", 7, 6817, 6, 2.9884013484063607, {0.49756748194109562}},
    {"discontinuous", IntegratorType::ODE45, "times backward", 4, 31, 4.7000000000000002, -6.5999999999999996, {-4}},
    {"linearStiff3", IntegratorType::EULER, "const 0..2 dt0.01", 201, 200, 201, 6.2790557174095223e+190, {3.9234118055029262e+185, -7.8389767873948472e+188, 7.0550791086553627e+190}},
    {"linearStiff3", IntegratorType::EULER, "const 0..1 dt0.3", 4, 3, 1.7999999999999998, -26369639.218499999, {-23.763500000000001, 272350, -26730899}},
    {"linearStiff3", IntegratorType::EULER, "const backward", 41, 40, 41, 2.0256161698086505e+68, {1.1167836588600778e+63, -2.2313337504024347e+66, 2.0082003753621912e+68}},
    {"linearStiff3", IntegratorType::EULER, "adaptive 1e-10/1e-8", 51, 50, 127.5, 5.923042705918837e+99, {3.3645093267575273e+94, -6.7222896348615392e+97, 6.0500606713753857e+99}},
    {"linearStiff3", IntegratorType::EULER, "adaptive backward", 101, 100, 50.500000000000007, 1.4990328166581251e+104, {7.663559303649364e+98, -1.5311791488691431e+102, 1.3780612339822287e+104}},
    {"linearStiff3", IntegratorType::EULER, "times uniform dt0.01", 21, 200, 21, 6.9767285768991808e+190, {3.9234118055022548e+185, -7.8389767873935067e+188, 7.0550791086541549e+190}},
    {"linearStiff3", IntegratorType::EULER, "times uniform dt0.5", 21, 20, 21, 8.0073539489883676e+39, {4.5484759070027281e+34, -9.0878548621914517e+37, 8.1790693759723073e+39}},
    {"linearStiff3", IntegratorType::EULER, "times irregular", 13, 66, 12.2940001, 1.5000752370937079e+89, {8.4357773577867195e+83, -1.685468316085787e+87, 1.5169214844772081e+89}},
    {"linearStiff3", IntegratorType::EULER, "times duplicates", 7, 20, 6, 8.0882363121094072e+39, {4.5484759070026976e+34, -9.0878548621913913e+37, 8.1790693759722505e+39}},
    {"linearStiff3", IntegratorType::EULER, "times backward", 4, 40, 4.7000000000000002, 1.9858982056947104e+68, {1.1167836588600519e+63, -2.2313337504023842e+66, 2.0082003753621457e+68}},
    {"linearStiff3", IntegratorType::MODIFIED_MIDPOINT, "const 0..1 dt0.3", 4, 9, 1.7999999999999998, -3.6525779270674137e+19, {-205405188333753.34, 4.1039980323045075e+17, -3.6935984636442714e+19}},
    {"linearStiff3", IntegratorType::MODIFIED_MIDPOINT, "const backward", 41, 120, 41, 1.3719840428292107e+169, {7.7149917898458081e+163, -1.5414553596111928e+167, 1.3873098236500736e+169}},
    {"linearStiff3", IntegratorType::MODIFIED_MIDPOINT, "adaptive 1e-10/1e-8", 51, 150, 127.5, 9.3781221992671876e+253, {5.2738961134746744e+248, -1.0537244434722402e+252, 9.4835199912501625e+253}},
    {"linearStiff3", IntegratorType::MODIFIED_MIDPOINT, "adaptive backward", 101, 300, 50.500000000000007, 8.8876220365091226e+226, {4.9711449713109898e+221, -9.9323476526793588e+224, 8.9391128874114226e+226}},
    {"linearStiff3", IntegratorType::MODIFIED_MIDPOINT, "times uniform dt0.5", 21, 60, 21, 3.8541991295380573e+101, {2.1674537159919472e+96, -4.3305725245519102e+99, 3.8975152720967194e+101}},
    {"linearStiff3", IntegratorType::MODIFIED_MIDPOINT, "times irregular", 13, 198, 12.2940001, 1.63296878791476e+253, {9.1831134775296765e+247, -1.8347860728104296e+251, 1.6513074655293867e+253}},
    {"linearStiff3", IntegratorType::MODIFIED_MIDPOINT, "times duplicates", 7, 60, 6, 3.8542312213882751e+101, {2.1674537159918993e+96, -4.3305725245518146e+99, 3.8975152720966336e+101}},
    {"linearStiff3", IntegratorType::MODIFIED_MIDPOINT, "times backward", 4, 120, 4.7000000000000002, 1.3719029850456615e+169, {7.7149917898453056e+163, -1.5414553596110915e+167, 1.3873098236499825e+169}},
    {"linearStiff3", IntegratorType::RK4, "const 0..1 dt0.3", 4, 12, 1.7999999999999998, 3.653066053577164e+25, {2.0543270781810118e+20, -4.1045455046067198e+23, 3.6940909543837532e+25}},
    {"linearStiff3", IntegratorType::RK4, "const backward", 41, 160, 41, 1.0940204092077503e+218, {6.1522780364077702e+212, -1.2292251516742724e+216, 1.106302636506845e+218}},
    {"linearStiff3", IntegratorType::RK4, "adaptive backward", 101, 400, 50.500000000000007, 8.0702021859245081e+280, {4.5312908335431025e+275, -9.0535190854191175e+278, 8.1481671768772061e+280}},
    {"linearStiff3", IntegratorType::RK4, "times uniform dt0.5", 21, 80, 21, 1.1142579626614623e+132, {6.2661055343676855e+126, -1.2519678857666636e+130, 1.1267710971899973e+132}},
    {"linearStiff3", IntegratorType::RK4, "times duplicates", 7, 80, 6, 1.1142576844378319e+132, {6.2661055343674991e+126, -1.2519678857666261e+130, 1.1267710971899638e+132}},
    {"linearStiff3", IntegratorType::RK4, "times backward", 4, 160, 4.7000000000000002, 1.0940165372680439e+218, {6.1522780364072349e+212, -1.2292251516741657e+216, 1.1063026365067492e+218}},
    {"linearStiff3", IntegratorType::ODE45, "const 0..1 dt0.3", 4, 19, 1.7999999999999998, 1.6878902095382078e+36, {9.4919679093075361e+30, -1.896495188279647e+34, 1.7068456694516826e+36}},
    {"linearStiff3", IntegratorType::ODE45, "const backward", 41, 241, 41, 2.8061902011851076e+298, {1.5780805093718891e+293, -3.1530048577250346e+296, 2.8377043719525313e+298}},
    {"linearStiff3", IntegratorType::ODE45, "adaptive 1e-6/1e-3", 1521, 10585, 3781.87775383913, 321.26429785628443, {0.0067723176762241903, -1.5893478811121416e-09, 1.4304130930009146e-07}},
    {"linearStiff3", IntegratorType::ODE45, "adaptive 1e-10/1e-8", 1625, 12193, 3776.8477279009176, 492.82194022387222, {0.006772317675429072, -1.0281398433320458e-13, 9.2532585899885499e-12}},
    {"linearStiff3", IntegratorType::ODE45, "times uniform dt0.01", 21, 4609, 21, 11.263511495361467, {0.13602563673014106, -9.505767454254895e-09, 8.555190708829403e-07}},
    {"linearStiff3", IntegratorType::ODE45, "times uniform dt0.5", 21, 4573, 21, 11.263512195075602, {0.13602563672748433, -4.1962263396263463e-09, 3.77660370566373e-07}},
    {"linearStiff3", IntegratorType::ODE45, "times irregular", 13, 6823, 12.2940001, 9.5335782100356408, {0.050041035224620882, -2.1918334585910893e-09, 1.9726501127320076e-07}},
    {"linearStiff3", IntegratorType::ODE45, "times duplicates", 7, 4561, 6, 5.4645436153552218, {0.13602563672639736, -2.0267668316076935e-09, 1.8240901484469326e-07}},
};
// clang-format on

TEST(Integrators, reproduceTheValuesOdeintComputed) {
  ASSERT_GE(GOLDEN_ROWS.size(), 200u);
  for (const GoldenRow& row : GOLDEN_ROWS) {
    const std::string what = std::string(row.problem) + " / " + integrator_type::toString(row.type) + " / " + row.label;
    const Trajectory trajectory = integrate(row.type, row.problem, row.label);
    ASSERT_EQ(trajectory.times.size(), row.observations) << what;
    EXPECT_EQ(trajectory.evaluations, row.evaluations) << what;
    scalar_t timeSum = 0.0;
    scalar_t stateSum = 0.0;
    for (size_t i = 0; i < trajectory.times.size(); ++i) {
      timeSum += trajectory.times[i];
      stateSum += trajectory.states[i].sum();
    }
    expectClose(timeSum, row.timeSum, what + " time checksum");
    expectClose(stateSum, row.stateSum, what + " state checksum");
    ASSERT_EQ(static_cast<size_t>(trajectory.states.back().size()), row.lastState.size()) << what;
    for (size_t i = 0; i < row.lastState.size(); ++i) {
      expectClose(trajectory.states.back()(static_cast<Eigen::Index>(i)), row.lastState[i], what + " final state");
    }
  }
}

const std::vector<IntegratorType> kOdeintTypes = {IntegratorType::EULER, IntegratorType::MODIFIED_MIDPOINT, IntegratorType::RK4,
                                                  IntegratorType::ODE45};

TEST(Integrators, integrateConstObservesEveryMultipleOfTheStepUpToTheFinalTime) {
  // Evaluations per step of the steppers: Euler 1, modified midpoint 3 (two sub-steps), RK4 4, and ODE45 6 after one
  // evaluation of the initial derivative, which its later steps reuse.
  const std::vector<std::pair<IntegratorType, size_t>> evaluationsPerStep = {
      {IntegratorType::EULER, 1}, {IntegratorType::MODIFIED_MIDPOINT, 3}, {IntegratorType::RK4, 4}, {IntegratorType::ODE45, 6}};
  for (const std::pair<IntegratorType, size_t>& entry : evaluationsPerStep) {
    // 1.0 / 0.3 is not a whole number of steps: the observations stop at the last multiple of dt before 1.0
    const Trajectory trajectory = integrate(entry.first, "linear2", "const 0..1 dt0.3");
    ASSERT_EQ(trajectory.times.size(), 4u) << integrator_type::toString(entry.first);
    for (size_t k = 0; k < trajectory.times.size(); ++k) {
      EXPECT_EQ(trajectory.times[k], 0.0 + static_cast<scalar_t>(k) * 0.3);
    }
    const size_t steps = trajectory.times.size() - 1;
    const size_t initialEvaluations = entry.first == IntegratorType::ODE45 ? 1 : 0;
    EXPECT_EQ(trajectory.evaluations, initialEvaluations + entry.second * steps) << integrator_type::toString(entry.first);

    // A final time that is a whole number of steps is reached despite rounding: 2.0 / 0.01 gives 201 observations.
    const Trajectory whole = integrate(entry.first, "linear2", "const 0..2 dt0.01");
    EXPECT_EQ(whole.times.size(), 201u);
    EXPECT_NEAR(whole.times.back(), 2.0, 1e-12);
  }
}

TEST(Integrators, integrateAdaptiveStartsAtTheInitialStateAndEndsExactlyAtTheFinalTime) {
  for (IntegratorType type : kOdeintTypes) {
    for (const char* absl_nonnull label : {"adaptive 1e-6/1e-3", "adaptive backward"}) {
      const Case c = testCase(label);
      const Problem p = problem("vanDerPol1");
      const std::unique_ptr<IntegratorBase> integrator = newIntegrator(type);
      const Trajectory trajectory = integrate(*integrator, p, c);
      ASSERT_GE(trajectory.times.size(), 2u);
      EXPECT_EQ(trajectory.times.front(), c.t0);
      EXPECT_TRUE(trajectory.states.front() == p.x0);
      EXPECT_NEAR(trajectory.times.back(), c.t1, 1e-12) << integrator_type::toString(type) << " " << label;
      for (size_t k = 1; k < trajectory.times.size(); ++k) {
        EXPECT_GT((trajectory.times[k] - trajectory.times[k - 1]) * (c.t1 - c.t0), 0.0) << "time must advance monotonically";
      }
    }
  }
}

TEST(Integrators, integrateAdaptiveOfOde45MeetsTheTolerance) {
  // x' = -x from 1: tighter tolerances must give a smaller error and more steps.
  const Problem decay{[](scalar_t, const vector_t& x) -> vector_t { return -x; }, vector_t::Ones(1)};
  size_t previousSteps = 0;
  scalar_t previousError = std::numeric_limits<scalar_t>::infinity();
  for (scalar_t tolerance : {1e-4, 1e-7, 1e-10}) {
    const std::unique_ptr<IntegratorBase> integrator = newIntegrator(IntegratorType::ODE45);
    const Trajectory trajectory = integrate(*integrator, decay, {Mode::kAdaptive, 0.0, 2.0, 0.01, tolerance, tolerance});
    const scalar_t error = std::abs(trajectory.states.back()(0) - std::exp(-2.0));
    EXPECT_LT(error, 100.0 * tolerance);
    EXPECT_LT(error, previousError);
    EXPECT_GT(trajectory.times.size(), previousSteps);
    previousError = error;
    previousSteps = trajectory.times.size();
  }
}

TEST(Integrators, integrateTimesObservesExactlyTheRequestedTimes) {
  for (IntegratorType type : kOdeintTypes) {
    for (const char* absl_nonnull label : {"times irregular", "times duplicates", "times backward"}) {
      const Case c = testCase(label);
      const Trajectory trajectory = integrate(type, "forced", label);
      ASSERT_EQ(trajectory.times.size(), c.times.size()) << integrator_type::toString(type) << " " << label;
      for (size_t k = 0; k < c.times.size(); ++k) {
        EXPECT_EQ(trajectory.times[k], c.times[k]);
        if (k > 0 && c.times[k] == c.times[k - 1]) {
          EXPECT_TRUE(trajectory.states[k] == trajectory.states[k - 1]) << "no step between two equal times";
        }
      }
    }
  }
  // A single time is observed with the initial state, and nothing is evaluated.
  const scalar_array_t single = {0.7};
  for (IntegratorType type : kOdeintTypes) {
    const std::unique_ptr<IntegratorBase> integrator = newIntegrator(type);
    const Trajectory trajectory = integrate(*integrator, problem("forced"), {Mode::kTimes, 0.0, 0.0, 0.01, 1e-6, 1e-3, single});
    ASSERT_EQ(trajectory.times.size(), 1u);
    EXPECT_EQ(trajectory.times.front(), 0.7);
    EXPECT_EQ(trajectory.evaluations, 0u);
  }
}

TEST(Integrators, anIntegratorObjectCarriesNothingFromOneIntegrationToTheNext) {
  for (IntegratorType type : kOdeintTypes) {
    for (const char* absl_nonnull label : {"const 0..2 dt0.01", "adaptive 1e-6/1e-3", "times irregular"}) {
      const std::unique_ptr<IntegratorBase> integrator = newIntegrator(type);
      const Trajectory first = integrate(*integrator, problem("coupled13"), testCase(label));
      const Trajectory second = integrate(*integrator, problem("coupled13"), testCase(label));
      EXPECT_EQ(first.times, second.times) << integrator_type::toString(type) << " " << label;
      EXPECT_EQ(first.evaluations, second.evaluations);
      ASSERT_EQ(first.states.size(), second.states.size());
      for (size_t k = 0; k < first.states.size(); ++k) {
        EXPECT_TRUE(first.states[k] == second.states[k]);
      }
    }
  }
}

TEST(Integrators, theMaximumNumberOfFunctionCallsEndsTheIntegration) {
  for (IntegratorType type : kOdeintTypes) {
    for (const char* absl_nonnull label : {"const 0..2 dt0.01", "adaptive 1e-6/1e-3", "times uniform dt0.01"}) {
      const std::unique_ptr<IntegratorBase> integrator = newIntegrator(type);
      try {
        integrate(*integrator, problem("vanDerPol1"), testCase(label), /*maxNumSteps=*/57);
        ADD_FAILURE() << integrator_type::toString(type) << " " << label << " did not stop";
      } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("maximum number of function calls"), std::string::npos) << error.what();
      }
    }
  }
}

TEST(Integrators, anEventOfTheEventHandlerEndsTheIntegrationAtTheObservation) {
  // x' = 1 from 0 crosses the threshold at t = 0.6.
  const Problem ramp{[](scalar_t, const vector_t& x) -> vector_t { return vector_t::Ones(x.size()); }, vector_t::Zero(1)};
  for (IntegratorType type : kOdeintTypes) {
    const std::shared_ptr<SystemEventHandler> handler = std::make_shared<ThresholdEventHandler>(/*threshold=*/0.6);
    const std::unique_ptr<IntegratorBase> integrator = newIntegrator(type, handler);
    scalar_array_t times;
    vector_array_t states;
    Observer observer(&states, &times);
    RecordingOde ode(ramp.flow, /*alternate=*/false);
    try {
      integrator->integrateAdaptive(ode, observer, ramp.x0, /*startTime=*/0.0, /*finalTime=*/5.0, /*dtInitial=*/0.01);
      ADD_FAILURE() << integrator_type::toString(type) << " ran past the event";
    } catch (size_t eventId) {
      EXPECT_EQ(eventId, ThresholdEventHandler::kEventId);
    }
    // the observer sees the state that triggered the event, and the integration stops there
    ASSERT_FALSE(states.empty());
    EXPECT_GT(states.back()(0), 0.6) << integrator_type::toString(type);
    EXPECT_LT(times.back(), 5.0);
    for (size_t k = 0; k + 1 < states.size(); ++k) {
      EXPECT_LE(states[k](0), 0.6);
    }
  }
}

TEST(Integrators, ode45GivesUpAfterFiveHundredRejectedStepSizes) {
  // Every attempt is rejected; the 501st throws, after 1 + 501 * 6 evaluations, before any state is accepted.
  for (Mode mode : {Mode::kAdaptive, Mode::kTimes}) {
    Case c{mode, 0.0, 1e300, 1e200, 0.0, 1e-3};
    c.times = {0.0, 1e300};
    const std::unique_ptr<IntegratorBase> integrator = newIntegrator(IntegratorType::ODE45);
    Trajectory trajectory;
    RecordingOde ode(problem("alternating").flow, /*alternate=*/true);
    Observer observer(&trajectory.states, &trajectory.times);
    try {
      if (mode == Mode::kAdaptive) {
        integrator->integrateAdaptive(ode, observer, problem("alternating").x0, c.t0, c.t1, c.dt, c.absTol, c.relTol);
      } else {
        integrator->integrateTimes(ode, observer, problem("alternating").x0, c.times.begin(), c.times.end(), c.dt, c.absTol, c.relTol);
      }
      ADD_FAILURE() << "no step size can be accepted";
    } catch (const std::runtime_error& error) {
      EXPECT_EQ(std::string(error.what()), "Max number of iterations exceeded (500). A new step size was not found.");
    }
    EXPECT_EQ(ode.times.size(), 1u + 501u * 6u);
    EXPECT_EQ(trajectory.times.size(), 1u);
  }
}

TEST(Integrators, integratorTypesParseTheirNamesAndRefuseRemovedOnes) {
  const std::vector<std::pair<IntegratorType, std::string>> names = {{IntegratorType::EULER, "EULER"},
                                                                     {IntegratorType::ODE45, "ODE45"},
                                                                     {IntegratorType::ODE45_OCS2, "ODE45_OCS2"},
                                                                     {IntegratorType::MODIFIED_MIDPOINT, "MODIFIED_MIDPOINT"},
                                                                     {IntegratorType::RK4, "RK4"}};
  for (const std::pair<IntegratorType, std::string>& entry : names) {
    EXPECT_EQ(integrator_type::toString(entry.first), entry.second);
    EXPECT_EQ(integrator_type::fromString(entry.second), entry.first);
    EXPECT_TRUE(newIntegrator(entry.first) != nullptr);
  }
  // The numbers of the remaining types did not move when the unused ones were removed.
  EXPECT_EQ(static_cast<int>(IntegratorType::EULER), 0);
  EXPECT_EQ(static_cast<int>(IntegratorType::ODE45), 1);
  EXPECT_EQ(static_cast<int>(IntegratorType::ODE45_OCS2), 2);
  EXPECT_EQ(static_cast<int>(IntegratorType::MODIFIED_MIDPOINT), 5);
  EXPECT_EQ(static_cast<int>(IntegratorType::RK4), 6);

  for (const char* absl_nonnull removed : {"ADAMS_BASHFORTH", "BULIRSCH_STOER", "RK5_VARIABLE", "ADAMS_BASHFORTH_MOULTON", "ode45", ""}) {
    try {
      integrator_type::fromString(removed);
      ADD_FAILURE() << removed << " was accepted";
    } catch (const std::invalid_argument& error) {
      const std::string message = error.what();
      EXPECT_NE(message.find(std::string("\"") + removed + "\""), std::string::npos) << message;
      EXPECT_NE(message.find("EULER, MODIFIED_MIDPOINT, ODE45, ODE45_OCS2, RK4"), std::string::npos) << message;
    }
  }
}

}  // namespace
}  // namespace ocs2
