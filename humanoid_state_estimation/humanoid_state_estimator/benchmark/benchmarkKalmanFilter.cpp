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

/**
 * Per-tick latency and heap allocations of KalmanFilter for a contact-aided
 * base estimator of the size the SA01 would run: base position and velocity
 * plus one 3D position per contact point, the world-frame IMU acceleration as
 * the input, and per stance contact a relative-position, a velocity and a
 * height measurement, every contact in stance.
 *
 * Two callers are compared. "naive" rebuilds its model structs every tick and
 * reads the estimate by value; "real-time" reserves the workspace, builds its
 * structs once, overwrites their values in place and reads into a
 * KalmanFilterState it owns, which is the pattern the KalmanFilter
 * documentation describes.
 *
 *   bazel run
 * //humanoid_state_estimation/humanoid_state_estimator:benchmark_kalman_filter
 *
 * On the robot, pin it to an isolated core and give it a real-time priority to
 * see the worst case the controller would, e.g. `sudo chrt -f 80 taskset -c 3
 * ./benchmark_kalman_filter` on a PREEMPT_RT kernel. The binary needs glibc,
 * whose malloc the allocation counter interposes.
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "ocs2_core/Types.h"

#include "humanoid_state_estimation/humanoid_state_estimator/test/AllocationCounter.h"
#include "humanoid_state_estimator/KalmanFilter.h"

namespace ocs2::humanoid::estimation {
namespace {

using Clock = std::chrono::steady_clock;

constexpr scalar_t kDt = 0.001;

matrix_t identity(Eigen::Index dim) {
  return matrix_t::Identity(dim, dim);
}

double elapsedMicroseconds(Clock::time_point from, Clock::time_point to) {
  return absl::ToDoubleMicroseconds(absl::FromChrono(to - from));
}

struct Phase {
  std::vector<double> microseconds;
  size_t allocations = 0;
};

struct Names {
  std::vector<std::string> contact;
  std::vector<std::string> pos_contact_rt_base_ewrt_world;
  std::vector<std::string> linvel_base_rt_world_ewrt_world;
  std::vector<std::string> height;
};

Names makeNames(int num_contacts) {
  Names names;
  for (int contact = 0; contact < num_contacts; ++contact) {
    names.contact.push_back(absl::StrCat("contact_", contact, "_position"));
    names.pos_contact_rt_base_ewrt_world.push_back(
        absl::StrCat("contact_", contact, "_relative_position"));
    names.linvel_base_rt_world_ewrt_world.push_back(
        absl::StrCat("contact_", contact, "_velocity"));
    names.height.push_back(absl::StrCat("contact_", contact, "_height"));
  }
  return names;
}

std::vector<KalmanFilterProcessModel> makeProcessModel(const Names& names) {
  std::vector<KalmanFilterProcessModel> processModel;
  processModel.push_back(
      {.stateName = "base_position",
       .stateTransition = {{"base_velocity", kDt * identity(3)}},
       .controlInputMatrix = {
           {"imu_acceleration", 0.5 * kDt * kDt * identity(3)}}});
  processModel.push_back(
      {.stateName = "base_velocity",
       .controlInputMatrix = {{"imu_acceleration", kDt * identity(3)}}});
  for (const std::string& contact : names.contact) {
    processModel.push_back(
        {.stateName = contact, .processNoiseCovariance = 1.0e-8 * identity(3)});
  }
  return processModel;
}

std::vector<KalmanFilterInput> makeInputs(
    const vector_t& linacc_base_rt_world_ewrt_world) {
  return {{.name = "imu_acceleration",
           .input = linacc_base_rt_world_ewrt_world,
           .inputNoiseCovariance = 1.0e-4 * identity(3)}};
}

std::vector<KalmanFilterMeasurement> makeMeasurements(
    const Names& names,
    const std::vector<vector_t>& pos_contacts_rt_world_ewrt_world,
    const vector_t& pos_base_rt_world_ewrt_world,
    const vector_t& linvel_base_rt_world_ewrt_world) {
  matrix_t e_z = matrix_t::Zero(1, 3);
  e_z(0, 2) = 1.0;
  std::vector<KalmanFilterMeasurement> measurements;
  for (size_t contact = 0; contact < names.contact.size(); ++contact) {
    measurements.push_back(
        {.name = names.pos_contact_rt_base_ewrt_world[contact],
         .measurement = pos_contacts_rt_world_ewrt_world[contact] -
                        pos_base_rt_world_ewrt_world,
         .measurementModel = {{names.contact[contact], identity(3)},
                              {"base_position", -identity(3)}},
         .measurementNoiseCovariance = 1.0e-6 * identity(3)});
    measurements.push_back(
        {.name = names.linvel_base_rt_world_ewrt_world[contact],
         .measurement = linvel_base_rt_world_ewrt_world,
         .measurementModel = {{"base_velocity", identity(3)}},
         .measurementNoiseCovariance = 1.0e-4 * identity(3)});
    measurements.push_back(
        {.name = names.height[contact],
         .measurement = pos_contacts_rt_world_ewrt_world[contact].tail(1),
         .measurementModel = {{names.contact[contact], e_z}},
         .measurementNoiseCovariance = 1.0e-6 * identity(1)});
  }
  return measurements;
}

void printPhase(absl::string_view label, Phase& phase, int iterations) {
  std::vector<double>& samples = phase.microseconds;
  std::sort(samples.begin(), samples.end());
  double sum = 0.0;
  for (const double sample : samples) {
    sum += sample;
  }
  const size_t count = samples.size();
  absl::PrintF(
      "    %-8s mean %7.2f  p50 %7.2f  p99 %7.2f  p99.9 %7.2f  max %8.2f us   "
      "%6.1f allocations/tick\n",
      label, sum / count, samples[count / 2], samples[count * 99 / 100],
      samples[count * 999 / 1000], samples[count - 1],
      static_cast<double>(phase.allocations) / iterations);
}

void run(int num_contacts, bool real_time_caller, int iterations) {
  const Names names = makeNames(num_contacts);
  std::vector<vector_t> pos_contacts_rt_world_ewrt_world;
  for (int contact = 0; contact < num_contacts; ++contact) {
    pos_contacts_rt_world_ewrt_world.push_back(
        (vector_t(3) << 0.1 * (contact % 4),
         contact < num_contacts / 2 ? 0.1 : -0.1, 0.0)
            .finished());
  }
  vector_t pos_base_rt_world_ewrt_world =
      (vector_t(3) << 0.0, 0.0, 0.8).finished();
  const vector_t linvel_base_rt_world_ewrt_world =
      (vector_t(3) << 0.3, 0.0, 0.0).finished();
  const vector_t linacc_base_rt_world_ewrt_world = vector_t::Zero(3);

  std::vector<KalmanFilterState> initialStates = {
      {.name = "base_position",
       .state = pos_base_rt_world_ewrt_world,
       .stateCovariance = 1.0e-4 * identity(3)},
      {.name = "base_velocity",
       .state = vector_t::Zero(3),
       .stateCovariance = identity(3)}};
  for (int contact = 0; contact < num_contacts; ++contact) {
    initialStates.push_back({.name = names.contact[contact],
                             .state = pos_contacts_rt_world_ewrt_world[contact],
                             .stateCovariance = 1.0e-2 * identity(3)});
  }
  absl::StatusOr<std::unique_ptr<KalmanFilter>> created =
      KalmanFilter::Create(initialStates);
  if (!created.ok()) {
    absl::PrintF("Create failed: %s\n", created.status().message());
    return;
  }
  std::unique_ptr<KalmanFilter> filter = std::move(*created);
  const int measurementDim = 7 * num_contacts;
  if (real_time_caller &&
      !filter->reserve(/*maxInputDim=*/3, measurementDim).ok()) {
    absl::PrintF("reserve failed\n");
    return;
  }

  std::vector<KalmanFilterProcessModel> processModel = makeProcessModel(names);
  std::vector<KalmanFilterInput> inputs =
      makeInputs(linacc_base_rt_world_ewrt_world);
  std::vector<KalmanFilterMeasurement> measurements = makeMeasurements(
      names, pos_contacts_rt_world_ewrt_world, pos_base_rt_world_ewrt_world,
      linvel_base_rt_world_ewrt_world);
  KalmanFilterState est_pos_base_rt_world_ewrt_world;
  KalmanFilterState est_linvel_base_rt_world_ewrt_world;

  Phase build;
  Phase predict;
  Phase correct;
  Phase read;
  Phase total;
  for (Phase* absl_nonnull phase :
       {&build, &predict, &correct, &read, &total}) {
    phase->microseconds.reserve(iterations);
  }

  const int warmup = iterations / 10;
  double checksum = 0.0;
  for (int iteration = -warmup; iteration < iterations; ++iteration) {
    pos_base_rt_world_ewrt_world += kDt * linvel_base_rt_world_ewrt_world;

    const Clock::time_point start = Clock::now();
    size_t allocations = heapAllocationCount();
    if (real_time_caller) {
      inputs[0].input = linacc_base_rt_world_ewrt_world;
      for (int contact = 0; contact < num_contacts; ++contact) {
        measurements[3 * contact].measurement =
            pos_contacts_rt_world_ewrt_world[contact] -
            pos_base_rt_world_ewrt_world;
      }
    } else {
      processModel = makeProcessModel(names);
      inputs = makeInputs(linacc_base_rt_world_ewrt_world);
      measurements = makeMeasurements(names, pos_contacts_rt_world_ewrt_world,
                                      pos_base_rt_world_ewrt_world,
                                      linvel_base_rt_world_ewrt_world);
    }
    const Clock::time_point built = Clock::now();
    const size_t build_allocations = heapAllocationCount() - allocations;

    allocations = heapAllocationCount();
    const absl::Status predicted = filter->predict(processModel, inputs);
    const Clock::time_point after_predict = Clock::now();
    const size_t predict_allocations = heapAllocationCount() - allocations;

    allocations = heapAllocationCount();
    const absl::Status corrected = filter->correct(measurements);
    const Clock::time_point after_correct = Clock::now();
    const size_t correct_allocations = heapAllocationCount() - allocations;

    allocations = heapAllocationCount();
    bool read_ok = true;
    if (real_time_caller) {
      read_ok =
          filter->getState("base_position", est_pos_base_rt_world_ewrt_world)
              .ok() &&
          filter->getState("base_velocity", est_linvel_base_rt_world_ewrt_world)
              .ok();
    } else {
      const absl::StatusOr<KalmanFilterState>
          read_pos_base_rt_world_ewrt_world = filter->getState("base_position");
      const absl::StatusOr<KalmanFilterState>
          read_linvel_base_rt_world_ewrt_world =
              filter->getState("base_velocity");
      read_ok = read_pos_base_rt_world_ewrt_world.ok() &&
                read_linvel_base_rt_world_ewrt_world.ok();
      if (read_ok) {
        est_pos_base_rt_world_ewrt_world.state =
            read_pos_base_rt_world_ewrt_world->state;
        est_linvel_base_rt_world_ewrt_world.state =
            read_linvel_base_rt_world_ewrt_world->state;
      }
    }
    const Clock::time_point after_read = Clock::now();
    const size_t read_allocations = heapAllocationCount() - allocations;

    if (!predicted.ok() || !corrected.ok() || !read_ok) {
      absl::PrintF("step failed: %s %s\n", predicted.message(),
                   corrected.message());
      return;
    }
    checksum += est_pos_base_rt_world_ewrt_world.state(0) +
                est_linvel_base_rt_world_ewrt_world.state(0);
    if (iteration >= 0) {
      build.microseconds.push_back(elapsedMicroseconds(start, built));
      predict.microseconds.push_back(elapsedMicroseconds(built, after_predict));
      correct.microseconds.push_back(
          elapsedMicroseconds(after_predict, after_correct));
      read.microseconds.push_back(
          elapsedMicroseconds(after_correct, after_read));
      total.microseconds.push_back(elapsedMicroseconds(start, after_read));
      build.allocations += build_allocations;
      predict.allocations += predict_allocations;
      correct.allocations += correct_allocations;
      read.allocations += read_allocations;
      total.allocations += build_allocations + predict_allocations +
                           correct_allocations + read_allocations;
    }
  }

  absl::PrintF(
      "  %d contacts (n = %d states, M = %d measurement rows), %s caller:\n",
      num_contacts, 6 + 3 * num_contacts, measurementDim,
      real_time_caller ? "real-time" : "naive");
  printPhase("build", build, iterations);
  printPhase("predict", predict, iterations);
  printPhase("correct", correct, iterations);
  printPhase("read", read, iterations);
  printPhase("TOTAL", total, iterations);
  absl::PrintF(
      "    (checksum %.6f; base velocity estimate x %.4f, truth 0.3000)\n\n",
      checksum, est_linvel_base_rt_world_ewrt_world.state(0));
}

}  // namespace
}  // namespace ocs2::humanoid::estimation

int main() {
  using ocs2::humanoid::estimation::run;
  absl::PrintF(
      "KalmanFilter per-tick cost (single thread, dt = 1 ms, every contact in "
      "stance)\n\n");
  run(/*num_contacts=*/2, /*real_time_caller=*/false, /*iterations=*/50000);
  run(/*num_contacts=*/2, /*real_time_caller=*/true, /*iterations=*/50000);
  run(/*num_contacts=*/8, /*real_time_caller=*/false, /*iterations=*/20000);
  run(/*num_contacts=*/8, /*real_time_caller=*/true, /*iterations=*/20000);
  return 0;
}
