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

#include "humanoid_nmpc/humanoid_mpc_ipc/test/PolicyTestData.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>

#include "ocs2_core/control/FeedforwardController.h"
#include "ocs2_core/control/LinearController.h"

namespace ocs2::humanoid::ipc::test_data {
namespace {

matrix_t randomMatrix(std::mt19937& generator, size_t rows, size_t cols) {
  std::normal_distribution<scalar_t> value(/*mean=*/0.0, /*stddev=*/10.0);
  matrix_t matrix(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));
  for (Eigen::Index j = 0; j < matrix.cols(); ++j) {
    for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
      matrix(i, j) = value(generator);
    }
  }
  return matrix;
}

}  // namespace

vector_t randomVector(std::mt19937& generator, size_t size) {
  std::normal_distribution<scalar_t> value(/*mean=*/0.0, /*stddev=*/10.0);
  vector_t vector(static_cast<Eigen::Index>(size));
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    vector(i) = value(generator);
  }
  return vector;
}

size_t randomSize(std::mt19937& generator, size_t low, size_t high) {
  return std::uniform_int_distribution<size_t>(low, high)(generator);
}

size_t randomMode(std::mt19937& generator) {
  return std::uniform_int_distribution<size_t>(/*a=*/0, /*b=*/size_t{1} << 40)(generator);
}

SystemObservation randomObservation(std::mt19937& generator, size_t stateDim, size_t inputDim) {
  SystemObservation observation;
  observation.time = std::uniform_real_distribution<scalar_t>(/*a=*/0.0, /*b=*/100.0)(generator);
  observation.state = randomVector(generator, stateDim);
  observation.input = randomVector(generator, inputDim);
  observation.mode = randomMode(generator);
  return observation;
}

ModeSchedule randomModeSchedule(std::mt19937& generator, size_t events, scalar_t startTime) {
  std::uniform_real_distribution<scalar_t> time(startTime, startTime + 1.0);
  scalar_array_t eventTimes(events);
  for (scalar_t& eventTime : eventTimes) {
    eventTime = time(generator);
  }
  std::sort(eventTimes.begin(), eventTimes.end());
  size_array_t modeSequence(events + 1);
  for (size_t& mode : modeSequence) {
    mode = randomMode(generator);
  }
  return ModeSchedule(std::move(eventTimes), std::move(modeSequence));
}

TargetTrajectories randomTargetTrajectories(std::mt19937& generator, size_t nodes, size_t stateDim, size_t inputDim) {
  std::uniform_real_distribution<scalar_t> step(/*a=*/0.0, /*b=*/0.5);
  TargetTrajectories targetTrajectories;
  scalar_t time = std::uniform_real_distribution<scalar_t>(/*a=*/0.0, /*b=*/100.0)(generator);
  for (size_t k = 0; k < nodes; ++k) {
    targetTrajectories.timeTrajectory.push_back(time);
    targetTrajectories.stateTrajectory.push_back(randomVector(generator, stateDim));
    if (inputDim > 0) {
      targetTrajectories.inputTrajectory.push_back(randomVector(generator, inputDim));
    }
    time += step(generator);
  }
  return targetTrajectories;
}

PerformanceIndex randomPerformanceIndex(std::mt19937& generator) {
  std::uniform_real_distribution<scalar_t> value(/*a=*/-1.0e3, /*b=*/1.0e3);
  PerformanceIndex performanceIndex;
  performanceIndex.merit = value(generator);
  performanceIndex.cost = value(generator);
  performanceIndex.dualFeasibilitiesSSE = value(generator);
  performanceIndex.dynamicsViolationSSE = value(generator);
  performanceIndex.equalityConstraintsSSE = value(generator);
  performanceIndex.inequalityConstraintsSSE = value(generator);
  performanceIndex.equalityLagrangian = value(generator);
  performanceIndex.inequalityLagrangian = value(generator);
  return performanceIndex;
}

CommandData randomCommandData(std::mt19937& generator, size_t stateDim, size_t inputDim, size_t targetNodes) {
  CommandData commandData;
  commandData.mpcInitObservation_ = randomObservation(generator, stateDim, inputDim);
  commandData.mpcTargetTrajectories_ = randomTargetTrajectories(generator, targetNodes, stateDim, inputDim);
  return commandData;
}

PrimalSolution randomPrimalSolution(std::mt19937& generator, const PolicyShape& shape) {
  if (shape.nodes == 0 || 3 * shape.events > shape.nodes - 1) {
    throw std::invalid_argument("randomPrimalSolution: a policy has at least one node and at most (nodes - 1) / 3 events");
  }
  // Post-event nodes spread evenly, so that every event has nodes of its own on either side.
  size_array_t postEventIndices;
  for (size_t event = 1; event <= shape.events; ++event) {
    postEventIndices.push_back(event * shape.nodes / (shape.events + 1));
  }

  PrimalSolution solution;
  std::uniform_real_distribution<scalar_t> step(/*a=*/0.005, /*b=*/0.02);
  scalar_t time = std::uniform_real_distribution<scalar_t>(/*a=*/0.0, /*b=*/100.0)(generator);
  size_t nextEvent = 0;
  for (size_t k = 0; k < shape.nodes; ++k) {
    if (nextEvent < postEventIndices.size() && postEventIndices[nextEvent] == k) {
      ++nextEvent;  // the post-event node shares the time of its pre-event node
    } else if (k > 0) {
      time += step(generator);
    }
    solution.timeTrajectory_.push_back(time);
    solution.stateTrajectory_.push_back(randomVector(generator, shape.stateDim));
    solution.inputTrajectory_.push_back(randomVector(generator, shape.inputDim));
  }

  scalar_array_t eventTimes;
  for (const size_t postEventIndex : postEventIndices) {
    eventTimes.push_back(solution.timeTrajectory_[postEventIndex - 1]);
  }
  size_array_t modeSequence(shape.events + 1);
  for (size_t& mode : modeSequence) {
    mode = randomMode(generator);
  }
  solution.postEventIndices_ = std::move(postEventIndices);
  solution.modeSchedule_ = ModeSchedule(std::move(eventTimes), std::move(modeSequence));

  if (shape.controllerType == ControllerType::LINEAR) {
    vector_array_t bias;
    matrix_array_t gain;
    for (size_t k = 0; k < shape.nodes; ++k) {
      bias.push_back(randomVector(generator, shape.inputDim));
      gain.push_back(randomMatrix(generator, shape.inputDim, shape.stateDim));
    }
    solution.controllerPtr_ = std::make_unique<LinearController>(solution.timeTrajectory_, std::move(bias), std::move(gain));
  } else {
    vector_array_t feedforward;
    for (size_t k = 0; k < shape.nodes; ++k) {
      feedforward.push_back(randomVector(generator, shape.inputDim));
    }
    solution.controllerPtr_ = std::make_unique<FeedforwardController>(solution.timeTrajectory_, std::move(feedforward));
  }
  return solution;
}

}  // namespace ocs2::humanoid::ipc::test_data
