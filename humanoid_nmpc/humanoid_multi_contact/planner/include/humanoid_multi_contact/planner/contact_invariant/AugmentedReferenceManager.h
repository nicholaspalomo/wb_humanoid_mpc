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

#include <cstddef>
#include <memory>
#include <utility>

#include "ocs2_core/Types.h"
#include "ocs2_core/reference/ModeSchedule.h"
#include "ocs2_core/reference/TargetTrajectories.h"
#include "ocs2_oc/synchronized_module/ReferenceManagerInterface.h"

namespace ocs2::humanoid::multi_contact {

/**
 * The reference manager of a contact-invariant problem: the physical
 * problem's, run before every solve on the physical part of the augmented
 * initial state. The mode schedule and the target trajectories are the
 * physical manager's, in the physical layout, which the physical terms read.
 *
 * As thread-safe as the physical manager.
 */
class AugmentedReferenceManager final : public ReferenceManagerInterface {
 public:
  /**
   * Wraps `physical`, whose state has `physicalStateDim` entries. A
   * std::shared_ptr because the solver holds its reference manager by one
   * (SolverBase::setReferenceManager()), and the physical terms keep
   * references to `physical`.
   */
  AugmentedReferenceManager(std::shared_ptr<ReferenceManagerInterface> physical,
                            int physicalStateDim);
  ~AugmentedReferenceManager() override = default;
  AugmentedReferenceManager(const AugmentedReferenceManager&) = delete;
  AugmentedReferenceManager& operator=(const AugmentedReferenceManager&) =
      delete;
  AugmentedReferenceManager(AugmentedReferenceManager&&) = delete;
  AugmentedReferenceManager& operator=(AugmentedReferenceManager&&) = delete;

  void preSolverRun(scalar_t initTime,
                    scalar_t finalTime,
                    const vector_t& initState,
                    size_t initMode) override;
  void reset() override { physical_->reset(); }

  const ModeSchedule& getModeSchedule() const override {
    return physical_->getModeSchedule();
  }
  void setModeSchedule(const ModeSchedule& modeSchedule) override {
    physical_->setModeSchedule(modeSchedule);
  }
  void setModeSchedule(ModeSchedule&& modeSchedule) override {
    physical_->setModeSchedule(std::move(modeSchedule));
  }
  const TargetTrajectories& getTargetTrajectories() const override {
    return physical_->getTargetTrajectories();
  }
  void setTargetTrajectories(
      const TargetTrajectories& targetTrajectories) override {
    physical_->setTargetTrajectories(targetTrajectories);
  }
  void setTargetTrajectories(TargetTrajectories&& targetTrajectories) override {
    physical_->setTargetTrajectories(std::move(targetTrajectories));
  }

 private:
  std::shared_ptr<ReferenceManagerInterface> physical_;
  int physicalStateDim_ = 0;
};

}  // namespace ocs2::humanoid::multi_contact
