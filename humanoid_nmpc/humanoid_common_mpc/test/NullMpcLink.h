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

#include <memory>

#include "ocs2_mpc/MRT_BASE.h"

#include "humanoid_common_mpc/mrt/MpcLink.h"

namespace ocs2::humanoid::test_support {

/**
 * An MpcLink with no MPC behind it: observations go nowhere and no policy ever arrives. For tests of an MRT joint
 * controller's own work in a cycle - its passive modes and its hold of WB_MPC - without a solver thread or a bus whose
 * work would be counted with it.
 */
class NullMpcLink final : public MpcLink {
 public:
  void start(const SystemObservation& /*initialObservation*/) override {}
  void stop() override {}
  MRT_BASE& mrt() override { return mrt_; }
  const MRT_BASE& mrt() const override { return mrt_; }

  /** A factory for the controllers' constructors. */
  static MpcLinkFactory factory() {
    return [](const MpcLink::ResetTargetFunction& /*resetTarget*/) { return std::make_unique<NullMpcLink>(); };
  }

 private:
  class NullMrt final : public MRT_BASE {
   public:
    void resetMpcNode(const TargetTrajectories& /*initTargetTrajectories*/) override {}
    void setCurrentObservation(const SystemObservation& /*observation*/) override {}
  };

  NullMrt mrt_;
};

}  // namespace ocs2::humanoid::test_support
