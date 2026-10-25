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

#include <optional>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"

namespace robot::mujoco_sim_interface {

/**
 * A viewer of the simulation, selected by its registry name (ViewerNames.h,
 * createViewer() in MujocoViewerRegistry.h) and started by
 * MujocoSimInterface::initSim(). Every viewer only displays the simulation: no
 * viewer input reaches the physics, the controller or the gantry.
 *
 * Lifecycle: create, start() once, destroy. The destructor stops and joins
 * every thread the viewer started; the simulator destroys its viewer before any
 * member the viewer reads. Unless an implementation says otherwise, the methods
 * are called from the thread that owns the simulator, and the const ones may be
 * called from any thread.
 */
class MujocoViewer {
 public:
  MujocoViewer() = default;
  MujocoViewer(const MujocoViewer&) = delete;
  MujocoViewer& operator=(const MujocoViewer&) = delete;
  MujocoViewer(MujocoViewer&&) = delete;
  MujocoViewer& operator=(MujocoViewer&&) = delete;
  virtual ~MujocoViewer() = default;

  /** The registry name of this viewer (ViewerNames.h). */
  virtual absl::string_view name() const = 0;

  /**
   * Starts the viewer's threads. A viewer that can fail at start-up waits until
   * it knows and returns why it did not start, having stopped every thread it
   * began; the simulator then tries the next viewer of its list.
   */
  virtual absl::Status start() = 0;

  /** False once the viewer has stopped on its own (its window was closed, or it
   * failed after start()). */
  virtual bool ok() const = 0;

  /** The TCP port the viewer serves, for a viewer that serves one (web); the
   * bound port when it was configured as 0. */
  virtual std::optional<int> port() const { return std::nullopt; }
};

}  // namespace robot::mujoco_sim_interface
