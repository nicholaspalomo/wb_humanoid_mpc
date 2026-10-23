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

#include <vector>

#include "absl/base/nullability.h"
#include "mujoco/mujoco.h"

namespace robot::mujoco_sim_interface {

/**
 * The mjModel arrays the simulator writes while it runs, copied out of its model with every state snapshot (MjState)
 * so that a viewer drawing with a private copy of the model (WebMujocoViewer, copyModelForOffscreenRendering()) computes
 * the contacts and forces the physics has, and not those of the model at start-up:
 * - the joint damping, which the zero-torque ragdoll mode raises and active control restores (dof_damping);
 * - the gantry weld's relpose, which the gantry height and the base's pose at locking move (eq_data). Whether the weld
 *   holds at all is mjData::eq_active, which the snapshot's mjData carries already: off while the robot walks;
 * - the projectile's collision flags (geom_ and body_contype / conaffinity), and its mass, inertia, the constants
 *   derived from them and its gravity compensation (body_mass, body_inertia, body_invweight0, body_subtreemass,
 *   body_gravcomp, dof_invweight0, dof_M0), which every throw and park change (Projectile.h).
 * Every array is copied whole, so a new write to another element of one is covered too; a write to another array needs
 * it added here (testOffscreenRender's private-model test compares the whole model).
 *
 * Sized for one model at construction. captureFrom() and applyTo() copy into the existing storage and allocate nothing,
 * and so does a copy assignment between two of one model, which is how MujocoSimInterface::readLatestMjState() hands
 * them over. Not thread-safe: it travels between threads inside an MjState, through the simulator's triple buffer.
 */
class MjRuntimeModelFields {
 public:
  /** No arrays: the fields of a moved-from MjState. captureFrom() and applyTo() copy nothing then. */
  MjRuntimeModelFields() = default;

  /** The arrays of `model`, sized for it and holding its current values. Allocates. */
  explicit MjRuntimeModelFields(const mjModel* absl_nonnull model);

  MjRuntimeModelFields(const MjRuntimeModelFields&) = default;
  MjRuntimeModelFields& operator=(const MjRuntimeModelFields&) = default;
  MjRuntimeModelFields(MjRuntimeModelFields&&) noexcept = default;
  MjRuntimeModelFields& operator=(MjRuntimeModelFields&&) noexcept = default;
  ~MjRuntimeModelFields() = default;

  /**
   * Copies the arrays out of `model`, which has the sizes of the model these fields were made for (that model or a
   * copy of it). Allocates nothing. Copies nothing when the sizes differ, which fits() tells.
   */
  void captureFrom(const mjModel* absl_nonnull model);

  /** Writes the arrays into `model`, under the same condition as captureFrom(). Allocates nothing. */
  void applyTo(mjModel* absl_nonnull model) const;

  /** True when `model` has the sizes these fields were made for; false for every model when they hold no arrays. */
  bool fits(const mjModel* absl_nonnull model) const;

 private:
  // Sized at construction, never resized: nv, neq x mjNEQDATA, ngeom, nbody (body_inertia 3 x nbody, body_invweight0
  // 2 x nbody); all empty in a default-constructed or moved-from value.
  std::vector<mjtNum> dofDamping_;
  std::vector<mjtNum> dofInvweight0_;
  std::vector<mjtNum> dofM0_;
  std::vector<mjtNum> eqData_;
  std::vector<int> geomContype_;
  std::vector<int> geomConaffinity_;
  std::vector<int> bodyContype_;
  std::vector<int> bodyConaffinity_;
  std::vector<mjtNum> bodyMass_;
  std::vector<mjtNum> bodyInertia_;
  std::vector<mjtNum> bodyInvweight0_;
  std::vector<mjtNum> bodySubtreemass_;
  std::vector<mjtNum> bodyGravcomp_;
};

}  // namespace robot::mujoco_sim_interface
