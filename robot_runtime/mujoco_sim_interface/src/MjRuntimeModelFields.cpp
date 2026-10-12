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

#include "mujoco_sim_interface/MjRuntimeModelFields.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "absl/base/nullability.h"
#include "mujoco/mujoco.h"

namespace robot::mujoco_sim_interface {
namespace {

/** Returns the first `count` elements of a model array; `count` zeros for a null array (MuJoCo's empty one). */
template <typename T>
std::vector<T> arrayOf(const T* absl_nullable field, int count) {
  std::vector<T> values(static_cast<size_t>(std::max(count, 0)));
  if (field != nullptr) std::copy_n(field, values.size(), values.begin());
  return values;
}

/** Copies `values.size()` elements of a model array into `values`; nothing from a null array. */
template <typename T>
void copyOut(const T* absl_nullable field, std::vector<T>& values) {
  if (field != nullptr) std::copy_n(field, values.size(), values.begin());
}

/** Copies `values` into the first `values.size()` elements of a model array; nothing into a null array. */
template <typename T>
void copyIn(const std::vector<T>& values, T* absl_nullable field) {
  if (field != nullptr) std::copy(values.begin(), values.end(), field);
}

}  // namespace

MjRuntimeModelFields::MjRuntimeModelFields(const mjModel* absl_nonnull model)
    : dofDamping_(arrayOf(model->dof_damping, model->nv)),
      dofInvweight0_(arrayOf(model->dof_invweight0, model->nv)),
      dofM0_(arrayOf(model->dof_M0, model->nv)),
      eqData_(arrayOf(model->eq_data, model->neq * mjNEQDATA)),
      geomContype_(arrayOf(model->geom_contype, model->ngeom)),
      geomConaffinity_(arrayOf(model->geom_conaffinity, model->ngeom)),
      bodyContype_(arrayOf(model->body_contype, model->nbody)),
      bodyConaffinity_(arrayOf(model->body_conaffinity, model->nbody)),
      bodyMass_(arrayOf(model->body_mass, model->nbody)),
      bodyInertia_(arrayOf(model->body_inertia, 3 * model->nbody)),
      bodyInvweight0_(arrayOf(model->body_invweight0, 2 * model->nbody)),
      bodySubtreemass_(arrayOf(model->body_subtreemass, model->nbody)),
      bodyGravcomp_(arrayOf(model->body_gravcomp, model->nbody)) {}

bool MjRuntimeModelFields::fits(const mjModel* absl_nonnull model) const {
  // One array per size is enough: they were all sized together, from one model.
  return model->nbody > 0 && dofDamping_.size() == static_cast<size_t>(model->nv) &&
         eqData_.size() == static_cast<size_t>(model->neq) * mjNEQDATA && geomContype_.size() == static_cast<size_t>(model->ngeom) &&
         bodyMass_.size() == static_cast<size_t>(model->nbody);
}

void MjRuntimeModelFields::captureFrom(const mjModel* absl_nonnull model) {
  if (!fits(model)) return;
  copyOut(model->dof_damping, dofDamping_);
  copyOut(model->dof_invweight0, dofInvweight0_);
  copyOut(model->dof_M0, dofM0_);
  copyOut(model->eq_data, eqData_);
  copyOut(model->geom_contype, geomContype_);
  copyOut(model->geom_conaffinity, geomConaffinity_);
  copyOut(model->body_contype, bodyContype_);
  copyOut(model->body_conaffinity, bodyConaffinity_);
  copyOut(model->body_mass, bodyMass_);
  copyOut(model->body_inertia, bodyInertia_);
  copyOut(model->body_invweight0, bodyInvweight0_);
  copyOut(model->body_subtreemass, bodySubtreemass_);
  copyOut(model->body_gravcomp, bodyGravcomp_);
}

void MjRuntimeModelFields::applyTo(mjModel* absl_nonnull model) const {
  if (!fits(model)) return;
  copyIn(dofDamping_, model->dof_damping);
  copyIn(dofInvweight0_, model->dof_invweight0);
  copyIn(dofM0_, model->dof_M0);
  copyIn(eqData_, model->eq_data);
  copyIn(geomContype_, model->geom_contype);
  copyIn(geomConaffinity_, model->geom_conaffinity);
  copyIn(bodyContype_, model->body_contype);
  copyIn(bodyConaffinity_, model->body_conaffinity);
  copyIn(bodyMass_, model->body_mass);
  copyIn(bodyInertia_, model->body_inertia);
  copyIn(bodyInvweight0_, model->body_invweight0);
  copyIn(bodySubtreemass_, model->body_subtreemass);
  copyIn(bodyGravcomp_, model->body_gravcomp);
}

}  // namespace robot::mujoco_sim_interface
