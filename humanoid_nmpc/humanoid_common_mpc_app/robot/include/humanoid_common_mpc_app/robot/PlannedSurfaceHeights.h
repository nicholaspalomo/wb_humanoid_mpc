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
#include <numbers>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "ocs2_core/Types.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/Types.h"
#include "humanoid_common_mpc_app/robot/ContactSurfaceHeightSource.h"
#include "humanoid_multi_contact/plan/multi_contact_plan.nproto.h"
#include "humanoid_state_estimator/ContactEstimationInputs.h"

namespace ocs2::humanoid {

/**
 * The surfaces of a multi-contact plan (contact_schedule_source:
 * "multi_contact_plan"): each contact stands on the patch of the plan it is
 * on, a seat, a stair tread or a box top, at that patch's height below the
 * contact, and a contact on no patch the plan knows, or on a steep one (a hand
 * on a wall), gets no height row.
 *
 * Create() collects, for every MPC contact (by name, as the MPC maps the
 * plan's contacts, PlanContactOrder), the distinct patches its contact phases
 * rest it on: a center and a unit normal (ContactPhase.des_pose_pad_rt_world).
 * fill() picks, for each contact, the patch whose center is nearest to the
 * contact's estimated position, and accepts it when it is closer than
 * kMaxSurfaceMatchDistance and tilted less than kMaxSurfaceTilt from the world
 * z axis; the height is the patch's plane at the contact's x and y,
 * h = c_z - (n_x (x - c_x) + n_y (y - c_y)) / n_z.
 *
 * Immutable after Create(); fill() allocates nothing and is safe to call
 * concurrently.
 */
class PlannedSurfaceHeights final : public ContactSurfaceHeightSource {
 public:
  /**
   * [m] How far a contact's estimated position may be from a planned pad's
   * center to stand on its patch. The MPC holds a stance contact within a few
   * centimeters of its planned pad (the clock hold waits for 1 cm along the
   * normal and 3 cm of the CoM, multi_contact_plan.clock_hold), and the
   * patches of one contact in the shipped scenarios are farther apart than
   * 0.2 m (stair treads, the box top and the floor); a contact farther than
   * 0.1 m from every pad has slipped or is not where the plan puts it.
   */
  static constexpr scalar_t kMaxSurfaceMatchDistance = 0.1;
  /**
   * [rad] The steepest patch that gives a height row. The row measures z
   * alone, so on a patch tilted by theta an xy error e of the contact
   * estimate becomes a height error e tan(theta): at 30 degrees 0.58 e.
   * Steeper patches (walls, the car's door frame) get none.
   */
  static constexpr scalar_t kMaxSurfaceTilt = std::numbers::pi / 6.0;

  /**
   * Collects the patches of `plan` for the contacts of `modelSettings`. A
   * phase of a contact the MPC does not have is skipped, as the MPC skips it.
   *
   * @return InvalidArgument naming the phase and its contact when its
   * des_pose_pad_rt_world is not six finite values with a unit normal.
   */
  static absl::StatusOr<std::unique_ptr<PlannedSurfaceHeights>> Create(
      const multi_contact::config::MultiContactPlan& plan,
      const ModelSettings& modelSettings);

  ~PlannedSurfaceHeights() override = default;

  /**
   * Gives each contact the height of its patch, or clears it (see the class
   * comment). Clears every contact when `estPosContactsRtWorldEwrtWorld` is
   * not one position per contact of `inputs` (an estimator that keeps none).
   */
  void fill(absl::Span<const vector3_t> estPosContactsRtWorldEwrtWorld,
            estimation::ContactEstimationInputs& inputs) const override;

  /** Returns the number of distinct patches the plan rests contact `contact`
   * on; 0 for a contact out of range. */
  size_t numPatches(size_t contact) const {
    return contact < patchesByContact_.size()
               ? patchesByContact_[contact].size()
               : 0;
  }

 private:
  /** One planned patch: its center and its unit normal, in the world. */
  struct Patch {
    vector3_t posSurfaceRtWorldEwrtWorld = vector3_t::Zero();
    vector3_t normalSurfaceEwrtWorld = vector3_t::UnitZ();
    /** False for a patch steeper than kMaxSurfaceTilt. */
    bool givesHeight = false;
  };

  /** Returns whether `patches` holds the patch of exactly this center and
   * unit normal (the dedup of Create()). Private because Patch is. */
  static bool hasPatch(absl::Span<const Patch> patches,
                       const vector3_t& posSurfaceRtWorldEwrtWorld,
                       const vector3_t& normalSurfaceEwrtWorld);

  explicit PlannedSurfaceHeights(std::vector<std::vector<Patch>> patches)
      : patchesByContact_(std::move(patches)) {}

  /** One entry per MPC contact, in contact order. */
  std::vector<std::vector<Patch>> patchesByContact_;
};

}  // namespace ocs2::humanoid
