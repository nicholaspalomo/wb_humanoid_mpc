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

#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicLayer.h"

#include "humanoid_common_mpc/common/MpcFormulationConfig.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFactory.h"

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> LocomotionHeuristicLayer::Create(
    const LocomotionHeuristicConfig& config,
    const LocomotionHeuristicModelParameters& model,
    const LocomotionHeuristicEnvironment& environment,
    bool verbose) {
  RETURN_IF_ERROR(config.validate());

  // A foothold heuristic under an online contact planner would do NOTHING, silently: ContactPlanningReferenceManager
  // overrides getSwingFootReference() and never calls nominalFoothold(), which is the only seam this family has. It is
  // rejected rather than warned about, both because a silent no-op is the worst outcome available and because the two
  // are alternatives on the merits - Bledt's stepping heuristics ARE the analytic answer to the question the
  // reduced-order planner solves numerically, and running both is two opinions fighting over one variable.
  if (environment.usesContactPlanning && !config.formulation.foothold.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicLayer] locomotion_heuristics.foothold lists ", absl::StrJoin(config.formulation.foothold, ", "),
                     " while ", kContactScheduleSourceKey, " is ", kContactPlannerContactScheduleSource,
                     ". The online contact planner supplies the swing foot's landing target itself and never consults the heuristic "
                     "one, so these would have no effect at all. They are the analytic ALTERNATIVE to that planner (Bledt section "
                     "4.3), not a companion to it: either set ",
                     kContactScheduleSourceKey, ": ", kGaitScheduleContactScheduleSource,
                     ", or empty the locomotion_heuristics.foothold list and tune the planner's own foothold terms in "
                     "contact_planning.yaml."));
  }

  // The foothold anchor keeps the feet apart laterally with the nominal step width, or - with hip_centered_stepping -
  // with the hips. With neither, every other foothold heuristic is a velocity- or rate-proportional correction around
  // the STANCE foot's own lateral line, and the swing foot is targeted onto the stance foot: towards self-collision
  // and zero support width. That cannot work, so it is refused.
  const bool listsAnchor = config.formulation.listed(HeuristicKind::FOOTHOLD, heuristic::kHipCenteredStepping);
  if (!config.formulation.foothold.empty() && !listsAnchor && !(environment.nominalStepWidth > 0.0)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "[LocomotionHeuristicLayer] locomotion_heuristics.foothold lists ", absl::StrJoin(config.formulation.foothold, ", "),
        " with model_settings.nominal_foothold.stepWidth at ", environment.nominalStepWidth,
        " and without 'hip_centered_stepping'. Nothing would then keep the feet apart: the other foothold heuristics correct a "
        "landing target that sits on the stance foot's own lateral line, so the swing foot would be aimed at the stance foot. "
        "Either add 'hip_centered_stepping' to the list, which places each foot relative to its own hip, or set "
        "model_settings.nominal_foothold.stepWidth to the lateral distance between the feet."));
  }

  auto layer = std::make_unique<LocomotionHeuristicLayer>();
  layer->formulation_ = config.formulation;
  // The running lists are, by definition, not an edit, so a reload that later changes them is reported against these.
  layer->reportedFormulation_ = config.formulation;
  layer->model_ = model;

  for (const std::string& name : config.formulation.basePose) {
    ASSIGN_OR_RETURN(std::unique_ptr<BasePoseHeuristic> heuristic, LocomotionHeuristicFactory::makeBasePoseHeuristic(name));
    RETURN_IF_ERROR(heuristic->configure(config, model));
    layer->basePose_.push_back(std::move(heuristic));
  }
  for (const std::string& name : config.formulation.foothold) {
    ASSIGN_OR_RETURN(std::unique_ptr<FootholdHeuristic> heuristic, LocomotionHeuristicFactory::makeFootholdHeuristic(name));
    RETURN_IF_ERROR(heuristic->configure(config, model));
    layer->footholdMovesAnchor_ = layer->footholdMovesAnchor_ || heuristic->movesAnchor();
    layer->foothold_.push_back(std::move(heuristic));
  }
  for (const std::string& name : config.formulation.wrench) {
    ASSIGN_OR_RETURN(std::unique_ptr<WrenchHeuristic> heuristic, LocomotionHeuristicFactory::makeWrenchHeuristic(name));
    RETURN_IF_ERROR(heuristic->configure(config, model));
    layer->wrenchNeedsWorldFrame_ = layer->wrenchNeedsWorldFrame_ || heuristic->producesHorizontalForce();
    layer->wrench_.push_back(std::move(heuristic));
  }

  // The basis-vector parameterization expresses the contact input in the LOCAL contact frame, so a horizontal
  // world-frame force has to be rotated into it by the foot's own orientation. The model supplies exactly that through
  // setContactForceInWorldFrame(), whose BasisInputsModelDecorator override takes a local copy of the Pinocchio data
  // and is documented as safe to call from several threads - so this combination WORKS, and the only thing it costs is
  // one forward-kinematics pass per shooting node per SQP iteration in the two input costs. That is a real cost on a
  // hot path, and it is paid only when a heuristic that needs it is listed, which is why it is reported rather than
  // hidden.
  if (environment.usesContactBasisVectorInputs && layer->wrenchNeedsWorldFrame_) {
    LOG(WARNING) << "[LocomotionHeuristicLayer] a listed wrench heuristic produces a HORIZONTAL force while "
                    "contactInputParameterization is basis_vectors. The contact-force reference is therefore written through the "
                    "state-aware setContactForceInWorldFrame(), which rotates it into the local contact frame and costs "
                    "one forward-kinematics pass per shooting node per SQP iteration in inputQuadraticCost and "
                    "stateInputQuadraticCost. Watch the solve time; the vertical-only heuristics do not pay this.";
  }
  // Configurations that work but do NOTHING. Warned rather than refused: the coefficients are ready and the operator
  // may be about to change the other setting, but nobody should tune a channel that cannot move.
  if (environment.listsComAndAcomTrackingCost && !layer->basePose_.empty()) {
    LOG(WARNING) << "[LocomotionHeuristicLayer] locomotion_heuristics.base_pose lists " << absl::StrJoin(config.formulation.basePose, ", ")
                 << " while costs lists com_and_acom_tracking_cost. The cost factory zeroes Q's and Q_final's base-pose blocks "
                    "in that mode and the ACoM cost reads the unshaped target, so these offsets reach no cost and change nothing. "
                    "Remove com_and_acom_tracking_cost from costs for them to act.";
  }
  if (environment.footPositionIsUntracked && !layer->foothold_.empty()) {
    LOG(WARNING) << "[LocomotionHeuristicLayer] locomotion_heuristics.foothold lists " << absl::StrJoin(config.formulation.foothold, ", ")
                 << " while task_space_foot_cost_weights.pos_x and pos_y are both zero (or task_space_foot_cost is not listed). "
                    "Nothing then tracks the landing target these shape, so they do not move the feet. Raise pos_x and pos_y for "
                    "them to act.";
  }
  for (const std::string& warning : config.formulation.warnings()) {
    LOG(WARNING) << "[LocomotionHeuristicLayer] " << warning;
  }

  if (verbose && !layer->empty()) {
    LOG(INFO) << "\n #### Locomotion heuristics (Bledt RPC, Appendix C):\n" << layer->summary() << model.summary();
  }
  return layer;
}

BasePoseOffset LocomotionHeuristicLayer::basePoseOffset(const BasePoseHeuristicContext& context) const {
  BasePoseOffset total;
  for (const std::unique_ptr<BasePoseHeuristic>& heuristic : basePose_) {
    total += heuristic->offset(context);
  }
  return total;
}

vector2_t LocomotionHeuristicLayer::footholdOffset(const FootholdHeuristicContext& context) const {
  vector2_t total = vector2_t::Zero();
  for (const std::unique_ptr<FootholdHeuristic>& heuristic : foothold_) {
    total += heuristic->offset(context);
  }
  return total;
}

vector3_t LocomotionHeuristicLayer::wrenchOffset(const WrenchHeuristicContext& context, size_t contactIndex) const {
  vector3_t total = vector3_t::Zero();
  for (const std::unique_ptr<WrenchHeuristic>& heuristic : wrench_) {
    total += heuristic->forceOffset(context, contactIndex);
  }
  return total;
}

absl::Status LocomotionHeuristicLayer::reconfigure(const LocomotionHeuristicConfig& config) {
  RETURN_IF_ERROR(config.validate());
  // Which heuristics are listed is structural: it decides whether the foothold seam has an opinion at all, and whether
  // the contact-force reference takes the cheap vertical-only path or the forward-kinematics one. Those are wired into
  // the reference manager and the two input costs at construction, so changing a list under a running solver would be
  // a different controller rather than a retuned one. The COEFFICIENTS are what the tuning GUI is for, and they are
  // exactly what this re-reads.
  bool listChanged = false;
  for (HeuristicKind kind : allHeuristicKinds()) {
    listChanged = listChanged || config.formulation.list(kind) != formulation_.list(kind);
  }
  bool alreadyReported = true;
  for (HeuristicKind kind : allHeuristicKinds()) {
    alreadyReported = alreadyReported && config.formulation.list(kind) == reportedFormulation_.list(kind);
  }
  if (listChanged && !alreadyReported) {
    for (HeuristicKind kind : allHeuristicKinds()) {
      if (config.formulation.list(kind) == formulation_.list(kind)) continue;
      LOG(WARNING) << "[LocomotionHeuristicLayer] the locomotion_heuristics." << heuristicKindName(kind)
                   << " list has changed on disk since start-up, from [" << absl::StrJoin(formulation_.list(kind), ", ") << "] to ["
                   << absl::StrJoin(config.formulation.list(kind), ", ")
                   << "]. Which heuristics are listed is wired in at construction and is NOT hot-reloadable; the "
                      "coefficients below it are. Restart the controller to apply the new list.";
    }
    reportedFormulation_ = config.formulation;
  }
  // Back to the running lists: forget the reported edit, so that making the same edit again is reported again.
  if (!listChanged) reportedFormulation_ = formulation_;

  // All or nothing: fresh heuristics of the RUNNING lists, configured with the new coefficients, swapped in only once
  // every one has accepted them.
  std::vector<std::unique_ptr<BasePoseHeuristic>> basePose;
  std::vector<std::unique_ptr<FootholdHeuristic>> foothold;
  std::vector<std::unique_ptr<WrenchHeuristic>> wrench;
  for (const std::string& name : formulation_.basePose) {
    ASSIGN_OR_RETURN(std::unique_ptr<BasePoseHeuristic> heuristic, LocomotionHeuristicFactory::makeBasePoseHeuristic(name));
    RETURN_IF_ERROR(heuristic->configure(config, model_));
    basePose.push_back(std::move(heuristic));
  }
  for (const std::string& name : formulation_.foothold) {
    ASSIGN_OR_RETURN(std::unique_ptr<FootholdHeuristic> heuristic, LocomotionHeuristicFactory::makeFootholdHeuristic(name));
    RETURN_IF_ERROR(heuristic->configure(config, model_));
    foothold.push_back(std::move(heuristic));
  }
  for (const std::string& name : formulation_.wrench) {
    ASSIGN_OR_RETURN(std::unique_ptr<WrenchHeuristic> heuristic, LocomotionHeuristicFactory::makeWrenchHeuristic(name));
    RETURN_IF_ERROR(heuristic->configure(config, model_));
    wrench.push_back(std::move(heuristic));
  }
  basePose_ = std::move(basePose);
  foothold_ = std::move(foothold);
  wrench_ = std::move(wrench);
  return absl::OkStatus();
}

std::string LocomotionHeuristicLayer::summary() const {
  if (empty()) return "  (none listed: the layer is an exact no-op)\n";
  std::string out;
  for (const std::unique_ptr<BasePoseHeuristic>& heuristic : basePose_) absl::StrAppend(&out, "  base_pose  ", heuristic->describe(), "\n");
  for (const std::unique_ptr<FootholdHeuristic>& heuristic : foothold_) absl::StrAppend(&out, "  foothold   ", heuristic->describe(), "\n");
  for (const std::unique_ptr<WrenchHeuristic>& heuristic : wrench_) absl::StrAppend(&out, "  wrench     ", heuristic->describe(), "\n");
  return out;
}

}  // namespace ocs2::humanoid
