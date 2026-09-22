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

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

#include "humanoid_common_mpc/common/StatusMacros.h"
#include "humanoid_common_mpc/locomotion_heuristics/LocomotionHeuristicFactory.h"

namespace ocs2::humanoid {

absl::StatusOr<std::unique_ptr<LocomotionHeuristicLayer>> LocomotionHeuristicLayer::Create(const LocomotionHeuristicConfig& config,
                                                                                           const LocomotionHeuristicModelParameters& model,
                                                                                           bool usesContactPlanning,
                                                                                           bool usesContactBasisVectorInputs,
                                                                                           bool verbose) {
  RETURN_IF_ERROR(config.validate());

  // A foothold heuristic under an online contact planner would do NOTHING, silently: ContactPlanningReferenceManager
  // overrides getSwingFootReference() and never calls nominalFoothold(), which is the only seam this family has. It is
  // rejected rather than warned about, both because a silent no-op is the worst outcome available and because the two
  // are alternatives on the merits - Bledt's stepping heuristics ARE the analytic answer to the question the
  // reduced-order planner solves numerically, and running both is two opinions fighting over one variable.
  if (usesContactPlanning && !config.formulation.foothold.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("[LocomotionHeuristicLayer] locomotion_heuristics.foothold lists ", absl::StrJoin(config.formulation.foothold, ", "),
                     " while useContactPlanning is true. The online contact planner supplies the swing foot's landing target itself and "
                     "never consults the heuristic one, so these would have no effect at all. They are the analytic ALTERNATIVE to "
                     "that planner (Bledt section 4.3), not a companion to it: either set useContactPlanning: false, or empty the "
                     "locomotion_heuristics.foothold list and tune the planner's own foothold terms in contact_planning.yaml."));
  }

  auto layer = std::make_unique<LocomotionHeuristicLayer>();
  layer->formulation_ = config.formulation;
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
  if (usesContactBasisVectorInputs && layer->wrenchNeedsWorldFrame_) {
    LOG(WARNING) << "[LocomotionHeuristicLayer] a listed wrench heuristic produces a HORIZONTAL force while "
                    "useContactBasisVectorInputs is true. The contact-force reference is therefore written through the "
                    "state-aware setContactForceInWorldFrame(), which rotates it into the local contact frame and costs "
                    "one forward-kinematics pass per shooting node per SQP iteration in inputQuadraticCost and "
                    "stateInputQuadraticCost. Watch the solve time; the vertical-only heuristics do not pay this.";
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
  for (HeuristicKind kind : allHeuristicKinds()) {
    if (config.formulation.list(kind) != formulation_.list(kind)) {
      LOG(WARNING) << "[LocomotionHeuristicLayer] the locomotion_heuristics." << heuristicKindName(kind)
                   << " list has changed on disk since start-up, from [" << absl::StrJoin(formulation_.list(kind), ", ") << "] to ["
                   << absl::StrJoin(config.formulation.list(kind), ", ")
                   << "]. Which heuristics are listed is wired in at construction and is NOT hot-reloadable; the "
                      "coefficients below it are. Restart the controller to apply the new list.";
    }
  }
  for (const std::unique_ptr<BasePoseHeuristic>& heuristic : basePose_) {
    RETURN_IF_ERROR(heuristic->configure(config, model_));
  }
  for (const std::unique_ptr<FootholdHeuristic>& heuristic : foothold_) {
    RETURN_IF_ERROR(heuristic->configure(config, model_));
  }
  for (const std::unique_ptr<WrenchHeuristic>& heuristic : wrench_) {
    RETURN_IF_ERROR(heuristic->configure(config, model_));
  }
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
