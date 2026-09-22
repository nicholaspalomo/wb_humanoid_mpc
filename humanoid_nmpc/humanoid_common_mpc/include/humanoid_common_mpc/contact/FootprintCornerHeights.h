/******************************************************************************
Copyright (c) 2026, Nicholas Palomo. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

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
#include <string>
#include <vector>

#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/common/Types.h"

namespace ocs2::humanoid {

/**
 * The world-frame heights of a set of points on one foot, and their Jacobian with respect to the state.
 *
 * Two terms of the contact-implicit formulation need the same geometry and must not disagree about it:
 *
 *   - GroundPenetrationConstraint needs every point to stay above the ground, so it reads all of them;
 *   - ContactComplementarityConstraint needs THE GAP - how far the foot is from touching - which is the height of the
 *     LOWEST point, not the height of the sole's centre.
 *
 * They used to disagree. Penetration was evaluated at the footprint corners while complementarity used the contact
 * frame, so a foot pitched onto its heel had a positive centre height and the complementarity term treated it as
 * airborne, penalising the contact force it was physically carrying. Since this formulation deliberately leaves the
 * foot's rocking rates free (ForceWeightedSlipConstraint constrains only the two tangential velocities and the spin
 * about the normal), pitched contact is the expected case rather than an exceptional one.
 *
 * WHY THIS CLASS EXISTS AT ALL, rather than a PinocchioEndEffectorKinematicsCppAd over the corner frames: that class
 * generates SIX code-generated models per frame set - position, velocity, orientation, orientation error, orientation
 * with respect to a plane, and angular velocity - and only the first is ever read here. Over a four-corner set that is
 * five large dead models per foot, which measurably slowed a clean build and pushed a test past its timeout. One tape
 * of N heights replaces them.
 *
 * The terrain height is deliberately NOT subtracted inside the tape: it is a configured scalar that the parameter
 * updater re-reads at runtime, and baking it into a generated model would freeze it at whatever it was when the model
 * was generated.
 */
class FootprintCornerHeights {
 public:
  /**
   * @param pinocchioInterface  the robot model; a CppAD copy is taken.
   * @param mpcRobotModelAD     used only to extract the generalized coordinates from the state.
   * @param frameNames          the points of this foot, e.g. the corner frames createPinocchioModel() adds for each
   *                            point of the contact polygon. Must be non-empty.
   * @param modelName           name of the generated library. It is the cache key, so it must change whenever the
   *                            geometry does; see the note on staleness in the class documentation of the callers.
   * @param modelSettings       supplies the model folder and whether to regenerate.
   */
  FootprintCornerHeights(const PinocchioInterface& pinocchioInterface,
                         const MpcRobotModelBase<ad_scalar_t>& mpcRobotModelAD,
                         std::vector<std::string> frameNames,
                         const std::string& modelName,
                         const ModelSettings& modelSettings);

  FootprintCornerHeights(const FootprintCornerHeights& rhs);
  ~FootprintCornerHeights() = default;
  FootprintCornerHeights& operator=(const FootprintCornerHeights&) = delete;

  std::unique_ptr<FootprintCornerHeights> clone() const { return std::make_unique<FootprintCornerHeights>(*this); }

  /** Number of points, i.e. the number of rows of getHeights() and getHeightsJacobian(). */
  size_t numCorners() const { return frameNames_.size(); }

  const std::vector<std::string>& frameNames() const { return frameNames_; }

  /** World-frame z of every point, in the order the frame names were given. */
  vector_t getHeights(const vector_t& state) const;

  /** d(heights) / d(state), numCorners() x stateDim. */
  matrix_t getHeightsJacobian(const vector_t& state) const;

 private:
  ad_vector_t heightsFunction(const ad_vector_t& state) const;

  /**
   * Tapes the height function and either generates or loads its library.
   *
   * Each instance builds its OWN CppAdInterface rather than copying the one it was cloned from, because the taped
   * function is a lambda over `this`: CppAdInterface's copy constructor carries that std::function across, so a clone
   * that outlived its source and then had to regenerate would tape through a dangling pointer. The SQP solver clones
   * the whole problem once per worker thread, so that is not a hypothetical lifetime.
   *
   * @param regenerate  true to compile the library afresh; false to load it, compiling only if it is missing. A clone
   *        always passes false: the source has already created it, and regenerating once per worker thread would turn
   *        a solver start-up into a series of compiler invocations.
   */
  void createAdInterface(bool regenerate);

  std::vector<std::string> frameNames_;
  /** Pinocchio frame indices of frameNames_, resolved once so the taped function does no string lookups. */
  std::vector<size_t> frameIds_;
  std::unique_ptr<MpcRobotModelBase<ad_scalar_t>> mpcRobotModelAdPtr_;
  PinocchioInterfaceCppAd pinocchioInterfaceCppAd_;
  std::string modelName_;
  std::string modelFolder_;
  bool verbose_;
  std::unique_ptr<CppAdInterface> adInterfacePtr_;
};

/**
 * The value of a smooth lower bound on a set of heights, together with the convex weights that are its gradient.
 */
struct SmoothMinimumHeight {
  /** [m] the smoothed minimum. Always in [min(heights), min(heights) + log(N) * smoothing]. */
  scalar_t value;
  /** The gradient of `value` with respect to the heights: non-negative and summing to one. */
  vector_t weights;
};

/**
 * A smooth, conservative stand-in for the MINIMUM of a set of corner heights - the gap between the foot and the ground.
 *
 * It is an UPPER bound on that minimum, never a lower one: the value lies in [min(h), min(h) + log(N) * s], so it
 * over-reports the clearance and never claims the foot is closer to the ground than it is. Which direction the error
 * runs in is the whole design, and is argued under WHY THE 1/N below.
 *
 *   softmin(h) = m - log( (1/N) sum_i exp(-(h_i - m) / s) ) * s,      m = min_i h_i,
 *
 * whose gradient with respect to h_i is the softmax weight w_i = exp(-(h_i - m)/s) / sum_j exp(-(h_j - m)/s).
 *
 * WHY NOT THE PLAIN MINIMUM. The exact min is what the physics wants, but it is non-smooth exactly where the foot is
 * flat, which is where a walking robot spends most of its stance. The complementarity residual is handed to a
 * quadratic penalty and then to an SQP solver that linearises it; at a flat foot the true min's gradient jumps between
 * the four corners', so the linearisation would flip between corners from iteration to iteration.
 *
 * WHY THE 1/N. Without it this is the standard log-sum-exp softmin, which at a flat foot returns m - log(N) * s: it
 * UNDER-reports the gap by 1.39 mm at the shipped smoothing, and does so worst precisely in the flat-footed stance
 * that is the common case. Under-reporting is the dangerous direction here, because the complementarity penalty is
 * two-sided: a residual of f_n * (h - g) with g < 0 is minimised by pushing the foot UP until the reported gap reaches
 * zero, so the robot would hover 1.39 mm above the ground under full load and never close the contact - and nothing
 * opposes it, since the penetration hinge is identically zero above the ground and the swing-foot z cost is switched
 * off in stance. Normalising by N makes the bound exact in value and in gradient at a flat foot, and makes it
 * one-sided in the SAFE direction everywhere: the result never falls below the true minimum, and over-reporting the
 * gap merely asks the solver to take a little load off a foot that is up on an edge.
 *
 * HOW BIG THE BIAS IS. It is s * log(N / k), where k is how many corners sit at the minimum - not a single number:
 *
 *   k = N, a flat foot                                                      exact
 *   k = 2, an EDGE down, the ordinary heel strike or toe-off of a sole      s * log 2 = 0.693 mm
 *   k = 1, a single corner, which needs pitch AND roll at once              s * log N = 1.386 mm, the bound
 *
 * Against the shipped penetration hinge those settle at about 0.094 mm and 0.187 mm of equilibrium penetration under
 * full body weight, which is below the compliance of any real sole.
 *
 * @param heights  [m] the corner heights; must be non-empty.
 * @param smoothing [m] the length scale s over which the minimum is blended; must be positive. The bound's worst-case
 *        error is log(N) * smoothing, so at the shipped 1 mm and four corners it is 1.39 mm, reached only when a
 *        single corner touches; an edge-down contact sees half of that.
 */
SmoothMinimumHeight smoothMinimumHeight(const vector_t& heights, scalar_t smoothing);

}  // namespace ocs2::humanoid
