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

#include <string>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace ocs2::humanoid::validation {

using golden_matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

/**
 * Where a golden file came from: the commit and the state of the worktree it was recorded on, and the SHA-256 of every
 * configuration file it depends on, so that a golden that no longer matches its configuration is recognized as stale
 * rather than read as a regression.
 */
struct GoldenProvenance {
  std::string gitCommit;      ///< the commit hash, or "unknown"
  std::string worktreeState;  ///< "clean", or what the uncommitted changes were (e.g. their diff's hash)
  /// (path as recorded, SHA-256 of its contents), in the order given.
  std::vector<std::pair<std::string, std::string>> configurationHashes;
  /// Free (key, value) notes: the robot, the label, the machine. Keys have no whitespace and no ':'.
  std::vector<std::pair<std::string, std::string>> notes;
};

/** One labeled matrix of a golden file. Labels have no whitespace. */
struct GoldenEntry {
  std::string label;
  golden_matrix_t value;
};

/**
 * Labeled matrices with their provenance, as plain text: every value at %.17g, so that a double reads back bit for bit
 * (NaN and the infinities as nan, inf and -inf, the sign of zero kept), in the style of the ProblemFingerprint of the
 * centroidal MPC tests and the contact-planning goldens of humanoid_common_mpc.
 *
 *   # wb_humanoid_mpc golden v1
 *   # git_commit: <hash>
 *   # worktree: <state>
 *   # config_sha256: <64 hex digits> <path>
 *   # note <key>: <value>
 *   matrix <label> <rows> <cols>
 *   <row 0: cols values separated by one space>
 *   ...
 */
struct GoldenFile {
  GoldenProvenance provenance;
  std::vector<GoldenEntry> entries;

  /** The matrix labeled `label`, or nullptr. */
  const golden_matrix_t* absl_nullable find(absl::string_view label) const;
};

/**
 * The provenance of a recording that depends on `configurationFiles`: each file's SHA-256, in the order given.
 * NotFound naming the first file that cannot be read.
 */
absl::StatusOr<GoldenProvenance> makeGoldenProvenance(std::string gitCommit,
                                                      std::string worktreeState,
                                                      const std::vector<std::string>& configurationFiles);

/**
 * The text of `golden`. InvalidArgument for a label that is empty, holds whitespace or repeats, and for a provenance
 * field or note that holds a line break (a note key also must not hold whitespace or ':').
 */
absl::StatusOr<std::string> formatGoldenFile(const GoldenFile& golden);

/** Parses formatGoldenFile()'s text; InvalidArgument naming the line of the first error. */
absl::StatusOr<GoldenFile> parseGoldenFile(absl::string_view text);

/** formatGoldenFile() into `path`, creating its directory; the errors of formatGoldenFile(), or Internal if unwritable. */
absl::Status writeGoldenFile(const std::string& path, const GoldenFile& golden);

/** parseGoldenFile() of the file at `path`; NotFound when it cannot be read. */
absl::StatusOr<GoldenFile> readGoldenFile(const std::string& path);

}  // namespace ocs2::humanoid::validation
