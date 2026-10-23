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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace robot {

/**
 * The absolute path of a file or directory of this repository, given by its path relative to the repository root, for
 * example "robot_models/drc_atlas/drc_atlas_description/urdf/atlas.urdf".
 *
 * Tests and tools read robot descriptions and configuration files from the Bazel runfiles of the running binary. The
 * runfiles hold exactly the files the binary's `data` attribute lists, as symlinks into the checkout, so they are
 * always current, never a copy of the source tree that could be out of date. The function looks in these places, in
 * order:
 *
 *   1. the runfiles the environment names: TEST_SRCDIR under `bazel test`, or RUNFILES_DIR / RUNFILES_MANIFEST_FILE
 *      when another Bazel binary started this one;
 *   2. the runfiles tree next to the executable, `<binary>.runfiles`. This covers `bazel run` and a binary started
 *      from `.bazel/bin`, directly or through a symlink.
 *
 * The checkout itself is never searched, not even under `bazel run`, which names it in BUILD_WORKSPACE_DIRECTORY: a
 * binary that found a file there would work on the developer's machine and fail on the robot computer it is deployed
 * to, and under `bazel test`. A file missing from `data` fails the same way everywhere.
 *
 * Returns NotFound when neither has the path. The message names the path, every place searched and the Bazel
 * package whose target has to go into the binary's `data`. Returns InvalidArgument for a path that is empty, absolute
 * or leaves the repository through "..".
 *
 * Not for a realtime thread: it reads the environment and the runfiles manifest and touches the file system. Resolve
 * paths at start-up.
 */
absl::StatusOr<std::string> resolveResourcePath(absl::string_view repositoryRelativePath);

}  // namespace robot
