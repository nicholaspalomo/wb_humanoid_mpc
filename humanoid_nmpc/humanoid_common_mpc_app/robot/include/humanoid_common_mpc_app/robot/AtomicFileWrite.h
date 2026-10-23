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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

/**
 * Crash-safe writes of the robot's persistent configuration files (RobotConfigDirectory, ConfigFileStore): a reader of
 * a file - the task file's watcher, the PD gains file's poll, the next start - sees its old or its new contents, never
 * a part of either, also after a power loss. POSIX; an error is the errno's absl::Status (absl::ErrnoToStatus) naming the
 * path. They block on the disk: for the main thread and the store's writer thread, never the realtime thread and not
 * the bus's IO thread.
 */
namespace ocs2::humanoid {

/**
 * Replaces the file at `path` with `contents`: writes them to `<path>.tmp.<pid>` (created exclusively, mode 0644),
 * fsyncs it, renames it over `path` and fsyncs the directory. A temporary of the same name left by a process of the
 * same pid that died while writing (a container's processes reuse their low pids) is removed first. DataLoss when the
 * file was replaced but its directory could not be synced afterwards (EIO, or a file system without directory fsync):
 * readers see the new contents, which a power loss may undo. On any other error the file at `path` is untouched and the
 * temporary is removed.
 */
absl::Status writeFileAtomically(absl::string_view path, absl::string_view contents);

/** The bytes of the file at `path`: NotFound when there is none, the errno's status otherwise, naming the path. */
absl::StatusOr<std::string> readFileBytes(absl::string_view path);

/**
 * Renames the file `from` over `to`, which it replaces atomically, and fsyncs the directory of `to`; DataLoss, as
 * writeFileAtomically(), when the rename was done but the directory could not be synced.
 */
absl::Status renameFile(absl::string_view from, absl::string_view to);

/**
 * Removes the file at `path` and fsyncs its directory; OK when there is no file there, DataLoss when it was removed but
 * the directory could not be synced.
 */
absl::Status removeFile(absl::string_view path);

}  // namespace ocs2::humanoid
