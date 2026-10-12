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

#include "mujoco_sim_interface/viewer/WebViewerOptions.h"

namespace robot::mujoco_sim_interface {

/**
 * What the OpenGL context of an offscreen viewer offers, as glGetString and glGetIntegerv report it
 * (EglContext::queryGlCapabilities()). Passive data; checkGlCapabilities() decides whether MuJoCo can render with it.
 */
struct GlCapabilities {
  std::string vendor;           // GL_VENDOR
  std::string renderer;         // GL_RENDERER, "llvmpipe (LLVM 20.1.2, 256 bits)" for instance
  std::string version;          // GL_VERSION, "4.5 (Compatibility Profile) Mesa 25.2.8" for instance
  std::string extensions;       // GL_EXTENSIONS, space-separated
  int maxRenderbufferSize = 0;  // [px] GL_MAX_RENDERBUFFER_SIZE
  int maxSamples = 0;           // GL_MAX_SAMPLES
};

/**
 * Checks, before MuJoCo creates its GPU context, every condition under which mjr_makeContext or mjr_resizeOffscreen
 * would end the process (mju_error) for a viewer of `options`: OpenGL 3.0, or 1.5 with GL_ARB_framebuffer_object;
 * an offscreen buffer of maxWidth x maxHeight within GL_MAX_RENDERBUFFER_SIZE; and msaaSamples within GL_MAX_SAMPLES.
 * FailedPrecondition names the limit, the value asked for and how to run without the web viewer.
 */
absl::Status checkGlCapabilities(const GlCapabilities& capabilities, const WebViewerOptions& options);

}  // namespace robot::mujoco_sim_interface
