/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

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

#include "humanoid_common_mpc/contact/ContactRectangle.h"

#include <vector>

namespace ocs2::humanoid {

ContactRectangle::ContactRectangle(const PolygonBounds& polygonBounds, const ContactCenterPoint& contactCenterPoint, scalar_t scaleFactor)
    : ContactPolygon({vector3_t(polygonBounds.x_min * scaleFactor, polygonBounds.y_min * scaleFactor, 0.0),
                      vector3_t(polygonBounds.x_max * scaleFactor, polygonBounds.y_min * scaleFactor, 0.0),
                      vector3_t(polygonBounds.x_max * scaleFactor, polygonBounds.y_max * scaleFactor, 0.0),
                      vector3_t(polygonBounds.x_min * scaleFactor, polygonBounds.y_max * scaleFactor, 0.0)},
                     PolygonBounds(polygonBounds.x_min, polygonBounds.x_max, polygonBounds.y_min, polygonBounds.y_max, scaleFactor),
                     contactCenterPoint) {}

}  // namespace ocs2::humanoid
