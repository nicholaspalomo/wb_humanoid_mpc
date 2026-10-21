/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

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

#include <ocs2_core/Types.h>

namespace ocs2 {

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionLinearApproximation::ScalarFunctionLinearApproximation(int nx, int nu) {
  resize(nx, nu);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionLinearApproximation& ScalarFunctionLinearApproximation::operator+=(const ScalarFunctionLinearApproximation& rhs) {
  f += rhs.f;
  dfdx += rhs.dfdx;
  dfdu += rhs.dfdu;
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionLinearApproximation& ScalarFunctionLinearApproximation::operator*=(scalar_t scalar) {
  f *= scalar;
  dfdx *= scalar;
  dfdu *= scalar;
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionLinearApproximation& ScalarFunctionLinearApproximation::resize(int nx, int nu) {
  dfdx.resize(nx);
  if (nu >= 0) {
    dfdu.resize(nu);
  } else {
    dfdu = vector_t();
  }
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionLinearApproximation& ScalarFunctionLinearApproximation::setZero(int nx, int nu) {
  f = 0.0;
  dfdx.setZero(nx);
  if (nu >= 0) {
    dfdu.setZero(nu);
  } else {
    dfdu = vector_t();
  }
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionLinearApproximation ScalarFunctionLinearApproximation::Zero(int nx, int nu) {
  ScalarFunctionLinearApproximation f;
  f.setZero(nx, nu);
  return f;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::ostream& operator<<(std::ostream& out, const ScalarFunctionLinearApproximation& f) {
  out << "f: " << f.f << '\n';
  out << "dfdx: " << f.dfdx.transpose() << '\n';
  out << "dfdu: " << f.dfdu.transpose() << '\n';
  return out;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionQuadraticApproximation::ScalarFunctionQuadraticApproximation(int nx, int nu) {
  resize(nx, nu);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionQuadraticApproximation& ScalarFunctionQuadraticApproximation::operator+=(const ScalarFunctionQuadraticApproximation& rhs) {
  f += rhs.f;
  dfdx += rhs.dfdx;
  dfdu += rhs.dfdu;
  dfdxx += rhs.dfdxx;
  dfdux += rhs.dfdux;
  dfduu += rhs.dfduu;
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionQuadraticApproximation& ScalarFunctionQuadraticApproximation::operator*=(scalar_t scalar) {
  f *= scalar;
  dfdx *= scalar;
  dfdu *= scalar;
  dfdxx *= scalar;
  dfdux *= scalar;
  dfduu *= scalar;
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionQuadraticApproximation& ScalarFunctionQuadraticApproximation::resize(int nx, int nu) {
  dfdx.resize(nx);
  dfdxx.resize(nx, nx);
  if (nu >= 0) {
    dfdu.resize(nu);
    dfdux.resize(nu, nx);
    dfduu.resize(nu, nu);
  } else {
    dfdu = vector_t();
    dfdux = matrix_t();
    dfduu = matrix_t();
  }
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionQuadraticApproximation& ScalarFunctionQuadraticApproximation::setZero(int nx, int nu) {
  f = 0.0;
  dfdx.setZero(nx);
  dfdxx.setZero(nx, nx);
  if (nu >= 0) {
    dfdu.setZero(nu);
    dfdux.setZero(nu, nx);
    dfduu.setZero(nu, nu);
  } else {
    dfdu = vector_t();
    dfdux = matrix_t();
    dfduu = matrix_t();
  }

  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
ScalarFunctionQuadraticApproximation ScalarFunctionQuadraticApproximation::Zero(int nx, int nu) {
  ScalarFunctionQuadraticApproximation f;
  f.setZero(nx, nu);
  return f;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::ostream& operator<<(std::ostream& out, const ScalarFunctionQuadraticApproximation& f) {
  out << "f: " << f.f << '\n';
  out << "dfdx: " << f.dfdx.transpose() << '\n';
  out << "dfdu: " << f.dfdu.transpose() << '\n';
  out << "dfdxx:\n" << f.dfdxx << '\n';
  out << "dfdux:\n" << f.dfdux << '\n';
  out << "dfduu:\n" << f.dfduu << '\n';
  return out;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation::VectorFunctionLinearApproximation(int nv, int nx, int nu) {
  resize(nv, nx, nu);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation& VectorFunctionLinearApproximation::resize(int nv, int nx, int nu) {
  f.resize(nv);
  dfdx.resize(nv, nx);
  if (nu >= 0) {
    dfdu.resize(nv, nu);
  } else {
    dfdu = matrix_t();
  }
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation& VectorFunctionLinearApproximation::setZero(int nv, int nx, int nu) {
  f.setZero(nv);
  dfdx.setZero(nv, nx);
  if (nu >= 0) {
    dfdu.setZero(nv, nu);
  } else {
    dfdu = matrix_t();
  }
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionLinearApproximation VectorFunctionLinearApproximation::Zero(int nv, int nx, int nu) {
  VectorFunctionLinearApproximation f;
  f.setZero(nv, nx, nu);
  return f;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::ostream& operator<<(std::ostream& out, const VectorFunctionLinearApproximation& f) {
  out << "f: " << f.f.transpose() << '\n';
  out << "dfdx:\n" << f.dfdx << '\n';
  out << "dfdu:\n" << f.dfdu << '\n';
  return out;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionQuadraticApproximation::VectorFunctionQuadraticApproximation(int nv, int nx, int nu) {
  resize(nv, nx, nu);
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionQuadraticApproximation& VectorFunctionQuadraticApproximation::resize(int nv, int nx, int nu) {
  f.resize(nv);
  dfdx.resize(nv, nx);
  dfdxx.resize(nv);
  dfdux.resize(nv);
  dfduu.resize(nv);
  for (size_t i = 0; i < static_cast<size_t>(nv); i++) {
    dfdxx[i].resize(nx, nx);
  }

  if (nu >= 0) {
    dfdu.resize(nv, nu);
    for (size_t i = 0; i < static_cast<size_t>(nv); i++) {
      dfdux[i].resize(nu, nx);
      dfduu[i].resize(nu, nu);
    }
  } else {
    dfdu = matrix_t();
    for (size_t i = 0; i < static_cast<size_t>(nv); i++) {
      dfdux[i] = matrix_t();
      dfduu[i] = matrix_t();
    }
  }
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionQuadraticApproximation& VectorFunctionQuadraticApproximation::setZero(int nv, int nx, int nu) {
  f.setZero(nv);
  dfdx.setZero(nv, nx);
  dfdxx.resize(nv);
  for (size_t i = 0; i < static_cast<size_t>(nv); i++) {
    dfdxx[i].setZero(nx, nx);
  }

  if (nu >= 0) {
    dfdu.setZero(nv, nu);
    dfdux.resize(nv);
    dfduu.resize(nv);
    for (size_t i = 0; i < static_cast<size_t>(nv); i++) {
      dfdux[i].setZero(nu, nx);
      dfduu[i].setZero(nu, nu);
    }
  } else {
    dfdu = matrix_t();
    dfdux.resize(nv);
    dfduu.resize(nv);
    for (size_t i = 0; i < static_cast<size_t>(nv); i++) {
      dfdux[i] = matrix_t();
      dfduu[i] = matrix_t();
    }
  }
  return *this;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
VectorFunctionQuadraticApproximation VectorFunctionQuadraticApproximation::Zero(int nv, int nx, int nu) {
  VectorFunctionQuadraticApproximation f;
  f.setZero(nv, nx, nu);
  return f;
}

/******************************************************************************************************/
/******************************************************************************************************/
/******************************************************************************************************/
std::ostream& operator<<(std::ostream& out, const VectorFunctionQuadraticApproximation& f) {
  out << "f: " << f.f.transpose() << '\n';
  out << "dfdx:\n" << f.dfdx << '\n';
  out << "dfdu:\n" << f.dfdu << '\n';
  for (size_t i = 0; i < static_cast<size_t>(f.f.rows()); i++) {
    out << "dfdxx[" << i << "]:\n" << f.dfdxx[i] << '\n';
  }
  for (size_t i = 0; i < static_cast<size_t>(f.f.rows()); i++) {
    out << "dfdux[" << i << "]:\n" << f.dfdux[i] << '\n';
  }
  for (size_t i = 0; i < static_cast<size_t>(f.f.rows()); i++) {
    out << "dfduu[" << i << "]:\n" << f.dfduu[i] << '\n';
  }
  return out;
}

}  // namespace ocs2
