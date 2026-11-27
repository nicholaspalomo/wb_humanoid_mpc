#ifndef CPPAD_CG_IDENTICAL_INCLUDED
#define CPPAD_CG_IDENTICAL_INCLUDED
/* --------------------------------------------------------------------------
 *  CppADCodeGen: C++ Algorithmic Differentiation with Source Code Generation:
 *    Copyright (C) 2012 Ciengis
 *
 *  CppADCodeGen is distributed under multiple licenses:
 *
 *   - Eclipse Public License Version 1.0 (EPL1), and
 *   - GNU General Public License Version 3 (GPL3).
 *
 *  EPL1 terms and conditions can be found in the file "epl-v10.txt", while
 *  terms and conditions for the GPL3 can be found in the file "gpl3.txt".
 * ----------------------------------------------------------------------------
 * Author: Joao Leal
 */

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace CppAD {

template<class Base>
inline bool IdenticalPar(const CppAD::cg::CG<Base>& x) {
    if (!x.isParameter()) {
        return false; // its value may change after tapping
    }
    return CppAD::IdenticalPar(x.getValue());
}

template<class Base>
inline bool IdenticalCon(const CppAD::cg::CG<Base>& x) {
    if (!x.isParameter()) {
        return false; // its value may change after tapping
    }
    return CppAD::IdenticalCon(x.getValue());
}

template<class Base>
inline bool IdenticalZero(const CppAD::cg::CG<Base>& x) {
    if (!x.isParameter()) {
        return false; // its value may change after tapping
    }
    return CppAD::IdenticalZero(x.getValue());
}

template<class Base>
inline bool IdenticalOne(const CppAD::cg::CG<Base>& x) {
    if (!x.isParameter()) {
        return false; // its value may change after tapping
    }
    return CppAD::IdenticalOne(x.getValue());
}

template<class Base>
inline bool IdenticalEqualPar(const CppAD::cg::CG<Base>& x,
                              const CppAD::cg::CG<Base>& y) {
    return x.isParameter() && y.isParameter() && CppAD::IdenticalEqualPar(x.getValue(), y.getValue());
}

template<class Base>
inline bool IdenticalEqualCon(const CppAD::cg::CG<Base>& x,
                              const CppAD::cg::CG<Base>& y) {
    return x.isParameter() && y.isParameter() && CppAD::IdenticalEqualCon(x.getValue(), y.getValue());
}

// LINT.IfChange(cg_constant_hash)
/**
 * The hash code CppAD's recorder files a constant under (local/recorder.hpp, put_con_par()), which picks the one
 * earlier constant the new one may share a parameter index with; IdenticalEqualCon() still decides whether it does.
 *
 * Local change of this fork (lib/ocs2/README.md). CppADCodeGen defined none, so CppAD's default applied
 * (local/hash_code.hpp), which sums the bytes of the object: for a CG its node pointer and the heap address of its
 * value. Equal constants then hashed apart or together by where malloc put them, and so did the deduplication of
 * the tape's constants, optimize()'s common subexpressions built on their indices, and the generated sources: two
 * generations of one function gave different C code. A constant now hashes the bytes of its value, as CppAD hashes
 * a double, so +0 and -0 hash apart and each keeps its sign. A variable, which IdenticalEqualCon() never matches,
 * gets a fixed code.
 */
template<class Base>
inline unsigned short hash_code(const CppAD::cg::CG<Base>& x) {
    static_assert(std::is_trivially_copyable<Base>::value && sizeof(Base) % sizeof(unsigned short) == 0,
                  "hash_code(CG<Base>) hashes the bytes of Base");
    if (!x.isParameter()) {
        return 0;
    }
    constexpr std::size_t wordCount = sizeof(Base) / sizeof(unsigned short);
    unsigned short words[wordCount];
    std::memcpy(words, &x.getValue(), sizeof(Base));  // CppAD's default reinterpret_casts; memcpy does not alias
    std::size_t sum = 0;
    for (std::size_t i = 0; i < wordCount; ++i) {
        sum += words[i];
    }
    return static_cast<unsigned short>(sum % CPPAD_HASH_TABLE_SIZE);
}
// LINT.ThenChange(//lib/ocs2/core/include/ocs2_core/automatic_differentiation/CppAdInterface.h:cppad_generator_tag)

} // END CppAD namespace

#endif
