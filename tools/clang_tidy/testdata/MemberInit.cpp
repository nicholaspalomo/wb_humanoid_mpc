#include "tools/clang_tidy/testdata/MemberInit.h"

namespace clang_tidy_testdata {

Counter::Counter() : count_(0) {}

int Counter::count() const {
  return count_;
}

}  // namespace clang_tidy_testdata
