#pragma once

namespace clang_tidy_testdata {

class Counter {
 public:
  Counter();
  int count() const;

 private:
  int count_;
};

}  // namespace clang_tidy_testdata
