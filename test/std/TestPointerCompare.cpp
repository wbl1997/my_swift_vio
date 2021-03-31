/**
 * @file TestPointerCompare.cpp
 * @brief Unit test for comparing pointers.
 * @author Jianzhu Huai
 */

#include <gtest/gtest.h>

class Keeper {
 private:
  const int* a_;

 public:
  void setPtr(const int& a) { a_ = &a; }
  const int* getPtr() const { return a_; }
};

TEST(StandardC, constPtrEqPtr) {
  int m('m'), n('n');
  Keeper mk, nk;
  mk.setPtr(m);
  nk.setPtr(n);

  const int* mp = mk.getPtr();
  int* rawmp = &m;
  EXPECT_TRUE(mp == rawmp);
  EXPECT_FALSE(mp == &n);
}
