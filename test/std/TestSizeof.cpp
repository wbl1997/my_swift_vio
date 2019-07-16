#include <gtest/gtest.h>
#include <iostream>


void foo(double** jacptr) {
  EXPECT_EQ(sizeof(jacptr), 8);
  ASSERT_EQ(sizeof(jacptr[0]), 8);
}

TEST(StandardC, Sizeof) {
  double* jac[3];
  foo(jac);

  ASSERT_EQ(sizeof(jac), 24);
  ASSERT_EQ(sizeof(jac[0]), 8);
}
