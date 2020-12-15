#include <gtest/gtest.h>
#include <iostream>


void foo(double** jacptr) {
  EXPECT_EQ(sizeof(jacptr), 8u);
  ASSERT_EQ(sizeof(jacptr[0]), 8u);
}

TEST(StandardC, Sizeof) {
  double* jac[3];
  foo(jac);

  ASSERT_EQ(sizeof(jac), 24u);
  ASSERT_EQ(sizeof(jac[0]), 8u);
}
