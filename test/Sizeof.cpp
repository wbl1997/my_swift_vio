#include <gtest/gtest.h>
#include <iostream>

#include <okvis/Player.hpp>

void testArrayArgument(double** jacptr) {
  EXPECT_EQ(sizeof(jacptr), 8);
  ASSERT_EQ(sizeof(jacptr[0]), 8);
}

TEST(StandardC, Sizeof) {
  double* jac[3];
  testArrayArgument(jac);

  ASSERT_EQ(sizeof(jac), 24);
  ASSERT_EQ(sizeof(jac[0]), 8);
}

TEST(StandardC, removeTrailingSlash) {
  std::string path1 = "/a/b";
  ASSERT_EQ(path1, okvis::removeTrailingSlash(path1));
  std::string path2 = "/a/b/";
  ASSERT_EQ("/a/b", okvis::removeTrailingSlash(path2));
  std::string path3 = "/a\\b\\";
  ASSERT_EQ("/a\\b", okvis::removeTrailingSlash(path3));
  std::string path4 = "/a/b//";
  ASSERT_EQ("/a/b", okvis::removeTrailingSlash(path4));
}
