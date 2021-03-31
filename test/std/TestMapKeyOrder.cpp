#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <unordered_map>

// proved that the keys of the map are ordered when accessed with the iterator
TEST(StandardC, MapKeyOrder) {
  std::map<uint64_t, char> ma;
  for (int i = 0; i < 1e4; ++i)
    ma.insert(std::make_pair(uint64_t(10 + i * 13 + std::rand() % 10),
                             char(std::rand() % 255)));
  uint64_t last = ma.begin()->first - 1;
  for (auto it = ma.begin(); it != ma.end(); ++it) {
    ASSERT_LT(last, it->first);
    last = it->first;
  }
}

TEST(StandardC, UnorderedMap) {
  std::unordered_map<std::string, double> ma = {
      {"mom", 5.4},     {"dad", 6.1},     {"brother", 5.9}, {"sister", 5.7},
      {"grandpa", 5.9}, {"grandma", 5.5}, {"aunt", 5.5},    {"uncle", 6.0}};

  std::string last = "apple";
  bool inOrder = true;
  for (std::unordered_map<std::string, double>::const_iterator it = ma.begin();
       it != ma.end(); ++it) {
    if (last.compare(it->first) > 0) {
      inOrder = false;
      break;
    }
    last = it->first;
  }
  EXPECT_FALSE(inOrder);
}

TEST(StandardC, OrderedSearchMap) {
  int modulo = 11;
  std::vector<uint64_t> vint;
  for (int j = 0; j < modulo / 2; ++j) {
    vint.push_back(std::rand() % modulo);
  }
  std::sort(vint.begin(), vint.end());
  std::map<uint64_t, int> mint;
  for (int j = 0; j < modulo + 2; ++j) {
    mint.emplace(std::rand() % modulo, j);
  }
  std::map<uint64_t, int> ref_mint = mint;
  // method 1:
  auto obsIter = mint.begin();
  for (uint64_t val : vint) {
    while (obsIter != mint.end() && obsIter->first < val) {
      ++obsIter;
    }
    if (obsIter == mint.end()) {
      break;
    }
    if (obsIter->first == val) {
      obsIter = mint.erase(obsIter);
    }
  }
  // method 2
  for (uint64_t val : vint) {
    for (auto obsIter = ref_mint.begin(); obsIter != ref_mint.end();
         ++obsIter) {
      if (obsIter->first == val) {
        ref_mint.erase(obsIter);
        break;
      }
    }
  }

  // check
  EXPECT_EQ(mint.size(), ref_mint.size());
  auto mit = mint.begin();
  auto ref_it = ref_mint.begin();
  for (; mit != mint.end(); ++mit, ++ref_it) {
    EXPECT_EQ(mit->first, ref_it->first);
    EXPECT_EQ(mit->second, ref_it->second);
  }
}
