#include <algorithm>
#include <deque>
#include <iostream>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

// Finding: (1) to remove entries based on their stored iterators are fragile,
// does not work for large deque

TEST(std, DequeIterator) {
  std::deque<int> q;
  size_t totalLength = 10;
  for (size_t jack = 0; jack < totalLength; ++jack) q.push_back(jack);

  std::vector<std::deque<int>::iterator> vToRemove;
  for (size_t jack = 0; jack < totalLength; jack += 2)
    vToRemove.push_back(q.begin() + jack);
  for (size_t jack = 0; jack < vToRemove.size(); ++jack) {
    q.erase(vToRemove[jack]);
  }

  size_t counter = 0;
  LOG(INFO) << "max value in deque " << q.back();
  try {
    for (auto it = q.begin(); it != q.end(); ++it, ++counter) {
      LOG(INFO) << *it << " " << (counter * 2 + 1);
      if (*it != (int)(counter * 2 + 1)) throw(*it);
    }
  } catch (int e) {
    LOG(INFO) << "even number in deque is not properly deleted at " << e;
  }

  q.clear();
  totalLength = 1e4;
  int checkOddNumber = 1e3 + 1;  // must be smaller than totalLength
  for (size_t jack = 0; jack < totalLength; ++jack)
    q.push_back(jack);
  for (auto it = q.begin(); it != q.end();) {
    if ((*it) % 2 == 0) {
      it = q.erase(it);
    } else {
      ++it;
    }
    if (*it >= checkOddNumber) {
      std::deque<int>::iterator idPos =
          std::find(q.begin(), q.end(), checkOddNumber);
      EXPECT_EQ(idPos - q.begin(), checkOddNumber / 2);
    }
  }
}
