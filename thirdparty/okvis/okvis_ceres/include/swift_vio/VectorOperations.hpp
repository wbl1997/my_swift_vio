#ifndef INCLUDE_SWIFT_VIO_REMOVE_FROM_VECTOR_HPP_
#define INCLUDE_SWIFT_VIO_REMOVE_FROM_VECTOR_HPP_

#include <vector>
#include <Eigen/Core>
#include <Eigen/StdVector>

namespace swift_vio {
template <typename Derived>
void removeUnsetMatrices(
    std::vector<Derived, Eigen::aligned_allocator<Derived>>* matrices,
    const std::vector<bool>& markers) {
  //  if (matrices->size() != markers.size()) {
  //    std::cerr << "The input size of matrices(" << matrices->size()
  //              << ") and markers(" << markers.size() << ") does not
  //              match.\n";
  //  }
  auto iter = matrices->begin();
  auto keepIter = matrices->begin();
  for (size_t i = 0; i < markers.size(); ++i) {
    if (!markers[i]) {
      ++iter;
    } else {
      if (keepIter != iter) *keepIter = *iter;
      ++iter;
      ++keepIter;
    }
  }
  matrices->resize(keepIter - matrices->begin());
}

template <typename T>
void removeUnsetElements(std::vector<T>* elements,
                         const std::vector<bool>& markers) {
  //  if (elements->size() != markers.size()) {
  //    std::cerr << "The input size of elements(" << elements->size()
  //              << ") and markers(" << markers.size() << ") does not
  //              match.\n";
  //  }
  auto iter = elements->begin();
  auto keepIter = elements->begin();
  for (size_t i = 0; i < markers.size(); ++i) {
    if (!markers[i]) {
      ++iter;
    } else {
      *keepIter = *iter;
      ++iter;
      ++keepIter;
    }
  }
  elements->resize(keepIter - elements->begin());
}

template <typename T>
void removeUnsetElements(std::vector<T>* elements,
                         const std::vector<bool>& markers, const int step) {
  //  if (elements->size() != markers.size()) {
  //    std::cerr << "The input size of elements(" << elements->size()
  //              << ") and markers(" << markers.size() << ") does not
  //              match.\n";
  //  }
  auto iter = elements->begin();
  auto keepIter = elements->begin();
  for (size_t i = 0; i < markers.size(); ++i) {
    if (!markers[i]) {
      iter += step;
    } else {
      for (int j = 0; j < step; ++j) {
        *keepIter = *iter;
        ++iter;
        ++keepIter;
      }
    }
  }
  elements->resize(keepIter - elements->begin());
}

/**
 * select matrics from a vector of Eigen types.
 * do not use this function with Eigen types.
 */
template <class T, class Index>
std::vector<T, Eigen::aligned_allocator<T>> selectMatrices(
    const std::vector<T, Eigen::aligned_allocator<T>>& inArray,
    const std::vector<Index>& indices) {
  std::vector<T, Eigen::aligned_allocator<T>> result;
  result.reserve(indices.size());
  for (auto val : indices) {
    result.push_back(inArray[val]);
  }
  return result;
}

/**
 * T, element type
 * Index, index type
 * warning: do not use this function with Eigen types.
 */
template <class T, class Index>
std::vector<T> selectElements(const std::vector<T>& inArray,
                              const std::vector<Index>& indices) {
  std::vector<T> result;
  result.reserve(indices.size());
  for (auto val : indices) {
    result.push_back(inArray[val]);
  }
  return result;
}

/**
 * @brief Does a vector contain a certain element.
 * @tparam Class of a vector element.
 * @param vector Vector to search element in.
 * @param query Element to search for.
 * @return True if query is an element of vector.
 */
template<class T>
bool vectorContains(const std::vector<T> & vector, const T & query){
  for(size_t i=0; i<vector.size(); ++i){
    if(vector[i] == query){
      return true;
    }
  }
  return false;
}

} // namespace swift_vio
#endif  // INCLUDE_SWIFT_VIO_REMOVE_FROM_VECTOR_HPP_
