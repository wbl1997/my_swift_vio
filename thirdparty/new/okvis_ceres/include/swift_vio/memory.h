#ifndef ASLAM_COMMON_MEMORY_H_
#define ASLAM_COMMON_MEMORY_H_

// adapted from maplab/aslam_cv2/aslam_cv_common/include/aslam/common/memory.h

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <Eigen/Core>
#include <Eigen/StdVector>

namespace Eigen {
template <typename Type>
using AlignedVector = std::vector<Type, Eigen::aligned_allocator<Type>>;

template <typename T>
using AlignedDeque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename KeyType, typename ValueType>
using AlignedMap =
    std::map<KeyType, ValueType, std::less<KeyType>,
             Eigen::aligned_allocator<std::pair<const KeyType, ValueType>>>;

template <typename KeyType, typename ValueType>
using AlignedUnorderedMap = std::unordered_map<
    KeyType, ValueType, std::hash<KeyType>, std::equal_to<KeyType>,
    Eigen::aligned_allocator<std::pair<const KeyType, ValueType>>>;

template <typename KeyType, typename ValueType>
using AlignedUnorderedMultimap = std::unordered_multimap<
    KeyType, ValueType, std::hash<KeyType>, std::equal_to<KeyType>,
    Eigen::aligned_allocator<std::pair<const KeyType, ValueType>>>;

template <typename Type>
using AlignedUnorderedSet =
    std::unordered_set<Type, std::hash<Type>, std::equal_to<Type>,
                       Eigen::aligned_allocator<Type>>;
} // namespace Eigen

#endif // ASLAM_COMMON_MEMORY_H_
