#include <vector>

#include "caffe2/core/context_gpu.h"
#include "caffe2/core/flags.h"
#include "caffe2/operators/conv_op_cache_cudnn.h"
#include <gtest/gtest.h>

CAFFE2_DECLARE_string(caffe_test_root);

namespace caffe2 {

TEST(AlgorithmsCacheTest, CachesCorrectly) {
  AlgorithmsCache<int> cache;
  int result = cache.getAlgorithm(
      std::vector<TIndex>(1), std::vector<TIndex>(1), 0, []() { return 5; });
  EXPECT_EQ(result, 5);

  int res2 = cache.getAlgorithm(
      std::vector<TIndex>(1), std::vector<TIndex>(1), 0, []() { return 10; });

  EXPECT_EQ(res2, 5);
}

TEST(AlgorithmsCacheTest, KeysDifferIfOneVectorIsEmpty) {
  AlgorithmsCache<int> cache;
  int result = cache.getAlgorithm(
      std::vector<TIndex>(1, 10), std::vector<TIndex>(), 0, []() { return 5; });
  EXPECT_EQ(result, 5);

  int res2 = cache.getAlgorithm(
      std::vector<TIndex>(), std::vector<TIndex>(1, 10), 0, []() {
        return 10;
      });

  EXPECT_EQ(res2, 10);
}

TEST(AlgorithmsCacheTest, KeysDifferIfFlagsAreDifferent) {
  AlgorithmsCache<int> cache;
  int result = cache.getAlgorithm(
      std::vector<TIndex>{2, 3, 4}, std::vector<TIndex>{5, 6}, 123, []() {
        return 5;
      });
  EXPECT_EQ(result, 5);

  int res2 = cache.getAlgorithm(
      std::vector<TIndex>{2, 3, 4}, std::vector<TIndex>{5, 6}, 456, []() {
        return 10;
      });

  EXPECT_EQ(res2, 10);

  int res3 = cache.getAlgorithm(
      std::vector<TIndex>{2, 3, 4}, std::vector<TIndex>{5, 6}, 456, []() {
        return 15;
      });

  EXPECT_EQ(res3, 10);
}

} // namespace caffe2
