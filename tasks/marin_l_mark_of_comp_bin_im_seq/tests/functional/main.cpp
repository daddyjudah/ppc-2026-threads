#include <gtest/gtest.h>

#include <array>
#include <tuple>

#include "marin_l_mark_of_comp_bin_im_seq/common/include/common.hpp"
#include "marin_l_mark_of_comp_bin_im_seq/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace marin_l_mark_of_comp_bin_im_seq {

class MarinLMarkOfCompBinImFuncTestsThreads : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  bool CheckTestOutputData(OutType &output_data) override {
    const auto &test_data = std::get<2>(GetParam());
    return output_data == std::get<1>(test_data);
  }

  InType GetTestInputData() override {
    const auto &test_data = std::get<2>(GetParam());
    return std::get<0>(test_data);
  }
};

TEST_P(MarinLMarkOfCompBinImFuncTestsThreads, ConnectedComponents) {
  ExecuteTest(GetParam());
}

TEST(MarinLMarkOfCompBinImValidation, EmptyInput) {
  MarinLMarkOfCompBinImSEQ task(Image{});
  ASSERT_FALSE(task.Validation());
}

TEST(MarinLMarkOfCompBinImValidation, EmptyRow) {
  Image input = {{}};
  MarinLMarkOfCompBinImSEQ task(input);
  ASSERT_FALSE(task.Validation());
}

TEST(MarinLMarkOfCompBinImSeqTest, ComponentAtImageBorder) {
  Image input = {{1, 1, 1}, {1, 0, 0}, {1, 0, 0}};
  Image expected = {{1, 1, 1}, {1, 0, 0}, {1, 0, 0}};

  MarinLMarkOfCompBinImSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetOutput(), expected);
}

TEST(MarinLMarkOfCompBinImSeqTest, AllDirections) {
  Image input = {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}};
  Image expected = {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}};

  MarinLMarkOfCompBinImSEQ task(input);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  EXPECT_EQ(task.GetOutput(), expected);
}

const std::array<TestType, 9> kTests = {{

    std::make_tuple(Image{{1, 1, 0}, {1, 1, 0}, {0, 0, 0}}, Image{{1, 1, 0}, {1, 1, 0}, {0, 0, 0}}),

    std::make_tuple(Image{{1, 0, 1}, {0, 0, 0}, {1, 0, 1}}, Image{{1, 0, 2}, {0, 0, 0}, {3, 0, 4}}),

    std::make_tuple(Image{{0, 0}, {0, 0}}, Image{{0, 0}, {0, 0}}),

    std::make_tuple(Image{{1, 1}, {1, 1}}, Image{{1, 1}, {1, 1}}),

    std::make_tuple(Image{{1, 0, 1, 1}, {1, 0, 0, 0}, {0, 0, 1, 0}, {1, 1, 0, 0}},
                    Image{{1, 0, 2, 2}, {1, 0, 0, 0}, {0, 0, 3, 0}, {4, 4, 0, 0}}),

    std::make_tuple(Image{{1}}, Image{{1}}),

    std::make_tuple(Image{{0}}, Image{{0}}),

    std::make_tuple(Image{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}}, Image{{1, 0, 0}, {0, 0, 0}, {0, 0, 0}}),

    std::make_tuple(Image{{1}, {1}, {1}}, Image{{1}, {1}, {1}})}};

namespace {

const auto kTestTasksList =
    ppc::util::AddFuncTask<MarinLMarkOfCompBinImSEQ, InType>(kTests, PPC_SETTINGS_marin_l_mark_of_comp_bin_im_seq);

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

INSTANTIATE_TEST_SUITE_P(FuncTests, MarinLMarkOfCompBinImFuncTestsThreads, kGtestValues);

}  // namespace

}  // namespace marin_l_mark_of_comp_bin_im_seq
