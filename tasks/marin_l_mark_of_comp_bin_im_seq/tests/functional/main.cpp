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

namespace {

void CheckValidation(MarinLMarkOfCompBinImSEQ &task) {
  EXPECT_TRUE(task.Validation());
}

void CheckPreProcessing(MarinLMarkOfCompBinImSEQ &task) {
  ASSERT_TRUE(task.PreProcessing());
}

void CheckRun(MarinLMarkOfCompBinImSEQ &task) {
  ASSERT_TRUE(task.Run());
}

void CheckPostProcessing(MarinLMarkOfCompBinImSEQ &task) {
  ASSERT_TRUE(task.PostProcessing());
}

void CheckOutput(MarinLMarkOfCompBinImSEQ &task, const Image &expected) {
  EXPECT_EQ(task.GetOutput(), expected);
}

void TestImage(const Image &input, const Image &expected) {
  MarinLMarkOfCompBinImSEQ task(input);

  CheckValidation(task);
  CheckPreProcessing(task);
  CheckRun(task);
  CheckPostProcessing(task);
  CheckOutput(task, expected);
}

void TestValidation(const Image &input, bool expected_valid) {
  MarinLMarkOfCompBinImSEQ task(input);
  EXPECT_EQ(task.Validation(), expected_valid);
}

}  // namespace

TEST_P(MarinLMarkOfCompBinImFuncTestsThreads, ConnectedComponents) {
  ExecuteTest(GetParam());
}

TEST(MarinLMarkOfCompBinImValidation, EmptyInput) {
  TestValidation(Image{}, false);
}

TEST(MarinLMarkOfCompBinImValidation, EmptyRow) {
  TestValidation(Image{{}}, false);
}

TEST(MarinLMarkOfCompBinImValidation, NonRectangularInput) {
  TestValidation(Image{{1, 0}, {1, 0, 1}}, false);
}

TEST(MarinLMarkOfCompBinImSeqTest, ComponentAtImageBorder) {
  TestImage({{1, 1, 1}, {1, 0, 0}, {1, 0, 0}}, {{1, 1, 1}, {1, 0, 0}, {1, 0, 0}});
}

TEST(MarinLMarkOfCompBinImSeqTest, AllDirections) {
  TestImage({{0, 1, 0}, {1, 1, 1}, {0, 1, 0}}, {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}});
}

TEST(MarinLMarkOfCompBinImSeqTest, ZeroPixel) {
  TestImage({{0}}, {{0}});
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
