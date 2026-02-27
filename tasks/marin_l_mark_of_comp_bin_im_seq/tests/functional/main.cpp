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

bool ValidateTask(const Image &input) {
  MarinLMarkOfCompBinImSEQ task(input);
  return task.Validation();
}

bool PreProcessTask(const Image &input) {
  MarinLMarkOfCompBinImSEQ task(input);
  return task.Validation() && task.PreProcessing();
}

bool RunTask(const Image &input) {
  MarinLMarkOfCompBinImSEQ task(input);
  return task.Validation() && task.PreProcessing() && task.Run();
}

bool FullProcessTask(const Image &input) {
  MarinLMarkOfCompBinImSEQ task(input);
  return task.Validation() && task.PreProcessing() && task.Run() && task.PostProcessing();
}

Image GetTaskOutput(const Image &input) {
  MarinLMarkOfCompBinImSEQ task(input);
  if (task.Validation() && task.PreProcessing() && task.Run() && task.PostProcessing()) {
    return task.GetOutput();
  }
  return Image{};
}

}  // namespace

TEST_P(MarinLMarkOfCompBinImFuncTestsThreads, ConnectedComponents) {
  ExecuteTest(GetParam());
}

TEST(MarinLMarkOfCompBinImValidation, EmptyInput) {
  EXPECT_FALSE(ValidateTask(Image{}));
}

TEST(MarinLMarkOfCompBinImValidation, EmptyRow) {
  Image input = {{}};
  EXPECT_FALSE(ValidateTask(input));
}

TEST(MarinLMarkOfCompBinImValidation, NonRectangularInput) {
  Image input = {{1, 0}, {1, 0, 1}};
  EXPECT_FALSE(ValidateTask(input));
}

TEST(MarinLMarkOfCompBinImSeqTest, ComponentAtImageBorder) {
  Image input = {{1, 1, 1}, {1, 0, 0}, {1, 0, 0}};
  Image expected = {{1, 1, 1}, {1, 0, 0}, {1, 0, 0}};

  EXPECT_TRUE(ValidateTask(input));
  EXPECT_TRUE(PreProcessTask(input));
  EXPECT_TRUE(RunTask(input));
  EXPECT_TRUE(FullProcessTask(input));
  EXPECT_EQ(GetTaskOutput(input), expected);
}

TEST(MarinLMarkOfCompBinImSeqTest, AllDirections) {
  Image input = {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}};
  Image expected = {{0, 1, 0}, {1, 1, 1}, {0, 1, 0}};

  EXPECT_TRUE(ValidateTask(input));
  EXPECT_TRUE(PreProcessTask(input));
  EXPECT_TRUE(RunTask(input));
  EXPECT_TRUE(FullProcessTask(input));
  EXPECT_EQ(GetTaskOutput(input), expected);
}

TEST(MarinLMarkOfCompBinImSeqTest, ZeroPixel) {
  Image input = {{0}};
  Image expected = {{0}};

  EXPECT_TRUE(ValidateTask(input));
  EXPECT_TRUE(PreProcessTask(input));
  EXPECT_TRUE(RunTask(input));
  EXPECT_TRUE(FullProcessTask(input));
  EXPECT_EQ(GetTaskOutput(input), expected);
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
