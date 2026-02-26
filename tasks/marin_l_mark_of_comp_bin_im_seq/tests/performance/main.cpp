#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "marin_l_mark_of_comp_bin_im_seq/common/include/common.hpp"
#include "marin_l_mark_of_comp_bin_im_seq/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace marin_l_mark_of_comp_bin_im_seq {

class MarinLMarkOfCompBinImPerfTestsThreads : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  InType GetTestInputData() override {
    const int size = 1000;

    InType img(size, std::vector<int>(size));

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> dist(0, 1);

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        img[i][j] = dist(gen);
      }
    }

    return img;
  }

  bool CheckTestOutputData(OutType &output_data) override {
    (void)output_data;
    return true;
  }
};

TEST_P(MarinLMarkOfCompBinImPerfTestsThreads, RunPerfModes) {
  ExecuteTest(GetParam());
}

namespace {

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MarinLMarkOfCompBinImSEQ>(PPC_SETTINGS_marin_l_mark_of_comp_bin_im_seq);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

INSTANTIATE_TEST_SUITE_P(PerfTests, MarinLMarkOfCompBinImPerfTestsThreads, kGtestValues);

}  // namespace

}  // namespace marin_l_mark_of_comp_bin_im_seq
