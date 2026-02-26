#pragma once

#include <vector>

#include "marin_l_mark_of_comp_bin_im_seq/common/include/common.hpp"
#include "task/include/task.hpp"

namespace marin_l_mark_of_comp_bin_im_seq {

class MarinLMarkOfCompBinImSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit MarinLMarkOfCompBinImSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void DFS(int x, int y);

  std::vector<std::vector<int>> image_;
  int current_label_{};
  int rows_{};
  int cols_{};
};

}  // namespace marin_l_mark_of_comp_bin_im_seq
