#pragma once

#include <vector>

#include "marin_l_mark_components/common/include/common.hpp"
#include "task/include/task.hpp"

namespace marin_l_mark_components {

class MarinLMarkComponentsTBB : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kTBB;
  }

  explicit MarinLMarkComponentsTBB(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  struct Run {
    int row;
    int l, r;
    int label;
  };

  void BuildRLE();
  void MergeRuns();
  void Flatten();
  void ExpandToImage();

  std::vector<uint8_t> binary_;
  std::vector<Run> runs_;
  std::vector<int> parent_;
  std::vector<int> labels_flat_;
  std::vector<std::vector<int>> row_runs_;

  int height_ = 0;
  int width_ = 0;
};

}  // namespace marin_l_mark_components
