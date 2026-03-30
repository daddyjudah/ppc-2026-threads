#pragma once

#include <cstdint>
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

  int height_ = 0;
  int width_ = 0;
  std::vector<int> labels_flat_;
  std::vector<std::uint8_t> binary_flat_;
  std::vector<int> parent_;

  void FirstPassTBB();
  void MergeBordersTBB();
  void SecondPassTBB();
};

}  // namespace marin_l_mark_components
