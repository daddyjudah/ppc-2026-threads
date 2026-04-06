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

  static bool IsBinary(const Image &img);

  void FirstPassTBB();
  void SecondPassTBB();
  void ConvertLabelsToOutput();

  std::vector<std::uint8_t> binary_;
  std::vector<int> labels_flat_;
  Labels labels_;
  std::vector<int> parent_;
  std::vector<int> root_to_compact_;
  int height_ = 0;
  int width_ = 0;
  int next_label_ = 1;
};

}  // namespace marin_l_mark_components
