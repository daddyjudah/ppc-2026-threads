#pragma once

#include <vector>

#include "marin_l_mark_components/common/include/common.hpp"
#include "task/include/task.hpp"

namespace marin_l_mark_components {

class MarinLMarkComponentsOMP : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kOMP;
  }

  explicit MarinLMarkComponentsOMP(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static bool IsBinary(const Image &img);

  void FirstPass();
  void SecondPass();

  void ProcessChunk(std::vector<int> &parent, int start_row, int end_row, int width);

  void MergeBorders(std::vector<int> &parent, int height, int width, int chunk, int num_threads);

  Image binary_;
  Labels labels_;
};

}  // namespace marin_l_mark_components
