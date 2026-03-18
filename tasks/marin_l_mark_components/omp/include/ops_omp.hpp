#pragma once

#include <omp.h>

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

  void ProcessRow(int row, int width, int &next_label, std::vector<int> &parent,
                  std::vector<std::vector<int>> &local_labels);
  void MergeLabels(int height, int width, std::vector<int> &parent);
  void RelabelComponents(int height, int width, std::vector<int> &parent);

  Image binary_;
  Labels labels_;
};

}  // namespace marin_l_mark_components
