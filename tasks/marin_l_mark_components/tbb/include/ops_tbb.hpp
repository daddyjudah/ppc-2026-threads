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

  void FirstPassTiles();
  void MergeTileBorders();
  void SecondPassTiles();

  std::vector<std::uint8_t> binary_;
  std::vector<int> labels_flat_;
  std::vector<int> parent_;
  std::vector<int> tile_offsets_;
  std::vector<int> tile_used_counts_;
  std::vector<int> root_to_compact_;
  std::vector<int> root_generation_;

  int height_ = 0;
  int width_ = 0;
  int tile_rows_ = 0;
  int tile_cols_ = 0;
  int tile_count_ = 0;
  int max_label_id_ = 0;
  int generation_id_ = 1;
};

}  // namespace marin_l_mark_components
