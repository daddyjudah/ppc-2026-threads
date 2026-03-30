#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;
constexpr int kTileHeight = 128;
constexpr int kTileWidth = 128;

struct TileBounds {
  int row_begin;
  int row_end;
  int col_begin;
  int col_end;
  int base_label;
};

int FindRoot(std::vector<int> &parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

void UnionLabels(std::vector<int> &parent, int a, int b) {
  int root_a = FindRoot(parent, a);
  int root_b = FindRoot(parent, b);
  if (root_a == root_b) {
    return;
  }
  if (root_a < root_b) {
    parent[root_b] = root_a;
  } else {
    parent[root_a] = root_b;
  }
}

int GetTileIndex(int tile_row, int tile_col, int tile_cols) {
  return (tile_row * tile_cols) + tile_col;
}

TileBounds GetTileBounds(int tile_index, int tile_cols, int height, int width, const std::vector<int> &tile_offsets) {
  const int tile_row = tile_index / tile_cols;
  const int tile_col = tile_index % tile_cols;

  const int row_begin = tile_row * kTileHeight;
  const int row_end = std::min(row_begin + kTileHeight, height);
  const int col_begin = tile_col * kTileWidth;
  const int col_end = std::min(col_begin + kTileWidth, width);

  return {
      .row_begin = row_begin,
      .row_end = row_end,
      .col_begin = col_begin,
      .col_end = col_end,
      .base_label = 1 + tile_offsets[static_cast<std::size_t>(tile_index)],
  };
}

void ProcessTile(const std::vector<std::uint8_t> &binary, std::vector<int> &labels_flat, std::vector<int> &parent,
                 std::vector<int> &tile_used_counts, int width, const TileBounds &tile, int tile_index) {
  const std::uint8_t *binary_ptr = binary.data();
  int *labels_ptr = labels_flat.data();
  int *parent_ptr = parent.data();
  int next_label = tile.base_label;

  for (int row = tile.row_begin; row < tile.row_end; ++row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
    const bool has_top_inside_tile = row > tile.row_begin;
    const std::size_t top_row_offset = has_top_inside_tile ? row_offset - static_cast<std::size_t>(width) : 0;

    for (int col = tile.col_begin; col < tile.col_end; ++col) {
      const std::size_t idx = row_offset + static_cast<std::size_t>(col);
      if (binary_ptr[idx] == 0U) {
        labels_ptr[idx] = 0;
        continue;
      }

      const int left_label = (col > tile.col_begin) ? labels_ptr[idx - 1ULL] : 0;
      const int top_label = has_top_inside_tile ? labels_ptr[top_row_offset + static_cast<std::size_t>(col)] : 0;

      if (left_label == 0) {
        if (top_label == 0) {
          parent_ptr[static_cast<std::size_t>(next_label)] = next_label;
          labels_ptr[idx] = next_label++;
          continue;
        }
        labels_ptr[idx] = top_label;
        continue;
      }

      if (top_label == 0) {
        labels_ptr[idx] = left_label;
        continue;
      }

      const int min_label = std::min(left_label, top_label);
      labels_ptr[idx] = min_label;
      if (left_label != top_label) {
        UnionLabels(parent, left_label, top_label);
      }
    }
  }

  tile_used_counts[static_cast<std::size_t>(tile_index)] = next_label - tile.base_label;
}

}  // namespace

MarinLMarkComponentsTBB::MarinLMarkComponentsTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsTBB::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (int pixel : row) {
      if (pixel != 0 && pixel != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MarinLMarkComponentsTBB::ValidationImpl() {
  const auto &img = GetInput().binary;
  if (img.empty() || img.front().empty()) {
    return false;
  }

  const std::size_t width = img.front().size();
  for (const auto &row : img) {
    if (row.size() != width) {
      return false;
    }
  }

  return IsBinary(img);
}

bool MarinLMarkComponentsTBB::PreProcessingImpl() {
  const auto &input_binary = GetInput().binary;
  height_ = static_cast<int>(input_binary.size());
  width_ = static_cast<int>(input_binary.front().size());

  if (height_ <= 0 || width_ <= 0) {
    return false;
  }

  const std::uint64_t pixels = static_cast<std::uint64_t>(height_) * static_cast<std::uint64_t>(width_);
  if (pixels > kMaxPixels) {
    return false;
  }

  binary_.assign(static_cast<std::size_t>(pixels), 0);
  labels_flat_.assign(static_cast<std::size_t>(pixels), 0);
  parent_.assign(static_cast<std::size_t>(pixels) + 1ULL, 0);
  root_to_compact_.assign(static_cast<std::size_t>(pixels) + 1ULL, 0);
  root_generation_.assign(static_cast<std::size_t>(pixels) + 1ULL, 0);
  generation_id_ = 1;
  max_label_id_ = 0;

  tile_rows_ = (height_ + kTileHeight - 1) / kTileHeight;
  tile_cols_ = (width_ + kTileWidth - 1) / kTileWidth;
  tile_count_ = tile_rows_ * tile_cols_;
  tile_offsets_.assign(static_cast<std::size_t>(tile_count_) + 1ULL, 0);
  tile_used_counts_.assign(static_cast<std::size_t>(tile_count_), 0);

  for (int tile_row = 0; tile_row < tile_rows_; ++tile_row) {
    const int row_begin = tile_row * kTileHeight;
    const int row_end = std::min(row_begin + kTileHeight, height_);
    for (int tile_col = 0; tile_col < tile_cols_; ++tile_col) {
      const int col_begin = tile_col * kTileWidth;
      const int col_end = std::min(col_begin + kTileWidth, width_);
      const int tile_index = GetTileIndex(tile_row, tile_col, tile_cols_);
      tile_offsets_[static_cast<std::size_t>(tile_index) + 1ULL] = (row_end - row_begin) * (col_end - col_begin);
    }
  }
  std::partial_sum(tile_offsets_.begin(), tile_offsets_.end(), tile_offsets_.begin());
  max_label_id_ = tile_offsets_[static_cast<std::size_t>(tile_count_)];

  std::uint8_t *binary_ptr = binary_.data();
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, height_, 32),
                            [&](const oneapi::tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row < range.end(); ++row) {
      const auto &src_row = input_binary[static_cast<std::size_t>(row)];
      const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
      for (int col = 0; col < width_; ++col) {
        binary_ptr[row_offset + static_cast<std::size_t>(col)] = static_cast<std::uint8_t>(src_row[col]);
      }
    }
  });

  return true;
}

bool MarinLMarkComponentsTBB::RunImpl() {
  FirstPassTiles();
  MergeTileBorders();
  SecondPassTiles();
  return true;
}

void MarinLMarkComponentsTBB::FirstPassTiles() {
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, tile_count_, 1),
                            [&](const oneapi::tbb::blocked_range<int> &range) {
    for (int tile_index = range.begin(); tile_index < range.end(); ++tile_index) {
      const TileBounds tile = GetTileBounds(tile_index, tile_cols_, height_, width_, tile_offsets_);
      ProcessTile(binary_, labels_flat_, parent_, tile_used_counts_, width_, tile, tile_index);
    }
  });
}

void MarinLMarkComponentsTBB::MergeTileBorders() {
  for (int tile_row = 0; tile_row < tile_rows_; ++tile_row) {
    for (int tile_col = 0; tile_col + 1 < tile_cols_; ++tile_col) {
      const int left_tile_index = GetTileIndex(tile_row, tile_col, tile_cols_);
      const TileBounds left_tile = GetTileBounds(left_tile_index, tile_cols_, height_, width_, tile_offsets_);
      const int border_col = left_tile.col_end;

      for (int row = left_tile.row_begin; row < left_tile.row_end; ++row) {
        const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
        const std::size_t left_idx = row_offset + static_cast<std::size_t>(border_col - 1);
        const std::size_t right_idx = row_offset + static_cast<std::size_t>(border_col);
        if ((binary_[left_idx] != 0U) && (binary_[right_idx] != 0U)) {
          const int left_label = labels_flat_[left_idx];
          const int right_label = labels_flat_[right_idx];
          if (left_label > 0 && right_label > 0 && left_label != right_label) {
            UnionLabels(parent_, left_label, right_label);
          }
        }
      }
    }
  }

  for (int tile_row = 0; tile_row + 1 < tile_rows_; ++tile_row) {
    for (int tile_col = 0; tile_col < tile_cols_; ++tile_col) {
      const int top_tile_index = GetTileIndex(tile_row, tile_col, tile_cols_);
      const TileBounds top_tile = GetTileBounds(top_tile_index, tile_cols_, height_, width_, tile_offsets_);
      const int border_row = top_tile.row_end;

      for (int col = top_tile.col_begin; col < top_tile.col_end; ++col) {
        const std::size_t top_idx =
            static_cast<std::size_t>(border_row - 1) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(col);
        const std::size_t bottom_idx =
            static_cast<std::size_t>(border_row) * static_cast<std::size_t>(width_) + static_cast<std::size_t>(col);
        if ((binary_[top_idx] != 0U) && (binary_[bottom_idx] != 0U)) {
          const int top_label = labels_flat_[top_idx];
          const int bottom_label = labels_flat_[bottom_idx];
          if (top_label > 0 && bottom_label > 0 && top_label != bottom_label) {
            UnionLabels(parent_, top_label, bottom_label);
          }
        }
      }
    }
  }

  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, tile_count_, 1),
                            [&](const oneapi::tbb::blocked_range<int> &range) {
    for (int tile_index = range.begin(); tile_index < range.end(); ++tile_index) {
      const int base_label = 1 + tile_offsets_[static_cast<std::size_t>(tile_index)];
      const int used_count = tile_used_counts_[static_cast<std::size_t>(tile_index)];
      for (int label = base_label; label < base_label + used_count; ++label) {
        parent_[static_cast<std::size_t>(label)] = FindRoot(parent_, label);
      }
    }
  });
}

void MarinLMarkComponentsTBB::SecondPassTiles() {
  if (max_label_id_ == 0) {
    return;
  }

  ++generation_id_;
  if (generation_id_ == 0) {
    generation_id_ = 1;
    std::fill(root_generation_.begin(), root_generation_.end(), 0);
  }

  int next_id = 1;
  for (int tile_index = 0; tile_index < tile_count_; ++tile_index) {
    const int base_label = 1 + tile_offsets_[static_cast<std::size_t>(tile_index)];
    const int used_count = tile_used_counts_[static_cast<std::size_t>(tile_index)];
    for (int label = base_label; label < base_label + used_count; ++label) {
      const int root = parent_[static_cast<std::size_t>(label)];
      if (root_generation_[static_cast<std::size_t>(root)] != generation_id_) {
        root_generation_[static_cast<std::size_t>(root)] = generation_id_;
        root_to_compact_[static_cast<std::size_t>(root)] = next_id++;
      }
    }
  }

  int *labels_ptr = labels_flat_.data();
  const int *parent_ptr = parent_.data();
  const int *compact_ptr = root_to_compact_.data();
  const int64_t pixels_count = static_cast<int64_t>(labels_flat_.size());

  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int64_t>(0, pixels_count, 1 << 20),
                            [&](const oneapi::tbb::blocked_range<int64_t> &range) {
    for (int64_t idx = range.begin(); idx < range.end(); ++idx) {
      const int label = labels_ptr[static_cast<std::size_t>(idx)];
      if (label == 0) {
        continue;
      }
      const int root = parent_ptr[static_cast<std::size_t>(label)];
      labels_ptr[static_cast<std::size_t>(idx)] = compact_ptr[static_cast<std::size_t>(root)];
    }
  });
}

bool MarinLMarkComponentsTBB::PostProcessingImpl() {
  OutType out;
  out.labels = Labels(static_cast<std::size_t>(height_), std::vector<int>(static_cast<std::size_t>(width_), 0));

  const int *labels_ptr = labels_flat_.data();
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, height_, 32),
                            [&](const oneapi::tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row < range.end(); ++row) {
      const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
      auto &out_row = out.labels[static_cast<std::size_t>(row)];
      for (int col = 0; col < width_; ++col) {
        out_row[static_cast<std::size_t>(col)] = labels_ptr[row_offset + static_cast<std::size_t>(col)];
      }
    }
  });

  GetOutput() = out;
  return true;
}

}  // namespace marin_l_mark_components
