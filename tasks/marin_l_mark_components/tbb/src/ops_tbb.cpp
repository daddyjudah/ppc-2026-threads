#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;
constexpr int kMinRowsPerStripe = 64;

struct StripeRange {
  int row_start;
  int row_end;
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

StripeRange GetStripeRange(int stripe, int height, int stripe_count, const std::vector<int> &stripe_offsets) {
  return {
      .row_start = (stripe * height) / stripe_count,
      .row_end = ((stripe + 1) * height) / stripe_count,
      .base_label = 1 + stripe_offsets[static_cast<std::size_t>(stripe)],
  };
}

void ProcessStripe(const std::vector<std::uint8_t> &binary, std::vector<int> &labels_flat, std::vector<int> &parent,
                   std::vector<int> &stripe_used_counts, int width, const StripeRange &stripe_range, int stripe) {
  const std::uint8_t *binary_ptr = binary.data();
  int *labels_ptr = labels_flat.data();
  int *parent_ptr = parent.data();
  int next_label = stripe_range.base_label;

  for (int row = stripe_range.row_start; row < stripe_range.row_end; ++row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
    const bool has_top = row > stripe_range.row_start;
    const std::size_t top_row_offset = has_top ? row_offset - static_cast<std::size_t>(width) : 0;

    for (int col = 0; col < width; ++col) {
      const std::size_t idx = row_offset + static_cast<std::size_t>(col);
      if (binary_ptr[idx] == 0U) {
        labels_ptr[idx] = 0;
        continue;
      }

      const int left_label = (col > 0) ? labels_ptr[idx - 1ULL] : 0;
      const int top_label = has_top ? labels_ptr[top_row_offset + static_cast<std::size_t>(col)] : 0;

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

  stripe_used_counts[static_cast<std::size_t>(stripe)] = next_label - stripe_range.base_label;
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
  max_label_id_ = 0;
  generation_id_ = 1;

  stripe_count_ = std::max(1, oneapi::tbb::this_task_arena::max_concurrency());
  stripe_count_ = std::min(stripe_count_, height_);
  stripe_count_ = std::min(stripe_count_, std::max(1, height_ / kMinRowsPerStripe));
  stripe_offsets_.assign(static_cast<std::size_t>(stripe_count_) + 1ULL, 0);
  stripe_used_counts_.assign(static_cast<std::size_t>(stripe_count_), 0);

  for (int stripe = 0; stripe < stripe_count_; ++stripe) {
    const int row_start = (stripe * height_) / stripe_count_;
    const int row_end = ((stripe + 1) * height_) / stripe_count_;
    stripe_offsets_[static_cast<std::size_t>(stripe) + 1ULL] = (row_end - row_start) * width_;
  }
  std::partial_sum(stripe_offsets_.begin(), stripe_offsets_.end(), stripe_offsets_.begin());
  max_label_id_ = stripe_offsets_[static_cast<std::size_t>(stripe_count_)];

  std::uint8_t *binary_ptr = binary_.data();
  oneapi::tbb::parallel_for(0, height_, [&](int row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
    const auto &src_row = input_binary[static_cast<std::size_t>(row)];
    for (int col = 0; col < width_; ++col) {
      binary_ptr[row_offset + static_cast<std::size_t>(col)] = static_cast<std::uint8_t>(src_row[col]);
    }
  });

  return true;
}

bool MarinLMarkComponentsTBB::RunImpl() {
  FirstPassTBB();
  MergeStripeBorders();
  SecondPassTBB();
  return true;
}

void MarinLMarkComponentsTBB::FirstPassTBB() {
  oneapi::tbb::parallel_for(0, stripe_count_, [&](int stripe) {
    const StripeRange stripe_range = GetStripeRange(stripe, height_, stripe_count_, stripe_offsets_);
    ProcessStripe(binary_, labels_flat_, parent_, stripe_used_counts_, width_, stripe_range, stripe);
  });
}

void MarinLMarkComponentsTBB::MergeStripeBorders() {
  for (int stripe = 0; stripe < stripe_count_ - 1; ++stripe) {
    const int border_row = ((stripe + 1) * height_) / stripe_count_;
    const std::size_t top_row_offset = static_cast<std::size_t>(border_row - 1) * static_cast<std::size_t>(width_);
    const std::size_t bottom_row_offset = static_cast<std::size_t>(border_row) * static_cast<std::size_t>(width_);

    for (int col = 0; col < width_; ++col) {
      const std::size_t top_idx = top_row_offset + static_cast<std::size_t>(col);
      const std::size_t bottom_idx = bottom_row_offset + static_cast<std::size_t>(col);
      if ((binary_[top_idx] != 0U) && (binary_[bottom_idx] != 0U)) {
        const int top_label = labels_flat_[top_idx];
        const int bottom_label = labels_flat_[bottom_idx];
        if (top_label > 0 && bottom_label > 0 && top_label != bottom_label) {
          UnionLabels(parent_, top_label, bottom_label);
        }
      }
    }
  }

  oneapi::tbb::parallel_for(0, stripe_count_, [&](int stripe) {
    const int base_label = 1 + stripe_offsets_[static_cast<std::size_t>(stripe)];
    const int used_count = stripe_used_counts_[static_cast<std::size_t>(stripe)];
    for (int label = base_label; label < base_label + used_count; ++label) {
      parent_[static_cast<std::size_t>(label)] = FindRoot(parent_, label);
    }
  });
}

void MarinLMarkComponentsTBB::SecondPassTBB() {
  if (height_ == 0 || width_ == 0 || max_label_id_ == 0) {
    return;
  }
  ++generation_id_;
  if (generation_id_ == 0) {
    generation_id_ = 1;
    std::fill(root_generation_.begin(), root_generation_.end(), 0);
  }

  int next_id = 1;
  for (int stripe = 0; stripe < stripe_count_; ++stripe) {
    const int base_label = 1 + stripe_offsets_[static_cast<std::size_t>(stripe)];
    const int used_count = stripe_used_counts_[static_cast<std::size_t>(stripe)];
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

  oneapi::tbb::parallel_for(int64_t{0}, pixels_count, [&](int64_t idx) {
    const int label = labels_ptr[static_cast<std::size_t>(idx)];
    if (label == 0) {
      return;
    }

    const int root = parent_ptr[static_cast<std::size_t>(label)];
    labels_ptr[static_cast<std::size_t>(idx)] = compact_ptr[static_cast<std::size_t>(root)];
  });
}

bool MarinLMarkComponentsTBB::PostProcessingImpl() {
  OutType out;
  out.labels = Labels(static_cast<std::size_t>(height_), std::vector<int>(static_cast<std::size_t>(width_), 0));

  const int *labels_ptr = labels_flat_.data();
  oneapi::tbb::parallel_for(0, height_, [&](int row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
    auto &out_row = out.labels[static_cast<std::size_t>(row)];
    for (int col = 0; col < width_; ++col) {
      out_row[static_cast<std::size_t>(col)] = labels_ptr[row_offset + static_cast<std::size_t>(col)];
    }
  });

  GetOutput() = out;
  return true;
}

}  // namespace marin_l_mark_components
