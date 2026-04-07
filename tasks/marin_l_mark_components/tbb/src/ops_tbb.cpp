#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "marin_l_mark_components/common/include/common.hpp"

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;

int FindRoot(std::vector<int> &parent, int x) {
  while (parent[static_cast<std::size_t>(x)] != x) {
    parent[static_cast<std::size_t>(x)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(x)])];
    x = parent[static_cast<std::size_t>(x)];
  }
  return x;
}

void UnionLabels(std::vector<int> &parent, int a, int b) {
  const int root_a = FindRoot(parent, a);
  const int root_b = FindRoot(parent, b);

  if (root_a == root_b) {
    return;
  }

  if (root_a < root_b) {
    parent[static_cast<std::size_t>(root_b)] = root_a;
  } else {
    parent[static_cast<std::size_t>(root_a)] = root_b;
  }
}

int MergeLabels(std::vector<int> &parent, int left_label, int top_label, int &next_label) {
  if (left_label == 0 && top_label == 0) {
    parent[static_cast<std::size_t>(next_label)] = next_label;
    return next_label++;
  }

  if (left_label == 0) {
    return top_label;
  }

  if (top_label == 0) {
    return left_label;
  }

  const int merged_label = std::min(left_label, top_label);
  if (left_label != top_label) {
    UnionLabels(parent, left_label, top_label);
  }

  return merged_label;
}

void ProcessPixel(const std::vector<std::uint8_t> &binary, std::vector<int> &labels_flat, std::vector<int> &parent,
                  int width, int row, int col, int &next_label) {
  const std::size_t idx =
      (static_cast<std::size_t>(row) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(col);
  if (binary[idx] == 0U) {
    return;
  }

  const int left_label = (col > 0) ? labels_flat[idx - 1ULL] : 0;
  const int top_label = (row > 0) ? labels_flat[idx - static_cast<std::size_t>(width)] : 0;
  labels_flat[idx] = MergeLabels(parent, left_label, top_label, next_label);
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
  labels_.clear();
  parent_.assign(static_cast<std::size_t>(pixels) + 1ULL, 0);
  root_to_compact_.assign(static_cast<std::size_t>(pixels) + 1ULL, 0);
  next_label_ = 1;

  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row != range.end(); ++row) {
      const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
      for (int col = 0; col < width_; ++col) {
        binary_[row_offset + static_cast<std::size_t>(col)] =
            static_cast<std::uint8_t>(input_binary[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
      }
    }
  });

  return true;
}

bool MarinLMarkComponentsTBB::RunImpl() {
  FirstPassTBB();
  SecondPassTBB();
  return true;
}

void MarinLMarkComponentsTBB::FirstPassTBB() {
  for (int row = 0; row < height_; ++row) {
    for (int col = 0; col < width_; ++col) {
      ProcessPixel(binary_, labels_flat_, parent_, width_, row, col, next_label_);
    }
  }

  for (int label = 1; label < next_label_; ++label) {
    parent_[static_cast<std::size_t>(label)] = FindRoot(parent_, label);
  }
}

void MarinLMarkComponentsTBB::SecondPassTBB() {
  if (next_label_ <= 1) {
    return;
  }

  int compact_label = 1;
  for (int label = 1; label < next_label_; ++label) {
    const int root = parent_[static_cast<std::size_t>(label)];
    if (root_to_compact_[static_cast<std::size_t>(root)] == 0) {
      root_to_compact_[static_cast<std::size_t>(root)] = compact_label++;
    }
  }

  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, labels_flat_.size()),
                    [&](const tbb::blocked_range<std::size_t> &range) {
    for (std::size_t idx = range.begin(); idx != range.end(); ++idx) {
      const int label = labels_flat_[idx];
      if (label == 0) {
        continue;
      }

      const int root = parent_[static_cast<std::size_t>(label)];
      labels_flat_[idx] = root_to_compact_[static_cast<std::size_t>(root)];
    }
  });
}

bool MarinLMarkComponentsTBB::PostProcessingImpl() {
  ConvertLabelsToOutput();
  OutType out;
  out.labels = labels_;
  GetOutput() = out;
  return true;
}

void MarinLMarkComponentsTBB::ConvertLabelsToOutput() {
  labels_.clear();
  labels_.resize(static_cast<std::size_t>(height_));

  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row != range.end(); ++row) {
      labels_[static_cast<std::size_t>(row)].resize(static_cast<std::size_t>(width_));
      const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);

      for (int col = 0; col < width_; ++col) {
        labels_[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
            labels_flat_[row_offset + static_cast<std::size_t>(col)];
      }
    }
  });
}

}  // namespace marin_l_mark_components
