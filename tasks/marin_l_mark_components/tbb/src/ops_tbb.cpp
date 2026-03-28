#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <numeric>

namespace marin_l_mark_components {

namespace {
inline int FindRoot(int *parent, int x) {
  int root = x;
  while (parent[root] != root) {
    root = parent[root];
  }
  return root;
}

inline void UnionLabels(int *parent, int a, int b) {
  int root_a = FindRoot(parent, a);
  int root_b = FindRoot(parent, b);
  while (root_a != root_b) {
    if (root_a < root_b) {
      parent[root_b] = root_a;
      root_b = FindRoot(parent, root_b);
    } else {
      parent[root_a] = root_b;
      root_a = FindRoot(parent, root_a);
    }
  }
}
}  // namespace

MarinLMarkComponentsTBB::MarinLMarkComponentsTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsTBB::ValidationImpl() {
  const auto &img = GetInput().binary;
  if (img.empty() || img[0].empty()) {
    return false;
  }
  return true;
}

bool MarinLMarkComponentsTBB::PreProcessingImpl() {
  const auto &input = GetInput().binary;
  height_ = static_cast<int>(input.size());
  width_ = static_cast<int>(input[0].size());
  size_t total = static_cast<size_t>(height_) * width_;

  binary_flat_.assign(total, 0);
  labels_flat_.assign(total, 0);
  parent_.resize(total + 1);
  std::iota(parent_.begin(), parent_.end(), 0);

  uint8_t *b_ptr = binary_flat_.data();
  tbb::parallel_for(0, height_, [&](int r) {
    const auto &row = input[r];
    for (int c = 0; c < width_; ++c) {
      b_ptr[r * width_ + c] = static_cast<uint8_t>(row[c]);
    }
  });
  return true;
}

void MarinLMarkComponentsTBB::FirstPassTBB() {
  int *p_ptr = parent_.data();
  int *l_ptr = labels_flat_.data();
  const uint8_t *b_ptr = binary_flat_.data();
  int w = width_;

  tbb::parallel_for(tbb::blocked_range<int>(0, height_, 64), [&](const tbb::blocked_range<int> &range) {
    for (int r = range.begin(); r < range.end(); ++r) {
      const int row_off = r * w;
      const uint8_t *row_bin = b_ptr + row_off;
      int *row_labels = l_ptr + row_off;

      for (int c = 0; c < w; ++c) {
        if (row_bin[c] == 0) {
          continue;
        }

        int top = (r > range.begin() && row_bin[c - w]) ? row_labels[c - w] : 0;
        int left = (c > 0 && row_bin[c - 1]) ? row_labels[c - 1] : 0;

        if (top == 0) {
          if (left == 0) {
            row_labels[c] = row_off + c + 1;
          } else {
            row_labels[c] = left;
          }
        } else if (left == 0) {
          row_labels[c] = top;
        } else {
          row_labels[c] = (top < left) ? top : left;
          if (top != left) {
            UnionLabels(p_ptr, top, left);
          }
        }
      }
    }
  }, tbb::static_partitioner());
}

void MarinLMarkComponentsTBB::MergeBordersTBB() {
  int *p_ptr = parent_.data();
  int *l_ptr = labels_flat_.data();
  const uint8_t *b_ptr = binary_flat_.data();
  int w = width_;

  tbb::parallel_for(1, height_, [&](int r) {
    const int curr_row = r * w;
    const int prev_row = curr_row - w;
    for (int c = 0; c < w; ++c) {
      if (b_ptr[curr_row + c] && b_ptr[prev_row + c]) {
        UnionLabels(p_ptr, l_ptr[curr_row + c], l_ptr[prev_row + c]);
      }
    }
  }, tbb::static_partitioner());
}

void MarinLMarkComponentsTBB::SecondPassTBB() {
  int *p_ptr = parent_.data();
  int *l_ptr = labels_flat_.data();
  int total = height_ * width_;

  tbb::parallel_for(tbb::blocked_range<int>(0, total, 20000), [&](const tbb::blocked_range<int> &range) {
    for (int i = range.begin(); i < range.end(); ++i) {
      if (l_ptr[i] != 0) {
        l_ptr[i] = FindRoot(p_ptr, l_ptr[i]);
      }
    }
  }, tbb::static_partitioner());
}

bool MarinLMarkComponentsTBB::RunImpl() {
  FirstPassTBB();
  MergeBordersTBB();
  SecondPassTBB();
  return true;
}

bool MarinLMarkComponentsTBB::PostProcessingImpl() {
  auto &output = GetOutput().labels;
  output.resize(height_);
  const int *l_ptr = labels_flat_.data();

  tbb::parallel_for(0, height_, [&](int r) { output[r].assign(l_ptr + r * width_, l_ptr + (r + 1) * width_); });
  return true;
}

}  // namespace marin_l_mark_components
