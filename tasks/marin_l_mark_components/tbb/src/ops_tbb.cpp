#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <numeric>

namespace marin_l_mark_components {

namespace {
int FindRoot(int *parent, int x) {
  int root = x;
  while (parent[root] != root) {
    root = parent[root];
  }
  return root;
}

void UnionLabels(int *parent, int a, int b) {
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

  tbb::parallel_for(tbb::blocked_range<int>(0, height_, 64), [&](const tbb::blocked_range<int> &range) {
    for (int r = range.begin(); r < range.end(); ++r) {
      int row_off = r * width_;
      for (int c = 0; c < width_; ++c) {
        int idx = row_off + c;
        if (b_ptr[idx] == 0) {
          continue;
        }

        int top = (r > range.begin() && b_ptr[idx - width_]) ? l_ptr[idx - width_] : 0;
        int left = (c > 0 && b_ptr[idx - 1]) ? l_ptr[idx - 1] : 0;

        if (top == 0 && left == 0) {
          l_ptr[idx] = idx + 1;
        } else if (top != 0 && left == 0) {
          l_ptr[idx] = top;
        } else if (top == 0 && left != 0) {
          l_ptr[idx] = left;
        } else {
          l_ptr[idx] = (top < left) ? top : left;
          if (top != left) {
            UnionLabels(p_ptr, top, left);
          }
        }
      }
    }
  });
}

void MarinLMarkComponentsTBB::MergeBordersTBB() {
  int *p_ptr = parent_.data();
  int *l_ptr = labels_flat_.data();
  const uint8_t *b_ptr = binary_flat_.data();

  tbb::parallel_for(1, height_, [&](int r) {
    int curr_row = r * width_;
    int prev_row = (r - 1) * width_;
    for (int c = 0; c < width_; ++c) {
      if (b_ptr[curr_row + c] && b_ptr[prev_row + c]) {
        UnionLabels(p_ptr, l_ptr[curr_row + c], l_ptr[prev_row + c]);
      }
    }
  });
}

void MarinLMarkComponentsTBB::SecondPassTBB() {
  int *p_ptr = parent_.data();
  int *l_ptr = labels_flat_.data();

  tbb::parallel_for(0, height_, [&](int r) {
    int off = r * width_;
    for (int c = 0; c < width_; ++c) {
      int idx = off + c;
      if (l_ptr[idx] != 0) {
        l_ptr[idx] = FindRoot(p_ptr, l_ptr[idx]);
      }
    }
  });
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
