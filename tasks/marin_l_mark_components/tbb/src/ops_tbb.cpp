#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <numeric>

namespace marin_l_mark_components {

namespace {
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
  size_t total_pixels = static_cast<size_t>(height_) * width_;

  binary_flat_.assign(total_pixels, 0);
  labels_flat_.assign(total_pixels, 0);
  parent_.resize(total_pixels + 1);
  std::iota(parent_.begin(), parent_.end(), 0);

  tbb::parallel_for(0, height_, [&](int r) {
    for (int c = 0; c < width_; ++c) {
      binary_flat_[r * width_ + c] = static_cast<uint8_t>(input[r][c]);
    }
  });

  return true;
}

void MarinLMarkComponentsTBB::FirstPassTBB() {
  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int> &range) {
    for (int r = range.begin(); r < range.end(); ++r) {
      for (int c = 0; c < width_; ++c) {
        size_t idx = static_cast<size_t>(r) * width_ + c;
        if (binary_flat_[idx] == 0) {
          continue;
        }

        int top = (r > range.begin()) ? labels_flat_[idx - width_] : 0;
        int left = (c > 0) ? labels_flat_[idx - 1] : 0;

        if (top == 0 && left == 0) {
          labels_flat_[idx] = static_cast<int>(idx + 1);
        } else if (top != 0 && left == 0) {
          labels_flat_[idx] = top;
        } else if (top == 0 && left != 0) {
          labels_flat_[idx] = left;
        } else {
          labels_flat_[idx] = std::min(top, left);
          if (top != left) {
            UnionLabels(parent_, top, left);
          }
        }
      }
    }
  });
}

void MarinLMarkComponentsTBB::MergeBordersTBB() {
  for (int r = 1; r < height_; ++r) {
    for (int c = 0; c < width_; ++c) {
      size_t curr = static_cast<size_t>(r) * width_ + c;
      size_t prev = curr - width_;
      if (binary_flat_[curr] && binary_flat_[prev]) {
        UnionLabels(parent_, labels_flat_[curr], labels_flat_[prev]);
      }
    }
  }
}

void MarinLMarkComponentsTBB::SecondPassTBB() {
  tbb::parallel_for(0, height_, [&](int r) {
    for (int c = 0; c < width_; ++c) {
      size_t idx = static_cast<size_t>(r) * width_ + c;
      if (labels_flat_[idx] != 0) {
        labels_flat_[idx] = FindRoot(parent_, labels_flat_[idx]);
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
  Labels output(height_, std::vector<int>(width_));
  for (int r = 0; r < height_; ++r) {
    for (int c = 0; c < width_; ++c) {
      output[r][c] = labels_flat_[r * width_ + c];
    }
  }
  GetOutput().labels = output;
  return true;
}

}  // namespace marin_l_mark_components
