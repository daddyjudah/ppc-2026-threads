#include "marin_l_mark_components/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;

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

void ProcessPixel(const Image &binary, Labels &labels, std::vector<int> &parent, int row, int col, int &next_label) {
  if (binary[row][col] == 0) {
    return;
  }

  int left = (col > 0) ? labels[row][col - 1] : 0;
  int top = (row > 0) ? labels[row - 1][col] : 0;

  if (left == 0 && top == 0) {
    labels[row][col] = next_label++;
    return;
  }

  if (left != 0 && top == 0) {
    labels[row][col] = left;
    return;
  }

  if (left == 0 && top != 0) {
    labels[row][col] = top;
    return;
  }

  if (left == top) {
    labels[row][col] = left;
  } else {
    int min_label = std::min(left, top);
    int max_label = std::max(left, top);
    labels[row][col] = min_label;
    UnionLabels(parent, min_label, max_label);
  }
}

}  // namespace

MarinLMarkComponentsOMP::MarinLMarkComponentsOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsOMP::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (int pixel : row) {
      if (pixel != 0 && pixel != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MarinLMarkComponentsOMP::ValidationImpl() {
  const auto &img = GetInput().binary;

  if (img.empty() || img.front().empty()) {
    return false;
  }

  size_t width = img.front().size();
  for (const auto &row : img) {
    if (row.size() != width) {
      return false;
    }
  }

  return IsBinary(img);
}

bool MarinLMarkComponentsOMP::PreProcessingImpl() {
  binary_ = GetInput().binary;

  int height = static_cast<int>(binary_.size());
  int width = static_cast<int>(binary_.front().size());

  if (height <= 0 || width <= 0) {
    return false;
  }

  uint64_t pixels = static_cast<uint64_t>(height) * width;
  if (pixels > kMaxPixels) {
    return false;
  }

  labels_.assign(height, std::vector<int>(width, 0));
  return true;
}

void MarinLMarkComponentsOMP::FirstPass() {
  int height = static_cast<int>(binary_.size());
  int width = static_cast<int>(binary_.front().size());

  int max_labels = height * width;

  std::vector<int> parent(max_labels + 1);
  for (int i = 0; i <= max_labels; ++i) {
    parent[i] = i;
  }

  int next_label = 1;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      ProcessPixel(binary_, labels_, parent, row, col, next_label);
    }
  }

  for (int i = 1; i < next_label; ++i) {
    parent[i] = FindRoot(parent, i);
  }

#pragma omp parallel for collapse(2)
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int label = labels_[row][col];
      if (label != 0) {
        labels_[row][col] = parent[label];
      }
    }
  }
}

void MarinLMarkComponentsOMP::SecondPass() {
  int height = static_cast<int>(labels_.size());
  int width = (height > 0) ? static_cast<int>(labels_[0].size()) : 0;

  if (height == 0 || width == 0) {
    return;
  }

  int max_label = 0;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      max_label = std::max(max_label, labels_[row][col]);
    }
  }

  if (max_label == 0) {
    return;
  }

  std::vector<bool> seen(max_label + 1, false);
  std::vector<int> unique_labels;

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int label = labels_[row][col];
      if (label != 0 && !seen[label]) {
        seen[label] = true;
        unique_labels.push_back(label);
      }
    }
  }

  std::sort(unique_labels.begin(), unique_labels.end());

  std::vector<int> compact_map(max_label + 1, 0);
  for (size_t i = 0; i < unique_labels.size(); ++i) {
    compact_map[unique_labels[i]] = static_cast<int>(i + 1);
  }

#pragma omp parallel for collapse(2)
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int label = labels_[row][col];
      if (label != 0) {
        labels_[row][col] = compact_map[label];
      }
    }
  }
}

bool MarinLMarkComponentsOMP::RunImpl() {
  FirstPass();
  SecondPass();
  return true;
}

bool MarinLMarkComponentsOMP::PostProcessingImpl() {
  GetOutput().labels = labels_;
  return true;
}

}  // namespace marin_l_mark_components
