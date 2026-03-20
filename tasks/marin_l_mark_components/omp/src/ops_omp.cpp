#include "marin_l_mark_components/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "marin_l_mark_components/common/include/common.hpp"

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
  const int root_a = FindRoot(parent, a);
  const int root_b = FindRoot(parent, b);

  if (root_a == root_b) {
    return;
  }

  if (root_a < root_b) {
    parent[root_b] = root_a;
  } else {
    parent[root_a] = root_b;
  }
}

inline int ResolveLabel(int left, int top) {
  if (left == 0) {
    return top;
  }
  if (top == 0) {
    return left;
  }
  return std::min(left, top);
}

inline void ProcessPixel(std::vector<int> &parent, std::vector<std::vector<int>> &labels,
                         const std::vector<std::vector<int>> &binary, int row, int col, int start_row, int &label) {
  auto &current_row = labels[row];

  if (binary[row][col] == 0) {
    current_row[col] = 0;
    return;
  }

  const int left = (col > 0) ? current_row[col - 1] : 0;
  const int top = (row > start_row) ? labels[row - 1][col] : 0;

  if (left == 0 && top == 0) {
    current_row[col] = label++;
    return;
  }

  const int min_label = ResolveLabel(left, top);
  current_row[col] = min_label;

  if (left != 0 && top != 0 && left != top) {
#pragma omp critical
    UnionLabels(parent, left, top);
  }
}

int FindMaxLabel(const Labels &labels, int height, int width) {
  int max_label = 0;

#pragma omp parallel for reduction(max : max_label) collapse(2) default(none) shared(labels, height, width)
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      max_label = std::max(max_label, labels[row][col]);
    }
  }

  return max_label;
}

void FillUsed(const Labels &labels, std::vector<char> &used, int height, int width) {
#pragma omp parallel for collapse(2) default(none) shared(labels, used, height, width)
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int val = labels[row][col];
      if (val != 0) {
        used[val] = 1;
      }
    }
  }
}

void ApplyMap(Labels &labels, const std::vector<int> &map, int height, int width) {
#pragma omp parallel for collapse(2) default(none) shared(labels, map, height, width)
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int val = labels[row][col];
      if (val != 0) {
        labels[row][col] = map[val];
      }
    }
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

  const std::size_t width = img.front().size();
  for (const auto &row : img) {
    if (row.size() != width) {
      return false;
    }
  }

  return IsBinary(img);
}

bool MarinLMarkComponentsOMP::PreProcessingImpl() {
  binary_ = GetInput().binary;

  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_.front().size());

  if (height <= 0 || width <= 0) {
    return false;
  }

  const std::uint64_t pixels = static_cast<std::uint64_t>(height) * static_cast<std::uint64_t>(width);

  if (pixels > kMaxPixels) {
    return false;
  }

  labels_.assign(height, std::vector<int>(width, 0));
  return true;
}

void MarinLMarkComponentsOMP::ProcessChunk(std::vector<int> &parent, int start_row, int end_row, int width) {
  int label = (start_row * width) + 1;

  for (int row = start_row; row < end_row; ++row) {
    for (int col = 0; col < width; ++col) {
      ProcessPixel(parent, labels_, binary_, row, col, start_row, label);
    }
  }
}

void MarinLMarkComponentsOMP::MergeBorders(std::vector<int> &parent, int height, int width, int chunk,
                                           int num_threads) {
  for (int thread_idx = 1; thread_idx < num_threads; ++thread_idx) {
    const int row = thread_idx * chunk;
    if (row >= height) {
      continue;
    }

    for (int col = 0; col < width; ++col) {
      if (binary_[row][col] == 0 || binary_[row - 1][col] == 0) {
        continue;
      }

      const int a = labels_[row][col];
      const int b = labels_[row - 1][col];

      if (a != b) {
        UnionLabels(parent, a, b);
      }
    }
  }
}

void MarinLMarkComponentsOMP::FirstPass() {
  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_[0].size());

  const int num_threads = omp_get_max_threads();
  const int chunk = (height + num_threads - 1) / num_threads;

  std::vector<int> parent((height * width) + 1);

#pragma omp parallel for default(none) shared(parent)
  for (std::size_t i = 0; i < parent.size(); ++i) {
    parent[i] = static_cast<int>(i);
  }

#pragma omp parallel default(none) shared(parent, height, width, chunk)
  {
    const int tid = omp_get_thread_num();
    const int start = tid * chunk;
    const int end = std::min(start + chunk, height);

    if (start < end) {
      ProcessChunk(parent, start, end, width);
    }
  }

  MergeBorders(parent, height, width, chunk, num_threads);

#pragma omp parallel for default(none) shared(parent)
  for (std::size_t i = 1; i < parent.size(); ++i) {
    parent[i] = FindRoot(parent, static_cast<int>(i));
  }

#pragma omp parallel for collapse(2) default(none) shared(height, width, parent)
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      if (labels_[row][col] != 0) {
        labels_[row][col] = parent[labels_[row][col]];
      }
    }
  }
}

void MarinLMarkComponentsOMP::SecondPass() {
  const int height = static_cast<int>(labels_.size());
  const int width = static_cast<int>(labels_[0].size());

  const int max_label = FindMaxLabel(labels_, height, width);

  if (max_label == 0) {
    return;
  }

  std::vector<char> used(max_label + 1, 0);
  FillUsed(labels_, used, height, width);

  std::vector<int> map(max_label + 1, 0);

  int next = 1;
  for (int i = 1; i <= max_label; ++i) {
    if (used[i] != 0) {
      map[i] = next++;
    }
  }

  ApplyMap(labels_, map, height, width);
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
