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

  uint64_t pixels = static_cast<uint64_t>(height) * static_cast<uint64_t>(width);
  if (pixels > kMaxPixels) {
    return false;
  }

  labels_.assign(height, std::vector<int>(width, 0));
  return true;
}

void MarinLMarkComponentsOMP::FirstPass() {
  int height = static_cast<int>(binary_.size());
  int width = static_cast<int>(binary_.front().size());
  int max_labels = height * width + 1;

  std::vector<int> parent(max_labels);
  for (int i = 0; i < max_labels; ++i) {
    parent[i] = i;
  }

  int num_threads = omp_get_max_threads();
  std::vector<int> thread_offsets(num_threads + 1, 0);
  std::vector<int> thread_start_row(num_threads);
  std::vector<int> thread_end_row(num_threads);

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int rows_per_thread = height / num_threads;
    thread_start_row[thread_id] = thread_id * rows_per_thread;
    thread_end_row[thread_id] = (thread_id == num_threads - 1) ? height : thread_start_row[thread_id] + rows_per_thread;
  }

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int start_row = thread_start_row[thread_id];
    int end_row = thread_end_row[thread_id];

    int next_label = 1;

    for (int row = start_row; row < end_row; ++row) {
      for (int col = 0; col < width; ++col) {
        ProcessPixel(binary_, labels_, parent, row, col, next_label);
      }
    }

    thread_offsets[thread_id + 1] = next_label - 1;
  }

  for (int i = 1; i <= num_threads; ++i) {
    thread_offsets[i] += thread_offsets[i - 1];
  }
#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int start_row = thread_start_row[thread_id];
    int end_row = thread_end_row[thread_id];
    int offset = thread_offsets[thread_id];

    for (int row = start_row; row < end_row; ++row) {
      for (int col = 0; col < width; ++col) {
        if (labels_[row][col] != 0) {
          labels_[row][col] += offset;
        }
      }
    }
  }

  int total_labels = thread_offsets[num_threads];

  for (int t = 1; t < num_threads; ++t) {
    int boundary_row = thread_start_row[t];
    if (boundary_row >= height || boundary_row <= 0) {
      continue;
    }

    for (int col = 0; col < width; ++col) {
      if (binary_[boundary_row][col] == 1 && binary_[boundary_row - 1][col] == 1) {
        int top_label = labels_[boundary_row - 1][col];
        int bottom_label = labels_[boundary_row][col];

        if (top_label != 0 && bottom_label != 0 && top_label != bottom_label) {
          UnionLabels(parent, top_label, bottom_label);
        }
      }
    }
  }

  for (int i = 1; i <= total_labels; ++i) {
    if (parent[i] != i) {
      parent[i] = FindRoot(parent, i);
    }
  }

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

  std::vector<int> unique_labels;

  int max_label = 0;
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int label = labels_[row][col];
      if (label > max_label) {
        max_label = label;
      }
    }
  }

  if (max_label == 0) {
    return;
  }

  std::vector<bool> seen(max_label + 1, false);

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      int label = labels_[row][col];
      if (label != 0 && !seen[label]) {
        seen[label] = true;
        unique_labels.push_back(label);
      }
    }
  }

  if (unique_labels.empty()) {
    return;
  }

  std::sort(unique_labels.begin(), unique_labels.end());

  std::vector<int> compact_map(max_label + 1, 0);
  for (size_t i = 0; i < unique_labels.size(); ++i) {
    compact_map[unique_labels[i]] = static_cast<int>(i + 1);
  }

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
