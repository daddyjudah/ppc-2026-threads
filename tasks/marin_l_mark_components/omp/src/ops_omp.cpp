#include "marin_l_mark_components/omp/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;

std::vector<int> &GetThreadLocalParent() {
  thread_local std::vector<int> parent_storage;
  return parent_storage;
}

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

void ProcessPixel(const std::vector<std::vector<int>> &binary, std::vector<std::vector<int>> &labels,
                  std::vector<int> &parent, int row, int col, int &next_label) {
  if (binary[row][col] == 0) {
    return;
  }

  int left_label = 0;
  int top_label = 0;

  if (col > 0) {
    left_label = labels[row][col - 1];
  }

  if (row > 0) {
    top_label = labels[row - 1][col];
  }

  if (left_label == 0 && top_label == 0) {
    labels[row][col] = next_label++;
    return;
  }

  if (left_label != 0 && top_label == 0) {
    labels[row][col] = left_label;
    return;
  }

  if (left_label == 0 && top_label != 0) {
    labels[row][col] = top_label;
    return;
  }

  const int min_label = std::min(left_label, top_label);
  labels[row][col] = min_label;

  if (left_label != top_label) {
    UnionLabels(parent, left_label, top_label);
  }
}

}  // namespace

MarinLMarkComponentsOMP::MarinLMarkComponentsOMP(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsOMP::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (const int pixel : row) {
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

  labels_.assign(static_cast<std::size_t>(height), std::vector<int>(static_cast<std::size_t>(width), 0));

  return true;
}

void MarinLMarkComponentsOMP::FirstPass() {
  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_.front().size());

  const int max_labels = height * width;
  const int num_threads_total = omp_get_max_threads();

  auto &global_parent = GetThreadLocalParent();
  global_parent.assign(static_cast<std::size_t>(max_labels) + 1ULL, 0);
  for (int i = 0; i <= max_labels; ++i) {
    global_parent[i] = i;
  }

  std::vector<int> thread_next_labels(num_threads_total, 1);
  std::vector<int> thread_start_row(num_threads_total);
  std::vector<int> thread_end_row(num_threads_total);

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int num_threads_curr = omp_get_num_threads();

    int rows_per_thread = height / num_threads_curr;
    thread_start_row[thread_id] = thread_id * rows_per_thread;
    thread_end_row[thread_id] =
        (thread_id == num_threads_curr - 1) ? height : thread_start_row[thread_id] + rows_per_thread;
  }

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int start_row = thread_start_row[thread_id];
    int end_row = thread_end_row[thread_id];

    int next_label = thread_next_labels[thread_id];

    for (int row = start_row; row < end_row; ++row) {
      for (int col = 0; col < width; ++col) {
        ProcessPixel(binary_, labels_, global_parent, row, col, next_label);
      }
    }

    thread_next_labels[thread_id] = next_label;
  }

  std::vector<int> thread_offsets(num_threads_total + 1, 0);
  for (int i = 0; i < num_threads_total; ++i) {
    thread_offsets[i + 1] = thread_offsets[i] + thread_next_labels[i] - 1;
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

#pragma omp parallel for
  for (int thread_id = 1; thread_id < num_threads_total; ++thread_id) {
    int boundary_row = thread_start_row[thread_id];

    if (boundary_row >= height || boundary_row <= 0) {
      continue;
    }

    for (int col = 0; col < width; ++col) {
      if (binary_[boundary_row][col] == 1 && binary_[boundary_row - 1][col] == 1) {
        int label_top = labels_[boundary_row - 1][col];
        int label_bottom = labels_[boundary_row][col];

        if (label_top != 0 && label_bottom != 0 && label_top != label_bottom) {
#pragma omp critical
          {
            UnionLabels(global_parent, label_top, label_bottom);
          }
        }
      }
    }
  }

  int max_label = 0;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      max_label = std::max(max_label, labels_[i][j]);
    }
  }

#pragma omp parallel for
  for (int label = 1; label <= max_label; ++label) {
    if (global_parent[label] != 0) {
      global_parent[label] = FindRoot(global_parent, label);
    }
  }

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int start_row = thread_start_row[thread_id];
    int end_row = thread_end_row[thread_id];

    for (int row = start_row; row < end_row; ++row) {
      for (int col = 0; col < width; ++col) {
        int label = labels_[row][col];
        if (label != 0) {
          labels_[row][col] = FindRoot(global_parent, label);
        }
      }
    }
  }
}

void MarinLMarkComponentsOMP::SecondPass() {
  const int height = static_cast<int>(labels_.size());
  const int width = height > 0 ? static_cast<int>(labels_.front().size()) : 0;

  if (height == 0 || width == 0) {
    return;
  }

  auto &parent = GetThreadLocalParent();

  int max_label = 0;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      max_label = std::max(max_label, labels_[i][j]);
    }
  }

  if (max_label == 0) {
    return;
  }

  std::vector<int> root_to_compact(static_cast<std::size_t>(max_label + 1), 0);
  std::vector<int> unique_roots;

  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int label = labels_[i][j];
      if (label != 0) {
        int root = FindRoot(parent, label);
        if (root_to_compact[root] == 0) {
          root_to_compact[root] = 1;
          unique_roots.push_back(root);
        }
      }
    }
  }

  std::sort(unique_roots.begin(), unique_roots.end());

  for (size_t i = 0; i < unique_roots.size(); ++i) {
    root_to_compact[unique_roots[i]] = static_cast<int>(i + 1);
  }

  const int num_threads = omp_get_max_threads();

#pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    int rows_per_thread = height / num_threads;
    int start_row = thread_id * rows_per_thread;
    int end_row = (thread_id == num_threads - 1) ? height : start_row + rows_per_thread;

    for (int row = start_row; row < end_row; ++row) {
      for (int col = 0; col < width; ++col) {
        int label = labels_[row][col];
        if (label != 0) {
          int root = FindRoot(parent, label);
          labels_[row][col] = root_to_compact[root];
        }
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
  OutType out;
  out.labels = labels_;
  GetOutput() = out;
  return true;
}

}  // namespace marin_l_mark_components
