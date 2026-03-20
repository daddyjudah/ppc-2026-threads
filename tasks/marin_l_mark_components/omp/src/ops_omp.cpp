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

void MarinLMarkComponentsOMP::FirstPass() {
  const int height = static_cast<int>(binary_.size());
  const int width = static_cast<int>(binary_[0].size());

  const int num_threads = omp_get_max_threads();
  const int chunk = (height + num_threads - 1) / num_threads;

  std::vector<int> parent((height * width) + 1);
  for (std::size_t index = 0; index < parent.size(); ++index) {
    parent[index] = static_cast<int>(index);
  }

  auto &labels = labels_;
  const auto &binary = binary_;

#pragma omp parallel default(none) shared(parent, height, width, chunk, num_threads, labels, binary)
  {
    const int thread_id = omp_get_thread_num();
    const int start_row = thread_id * chunk;
    int end_row = start_row + chunk;
    if (end_row > height) {
      end_row = height;
    }

    if (start_row < end_row) {
      int label = (start_row * width) + 1;

      for (int row_idx = start_row; row_idx < end_row; ++row_idx) {
        auto &row = labels[row_idx];

        for (int col_idx = 0; col_idx < width; ++col_idx) {
          if (binary[row_idx][col_idx] == 0) {
            row[col_idx] = 0;
            continue;
          }

          const int left = (col_idx > 0) ? row[col_idx - 1] : 0;
          const int top = (row_idx > start_row) ? labels[row_idx - 1][col_idx] : 0;

          if (left == 0 && top == 0) {
            row[col_idx] = label++;
          } else if (left != 0 && top == 0) {
            row[col_idx] = left;
          } else if (left == 0 && top != 0) {
            row[col_idx] = top;
          } else {
            const int mn = (left < top) ? left : top;
            const int mx = (left > top) ? left : top;

            row[col_idx] = mn;

#pragma omp critical
            {
              UnionLabels(parent, mn, mx);
            }
          }
        }
      }
    }
  }

  for (int thread_idx = 1; thread_idx < num_threads; ++thread_idx) {
    const int row_idx = thread_idx * chunk;
    if (row_idx >= height) {
      continue;
    }

    for (int col_idx = 0; col_idx < width; ++col_idx) {
      if (binary[row_idx][col_idx] != 0 && binary[row_idx - 1][col_idx] != 0) {
        const int a = labels[row_idx][col_idx];
        const int b = labels[row_idx - 1][col_idx];

        if (a != b) {
          UnionLabels(parent, a, b);
        }
      }
    }
  }

#pragma omp parallel for default(none) shared(parent)
  for (std::size_t index = 1; index < parent.size(); ++index) {
    parent[index] = FindRoot(parent, static_cast<int>(index));
  }

#pragma omp parallel for collapse(2) default(none) shared(height, width, labels, parent)
  for (int row_idx = 0; row_idx < height; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      if (labels[row_idx][col_idx] != 0) {
        labels[row_idx][col_idx] = parent[labels[row_idx][col_idx]];
      }
    }
  }
}

void MarinLMarkComponentsOMP::SecondPass() {
  const int height = static_cast<int>(labels_.size());
  const int width = static_cast<int>(labels_[0].size());

  int max_label = 0;

#pragma omp parallel for reduction(max : max_label) collapse(2) default(none) shared(height, width)
  for (int row_idx = 0; row_idx < height; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      max_label = std::max(max_label, labels_[row_idx][col_idx]);
    }
  }

  if (max_label == 0) {
    return;
  }

  std::vector<char> used(max_label + 1, 0);

#pragma omp parallel for collapse(2) default(none) shared(height, width, used)
  for (int row_idx = 0; row_idx < height; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      if (labels_[row_idx][col_idx] != 0) {
        used[labels_[row_idx][col_idx]] = 1;
      }
    }
  }

  std::vector<int> map(max_label + 1, 0);
  int next_label = 1;

  for (int label_idx = 1; label_idx <= max_label; ++label_idx) {
    if (used[label_idx] != 0) {
      map[label_idx] = next_label++;
    }
  }

#pragma omp parallel for collapse(2) default(none) shared(height, width, map)
  for (int row_idx = 0; row_idx < height; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      if (labels_[row_idx][col_idx] != 0) {
        labels_[row_idx][col_idx] = map[labels_[row_idx][col_idx]];
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
