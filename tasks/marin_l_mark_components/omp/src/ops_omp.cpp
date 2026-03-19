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
  int height = binary_.size();
  int width = binary_[0].size();

  int num_threads = omp_get_max_threads();
  int chunk = (height + num_threads - 1) / num_threads;

  std::vector<int> parent(height * width + 1);
  for (int i = 0; i < (int)parent.size(); ++i) {
    parent[i] = i;
  }

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int start = tid * chunk;
    int end = std::min(height, start + chunk);

    if (start < end) {
      int label = start * width + 1;

      for (int r = start; r < end; ++r) {
        auto &row = labels_[r];

        for (int c = 0; c < width; ++c) {
          if (!binary_[r][c]) {
            row[c] = 0;
            continue;
          }

          int left = (c > 0) ? row[c - 1] : 0;
          int top = (r > start) ? labels_[r - 1][c] : 0;

          if (left == 0 && top == 0) {
            row[c] = label++;
          } else if (left != 0 && top == 0) {
            row[c] = left;
          } else if (left == 0 && top != 0) {
            row[c] = top;
          } else {
            int mn = std::min(left, top);
            int mx = std::max(left, top);
            row[c] = mn;

            // 🔥 редкий случай → можно позволить critical
#pragma omp critical
            {
              UnionLabels(parent, mn, mx);
            }
          }
        }
      }
    }
  }

  // 🔥 merge границ блоков
  for (int t = 1; t < num_threads; ++t) {
    int r = t * chunk;
    if (r >= height) {
      continue;
    }

    for (int c = 0; c < width; ++c) {
      if (binary_[r][c] && binary_[r - 1][c]) {
        int a = labels_[r][c];
        int b = labels_[r - 1][c];
        if (a != b) {
          UnionLabels(parent, a, b);
        }
      }
    }
  }

#pragma omp parallel for
  for (int i = 1; i < (int)parent.size(); ++i) {
    parent[i] = FindRoot(parent, i);
  }

#pragma omp parallel for collapse(2)
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      if (labels_[r][c]) {
        labels_[r][c] = parent[labels_[r][c]];
      }
    }
  }
}

void MarinLMarkComponentsOMP::SecondPass() {
  int height = labels_.size();
  int width = labels_[0].size();

  int max_label = 0;

#pragma omp parallel for reduction(max : max_label) collapse(2)
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      max_label = std::max(max_label, labels_[r][c]);
    }
  }

  if (max_label == 0) {
    return;
  }

  std::vector<char> used(max_label + 1, 0);

#pragma omp parallel for collapse(2)
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      if (labels_[r][c] != 0) {
        used[labels_[r][c]] = 1;
      }
    }
  }

  std::vector<int> map(max_label + 1, 0);
  int next = 1;

  for (int i = 1; i <= max_label; ++i) {
    if (used[i]) {
      map[i] = next++;
    }
  }

#pragma omp parallel for collapse(2)
  for (int r = 0; r < height; ++r) {
    for (int c = 0; c < width; ++c) {
      if (labels_[r][c] != 0) {
        labels_[r][c] = map[labels_[r][c]];
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
