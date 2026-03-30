#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <tbb/tbb.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace marin_l_mark_components {

namespace {
// Быстрый поиск корня с компрессией пути
inline int FindRoot(int *parent, int x) {
  int root = x;
  while (parent[root] != root) {
    parent[root] = parent[parent[root]];  // Сжатие пути "на лету"
    root = parent[root];
  }
  return root;
}

// Детерминированное объединение: всегда к меньшему индексу
inline void UnionLabels(int *parent, int a, int b) {
  int root_a = FindRoot(parent, a);
  int root_b = FindRoot(parent, b);
  if (root_a != root_b) {
    if (root_a < root_b) {
      parent[root_b] = root_a;
    } else {
      parent[root_a] = root_b;
    }
  }
}
}  // namespace

MarinLMarkComponentsTBB::MarinLMarkComponentsTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsTBB::ValidationImpl() {
  return !GetInput().binary.empty() && !GetInput().binary[0].empty();
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

  tbb::parallel_for(0, height_, [&](int r) {
    std::copy(input[r].begin(), input[r].end(), binary_flat_.begin() + (size_t)r * width_);
  });
  return true;
}

// ИСПРАВЛЕНО: тип возвращаемого значения изменен на bool
bool MarinLMarkComponentsTBB::RunImpl() {
  int *l_ptr = labels_flat_.data();
  const uint8_t *b_ptr = binary_flat_.data();
  int *p_ptr = parent_.data();
  int w = width_;
  int h = height_;

  // 1. ПАРАЛЛЕЛЬНЫЙ ПРОХОД: Локальная разметка в каждой строке
  tbb::parallel_for(0, h, [&](int r) {
    size_t row_off = (size_t)r * w;
    int last_label = 0;
    for (int c = 0; c < w; ++c) {
      size_t idx = row_off + c;
      if (b_ptr[idx]) {
        if (c > 0 && b_ptr[idx - 1]) {
          l_ptr[idx] = last_label;
        } else {
          l_ptr[idx] = static_cast<int>(idx + 1);
          last_label = l_ptr[idx];
        }
      }
    }
  });

  // 2. СТРАТЕГИЯ MERGE: Склеиваем границы строк
  tbb::spin_mutex merge_mutex;
  tbb::parallel_for(1, h, [&](int r) {
    size_t curr_row = (size_t)r * w;
    size_t prev_row = curr_row - w;
    for (int c = 0; c < w; ++c) {
      if (b_ptr[curr_row + c] && b_ptr[prev_row + c]) {
        tbb::spin_mutex::scoped_lock lock(merge_mutex);
        UnionLabels(p_ptr, l_ptr[curr_row + c], l_ptr[prev_row + c]);
      }
    }
  });

  // 3. ФИНАЛИЗАЦИЯ: Сплющивание DSU (Параллельно)
  size_t total = (size_t)h * w;
  tbb::parallel_for(tbb::blocked_range<size_t>(0, total, 10000), [&](const tbb::blocked_range<size_t> &range) {
    for (size_t i = range.begin(); i < range.end(); ++i) {
      if (l_ptr[i] > 0) {
        l_ptr[i] = FindRoot(p_ptr, l_ptr[i]);
      }
    }
  });

  // 4. НОРМАЛИЗАЦИЯ: Последовательно (для тестов)
  std::vector<int> lookup(total + 1, 0);
  int next_label = 1;
  for (size_t i = 0; i < total; ++i) {
    if (l_ptr[i] > 0) {
      int root = l_ptr[i];
      if (lookup[root] == 0) {
        lookup[root] = next_label++;
      }
      l_ptr[i] = lookup[root];
    }
  }

  return true;  // Не забываем вернуть true
}

bool MarinLMarkComponentsTBB::PostProcessingImpl() {
  auto &output = GetOutput().labels;
  output.resize(height_);
  int w = width_;
  tbb::parallel_for(0, height_, [&](int r) {
    output[r].assign(labels_flat_.data() + (size_t)r * w, labels_flat_.data() + (size_t)(r + 1) * w);
  });
  return true;
}

}  // namespace marin_l_mark_components
