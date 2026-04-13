#include "marin_l_mark_components/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

#include "marin_l_mark_components/common/include/common.hpp"

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;

int FindRoot(std::vector<int> &parent, int x) {
  int root = x;
  while (parent[static_cast<std::size_t>(root)] != root) {
    root = parent[static_cast<std::size_t>(root)];
  }

  // Сжатие путей (Path Compression)
  int current = x;
  while (current != root) {
    const int next = parent[static_cast<std::size_t>(current)];
    parent[static_cast<std::size_t>(current)] = root;
    current = next;
  }
  return root;
}

void UnionLabels(std::vector<int> &parent, int a, int b) {
  const int root_a = FindRoot(parent, a);
  const int root_b = FindRoot(parent, b);

  if (root_a != root_b) {
    if (root_a < root_b) {
      parent[static_cast<std::size_t>(root_b)] = root_a;
    } else {
      parent[static_cast<std::size_t>(root_a)] = root_b;
    }
  }
}

}  // namespace

MarinLMarkComponentsSTL::MarinLMarkComponentsSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsSTL::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (const int pixel : row) {
      if (pixel != 0 && pixel != 1) {
        return false;
      }
    }
  }
  return true;
}

// Строго выровненная валидация
bool MarinLMarkComponentsSTL::ValidationImpl() {
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

bool MarinLMarkComponentsSTL::PreProcessingImpl() {
  const auto &img = GetInput().binary;
  height_ = static_cast<int>(img.size());
  width_ = static_cast<int>(img.front().size());

  if (height_ <= 0 || width_ <= 0) {
    return false;
  }

  const std::uint64_t total_pixels = static_cast<std::uint64_t>(height_) * static_cast<std::uint64_t>(width_);
  if (total_pixels > kMaxPixels) {
    return false;
  }

  binary_flat_.assign(static_cast<std::size_t>(total_pixels), 0);
  labels_flat_.assign(static_cast<std::size_t>(total_pixels), 0);

  // Определяем количество потоков
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 4;
  }
  int num_stripes = std::min(height_, static_cast<int>(num_threads));

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(num_stripes));

  // Параллельное уплощение (flattening)
  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    threads.emplace_back([this, &img, stripe, num_stripes]() {
      const int start_row = (stripe * height_) / num_stripes;
      const int end_row = ((stripe + 1) * height_) / num_stripes;

      for (int row = start_row; row < end_row; ++row) {
        const auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
        for (int col = 0; col < width_; ++col) {
          binary_flat_[row_offset + static_cast<std::size_t>(col)] =
              static_cast<std::uint8_t>(img[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
        }
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  return true;
}

bool MarinLMarkComponentsSTL::RunImpl() {
  if (height_ == 0 || width_ == 0) {
    return true;
  }

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 4;
  }
  int num_stripes = std::min(height_, static_cast<int>(num_threads));

  std::vector<int> stripe_bounds(static_cast<std::size_t>(num_stripes) + 1ULL, 0);
  std::vector<int> stripe_base_label(static_cast<std::size_t>(num_stripes), 0);
  std::vector<int> stripe_max_used(static_cast<std::size_t>(num_stripes), 0);

  int total_max_labels = 1;

  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    stripe_bounds[static_cast<std::size_t>(stripe)] = (stripe * height_) / num_stripes;
    stripe_base_label[static_cast<std::size_t>(stripe)] = total_max_labels;

    // Выделяем каждому страйпу независимое пространство ID, чтобы избежать гонок (Race Conditions)
    const int stripe_height = ((stripe + 1) * height_) / num_stripes - (stripe * height_) / num_stripes;
    total_max_labels += ((stripe_height * width_) / 2) + 1;
  }
  stripe_bounds[static_cast<std::size_t>(num_stripes)] = height_;

  std::vector<int> parent(static_cast<std::size_t>(total_max_labels));
  for (int i = 0; i < total_max_labels; ++i) {
    parent[static_cast<std::size_t>(i)] = i;
  }

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(num_stripes));

  // === ПЕРВЫЙ ПРОХОД: Локальная маркировка ===
  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    workers.emplace_back([this, &parent, &stripe_bounds, &stripe_base_label, &stripe_max_used, stripe]() {
      const int start_row = stripe_bounds[static_cast<std::size_t>(stripe)];
      const int end_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];
      int next_label = stripe_base_label[static_cast<std::size_t>(stripe)];

      for (int row = start_row; row < end_row; ++row) {
        const auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
        const auto prev_row_offset = static_cast<std::size_t>(row - 1) * static_cast<std::size_t>(width_);

        for (int col = 0; col < width_; ++col) {
          const auto idx = row_offset + static_cast<std::size_t>(col);
          if (binary_flat_[idx] == 0U) {
            continue;
          }

          const int left_label = (col > 0) ? labels_flat_[idx - 1ULL] : 0;
          const int top_label = (row > start_row) ? labels_flat_[prev_row_offset + static_cast<std::size_t>(col)] : 0;

          if (left_label == 0 && top_label == 0) {
            labels_flat_[idx] = next_label++;
          } else if (left_label != 0 && top_label == 0) {
            labels_flat_[idx] = left_label;
          } else if (left_label == 0 && top_label != 0) {
            labels_flat_[idx] = top_label;
          } else {
            const int min_label = std::min(left_label, top_label);
            labels_flat_[idx] = min_label;
            if (left_label != top_label) {
              UnionLabels(parent, left_label, top_label);
            }
          }
        }
      }
      stripe_max_used[static_cast<std::size_t>(stripe)] = next_label;
    });
  }

  for (auto &t : workers) {
    t.join();
  }
  workers.clear();

  // === МЕРДЖ ГРАНИЦ ===
  for (int stripe = 0; stripe < num_stripes - 1; ++stripe) {
    const int boundary_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];
    const auto row_offset = static_cast<std::size_t>(boundary_row) * static_cast<std::size_t>(width_);
    const auto prev_row_offset = static_cast<std::size_t>(boundary_row - 1) * static_cast<std::size_t>(width_);

    for (int col = 0; col < width_; ++col) {
      const auto bottom_idx = row_offset + static_cast<std::size_t>(col);
      const auto top_idx = prev_row_offset + static_cast<std::size_t>(col);

      if (binary_flat_[bottom_idx] == 1U && binary_flat_[top_idx] == 1U) {
        UnionLabels(parent, labels_flat_[bottom_idx], labels_flat_[top_idx]);
      }
    }
  }

  // === КОМПАКТИЗАЦИЯ ===
  std::vector<int> compacted(static_cast<std::size_t>(total_max_labels), 0);
  int next_compact_id = 1;

  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    for (int label = stripe_base_label[static_cast<std::size_t>(stripe)];
         label < stripe_max_used[static_cast<std::size_t>(stripe)]; ++label) {
      const int root = FindRoot(parent, label);
      if (compacted[static_cast<std::size_t>(root)] == 0) {
        compacted[static_cast<std::size_t>(root)] = next_compact_id++;
      }
      compacted[static_cast<std::size_t>(label)] = compacted[static_cast<std::size_t>(root)];
    }
  }

  // === ВТОРОЙ ПРОХОД: Применение финальных лейблов ===
  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    workers.emplace_back([this, &stripe_bounds, &compacted, stripe]() {
      const int start_row = stripe_bounds[static_cast<std::size_t>(stripe)];
      const int end_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];

      for (int row = start_row; row < end_row; ++row) {
        const auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
        for (int col = 0; col < width_; ++col) {
          const auto idx = row_offset + static_cast<std::size_t>(col);
          const int label = labels_flat_[idx];
          if (label != 0) {
            labels_flat_[idx] = compacted[static_cast<std::size_t>(label)];
          }
        }
      }
    });
  }

  for (auto &t : workers) {
    t.join();
  }

  return true;
}

// Строго выровненный пост-процессинг, обходящий баг resize в GCC 14 (array-bounds memmove)
bool MarinLMarkComponentsSTL::PostProcessingImpl() {
  labels_out_.clear();
  labels_out_.reserve(static_cast<std::size_t>(height_));

  for (int i = 0; i < height_; ++i) {
    labels_out_.emplace_back(static_cast<std::size_t>(width_));
  }

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 4;
  }
  int num_stripes = std::min(height_, static_cast<int>(num_threads));

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(num_stripes));

  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    threads.emplace_back([this, stripe, num_stripes]() {
      const int start_row = (stripe * height_) / num_stripes;
      const int end_row = ((stripe + 1) * height_) / num_stripes;

      for (int row = start_row; row < end_row; ++row) {
        const auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
        std::copy(labels_flat_.begin() + static_cast<std::ptrdiff_t>(row_offset),
                  labels_flat_.begin() + static_cast<std::ptrdiff_t>(row_offset) + width_,
                  labels_out_[static_cast<std::size_t>(row)].begin());
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }

  GetOutput().labels = std::move(labels_out_);
  return true;
}

}  // namespace marin_l_mark_components
