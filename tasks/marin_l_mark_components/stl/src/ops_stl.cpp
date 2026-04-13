#include "marin_l_mark_components/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "marin_l_mark_components/common/include/common.hpp"

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;
// Удалена неиспользуемая константа kMinRowsPerStripe

struct StripeSetup {
  int num_stripes = 1;
  int total_max_labels = 1;
  std::vector<int> stripe_bounds;
  std::vector<int> stripe_base_label;
};

int FindRoot(std::vector<int> &parent, int x) {
  int root = x;
  while (parent[root] != root) {
    root = parent[root];
  }
  int current = x;
  while (current != root) {
    const int next = parent[current];
    parent[current] = root;
    current = next;
  }
  return root;
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

StripeSetup BuildSmartStripeSetup(int height, int width, const std::vector<int> &row_weights, int total_weight) {
  StripeSetup setup;
  int num_threads = static_cast<int>(std::thread::hardware_concurrency());
  setup.num_stripes = std::min(height, std::max(1, num_threads * 2));

  setup.stripe_bounds.push_back(0);

  if (total_weight == 0) {
    for (int i = 1; i <= setup.num_stripes; ++i) {
      setup.stripe_bounds.push_back((i * height) / setup.num_stripes);
    }
  } else {
    int target_weight = total_weight / setup.num_stripes;
    int current_weight = 0;
    int current_stripe = 0;

    for (int r = 0; r < height; ++r) {
      current_weight += row_weights[r];

      if (current_weight >= target_weight && current_stripe < setup.num_stripes - 1) {
        setup.stripe_bounds.push_back(r + 1);
        current_weight = 0;
        current_stripe++;
      }
    }

    while (static_cast<int>(setup.stripe_bounds.size()) <= setup.num_stripes) {
      setup.stripe_bounds.push_back(height);
    }
    setup.stripe_bounds.back() = height;
  }

  setup.stripe_base_label.assign(setup.num_stripes, 0);
  for (int stripe = 0; stripe < setup.num_stripes; ++stripe) {
    setup.stripe_base_label[stripe] = setup.total_max_labels;
    const int stripe_height = setup.stripe_bounds[stripe + 1] - setup.stripe_bounds[stripe];
    setup.total_max_labels += ((stripe_height * width) / 2) + 1;
  }

  return setup;
}

void AssignPixelLabel(std::vector<int> &labels_flat, std::vector<int> &parent, std::size_t idx, int left_label,
                      int top_label, int &next_label) {
  if (left_label == 0 && top_label == 0) {
    labels_flat[idx] = next_label++;
    return;
  }
  if (left_label != 0 && top_label == 0) {
    labels_flat[idx] = left_label;
    return;
  }
  if (left_label == 0 && top_label != 0) {
    labels_flat[idx] = top_label;
    return;
  }

  const int min_label = std::min(left_label, top_label);
  labels_flat[idx] = min_label;
  if (left_label != top_label) {
    UnionLabels(parent, left_label, top_label);
  }
}

void LabelStripe(const std::vector<std::uint8_t> &binary_flat, std::vector<int> &labels_flat, std::vector<int> &parent,
                 const std::vector<int> &stripe_bounds, const std::vector<int> &stripe_base_label,
                 std::vector<int> &stripe_max_used, int width, int stripe) {
  const int start_row = stripe_bounds[stripe];
  const int end_row = stripe_bounds[stripe + 1];
  int next_label = stripe_base_label[stripe];

  for (int row = start_row; row < end_row; ++row) {
    const auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
    const auto prev_row_offset = static_cast<std::size_t>(row - 1) * static_cast<std::size_t>(width);

    for (int col = 0; col < width; ++col) {
      const auto idx = row_offset + static_cast<std::size_t>(col);
      if (binary_flat[idx] == 0U) {
        continue;
      }

      const int left_label = (col > 0) ? labels_flat[idx - 1ULL] : 0;
      const int top_label = (row > start_row) ? labels_flat[prev_row_offset + static_cast<std::size_t>(col)] : 0;
      AssignPixelLabel(labels_flat, parent, idx, left_label, top_label, next_label);
    }
  }
  stripe_max_used[stripe] = next_label;
}

void MergeStripeBorders(const std::vector<std::uint8_t> &binary_flat, const std::vector<int> &stripe_bounds,
                        std::vector<int> &labels_flat, std::vector<int> &parent, int width, int num_stripes) {
  for (int stripe = 0; stripe < num_stripes - 1; ++stripe) {
    const int boundary_row = stripe_bounds[stripe + 1];
    if (boundary_row == 0 || boundary_row == stripe_bounds.back()) {
      continue;
    }

    const auto row_offset = static_cast<std::size_t>(boundary_row) * static_cast<std::size_t>(width);
    const auto prev_row_offset = static_cast<std::size_t>(boundary_row - 1) * static_cast<std::size_t>(width);

    for (int col = 0; col < width; ++col) {
      const auto bottom_idx = row_offset + static_cast<std::size_t>(col);
      const auto top_idx = prev_row_offset + static_cast<std::size_t>(col);
      if (binary_flat[bottom_idx] == 1U && binary_flat[top_idx] == 1U) {
        UnionLabels(parent, labels_flat[bottom_idx], labels_flat[top_idx]);
      }
    }
  }
}

std::vector<int> BuildCompactedLabels(std::vector<int> &parent, const std::vector<int> &stripe_base_label,
                                      const std::vector<int> &stripe_max_used, int total_max_labels, int num_stripes) {
  std::vector<int> compacted(static_cast<std::size_t>(total_max_labels), 0);
  int next_compact_id = 1;

  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    for (int label = stripe_base_label[stripe]; label < stripe_max_used[stripe]; ++label) {
      const int root = FindRoot(parent, label);
      if (compacted[static_cast<std::size_t>(root)] == 0) {
        compacted[static_cast<std::size_t>(root)] = next_compact_id++;
      }
      compacted[static_cast<std::size_t>(label)] = compacted[static_cast<std::size_t>(root)];
    }
  }
  return compacted;
}

}  // namespace

MarinLMarkComponentsSTL::MarinLMarkComponentsSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsSTL::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (int pixel : row) {
      if (pixel != 0 && pixel != 1) {
        return false;
      }
    }
  }
  return true;
}

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

  binary_flat_.assign(total_pixels, 0);
  labels_flat_.assign(total_pixels, 0);
  row_weights_.assign(height_, 0);

  int num_threads = static_cast<int>(std::thread::hardware_concurrency());
  std::vector<std::thread> threads;
  int chunk_size = std::max(1, height_ / num_threads);

  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? height_ : start + chunk_size;
    if (start >= height_) {
      break;
    }

    threads.emplace_back([this, &img, start, end]() {
      for (int row = start; row < end; ++row) {
        int weight = 0;
        const auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
        for (int col = 0; col < width_; ++col) {
          uint8_t val = static_cast<std::uint8_t>(img[row][col]);
          binary_flat_[row_offset + static_cast<std::size_t>(col)] = val;
          weight += val;
        }
        row_weights_[row] = weight;
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

  int total_weight = 0;
  for (int w : row_weights_) {
    total_weight += w;
  }

  const StripeSetup setup = BuildSmartStripeSetup(height_, width_, row_weights_, total_weight);

  std::vector<int> parent(setup.total_max_labels);
  for (int i = 0; i < setup.total_max_labels; ++i) {
    parent[i] = i;
  }

  std::vector<int> stripe_max_used(setup.num_stripes, 0);
  std::vector<std::thread> workers;

  for (int stripe = 0; stripe < setup.num_stripes; ++stripe) {
    workers.emplace_back([this, &parent, &setup, &stripe_max_used, stripe]() {
      LabelStripe(binary_flat_, labels_flat_, parent, setup.stripe_bounds, setup.stripe_base_label, stripe_max_used,
                  width_, stripe);
    });
  }
  for (auto &t : workers) {
    t.join();
  }
  workers.clear();

  MergeStripeBorders(binary_flat_, setup.stripe_bounds, labels_flat_, parent, width_, setup.num_stripes);

  const std::vector<int> compacted =
      BuildCompactedLabels(parent, setup.stripe_base_label, stripe_max_used, setup.total_max_labels, setup.num_stripes);

  for (int stripe = 0; stripe < setup.num_stripes; ++stripe) {
    workers.emplace_back([this, &setup, &compacted, stripe]() {
      const int start_row = setup.stripe_bounds[stripe];
      const int end_row = setup.stripe_bounds[stripe + 1];

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

bool MarinLMarkComponentsSTL::PostProcessingImpl() {
  labels_out_.clear();
  labels_out_.reserve(height_);
  for (int i = 0; i < height_; ++i) {
    labels_out_.emplace_back(width_);
  }

  int num_threads = static_cast<int>(std::thread::hardware_concurrency());
  std::vector<std::thread> threads;
  int chunk_size = std::max(1, height_ / num_threads);

  for (int i = 0; i < num_threads; ++i) {
    int start = i * chunk_size;
    int end = (i == num_threads - 1) ? height_ : start + chunk_size;
    if (start >= height_) {
      break;
    }

    threads.emplace_back([this, start, end]() {
      for (int row = start; row < end; ++row) {
        const auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
        std::copy(labels_flat_.begin() + static_cast<std::ptrdiff_t>(row_offset),
                  labels_flat_.begin() + static_cast<std::ptrdiff_t>(row_offset) + width_, labels_out_[row].begin());
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
