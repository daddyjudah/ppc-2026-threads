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

unsigned int GetThreadCount() {
  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) {
    num_threads = 4;
  }
  return num_threads;
}

std::vector<int> BuildStripeBounds(int height, int num_stripes) {
  std::vector<int> stripe_bounds(static_cast<std::size_t>(num_stripes) + 1ULL, 0);
  for (int stripe = 0; stripe <= num_stripes; ++stripe) {
    stripe_bounds[static_cast<std::size_t>(stripe)] = (stripe * height) / num_stripes;
  }
  return stripe_bounds;
}

int FindRoot(std::vector<int> &parent, int x) {
  int root = x;
  while (parent[static_cast<std::size_t>(root)] != root) {
    root = parent[static_cast<std::size_t>(root)];
  }

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

int ResolvePixelLabel(int left_label, int top_label, std::vector<int> &parent) {
  if (left_label == 0 && top_label == 0) {
    return 0;
  }
  if (left_label == 0) {
    return top_label;
  }
  if (top_label == 0) {
    return left_label;
  }

  const int min_label = std::min(left_label, top_label);
  if (left_label != top_label) {
    UnionLabels(parent, left_label, top_label);
  }
  return min_label;
}

int GetLeftLabel(const std::vector<int> &labels_flat, std::size_t idx, int col) {
  if (col <= 0) {
    return 0;
  }
  return labels_flat[idx - 1ULL];
}

int GetTopLabel(const std::vector<int> &labels_flat, std::size_t prev_row_offset, int row, int start_row, int col) {
  if (row <= start_row) {
    return 0;
  }
  return labels_flat[prev_row_offset + static_cast<std::size_t>(col)];
}

void ProcessStripeFirstPass(const std::vector<std::uint8_t> &binary_flat, std::vector<int> &labels_flat, int width,
                            int start_row, int end_row, int base_label, std::vector<int> &parent, int &max_used) {
  int next_label = base_label;
  for (int row = start_row; row < end_row; ++row) {
    const auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
    const auto prev_row_offset = static_cast<std::size_t>(row - 1) * static_cast<std::size_t>(width);

    for (int col = 0; col < width; ++col) {
      const auto idx = row_offset + static_cast<std::size_t>(col);
      if (binary_flat[idx] == 0U) {
        continue;
      }

      const int left_label = GetLeftLabel(labels_flat, idx, col);
      const int top_label = GetTopLabel(labels_flat, prev_row_offset, row, start_row, col);
      const int label = ResolvePixelLabel(left_label, top_label, parent);
      if (label == 0) {
        labels_flat[idx] = next_label++;
        continue;
      }
      labels_flat[idx] = label;
    }
  }
  max_used = next_label;
}

void MergeStripeBoundaries(const std::vector<std::uint8_t> &binary_flat, const std::vector<int> &labels_flat, int width,
                           const std::vector<int> &stripe_bounds, std::vector<int> &parent) {
  const int num_stripes = static_cast<int>(stripe_bounds.size()) - 1;
  for (int stripe = 0; stripe < num_stripes - 1; ++stripe) {
    const int boundary_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];
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

std::vector<int> BuildCompactedLabels(const std::vector<int> &stripe_base_label,
                                      const std::vector<int> &stripe_max_used, std::vector<int> &parent,
                                      int total_max_labels) {
  std::vector<int> compacted(static_cast<std::size_t>(total_max_labels), 0);
  int next_compact_id = 1;

  for (std::size_t stripe = 0; stripe < stripe_base_label.size(); ++stripe) {
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

void ApplyCompactedLabels(std::vector<int> &labels_flat, int width, int start_row, int end_row,
                          const std::vector<int> &compacted) {
  for (int row = start_row; row < end_row; ++row) {
    const auto row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
    for (int col = 0; col < width; ++col) {
      const auto idx = row_offset + static_cast<std::size_t>(col);
      const int label = labels_flat[idx];
      if (label != 0) {
        labels_flat[idx] = compacted[static_cast<std::size_t>(label)];
      }
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

  const int num_stripes = std::min(height_, static_cast<int>(GetThreadCount()));

  std::vector<std::thread> threads;
  threads.reserve(static_cast<std::size_t>(num_stripes));

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

  const int num_stripes = std::min(height_, static_cast<int>(GetThreadCount()));

  const std::vector<int> stripe_bounds = BuildStripeBounds(height_, num_stripes);
  std::vector<int> stripe_base_label(static_cast<std::size_t>(num_stripes), 0);
  std::vector<int> stripe_max_used(static_cast<std::size_t>(num_stripes), 0);

  int total_max_labels = 1;

  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    stripe_base_label[static_cast<std::size_t>(stripe)] = total_max_labels;

    const int stripe_height = (((stripe + 1) * height_) / num_stripes) - ((stripe * height_) / num_stripes);
    total_max_labels += ((stripe_height * width_) / 2) + 1;
  }

  std::vector<int> parent(static_cast<std::size_t>(total_max_labels));
  for (int i = 0; i < total_max_labels; ++i) {
    parent[static_cast<std::size_t>(i)] = i;
  }

  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(num_stripes));

  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    workers.emplace_back([this, &parent, &stripe_bounds, &stripe_base_label, &stripe_max_used, stripe]() {
      const int start_row = stripe_bounds[static_cast<std::size_t>(stripe)];
      const int end_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];
      ProcessStripeFirstPass(binary_flat_, labels_flat_, width_, start_row, end_row,
                             stripe_base_label[static_cast<std::size_t>(stripe)], parent,
                             stripe_max_used[static_cast<std::size_t>(stripe)]);
    });
  }

  for (auto &t : workers) {
    t.join();
  }
  workers.clear();

  MergeStripeBoundaries(binary_flat_, labels_flat_, width_, stripe_bounds, parent);

  const std::vector<int> compacted = BuildCompactedLabels(stripe_base_label, stripe_max_used, parent, total_max_labels);

  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    workers.emplace_back([this, &stripe_bounds, &compacted, stripe]() {
      const int start_row = stripe_bounds[static_cast<std::size_t>(stripe)];
      const int end_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];
      ApplyCompactedLabels(labels_flat_, width_, start_row, end_row, compacted);
    });
  }

  for (auto &t : workers) {
    t.join();
  }

  return true;
}

bool MarinLMarkComponentsSTL::PostProcessingImpl() {
  labels_out_.clear();
  labels_out_.reserve(static_cast<std::size_t>(height_));

  for (int i = 0; i < height_; ++i) {
    labels_out_.emplace_back(static_cast<std::size_t>(width_));
  }

  const int num_stripes = std::min(height_, static_cast<int>(GetThreadCount()));

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
