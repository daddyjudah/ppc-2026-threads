#include "marin_l_mark_components/all/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "marin_l_mark_components/common/include/common.hpp"
#include "oneapi/tbb/blocked_range.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task_arena.h"

namespace marin_l_mark_components {

namespace {

constexpr std::uint64_t kMaxPixels = 100000000ULL;
constexpr int kMinRowsPerStripe = 64;

struct StripeSetup {
  int num_stripes = 1;
  int total_max_labels = 1;
  std::vector<int> stripe_bounds;
  std::vector<int> stripe_base_label;
};

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
  if (root_a == root_b) {
    return;
  }

  if (root_a < root_b) {
    parent[static_cast<std::size_t>(root_b)] = root_a;
  } else {
    parent[static_cast<std::size_t>(root_a)] = root_b;
  }
}

StripeSetup BuildStripeSetup(int height, int width) {
  StripeSetup setup;
  if (height <= 0 || width <= 0) {
    return setup;
  }

  setup.num_stripes = std::min(height, oneapi::tbb::this_task_arena::max_concurrency() * 2);
  if (setup.num_stripes > 0 && height / setup.num_stripes < kMinRowsPerStripe) {
    setup.num_stripes = std::max(1, height / kMinRowsPerStripe);
  }
  setup.num_stripes = std::max(1, setup.num_stripes);

  setup.stripe_bounds.assign(static_cast<std::size_t>(setup.num_stripes) + 1ULL, 0);
  for (int stripe = 0; stripe <= setup.num_stripes; ++stripe) {
    setup.stripe_bounds[static_cast<std::size_t>(stripe)] = (stripe * height) / setup.num_stripes;
  }

  setup.stripe_base_label.assign(static_cast<std::size_t>(setup.num_stripes), 0);
  for (int stripe = 0; stripe < setup.num_stripes; ++stripe) {
    setup.stripe_base_label[static_cast<std::size_t>(stripe)] = setup.total_max_labels;
    const int stripe_height = setup.stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL] -
                              setup.stripe_bounds[static_cast<std::size_t>(stripe)];
    setup.total_max_labels += ((stripe_height * width) / 2) + 1;
  }

  return setup;
}

void InitializeParents(std::vector<int> &parent, int total_max_labels) {
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, total_max_labels),
                            [&](const oneapi::tbb::blocked_range<int> &range) {
    for (int label = range.begin(); label != range.end(); ++label) {
      parent[static_cast<std::size_t>(label)] = label;
    }
  });
}

int MergeLabels(std::vector<int> &parent, int left_label, int top_label, int &next_label) {
  if (left_label == 0 && top_label == 0) {
    return next_label++;
  }
  if (left_label == 0) {
    return top_label;
  }
  if (top_label == 0) {
    return left_label;
  }

  const int merged_label = std::min(left_label, top_label);
  if (left_label != top_label) {
    UnionLabels(parent, left_label, top_label);
  }
  return merged_label;
}

void LabelStripe(const std::vector<std::uint8_t> &binary_flat, std::vector<int> &labels_flat, std::vector<int> &parent,
                 const std::vector<int> &stripe_bounds, const std::vector<int> &stripe_base_label,
                 std::vector<int> &stripe_max_used, int width, int stripe) {
  const int start_row = stripe_bounds[static_cast<std::size_t>(stripe)];
  const int end_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];
  int next_label = stripe_base_label[static_cast<std::size_t>(stripe)];

  for (int row = start_row; row < end_row; ++row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
    const std::size_t prev_row_offset = row_offset - static_cast<std::size_t>(width);

    for (int col = 0; col < width; ++col) {
      const std::size_t idx = row_offset + static_cast<std::size_t>(col);
      if (binary_flat[idx] == 0U) {
        continue;
      }

      const int left_label = (col > 0) ? labels_flat[idx - 1ULL] : 0;
      const int top_label = (row > start_row) ? labels_flat[prev_row_offset + static_cast<std::size_t>(col)] : 0;
      labels_flat[idx] = MergeLabels(parent, left_label, top_label, next_label);
    }
  }

  stripe_max_used[static_cast<std::size_t>(stripe)] = next_label;
}

void MergeStripeBorders(const std::vector<std::uint8_t> &binary_flat, const std::vector<int> &stripe_bounds,
                        std::vector<int> &labels_flat, std::vector<int> &parent, int width, int num_stripes) {
  for (int stripe = 0; stripe < num_stripes - 1; ++stripe) {
    const int boundary_row = stripe_bounds[static_cast<std::size_t>(stripe) + 1ULL];
    const std::size_t row_offset = static_cast<std::size_t>(boundary_row) * static_cast<std::size_t>(width);
    const std::size_t prev_row_offset = static_cast<std::size_t>(boundary_row - 1) * static_cast<std::size_t>(width);

    for (int col = 0; col < width; ++col) {
      const std::size_t bottom_idx = row_offset + static_cast<std::size_t>(col);
      const std::size_t top_idx = prev_row_offset + static_cast<std::size_t>(col);
      if (binary_flat[top_idx] == 1U && binary_flat[bottom_idx] == 1U) {
        UnionLabels(parent, labels_flat[top_idx], labels_flat[bottom_idx]);
      }
    }
  }
}

void BuildCompactedLabels(std::vector<int> &labels_flat, std::vector<int> &parent,
                          const std::vector<int> &stripe_base_label, const std::vector<int> &stripe_max_used,
                          int total_max_labels, int num_stripes) {
  for (int label = 1; label < total_max_labels; ++label) {
    parent[static_cast<std::size_t>(label)] = FindRoot(parent, label);
  }

  std::vector<int> compacted(static_cast<std::size_t>(total_max_labels), 0);
  int next_compact_id = 1;

  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    for (int label = stripe_base_label[static_cast<std::size_t>(stripe)];
         label < stripe_max_used[static_cast<std::size_t>(stripe)]; ++label) {
      const int root = parent[static_cast<std::size_t>(label)];
      if (compacted[static_cast<std::size_t>(root)] == 0) {
        compacted[static_cast<std::size_t>(root)] = next_compact_id++;
      }
      compacted[static_cast<std::size_t>(label)] = compacted[static_cast<std::size_t>(root)];
    }
  }

  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, labels_flat.size()),
                            [&](const oneapi::tbb::blocked_range<std::size_t> &range) {
    for (std::size_t idx = range.begin(); idx != range.end(); ++idx) {
      const int label = labels_flat[idx];
      if (label != 0) {
        labels_flat[idx] = compacted[static_cast<std::size_t>(label)];
      }
    }
  });
}

}  // namespace

MarinLMarkComponentsALL::MarinLMarkComponentsALL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsALL::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (int pixel : row) {
      if (pixel != 0 && pixel != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MarinLMarkComponentsALL::ValidationImpl() {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

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

bool MarinLMarkComponentsALL::PreProcessingImpl() {
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

  local_binary_flat_.assign(static_cast<std::size_t>(total_pixels), 0);
  global_labels_flat_.assign(static_cast<std::size_t>(total_pixels), 0);
  labels_out_.clear();

  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, height_),
                            [&](const oneapi::tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row != range.end(); ++row) {
      const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
      for (int col = 0; col < width_; ++col) {
        local_binary_flat_[row_offset + static_cast<std::size_t>(col)] =
            static_cast<std::uint8_t>(img[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
      }
    }
  });

  return true;
}

bool MarinLMarkComponentsALL::RunImpl() {
  const StripeSetup setup = BuildStripeSetup(height_, width_);
  std::vector<int> parent(static_cast<std::size_t>(setup.total_max_labels), 0);
  InitializeParents(parent, setup.total_max_labels);

  std::vector<int> stripe_max_used(static_cast<std::size_t>(setup.num_stripes), 0);
  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, setup.num_stripes),
                            [&](const oneapi::tbb::blocked_range<int> &range) {
    for (int stripe = range.begin(); stripe != range.end(); ++stripe) {
      LabelStripe(local_binary_flat_, global_labels_flat_, parent, setup.stripe_bounds, setup.stripe_base_label,
                  stripe_max_used, width_, stripe);
    }
  });

  MergeStripeBorders(local_binary_flat_, setup.stripe_bounds, global_labels_flat_, parent, width_, setup.num_stripes);
  BuildCompactedLabels(global_labels_flat_, parent, setup.stripe_base_label, stripe_max_used, setup.total_max_labels,
                       setup.num_stripes);

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool MarinLMarkComponentsALL::PostProcessingImpl() {
  ConvertLabelsToOutput();
  OutType out;
  out.labels = labels_out_;
  GetOutput() = out;
  return true;
}

void MarinLMarkComponentsALL::ConvertLabelsToOutput() {
  labels_out_.resize(static_cast<std::size_t>(height_));

  oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, height_),
                            [&](const oneapi::tbb::blocked_range<int> &range) {
    for (int row = range.begin(); row != range.end(); ++row) {
      labels_out_[static_cast<std::size_t>(row)].resize(static_cast<std::size_t>(width_));
      const std::size_t row_offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(width_);
      for (int col = 0; col < width_; ++col) {
        labels_out_[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
            global_labels_flat_[row_offset + static_cast<std::size_t>(col)];
      }
    }
  });
}

}  // namespace marin_l_mark_components
