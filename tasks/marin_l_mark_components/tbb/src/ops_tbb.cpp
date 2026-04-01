#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include <algorithm>

namespace marin_l_mark_components {

namespace {

constexpr int kMinRowsPerStripe = 64;

int FindRoot(std::vector<int> &parent, int x) {
  int root = x;
  while (parent[root] != root) {
    root = parent[root];
  }

  int curr = x;
  while (curr != root) {
    int nxt = parent[curr];
    parent[curr] = root;
    curr = nxt;
  }
  return root;
}

void Union(std::vector<int> &parent, int a, int b) {
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

bool MarinLMarkComponentsTBB::IsBinary(const Image &img) {
  for (const auto &row : img) {
    for (int pixel : row) {
      if (pixel != 0 && pixel != 1) {
        return false;
      }
    }
  }
  return true;
}

bool MarinLMarkComponentsTBB::ValidationImpl() {
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

bool MarinLMarkComponentsTBB::PreProcessingImpl() {
  const auto &img = GetInput().binary;
  height_ = static_cast<int>(img.size());
  width_ = static_cast<int>(img.front().size());

  if (height_ <= 0 || width_ <= 0) {
    return false;
  }

  const std::size_t total_pixels = static_cast<std::size_t>(height_) * static_cast<std::size_t>(width_);

  binary_flat_.assign(total_pixels, 0);
  labels_flat_.assign(total_pixels, 0);

  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int> &r) {
    for (int y = r.begin(); y != r.end(); ++y) {
      const std::size_t row_offset = static_cast<std::size_t>(y) * width_;
      for (int x = 0; x < width_; ++x) {
        binary_flat_[row_offset + x] = static_cast<std::uint8_t>(img[y][x]);
      }
    }
  });

  return true;
}

bool MarinLMarkComponentsTBB::RunImpl() {
  if (height_ == 0 || width_ == 0) {
    return true;
  }

  int max_threads = tbb::this_task_arena::max_concurrency();
  int num_stripes = std::min(height_, max_threads * 2);
  if (num_stripes > 0 && height_ / num_stripes < kMinRowsPerStripe) {
    num_stripes = std::max(1, height_ / kMinRowsPerStripe);
  }

  std::vector<int> stripe_bounds(num_stripes + 1, 0);
  for (int i = 0; i <= num_stripes; ++i) {
    stripe_bounds[i] = i * height_ / num_stripes;
  }

  std::vector<int> stripe_base_label(num_stripes);
  int total_max_labels = 1;

  for (int i = 0; i < num_stripes; ++i) {
    stripe_base_label[i] = total_max_labels;
    int stripe_height = stripe_bounds[i + 1] - stripe_bounds[i];

    total_max_labels += (stripe_height * width_) / 2 + 1;
  }

  std::vector<int> parent(total_max_labels);
  tbb::parallel_for(0, total_max_labels, [&](int i) { parent[i] = i; });

  std::vector<int> stripe_max_used(num_stripes, 0);

  tbb::parallel_for(0, num_stripes, [&](int stripe) {
    int start_y = stripe_bounds[stripe];
    int end_y = stripe_bounds[stripe + 1];
    int next_label = stripe_base_label[stripe];

    for (int y = start_y; y < end_y; ++y) {
      int row_offset = y * width_;
      int prev_row_offset = (y - 1) * width_;

      for (int x = 0; x < width_; ++x) {
        if (binary_flat_[row_offset + x] == 0) {
          continue;
        }

        int left = (x > 0) ? labels_flat_[row_offset + x - 1] : 0;
        int top = (y > start_y) ? labels_flat_[prev_row_offset + x] : 0;

        if (left == 0 && top == 0) {
          labels_flat_[row_offset + x] = next_label++;
        } else if (left != 0 && top == 0) {
          labels_flat_[row_offset + x] = left;
        } else if (left == 0 && top != 0) {
          labels_flat_[row_offset + x] = top;
        } else {
          int min_l = std::min(left, top);
          labels_flat_[row_offset + x] = min_l;
          if (left != top) {
            Union(parent, left, top);
          }
        }
      }
    }
    stripe_max_used[stripe] = next_label;
  });

  for (int stripe = 0; stripe < num_stripes - 1; ++stripe) {
    int boundary_y = stripe_bounds[stripe + 1];
    int row_offset = boundary_y * width_;
    int prev_row_offset = (boundary_y - 1) * width_;

    for (int x = 0; x < width_; ++x) {
      if (binary_flat_[row_offset + x] == 1 && binary_flat_[prev_row_offset + x] == 1) {
        Union(parent, labels_flat_[row_offset + x], labels_flat_[prev_row_offset + x]);
      }
    }
  }

  std::vector<int> compacted(total_max_labels, 0);
  int current_id = 1;

  for (int stripe = 0; stripe < num_stripes; ++stripe) {
    for (int l = stripe_base_label[stripe]; l < stripe_max_used[stripe]; ++l) {
      int root = FindRoot(parent, l);
      if (compacted[root] == 0) {
        compacted[root] = current_id++;
      }
      compacted[l] = compacted[root];
    }
  }

  tbb::parallel_for(0, num_stripes, [&](int stripe) {
    int start_y = stripe_bounds[stripe];
    int end_y = stripe_bounds[stripe + 1];

    for (int y = start_y; y < end_y; ++y) {
      int row_offset = y * width_;
      for (int x = 0; x < width_; ++x) {
        int l = labels_flat_[row_offset + x];
        if (l != 0) {
          labels_flat_[row_offset + x] = compacted[l];
        }
      }
    }
  });

  return true;
}

bool MarinLMarkComponentsTBB::PostProcessingImpl() {
  labels_out_.clear();
  labels_out_.reserve(height_);

  for (int i = 0; i < height_; ++i) {
    labels_out_.emplace_back(width_);
  }

  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int> &r) {
    for (int y = r.begin(); y != r.end(); ++y) {
      const std::size_t row_offset = static_cast<std::size_t>(y) * width_;

      std::copy(labels_flat_.begin() + row_offset, labels_flat_.begin() + row_offset + width_, labels_out_[y].begin());
    }
  });

  auto &out = GetOutput();
  out.labels = std::move(labels_out_);

  return true;
}

}  // namespace marin_l_mark_components
