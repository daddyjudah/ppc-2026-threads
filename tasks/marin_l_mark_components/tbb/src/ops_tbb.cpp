#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/spin_mutex.h>

#include <algorithm>
#include <vector>

namespace marin_l_mark_components {

namespace {

int Find(std::vector<int> &p, int x) {
  while (p[x] != x) {
    p[x] = p[p[x]];
    x = p[x];
  }
  return x;
}

void Union(std::vector<int> &p, int a, int b) {
  int ra = Find(p, a);
  int rb = Find(p, b);

  if (ra != rb) {
    if (ra < rb) {
      p[rb] = ra;
    } else {
      p[ra] = rb;
    }
  }
}

}  // namespace

MarinLMarkComponentsTBB::MarinLMarkComponentsTBB(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkComponentsTBB::ValidationImpl() {
  const auto &img = GetInput().binary;
  if (img.empty() || img[0].empty()) {
    return false;
  }

  size_t w = img[0].size();
  for (auto &r : img) {
    if (r.size() != w) {
      return false;
    }
  }
  return true;
}

bool MarinLMarkComponentsTBB::PreProcessingImpl() {
  const auto &in = GetInput().binary;

  height_ = (int)in.size();
  width_ = (int)in[0].size();

  binary_.resize(height_ * width_);

  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int> &range) {
    for (int r = range.begin(); r < range.end(); ++r) {
      int base = r * width_;
      for (int c = 0; c < width_; c++) {
        binary_[base + c] = (uint8_t)in[r][c];
      }
    }
  });

  return true;
}

void MarinLMarkComponentsTBB::BuildRLE() {
  std::vector<int> row_counts(height_, 0);

  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int> &range) {
    for (int r = range.begin(); r < range.end(); ++r) {
      int c = 0, count = 0;
      int base = r * width_;

      while (c < width_) {
        while (c < width_ && binary_[base + c] == 0) {
          c++;
        }
        if (c >= width_) {
          break;
        }

        while (c < width_ && binary_[base + c] == 1) {
          c++;
        }
        count++;
      }

      row_counts[r] = count;
    }
  });

  offsets_.assign(height_ + 1, 0);
  for (int i = 0; i < height_; i++) {
    offsets_[i + 1] = offsets_[i] + row_counts[i];
  }

  int total_runs = offsets_[height_];

  runs_.resize(total_runs);
  parent_.resize(total_runs);
  locks_.resize(total_runs);

  tbb::parallel_for(tbb::blocked_range<int>(0, height_), [&](const tbb::blocked_range<int> &range) {
    for (int r = range.begin(); r < range.end(); ++r) {
      int c = 0;
      int idx = offsets_[r];
      int base = r * width_;

      while (c < width_) {
        while (c < width_ && binary_[base + c] == 0) {
          c++;
        }
        if (c >= width_) {
          break;
        }

        int start = c;
        while (c < width_ && binary_[base + c] == 1) {
          c++;
        }

        runs_[idx] = {r, start, c - 1, idx};
        idx++;
      }
    }
  });

  tbb::parallel_for(0, total_runs, [&](int i) { parent_[i] = i; });
}

void MarinLMarkComponentsTBB::MergeRuns() {
  tbb::parallel_for(1, height_, [&](int r) {
    int prev_begin = offsets_[r - 1];
    int prev_end = offsets_[r];

    int curr_begin = offsets_[r];
    int curr_end = offsets_[r + 1];

    int i = prev_begin;
    int j = curr_begin;

    while (i < prev_end && j < curr_end) {
      const Run &A = runs_[i];
      const Run &B = runs_[j];

      if (A.r < B.l) {
        i++;
      } else if (B.r < A.l) {
        j++;
      } else {
        int a = A.label;
        int b = B.label;

        if (a != b) {
          int lock_id = std::min(a, b);
          tbb::spin_mutex::scoped_lock lock(locks_[lock_id]);
          Union(parent_, a, b);
        }

        if (A.r < B.r) {
          i++;
        } else {
          j++;
        }
      }
    }
  });
}

void MarinLMarkComponentsTBB::ExpandToImage() {
  labels_flat_.assign(height_ * width_, 0);

  tbb::parallel_for(tbb::blocked_range<int>(0, (int)runs_.size()), [&](const tbb::blocked_range<int> &range) {
    for (int i = range.begin(); i < range.end(); ++i) {
      const Run &run = runs_[i];

      int root = Find(parent_, run.label);
      int label = root + 1;

      int base = run.row * width_;
      for (int c = run.l; c <= run.r; c++) {
        labels_flat_[base + c] = label;
      }
    }
  });
}

bool MarinLMarkComponentsTBB::RunImpl() {
  BuildRLE();
  MergeRuns();
  ExpandToImage();
  return true;
}

bool MarinLMarkComponentsTBB::PostProcessingImpl() {
  Labels out(height_, std::vector<int>(width_));

  tbb::parallel_for(0, height_, [&](int r) {
    int base = r * width_;
    for (int c = 0; c < width_; c++) {
      out[r][c] = labels_flat_[base + c];
    }
  });

  GetOutput().labels = out;
  return true;
}

}  // namespace marin_l_mark_components
