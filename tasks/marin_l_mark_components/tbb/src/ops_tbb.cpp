#include "marin_l_mark_components/tbb/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

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
    p[rb] = ra;
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

  tbb::parallel_for(0, height_, [&](int r) {
    for (int c = 0; c < width_; c++) {
      binary_[r * width_ + c] = (uint8_t)in[r][c];
    }
  });

  return true;
}

void MarinLMarkComponentsTBB::BuildRLE() {
  std::vector<std::vector<Run>> per_row(height_);

  tbb::parallel_for(0, height_, [&](int r) {
    int c = 0;
    while (c < width_) {
      while (c < width_ && binary_[r * width_ + c] == 0) {
        c++;
      }
      if (c >= width_) {
        break;
      }

      int start = c;
      while (c < width_ && binary_[r * width_ + c] == 1) {
        c++;
      }

      per_row[r].push_back({r, start, c - 1, -1});
    }
  });

  size_t total_runs = 0;
  for (const auto &row : per_row) {
    total_runs += row.size();
  }

  runs_.clear();
  runs_.reserve(total_runs);

  row_runs_.clear();
  row_runs_.resize(height_);

  for (int r = 0; r < height_; r++) {
    for (auto &run : per_row[r]) {
      int id = static_cast<int>(runs_.size());
      run.label = id;

      runs_.push_back(run);
      row_runs_[r].push_back(id);
    }
  }

  parent_.resize(runs_.size());
  for (size_t i = 0; i < runs_.size(); i++) {
    parent_[i] = static_cast<int>(i);
  }
}

void MarinLMarkComponentsTBB::MergeRuns() {
  for (int r = 1; r < height_; r++) {
    const auto &prev = row_runs_[r - 1];
    const auto &curr = row_runs_[r];

    int i = 0, j = 0;

    while (i < (int)prev.size() && j < (int)curr.size()) {
      auto &A = runs_[prev[i]];
      auto &B = runs_[curr[j]];

      if (A.r < B.l) {
        i++;
      } else if (B.r < A.l) {
        j++;
      } else {
        Union(parent_, A.label, B.label);

        if (A.r < B.r) {
          i++;
        } else {
          j++;
        }
      }
    }
  }
}

void MarinLMarkComponentsTBB::Flatten() {
  for (size_t i = 0; i < parent_.size(); i++) {
    parent_[i] = Find(parent_, i);
  }
}

void MarinLMarkComponentsTBB::ExpandToImage() {
  labels_flat_.assign(height_ * width_, 0);

  tbb::parallel_for(0, (int)runs_.size(), [&](int i) {
    auto &run = runs_[i];
    int label = parent_[run.label];

    for (int c = run.l; c <= run.r; c++) {
      labels_flat_[run.row * width_ + c] = label + 1;
    }
  });
}

bool MarinLMarkComponentsTBB::RunImpl() {
  BuildRLE();
  MergeRuns();
  Flatten();
  ExpandToImage();
  return true;
}

bool MarinLMarkComponentsTBB::PostProcessingImpl() {
  Labels out(height_, std::vector<int>(width_));

  tbb::parallel_for(0, height_, [&](int r) {
    for (int c = 0; c < width_; c++) {
      out[r][c] = labels_flat_[r * width_ + c];
    }
  });

  GetOutput().labels = out;
  return true;
}

}  // namespace marin_l_mark_components
