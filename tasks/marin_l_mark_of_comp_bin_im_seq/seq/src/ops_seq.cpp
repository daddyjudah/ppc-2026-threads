#include "marin_l_mark_of_comp_bin_im_seq/seq/include/ops_seq.hpp"

#include <numeric>
#include <stack>
#include <vector>

#include "marin_l_mark_of_comp_bin_im_seq/common/include/common.hpp"
#include "util/include/util.hpp"

namespace marin_l_mark_of_comp_bin_im_seq {

MarinLMarkOfCompBinImSEQ::MarinLMarkOfCompBinImSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkOfCompBinImSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.empty()) {
    return false;
  }

  if (input[0].empty()) {
    return false;
  }

  const size_t cols = input[0].size();

  for (const auto &row : input) {
    if (row.size() != cols) {
      return false;
    }
  }

  return true;
}

bool MarinLMarkOfCompBinImSEQ::PreProcessingImpl() {
  image_ = GetInput();
  rows_ = image_.size();
  cols_ = image_[0].size();
  GetOutput().assign(rows_, std::vector<int>(cols_, 0));
  return true;
}

void MarinLMarkOfCompBinImSEQ::DFS(int x, int y) {
  std::stack<std::pair<int, int>> st;
  image_[x][y] = current_label_;
  st.push({x, y});

  while (!st.empty()) {
    auto [cx, cy] = st.top();
    st.pop();

    int dx[] = {1, -1, 0, 0};
    int dy[] = {0, 0, 1, -1};

    for (int i = 0; i < 4; ++i) {
      int nx = cx + dx[i];
      int ny = cy + dy[i];

      if (nx >= 0 && nx < rows_ && ny >= 0 && ny < cols_ && image_[nx][ny] == 1) {
        image_[nx][ny] = current_label_;
        st.push({nx, ny});
      }
    }
  }
}

bool MarinLMarkOfCompBinImSEQ::RunImpl() {
  current_label_ = 2;

  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (image_[i][j] == 1) {
        DFS(i, j);
        current_label_++;
      }
    }
  }

  for (int i = 0; i < rows_; ++i) {
    for (int j = 0; j < cols_; ++j) {
      if (image_[i][j] > 0) {
        image_[i][j] -= 1;
      }
    }
  }

  GetOutput() = image_;
  return true;
}

bool MarinLMarkOfCompBinImSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace marin_l_mark_of_comp_bin_im_seq
