#include "marin_l_mark_of_comp_bin_im_seq/seq/include/ops_seq.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stack>
#include <utility>
#include <vector>

#include "marin_l_mark_of_comp_bin_im_seq/common/include/common.hpp"

namespace marin_l_mark_of_comp_bin_im_seq {

MarinLMarkOfCompBinImSEQ::MarinLMarkOfCompBinImSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MarinLMarkOfCompBinImSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.empty() || input[0].empty()) {
    return false;
  }

  const std::size_t cols = input[0].size();

  return std::ranges::all_of(input, [cols](const auto &row) { return row.size() == cols; });
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
  st.emplace(x, y);

  constexpr std::array<std::pair<int, int>, 4> kDirections{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

  while (!st.empty()) {
    auto [cx, cy] = st.top();
    st.pop();

    for (const auto &[dx, dy] : kDirections) {
      int nx = cx + dx;
      int ny = cy + dy;

      if (nx >= 0 && ny >= 0 && static_cast<std::size_t>(nx) < rows_ && static_cast<std::size_t>(ny) < cols_ &&
          image_[nx][ny] == 1) {
        image_[nx][ny] = current_label_;
        st.emplace(nx, ny);
      }
    }
  }
}

bool MarinLMarkOfCompBinImSEQ::RunImpl() {
  current_label_ = 2;

  for (std::size_t i = 0; i < rows_; ++i) {
    for (std::size_t j = 0; j < cols_; ++j) {
      if (image_[i][j] == 1) {
        DFS(i, j);
        current_label_++;
      }
    }
  }

  for (std::size_t i = 0; i < rows_; ++i) {
    for (std::size_t j = 0; j < cols_; ++j) {
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
