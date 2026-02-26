#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace marin_l_mark_of_comp_bin_im_seq {

using Image = std::vector<std::vector<int>>;

using InType = Image;
using OutType = Image;
using TestType = std::tuple<Image, Image>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace marin_l_mark_of_comp_bin_im_seq
