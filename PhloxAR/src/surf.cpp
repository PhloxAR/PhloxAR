/*
 * This file contains much code ported from dlib
 * DLIB is under the following copyright and license:
 *     Copyright (c) 2009 Davis E. King (davis@dlib.net)
 *     License: Boost Software License
 *
 * Copyright 2016(c) Matthias Y. Chen
 * <matthiasychen@gmail.com/matthias_cy@outlook.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "include/array.hpp"
#include "include/dispatch.hpp"
#include "include/utils.hpp"

#include <cmath>
#include <vector>
#include <limits>
#include <sstream>
#include <algorithm>

namespace {

  const char TypeErrorMsg[] = "Type not understood."
  " This is caused by either a direct call to surf (which is dangerous: types"
  " are not checked!) or a bug in surf.py.\n";

  /*
   * SURF: Speed-Up Robust Features
   * The implementation here is a port from DLIB, which is in turn influenced by
   * the very well documented OpenSURF library and its corresponding description
   * of how the fast Hessian algorithm functions: "Notes on the OpenSURF Library"
   * by
   */
  typedef numpy::aligned_array<double> integral_image_type;

  template <typename T>
  double sum_rect(numpy::aligned_array<T> integral, int y0, int x0, int y1, int x1) {
    y0 = std::max<int>(y0-1, 0);
    x0 = std::max<int>(x0-1, 0);
    y1 = std::min<int>(y1-1, integral.dim(0) - 1);
    x1 = std::min<int>(x1-1, integral.dim(1) - 1);

    const T A = integral.at(y0,x0);
    const T B = integral.at(y0,x1);
    const T C = integral.at(y1,x0);
    const T D = integral.at(y1,x1);

    // This expression, unlike equivalent alternatives,
    // has no overflows. (D > B) and (C > A) and (D-B) > (C-A)
    return (D - B) - (C - A);
  }

};