/**
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

  template <typename T>
  double csum_rect(numpy::aligned_array<T> integral, int y, int x,
                   const int dy, const int dx, int h, int w) {
    int y0 = y + dy - h/2;
    int x0 = x + dx - w/2;
    int y1 = y0 + h;
    int x1 = x0 + w;

    return sum_rect(integral, y0, x0, x1, y1);
  }

  double harr_x(const integral_image_type& integral, int y, int x, const int w) {
    const double left = sum_rect(integral, y - w/2, x - w/2, y - w/2 + w, x);
    const double right = sum_rect(integral, y - w/2, x, y - w/2 + w, x - w/2 + w);
    return left - right;
  }

  double harr_y(const integral_image_type& integral, int y, int x, const int w) {
    const double top = sum_rect(integral, y - w/2, x - w/2, y, x - w/2 + w);
    const double bottom = sum_rect(integral, y, x - w/2, y - w/2 + w, x - w/2 + w);
    return top - bottom;
  }

  int round(double f) {
    if (f > 0)
      return int(f + .5);

    return int(f - .5);
  }

  int get_border_size(const int octave, const int nr_intervals) {
    const double lobe_size = std::pow(2.0, octave + 1.0) * (nr_intervals + 1) + 1;
    const double filter_size = 3 * lobe_size;

    const int bs = static_cast<int>(std::ceil(filter_size / 2.0));

    return bs;
  }

  int get_step_size(const int initial_step_size, const int octave) {
    return initial_step_size * static_cast<int>(std::pow(2.0, double(octave)) + 0.5);
  }

  struct hessian_pyramid {
    typedef std::vector<numpy::aligned_array<double> > pyramid_type;
    pyramid_type  pyr;

    double get_laplacian(int o, int i, int r, int c) const {
      return pyr[o].at(i, r, c) < 0 ? -1. : +1.;
    }

    double get_value(int o, int i, int r, int c) const {
      return std::abs(pyr[o].at(i, r, c));
    }

    int nr_intervals() const { return pyr[0].dim(0); }
    int nr_octave() const { return pyr.size(); }
    int nr(const int o) const { return pyr[o].dim(1); }
    int nc(const int o) cosnt { return pyr[o].dim(2); }
  };

  inline bool is_maximum_in_region(const hessian_pyramid& pyr, const int o,
                                   const int i, const int r, const int c) {
    // First check if this point is near the edge of the octave
    // If it is then we say it isn't a maximum as these points are
    // not as reliable.
    if (i <= 0 || i+1 >= pyr.nr_intervals())
      return false;

    assert(r > 0);
    assert(c > 0);

    const double val = pyr.get_value(o, i, r, c);

    // now check if there are any bigger values around this guy
    for (int ii = i-1; ii <= i+1; ++ii) {
      for (int rr = r-1; rr <= r+1; ++rr) {
        for (int cc = c-1; cc <= c+1; ++cc) {
          if (pyr.get_value(o, ii, rr, cc) > val)
            return false;
        }
      }
    }

    return true;
  }

  const double pi = 3.141592635898;

  struct double_v2 {
    double_v2() {
      data_[0] = 0.;
      data_[1] = 0.;
    }

    double_v2(double y, double x) {
      data_[0] = y;
      data_[1] = x;
    }

    double& y() { return data_[0]; }
    double& x() { return data_[1]; }

    double y() { return data_[0]; }
    double x() { return data_[1]; }

    double angle() const { return std::atan2(data_[1], data_[0]); }
    double norm2() const { return data_[0] * data_[0] + data_[1] * data_[1]; }

    double_v2 abs() const {
      return double_v2(std::abs(data_[0]), std::abs(data_[1]));
    }

    void clear() {
      data_[0] = 0.;
      data_[1] = 0.;
    }

    double_v2& operator += (const double_v2& rhs) {
      data_[0] += rhs.data_[0];
      data_[1] += rhs.data_[1];
      return *this;
    }

    double_v2& operator -= (const double_v2& rhs) {
      data_[0] -= rhs.data_[0];
      data_[1] -= rhs.data_[1];
      return *this;
    }

  private:
    double data_[2];
  };

  inline bool operator < (const double_v2& lhs, const double_v2& rhs) {
    return (lhs.y() == rhs.y()) ? (lhs.x() < rhs.x()) : (lhs.y() < rhs.y());
  }

  struct interest_point {
    interest_point()
        : scale(0), score(0), laplacian(0)
    { }

    double& y() { return center_.y(); }
    double& x() { return center_.x(); }

    double y() const { return center_.y(); }
    double x() const { return center_.x(); }

    double_v2& center() { return center_; }
    const double_v2& center() const { return center_; }

    double_v2 center_;
    double scale;
    double score;
    double laplacian;

    bool operator < (const interest_point& p) const { return score < p.score; }

    static const size_t ndoubles = 2 + 3;

    void dump(double out[ndoubles]) const {
      out[0] = center_.y();
      out[1] = center_.x();
      out[2] = scale;
      out[3] = score;
      out[4] = laplacian;
    }

    static interest_point load(const double in[ndoubles]) {
      interest_point res;
      res.center_.y() = in[0];
      res.center_.x() = in[1];
      res.scale = in[2];
      res.score = in[3];
      res.laplacian = in[4];
      return res;
    }
  };

  inline const interest_point iterpolate_point(const hessian_pyramid& pyr,
                                               const int o, const int i,
                                               const int r, const int c,
                                               const int initial_step_size) {
    // TODO
  }

  void get_interest_points(const hessian_pyramid& pyr, double threshold,
                           std::vector<interest_point>& result_points,
                           const int initial_step_size) {
    // TODO
  }

  template <typename T>
  void build_pyramid(numpy::aligned_array<T> integral,
                     hessian_pyramid& hpyramid,
                     const int nr_octaves, const int nr_intervals,
                     const int initial_step_size) {
    // TODO
  }

  template <typename T>
  void integral(numpy::aligned_array<T> array) {
    // TODO
  }

  struct surf_point {
    // TODO
  };

  inline double gaussian(cosnt double x, const double y, const double sigma) {
    return 1.0/(sigma*sigma*2*pi) * std::exp(-(x*x + y*y) / (2*sig*sig));
  }

  inline bool between_angles(const double a1, double a) {
    // TODO
  }

  double compute_dominant_angle(const integral_image_type& img,
                                const double_v2& center, const double scale) {
    // TODO
  }

  inline double_v2 rotate_point(const double_v2& p, const double sin_angle,
                                const double cos_angle) {
    // TODO
  }

  void compute_surf_descriptor(const integral_image_type& img,
                               double_v2 center, const double scale,
                               const double angle, double des[64]) {
    // TODO
  }

  std::vector<surf_point> compute_descriptors(const integral_image_type& int_img,
                                              const std::vector<interest_point>& points,
                                              const int max_points) {
    // TODO
  }

  template<typename T>
  std::vector<surf_point> get_surf_points(const numpy::aligned_array<T>& int_img,
                                          const int nr_octaves,
                                          const int nr_intervals,
                                          const int initial_step_size,
                                          const float threshold,
                                          const int max_points) {
    // TODO
  }

  PyObject* py_surf(PyObject* self, PyObject* args) {
    // TODO
  }

  PyObject* py_descriptors(PyObject* self, PyObject* args) {
    // TODO
  }

  PyObject* py_interest_points(PyObject* self, PyObject* args) {
    // TODO
  }

  PyObject* py_pyramid(PyObject* self, PyObject* args) {
    // TODO
  }

  PyObject* py_integral(PyObject* self, PyObject* args) {
    // TODO
  }

  PyObject* py_sum_rect(PyObject* self, PyObject* args) {
    // TODO
  }

  PyMethodDef methods[] = {
      {"integral",(PyCFunction)py_integral, METH_VARARGS, NULL},
      {"pyramid",(PyCFunction)py_pyramid, METH_VARARGS, NULL},
      {"interest_points",(PyCFunction)py_interest_points, METH_VARARGS, NULL},
      {"sum_rect",(PyCFunction)py_sum_rect, METH_VARARGS, NULL},
      {"descriptors",(PyCFunction)py_descriptors, METH_VARARGS, NULL},
      {"surf",(PyCFunction)py_surf, METH_VARARGS, NULL},
      {NULL, NULL,0,NULL},
  };

};  // namespace

DELARE_MODULE(surf);