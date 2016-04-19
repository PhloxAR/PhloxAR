/*
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

#ifndef __PHLOX_ARRAY_HPP__
#define __PHLOX_ARRAY_HPP__

#include <iterator>
#include <algorithm>
#include <cstring>
#include <ostream>
#include <iostream>
#include <vector>
#include <cassert>

#include "numpy.hpp"

#ifndef __GUNC__
#define __PRETTY_FUNCTION__ ""
#endif

template <typename T>
struct filter_iterator;

namespace numpy {

  typedef npy_intp index_type;
  const unsigned index_type_number = NPY_INTP;

  struct position {
    position(): _nd(0) {}
    position(const npy_intp* pos, int nd): _nd(nd) {
      for (int i = 0; i != _nd; ++i) {
        _pos[i] = pos[i];
      }
    }

    npy_intp operator [](unsigned pos) const {
      return this->_pos[pos];
    }

    int ndim() const {
      return _nd;
    }

    int _nd;
    npy_intp _pos[NPY_MAXDIMS];

    bool operator ==(const position& rhs) {
      return !std::memcmp(this->_pos, rhs._pos, sizeof(this->_pos[0])*this->_nd);
    }

    bool operator !=(const position& rhs) {
      return !(*this == rhs);
    }

    static position from1(npy_intp p0) {
      position res;
      res._nd = 1;
      res._pos[0] = p0;
      return res;
    }

    static position from2(npy_intp p0, npy_intp p1) {
      position res;
      res._nd = 2;
      res.pos[0] = p0;
      res.pos[1] = p1;
      return res;
    }

    static position from3(npy_intp p0, npy_intp p1, npy_intp p2) {
      position res;
      res.nd_ = 3;
      res.position_[0] = p0;
      res.position_[1] = p1;
      res.position_[2] = p2;
      return res;
    }

  };

  inline position operator +=(const position& l, const position& r) {
    assert(l._nd == r._nd);
    for (int i = 0; i != l._nd; ++i) {
      l._pos[i] += r._pos[i];
    }
    return l;
  }

  inline position operator +(const position& l, const position& r) {
    assert(l._nd == r._nd);
    position res = l;
    res += r;
    return res;
  }

  inline position operator -=(const position& l, const position& r) {
    assert(l._nd == r._nd);
    for (int i = 0; i != l._nd; ++i) {
      l._pos[i] -= r._pos[i];
    }
    return l;
  }

  inline position operator -(const position& l, const position& r) {
    assert(l._nd == r._nd);
    position res = l;
    res -= r;
    return res;
  }

  inline bool operator ==(const position& l, const position& r) {
    if (l._nd != r._nd) return false;
    for (int i = 0; i != r._nd; ++i) {
      if (l._pos[i] != r._pos[i])
        return false
    }
    return true;
  }

  inline bool operator != (const position& l, const position& r) {
    return !(l == r);
  }

  template <typename T>
  T& operator <<(T& out, const numpy::position& p) {
    out << "[";
    for (int d = 0; d != p._nd; ++d) {
      out << p._pos[d] << ":";
    }
    out << "]";
    return out;
  }

  struct position_vector {
  public:
    position_vector(const int size): _size(size) {}

    position operator [](const unsigned i) const {
      assert(i*_size < _data.size());
      position res(&_data[i*_size], _size);
      return res;
    }

    void push_back(const poistion& pos) {
      assert(pos.ndim() == _size);
      for (int d = 0; d != size; ++d)
        _data.push_back(pos[d]);
    }

    unsigned size() const { return _data.size() / _size; }
    bool empty() cosnt { return _data.empty(); }

  protected:
    const int _size;
    std::vector<npy_intp> _data;
  };

  struct position_stack: position_vector {
  public:
    position_stack(const int size): position_vector(size) {}

    position pop() {
      assert(!empty());
      position res(&_data[_data.size() - _size], _size);
      _data.erase(_data.end() - size, _data.end());
      return res;
    }

    void push(const position& pos) {
      this->push_back(pos);
    }
  };

  struct position_queue: position_vector {
  public:
    position_queue(const int size): position_vector(size), _next(0) {}

    void push(const position& pos) { this->push_back(pos); }
    unsigned size() const { return this->position_vector::size() - _next; }
    bool empty() const { return (*this)[_next]; }

    position top() const {
      return (*this)[_next];
    }

    void pop() {
      ++_next;
      if (_next == 512) {
        _data.erase(_data.begin(), _data.begin() + _next * _size);
        _next = 0;
      }
    }

    position top_pop() {
      position pos = this->top();
      this->pop();
      return pos;
    }

  protected:
    unsigned _next;
  };

  template <typename BaseType>
  struct iterator_data: std::iterator<std::forward_iterator_tag, BaseType> {
    friend struct::filter_iterator<BaseType>;

  protected:
#ifdef _GLIBCXX_DEBUG
    const PyArrayObject* base;
#endif
    BaseType* _data;
    int _steps[NPY_MAXDIMS];
    int _dims[NPY_MAXDIMS];
    ::numpy::position _pos;

  public:
    iterator_base(PyArrayObject* array) {
#ifdef _GLIBCXX_DEBUG
      base = array;
#endif
      assert(PyArray_Check(array));
      int nd = PyArray_NDIM(array);
      _pos._nd = nd;
      _data = ndarray_cast<BaseType*>(array);
      std::fill(_pos._pos, _pos._pos+nd, 0);

      unsigned cummul = 0;

      for (int i = 0; i != _pos._nd; ++i) {
        _dims[i] = PyArray_DIM(array, nd-i-1);
        _steps[i] = PyArray_STRIDE(array, nd-i-1) / sizeof(BaseType) - cummul;
        cummul *= PyArray_DIM(array, nd-i-1);
        cummul += _steps[i]*PyArray_DIM(array, nd-i-1);
      }
    }

    iterator_base& operator ++() {
      for (int i = 0; i != _pos._nd; ++i) {
        _data += _steps[i];
        ++pos_.pos_[i];
        if (_pos._pos[i] != _dims[i]) {
          return *this;
        }
        _pos._pos[i] = 0;
      }
      return *this;
    }

    int index(unsigned i) const { return index_rev(_pos._nd-i-1); }
    int index_rev(unsigned i) const { return _pos._pos[i]; }
    npy_intp dimension(unsigned i) const { return dimension_rev(_pos._nd-i-1); }
    npy_intp dimension_rev(unsigned i) const { return _dims[i]; }

    bool operator ==(const iterator_base& rhs) { return this->_pos == rhs->_pos; }
    bool operator !=(const iterator_base& rhs) { return !(*this == rhs); }

    ::numpy::position position() const {
      ::numpy::position res = _pos;
      std::reverse(res._pos, res._pos+res._nd);
      return res;
    }

    friend inline std::ostream& operator <<(std::ostream& out, const iterator_base& iter) {
      return out << "I {" << iter._pos << "}";
    }

    bool is_valid() const {
#ifdef _GLIBCXX_DEBUG
      ::numpy::position p = this->position();
      for (int r = 0; r != p.ndim(); ++r) {
        if (p[r] < 0 || p[r] >= PyArray_DIM(base, r)) return false;
      }
#endif
      return true;
    }
  };

  template <typename T>
  struct no_cast {
    typedef T type;
  };

  template <typename T>
  struct no_const<const T> {
    typedef T type;
  };

  template <typename BaseType>
  class iterator_type: public iterator_base<BaseType> {
  public:
    iterator_type(PyArrayObject* array): iterator_base<BaseType>(array) {}

    BaseType operator *() const {
      assert(this->is_valid());
      typename no_const<BaseType>::type res;
      std::memcpy(&res, this->_data, sizeof(res));
      return res;
    }
  };

  template <typename BaseType>
  class aligned_iterator_type: public iterator_base<BaseType> {
  public:
    aligned_iterator_type(PyArrayObject* array): iterator_base<BaseType>(array) {
      assert(PyArray_ISALIGNED(array));
    }

    BaseType& operator *() const {
      assert(this->is_valid());
      return *this->_data;
    }
  };

  template <typename BaseType>
  class array_base {
  protected:
    PyArrayObject* _array;

    void* raw_data(const position& pos) const {
      assert(this->validposition(pos));
      return PyArray_GetPtr(_array, const_cast<npy_intp*>(pos._pos));
    }

  public:
    array_base(const array_base<BaseType>& other): array_(other._array) {
      if (sizeof(BaseType) != PyArray_ITEMSIZE(_array)) {
        std::cerr << "Phlox:" << __PRETTY_FUNCTION__ << " mix up of array types"
                  << " [using size " << sizeof(BaseType) << " expecting "
                  << PyArray_ITEMSIZE(_array) << "]\n";
        assert(false);
      }
      Py_INCREF(_array);
    }

    array_base(PyArrayObject* array) :_array(array) {
      if (sizeof(BaseType) != PyArray_ITEMSIZE(_array)) {
        std::cerr << "Phlox:" << __PRETTY_FUNCTION__ << " mix up of array types"
                  << " [using size " <<sizeof(BaseType) << " expecting "
                  << PyArray_ITEMSIZE(_array) << "]\n";
        assert(false);
      }
      Py_INCREF(_array);
    }

    ~array_base() {
      Py_XDECREF(_array);
    }

    array_base<BaseType>& operator =(const BaseType& rhs) {
      array_base<BaseType> na(rhs);
      this->swap(na);
    }
    void swap(array_base<BaseType>& other) {
      std::swap(this->_array, other._array);
    }

    index_type size() const { return PyArray_SIZE(_array); }
    index_type size(index_type i) const {
      return this->dim(i);
    }
    index_type ndim() const { return PyArray_NDIM(_array); }
    index_type ndims() const { return PyArray_NDIM(_array); }
    index_type dim(index_type i) const {
      assert(i < this->ndims());
      return PyArray_DIM(_array, i);
    }

    unsigned stride(unsigned i) const {
      return PyArray_STRIDE(_array, i)/sizeof(BaseType);
    }

    PyArrayObject* raw_array() const { return _array; }
    void* raw_data() const { return PyArray_DATA(_array); }
    const npy_intp* raw_dims() const { return PyArray_DIMS(_array); }

    bool validposition(const position& pos) const {
      if (ndims() != pos._nd) {
        return false;
      }
      for (int i=0; i != pos._nd; ++i) {
        if (pos[i] < 0 || pos[i] >= this->dim(i))
          return false;
      }
      return true;
    }

    bool is_aligned() const {
      return PyArray_ISALIGNED(_array);
    }

    BaseType at(const position& pos) const {
      BaseType val;
      void* datap = raw_data(pos);
      std::memcpy(&val, datap, sizeof(BaseType));
      return val;
    }

    npy_intp raw_stride(npy_intp i) const {
      return PyArray_STRIDE(this->_array, i);
    }
  };

  template<typename BaseType>
  struct array : public array_base<BaseType> {
  public:
    array(PyArrayObject* array): array_base<BaseType>(array) {}

    typedef iterator_type<BaseType> iterator;
    typedef iterator_type<const BaseType> const_iterator;

    iterator begin() {
      return iterator(this->_array);
    }

    const_iterator begin() const {
      return const_iterator(this->_array);
    }

    iterator end() {
      iterator res = begin();
      for (unsigned i = 0, N = this->size(); i!= N; ++i) {
        ++res;
      }
      return res;
    }

    const_iterator end() const {
      const_iterator res = begin();
      for (unsigned i = 0, N = this->size(); i!= N; ++i) {
        ++res;
      }
      return res;
    }
  };

  template <typename BaseType>
  struct aligned_array: public array_base<BaseType> {
  private:
    bool _is_carray;

  public:
    aligned_array(PyArrayObject* array)
      : array_base<BaseType>(array) ,_is_carray(PyArray_ISCARRAY(array)) {
      assert(PyArray_ISALIGNED(array));
    }

    aligned_array(const aligned_array<BaseType>& other)
      :array_base<BaseType>(other)
      ,_is_carray(other._is_carray)
    { }

    typedef aligned_iterator_type<BaseType> iterator;
    typedef aligned_iterator_type<const BaseType> const_iterator;

    const_iterator begin() const {
      return const_iterator(this->_array);
    }

    iterator begin() {
      return iterator(this->_array);
    }

    iterator end() {
      iterator res = begin();
      for (unsigned i = 0, N = this->size(); i!= N; ++i) {
        ++res;
      }
      return res;
    }

    npy_intp stride(npy_intp i) const {
      return this->raw_stride(i)/sizeof(BaseType);
    }

    bool is_carray() const { return _is_carray; }

    BaseType* data() {
      return reinterpret_cast<BaseType*>(PyArray_DATA(this->_array));
    }

    BaseType* data(npy_intp p0) {
      assert(p0 < this->dim(0));
      return reinterpret_cast<BaseType*>(PyArray_GETPTR1(this->_array, p0));
    }

    BaseType* data(npy_intp p0, npy_intp p1) {
      assert(p0 < this->dim(0));
      assert(p1 < this->dim(1));
      return reinterpret_cast<BaseType*>(PyArray_GETPTR2(this->_array, p0, p1));
    }

    BaseType* data(npy_intp p0, npy_intp p1, npy_intp p2) {
      assert(p0 < this->dim(0));
      assert(p1 < this->dim(1));
      assert(p2 < this->dim(2));
      return reinterpret_cast<BaseType*>(PyArray_GETPTR3(this->_array, p0, p1, p2));
    }

    const BaseType* data() const {
      return reinterpret_cast<const BaseType*>(PyArray_DATA(this->_array));
    }

    const BaseType* data(const position& pos) const {
      return reinterpret_cast<const BaseType*>(this->raw_data(pos));
    }

    const BaseType* data(npy_intp p0) const {
      assert(p0 < this->dim(0));
      return reinterpret_cast<const BaseType*>(PyArray_GETPTR1(this->_array, p0));
    }

    const BaseType* data(npy_intp p0, npy_intp p1) const {
      assert(p0 < this->dim(0));
      assert(p1 < this->dim(1));
      return reinterpret_cast<const BaseType*>(PyArray_GETPTR2(this->_array, p0, p1));
    }

    const BaseType* data(npy_intp p0, npy_intp p1, npy_intp p2) const {
      assert(p0 < this->dim(0));
      assert(p1 < this->dim(1));
      assert(p2 < this->dim(2));
      return reinterpret_cast<const BaseType*>(PyArray_GETPTR3(this->_array, p0, p1, p2));
    }

    BaseType* data(const position& pos) {
      return reinterpret_cast<BaseType*>(this->raw_data(pos));
    }

    BaseType& at(const position& pos) {
      return *data(pos);
    }

    BaseType at(const position& pos) const {
      return *data(pos);
    }

    BaseType& at_flat(npy_intp p) {
      if (_is_carray) return data()[p];

      BaseType* base = this->data();
      for (int d = this->ndims() - 1; d >= 0; --d) {
        int c = (p % this->dim(d));
        p /= this->dim(d-1);
        base += c * this->stride(d);
      }
      return *base;
    }

    BaseType at_flat(npy_intp p) const {
      return const_cast< aligned_array<BaseType>* >(this)->at_flat(p);
    }

    int pos_to_flat(const position& pos) const {
      npy_intp res = 0;
      int cummul = 1;
      for (int d = this->ndims() -1; d >= 0; --d) {
        res += pos._pos[d] * cummul;
        cummul *= this->dim(d);
      }
      return res;
    }

    numpy::position flat_to_pos(int p) const {
      numpy::position res;
      res.nd_ = this->ndims();
      for (int d = this->ndims() - 1; d >= 0; --d) {
        res.position_[d] = (p % this->dim(d));
        p /= this->dim(d);
      }
      if (p) res._pos[0] += p * this->dim(0);
      return res;
    }

    BaseType at(int p0) const {
      return *static_cast<BaseType*>(PyArray_GETPTR1(this->_array, p0));
    }

    BaseType& at(int p0) {
      assert(p0 < this->dim(0));
      return *static_cast<BaseType*>(PyArray_GETPTR1(this->_array, p0));
    }

    BaseType at(int p0, int p1) const {
      assert(p0 < this->dim(0));
      assert(p1 < this->dim(1));
      return *static_cast<BaseType*>(PyArray_GETPTR2(this->_array, p0, p1));
    }

    BaseType& at(int p0, int p1) {
      assert(p0 < this->dim(0));
      assert(p1 < this->dim(1));
      return *static_cast<BaseType*>(PyArray_GETPTR2(this->_array, p0, p1));
    }

    BaseType at(int p0, int p1, int p2) const {
      assert(p0 < this->dim(0));
      assert(p1 < this->dim(1));
      assert(p2 < this->dim(2));
      return *static_cast<BaseType*>(PyArray_GETPTR3(this->_array, p0, p1, p2));
    }

    BaseType& at(int p0, int p1, int p2) {
      assert(p0 < this->dim(0));
      assert(p1 < this->dim(1));
      assert(p2 < this->dim(2));
      return *static_cast<BaseType*>(PyArray_GETPTR3(this->_array, p0, p1, p2));
    }
  };

  template <typename BaseType>
  aligned_array<BaseType> new_array(const npy_intp ndims, const npy_intp* dims) {
    assert(ndims < NPY_MAXDIMS);
    for (int d = 0; d != ndims; ++d) {
      assert(dims[d] >= 0);
    }

    aligned_array<BaseType> res(reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(ndims, const_cast<npy_intp*>(dims), dtype_code<BaseType>())));
    // SimpleNew returns an object with count = 1
    // constructing an array sets it to 2.
    Py_XDECREF(res.raw_array());
    return res;
  }

  template <typename BaseType>
  aligned_array<BaseType> new_array(int s0) {
    npy_intp dim = s0;
    return new_array<BaseType>(1, &dim);
  }

  template <typename BaseType>
  aligned_array<BaseType> new_array(int s0, int s1) {
    npy_intp dims[2];
    dims[0] = s0;
    dims[1] = s1;
    return new_array<BaseType>(2, dims);
  }

  template <typename BaseType>
  aligned_array<BaseType> new_array(int s0, int s1, int s2) {
    npy_intp dims[3];
    dims[0] = s0;
    dims[1] = s1;
    dims[2] = s2;
    return new_array<BaseType>(3, dims);
  }

  template <typename BaseType>
  aligned_array<BaseType> array_like(const array_base<BaseType>& orig) {
    PyArrayObject* array = orig.raw_array();
    return aligned_array<BaseType>((PyArrayObject*)PyArray_SimpleNew(PyArray_NDIM(array), PyArray_DIMS(array), PyArray_TYPE(array)));
  }

  inline
  bool same_shape(PyArrayObject* a, PyArrayObject* b) {
    if (PyArray_NDIM(a) != PyArray_NDIM(b)) return false;
    const int n = PyArray_NDIM(a);
    for (int i = 0; i != n; ++i) {
      if (PyArray_DIM(a, i) != PyArray_DIM(b, i)) return false;
    }
    return true;
  }

  inline bool are_arrays(PyArrayObject* a) { return PyArray_Check(a); }

  inline bool are_arrays(PyArrayObject* a, PyArrayObject* b) {
    return PyArray_Check(a) && PyArray_Check(b);
  }

  inline bool are_arrays(PyArrayObject* a, PyArrayObject* b, PyArrayObject* c) {
    return PyArray_Check(a) && PyArray_Check(b) && PyArray_Check(c);
  }

  inline bool arrays_of_same_shape_type(PyArrayObject* a, PyArrayObject* b) {
    return are_arrays(a,b) &&
      PyArray_EquivTypenums(PyArray_TYPE(a), PyArray_TYPE(b)) &&
      same_shape(a,b);
  }

  inline bool equiv_typenums(PyArrayObject* a, PyArrayObject* b) {
    return PyArray_EquivTypenums(PyArray_TYPE(a), PyArray_TYPE(b));
  }

  inline bool equiv_typenums(PyArrayObject* a, PyArrayObject* b, PyArrayObject* c) {
    return equiv_typenums(a, b) && equiv_typenums(a, c);
  }

  inline bool equiv_typenums(PyArrayObject* a, PyArrayObject* b, PyArrayObject* c, PyArrayObject* d) {
    return equiv_typenums(a, b) && equiv_typenums(a, c) && equiv_typenums(a, d);
  }

  inline bool is_carray(PyArrayObject* a) { return are_arrays(a) && PyArray_ISCARRAY(a); }

}  // namespace numpy

#endif  // __PHLOX_ARRAY_HPP__
