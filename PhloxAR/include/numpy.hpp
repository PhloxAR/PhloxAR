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

#ifndef __PHLOX_NUMPY_HPP__
#define __PHLOX_NUMPY_HPP__

#include <complex>
#include <Python.h>
#include <numpy/ndarrayobject.h>

namespace numpy {

  template <typename T>
  inline npy_intp dtype_code();

#define DECLARE_DTYPE_CODE(type, constant) \
  template <> inline npy_intp dtype_code<type>() {return constant;} \
  template <> inline npy_intp dtype_code<const type>() {return constant;} \
  template <> inline npy_intp dtype_code<volatile type>() {return constant;} \
  template <> inline npy_intp dtype_code<volatile const type>() {return constant;}

  DECLARE_DTYPE_CODE(bool, NPY_BOOL)
  DECLARE_DTYPE_CODE(float, NPY_FLOAT)
  DECLARE_DTYPE_CODE(char, NPY_BYTE)
  DECLARE_DTYPE_CODE(unsigned char, NPY_UBYTE)
  DECLARE_DTYPE_CODE(short, NPY_SHORT)
  DECLARE_DTYPE_CODE(unsigned short, NPY_USHORT)
  DECLARE_DTYPE_CODE(int, NPY_INT)
  DECLARE_DTYPE_CODE(long, NPY_LONG)
  DECLARE_DTYPE_CODE(unsigned long, NPY_ULONG)
  DECLARE_DTYPE_CODE(long long, NPY_LONGLONG)
  DECLARE_DTYPE_CODE(unsigned long long, NPY_ULONGLONG)
  DECLARE_DTYPE_CODE(double, NPY_DOUBLE)
#if defined(NPY_FLOAT128)
  DECLARE_DTYPE_CODE(npy_float128, NPY_FLOAT128)
#endif  // NPY_FLOAT128
  DECLARE_DTYPE_CODE(std::complex<float>, NPY_CFLOAT)
  DECLARE_DTYPE_CODE(std::complex<double>, NPY_CDOUBLE)
  DECLARE_DTYPE_CODE(unsigned int, NPY_UINT)

  template <typename T>
  bool check_type(PyArrayObject* o) {
    return PyArray_EquivTypenums(PyArray_TYPE(o), dtype_code<T>());
  }

  template <typename T>
  bool check_type(PyObject* o) {
    return check_type<T>(reinterpret_cast<PyArrayObject*>(o));
  }

  template <typename T>
  struct no_ptr {
    typedef T type;
  };

  template <typename T>
  struct no_ptr<T*> {
    typedef T type;
  };

  template <typename T>
  struct no_ptr<const T*> {
    typedef T type;
  };

  template <typename T>
  T ndarray_cast(PyArrayObject* o) {
    assert(check_type<typename no_ptr<T>::type>(o));
    assert(PyArray_ISALINGNED(o));
    void* as_voidp = PyArray_DATA(o);
    return const_cast<T>(static_cast<T>(as_voidp));
  }

  template <typename T>
  T ndarray_cast(PyObject* po) {
    assert(PyArray_Check(po));
    return ndarray_cast<T>((PyArrayObject*)po);
  }

}

#endif  // __PHLOX_NUMPY_HPP__
