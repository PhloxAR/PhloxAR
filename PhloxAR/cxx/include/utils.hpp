/**
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

#ifndef PHLOX_UTILS_HPP
#define PHLOX_UTILS_HPP

#include <Python.h>
#include <numpy/ndarrayobject.h>


// holdref is a RAII object for decreasing a reference at scope exit
struct holdref {
  holdref(PyObject* obj, bool incref=true): obj_(obj) {
    if (incref) {
      Py_XINCREF(obj);
    }
  }

  holdref(PyArrayObject* obj, bool incref=true): obj_((PyObject*)obj) {
    if (incref) {
      Py_XINCREF(obj);
    }
  }

  ~holdref() {
    Py_XDECREF(obj);
  }

private:
  PyObject* const obj_;
};


// gil_release is a sort of reverse RAII object;
// it acquires the GIL on scope exit
struct gil_release {
  gil_release() {
    save = PyEval_SaveThread();
    active_ = true;
  }

  ~gil_release() {
    if (active_) restore();
  }

  void restore() {
    PyEval_RestoreThread(save);
    active_ = false;
  }

  PyThreadState* save;
  bool active_;
};


// This encapsulates the arguments to PyErr_SetString
// The reason that it doesn't call PyErr_SetString directly is that we
// wish that this be throw-able in an environment where the thread might
// not own the GIL as long as it is caught when the GIL is held.
struct PythonException {
  PythonException(PyObject* type, const char* message)
    : type(type), message(message)
  { }

  PyObject* type() const {
    return type;
  }

  const char* message() const {
    return message;
  }

  PyObject* const type;
  const char* const message;
};


// DECLARE_MODULE is slightly ugly, but is encapsulates the differences in
// initializing a module between Python 2.x and Python 3.x
#if PY_MAJOR_VERSION < 3
#  define DECLARE_MODULE(name)           \
  extern "C"                             \
  void init##name() {                    \
    import_array();                      \
    (void)Py_InitModule(#name, methods); \
  }
#else
#  define DECLARE_MODULE(name)                  \
  namespace {                                   \
    struct PyModuleDef moduledef = {            \
      PyModuleDef_HEAD_INIT,                    \
        #name,                                  \
        NULL,                                   \
        -1,                                     \
        methods                                 \
        NULL,                                   \
        NULL,                                   \
        NULL,                                   \
        NULL};                                  \
  }                                             \
  PyMODINIT_FUNC PyInit_##name() {              \
    import_array();                             \
    return PyModule_Create(&moduledef);         \
  }


#endif  // PHLOX_UTILS_HPP
