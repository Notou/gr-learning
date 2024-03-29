/*
 * Copyright 2024 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually
 * edited  */
/* The following lines can be configured to regenerate this file during cmake */
/* If manual edits are made, the following tags should be modified accordingly.
 */
/* BINDTOOL_GEN_AUTOMATIC(0) */
/* BINDTOOL_USE_PYGCCXML(0) */
/* BINDTOOL_HEADER_FILE(ber_bf.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(0551ea820cd4ce8e8a882a428f148fed) */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <gnuradio/learning/ber_bf.h>
// pydoc.h is automatically generated in the build directory
#include <ber_bf_pydoc.h>

void bind_ber_bf(py::module &m) {

  using ber_bf = ::gr::learning::ber_bf;

  py::class_<ber_bf, gr::block, gr::basic_block, std::shared_ptr<ber_bf>>(
      m, "ber_bf", D(ber_bf))

      .def(py::init(&ber_bf::make), py::arg("test_mode") = false,
           py::arg("berminerrors") = 100, py::arg("ber_limit") = -7.,
           py::arg("bitsperbyte") = 8, D(ber_bf, make))

      .def("total_errors", &ber_bf::total_errors, D(ber_bf, total_errors))

      .def("reset_counters", &ber_bf::reset_counters, D(ber_bf, reset_counters))

      ;
}
