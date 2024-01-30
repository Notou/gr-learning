# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT learning_sources)
    message(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT learning_sources)

########################################################################
# Check for pygccxml
########################################################################
gr_python_check_module_raw("pygccxml" "import pygccxml" PYGCCXML_FOUND)

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND learning_python_files 
align_python.cc
ber_bf_python.cc
correlator_python.cc
head_python.cc
packet_isolator_c_python.cc
python_bindings.cc)

gr_pybind_make_oot(learning ../../.. gr::learning "${learning_python_files}")

# copy bindings extension for use in QA test module
add_custom_command(
    TARGET learning_python
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:learning_python>
            ${PROJECT_BINARY_DIR}/test_modules/gnuradio/learning/)

install(
    TARGETS learning_python
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/learning
    COMPONENT pythonapi)
