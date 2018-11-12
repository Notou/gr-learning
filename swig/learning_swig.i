/* -*- c++ -*- */

#define LEARNING_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
%include "learning_swig_doc.i"

%{
#include "learning/packet_isolator_c.h"
#include "learning/correlator.h"
%}


%include "learning/packet_isolator_c.h"
GR_SWIG_BLOCK_MAGIC2(learning, packet_isolator_c);
%include "learning/correlator.h"
GR_SWIG_BLOCK_MAGIC2(learning, correlator);
