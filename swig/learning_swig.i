/* -*- c++ -*- */

#define LEARNING_API

%include "gnuradio.i"			// the common stuff

//load generated python docstrings
%include "learning_swig_doc.i"

%{
#include "learning/packet_isolator_c.h"
#include "learning/correlator.h"
#include "learning/head.h"
#include "learning/ber_bf.h"
#include "learning/align.h"
%}


%include "learning/packet_isolator_c.h"
GR_SWIG_BLOCK_MAGIC2(learning, packet_isolator_c);
%include "learning/correlator.h"
GR_SWIG_BLOCK_MAGIC2(learning, correlator);
%include "learning/head.h"
GR_SWIG_BLOCK_MAGIC2(learning, head);
%include "learning/ber_bf.h"
GR_SWIG_BLOCK_MAGIC2(learning, ber_bf);
%include "learning/align.h"
GR_SWIG_BLOCK_MAGIC2(learning, align);
