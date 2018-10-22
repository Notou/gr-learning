/* -*- c++ -*- */
/*
 * Copyright 2018 Cyrille Morin.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_LEARNING_PACKET_ISOLATOR_C_IMPL_H
#define INCLUDED_LEARNING_PACKET_ISOLATOR_C_IMPL_H

#include <learning/packet_isolator_c.h>

namespace gr {
  namespace learning {

    class packet_isolator_c_impl : public packet_isolator_c
    {
     private:
       int d_payload_length;
       int d_preamble_length;
       int d_lookup_window;
       int d_pack_length;
       int d_transmit_preamble;
       char* d_tag_name;

       uint64_t read;
       uint64_t history_read;
       uint64_t max_offset;

       uint64_t max_tag_offset;

       uint64_t written;
       int output_number;
       uint64_t position;
       uint64_t tag_offset;

     public:
      packet_isolator_c_impl(int payload_length, int preamble_length, int lookup_window, char* tag_name);
      ~packet_isolator_c_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);
    };

  } // namespace learning
} // namespace gr

#endif /* INCLUDED_LEARNING_PACKET_ISOLATOR_C_IMPL_H */
