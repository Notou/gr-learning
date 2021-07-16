/* -*- c++ -*- */
/*
 * Copyright 2019 Cyrille Morin.
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

#ifndef INCLUDED_LEARNING_ALIGN_IMPL_H
#define INCLUDED_LEARNING_ALIGN_IMPL_H

#include <learning/align.h>

namespace gr {
  namespace learning {

    class align_impl : public align
    {
     private:
      // Nothing to declare in this block.
      char* d_tag_name;
      int d_frame_size;
      int d_label_frame_size;
      int d_user_symbols;
      int d_vec_len;

      int d_previous_packet = -1;
      int d_last_packet_loss = -1;

     public:
      align_impl(char* tag_name, int frame_size, int label_frame_size, int vec_len, int user_symbols);
      ~align_impl();

      // Where all the action really happens
      void forecast (int noutput_items, gr_vector_int &ninput_items_required);

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

    };

  } // namespace learning
} // namespace gr

#endif /* INCLUDED_LEARNING_ALIGN_IMPL_H */
