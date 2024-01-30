/* -*- c++ -*- */
/*
 * Copyright 2019 gr-learning author.
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

#ifndef INCLUDED_LEARNING_BER_BF_IMPL_H
#define INCLUDED_LEARNING_BER_BF_IMPL_H

#include <gnuradio/learning/ber_bf.h>

namespace gr {
  namespace learning {

    class ber_bf_impl : public ber_bf
    {
     private:
       long d_total_errors;
       long d_total;
       bool d_test_mode;
       int d_berminerrors;
       float d_ber_limit;
       int d_bitsperbyte;

       inline float calculate_log_ber() const;
       inline void update_counters(const int items,
                                   const unsigned char* inbuffer0,
                                   const unsigned char* inbuffer1);

     public:
      ber_bf_impl(bool d_test_mode = false, int berminerrors = 100, float ber_limit = -7.0, int bitsperbyte = 8);
      ~ber_bf_impl();

      int general_work(int noutput_items,
           gr_vector_int &ninput_items,
           gr_vector_const_void_star &input_items,
           gr_vector_void_star &output_items);

      long total_errors() { return d_total_errors; };
      float reset_counters();

    };

  } // namespace learning
} // namespace gr

#endif /* INCLUDED_LEARNING_BER_BF_IMPL_H */
