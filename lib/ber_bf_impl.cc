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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ber_bf_impl.h"
#include <gnuradio/io_signature.h>
#include <math.h>
#include <volk/volk.h>

namespace gr {
namespace learning {

ber_bf::sptr ber_bf::make(bool test_mode, int berminerrors, float ber_limit,
                          int bitsperbyte) {
  return gnuradio::get_initial_sptr(
      new ber_bf_impl(test_mode, berminerrors, ber_limit, bitsperbyte));
}

/*
 * The private constructor
 */
ber_bf_impl::ber_bf_impl(bool test_mode, int berminerrors, float ber_limit,
                         int bitsperbyte)
    : gr::block("ber_bf", io_signature::make(2, 2, sizeof(unsigned char)),
                io_signature::make(1, 1, sizeof(float))),
      d_total_errors(0), d_total(0), d_test_mode(test_mode),
      d_berminerrors(berminerrors), d_ber_limit(ber_limit),
      d_bitsperbyte(bitsperbyte) {}

inline float ber_bf_impl::calculate_log_ber() const {
  return log10(((double)d_total_errors) / (d_total * d_bitsperbyte));
}

inline void ber_bf_impl::update_counters(const int items,
                                         const unsigned char *inbuffer0,
                                         const unsigned char *inbuffer1) {
  uint32_t ret;
  for (int i = 0; i < items; i++) {
    volk_32u_popcnt(&ret, static_cast<uint32_t>(inbuffer0[i] ^ inbuffer1[i]));
    d_total_errors += ret;
  }
  d_total += items;
}

float ber_bf_impl::reset_counters() {
  float ber = calculate_log_ber();
  d_total_errors = d_total = 0;
  return ber;
}

/*
 * Our virtual destructor.
 */
ber_bf_impl::~ber_bf_impl() {}

int ber_bf_impl::general_work(int noutput_items, gr_vector_int &ninput_items,
                              gr_vector_const_void_star &input_items,
                              gr_vector_void_star &output_items) {
  unsigned char *inbuffer0 = (unsigned char *)input_items[0];
  unsigned char *inbuffer1 = (unsigned char *)input_items[1];
  float *outbuffer = (float *)output_items[0];

  int items =
      ninput_items[0] <= ninput_items[1] ? ninput_items[0] : ninput_items[1];

  if (d_test_mode) {
    if (d_total_errors >= d_berminerrors) {
      return WORK_DONE;
    } else {
      if (items > 0) {
        update_counters(items, inbuffer0, inbuffer1);
      }
      consume_each(items);

      if (d_total_errors >= d_berminerrors) {
        outbuffer[0] = calculate_log_ber();
        d_logger->info("    {} over {} --> {}", d_total_errors,
                       (d_total * d_bitsperbyte), outbuffer[0]);
        return 1;
      }
      // check for total_errors to prevent early shutdown at high SNR
      // simulations
      else if (calculate_log_ber() < d_ber_limit && d_total_errors > 0) {
        d_logger->info("    Min. BER limit reached");
        outbuffer[0] = d_ber_limit;
        d_total_errors = d_berminerrors + 1;
        return 1;
      } else {
        return 0;
      }
    }
  } else { // streaming mode
    if (items > 0) {
      update_counters(items, inbuffer0, inbuffer1);
      outbuffer[0] = calculate_log_ber();
      consume_each(items);
      return 1;
    } else {
      return 0;
    }
  }
}

} /* namespace learning */
} /* namespace gr */
