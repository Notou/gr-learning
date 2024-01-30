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

#ifndef INCLUDED_LEARNING_BER_BF_H
#define INCLUDED_LEARNING_BER_BF_H

#include <gnuradio/learning/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace learning {

    /*!
     * \brief <+description of block+>
     * \ingroup learning
     *
     * \details
     *
     * This block measures the bit error rate between two streams of
     * packed data. It compares the bits of each streams and counts
     * the number of incorrect bits between them. It outputs the log
     * of the bit error rate, so a value of -X is 10^{-X} bit errors.
     *
     * When the \p mode is set to false (default), it is in streaming
     * mode. This means that the output is constantly producing the
     * current value of the BER. In this mode, there is a single
     * output BER calculation per chunk of bytes passed to it, so
     * there is no exact timing between calculations of BER. In this
     * mode, the other two parameters to the constructor are ignored.
     *
     * When \p mode is true, the block is in test mode. This mode is
     * used in the ber_curve_gen example and for other offline
     * analysis of BER curves. Here, the block waits until at least \p
     * berminerrors are observed and then produces a BER
     * calculation. The parameter \p ber_limit helps make sure that
     * the simulation is controlled. If the BER calculation drops
     * below the \p ber_limit setting, the block will exit and simply
     * return the set limit; the real BER is therefore some amount
     * lower than this.
     *
     * Note that this block takes in data as packed bytes with 8-bits
     * per byte used. It outputs a stream of floats as the log-scale BER.
     */
    class LEARNING_API ber_bf : virtual public gr::block
    {
     public:
      typedef std::shared_ptr<ber_bf> sptr;

      /*!
       * Get total number of errors counter value.
       */
      virtual long total_errors() = 0;

      virtual float reset_counters() = 0;

      /*!
       * Calculate the BER between two streams of data.
       *
       * \param test_mode false for normal streaming mode (default);
       *        true for test mode.
       * \param berminerrors the block needs to observe this many
       *        errors before outputting a result. Only valid when
       *        test_mode=true.
       * \param ber_limit if the BER calculation falls below this
       *        limit, produce this value and exit. Only valid when
       *        test_mode=true.
       * \param bitsperbyte number of data bits per byte
       */
      static sptr make(bool test_mode = false, int berminerrors = 100, float ber_limit = -7.0, int bitsperbyte = 8);
    };

  } // namespace learning
} // namespace gr

#endif /* INCLUDED_LEARNING_BER_BF_H */
