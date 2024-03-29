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


#ifndef INCLUDED_LEARNING_PACKET_ISOLATOR_C_H
#define INCLUDED_LEARNING_PACKET_ISOLATOR_C_H

#include <gnuradio/learning/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace learning {

    /*!
     * \brief <+description of block+>
     * \ingroup learning
     *
     */
    class LEARNING_API packet_isolator_c : virtual public gr::block
    {
     public:
      typedef std::shared_ptr<packet_isolator_c> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of learning::packet_isolator_c.
       *
       * To avoid accidental use of raw pointers, learning::packet_isolator_c's
       * constructor is in a private implementation
       * class. learning::packet_isolator_c::make is the public interface for
       * creating new instances.
       */
      static sptr make(int payload_length=0, int preamble_length = 80, int lookup_window = 30, std::string tag_name = "corr_est");
    };

  } // namespace learning
} // namespace gr

#endif /* INCLUDED_LEARNING_PACKET_ISOLATOR_C_H */
