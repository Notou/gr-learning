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

#ifndef INCLUDED_LEARNING_ALIGN_H
#define INCLUDED_LEARNING_ALIGN_H

#include <learning/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace learning {

    /*!
     * \brief <+description of block+>
     * \ingroup learning
     *
     */
    class LEARNING_API align : virtual public gr::block
    {
     public:
      typedef boost::shared_ptr<align> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of learning::align.
       *
       * To avoid accidental use of raw pointers, learning::align's
       * constructor is in a private implementation
       * class. learning::align::make is the public interface for
       * creating new instances.
       */
      static sptr make(char* tag_name, int frame_size, int label_frame_size, int vec_len, int user_symbols);
    };

  } // namespace learning
} // namespace gr

#endif /* INCLUDED_LEARNING_ALIGN_H */
