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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include <pmt/pmt.h>
#include "align_impl.h"

namespace gr {
  namespace learning {

    align::sptr
    align::make(char* tag_name, int frame_size, int label_frame_size, int vec_len, int user_symbols)
    {
      return gnuradio::get_initial_sptr
        (new align_impl(tag_name, frame_size, label_frame_size, vec_len, user_symbols));
    }


    /*
     * The private constructor
     */
    align_impl::align_impl(char* tag_name, int frame_size, int label_frame_size, int vec_len, int user_symbols)
      : gr::block("Align",
              gr::io_signature::make2(2, 2, sizeof(gr_complex)* vec_len, sizeof(char)),
              gr::io_signature::make2(2, 2, sizeof(gr_complex)* vec_len, sizeof(char))),
          d_frame_size(frame_size),
          d_label_frame_size(label_frame_size),
          d_tag_name(tag_name),
          d_vec_len(vec_len),
          d_user_symbols(user_symbols)
    {
      printf("%s\n", d_tag_name);
      d_tag_name = "packet_num";
      printf("%s\n", d_tag_name);
      printf("BLALALALALALALALALALALALAAAAAAAAAAAAAAAAAAALALALALAL!!!\n");
      set_tag_propagation_policy(TPP_DONT);
      set_output_multiple(d_frame_size);
      if (d_vec_len <= 1) {
        d_user_symbols = 1;
      }
    }

    /*
     * Our virtual destructor.
     */
    align_impl::~align_impl()
    {
    }

    void
    align_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = d_frame_size / d_user_symbols;
      ninput_items_required[1] = d_label_frame_size;
    }

    int
    align_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items,
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      // printf("Work called\n" );
      // I/O Variables
      const gr_complex *in0 = (const gr_complex *) input_items[0];
      const char *in1 = (const char *) input_items[1];
      gr_complex *out0 = (gr_complex *) output_items[0];
      char *out1 = (char *) output_items[1];

      // Indeces
      uint64_t read0 = nitems_read(0);
      uint64_t read1 = nitems_read(1);

      // Get tags
      std::vector<tag_t> tags_0;
      get_tags_in_window(tags_0, 0, 0, ninput_items[0], pmt::intern("packet_num"));
      std::vector<tag_t> tags_1;
      get_tags_in_window(tags_1, 1, 0, ninput_items[1], pmt::intern("packet_num"));

      // Variables init
      uint value0 = 0;
      uint value1 = -1;
      uint64_t offset0 = 0;
      uint64_t offset1 = 0;

      if (tags_0.size() == 0) {
        // printf("%s\n", "Align: We don't have any tag in stream 0");
        consume(0, ninput_items[0]);
        return 0;
      }
      if (tags_1.size() == 0) {
        // printf("%s\n", "Align: We don't have any tag in stream 1");
        consume(1, ninput_items[1]);
        return 0;
      }
      // printf("%s %d %d\n", "Align, number of packets in in0  and in1 buffer", tags_0.size(), tags_1.size());

      // for(int i=0; i < tags_0.size(); i++){
      //   printf("%s %d\n", "Input 0 tag at input ", pmt::to_uint64(tags_0[i].value));
      // }
      // for(int i=0; i < tags_1.size(); i++){
      //   printf("%s %d\n", "Input 1 tag at input ", pmt::to_uint64(tags_1[i].value));
      // }
      // std::vector<tag_t> tags;
      // get_tags_in_window(tags, 0, 0, ninput_items[0], pmt::intern(d_tag_name));
      // if (tags.size() > 0) {
      //   printf("%s %d\n", "There is thas amount of tags seen with the tag name: ", tags.size());
      // }

      // Find the tag in each input that has the same value as the other and is the lowest possible. We assume that the values are increasing
      for(int i=0; i < tags_0.size(); i++){
        value0 = pmt::to_uint64(tags_0[i].value);

        for(int j=0; j < tags_1.size(); j++){
          value1 = pmt::to_uint64(tags_1[j].value);
          offset1 = tags_1[j].offset;

          if (value0 <= value1) {
            break;
          }
          if (value0 > value1) {
            continue;
          }
        }
        if (value0 == value1) {
          offset0 = tags_0[i].offset;
          break;
        }
        if (value0 < value1) {
          continue;
        }
      }

      // printf("%s\n", "Work function called");
      // No correspondance between the streams
      if (value0 != value1) {
        // printf("%s %d %d\n", "No correspondance found between the streams:", value0, value1);
        if (value0 > value1) {
          consume(1, ninput_items[1]);
        } else {
          consume(0, ninput_items[0]);
        }
        return 0;
      }

      //Index of the selected tags relative to the input arays
      uint64_t position0 = offset0 - read0;
      uint64_t position1 = offset1 - read1;

      // Process a frame only if it has been fully received
      if (position0 + (d_frame_size/ d_user_symbols) > ninput_items[0]) {
        // printf("%s %d >= %d\n", "Stream 0 too late, consuming", position0 + d_frame_size, ninput_items[0]);
        consume(0, position0);
        return 0;
      }
      if (position1 + d_label_frame_size > ninput_items[1]) {
        // printf("%s\n", "Stream 1 too late, consuming");
        consume(1, position1);
        return 0;
      }

      // Detect packet loss
      if (value0 != d_previous_packet + 1) {
        // printf("%d %d\n", value0, d_previous_packet);
        printf("Packet lost: %d in a row. Value is: %d Packets since last loss: %d\n", value0-(d_previous_packet+1), value0, value0-d_last_packet_loss);
        d_last_packet_loss = value0;
      }

      if (noutput_items < d_frame_size) {
        printf("%s\n", "Output size not big enough" );
      }

      memcpy(out0, in0 + position0, (d_frame_size/ d_user_symbols)* d_vec_len *sizeof(gr_complex)); // The starting position should also be *d_vec_len, right?
      memcpy(out1, in1 + position1, d_label_frame_size*sizeof(char));

      add_item_tag(0, nitems_written(0), pmt::intern("packet_num"), pmt::from_uint64(value0));
      add_item_tag(1, nitems_written(1), pmt::intern("packet_num"), pmt::from_uint64(value0));
      d_previous_packet = value0;
      consume(0, position0 + (d_frame_size/ d_user_symbols));
      consume(1, position1 + d_label_frame_size);
      produce(0, (d_frame_size/ d_user_symbols));
      produce(1, d_label_frame_size);

      // Tell runtime system how many output items we produced.
      return 0;
    }

  } /* namespace learning */
} /* namespace gr */
