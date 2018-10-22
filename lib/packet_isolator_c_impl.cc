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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include <pmt/pmt.h>
#include "packet_isolator_c_impl.h"

namespace gr {
  namespace learning {

    packet_isolator_c::sptr
    packet_isolator_c::make(int payload_length, int preamble_length, int lookup_window, char* tag_name)
    {
      return gnuradio::get_initial_sptr
        (new packet_isolator_c_impl(payload_length, preamble_length, lookup_window, tag_name));
    }

    /*
     * The private constructor
     */
    packet_isolator_c_impl::packet_isolator_c_impl(int payload_length, int preamble_length, int lookup_window, char* tag_name)
      : gr::block("packet_isolator_c",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex)))
    {
      d_transmit_preamble = 0;
      d_payload_length = payload_length;
      d_preamble_length = preamble_length;
      d_lookup_window = lookup_window;
      d_pack_length = (d_transmit_preamble * d_preamble_length) + d_payload_length; //=d_payload_length if d_transmit_preamble=0
      d_tag_name = tag_name;
      set_tag_propagation_policy(TPP_DONT);
      set_history(std::max(d_preamble_length, d_lookup_window)+d_payload_length);
      set_min_noutput_items(d_pack_length *5);


    }

    /*
     * Our virtual destructor.
     */
    packet_isolator_c_impl::~packet_isolator_c_impl()
    {
    }

    void
    packet_isolator_c_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
          ninput_items_required[0] = int(noutput_items);
    }

    int
    packet_isolator_c_impl::general_work (int noutput_items,
                       gr_vector_int &ninput_items, //Does not include the history
                       gr_vector_const_void_star &input_items,
                       gr_vector_void_star &output_items)
    {
      const gr_complex *in = (const gr_complex *) input_items[0];
      gr_complex *out = (gr_complex *) output_items[0];

      read = nitems_read(0);
      history_read = (read - history());
      max_offset = read + ninput_items[0];
      // printf("%s %d\n", "Work starting. Input samples: ", ninput_items[0]);
      // printf("%s %d %s %d\n", "History: ", history(), "History read:", history_read);
      // printf("%s %d %s %d\n", "Item read: ", read, "length:", d_pack_length);
      //We don't want to process tags so close to the end of the input array that we don't have access to the payload samples
      max_tag_offset = max_offset - d_payload_length - d_preamble_length - d_lookup_window;

      written = nitems_written(0);
      output_number = 0; //Number of samples we are outputting


      std::vector<tag_t> tags;
      get_tags_in_range(tags, 0,  history_read , max_offset, pmt::intern(d_tag_name));

      //std::vector<tag_t> relevant_tags;
      //get_tags_in_range(relevant_tags, 0,  history_read + d_preamble_length , max_offset);
      tag_t relevant_tag;
      int i=0;
      // printf("%s %d\n", "Tag number: ", tags.size());
      //printf("%s %d\n", "Total tag number: ", relevant_tags.size());
      //printf("%s %d to %d\n", "Search range: ", history_read + d_preamble_length, max_offset);
      for(int i=0; i < tags.size();i++){
        if(tags[i].offset > max_tag_offset){ //Just in case
          break;
        }
        relevant_tag = tags[i];
        for(int j=1; i+j < tags.size();j++){
          if(tags[i+j].offset > tags[i].offset + d_lookup_window){
            break;
          }

          if (pmt::to_double(relevant_tag.value) < pmt::to_double(tags[i+j].value)){
            relevant_tag = tags[i+j];
          }
        }


        position = relevant_tag.offset - history_read; //Position of the tag relative to the input array

        output_number += d_pack_length;
        if (output_number > noutput_items){ //We don't want to write more than the size of the output array
          consume_each(position);
          // printf("%s %d %s %d %s %d %s %d\n", "Work finishing early. Output samples: ", output_number, "consumed:", position-history()+1, "Output number:", noutput_items, "position:", position);
          return output_number - d_pack_length;
        }
        //Forward a slice of inputs
        memcpy(out + i*d_pack_length, in + position+(d_preamble_length*(!d_transmit_preamble)), d_pack_length * sizeof(gr_complex));
        //out[i*d_pack_length:(i+1)*d_pack_length] = in[position-d_preamble_length:position+d_payload_length]

        //Write a tag at the beginning of the payload
        tag_offset = written + i*d_pack_length + (relevant_tag.offset - history_read - (position+(d_preamble_length*d_transmit_preamble)));
        add_item_tag(0, tag_offset, pmt::intern("header_start"), pmt::from_long(0));



      }


      // for(int i = 0; i < relevant_tags.size(); i++){
      //
      //   //Custom tag propagation scheme.
      //   // tags_to_propagate = self.get_tags_in_range(0, tags[i].offset-d_preamble_length, tags[i].offset+d_payload_length)
      //   // for tag in tags_to_propagate{
      //   //   tag_offset = written + i*d_pack_length + (tag.offset - history_read - (position-d_preamble_length))
      //   //   add_item_tag(0, tag_offset, tag.key, tag.value, tag.srcid)
      //   // }
      // }

      // Tell runtime system how many input items we consumed on
      // each input stream.
      consume_each(ninput_items[0]);

      // Tell runtime system how many output items we produced.
      // printf("%s %d\n", "Work finishing. Output samples: ", output_number);
      return output_number;
    }

  } /* namespace learning */
} /* namespace gr */


/*  TO USE WHEN THE CORRELATION IS TAGGED AT THE END OF THE VECTOR!
int
packet_isolator_c_impl::general_work (int noutput_items,
                   gr_vector_int &ninput_items, //Does not include the history
                   gr_vector_const_void_star &input_items,
                   gr_vector_void_star &output_items)
{
  const gr_complex *in = (const gr_complex *) input_items[0];
  gr_complex *out = (gr_complex *) output_items[0];

  read = nitems_read(0);
  history_read = (read - history());
  max_offset = read + ninput_items[0];
  // printf("%s %d\n", "Work starting. Input samples: ", ninput_items[0]);
  // printf("%s %d %s %d\n", "History: ", history(), "History read:", history_read);
  // printf("%s %d %s %d\n", "Item read: ", read, "length:", d_pack_length);
  //We don't want to process tags so close to the end of the input array that we don't have access to the payload samples
  max_tag_offset = max_offset - d_payload_length - d_lookup_window;

  written = nitems_written(0);
  output_number = 0; //Number of samples we are outputting


  std::vector<tag_t> tags;
  get_tags_in_range(tags, 0,  history_read + d_preamble_length , max_offset, pmt::intern(d_tag_name));

  //std::vector<tag_t> relevant_tags;
  //get_tags_in_range(relevant_tags, 0,  history_read + d_preamble_length , max_offset);
  tag_t relevant_tag;
  int i=0;
  // printf("%s %d\n", "Tag number: ", tags.size());
  //printf("%s %d\n", "Total tag number: ", relevant_tags.size());
  //printf("%s %d to %d\n", "Search range: ", history_read + d_preamble_length, max_offset);
  for(int i=0; i < tags.size();i++){
    if(tags[i].offset > max_tag_offset){ //Just in case
      break;
    }
    relevant_tag = tags[i];
    for(int j=1; i+j < tags.size();j++){
      if(tags[i+j].offset > tags[i].offset + d_lookup_window){
        break;
      }

      if (pmt::to_double(relevant_tag.value) < pmt::to_double(tags[i+j].value)){
        relevant_tag = tags[i+j];
      }
    }


    position = relevant_tag.offset - history_read; //Position of the tag relative to the input array

    output_number += d_pack_length;
    if (output_number > noutput_items){ //We don't want to write more than the size of the output array
      consume_each(position-d_preamble_length);
      // printf("%s %d %s %d %s %d %s %d\n", "Work finishing early. Output samples: ", output_number, "consumed:", position-history()+1, "Output number:", noutput_items, "position:", position);
      return output_number - d_pack_length;
    }
    //Forward a slice of inputs
    memcpy(out + i*d_pack_length, in + position-(d_preamble_length*d_transmit_preamble), d_pack_length * sizeof(gr_complex));
    //out[i*d_pack_length:(i+1)*d_pack_length] = in[position-d_preamble_length:position+d_payload_length]

    //Write a tag at the beginning of the payload
    tag_offset = written + i*d_pack_length + (relevant_tag.offset - history_read - (position-(d_preamble_length*d_transmit_preamble)));
    add_item_tag(0, tag_offset, pmt::intern("header_start"), pmt::from_long(0));



  }


  // for(int i = 0; i < relevant_tags.size(); i++){
  //
  //   //Custom tag propagation scheme.
  //   // tags_to_propagate = self.get_tags_in_range(0, tags[i].offset-d_preamble_length, tags[i].offset+d_payload_length)
  //   // for tag in tags_to_propagate{
  //   //   tag_offset = written + i*d_pack_length + (tag.offset - history_read - (position-d_preamble_length))
  //   //   add_item_tag(0, tag_offset, tag.key, tag.value, tag.srcid)
  //   // }
  // }

  // Tell runtime system how many input items we consumed on
  // each input stream.
  consume_each(ninput_items[0]);

  // Tell runtime system how many output items we produced.
  // printf("%s %d\n", "Work finishing. Output samples: ", output_number);
  return output_number;
}*/
