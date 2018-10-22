#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2018 Cyrille Morin.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

import numpy as np
from gnuradio import gr
import pmt

class packet_header(gr.basic_block):
    """Insert a header containing a packet number based on the value of a received tag
    We assume that the tag is at the beginning of the preamble.
    The header will be inserted after that preamble"""

    def __init__(self, header_byte_length=2, preamble_byte_length = 5, tag_name = "step", payload_byte_length = 10):
        """arguments to this function show up as parameters in GRC"""
        gr.basic_block.__init__(
            self,
            name='Packet number inserter',   # will show up in GRC
            in_sig=[np.int8],
            out_sig=[np.int8]
        )

        self.header_byte_length = header_byte_length
        self.preamble_byte_length = preamble_byte_length
        self.tag_name = tag_name
        self.payload_byte_length = payload_byte_length
        self.packet_length = preamble_byte_length + preamble_byte_length + payload_byte_length
        self.set_relative_rate((self.packet_length)/(preamble_byte_length + payload_byte_length))
        self.set_output_multiple(self.packet_length *2)
        self.set_tag_propagation_policy(0)

    def general_work(self, input_items, output_items):
        """example: multiply with constant"""
        tags = self.get_tags_in_window(0, 0 , len(input_items[0]), pmt.intern(self.tag_name))
        read = self.nitems_read(0)
        written = self.nitems_written(0)

        output_position = 0
        for tag in tags:
            input_tag_position = tag.offset - read

            if output_position > (len(output_items[0]) - self.packet_length):
                self.consume(0, input_tag_position)
                return output_position

            if input_tag_position > len(input_items[0]) - (self.preamble_byte_length + self.payload_byte_length):
                #print("Not enough reading space, returning")
                self.consume(0, input_tag_position)
                return output_position
            tag_offset = output_position + written
            self.add_item_tag(0, tag_offset, tag.key, tag.value, tag.srcid)

            #Copy preamble
            output_items[0][output_position: output_position+ self.preamble_byte_length] = input_items[0][input_tag_position: input_tag_position + self.preamble_byte_length]
            output_position += self.preamble_byte_length

            #Insert header
            for i in range(self.header_byte_length):
                index = np.uint32( pmt.to_python(tag.value) )
                output_items[0][output_position + i :output_position + i + 1] = index % (2**(8*(i+1))) / (2**(8*(i)))
            tag_offset = output_position + written
            self.add_item_tag(0, tag_offset, pmt.intern("header_start"), pmt.to_pmt(0))
            output_position += self.header_byte_length

            #Copy payload
            output_items[0][output_position:output_position + self.payload_byte_length] = input_items[0][input_tag_position + self.preamble_byte_length: input_tag_position + self.preamble_byte_length + self.payload_byte_length]
            output_position += self.payload_byte_length


        return output_position
