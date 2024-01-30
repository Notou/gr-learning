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

class header_reader(gr.sync_block):
    """Reads a packet number header written by a Packet number inserter block and writes a tag with this number
    The position of the header is received by a tag.
    We write the new tag at the beginning of the preamble"""

    def __init__(self, header_byte_length=2, preamble_byte_length = 5, tag_name = "header_start", new_tag_name= "step", threshold = 10):
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Packet number reader',
            in_sig=[np.int8],
            out_sig=[np.int8]
        )

        self.header_byte_length = header_byte_length
        self.preamble_byte_length = preamble_byte_length
        self.tag_name = tag_name
        self.threshold = threshold
        self.new_tag_name = new_tag_name

        self.packet_length = preamble_byte_length + header_byte_length
        self.set_relative_rate(1)
        self.set_output_multiple(self.packet_length *2)
        self.set_history((preamble_byte_length + self.header_byte_length)*8)
        self.previous_packet = np.uint32(0)

    def work(self, input_items, output_items):
        """example: multiply with constant"""
        read = self.nitems_read(0)
        history_read = max(read - self.history()+1,0)
        written = self.nitems_written(0)

        start_pos = (self.preamble_byte_length*8)
        process_length = len(output_items[0])
        stop_pos = start_pos + process_length -1
        # We don't want to read a tag if we cannot access the header that comes after
        tags = self.get_tags_in_window(0, start_pos , stop_pos , pmt.intern(self.tag_name))

        for tag in tags:
            position = tag.offset - read + (self.history()-1) #Position of the tag relative to the input array
            if position > stop_pos:
                #print("Outside upper limit")
                break
            char = ""
            for byte in reversed(range(self.header_byte_length)):
                for bit in (range(8)):
                    char += str(input_items[0][position + 8*byte + bit])

            #Detect outliers in the sequence of packets
            if abs(self.previous_packet-int(char, 2)) < self.threshold:
                self.add_item_tag(0, tag.offset - (self.preamble_byte_length)*8, pmt.intern(self.new_tag_name), pmt.to_pmt(np.int64(int(char, 2))))
                #print("Packet number disparity!!!!!!!!:", self.previous_packet)

            self.previous_packet = int(char, 2)

        output_items[0][:] = input_items[0][-len(output_items[0]):]
        return process_length
