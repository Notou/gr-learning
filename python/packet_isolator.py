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

class packet_isolator(gr.basic_block):
    """
    This block forwards a number of input samples when a correlation tag is recieved.

    When a tag is recieved it will look forwards a bit and if another tag with a higher value is found it will only retain that one

    Preamble length: number of samples to forward before the tag
    Payload length: number of samples to forward after the tag
    Tag name: name of the tag to look for
    Lookup window: the number of samples in which to look for other tags with an higher value
    """
    def __init__(self, payload_length=0, preamble_length=80, lookup_window = 30, tag_name = "corr_est"):
        gr.basic_block.__init__(self,
            name="packet_isolator",
            in_sig=[np.complex64],
            out_sig=[np.complex64])

        self.payload_length = payload_length
        self.preamble_length = preamble_length
        self.lookup_window = lookup_window
        self.tag_name = tag_name
        self.set_tag_propagation_policy(0)
        self.set_history(max(self.preamble_length, self.lookup_window)+self.payload_length)
        self.set_output_multiple((self.preamble_length + self.payload_length) *2) #We would like to use set_min_output_items but it's not available in python yet
        self.pack_length = self.preamble_length+self.payload_length

    def forecast(self, noutput_items, ninput_items_required):
        # setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
                ninput_items_required[i] = int(noutput_items/2.0)+1

    def general_work(self, input_items, output_items):
        """ """
        read = self.nitems_read(0)
        history_read = max(read - self.history()+1,0)
        max_offset = history_read + len(input_items[0])

        #We don't want to process tags so close to the end of the input array that we don't have access to the payload samples
        max_tag_offset = max_offset - self.payload_length - self.lookup_window

        written = self.nitems_written(0)



        tags = self.get_tags_in_range(0,  history_read + self.preamble_length , max_offset, pmt.intern(self.tag_name))
        tags = list(tags) #So we can remove low valued tags from the list

        i=0
        while i < len(tags):
            if tags[i].offset > max_tag_offset: #Just in case
                tags.remove(tags[i])
                continue
            next_index=i+1
            while next_index < len(tags):
                if tags[next_index].offset > tags[i].offset + self.lookup_window:
                    break
                if pmt.to_python(tags[i].value) > pmt.to_python(tags[next_index].value):
                    tags.remove(tags[next_index])
                else:
                    tags.remove(tags[i])
            i += 1

        output_number = 0 #Number of samples we are outputting
        for i in range(len(tags)):
            position = tags[i].offset - history_read #Position of the tag relative to the input array

            output_number += self.pack_length
            if output_number > len(output_items[0]): #We don't want to write more than the size of the output array
                self.consume(0, position)
                return output_number

            #Forward a slice of inputs
            output_items[0][i*self.pack_length:(i+1)*self.pack_length] = input_items[0][position-self.preamble_length:position+self.payload_length]

            #Write a tag at the beginning of the payload
            tag_offset = written + i*self.pack_length + (tags[i].offset - history_read - (position-self.preamble_length))
            self.add_item_tag(0, tag_offset+20, pmt.intern("header_start"), pmt.to_pmt(0))

            #Custom tag propagation scheme.
            tags_to_propagate = self.get_tags_in_range(0, tags[i].offset-self.preamble_length, tags[i].offset+self.payload_length)
            for tag in tags_to_propagate:
                tag_offset = written + i*self.pack_length + (tag.offset - history_read - (position-self.preamble_length))
                self.add_item_tag(0, tag_offset, tag.key, tag.value, tag.srcid)

        self.consume(0, len(input_items[0])-(self.history()-1))
        return output_number
