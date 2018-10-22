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

import numpy
from gnuradio import gr
import pmt

class tag_numerotation(gr.sync_block):
    """
    Adds a tag to a stream every "Tag interval" samples.
    The value of the tag increases by one each time a tag is added.
    """

    def __init__(self, tag_name, interval, io_type = None):
        print(io_type)
        sig_type = None
        if io_type == "cc":
            sig_type = numpy.complex64
        if io_type == "bb":
            sig_type = numpy.int8
        if io_type == "ff":
            sig_type = numpy.float32
        if io_type == None:
            return
        gr.sync_block.__init__(self,
            name="Tag numerotation",
            in_sig=[sig_type],
            out_sig=[sig_type])
        self.tag_name = tag_name
        self.interval = interval
        self.index = numpy.uint32(0)
        self.offset = 0
        self.id = pmt.intern('tag_numerator')


    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        
        out[:] = in0
        input_size = len(input_items[0])
        for i in range(self.offset, input_size, self.interval):
            self.add_item_tag( 0, self.interval * self.index, pmt.intern(self.tag_name), pmt.to_pmt(numpy.int64(self.index)), self.id)
            self.index +=1
        self.offset = (input_size - self.offset) % self.interval
        return len(output_items[0])
