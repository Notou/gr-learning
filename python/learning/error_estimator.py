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

class error_estimator(gr.basic_block):
    """
    docstring for block error_estimator
    """
    def __init__(self, tag_name, frame_size, io_type = None, packet_loss_penalty = 10):
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

        gr.basic_block.__init__(self,
            name="error_estimator",
            in_sig=[sig_type, sig_type],
            out_sig=None)
        self.frame_size = frame_size
        self.tag_name = tag_name
        self.message_port_register_out(pmt.intern("error"))
        self.packet_loss_penalty = packet_loss_penalty
        if packet_loss_penalty < 0:
            print("WARN: Packet loss penalty is negative")
        self.previous_packet = -1

    def forecast(self, noutput_items, ninput_items_required):
        # setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
                ninput_items_required[i] = self.frame_size


    def general_work(self, input_items, output_items):
        if  len(input_items[0]) < self.frame_size: #Just in case
            print("Err input size: ", len(input_items[0]), " ", len(input_items[1]))
            return 0

        read0 = self.nitems_read(0)
        read1 = self.nitems_read(1)
        in0 = input_items[0]
        in1 = input_items[1]
        tags_0 = self.get_tags_in_window(0, 0, len(in0), pmt.intern("step"))
        tags_1 = self.get_tags_in_window(1, 0, len(in1), pmt.intern("step"))
        value0 = 0
        value1 = -1
        offset0 = 0
        offset1 = 0

        if len(tags_0) == 0:
            #print("We don't have any tag in stream 0")
            self.consume(0, len(in0))
            return 0
        if len(tags_1) == 0:
            #print("We don't have any tag in stream 1")
            self.consume(1, len(in1))
            return 0

        #Find the tag in each input that has the same value as the other and is the lowest possible. We assume that the values are increasing
        for tag0 in tags_0:
            value0 = pmt.to_python(tag0.value)

            for tag1 in tags_1:
                value1 = pmt.to_python(tag1.value)
                offset1 = tag1.offset

                if value0 <= value1:
                    break
                if value0 > value1:
                    continue
            if value0 == value1:
                offset0 = tag0.offset
                break
            if value0 < value1:
                continue

        if value0 != value1:
            #print("Cannot make correspondance between the two streams.", value0, value1)
            if value0 > value1:
                self.consume(1, len(in1))
            if value0 < value1:
                self.consume(0, len(in0))
            return 0

        #Index of the selected tags relative to the input arays
        position0 = offset0 - read0
        position1 = offset1 - read1

        #We process a frame only if it has been fully received
        if position0 + self.frame_size >= len(in0):
            #print("Stream 0 too late, consuming")
            self.consume(0, position0)
            return position0
        if position1 + self.frame_size >= len(in1):
            #print("Stream 1 too late, consuming")
            self.consume(1, position1)
            return position1

        #Detect packet packet loss
        if value0 != self.previous_packet+1:
            for i in range(self.previous_packet+1, value0):
                print(self.packet_loss_penalty, i)
                self.message_port_pub(pmt.intern("error"), pmt.to_pmt(self.packet_loss_penalty))


        diff = in0[position0:position0 + self.frame_size] - in1[position1:position1 + self.frame_size]
        norm = numpy.linalg.norm(diff, ord = 2)

        self.message_port_pub(pmt.intern("error"), pmt.to_pmt(norm.item()))
        print(norm, value0)

        self.previous_packet = value0
        self.consume(0, position0 + self.frame_size)
        self.consume(1, position1 + self.frame_size)

        return self.frame_size
