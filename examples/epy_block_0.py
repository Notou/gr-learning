"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import pmt

class alignement(gr.basic_block):  # other base classes are basic_block, decim_block, interp_block
    """Align two stream according to a packet number set in a tag"""

    def __init__(self, tag_name="packet_num", frame_size=384):

        gr.basic_block.__init__(self,
            name="Align",
            in_sig=[np.complex64, np.int8],
            out_sig=[np.complex64, np.int8])
        self.frame_size = frame_size
        self.tag_name = tag_name
        self.previous_packet = -1
        self.last_packet_loss = -1
        self.set_tag_propagation_policy(0)
        self.set_output_multiple(self.frame_size)

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
        out0 = output_items[0]
        out1 = output_items[1]
        tags_0 = self.get_tags_in_window(0, 0, len(in0), pmt.intern(self.tag_name))
        tags_1 = self.get_tags_in_window(1, 0, len(in1), pmt.intern(self.tag_name))
        value0 = 0
        value1 = -1
        offset0 = 0
        offset1 = 0

        # print("function called: in0: {} in1: {}".format(len(in0), len(in1)))
        if len(tags_0) == 0:
            print("We don't have any tag in stream 0")
            self.consume(0, len(in0))
            return 0
        if len(tags_1) == 0:
            print("We don't have any tag in stream 1")
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
            return 0
        if position1 + self.frame_size >= len(in1):
            #print("Stream 1 too late, consuming")
            self.consume(1, position1)
            return 0

        #Detect packet packet loss
        if value0 != self.previous_packet+1:
            # for i in range(self.previous_packet+1, value0):
            print("Packet lost: {} in a row. Value is: {} Packets since last loss: {}".format(value0-(self.previous_packet+1), value0, value0-self.last_packet_loss))
            self.last_packet_loss = value0
                # self.message_port_pub(pmt.intern("error"), pmt.to_pmt(self.packet_loss_penalty))
        if len(out0) < self.frame_size:
            print("output size 0 not big enough")
        if len(out1) < self.frame_size:
            print("output size 1 not big enough")
        out0[:self.frame_size] = in0[position0:position0 + self.frame_size]
        out1[:self.frame_size] = in1[position1:position1 + self.frame_size]
        self.add_item_tag(0, self.nitems_written(0), pmt.intern(self.tag_name), pmt.to_pmt(value0))
        # print("outputting packet num: {}".format(value0))
        # diff = in0[position0:position0 + self.frame_size] - in1[position1:position1 + self.frame_size]
        # norm = numpy.linalg.norm(diff, ord = 2)

        # self.message_port_pub(pmt.intern("error"), pmt.to_pmt(norm.item()))
        # print(norm, value0)

        self.previous_packet = value0
        self.consume(0, position0 + self.frame_size)
        self.consume(1, position1 + self.frame_size)

        return self.frame_size
