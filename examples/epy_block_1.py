"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import pmt

class packet_header(gr.basic_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const
    We assume that the tag is at the beginning of the preamble"""

    def __init__(self, header_byte_length=2, preamble_byte_length = 5, tag_name = "step", payload_byte_length = 10):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.basic_block.__init__(
            self,
            name='Packet number inserter',   # will show up in GRC
            in_sig=[np.int8],
            out_sig=[np.int8]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
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
        #print("Read", read)
        #print("Written", written)
        output_position = 0
        for tag in tags:
            input_tag_position = tag.offset - read
            #print("Output:", output_position, len(output_items[0]), type(output_items[0][0]))
            if output_position > (len(output_items[0]) - self.packet_length):
                #print("Not enough writing space, returning")
                #print("Byte decomposition,", index, "becomes:", output_items[0][self.preamble_byte_length], output_items[0][self.preamble_byte_length+1])
                self.consume(0, input_tag_position)
                return output_position
            #print("Input:", input_tag_position, len(input_items[0]))
            if input_tag_position > len(input_items[0]) - (self.preamble_byte_length + self.payload_byte_length):
                print("Not enough reading space, returning")
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
                #print(pmt.to_python(tag.value) % (2**(8*(i+1))) / (2**(8*(i))))
            tag_offset = output_position + written
            self.add_item_tag(0, tag_offset, pmt.intern("header_start"), pmt.to_pmt(0))
            output_position += self.header_byte_length
            #print(output_position, "len", len(output_items[0]), output_position + self.payload_byte_length)
            #Copy payload
            output_items[0][output_position:output_position + self.payload_byte_length] = input_items[0][input_tag_position + self.preamble_byte_length: input_tag_position + self.preamble_byte_length + self.payload_byte_length]
            output_position += self.payload_byte_length
            #print(tag.key, tag.value, tag.offset)

        #output_items[0][:] = input_items[0]
        return output_position
