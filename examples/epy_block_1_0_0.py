"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import pmt

class header_reader(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const
    We assume that the tag is at the beginning of the preamble"""

    def __init__(self, header_byte_length=2, preamble_byte_length = 5, tag_name = "header_start", payload_byte_length = 10, threshold = 10):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Packet number reader',   # will show up in GRC
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
        self.set_relative_rate(1)
        self.set_output_multiple(self.packet_length *2)
        self.set_history((preamble_byte_length + self.header_byte_length)*8)
        self.packet_number = np.uint32(0)
        self.threshold = threshold

    def work(self, input_items, output_items):
        """example: multiply with constant"""
        read = self.nitems_read(0)
        history_read = max(read - self.history()+1,0)
        written = self.nitems_written(0)
        #print("Read", read, history_read)
        #print("Written", written)
        start_pos = (self.preamble_byte_length*8)
        process_length = len(output_items[0])
        stop_pos = start_pos + process_length -1
        #print("Stop position", stop_pos, len(input_items[0]) - stop_pos)
        tags = self.get_tags_in_window(0, start_pos , stop_pos , pmt.intern(self.tag_name)) # We don't want to read a tag if we cannot access the header that comes after

        #print("Input:", len(input_items[0]), "Output:", len(output_items[0]), len(input_items[0])-(self.history()-1)-len(output_items[0]))

        for tag in tags:
            position = tag.offset - read + (self.history()-1)
            #print("Position:", position, "offset", tag.offset)
            if position > stop_pos:
                print("Outside upper limit")
                break
            char = ""
            for byte in reversed(range(self.header_byte_length)):
                for bit in (range(8)):

                    char += str(input_items[0][position + 8*byte + bit])
                    # print(position + 8*byte + bit, input_items[0][position + 8*byte + bit])
            #print("Packet number:", self.packet_number)

            if abs(self.packet_number-int(char, 2)) > self.threshold:
                print("Packet number disparity!!!!!!!!:", self.packet_number)
            else:
                self.add_item_tag(0, tag.offset - (self.preamble_byte_length)*8, pmt.intern("step"), pmt.to_pmt(np.int64(int(char, 2))))

            self.packet_number = int(char, 2)

        output_items[0][:] = input_items[0][-len(output_items[0]): ]

        return process_length
