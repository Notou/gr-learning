"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import pmt

class blk(gr.basic_block):  # other base classes are basic_block, decim_block, interp_block
    """Open stream on reception of corr_est tags from the correlation estimator block"""

    def __init__(self, payload_length=0, preamble_length=80, lookup_window = 30):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.basic_block.__init__(
            self,
            name='Isolator',   # will show up in GRC
            in_sig=[np.complex64],
            out_sig=[np.complex64]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.payload_length = payload_length
        self.preamble_length = preamble_length
        self.lookup_window = lookup_window
        self.set_tag_propagation_policy(0)
        self.set_history(max(self.preamble_length, self.lookup_window)+self.payload_length)
        # print [method for method in dir(gr.block) if callable(getattr(gr.block, method))]
        # import inspect
        # print inspect.getmro(blk)
        # print issubclass(blk, gr.basic_block)
        self.set_output_multiple((self.preamble_length + self.payload_length) *2)

    def forecast(self, noutput_items, ninput_items_required):
        # setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
                ninput_items_required[i] = int(noutput_items/2.0)+1

    def general_work(self, input_items, output_items):
        """example: multiply with constant"""
        read = self.nitems_read(0)
        history_read = max(read - self.history()+1,0)
        max_offset = history_read + len(input_items[0])
        #print("read",read, max_offset, "length", len(input_items[0]))
        tags = self.get_tags_in_range(0, history_read , max_offset, pmt.intern("step"))
        #for tag in tags:
            #print(tag.key, tag.value, tag.offset)


        tags = self.get_tags_in_range(0,  history_read + self.preamble_length , max_offset, pmt.intern("corr_est"))
        tags = list(tags)
        #print("Input size: ", len(input_items[0]), "   ", len(tags), " tags: ", "type: ", type(tags))
        #if len(tags)>0 :
        written = self.nitems_written(0)
        max_tag_offset = max_offset - self.payload_length - self.lookup_window
        i=0
        while i < len(tags):
            if tags[i].offset > max_tag_offset:
                tags.remove(tags[i])
                continue
            next_index=i+1
            while next_index < len(tags):
                #print(tags[next_index].key, tags[next_index].value, tags[next_index].offset)
                if tags[next_index].offset > tags[i].offset + self.lookup_window:
                    break
                if pmt.to_python(tags[i].value) > pmt.to_python(tags[next_index].value):
                    #print("removing next: curr=", tags[i].value, ">  next=", tags[next_index].value, " types: ", type(tags[i].value), type(tags[next_index].value))
                    tags.remove(tags[next_index])
                else:
                    #print("removing current: curr=", tags[i].value, "<  next=", tags[next_index].value)
                    tags.remove(tags[i])
            #print(tags[i].key, tags[i].value, tags[i].offset)
            i += 1
        output_number = 0
        pack_length = self.preamble_length+self.payload_length
        #output_items[0].resize((pack_length*len(tags),))
        #print(type(output_items[0]), output_items[0].dtype, output_items[0].shape)
        #print(len(tags))
        for i in range(len(tags)):
            #print(tags[i].key, tags[i].value, tags[i].offset)
            position = tags[i].offset - history_read
            # if position < self.history():
            #     print("Start copy at ", position)
            output_number += pack_length
            if output_number > len(output_items[0]):
                print("Too much input was given")
                self.consume(0, position)
                return output_number

            #print("output number: ", output_number)
            output_items[0][i*pack_length:(i+1)*pack_length] = input_items[0][position-self.preamble_length:position+self.payload_length]

            tag_offset = written + i*pack_length + (tags[i].offset - history_read - (position-self.preamble_length))
            self.add_item_tag(0, tag_offset+20, pmt.intern("header_start"), pmt.to_pmt(0))
            tags_to_propagate = self.get_tags_in_range(0, tags[i].offset-self.preamble_length, tags[i].offset+self.payload_length)
            for tag in tags_to_propagate:
                #if pmt.to_python(tag.key) ==  "step":
                    #print("Seen: ", tag.key, tag.value, tag.offset)
                tag_offset = written + i*pack_length + (tag.offset - history_read - (position-self.preamble_length))
                self.add_item_tag(0, tag_offset, tag.key, tag.value, tag.srcid)
        self.consume(0, len(input_items[0])-(self.history()-1))
        if read < self.history():
            self.consume(0, self.history()-1)
        return output_number
