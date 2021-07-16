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
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self, tag_name="chan_est"):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.basic_block.__init__(
            self,
            name='Embedded Python Block',   # will show up in GRC
            in_sig=[(np.complex64,64)],
            out_sig=[(np.complex64,64)]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.tag_name = tag_name
        self.output = np.array(64,dtype=np.complex64)

    def forecast(self, noutput_items, ninput_items_required):
        #setup size of input_items[i] for work call
        ninput_items_required[0] = noutput_items

    def general_work(self, input_items, output_items):
        """example: multiply with constant"""
        tags = self.get_tags_in_window(0,  0 , len(input_items[0]), pmt.intern('ofdm_sync_chan_taps'))
        # print(len(tags))
        for i in range(len(tags)):
            output_items[0][i] = pmt.to_python(tags[i].value)
            # print(pmt.to_python(tag.value))
        self.consume(0, len(input_items[0]))
        return len(tags)
