"""
Embedded Python Blocks:

Each time this file is saved, GRC will instantiate the first class it finds
to get ports and parameters of your block. The arguments to __init__  will
be the parameters. All of them are required to have default values!
"""

import numpy as np
from gnuradio import gr
import pmt

class trigger(gr.basic_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded python block that forwards a chunk of 'len' samples when the trigger input receives a 1. Input and trigger streams need to be synchronised."""

    def __init__(self, fft_len = 64, cp_len = 16, length=10, tag_key="frame_len"):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.basic_block.__init__(
            self,
            name='Trigger',   # will show up in GRC
            in_sig=[np.complex64, np.int8],
            out_sig=[np.complex64]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).
        self.fft_len = fft_len
        self.cp_len = cp_len
        self.length = length
        self.tag_key = tag_key
        self.to_forward = 0
        self.sample_in_frame = 0

    def forecast(self, noutput_items, ninput_items_required):
        # setup size of input_items[i] for work call
        for i in range(len(ninput_items_required)):
                ninput_items_required[i] = int(noutput_items)

    def general_work(self, input_items, output_items):
        """forwards item upon trigger reception"""
        output_samples = 0
        input = input_items[0]
        trigger = input_items[1]
        work_samples = np.minimum(len(output_items[0]),np.minimum(len(input), len(trigger)))
        for i in range(work_samples):
            if trigger[i] == 1: # Triggered
                self.add_item_tag(0, output_samples+self.nitems_written(0), pmt.intern(self.tag_key), pmt.to_pmt(self.length))
                self.to_forward = self.length
            if self.to_forward > 0:
                if self.sample_in_frame < self.fft_len:
                    output_items[0][output_samples] = input[i]
                    output_samples += 1
                self.sample_in_frame = (self.sample_in_frame + 1) % (self.fft_len+self.cp_len)
                if self.sample_in_frame == 0:
                    self.to_forward -= 1

        self.consume(0, work_samples)
        self.consume(1, work_samples)
        return output_samples
