#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2019 Cyrille Morin.
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
import time
from gnuradio import gr
from scipy import special

class sweeper(gr.sync_block):

    """
    Sweeps a range of snr values to gather a BER curve
    """
    def __init__(self, top_block, snrs, error_threshold, ber_block=None, probe_block = None, snr_variable = None):

        self.top_block = top_block
        self.ber_block = ber_block
        self.probe_block = probe_block
        self.snr_variable = snr_variable

        self.error_threshold = error_threshold

        self.index = 0
        self.snrs = snrs
        self.results = np.zeros((len(self.snrs),))

        gr.sync_block.__init__(self,
        name="Sweeper",
        in_sig=[],
        out_sig=[(np.float32,len(self.snrs)),(np.float32,len(self.snrs))])

        self.started = False

    def start_sweep(self):
        self.index = 0
        self.snr_variable(self.snrs[self.index])
        # self.results = np.zeros((len(self.snrs),))

        # time.sleep(1)
        getattr(self.top_block, self.ber_block).reset_counters()
        self.started = True

    def stop_sweep(self):
        self.started = False
        getattr(self.top_block, self.ber_block).reset_counters()
        print(self.results)

    def set_error_count(self, errors):
        self.error_count = errors

        if self.started:
            # Get BER info
            self.results[self.index] = getattr(self.top_block, self.probe_block).level()
            if errors > self.error_threshold:
                self.index += 1
                if self.index >= len(self.snrs):
                    print(self.index, len(self.snrs))
                    self.stop_sweep()
                    return
                self.snr_variable(self.snrs[self.index])
                print("SNR to: {} and resetting ber".format(self.snrs[self.index]))
                # Reset BER Block
                time.sleep(1.5)
                getattr(self.top_block, self.ber_block).reset_counters()
                # print(self.results)

    def work(self, input_items, output_items):
        out = output_items[0]
        time.sleep(0.5)
        out[0] = self.results
        output_items[1][0] = np.log10(0.0000001+special.erfc(np.sqrt(10**(self.snrs/10))))

        # print(len(output_items[0]))

        return 1
