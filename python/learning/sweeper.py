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
    def __init__(self, top_block, snrs, error_threshold, ber_block_1=None, ber_block_2=None, probe_block_1 = None, probe_block_2 = None, snr_variable = None):

        self.top_block = top_block
        self.ber_block_1 = ber_block_1
        self.ber_block_2 = ber_block_2
        self.probe_block_1 = probe_block_1
        self.probe_block_2 = probe_block_2
        self.snr_variable = snr_variable

        self.error_threshold = error_threshold

        self.index = 0
        self.snrs = snrs
        self.results_1 = np.ones((len(self.snrs),))*0.000001
        self.results_2 = np.log10(0.0000001+special.erfc(np.sqrt(10**(self.snrs/10))))

        gr.sync_block.__init__(self,
        name="Sweeper",
        in_sig=[],
        out_sig=[(np.float32,len(self.snrs)),(np.float32,len(self.snrs))])

        self.started = False

    def start_sweep(self):
        print("Start sweep")
        self.index = 0
        self.snr_variable(self.snrs[self.index])
        # self.results = np.zeros((len(self.snrs),))

        # time.sleep(1)
        getattr(self.top_block, self.ber_block_1).reset_counters()
        getattr(self.top_block, self.ber_block_2).reset_counters()
        self.started = True

    def stop_sweep(self):
        print("Stop sweep")
        self.started = False
        getattr(self.top_block, self.ber_block_1).reset_counters()
        getattr(self.top_block, self.ber_block_2).reset_counters()
        print(self.results_1)
        print(self.results_2)

    def set_error_count(self, errors):
        # self.error_count = errors
        if self.started:
            print("Sweeper called")
            # Get BER info
            self.results_1[self.index] = getattr(self.top_block, self.probe_block_1).level()
            self.results_2[self.index] = getattr(self.top_block, self.probe_block_2).level()

            errors_1 = getattr(self.top_block, self.ber_block_1).total_errors()
            errors_2 = getattr(self.top_block, self.ber_block_2).total_errors()
            print("Error 1: {}  Error 2: {}".format(errors_1, errors_2))
            if errors_1 > self.error_threshold and errors_2 > self.error_threshold:
                self.index += 1
                if self.index >= len(self.snrs):
                    print(self.index, len(self.snrs))
                    self.stop_sweep()
                    return
                self.snr_variable(self.snrs[self.index])
                print("SNR to: {} and resetting ber".format(self.snrs[self.index]))
                # Reset BER Block
                time.sleep(1.5)
                getattr(self.top_block, self.ber_block_1).reset_counters()
                getattr(self.top_block, self.ber_block_2).reset_counters()
                # print(self.results)

    def work(self, input_items, output_items):
        out_1 = output_items[0]
        out_2 = output_items[1]
        # time.sleep(0.5)
        out_1[0] = self.results_1
        out_2[0] = self.results_2
        # output_items[1][0] = np.log10(0.0000001+special.erfc(np.sqrt(10**(self.snrs/10))))

        # print(len(output_items[0]))

        return 1
