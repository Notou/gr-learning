#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: RL Mod training
# Author: Cyrille Morin
# GNU Radio version: 3.8.1.0

from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import eng_notation
import bokehgui
from gnuradio import fec
from gnuradio import fft
from gnuradio.fft import window
from gnuradio import gr
from gnuradio.filter import firdes
import sys
import signal
import time
import functools
from bokeh.client import push_session
from bokeh.plotting import curdoc
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import uhd
from gnuradio import zeromq
from gnuradio.digital.utils import tagged_streams
import learning
import numpy as np

class rl_mod_TX(gr.top_block):
    def __init__(self, doc, ip="127.0.0.1"):
        gr.top_block.__init__(self, "RL Mod training")
        self.doc = doc
        self.plot_lst = []
        self.widget_lst = []

        ##################################################
        # Variables
        ##################################################
        self.pilot_symbols = pilot_symbols = ((1, 1, 1, -1,),)
        self.pilot_carriers = pilot_carriers = ((-21, -7, 7, 21,),)
        self.packet_length_tag_key = packet_length_tag_key = "packet_length"
        self.occupied_carriers = occupied_carriers = (list(range(-22, -21)) + list(range(-20, -7)) + list(range(-6, 0)) + list(range(1, 7)) + list(range(8, 21)) + list(range(22, 23)),)
        self.length_tag_key_0 = length_tag_key_0 = "frame_len"
        self.length_tag_key = length_tag_key = "packet_len"
        self.header_mod = header_mod = digital.constellation_bpsk()
        self.fft_len = fft_len = 64
        self.bits_per_symbol = bits_per_symbol = 4
        self.variable_0 = variable_0 = len(occupied_carriers[0])
        self.tx_lr = tx_lr = -3
        self.tx_explo = tx_explo = 0.15
        self.training_mod = training_mod = 1
        self.timestamp = timestamp = '1'
        self.t_state = t_state = 1
        self.sync_word2 = sync_word2 = [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 0, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
        self.sync_word1 = sync_word1 = [0., 0., 0., 0., 0., 0., 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., -1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., -1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 1.41421356, 0., 0., 0., 0., 0., 0.]
        self.samp_rate = samp_rate = 0.3e6
        self.rolloff = rolloff = 0
        self.puncpat = puncpat = '11'
        self.payload_mod = payload_mod = digital.qam_constellation(constellation_points=2**bits_per_symbol)
        self.payload_equalizer = payload_equalizer = digital.ofdm_equalizer_static(fft_len,  occupied_carriers, pilot_carriers, pilot_symbols, 1)
        self.packet_len = packet_len = 400
        self.ldpc_enc = ldpc_enc = fec.ldpc_encoder_make(gr.prefix() + "/share/gnuradio/fec/ldpc/" + "n_0100_k_0042_gap_02.alist")
        self.header_formatter = header_formatter = digital.packet_header_ofdm(occupied_carriers, n_syms=1, len_tag_key=packet_length_tag_key, frame_len_tag_key=length_tag_key_0, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=False)
        self.header_equalizer = header_equalizer = digital.ofdm_equalizer_simpledfe(fft_len, header_mod.base(), occupied_carriers, pilot_carriers, pilot_symbols)
        self.hdr_format = hdr_format = digital.header_format_ofdm(occupied_carriers, 1, length_tag_key,)
        self.gain = gain = 10
        self.freq = freq = 900e6
        self.bits_per_symbol_0 = bits_per_symbol_0 = len(occupied_carriers[0])

        ##################################################
        # Blocks
        ##################################################
        self.tx_lr_slider = bokehgui.slider(self.widget_lst, 'TX Learning rate (log)' +":", -5, 0, 0.1, 1, -3)
        self.tx_lr_slider.add_callback(lambda attr, old, new: self.set_tx_lr(new))
        self.tx_explo_slider = bokehgui.slider(self.widget_lst, 'TX Exploration noise' +":", 0.001, 0.5, 0.001, 1, 0.15)
        self.tx_explo_slider.add_callback(lambda attr, old, new: self.set_tx_explo(new))
        self.training_mod_textbox = bokehgui.textbox(self.widget_lst, str(1), 'Alternate training every' +": ")
        self.training_mod_textbox.add_callback(
          lambda attr, old, new: self.set_training_mod(int(new)))
        self._t_state_options = [    1,     0,   ]
        self._t_state_labels = [      'On',      'Off',  ]

        self.t_state_radiobutton = bokehgui.radiobutton(self.widget_lst, None, self._t_state_labels, inline = True)
        self.t_state_radiobutton.add_callback(
                  lambda new: self.set_t_state(int(self._t_state_options[new])))
        self.gain_slider = bokehgui.slider(self.widget_lst, 'Amplitude' +":", 0, 90, 0.5, 1, 10)
        self.gain_slider.add_callback(lambda attr, old, new: self.set_gain(new))
        self.zeromq_sub_msg_source_0_1 = zeromq.sub_msg_source('tcp://127.0.0.1:50002', 100)
        self.zeromq_pub_msg_sink_0_0 = zeromq.pub_msg_sink('tcp://127.0.0.1:50001', 100)
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(("", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0,1)),
            ),
            length_tag_key,
        )


        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)

        self.uhd_usrp_sink_0.set_time_now(uhd.time_spec(time.time()), uhd.ALL_MBOARDS)

        self.uhd_usrp_sink_0.set_center_freq(freq, 0)

        self.uhd_usrp_sink_0.set_antenna('TX/RX', 0)

        self.uhd_usrp_sink_0.set_gain(gain, 0)
        self.timestamp_textbox = bokehgui.textbox(self.widget_lst, str('1'), 'Load timestamp' +": ")
        self.timestamp_textbox.add_callback(
          lambda attr, old, new: self.set_timestamp(str(new)))
        self.learning_tag_numerotation_0 = learning.tag_numerotation('packet_num', packet_len, 4096, "bb")
        self.learning_rl_mod_0 = learning.rl_mod('packet_num', bits_per_symbol, packet_len, 1, training_mod, 10**tx_lr , tx_explo, t_state , '/home/cyrille/Gnu-Radio/modules/gr-learning/examples/saved_models')
        self.fft_vxx_0_0 = fft.fft_vcc(fft_len, False, (), True, 1)
        self.fec_extended_encoder_1_0_0 = fec.extended_encoder(encoder_obj_list=ldpc_enc, threading= None, puncpat=puncpat)
        self.digital_protocol_formatter_bb_0 = digital.protocol_formatter_bb(hdr_format, length_tag_key)
        self.digital_ofdm_cyclic_prefixer_0 = digital.ofdm_cyclic_prefixer(
            fft_len,
            fft_len + fft_len//4,
            rolloff,
            length_tag_key)
        self.digital_ofdm_carrier_allocator_cvc_0 = digital.ofdm_carrier_allocator_cvc( fft_len, occupied_carriers, pilot_carriers, pilot_symbols, (sync_word1, sync_word2), length_tag_key, True)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc(header_mod.points(), 1)
        self.bokehgui_time_const_x_0 = bokehgui.time_sink_c_proc(1024, samp_rate, "Transmitted noiseless constellation",  1)

        self.bokehgui_time_const_x_0_plot = bokehgui.const_sink_c(self.doc, self.plot_lst, self.bokehgui_time_const_x_0, is_message = False)
        labels = ['', '', '', '', '',
                  '', '', '', '', '']
        legend_list = []
        for i in range(  1 ):
          if len(labels[i]) == 0:
              if(i % 2 == 0):
                  legend_list.append("Re{{Data {0}}}".format(i/2))
              else:
                  legend_list.append("Im{{Data {0}}}".format(i/2))
          else:
              legend_list.append(labels[i])

        self.bokehgui_time_const_x_0_plot.initialize(update_time = 500,
                                    legend_list = legend_list)

        self.bokehgui_time_const_x_0_plot.set_y_axis([-2, 2])
        self.bokehgui_time_const_x_0_plot.set_y_label('Q Channel' + '(' +""+')')

        self.bokehgui_time_const_x_0_plot.set_x_label('I Channel' + '(' +""+')')
        self.bokehgui_time_const_x_0_plot.enable_tags(-1, False)
        self.bokehgui_time_const_x_0_plot.set_trigger_mode(bokehgui.TRIG_MODE_FREE, bokehgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.bokehgui_time_const_x_0_plot.enable_grid(False)
        self.bokehgui_time_const_x_0_plot.enable_axis_labels(True)
        self.bokehgui_time_const_x_0_plot.disable_legend(not True)
        self.bokehgui_time_const_x_0_plot.set_layout(*((0,1,2,2)))
        colors = ["blue", "red", "green", "black", "cyan",
                  "magenta", "yellow", "blue", "blue", "blue"]
        widths = [1, 1, 1, 1, 1,
                  1, 1, 1, 1, 1]
        markers = ['o', 'o', 'o', 'o', 'o',
                   'o', 'o', 'o', 'o', 'o']
        alphas = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0]
        for i in range(    1  ):
          self.bokehgui_time_const_x_0_plot.format_line(i, colors[i], widths[i], 'None', markers[i], alphas[i])
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_char*1, samp_rate/6,True)
        self.blocks_tagged_stream_to_pdu_0_0_0 = blocks.tagged_stream_to_pdu(blocks.byte_t, 'packet_len')
        self.blocks_tagged_stream_mux_0 = blocks.tagged_stream_mux(gr.sizeof_gr_complex*1, length_tag_key, 0)
        self.blocks_stream_to_tagged_stream_0_0 = blocks.stream_to_tagged_stream(gr.sizeof_char, 1, packet_len, length_tag_key)
        self.blocks_repack_bits_bb_0_0_0 = blocks.repack_bits_bb(8, 1, length_tag_key, False, gr.GR_LSB_FIRST)
        self.blocks_repack_bits_bb_0_0 = blocks.repack_bits_bb(1, bits_per_symbol, '', False, gr.GR_LSB_FIRST)
        self.blocks_multiply_const_xx_0 = blocks.multiply_const_cc(0.01, 1)
        self.analog_random_source_x_0_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 2, 1000))), True)

        if self.widget_lst:
            input_t = bokehgui.bokeh_layout.widgetbox(self.widget_lst)
            widgetbox = bokehgui.bokeh_layout.WidgetLayout(input_t)
            widgetbox.set_layout(*((0, 0, 2, 1)))
            list_obj = [widgetbox] + self.plot_lst
        else:
            list_obj = self.plot_lst
        layout_t = bokehgui.bokeh_layout.create_layout(list_obj, "fixed")
        self.doc.add_root(layout_t)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.blocks_tagged_stream_to_pdu_0_0_0, 'pdus'), (self.zeromq_pub_msg_sink_0_0, 'in'))
        self.msg_connect((self.zeromq_sub_msg_source_0_1, 'out'), (self.learning_rl_mod_0, 'losses'))
        self.connect((self.analog_random_source_x_0_0, 0), (self.fec_extended_encoder_1_0_0, 0))
        self.connect((self.blocks_multiply_const_xx_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0, 0), (self.blocks_stream_to_tagged_stream_0_0, 0))
        self.connect((self.blocks_repack_bits_bb_0_0_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.digital_protocol_formatter_bb_0, 0))
        self.connect((self.blocks_stream_to_tagged_stream_0_0, 0), (self.learning_tag_numerotation_0, 0))
        self.connect((self.blocks_tagged_stream_mux_0, 0), (self.digital_ofdm_carrier_allocator_cvc_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_tagged_stream_to_pdu_0_0_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.blocks_tagged_stream_mux_0, 0))
        self.connect((self.digital_ofdm_carrier_allocator_cvc_0, 0), (self.fft_vxx_0_0, 0))
        self.connect((self.digital_ofdm_cyclic_prefixer_0, 0), (self.blocks_multiply_const_xx_0, 0))
        self.connect((self.digital_protocol_formatter_bb_0, 0), (self.blocks_repack_bits_bb_0_0_0, 0))
        self.connect((self.fec_extended_encoder_1_0_0, 0), (self.blocks_repack_bits_bb_0_0, 0))
        self.connect((self.fft_vxx_0_0, 0), (self.digital_ofdm_cyclic_prefixer_0, 0))
        self.connect((self.learning_rl_mod_0, 0), (self.blocks_tagged_stream_mux_0, 1))
        self.connect((self.learning_rl_mod_0, 1), (self.bokehgui_time_const_x_0, 0))
        self.connect((self.learning_tag_numerotation_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.learning_tag_numerotation_0, 0), (self.learning_rl_mod_0, 0))

    def get_pilot_symbols(self):
        return self.pilot_symbols

    def set_pilot_symbols(self, pilot_symbols):
        self.pilot_symbols = pilot_symbols
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len,  self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))

    def get_pilot_carriers(self):
        return self.pilot_carriers

    def set_pilot_carriers(self, pilot_carriers):
        self.pilot_carriers = pilot_carriers
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len,  self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))

    def get_packet_length_tag_key(self):
        return self.packet_length_tag_key

    def set_packet_length_tag_key(self, packet_length_tag_key):
        self.packet_length_tag_key = packet_length_tag_key
        self.set_header_formatter(digital.packet_header_ofdm(self.occupied_carriers, n_syms=1, len_tag_key=self.packet_length_tag_key, frame_len_tag_key=self.length_tag_key_0, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=False))

    def get_occupied_carriers(self):
        return self.occupied_carriers

    def set_occupied_carriers(self, occupied_carriers):
        self.occupied_carriers = occupied_carriers
        self.set_bits_per_symbol_0(len(self.occupied_carriers[0]))
        self.set_hdr_format(digital.header_format_ofdm(self.occupied_carriers, 1, self.length_tag_key,))
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_header_formatter(digital.packet_header_ofdm(self.occupied_carriers, n_syms=1, len_tag_key=self.packet_length_tag_key, frame_len_tag_key=self.length_tag_key_0, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=False))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len,  self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))
        self.set_variable_0(len(self.occupied_carriers[0]))

    def get_length_tag_key_0(self):
        return self.length_tag_key_0

    def set_length_tag_key_0(self, length_tag_key_0):
        self.length_tag_key_0 = length_tag_key_0
        self.set_header_formatter(digital.packet_header_ofdm(self.occupied_carriers, n_syms=1, len_tag_key=self.packet_length_tag_key, frame_len_tag_key=self.length_tag_key_0, bits_per_header_sym=header_mod.bits_per_symbol(), bits_per_payload_sym=8, scramble_header=False))

    def get_length_tag_key(self):
        return self.length_tag_key

    def set_length_tag_key(self, length_tag_key):
        self.length_tag_key = length_tag_key
        self.set_hdr_format(digital.header_format_ofdm(self.occupied_carriers, 1, self.length_tag_key,))

    def get_header_mod(self):
        return self.header_mod

    def set_header_mod(self, header_mod):
        self.header_mod = header_mod

    def get_fft_len(self):
        return self.fft_len

    def set_fft_len(self, fft_len):
        self.fft_len = fft_len
        self.set_header_equalizer(digital.ofdm_equalizer_simpledfe(self.fft_len, header_mod.base(), self.occupied_carriers, self.pilot_carriers, self.pilot_symbols))
        self.set_payload_equalizer(digital.ofdm_equalizer_static(self.fft_len,  self.occupied_carriers, self.pilot_carriers, self.pilot_symbols, 1))

    def get_bits_per_symbol(self):
        return self.bits_per_symbol

    def set_bits_per_symbol(self, bits_per_symbol):
        self.bits_per_symbol = bits_per_symbol
        self.set_payload_mod(digital.qam_constellation(constellation_points=2**self.bits_per_symbol))
        self.blocks_repack_bits_bb_0_0.set_k_and_l(1,self.bits_per_symbol)

    def get_variable_0(self):
        return self.variable_0

    def set_variable_0(self, variable_0):
        self.variable_0 = variable_0

    def get_tx_lr(self):
        return self.tx_lr

    def set_tx_lr(self, tx_lr):
        self.tx_lr = tx_lr
        self.learning_rl_mod_0.set_lr(10**self.tx_lr )

    def get_tx_explo(self):
        return self.tx_explo

    def set_tx_explo(self, tx_explo):
        self.tx_explo = tx_explo
        self.learning_rl_mod_0.set_explo_noise(self.tx_explo)

    def get_training_mod(self):
        return self.training_mod

    def set_training_mod(self, training_mod):
        self.training_mod = training_mod
        self.learning_rl_mod_0.set_alternate(self.training_mod)

    def get_timestamp(self):
        return self.timestamp

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp
        self.learning_rl_mod_0.load_model(self.timestamp)

    def get_t_state(self):
        return self.t_state

    def set_t_state(self, t_state):
        self.t_state = t_state
        self.learning_rl_mod_0.set_training_state(self.t_state )

    def get_sync_word2(self):
        return self.sync_word2

    def set_sync_word2(self, sync_word2):
        self.sync_word2 = sync_word2

    def get_sync_word1(self):
        return self.sync_word1

    def set_sync_word1(self, sync_word1):
        self.sync_word1 = sync_word1

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.blocks_throttle_0.set_sample_rate(self.samp_rate/6)
        self.bokehgui_time_const_x_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_rolloff(self):
        return self.rolloff

    def set_rolloff(self, rolloff):
        self.rolloff = rolloff

    def get_puncpat(self):
        return self.puncpat

    def set_puncpat(self, puncpat):
        self.puncpat = puncpat

    def get_payload_mod(self):
        return self.payload_mod

    def set_payload_mod(self, payload_mod):
        self.payload_mod = payload_mod

    def get_payload_equalizer(self):
        return self.payload_equalizer

    def set_payload_equalizer(self, payload_equalizer):
        self.payload_equalizer = payload_equalizer

    def get_packet_len(self):
        return self.packet_len

    def set_packet_len(self, packet_len):
        self.packet_len = packet_len
        self.blocks_stream_to_tagged_stream_0_0.set_packet_len(self.packet_len)
        self.blocks_stream_to_tagged_stream_0_0.set_packet_len_pmt(self.packet_len)

    def get_ldpc_enc(self):
        return self.ldpc_enc

    def set_ldpc_enc(self, ldpc_enc):
        self.ldpc_enc = ldpc_enc

    def get_header_formatter(self):
        return self.header_formatter

    def set_header_formatter(self, header_formatter):
        self.header_formatter = header_formatter

    def get_header_equalizer(self):
        return self.header_equalizer

    def set_header_equalizer(self, header_equalizer):
        self.header_equalizer = header_equalizer

    def get_hdr_format(self):
        return self.hdr_format

    def set_hdr_format(self, hdr_format):
        self.hdr_format = hdr_format

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_sink_0.set_gain(self.gain, 0)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_sink_0.set_center_freq(self.freq, 0)

    def get_bits_per_symbol_0(self):
        return self.bits_per_symbol_0

    def set_bits_per_symbol_0(self, bits_per_symbol_0):
        self.bits_per_symbol_0 = bits_per_symbol_0

def argument_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--ip", dest="ip", type=str, default="127.0.0.1",
        help="Set IP [default=%(default)r]")
    return parser

def main(top_block_cls=rl_mod_TX, options=None):
    if options is None:
        options = argument_parser().parse_args()
    serverProc, port = bokehgui.utils.create_server()
    def killProc(signum, frame, tb):
        tb.stop()
        tb.wait()
        serverProc.terminate()
        serverProc.kill()
    time.sleep(1)
    try:
        # Define the document instance
        doc = curdoc()
        doc.title = "RL Mod training - Cyrille Morin"
        session = push_session(doc, session_id="rl_mod_TX",
                               url = "http://localhost:" + port + "/bokehgui")
        # Create Top Block instance
        tb = top_block_cls(doc, ip=options.ip)
        try:
            tb.start()
            signal.signal(signal.SIGTERM, functools.partial(killProc, tb=tb))
            session.loop_until_closed()
        finally:
            print("Exiting the simulation. Stopping Bokeh Server")
            tb.stop()
            tb.wait()
    finally:
        serverProc.terminate()
        serverProc.kill()


if __name__ == '__main__':
    main()
